/*
 * test_m4t_ternary_matmul.c — tests for MTFP19 × packed-ternary matmul.
 *
 * Coverage:
 *   1. small_golden          — hand-computed 2×4×3 matmul
 *   2. random_vs_reference   — random M/K/N matrices vs int64 reference,
 *                              including K not divisible by 16 (tail)
 *   3. saturation_clamp      — inputs that overflow MTFP19, verify clamp
 *   4. saturation_flags      — flags reflect saturation per cell
 *   5. zero_dim              — M=0 / N=0 / K=0 edge cases
 *   6. determinism           — same inputs → same outputs across calls
 */

#include "m4t_ternary_matmul.h"
#include "m4t_trit_pack.h"
#include "m4t_mtfp.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── RNG ────────────────────────────────────────────────────────────────── */
static uint32_t g_rng = 0xfeedfaceu;
static uint32_t xs32(void) {
    uint32_t x = g_rng; x ^= x << 13; x ^= x >> 17; x ^= x << 5; g_rng = x; return x;
}
static int32_t rand_mtfp19(void) {
    int64_t span = (int64_t)M4T_MTFP_MAX_VAL * 2 + 1;
    return (int32_t)((int64_t)(xs32() % (uint64_t)span) - (int64_t)M4T_MTFP_MAX_VAL);
}
static m4t_trit_t rand_trit(void) {
    return (m4t_trit_t)((int)(xs32() % 3u) - 1);
}
static int rand_int(int lo, int hi) {
    return lo + (int)(xs32() % (uint32_t)(hi - lo + 1));
}

/* Pack an unpacked trit row [N,K] into packed [N, ceil(K/4)] format. */
static void pack_trits_2d(uint8_t* dst, const m4t_trit_t* src, int N, int K) {
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    for (int j = 0; j < N; j++) {
        m4t_pack_trits_1d(dst + (size_t)j * Kp, src + (size_t)j * K, K);
    }
}

/* Reference: int64 matmul using unpacked trits and saturating clamp on store. */
static void ref_matmul(
    int32_t* Y, const int32_t* X, const m4t_trit_t* W,
    int M, int K, int N)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int64_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int64_t)X[i*K+k] * (int64_t)W[j*K+k];
            }
            int32_t out;
            if (acc >  (int64_t)M4T_MTFP_MAX_VAL) out =  M4T_MTFP_MAX_VAL;
            else if (acc < -(int64_t)M4T_MTFP_MAX_VAL) out = -M4T_MTFP_MAX_VAL;
            else out = (int32_t)acc;
            Y[i*N+j] = out;
        }
    }
}

/* ── Tests ──────────────────────────────────────────────────────────────── */

static int test_small_golden(void) {
    /* M=2, K=4, N=3.
     * X = [[1,  2,  3,  4],
     *      [5, -6,  7, -8]]
     * W (unpacked, [N,K]) =
     *     [[+1, -1,  0, +1],
     *      [ 0,  0, +1,  0],
     *      [-1, +1, -1, +1]]
     * W^T:  [[+1,  0, -1],
     *        [-1,  0, +1],
     *        [ 0, +1, -1],
     *        [+1,  0, +1]]
     * Y = X @ W^T:
     *   Y[0,0] =  1·1 +  2·-1 +  3·0 +  4·1  =  3
     *   Y[0,1] =  1·0 +  2·0  +  3·1 +  4·0  =  3
     *   Y[0,2] =  1·-1 + 2·1  +  3·-1+  4·1  =  2
     *   Y[1,0] =  5·1 + -6·-1 +  7·0 + -8·1  =  3
     *   Y[1,1] =  5·0 + -6·0  +  7·1 + -8·0  =  7
     *   Y[1,2] =  5·-1+ -6·1  +  7·-1+ -8·1  = -26 */
    int32_t X[8] = { 1, 2, 3, 4,  5, -6, 7, -8 };
    m4t_trit_t W[12] = {
         1, -1,  0,  1,
         0,  0,  1,  0,
        -1,  1, -1,  1
    };
    uint8_t W_packed[3 * M4T_TRIT_PACKED_BYTES(4)];
    pack_trits_2d(W_packed, W, 3, 4);

    int32_t Y[6];
    m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, NULL, 2, 4, 3);

    int32_t expected[6] = { 3, 3, 2,  3, 7, -26 };
    for (int i = 0; i < 6; i++) {
        if (Y[i] != expected[i]) {
            printf("FAIL small_golden[%d]: got=%d want=%d\n",
                   i, Y[i], expected[i]);
            return 1;
        }
    }
    return 0;
}

static int test_random_vs_reference(void) {
    g_rng = 0xfeedfaceu;
    for (int trial = 0; trial < 200; trial++) {
        int M = rand_int(1, 4);
        int N = rand_int(1, 4);
        int K = rand_int(1, 100);   /* small enough to not saturate randomly */

        int32_t* X = malloc((size_t)M * K * sizeof(int32_t));
        m4t_trit_t* W = malloc((size_t)N * K);
        int Kp = M4T_TRIT_PACKED_BYTES(K);
        uint8_t* W_packed = malloc((size_t)N * (size_t)Kp);
        int32_t* Y = malloc((size_t)M * N * sizeof(int32_t));
        int32_t* Yref = malloc((size_t)M * N * sizeof(int32_t));

        /* Use small mantissas to keep accumulator in MTFP19 range. */
        for (int i = 0; i < M*K; i++) X[i] = (int32_t)((int)(xs32() % 1001u) - 500);
        for (int i = 0; i < N*K; i++) W[i] = rand_trit();
        pack_trits_2d(W_packed, W, N, K);

        m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, NULL, M, K, N);
        ref_matmul(Yref, X, W, M, K, N);

        for (int i = 0; i < M*N; i++) {
            if (Y[i] != Yref[i]) {
                printf("FAIL random trial %d cell %d: got=%d ref=%d (M=%d K=%d N=%d)\n",
                       trial, i, Y[i], Yref[i], M, K, N);
                free(X); free(W); free(W_packed); free(Y); free(Yref);
                return 1;
            }
        }
        free(X); free(W); free(W_packed); free(Y); free(Yref);
    }
    return 0;
}

static int test_saturation_clamp(void) {
    /* Inputs that DO overflow MTFP19. Use K cells of MAX_VAL × +1 → K * MAX. */
    int K = 4;
    int32_t X[4] = {
        M4T_MTFP_MAX_VAL,
        M4T_MTFP_MAX_VAL,
        M4T_MTFP_MAX_VAL,
        M4T_MTFP_MAX_VAL,
    };
    m4t_trit_t W[4] = { 1, 1, 1, 1 };
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    uint8_t W_packed[1 * 1];   /* Kp = 1 for K=4 */
    (void)Kp;
    pack_trits_2d(W_packed, W, 1, K);

    int32_t Y[1];
    m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, NULL, 1, K, 1);
    /* True sum = 4 * MAX_VAL ≈ 2.32e9; saturates to MAX_VAL. */
    if (Y[0] != M4T_MTFP_MAX_VAL) {
        printf("FAIL sat_clamp: got=%d want=%d\n", Y[0], M4T_MTFP_MAX_VAL);
        return 1;
    }

    /* Negative saturation. */
    m4t_trit_t Wn[4] = { -1, -1, -1, -1 };
    pack_trits_2d(W_packed, Wn, 1, K);
    m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, NULL, 1, K, 1);
    if (Y[0] != -M4T_MTFP_MAX_VAL) {
        printf("FAIL sat_clamp neg: got=%d want=%d\n", Y[0], -M4T_MTFP_MAX_VAL);
        return 1;
    }
    return 0;
}

static int test_saturation_flags(void) {
    /* Two output cells: one saturates, one doesn't. */
    int K = 4;
    int32_t X[4] = {
        M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL
    };
    m4t_trit_t W[8] = {
        1, 1, 1, 1,    /* output cell 0: saturates */
        0, 0, 0, 0,    /* output cell 1: zero, no saturation */
    };
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    uint8_t W_packed[2 * 1];
    (void)Kp;
    pack_trits_2d(W_packed, W, 2, K);

    int32_t Y[2];
    int M = 1, N = 2;
    size_t fb = M4T_FLAG_BYTES(M * N);
    uint8_t flags[1];   /* M*N=2 cells → ceil(2/4)=1 byte */
    memset(flags, 0, fb);

    m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, flags, M, K, N);

    if (Y[0] != M4T_MTFP_MAX_VAL || Y[1] != 0) {
        printf("FAIL sat_flags Y: Y[0]=%d Y[1]=%d\n", Y[0], Y[1]);
        return 1;
    }
    int got_sat_0 = m4t_flag_test(flags, 0, M4T_FLAG_SATURATED) ? 1 : 0;
    int got_sat_1 = m4t_flag_test(flags, 1, M4T_FLAG_SATURATED) ? 1 : 0;
    int got_rnd_0 = m4t_flag_test(flags, 0, M4T_FLAG_ROUNDED)   ? 1 : 0;

    if (!got_sat_0) { printf("FAIL sat_flags: cell 0 should be SATURATED\n"); return 1; }
    if (got_sat_1)  { printf("FAIL sat_flags: cell 1 should NOT be SATURATED\n"); return 1; }
    if (got_rnd_0)  { printf("FAIL sat_flags: ROUNDED never set by matmul\n"); return 1; }
    return 0;
}

static int test_zero_dim(void) {
    int32_t Y[4] = { 99, 99, 99, 99 };
    int32_t X[4] = { 1, 1, 1, 1 };
    uint8_t W_packed[1] = { 0 };

    m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, NULL, 0, 4, 1);
    if (Y[0] != 99) { printf("FAIL zero_dim M=0\n"); return 1; }

    m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, NULL, 1, 4, 0);
    if (Y[0] != 99) { printf("FAIL zero_dim N=0\n"); return 1; }

    m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, NULL, 1, 0, 1);
    if (Y[0] != 0) { printf("FAIL zero_dim K=0: got=%d\n", Y[0]); return 1; }

    return 0;
}

static int test_determinism(void) {
    int M = 3, K = 64, N = 3;
    int32_t* X = malloc((size_t)M * K * sizeof(int32_t));
    m4t_trit_t* W = malloc((size_t)N * K);
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    uint8_t* W_packed = malloc((size_t)N * (size_t)Kp);
    int32_t Y1[9], Y2[9];

    g_rng = 0xa5a5a5a5u;
    for (int i = 0; i < M*K; i++) X[i] = rand_mtfp19();
    for (int i = 0; i < N*K; i++) W[i] = rand_trit();
    pack_trits_2d(W_packed, W, N, K);

    m4t_mtfp_ternary_matmul_bt(Y1, X, W_packed, NULL, M, K, N);
    m4t_mtfp_ternary_matmul_bt(Y2, X, W_packed, NULL, M, K, N);

    int ok = (memcmp(Y1, Y2, sizeof(Y1)) == 0);
    free(X); free(W); free(W_packed);
    if (!ok) {
        printf("FAIL determinism\n");
        return 1;
    }
    return 0;
}

int main(void) {
    if (test_small_golden())          return 1;
    if (test_random_vs_reference())   return 1;
    if (test_saturation_clamp())      return 1;
    if (test_saturation_flags())      return 1;
    if (test_zero_dim())              return 1;
    if (test_determinism())           return 1;
    printf("m4t_ternary_matmul: all 6 tests passed\n");
    return 0;
}
