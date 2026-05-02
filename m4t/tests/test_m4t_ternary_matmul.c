/*
 * test_m4t_ternary_matmul.c — tests for MTFP19 × packed-ternary matmul.
 *
 * Coverage:
 *   1. small_golden          — hand-computed 2×4×3 matmul
 *   2. random_vs_reference   — random M/K/N matrices vs int64 reference,
 *                              including K not divisible by 16 (tail)
 *   3. long_k                — K=1M vs int64 reference; stress test
 *   4. saturation_clamp      — inputs that overflow MTFP19, verify clamp
 *   5. saturation_flags      — flags reflect saturation per cell
 *   6. partial_block         — trailing-block flag bits past M·N stay zero
 *   7. invalid_trit_code     — 0b11 (reserved) treated as zero per LUT
 *   8. zero_dim              — M=0 / N=0 / K=0 edge cases
 *   9. determinism           — same inputs → same outputs across calls
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

/* Long-K stress test. K=1M exercises the NEON inner loop hundreds of
 * thousands of times; bit-exact comparison to int64 reference catches
 * accumulator-handling bugs that small-K random tests would miss. */
static int test_long_k(void) {
    int K = 1000000;
    int M = 1, N = 1;
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    int32_t* X = malloc((size_t)K * sizeof(int32_t));
    m4t_trit_t* W = malloc((size_t)K);
    uint8_t* W_packed = malloc((size_t)Kp);

    /* Use MTFP4-magnitude operands so that K=1M sum stays in MTFP19 range
     * (K · 40 = 40M ≪ MTFP19_MAX). Tests the algorithm without forcing
     * saturation; saturation paths are covered by other tests. */
    g_rng = 0xc001d00du;
    for (int k = 0; k < K; k++) {
        X[k] = (int32_t)((int)(xs32() % 81u) - 40);
        W[k] = (m4t_trit_t)((int)(xs32() % 3u) - 1);
    }
    pack_trits_2d(W_packed, W, N, K);

    int32_t Y, Yref;
    m4t_mtfp_ternary_matmul_bt(&Y, X, W_packed, NULL, M, K, N);

    /* int64 reference. */
    int64_t acc = 0;
    for (int k = 0; k < K; k++) acc += (int64_t)X[k] * (int64_t)W[k];
    Yref = (int32_t)acc;

    int ok = (Y == Yref);
    free(X); free(W); free(W_packed);
    if (!ok) {
        printf("FAIL long_k: got=%d ref=%d\n", Y, Yref);
        return 1;
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
    uint8_t W_packed[1];   /* M4T_TRIT_PACKED_BYTES(4) = 1 */
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
    uint8_t W_packed[2];   /* 2 rows × M4T_TRIT_PACKED_BYTES(4)=1 byte each */
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

/* For M·N not a multiple of 4, the trailing flag byte has bits for
 * cells past the tensor's M·N output. Kernel must not touch those bits.
 * Test forces M·N ∈ {1, 2, 3, 5, 6, 7} and verifies trailing bits stay
 * zero after a saturating matmul (which would set unmasked bits if the
 * kernel mis-indexed). */
static int test_partial_block(void) {
    /* Pick M·N=5 so the second flag byte holds bits for cells 4-7,
     * but only cell 4 is in-tensor. Bits 2-7 of byte 1 must stay zero. */
    int M = 1, N = 5, K = 4;
    int32_t X[4] = {
        M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL, M4T_MTFP_MAX_VAL
    };
    /* All 5 output cells saturate. */
    m4t_trit_t W[5 * 4];
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) W[j * K + k] = 1;
    }
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    uint8_t W_packed[5 * 1];   /* Kp=1 for K=4 */
    (void)Kp;
    pack_trits_2d(W_packed, W, N, K);

    int32_t Y[5];
    size_t fb = M4T_FLAG_BYTES(M * N);
    uint8_t flags[2] = { 0xAA, 0xAA };  /* sentinel; kernel should overwrite */
    /* But we also pre-zero any trailing bits we'll inspect. Actually,
     * kernel uses sticky-OR so it DOESN'T overwrite — it ORs bits in.
     * So we initialize to zero and check only trailing bits stay zero. */
    memset(flags, 0, fb);

    m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, flags, M, K, N);

    /* M·N = 5, so flag bytes hold cells [0..3] in byte 0 (full), cell 4
     * in byte 1 bits 0-1 only. Bits 2-7 of byte 1 are out-of-tensor and
     * must remain zero. */
    int last_cells_used = M * N - 4;       /* = 1 */
    uint8_t used_mask = (uint8_t)((1u << (last_cells_used * 2)) - 1u);  /* 0x03 */
    uint8_t unused_mask = (uint8_t)~used_mask;                          /* 0xFC */
    if ((flags[fb - 1] & unused_mask) != 0) {
        printf("FAIL partial_block: trailing bits set "
               "(byte=0x%02x mask=0x%02x)\n",
               flags[fb - 1], unused_mask);
        return 1;
    }
    /* Sanity: cell 4 should also be flagged saturated. */
    if (!m4t_flag_test(flags, 4, M4T_FLAG_SATURATED)) {
        printf("FAIL partial_block: cell 4 SATURATED bit not set\n");
        return 1;
    }
    return 0;
}

/* Reserved trit code 0b11 should be treated as zero (per
 * M4T_TRIT_DECODE_LUT). Both NEON and scalar paths must agree. We
 * inject 0b11 into the packed weight buffer and verify the kernel
 * produces the same output as if those positions were 0b00. */
static int test_invalid_trit_code(void) {
    int K = 20;  /* > 16 so NEON path runs, plus a tail */
    int M = 1, N = 1;

    int32_t X[20];
    for (int k = 0; k < K; k++) X[k] = 1000;  /* uniform, easy to verify */

    /* Build two packed buffers:
     *   W_normal: alternating +1, 0  → expected sum = 10 * 1000 = 10000
     *   W_with_11: same but with code 0b11 (reserved) instead of 0b00
     *              for the zero positions → must produce SAME sum. */
    int Kp = M4T_TRIT_PACKED_BYTES(K);  /* (20+3)/4 = 5 */
    uint8_t W_normal[5];
    uint8_t W_with_11[5];

    /* Pack: alternating +1 (code 01), 0 (code 00 or 11). */
    memset(W_normal, 0, (size_t)Kp);
    memset(W_with_11, 0, (size_t)Kp);
    for (int k = 0; k < K; k++) {
        uint8_t code_01 = 0x01u;          /* +1 */
        uint8_t code_00 = 0x00u;          /* 0 */
        uint8_t code_11 = 0x03u;          /* reserved → treated as 0 */
        uint8_t code_normal = (k % 2 == 0) ? code_01 : code_00;
        uint8_t code_alt    = (k % 2 == 0) ? code_01 : code_11;
        W_normal[k >> 2]  |= (uint8_t)(code_normal << ((k & 3) * 2));
        W_with_11[k >> 2] |= (uint8_t)(code_alt    << ((k & 3) * 2));
    }

    int32_t Y_normal, Y_with_11;
    m4t_mtfp_ternary_matmul_bt(&Y_normal,  X, W_normal,  NULL, M, K, N);
    m4t_mtfp_ternary_matmul_bt(&Y_with_11, X, W_with_11, NULL, M, K, N);

    if (Y_normal != Y_with_11) {
        printf("FAIL invalid_trit_code: normal=%d with_11=%d\n",
               Y_normal, Y_with_11);
        return 1;
    }
    /* Sanity: expected sum = 10 cells of +1 × 1000 = 10000. */
    if (Y_normal != 10000) {
        printf("FAIL invalid_trit_code expected: got=%d want=10000\n", Y_normal);
        return 1;
    }
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

/* Ternary × ternary via the SDOT-delegating wrapper. Verifies
 *   m4t_ternary_dot_matmul_bt(Y, X, W, M, K, N)
 * produces the same int32 dot product as the reference open-coded
 * loop, across diverse shapes and seeds. */
static int test_ternary_dot_random_vs_reference(void) {
    g_rng = 0xa5a5a5a5u;
    for (int trial = 0; trial < 200; trial++) {
        int M = rand_int(1, 8);
        int N = rand_int(1, 8);
        int K = rand_int(1, 200);

        m4t_trit_t* X = malloc((size_t)M * K);
        m4t_trit_t* W = malloc((size_t)N * K);
        int32_t* Y = malloc((size_t)M * N * sizeof(int32_t));
        int32_t* Yref = malloc((size_t)M * N * sizeof(int32_t));

        for (int i = 0; i < M*K; i++) X[i] = rand_trit();
        for (int i = 0; i < N*K; i++) W[i] = rand_trit();

        m4t_ternary_dot_matmul_bt(Y, X, W, M, K, N);

        /* Reference: int32 sum of int8 × int8. */
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++) {
                    acc += (int32_t)X[i*K + k] * (int32_t)W[j*K + k];
                }
                Yref[i*N + j] = acc;
            }
        }

        for (int i = 0; i < M*N; i++) {
            if (Y[i] != Yref[i]) {
                printf("FAIL ternary_dot trial %d cell %d: "
                       "got=%d ref=%d (M=%d K=%d N=%d)\n",
                       trial, i, Y[i], Yref[i], M, K, N);
                free(X); free(W); free(Y); free(Yref);
                return 1;
            }
        }
        free(X); free(W); free(Y); free(Yref);
    }
    return 0;
}

int main(void) {
    if (test_small_golden())                       return 1;
    if (test_random_vs_reference())                return 1;
    if (test_long_k())                             return 1;
    if (test_saturation_clamp())                   return 1;
    if (test_saturation_flags())                   return 1;
    if (test_partial_block())                      return 1;
    if (test_invalid_trit_code())                  return 1;
    if (test_zero_dim())                           return 1;
    if (test_determinism())                        return 1;
    if (test_ternary_dot_random_vs_reference())    return 1;
    printf("m4t_ternary_matmul: all 10 tests passed\n");
    return 0;
}
