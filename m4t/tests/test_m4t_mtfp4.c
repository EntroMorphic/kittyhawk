/*
 * test_m4t_mtfp4.c — tests for the SDOT ternary matmul + cell-width
 * conversions. Hand-derived golden values where small enough; bit-exact
 * int64 reference for the matmul (since SDOT must be exact per §8.4).
 *
 * Coverage:
 *   1. mtfp4_clamp_basic        — clamping at MTFP4 boundaries
 *   2. sdot_matmul_small        — golden-value 2x2x4 matmul
 *   3. sdot_matmul_large        — random K up to 1024 vs int64 reference
 *   4. sdot_matmul_max_bound    — extreme inputs, verify exact (no clamp)
 *   5. sdot_zero_dim            — M=0/N=0/K=0 edge cases
 *   6. widen_exact              — mtfp4_to_mtfp19 round-trip
 *   7. narrow_round             — mtfp19_to_mtfp4 round-to-nearest
 *   8. narrow_saturate          — mtfp19_to_mtfp4 saturation
 *   9. narrow_flags             — ROUNDED / SATURATED bits exact
 *  10. roundtrip_widen_narrow   — widen then narrow recovers exact value
 */

#include "m4t_mtfp4.h"
#include "m4t_mtfp.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT_EQ_I32(got, want, msg) do { \
    if ((int32_t)(got) != (int32_t)(want)) { \
        printf("FAIL %s: got=%d want=%d\n", (msg), (int)(got), (int)(want)); \
        return 1; \
    } \
} while (0)

/* ── xorshift32 RNG ─────────────────────────────────────────────────────── */
static uint32_t g_rng = 0xdeadbeefu;
static uint32_t xs32(void) {
    uint32_t x = g_rng; x ^= x << 13; x ^= x >> 17; x ^= x << 5; g_rng = x; return x;
}
static int8_t rand_mtfp4(void) {
    /* Uniform in [-40, 40]. */
    return (int8_t)((int)(xs32() % 81u) - 40);
}
static int8_t rand_trit(void) {
    /* Uniform in {-1, 0, +1}. */
    return (int8_t)((int)(xs32() % 3u) - 1);
}
static int32_t rand_mtfp19(void) {
    int64_t span = (int64_t)M4T_MTFP_MAX_VAL * 2 + 1;
    return (int32_t)((int64_t)(xs32() % (uint64_t)span) - (int64_t)M4T_MTFP_MAX_VAL);
}
static int rand_int(int lo, int hi) {
    return lo + (int)(xs32() % (uint32_t)(hi - lo + 1));
}

/* ── Tests ──────────────────────────────────────────────────────────────── */

static int test_clamp_basic(void) {
    ASSERT_EQ_I32(m4t_mtfp4_clamp(0), 0, "clamp(0)");
    ASSERT_EQ_I32(m4t_mtfp4_clamp(40), 40, "clamp(40)");
    ASSERT_EQ_I32(m4t_mtfp4_clamp(-40), -40, "clamp(-40)");
    ASSERT_EQ_I32(m4t_mtfp4_clamp(41), 40, "clamp(41)");
    ASSERT_EQ_I32(m4t_mtfp4_clamp(-41), -40, "clamp(-41)");
    ASSERT_EQ_I32(m4t_mtfp4_clamp(1000000), 40, "clamp(huge)");
    return 0;
}

/* Golden-value matmul: 2 rows × 4 cols × K=4. Computed by hand. */
static int test_sdot_matmul_small(void) {
    /* X = [[1, 2, 3, 4], [-1, -2, -3, -4]]  (M=2, K=4) */
    int8_t X[8] = { 1, 2, 3, 4,  -1, -2, -3, -4 };
    /* W = [[+1,-1, 0,+1], [-1, 0,+1,+1], [ 0, 0,+1, 0], [+1,+1,+1,+1]]  (N=4, K=4) */
    int8_t W[16] = {
         1, -1,  0,  1,
        -1,  0,  1,  1,
         0,  0,  1,  0,
         1,  1,  1,  1
    };
    /* Y = X @ W^T:
     *   Y[0,0] = 1·1 + 2·-1 + 3·0 + 4·1 =  3
     *   Y[0,1] = 1·-1 + 2·0 + 3·1 + 4·1 =  6
     *   Y[0,2] = 1·0 + 2·0 + 3·1 + 4·0 =  3
     *   Y[0,3] = 1·1 + 2·1 + 3·1 + 4·1 = 10
     *   Y[1,*] = -Y[0,*] = [-3, -6, -3, -10] */
    int32_t Y[8];
    m4t_mtfp4_sdot_matmul_bt(Y, X, W, 2, 4, 4);
    int32_t expected[8] = { 3, 6, 3, 10,  -3, -6, -3, -10 };
    for (int i = 0; i < 8; i++) {
        if (Y[i] != expected[i]) {
            printf("FAIL sdot_small[%d]: got=%d want=%d\n",
                   i, Y[i], expected[i]);
            return 1;
        }
    }
    return 0;
}

/* Random large-K matmul: bit-exact vs int64 reference. */
static int test_sdot_matmul_large(void) {
    g_rng = 0xc001d00du;
    for (int trial = 0; trial < 200; trial++) {
        int M = rand_int(1, 4);
        int N = rand_int(1, 4);
        int K = rand_int(16, 1024);    /* exercises the NEON loop and tail */
        int8_t* X = malloc((size_t)M * K);
        int8_t* W = malloc((size_t)N * K);
        int32_t* Y = malloc((size_t)M * N * sizeof(int32_t));
        int32_t* Yref = malloc((size_t)M * N * sizeof(int32_t));

        for (int i = 0; i < M * K; i++) X[i] = rand_mtfp4();
        for (int i = 0; i < N * K; i++) W[i] = rand_trit();

        m4t_mtfp4_sdot_matmul_bt(Y, X, W, M, K, N);

        /* int64 reference. */
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int64_t acc = 0;
                for (int k = 0; k < K; k++) {
                    acc += (int64_t)X[i*K+k] * (int64_t)W[j*K+k];
                }
                Yref[i*N+j] = (int32_t)acc;
            }
        }

        for (int i = 0; i < M*N; i++) {
            if (Y[i] != Yref[i]) {
                printf("FAIL sdot_large trial %d cell %d: got=%d ref=%d (M=%d K=%d N=%d)\n",
                       trial, i, Y[i], Yref[i], M, K, N);
                free(X); free(W); free(Y); free(Yref);
                return 1;
            }
        }
        free(X); free(W); free(Y); free(Yref);
    }
    return 0;
}

/* Worst-case extreme inputs: all X=±MAX_VAL_4, all W=±1, verify exact. */
static int test_sdot_matmul_max_bound(void) {
    int K = 4096;
    int8_t* X = malloc((size_t)K);
    int8_t* W = malloc((size_t)K);
    int32_t Y[1];

    /* All X = +40, all W = +1: Y = K · 40 = 163840 */
    for (int k = 0; k < K; k++) { X[k] = 40; W[k] = 1; }
    m4t_mtfp4_sdot_matmul_bt(Y, X, W, 1, K, 1);
    if (Y[0] != K * 40) {
        printf("FAIL max_bound +1: got=%d want=%d\n", Y[0], K * 40);
        free(X); free(W); return 1;
    }

    /* All X = +40, all W = -1: Y = -K · 40 */
    for (int k = 0; k < K; k++) W[k] = -1;
    m4t_mtfp4_sdot_matmul_bt(Y, X, W, 1, K, 1);
    if (Y[0] != -K * 40) {
        printf("FAIL max_bound -1: got=%d want=%d\n", Y[0], -K * 40);
        free(X); free(W); return 1;
    }

    /* Verify result is in MTFP19 range (we should never need to saturate). */
    if (Y[0] < -M4T_MTFP_MAX_VAL || Y[0] > M4T_MTFP_MAX_VAL) {
        printf("FAIL max_bound: |Y|=%d outside MTFP19\n", Y[0]);
        free(X); free(W); return 1;
    }

    free(X); free(W);
    return 0;
}

static int test_sdot_zero_dim(void) {
    int32_t Y[4] = { 99, 99, 99, 99 };  /* sentinel */
    int8_t X[4] = { 1, 1, 1, 1 };
    int8_t W[4] = { 1, 1, 1, 1 };

    /* M=0 — kernel must not touch Y. */
    m4t_mtfp4_sdot_matmul_bt(Y, X, W, 0, 4, 1);
    if (Y[0] != 99) { printf("FAIL zero_dim M=0\n"); return 1; }

    /* N=0 — ditto. */
    m4t_mtfp4_sdot_matmul_bt(Y, X, W, 1, 4, 0);
    if (Y[0] != 99) { printf("FAIL zero_dim N=0\n"); return 1; }

    /* K=0 — Y[0,0] = empty sum = 0. */
    m4t_mtfp4_sdot_matmul_bt(Y, X, W, 1, 0, 1);
    if (Y[0] != 0) { printf("FAIL zero_dim K=0: got=%d\n", Y[0]); return 1; }

    return 0;
}

static int test_widen_exact(void) {
    int8_t src[5] = { 0, 1, -1, 40, -40 };
    int32_t dst[5];
    m4t_mtfp4_to_mtfp19(dst, src, 5);
    int32_t expected[5] = { 0, 6561, -6561, 40 * 6561, -40 * 6561 };
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ_I32(dst[i], expected[i], "widen");
    }
    return 0;
}

static int test_narrow_round(void) {
    /* Verify base-3 round-to-nearest by examining values straddling the
     * rescale boundary. SCALE = 6561; halfway point is 3280.5 (integer
     * remainders never hit it).
     *   src  =     0:        q=0,    rem=0           → 0,    not rounded
     *   src  =  3280:        q=0,    rem=3280  <half → 0,    rounded
     *   src  =  3281:        q=0,    rem=3281  >half → 1,    rounded
     *   src  =  6561:        q=1,    rem=0           → 1,    not rounded
     *   src  = -3280:        q=0,    rem=-3280 <half → 0,    rounded
     *   src  = -3281:        q=0,    rem=-3281 >half →-1,    rounded */
    int32_t src[6] = { 0, 3280, 3281, 6561, -3280, -3281 };
    int8_t dst[6];
    m4t_mtfp19_to_mtfp4(dst, src, NULL, 6);
    int8_t expected[6] = { 0, 0, 1, 1, 0, -1 };
    for (int i = 0; i < 6; i++) {
        ASSERT_EQ_I32(dst[i], expected[i], "narrow_round");
    }
    return 0;
}

static int test_narrow_saturate(void) {
    /* Inputs that exceed MTFP4 range after rescale clamp at ±40. */
    int32_t src[3] = {
        40 * 6561 + 6561,   /* rounds to 41, saturates to 40 */
        -40 * 6561 - 6561,  /* rounds to -41, saturates to -40 */
        100 * 6561,         /* rounds to 100, saturates to 40 */
    };
    int8_t dst[3];
    m4t_mtfp19_to_mtfp4(dst, src, NULL, 3);
    ASSERT_EQ_I32(dst[0],  40, "sat +");
    ASSERT_EQ_I32(dst[1], -40, "sat -");
    ASSERT_EQ_I32(dst[2],  40, "sat huge");
    return 0;
}

static int test_narrow_flags(void) {
    /* Mix of rounded and saturated cells. */
    int32_t src[8] = {
        0,           /* exact zero, no flags */
        3280,        /* rounded, not saturated */
        6561,        /* exact, no flags */
        100 * 6561,  /* rounded ×0 (exact div), saturated */
        40 * 6561,   /* exact div, no flags (exactly 40 = MAX) */
        40 * 6561 + 1, /* rounded (rem=1), not saturated (still rounds to 40) */
        50 * 6561 + 10, /* rounded AND saturated */
        -3281,       /* rounded, not saturated */
    };
    int n = 8;
    size_t fb = M4T_FLAG_BYTES(n);
    int8_t dst[8];
    uint8_t flags[2];
    memset(flags, 0, fb);
    m4t_mtfp19_to_mtfp4(dst, src, flags, n);

    /* Per-cell expectations:
     *   0: no flags
     *   1: ROUNDED only
     *   2: no flags
     *   3: SATURATED only (100*6561 / 6561 = 100 exact, then clamp to 40)
     *   4: no flags
     *   5: ROUNDED only (40*6561+1 / 6561 = 40 with rem=1, no sat)
     *   6: ROUNDED + SATURATED
     *   7: ROUNDED only */
    uint8_t expected[8] = {
        0,
        M4T_FLAG_ROUNDED,
        0,
        M4T_FLAG_SATURATED,
        0,
        M4T_FLAG_ROUNDED,
        M4T_FLAG_ROUNDED | M4T_FLAG_SATURATED,
        M4T_FLAG_ROUNDED,
    };
    for (int i = 0; i < n; i++) {
        uint8_t got_sat = m4t_flag_test(flags, i, M4T_FLAG_SATURATED);
        uint8_t got_rnd = m4t_flag_test(flags, i, M4T_FLAG_ROUNDED);
        uint8_t got = (got_sat ? M4T_FLAG_SATURATED : 0) |
                      (got_rnd ? M4T_FLAG_ROUNDED   : 0);
        if (got != expected[i]) {
            printf("FAIL narrow_flags cell %d: got=0x%02x want=0x%02x\n",
                   i, got, expected[i]);
            return 1;
        }
    }
    return 0;
}

static int test_roundtrip_widen_narrow(void) {
    /* For values in MTFP4 range, widen then narrow recovers exact. */
    int8_t src[5] = { 0, 1, -1, 40, -40 };
    int32_t mid[5];
    int8_t back[5];
    m4t_mtfp4_to_mtfp19(mid, src, 5);
    m4t_mtfp19_to_mtfp4(back, mid, NULL, 5);
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ_I32(back[i], src[i], "roundtrip");
    }
    return 0;
}

int main(void) {
    if (test_clamp_basic())                return 1;
    if (test_sdot_matmul_small())          return 1;
    if (test_sdot_matmul_large())          return 1;
    if (test_sdot_matmul_max_bound())      return 1;
    if (test_sdot_zero_dim())              return 1;
    if (test_widen_exact())                return 1;
    if (test_narrow_round())               return 1;
    if (test_narrow_saturate())            return 1;
    if (test_narrow_flags())               return 1;
    if (test_roundtrip_widen_narrow())     return 1;
    printf("m4t_mtfp4: all 10 tests passed\n");
    (void)rand_mtfp19;  /* unused — kept for future tests */
    return 0;
}
