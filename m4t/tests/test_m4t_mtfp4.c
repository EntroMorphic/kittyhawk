/*
 * test_m4t_mtfp4.c — tests for MTFP4 routing cell and SDOT matmul.
 *
 * Golden values are hand-derived. No float.
 */

#include "m4t_types.h"
#include "m4t_mtfp4.h"
#include "m4t_mtfp.h"

#include <stdio.h>
#include <string.h>

#define ASSERT_EQ_I32(actual, expected, msg) do { \
    if ((actual) != (expected)) { \
        fprintf(stderr, "FAIL: %s — got %d, expected %d (line %d)\n", \
                (msg), (int)(actual), (int)(expected), __LINE__); \
        return 1; \
    } \
} while (0)

#define F4_MAX  M4T_MTFP4_MAX_VAL   /* 40 */
#define F4_S    M4T_MTFP4_SCALE     /* 9 */

/* ── Scalar arithmetic ─────────────────────────────────────────────────── */

static int test_f4_scalar(void) {
    /* add: 9 + 9 = 18 (real 1.0 + 1.0 = 2.0) */
    ASSERT_EQ_I32(m4t_mtfp4_add(F4_S, F4_S), 2 * F4_S, "f4 add 1+1");
    ASSERT_EQ_I32(m4t_mtfp4_sub(F4_S, F4_S), 0, "f4 sub 1-1");
    ASSERT_EQ_I32(m4t_mtfp4_neg(F4_S), -F4_S, "f4 neg");

    /* mul: 9 * 18 = 162 → (162 + 4) / 9 = 18 = 2S (1.0 * 2.0 = 2.0) */
    ASSERT_EQ_I32(m4t_mtfp4_mul(F4_S, 2 * F4_S), 2 * F4_S, "f4 mul 1*2");

    /* mul: 4 * 4 = 16 → (16 + 4) / 9 = 2 (real ≈ 0.44 * 0.44 ≈ 0.198 → cell 2 ≈ 0.222) */
    ASSERT_EQ_I32(m4t_mtfp4_mul(4, 4), 2, "f4 mul 4*4");

    /* trit: 9 * +1 = 9, 9 * -1 = -9, 9 * 0 = 0 */
    ASSERT_EQ_I32(m4t_mtfp4_mul_trit(F4_S,  1),  F4_S, "f4 trit*+1");
    ASSERT_EQ_I32(m4t_mtfp4_mul_trit(F4_S, -1), -F4_S, "f4 trit*-1");
    ASSERT_EQ_I32(m4t_mtfp4_mul_trit(F4_S,  0),  0, "f4 trit*0");
    return 0;
}

/* ── Saturation ────────────────────────────────────────────────────────── */

static int test_f4_saturation(void) {
    ASSERT_EQ_I32(m4t_mtfp4_add(F4_MAX, 1), F4_MAX, "f4 add sat");
    ASSERT_EQ_I32(m4t_mtfp4_add(-F4_MAX, -1), -F4_MAX, "f4 add sat neg");
    ASSERT_EQ_I32(m4t_mtfp4_sub(F4_MAX, -F4_MAX), F4_MAX, "f4 sub sat");
    ASSERT_EQ_I32(m4t_mtfp4_mul(F4_MAX, 2 * F4_S), F4_MAX, "f4 mul sat");
    return 0;
}

/* ── SDOT ternary matmul ───────────────────────────────────────────────── */

static int test_f4_sdot_small(void) {
    /* M=1, K=4, N=2.
     * X = [1, 2, 3, 4]  (MTFP4 cells, real ≈ 0.11, 0.22, 0.33, 0.44)
     * W[0] = [+1, +1, +1, +1]  → dot = 1+2+3+4 = 10
     * W[1] = [+1, -1, +1, -1]  → dot = 1-2+3-4 = -2
     */
    enum { M = 1, K = 4, N = 2 };
    m4t_mtfp4_t X[4] = { 1, 2, 3, 4 };
    m4t_trit_t W[8] = {
        1,  1,  1,  1,
        1, -1,  1, -1
    };
    m4t_mtfp4_t Y[2];
    m4t_mtfp4_sdot_matmul_bt(Y, X, W, M, K, N);

    ASSERT_EQ_I32(Y[0],  10, "sdot small[0]");
    ASSERT_EQ_I32(Y[1],  -2, "sdot small[1]");
    return 0;
}

static int test_f4_sdot_k32(void) {
    /* K=32: exercises NEON SDOT path (2 blocks of 16) + zero tail.
     * X[k] = 1 for all k. W[0] = all +1 → dot = 32. W[1] = all -1 → dot = -32.
     * Both clamp to ±40. */
    enum { M = 1, K = 32, N = 2 };
    m4t_mtfp4_t X[32];
    m4t_trit_t W[64];
    for (int k = 0; k < K; k++) { X[k] = 1; W[k] = 1; W[K + k] = -1; }

    m4t_mtfp4_t Y[2];
    m4t_mtfp4_sdot_matmul_bt(Y, X, W, M, K, N);

    ASSERT_EQ_I32(Y[0],  32, "sdot k32 all+1");
    ASSERT_EQ_I32(Y[1], -32, "sdot k32 all-1");
    return 0;
}

static int test_f4_sdot_k17_tail(void) {
    /* K=17: 1 SDOT block (16) + 1 scalar element.
     * X[k] = 2 for all k. W = all +1 → dot = 34.
     * Clamped to 40. */
    enum { M = 1, K = 17, N = 1 };
    m4t_mtfp4_t X[17];
    m4t_trit_t W[17];
    for (int k = 0; k < K; k++) { X[k] = 2; W[k] = 1; }

    m4t_mtfp4_t Y[1];
    m4t_mtfp4_sdot_matmul_bt(Y, X, W, M, K, N);

    ASSERT_EQ_I32(Y[0], 34, "sdot k17 tail");
    return 0;
}

static int test_f4_sdot_saturation(void) {
    /* K=64, X[k]=F4_MAX=40, W=all+1 → raw dot = 2560 → clamped to 40. */
    enum { M = 1, K = 64, N = 1 };
    m4t_mtfp4_t X[64];
    m4t_trit_t W[64];
    for (int k = 0; k < K; k++) { X[k] = F4_MAX; W[k] = 1; }

    m4t_mtfp4_t Y[1];
    m4t_mtfp4_sdot_matmul_bt(Y, X, W, M, K, N);

    ASSERT_EQ_I32(Y[0], F4_MAX, "sdot sat");
    return 0;
}

/* ── Conversion MTFP19 ↔ MTFP4 ────────────────────────────────────────── */

static int test_f4_conversion(void) {
    /* MTFP19 cell for real 1.0 = 59049. Convert to MTFP4: 59049/6561 = 9 = F4_S. */
    m4t_mtfp_t src19[4] = {
        M4T_MTFP_SCALE,       /* 1.0 → 9 */
        -M4T_MTFP_SCALE,      /* -1.0 → -9 */
        0,                     /* 0.0 → 0 */
        2 * M4T_MTFP_SCALE    /* 2.0 → 18 */
    };

    m4t_mtfp4_t dst4[4];
    m4t_mtfp19_to_mtfp4(dst4, src19, 4);

    ASSERT_EQ_I32(dst4[0],  F4_S, "19→4 +1.0");
    ASSERT_EQ_I32(dst4[1], -F4_S, "19→4 -1.0");
    ASSERT_EQ_I32(dst4[2],     0, "19→4 0.0");
    ASSERT_EQ_I32(dst4[3], 2*F4_S, "19→4 +2.0");

    /* Round-trip: 4→19→4 should be identity for representable values. */
    m4t_mtfp_t rt19[4];
    m4t_mtfp4_t rt4[4];
    m4t_mtfp4_to_mtfp19(rt19, dst4, 4);
    m4t_mtfp19_to_mtfp4(rt4, rt19, 4);

    for (int i = 0; i < 4; i++) {
        ASSERT_EQ_I32(rt4[i], dst4[i], "roundtrip 4→19→4");
    }

    /* Saturation on conversion: large MTFP19 value → clamps at F4_MAX. */
    m4t_mtfp_t big[1] = { M4T_MTFP_MAX_VAL };
    m4t_mtfp4_t clamped[1];
    m4t_mtfp19_to_mtfp4(clamped, big, 1);
    ASSERT_EQ_I32(clamped[0], F4_MAX, "19→4 sat");

    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    if (test_f4_scalar())         return 1;
    if (test_f4_saturation())     return 1;
    if (test_f4_sdot_small())     return 1;
    if (test_f4_sdot_k32())       return 1;
    if (test_f4_sdot_k17_tail())  return 1;
    if (test_f4_sdot_saturation()) return 1;
    if (test_f4_conversion())     return 1;
    printf("m4t_mtfp4: all tests passed\n");
    return 0;
}
