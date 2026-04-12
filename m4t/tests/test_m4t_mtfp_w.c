/*
 * test_m4t_mtfp_w.c — tests for MTFP39 wide-cell arithmetic.
 *
 * Mirrors the MTFP19 tests from test_m4t_smoke.c but for int64 cells
 * with 39-trit range. Golden values are hand-derived integers. No float.
 */

#include "m4t_types.h"
#include "m4t_mtfp_w.h"
#include "m4t_trit_pack.h"

#include <stdio.h>
#include <string.h>

#define ASSERT_EQ_I64(actual, expected, msg) do { \
    if ((actual) != (expected)) { \
        fprintf(stderr, "FAIL: %s — got %lld, expected %lld (line %d)\n", \
                (msg), (long long)(actual), (long long)(expected), __LINE__); \
        return 1; \
    } \
} while (0)

#define W_ONE  ((m4t_mtfp_w_t)M4T_MTFPW_SCALE)
#define W_TWO  ((m4t_mtfp_w_t)(2 * M4T_MTFPW_SCALE))
#define W_HALF ((m4t_mtfp_w_t)(M4T_MTFPW_SCALE / 2))
#define W_MAX  M4T_MTFPW_MAX_VAL

/* ── Scalar arithmetic ─────────────────────────────────────────────────── */

static int test_w_scalar(void) {
    ASSERT_EQ_I64(m4t_mtfp_w_add(W_ONE, W_ONE), W_TWO, "w add 1+1");
    ASSERT_EQ_I64(m4t_mtfp_w_sub(W_ONE, W_ONE), 0, "w sub 1-1");
    ASSERT_EQ_I64(m4t_mtfp_w_neg(W_ONE), -W_ONE, "w neg 1");
    ASSERT_EQ_I64(m4t_mtfp_w_mul(W_ONE, W_TWO), W_TWO, "w mul 1*2");

    /* 0.5 * 0.5 = 0.25. S=59049, half=29524.
     * prod = 29524 * 29524 = 871666576. + 29524 = 871696100. / 59049 = 14762. */
    ASSERT_EQ_I64(m4t_mtfp_w_mul(W_HALF, W_HALF), 14762, "w mul 0.5*0.5");

    ASSERT_EQ_I64(m4t_mtfp_w_mul_trit(W_ONE,  1),  W_ONE, "w trit*+1");
    ASSERT_EQ_I64(m4t_mtfp_w_mul_trit(W_ONE, -1), -W_ONE, "w trit*-1");
    ASSERT_EQ_I64(m4t_mtfp_w_mul_trit(W_ONE,  0),  0, "w trit*0");
    return 0;
}

/* ── Saturation ────────────────────────────────────────────────────────── */

static int test_w_saturation(void) {
    ASSERT_EQ_I64(m4t_mtfp_w_add(W_MAX, W_MAX), W_MAX, "w add sat");
    ASSERT_EQ_I64(m4t_mtfp_w_add(-W_MAX, -W_MAX), -W_MAX, "w add sat neg");
    ASSERT_EQ_I64(m4t_mtfp_w_sub(W_MAX, -W_MAX), W_MAX, "w sub sat");
    ASSERT_EQ_I64(m4t_mtfp_w_mul(W_MAX, W_TWO), W_MAX, "w mul sat");
    ASSERT_EQ_I64(m4t_mtfp_w_mul(W_MAX, -W_TWO), -W_MAX, "w mul sat neg");
    ASSERT_EQ_I64(m4t_mtfp_w_mul_trit(W_MAX, -1), -W_MAX, "w trit sat");
    return 0;
}

/* ── Vector ops ────────────────────────────────────────────────────────── */

static int test_w_vec_ops(void) {
    m4t_mtfp_w_t a[5], b[5], c[5];
    for (int i = 0; i < 5; i++) { a[i] = (m4t_mtfp_w_t)(i * 100); b[i] = (m4t_mtfp_w_t)(i * 10); }

    m4t_mtfp_w_vec_add(c, a, b, 5);
    for (int i = 0; i < 5; i++) ASSERT_EQ_I64(c[i], (m4t_mtfp_w_t)(i * 110), "w vec_add");

    memcpy(c, a, sizeof(a));
    m4t_mtfp_w_vec_add_inplace(c, b, 5);
    for (int i = 0; i < 5; i++) ASSERT_EQ_I64(c[i], (m4t_mtfp_w_t)(i * 110), "w vec_add_ip");

    for (int i = 0; i < 5; i++) c[i] = (m4t_mtfp_w_t)(i * 200);
    m4t_mtfp_w_vec_sub_inplace(c, b, 5);
    for (int i = 0; i < 5; i++) ASSERT_EQ_I64(c[i], (m4t_mtfp_w_t)(i * 190), "w vec_sub_ip");

    /* Saturation in vec_add */
    for (int i = 0; i < 5; i++) { a[i] = W_MAX; b[i] = W_MAX; }
    m4t_mtfp_w_vec_add(c, a, b, 5);
    for (int i = 0; i < 5; i++) ASSERT_EQ_I64(c[i], W_MAX, "w vec_add sat");

    m4t_mtfp_w_vec_zero(c, 5);
    for (int i = 0; i < 5; i++) ASSERT_EQ_I64(c[i], 0, "w vec_zero");
    return 0;
}

/* ── Vec scale ─────────────────────────────────────────────────────────── */

static int test_w_vec_scale(void) {
    const m4t_mtfp_w_t S = (m4t_mtfp_w_t)M4T_MTFPW_SCALE;
    m4t_mtfp_w_t src[3] = { S, -2*S, 0 };
    m4t_mtfp_w_t dst[3];
    m4t_mtfp_w_vec_scale(dst, src, 3*S, 3);
    ASSERT_EQ_I64(dst[0],  3*S, "w scale 1*3");
    ASSERT_EQ_I64(dst[1], -6*S, "w scale -2*3");
    ASSERT_EQ_I64(dst[2],    0, "w scale 0*3");
    return 0;
}

/* ── Odd-n vector test (W5) ────────────────────────────────────────────── */

static int test_w_vec_odd_n(void) {
    /* n=3: NEON processes 2 elements, scalar tail handles 1. */
    m4t_mtfp_w_t a[3] = { 100, 200, 300 };
    m4t_mtfp_w_t b[3] = { 10, 20, 30 };
    m4t_mtfp_w_t c[3];
    m4t_mtfp_w_vec_add(c, a, b, 3);
    ASSERT_EQ_I64(c[0], 110, "w odd_n[0]");
    ASSERT_EQ_I64(c[1], 220, "w odd_n[1]");
    ASSERT_EQ_I64(c[2], 330, "w odd_n[2]");

    /* n=1: pure scalar. */
    m4t_mtfp_w_t x[1] = { W_MAX };
    m4t_mtfp_w_t y[1] = { 1 };
    m4t_mtfp_w_t z[1];
    m4t_mtfp_w_vec_add(z, x, y, 1);
    ASSERT_EQ_I64(z[0], W_MAX, "w n1 sat");
    return 0;
}

/* ── Dense matmul ──────────────────────────────────────────────────────── */

static int test_w_matmul_bt(void) {
    /* X=[[2,0],[0,3]], W=[[1,2],[3,4]]. Y = X @ W^T.
     * Y[0,0]=2*1+0*2=2, Y[0,1]=2*3+0*4=6,
     * Y[1,0]=0*1+3*2=6, Y[1,1]=0*3+3*4=12. */
    const m4t_mtfp_w_t S = (m4t_mtfp_w_t)M4T_MTFPW_SCALE;
    m4t_mtfp_w_t X[4] = { 2*S, 0, 0, 3*S };
    m4t_mtfp_w_t W[4] = { S, 2*S, 3*S, 4*S };
    m4t_mtfp_w_t Y[4];

    m4t_mtfp_w_matmul_bt(Y, X, W, 2, 2, 2);
    ASSERT_EQ_I64(Y[0],  2*S, "w matmul[0,0]");
    ASSERT_EQ_I64(Y[1],  6*S, "w matmul[0,1]");
    ASSERT_EQ_I64(Y[2],  6*S, "w matmul[1,0]");
    ASSERT_EQ_I64(Y[3], 12*S, "w matmul[1,1]");
    return 0;
}

/* ── Ternary matmul ────────────────────────────────────────────────────── */

static int test_w_ternary_matmul_bt(void) {
    /* X = [1, 2, 3, 4] (cells = i*S), W = [[+1,+1,+1,+1], [+1,-1,+1,-1]]
     * Y[0] = 1+2+3+4 = 10, Y[1] = 1-2+3-4 = -2 */
    enum { M = 1, K = 4, N = 2, KP = 1 };
    const m4t_mtfp_w_t S = (m4t_mtfp_w_t)M4T_MTFPW_SCALE;

    m4t_mtfp_w_t X[K];
    for (int k = 0; k < K; k++) X[k] = (m4t_mtfp_w_t)((k + 1) * S);

    m4t_trit_t Wt[N * K];
    for (int k = 0; k < K; k++) Wt[k] = 1;
    for (int k = 0; k < K; k++) Wt[K + k] = (k & 1) ? -1 : 1;

    uint8_t Wp[N * KP];
    m4t_pack_trits_rowmajor(Wp, Wt, N, K);

    m4t_mtfp_w_t Y[N];
    m4t_mtfp_w_ternary_matmul_bt(Y, X, Wp, M, K, N);

    ASSERT_EQ_I64(Y[0],  10 * S, "w ternary [0]");
    ASSERT_EQ_I64(Y[1],  -2 * S, "w ternary [1]");
    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    if (test_w_scalar())             return 1;
    if (test_w_saturation())         return 1;
    if (test_w_vec_ops())            return 1;
    if (test_w_vec_scale())          return 1;
    if (test_w_vec_odd_n())          return 1;
    if (test_w_matmul_bt())          return 1;
    if (test_w_ternary_matmul_bt())  return 1;
    printf("m4t_mtfp_w: all tests passed\n");
    return 0;
}
