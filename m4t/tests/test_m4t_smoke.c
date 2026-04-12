/*
 * test_m4t_smoke.c — smoke tests for M4T's MTFP core and
 * packed-trit matmul.
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * Golden references are hand-computed integer constants. Absolutely no
 * float, no Python-generated expected values. Equality is exact.
 */

#include "m4t_types.h"
#include "m4t_mtfp.h"
#include "m4t_trit_pack.h"
#include "m4t_ternary_matmul.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FAIL(msg) do { \
    fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__); \
    return 1; \
} while (0)

#define ASSERT_EQ_I32(actual, expected, msg) do { \
    if ((actual) != (expected)) { \
        fprintf(stderr, "FAIL: %s — got %d, expected %d (line %d)\n", \
                (msg), (int)(actual), (int)(expected), __LINE__); \
        return 1; \
    } \
} while (0)

#define ASSERT_EQ_I64(actual, expected, msg) do { \
    if ((actual) != (expected)) { \
        fprintf(stderr, "FAIL: %s — got %lld, expected %lld (line %d)\n", \
                (msg), (long long)(actual), (long long)(expected), __LINE__); \
        return 1; \
    } \
} while (0)

/* ── Constants and helpers ─────────────────────────────────────────────── */

/* Real value 1 → MTFP cell 59049. */
#define M_ONE  ((m4t_mtfp_t)M4T_MTFP_SCALE)
#define M_TWO  ((m4t_mtfp_t)(2 * M4T_MTFP_SCALE))
#define M_HALF ((m4t_mtfp_t)(M4T_MTFP_SCALE / 2))
#define M_ZERO ((m4t_mtfp_t)0)

/* ── Scalar arithmetic ─────────────────────────────────────────────────── */

static int test_scalar_arith(void) {
    ASSERT_EQ_I32(m4t_mtfp_add(M_ONE, M_ONE), M_TWO, "add 1+1");
    ASSERT_EQ_I32(m4t_mtfp_sub(M_ONE, M_ONE), M_ZERO, "sub 1-1");
    ASSERT_EQ_I32(m4t_mtfp_neg(M_ONE), -M_ONE, "neg 1");

    /* 1.0 * 2.0 = 2.0 → cell 118098 */
    ASSERT_EQ_I32(m4t_mtfp_mul(M_ONE, M_TWO), M_TWO, "mul 1*2");
    /* 0.5 * 0.5 = 0.25 → cell 14762 (rounded from 0.25 * 59049 = 14762.25) */
    {
        m4t_mtfp_t q = m4t_mtfp_mul(M_HALF, M_HALF);
        /* Expected: round((0.5*S)*(0.5*S)/S) = round(S/4) = round(14762.25) = 14762 */
        ASSERT_EQ_I32(q, 14762, "mul 0.5*0.5");
    }

    /* trit multiply */
    ASSERT_EQ_I32(m4t_mtfp_mul_trit(M_ONE,  1),  M_ONE, "trit * +1");
    ASSERT_EQ_I32(m4t_mtfp_mul_trit(M_ONE, -1), -M_ONE, "trit * -1");
    ASSERT_EQ_I32(m4t_mtfp_mul_trit(M_ONE,  0),  M_ZERO, "trit *  0");
    return 0;
}

/* ── mtfp_mul boundary / saturation ───────────────────────────────────── */

static int test_mul_boundary(void) {
    const m4t_mtfp_t S = (m4t_mtfp_t)M4T_MTFP_SCALE;
    const m4t_mtfp_t MAX_CELL = M4T_MTFP_MAX_VAL;

    /* Identity: x * 1 == x, at the upper and lower boundary. */
    ASSERT_EQ_I32(m4t_mtfp_mul(MAX_CELL, S),  MAX_CELL, "mul +MAX * 1");
    ASSERT_EQ_I32(m4t_mtfp_mul(-MAX_CELL, S), -MAX_CELL, "mul -MAX * 1");

    /* Doubling MAX must saturate to MAX (not wrap, not exceed spec). */
    ASSERT_EQ_I32(m4t_mtfp_mul(MAX_CELL, 2*S),  MAX_CELL, "mul +MAX * 2 saturates");
    ASSERT_EQ_I32(m4t_mtfp_mul(-MAX_CELL, 2*S), -MAX_CELL, "mul -MAX * 2 saturates");

    /* Large overshoot must still saturate. */
    ASSERT_EQ_I32(m4t_mtfp_mul(MAX_CELL, 10*S),  MAX_CELL, "mul +MAX * 10 saturates");
    ASSERT_EQ_I32(m4t_mtfp_mul(-MAX_CELL, 10*S), -MAX_CELL, "mul -MAX * 10 saturates");

    /* Mixed sign saturation. */
    ASSERT_EQ_I32(m4t_mtfp_mul(MAX_CELL, -2*S), -MAX_CELL, "mul +MAX * -2 saturates");
    ASSERT_EQ_I32(m4t_mtfp_mul(-MAX_CELL, -2*S),  MAX_CELL, "mul -MAX * -2 saturates");
    return 0;
}

/* ── add / sub saturation (NEW2) ──────────────────────────────────────── */

static int test_add_sub_saturation(void) {
    const m4t_mtfp_t MAX_CELL = M4T_MTFP_MAX_VAL;

    /* Scalar add saturates on overflow of spec. */
    ASSERT_EQ_I32(m4t_mtfp_add(MAX_CELL, MAX_CELL),   MAX_CELL, "add +MAX +MAX");
    ASSERT_EQ_I32(m4t_mtfp_add(-MAX_CELL, -MAX_CELL), -MAX_CELL, "add -MAX -MAX");

    /* Scalar sub saturates. */
    ASSERT_EQ_I32(m4t_mtfp_sub(MAX_CELL, -MAX_CELL),  MAX_CELL, "sub +MAX -(-MAX)");
    ASSERT_EQ_I32(m4t_mtfp_sub(-MAX_CELL, MAX_CELL), -MAX_CELL, "sub -MAX -(+MAX)");

    /* Non-overflow case unaffected. */
    ASSERT_EQ_I32(m4t_mtfp_add(100, 200), 300, "add small");
    ASSERT_EQ_I32(m4t_mtfp_sub(300, 200), 100, "sub small");
    return 0;
}

static int test_vec_add_saturation(void) {
    /* Sized for NEON head (4 lanes) + scalar tail (1 lane). */
    m4t_mtfp_t a[5], b[5], c[5];
    for (int i = 0; i < 5; i++) {
        a[i] = M4T_MTFP_MAX_VAL;
        b[i] = M4T_MTFP_MAX_VAL;
    }

    m4t_mtfp_vec_add(c, a, b, 5);
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ_I32(c[i], M4T_MTFP_MAX_VAL, "vec_add sat (NEON+tail)");
    }

    /* Negative saturation. */
    for (int i = 0; i < 5; i++) { a[i] = -M4T_MTFP_MAX_VAL; b[i] = -M4T_MTFP_MAX_VAL; }
    m4t_mtfp_vec_add(c, a, b, 5);
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ_I32(c[i], -M4T_MTFP_MAX_VAL, "vec_add sat neg");
    }

    /* In-place. */
    for (int i = 0; i < 5; i++) { c[i] = M4T_MTFP_MAX_VAL; b[i] = M4T_MTFP_MAX_VAL; }
    m4t_mtfp_vec_add_inplace(c, b, 5);
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ_I32(c[i], M4T_MTFP_MAX_VAL, "vec_add_inplace sat");
    }
    return 0;
}

/* ── Vector arithmetic ─────────────────────────────────────────────────── */

static int test_vector_ops(void) {
    m4t_mtfp_t a[17], b[17], c[17];
    for (int i = 0; i < 17; i++) {
        a[i] = (m4t_mtfp_t)(i * 100);
        b[i] = (m4t_mtfp_t)(i * 10);
    }

    m4t_mtfp_vec_add(c, a, b, 17);
    for (int i = 0; i < 17; i++) {
        ASSERT_EQ_I32(c[i], (m4t_mtfp_t)(i * 110), "vec_add");
    }

    memcpy(c, a, sizeof(a));
    m4t_mtfp_vec_add_inplace(c, b, 17);
    for (int i = 0; i < 17; i++) {
        ASSERT_EQ_I32(c[i], (m4t_mtfp_t)(i * 110), "vec_add_inplace");
    }

    m4t_mtfp_vec_zero(c, 17);
    for (int i = 0; i < 17; i++) {
        ASSERT_EQ_I32(c[i], 0, "vec_zero");
    }
    return 0;
}

/* ── vec_scale and bias_add (N5) ──────────────────────────────────────── */

static int test_vec_scale(void) {
    const m4t_mtfp_t S = (m4t_mtfp_t)M4T_MTFP_SCALE;
    m4t_mtfp_t src[5] = { S, 2*S, -S, 0, 3*S };
    m4t_mtfp_t dst[5];

    /* Scale by 2.0 (cell = 2*S). Result = input * 2. */
    m4t_mtfp_vec_scale(dst, src, 2*S, 5);
    ASSERT_EQ_I32(dst[0], 2*S,  "scale 1*2");
    ASSERT_EQ_I32(dst[1], 4*S,  "scale 2*2");
    ASSERT_EQ_I32(dst[2], -2*S, "scale -1*2");
    ASSERT_EQ_I32(dst[3], 0,    "scale 0*2");
    ASSERT_EQ_I32(dst[4], 6*S,  "scale 3*2");

    /* Scale by 0 → all zero. */
    m4t_mtfp_vec_scale(dst, src, 0, 5);
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ_I32(dst[i], 0, "scale *0");
    }
    return 0;
}

static int test_bias_add(void) {
    const m4t_mtfp_t S = (m4t_mtfp_t)M4T_MTFP_SCALE;
    /* 2 batch rows, dim=3.  x = [[1,2,3],[4,5,6]], bias = [10,20,30] */
    m4t_mtfp_t x[6] = { 1*S, 2*S, 3*S, 4*S, 5*S, 6*S };
    m4t_mtfp_t b[3] = { 10*S, 20*S, 30*S };

    m4t_mtfp_bias_add(x, b, 2, 3);
    ASSERT_EQ_I32(x[0], 11*S, "bias row0 col0");
    ASSERT_EQ_I32(x[1], 22*S, "bias row0 col1");
    ASSERT_EQ_I32(x[2], 33*S, "bias row0 col2");
    ASSERT_EQ_I32(x[3], 14*S, "bias row1 col0");
    ASSERT_EQ_I32(x[4], 25*S, "bias row1 col1");
    ASSERT_EQ_I32(x[5], 36*S, "bias row1 col2");
    return 0;
}

/* ── Integer square root ───────────────────────────────────────────────── */

static int test_isqrt(void) {
    ASSERT_EQ_I64(m4t_isqrt64(0), 0, "isqrt 0");
    ASSERT_EQ_I64(m4t_isqrt64(1), 1, "isqrt 1");
    ASSERT_EQ_I64(m4t_isqrt64(4), 2, "isqrt 4");
    ASSERT_EQ_I64(m4t_isqrt64(9), 3, "isqrt 9");
    ASSERT_EQ_I64(m4t_isqrt64(16), 4, "isqrt 16");
    ASSERT_EQ_I64(m4t_isqrt64(100), 10, "isqrt 100");
    ASSERT_EQ_I64(m4t_isqrt64(1000000), 1000, "isqrt 1e6");
    /* floor(sqrt(10)) = 3 */
    ASSERT_EQ_I64(m4t_isqrt64(10), 3, "isqrt 10 (floor)");
    return 0;
}

/* ── Pack / unpack round-trip ──────────────────────────────────────────── */

static int test_pack_roundtrip(void) {
    const int N = 19;  /* not a multiple of 4 — exercises the tail */
    const m4t_trit_t src[19] = {
        1, 0, -1, 1, 0, 1, -1, -1, 0, 1,
        -1, 0, 1, 1, -1, 0, 1, -1, 0
    };
    uint8_t packed[M4T_TRIT_PACKED_BYTES(19)];
    m4t_trit_t round[19];

    m4t_pack_trits_1d(packed, src, N);
    m4t_unpack_trits_1d(round, packed, N);
    for (int i = 0; i < N; i++) {
        ASSERT_EQ_I32(round[i], src[i], "pack/unpack trit");
    }
    return 0;
}

/* ── Popcount distance ─────────────────────────────────────────────────── */

static int test_popcount_dist(void) {
    /* 17 bytes: first 16 exercise NEON path, last 1 exercises scalar tail. */
    uint8_t a[17] = { 0 };
    uint8_t b[17] = { 0 };
    uint8_t mask[17];
    memset(mask, 0xFF, sizeof(mask));

    /* All zero → distance 0. */
    ASSERT_EQ_I32(m4t_popcount_dist(a, b, mask, 17), 0, "dist 0");

    /* Differ by 1 bit in byte 0. */
    b[0] = 0x01;
    ASSERT_EQ_I32(m4t_popcount_dist(a, b, mask, 17), 1, "dist 1 in byte 0");

    /* Differ by 4 bits in byte 16 (scalar tail). */
    b[16] = 0x0F;
    ASSERT_EQ_I32(m4t_popcount_dist(a, b, mask, 17), 1 + 4, "dist incl tail");

    /* Mask suppresses byte 16 mismatch. */
    mask[16] = 0x00;
    ASSERT_EQ_I32(m4t_popcount_dist(a, b, mask, 17), 1, "mask suppresses tail");
    return 0;
}

/* ── Ternary matmul: MTFP × packed trits ───────────────────────────────── */

static int test_ternary_matmul_bt_small(void) {
    /* Y[1,2] = X[1,8] @ W^T where W is [2,8] ternary.
     *
     * X = [1, 2, 3, 4, 5, 6, 7, 8] (in MTFP units: cell = i*S)
     * W[0] = [+1, +1, +1, +1, +1, +1, +1, +1]  → row dot = 1+2+3+4+5+6+7+8 = 36
     * W[1] = [+1, -1, +1, -1, +1, -1, +1, -1]  → row dot = 1-2+3-4+5-6+7-8 = -4
     */
    enum { M = 1, K = 8, N = 2 };
    enum { KP = M4T_TRIT_PACKED_BYTES(K) };

    m4t_mtfp_t X[K];
    for (int k = 0; k < K; k++) X[k] = (m4t_mtfp_t)((k + 1) * M4T_MTFP_SCALE);

    m4t_trit_t W[N * K];
    for (int k = 0; k < K; k++) W[k] = 1;
    for (int k = 0; k < K; k++) W[K + k] = (k & 1) ? -1 : 1;

    uint8_t W_packed[N * KP];
    m4t_pack_trits_rowmajor(W_packed, W, N, K);

    m4t_mtfp_t Y[2];
    m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, M, K, N);

    ASSERT_EQ_I32(Y[0], (m4t_mtfp_t)(36 * M4T_MTFP_SCALE), "matmul row 0");
    ASSERT_EQ_I32(Y[1], (m4t_mtfp_t)(-4 * M4T_MTFP_SCALE), "matmul row 1");
    return 0;
}

static int test_ternary_matmul_bt_k32(void) {
    /* K=32 crosses the 16-trit NEON block boundary exactly twice plus zero
     * tail. Validates that accumulators compose correctly across blocks.
     *
     * X[k] = 1 (cell = S) for all k
     * W[0] = all +1  → dot = 32
     * W[1] = all -1  → dot = -32
     * W[2] = +1 on even k, -1 on odd → dot = 0 (16*1 + 16*-1)
     */
    enum { M = 1, K = 32, N = 3 };
    enum { KP = M4T_TRIT_PACKED_BYTES(K) };

    m4t_mtfp_t X[K];
    for (int k = 0; k < K; k++) X[k] = (m4t_mtfp_t)M4T_MTFP_SCALE;

    m4t_trit_t W[N * K];
    for (int k = 0; k < K; k++) {
        W[0 * K + k] =  1;
        W[1 * K + k] = -1;
        W[2 * K + k] = (k & 1) ? -1 : 1;
    }

    uint8_t W_packed[N * KP];
    m4t_pack_trits_rowmajor(W_packed, W, N, K);

    m4t_mtfp_t Y[3];
    m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, M, K, N);

    ASSERT_EQ_I32(Y[0], (m4t_mtfp_t)( 32 * M4T_MTFP_SCALE), "k32 all +1");
    ASSERT_EQ_I32(Y[1], (m4t_mtfp_t)(-32 * M4T_MTFP_SCALE), "k32 all -1");
    ASSERT_EQ_I32(Y[2], M_ZERO, "k32 alternating");
    return 0;
}

/* ── Dense MTFP × MTFP matmul (T2) ─────────────────────────────────────── */

static int test_dense_matmul_2x2(void) {
    /* X = [[2, 0], [0, 3]], W = [[1, 2], [3, 4]]  (real values, via MTFP)
     * Y = X @ W = [[2, 4], [9, 12]] (real)
     */
    const m4t_mtfp_t S = (m4t_mtfp_t)M4T_MTFP_SCALE;
    m4t_mtfp_t X[4] = { 2*S, 0, 0, 3*S };
    m4t_mtfp_t W[4] = { S, 2*S, 3*S, 4*S };
    m4t_mtfp_t Y[4] = { 0 };

    m4t_mtfp_matmul(Y, X, W, 2, 2, 2);
    ASSERT_EQ_I32(Y[0],  2*S, "dense Y[0,0]");
    ASSERT_EQ_I32(Y[1],  4*S, "dense Y[0,1]");
    ASSERT_EQ_I32(Y[2],  9*S, "dense Y[1,0]");
    ASSERT_EQ_I32(Y[3], 12*S, "dense Y[1,1]");
    return 0;
}

static int test_dense_matmul_bt_2x2(void) {
    /* X = [[2, 0], [0, 3]], W_bt viewed as [N=2, K=2] = [[1, 2], [3, 4]]
     * Y = X @ W^T
     * Y[0,0] = 2*1 + 0*2 = 2
     * Y[0,1] = 2*3 + 0*4 = 6
     * Y[1,0] = 0*1 + 3*2 = 6
     * Y[1,1] = 0*3 + 3*4 = 12
     */
    const m4t_mtfp_t S = (m4t_mtfp_t)M4T_MTFP_SCALE;
    m4t_mtfp_t X[4] = { 2*S, 0, 0, 3*S };
    m4t_mtfp_t W[4] = { S, 2*S, 3*S, 4*S };
    m4t_mtfp_t Y[4] = { 0 };

    m4t_mtfp_matmul_bt(Y, X, W, 2, 2, 2);
    ASSERT_EQ_I32(Y[0],  2*S, "dense_bt Y[0,0]");
    ASSERT_EQ_I32(Y[1],  6*S, "dense_bt Y[0,1]");
    ASSERT_EQ_I32(Y[2],  6*S, "dense_bt Y[1,0]");
    ASSERT_EQ_I32(Y[3], 12*S, "dense_bt Y[1,1]");
    return 0;
}

/* ── Ternary matmul tail + multi-row (T6, T7) ──────────────────────────── */

static int test_ternary_matmul_bt_k17_tail(void) {
    /* K=17: NEON block runs once (k=0..15), scalar tail handles k=16.
     * X[k] = 1 (cell = S) for all k.
     *   W[0] = all +1               → dot = 17
     *   W[1] = all -1               → dot = -17
     *   W[2] = +1 on even k, -1 odd → dot = 9 - 8 = 1  (9 evens in [0..16])
     */
    const int M = 1, K = 17, N = 3;
    const m4t_mtfp_t S = (m4t_mtfp_t)M4T_MTFP_SCALE;
    enum { KP17 = 5 };  /* (17 + 3) / 4 = 5 */

    m4t_mtfp_t X[17];
    for (int k = 0; k < K; k++) X[k] = S;

    m4t_trit_t W[3 * 17];
    for (int k = 0; k < K; k++) {
        W[0 * K + k] =  1;
        W[1 * K + k] = -1;
        W[2 * K + k] = (k & 1) ? -1 : 1;
    }

    uint8_t W_packed[3 * KP17];
    m4t_pack_trits_rowmajor(W_packed, W, N, K);

    m4t_mtfp_t Y[3];
    m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, M, K, N);

    ASSERT_EQ_I32(Y[0],  17 * S, "k17 all +1");
    ASSERT_EQ_I32(Y[1], -17 * S, "k17 all -1");
    ASSERT_EQ_I32(Y[2],      S, "k17 alternating");
    return 0;
}

static int test_ternary_matmul_bt_m3(void) {
    /* M=3 exercises the multi-row outer loop.
     * K=8, N=2.
     *   X[0] = [1]*8
     *   X[1] = [2]*8
     *   X[2] = [0]*8
     *   W[0] = all +1
     *   W[1] = all -1
     * Row dots:
     *   (i=0): +8, -8
     *   (i=1): +16, -16
     *   (i=2):   0,   0
     */
    const int M = 3, K = 8, N = 2;
    const m4t_mtfp_t S = (m4t_mtfp_t)M4T_MTFP_SCALE;
    enum { KP8 = 2 };  /* 8 / 4 = 2 */

    m4t_mtfp_t X[24];
    for (int k = 0; k < K; k++) X[0 * K + k] = S;
    for (int k = 0; k < K; k++) X[1 * K + k] = 2 * S;
    for (int k = 0; k < K; k++) X[2 * K + k] = 0;

    m4t_trit_t W[2 * 8];
    for (int k = 0; k < K; k++) { W[0 * K + k] =  1; W[1 * K + k] = -1; }

    uint8_t W_packed[2 * KP8];
    m4t_pack_trits_rowmajor(W_packed, W, N, K);

    m4t_mtfp_t Y[6];
    m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, M, K, N);

    ASSERT_EQ_I32(Y[0*N + 0],   8 * S, "m3 row0 col0");
    ASSERT_EQ_I32(Y[0*N + 1],  -8 * S, "m3 row0 col1");
    ASSERT_EQ_I32(Y[1*N + 0],  16 * S, "m3 row1 col0");
    ASSERT_EQ_I32(Y[1*N + 1], -16 * S, "m3 row1 col1");
    ASSERT_EQ_I32(Y[2*N + 0],       0, "m3 row2 col0");
    ASSERT_EQ_I32(Y[2*N + 1],       0, "m3 row2 col1");
    return 0;
}

/* ── LayerNorm sanity ──────────────────────────────────────────────────── */

/* Symmetric-input LayerNorm with analytically derived integer expected
 * output. This test is designed to bracket the rounding-asymmetry bug
 * (NEW1) in the scale multiply: the broken code rounds negative
 * intermediates toward zero, off by one ULP.
 *
 * Derivation (S = 59049):
 *   x          = [-3S, -S, +S, +3S]              (real [-3, -1, +1, +3])
 *   mean       = 0
 *   var_sum    = 9S^2 + S^2 + S^2 + 9S^2 = 20 S^2
 *   var        = 20 S^2 / 4 = 5 S^2
 *   isqrt(5 S^2) = 132037   (132037^2 = 17,433,769,369;
 *                            132038^2 = 17,434,033,444; target 17,433,922,005)
 *   rstd       = S^2 / 132037 = 26407  (int div, remainder 83342)
 *   norm[k]    = (k * S) * rstd / S = k * 26407
 *   norm       = [-79221, -26407, +26407, +79221]
 *
 * With weight = S (real 1.0) and bias = 0, `scaled = norm` iff the
 * scale-multiply rounds symmetrically around zero. Under the pre-fix
 * asymmetric rounding, the two negative lanes come out as -79220 and
 * -26406 — that's the bug.
 */
static int test_layernorm_symmetric_row(void) {
    const int rows = 1, cols = 4;
    const m4t_mtfp_t S = (m4t_mtfp_t)M4T_MTFP_SCALE;

    m4t_mtfp_t x[4] = { -3*S, -S, +S, +3*S };
    m4t_mtfp_t w[4] = {  S,  S,  S,  S };
    m4t_mtfp_t b[4] = {  0,  0,  0,  0 };
    m4t_mtfp_t y[4];

    m4t_mtfp_layernorm(y, x, w, b, (m4t_mtfp_t)0, rows, cols);

    ASSERT_EQ_I32(y[0], -79221, "ln symmetric y[0]");
    ASSERT_EQ_I32(y[1], -26407, "ln symmetric y[1]");
    ASSERT_EQ_I32(y[2], +26407, "ln symmetric y[2]");
    ASSERT_EQ_I32(y[3], +79221, "ln symmetric y[3]");
    return 0;
}

/* Same row, but with a nonzero bias. Output = y_symmetric + bias_j. */
static int test_layernorm_symmetric_row_with_bias(void) {
    const int rows = 1, cols = 4;
    const m4t_mtfp_t S = (m4t_mtfp_t)M4T_MTFP_SCALE;

    m4t_mtfp_t x[4] = { -3*S, -S, +S, +3*S };
    m4t_mtfp_t w[4] = {  S,  S,  S,  S };
    m4t_mtfp_t b[4] = {  100, -200, 300, -400 };
    m4t_mtfp_t y[4];

    m4t_mtfp_layernorm(y, x, w, b, (m4t_mtfp_t)0, rows, cols);

    ASSERT_EQ_I32(y[0], -79221 + 100, "ln sym+bias y[0]");
    ASSERT_EQ_I32(y[1], -26407 - 200, "ln sym+bias y[1]");
    ASSERT_EQ_I32(y[2], +26407 + 300, "ln sym+bias y[2]");
    ASSERT_EQ_I32(y[3], +79221 - 400, "ln sym+bias y[3]");
    return 0;
}

/* Zero cols must not crash. */
static int test_layernorm_zero_cols(void) {
    m4t_mtfp_t y[1] = { 0xdead };
    m4t_mtfp_layernorm(y, y, y, y, (m4t_mtfp_t)1, 0, 0);
    /* Early return; y is untouched. Passing this line means no crash. */
    return 0;
}

static int test_layernorm_constant_row(void) {
    /* A row of all-equal values has zero variance; LayerNorm output equals
     * bias (normalized part is 0, scale·0 + bias = bias). */
    const int rows = 1, cols = 8;
    m4t_mtfp_t x[8], y[8], w[8], b[8];
    for (int j = 0; j < cols; j++) {
        x[j] = (m4t_mtfp_t)(5 * M4T_MTFP_SCALE);
        w[j] = M_ONE;
        b[j] = (m4t_mtfp_t)(j * M4T_MTFP_SCALE);
    }

    /* eps in MTFP real units = 1e-5 * S ≈ 0 at this resolution; use 1 cell. */
    m4t_mtfp_layernorm(y, x, w, b, (m4t_mtfp_t)1, rows, cols);

    for (int j = 0; j < cols; j++) {
        ASSERT_EQ_I32(y[j], b[j], "ln constant row → bias");
    }
    return 0;
}

/* ── Boundary at new MAX_VAL (R5) ──────────────────────────────────────── */

static int test_max_val_boundary(void) {
    const m4t_mtfp_t MAX_CELL = M4T_MTFP_MAX_VAL;  /* 581,130,733 = (3^19-1)/2 */

    /* add at boundary saturates */
    ASSERT_EQ_I32(m4t_mtfp_add(MAX_CELL, 1), MAX_CELL, "add MAX+1 sat");
    ASSERT_EQ_I32(m4t_mtfp_add(-MAX_CELL, -1), -MAX_CELL, "add -MAX-1 sat");

    /* sub at boundary saturates */
    ASSERT_EQ_I32(m4t_mtfp_sub(-MAX_CELL, 1), -MAX_CELL, "sub -MAX-1 sat");

    /* mul at boundary: MAX * 1.0 = MAX */
    const m4t_mtfp_t S = (m4t_mtfp_t)M4T_MTFP_SCALE;
    ASSERT_EQ_I32(m4t_mtfp_mul(MAX_CELL, S), MAX_CELL, "mul MAX*1.0");
    ASSERT_EQ_I32(m4t_mtfp_mul(-MAX_CELL, S), -MAX_CELL, "mul -MAX*1.0");

    /* mul_trit at boundary (R7) */
    ASSERT_EQ_I32(m4t_mtfp_mul_trit(MAX_CELL, 1), MAX_CELL, "mul_trit MAX*+1");
    ASSERT_EQ_I32(m4t_mtfp_mul_trit(MAX_CELL, -1), -MAX_CELL, "mul_trit MAX*-1");
    ASSERT_EQ_I32(m4t_mtfp_mul_trit(MAX_CELL, 0), 0, "mul_trit MAX*0");
    ASSERT_EQ_I32(m4t_mtfp_mul_trit(-MAX_CELL, -1), MAX_CELL, "mul_trit -MAX*-1");

    /* clamp64 at boundary */
    ASSERT_EQ_I32(m4t_mtfp_clamp64((int64_t)MAX_CELL + 1), MAX_CELL, "clamp64 MAX+1");
    ASSERT_EQ_I32(m4t_mtfp_clamp64(-(int64_t)MAX_CELL - 1), -MAX_CELL, "clamp64 -MAX-1");
    ASSERT_EQ_I32(m4t_mtfp_clamp64((int64_t)MAX_CELL), MAX_CELL, "clamp64 exact MAX");

    /* vec_add at boundary */
    m4t_mtfp_t va[5], vb[5], vc[5];
    for (int i = 0; i < 5; i++) { va[i] = MAX_CELL; vb[i] = 1; }
    m4t_mtfp_vec_add(vc, va, vb, 5);
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ_I32(vc[i], MAX_CELL, "vec_add MAX+1 sat");
    }
    return 0;
}

/* ── fan_in_normalize rounding (R12) ──────────────────────────────────── */

static int test_fan_in_normalize_rounding(void) {
    /* fan_in = 4 → isqrt(4) = 2 → divide each cell by 2.
     * For positive cells: round-half-away-from-zero.
     * 3 / 2 = 1 (round of 1.5 → 2? No: (3 + 1)/2 = 2. Yes, with half=1.)
     * -3 / 2 = -2 (symmetric: (-3 - 1)/2 = -2. Correct.)
     */
    m4t_mtfp_t x[4] = { 3, -3, 4, -4 };
    m4t_mtfp_fan_in_normalize(x, 4, 4);

    /* 3: (3 + 1) / 2 = 2 */
    ASSERT_EQ_I32(x[0], 2, "fan_in +3/2 rounds up");
    /* -3: (-3 - 1) / 2 = -2 */
    ASSERT_EQ_I32(x[1], -2, "fan_in -3/2 rounds down");
    /* 4: (4 + 1) / 2 = 2 */
    ASSERT_EQ_I32(x[2], 2, "fan_in +4/2 exact");
    /* -4: (-4 - 1) / 2 = -2 */
    ASSERT_EQ_I32(x[3], -2, "fan_in -4/2 exact");
    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    if (test_scalar_arith())            return 1;
    if (test_mul_boundary())             return 1;
    if (test_add_sub_saturation())       return 1;
    if (test_vec_add_saturation())       return 1;
    if (test_vector_ops())               return 1;
    if (test_vec_scale())                return 1;
    if (test_bias_add())                 return 1;
    if (test_isqrt())                    return 1;
    if (test_pack_roundtrip())           return 1;
    if (test_popcount_dist())            return 1;
    if (test_ternary_matmul_bt_small())  return 1;
    if (test_ternary_matmul_bt_k32())    return 1;
    if (test_ternary_matmul_bt_k17_tail()) return 1;
    if (test_ternary_matmul_bt_m3())     return 1;
    if (test_dense_matmul_2x2())         return 1;
    if (test_dense_matmul_bt_2x2())      return 1;
    if (test_layernorm_constant_row())   return 1;
    if (test_layernorm_symmetric_row())  return 1;
    if (test_layernorm_symmetric_row_with_bias()) return 1;
    if (test_layernorm_zero_cols())      return 1;
    if (test_max_val_boundary())         return 1;
    if (test_fan_in_normalize_rounding()) return 1;
    printf("m4t_smoke: all tests passed\n");
    return 0;
}
