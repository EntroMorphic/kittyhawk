/*
 * test_m4t_ternary_matmul.c — direct tests for MTFP19 × packed-trit matmul.
 *
 * Covers exact small products, zero rows, NEON-width K=16, NEON+tail K=20,
 * and saturating clamp on store. Golden values are hand-derived integers.
 */

#include "m4t_types.h"
#include "m4t_trit_pack.h"
#include "m4t_ternary_matmul.h"

#include <stdio.h>

#define ASSERT_EQ_I32(actual, expected, msg) do { \
    if ((actual) != (expected)) { \
        fprintf(stderr, "FAIL: %s — got %d, expected %d (line %d)\n", \
                (msg), (int)(actual), (int)(expected), __LINE__); \
        return 1; \
    } \
} while (0)

static int test_small_exact(void) {
    enum { M = 2, K = 4, N = 3 };
    m4t_mtfp_t X[M * K] = {
         1,  2,  3,  4,
         5, -6,  7, -8
    };
    m4t_trit_t W[N * K] = {
         1,  1,  1,  1,
         1, -1,  0,  1,
         0,  0,  0,  0
    };
    uint8_t W_packed[N * M4T_TRIT_PACKED_BYTES(K)];
    m4t_mtfp_t Y[M * N];

    m4t_pack_trits_rowmajor(W_packed, W, N, K);
    m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, M, K, N);

    ASSERT_EQ_I32(Y[0 * N + 0], 10, "small row0 col0");
    ASSERT_EQ_I32(Y[0 * N + 1],  3, "small row0 col1");
    ASSERT_EQ_I32(Y[0 * N + 2],  0, "small row0 col2");
    ASSERT_EQ_I32(Y[1 * N + 0], -2, "small row1 col0");
    ASSERT_EQ_I32(Y[1 * N + 1],  3, "small row1 col1");
    ASSERT_EQ_I32(Y[1 * N + 2],  0, "small row1 col2");
    return 0;
}

static int test_k16_neon_exact(void) {
    enum { M = 1, K = 16, N = 2 };
    m4t_mtfp_t X[K];
    m4t_trit_t W[N * K];
    uint8_t W_packed[N * M4T_TRIT_PACKED_BYTES(K)];
    m4t_mtfp_t Y[N];

    for (int k = 0; k < K; k++) {
        X[k] = (m4t_mtfp_t)(k + 1);
        W[0 * K + k] = 1;
        W[1 * K + k] = (k < 8) ? 1 : -1;
    }

    m4t_pack_trits_rowmajor(W_packed, W, N, K);
    m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, M, K, N);

    ASSERT_EQ_I32(Y[0], 136, "k16 all-positive");
    ASSERT_EQ_I32(Y[1], -64, "k16 mixed signs");
    return 0;
}

static int test_k20_neon_plus_tail(void) {
    enum { M = 1, K = 20, N = 1 };
    m4t_mtfp_t X[K];
    m4t_trit_t W[K];
    uint8_t W_packed[M4T_TRIT_PACKED_BYTES(K)];
    m4t_mtfp_t Y[1];

    for (int k = 0; k < K; k++) {
        X[k] = 10;
        W[k] = (k < 16) ? 1 : ((k == 16 || k == 17) ? -1 : 0);
    }

    m4t_pack_trits_1d(W_packed, W, K);
    m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, M, K, N);

    ASSERT_EQ_I32(Y[0], 140, "k20 neon plus tail");
    return 0;
}

static int test_store_saturation(void) {
    enum { M = 1, K = 4, N = 2 };
    m4t_mtfp_t X[K] = {
        M4T_MTFP_MAX_VAL,
        M4T_MTFP_MAX_VAL,
        M4T_MTFP_MAX_VAL,
        M4T_MTFP_MAX_VAL
    };
    m4t_trit_t W[N * K] = {
         1,  1,  1,  1,
        -1, -1, -1, -1
    };
    uint8_t W_packed[N * M4T_TRIT_PACKED_BYTES(K)];
    m4t_mtfp_t Y[N];

    m4t_pack_trits_rowmajor(W_packed, W, N, K);
    m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, M, K, N);

    ASSERT_EQ_I32(Y[0],  M4T_MTFP_MAX_VAL, "positive saturation");
    ASSERT_EQ_I32(Y[1], -M4T_MTFP_MAX_VAL, "negative saturation");
    return 0;
}

int main(void) {
    if (test_small_exact()) return 1;
    if (test_k16_neon_exact()) return 1;
    if (test_k20_neon_plus_tail()) return 1;
    if (test_store_saturation()) return 1;
    printf("m4t_ternary_matmul: all tests passed\n");
    return 0;
}
