/*
 * test_m4t_ternary_rowskip.c — bit-exact verification of the row-skip
 * dense kernel against m4t_ternary_5in8_matmul_bt.
 *
 * Coverage:
 *   1. Encoder: count of empty K-rows matches a manual scan of the packed W.
 *   2. Bit-exact: rowskip == dense across:
 *      - Synthetic shapes with controlled empty-row injection
 *      - BitNet shapes (K=N=2560, K=2560 N=6912, K=6912 N=2560)
 *      - Various empty-row fractions: 0%, 5%, 15%, 44%, 90%, 100%
 *      - K%5 != 0, K < 80, K = 1
 *      - M = 0, 1, 2, 4, 8, 32
 */

#include "m4t_ternary_rowskip.h"
#include "m4t_ternary_matmul.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int g_fails = 0;

static uint32_t xs_state = 0xFEEDFACEu;
static uint32_t xs(void) {
    uint32_t x = xs_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    xs_state = x;
    return x;
}

/* Generate random 5-in-8 packed weights with target zero density.
 * Optionally inject empty K-rows: with probability `empty_row_prob`,
 * mark a K-position as "empty" — set W[k, j] = 0 for all j. */
static void rand_packed_with_empties(uint8_t* W_packed, int K, int N,
                                      double zero_prob,
                                      double empty_row_prob,
                                      int* expected_n_empty) {
    int Kp = (K + 4) / 5;
    static const int p3[5] = {1, 3, 9, 27, 81};

    /* Pre-decide which K-rows are empty. */
    uint8_t* row_empty = (uint8_t*)calloc((size_t)K, 1);
    int n_empty = 0;
    for (int k = 0; k < K; k++) {
        uint32_t r = xs() % 1000000u;
        if (r < (uint32_t)(empty_row_prob * 1e6)) {
            row_empty[k] = 1;
            n_empty++;
        }
    }
    *expected_n_empty = n_empty;

    for (int j = 0; j < N; j++) {
        for (int b = 0; b < Kp; b++) {
            uint8_t byte = 0;
            for (int d = 0; d < 5; d++) {
                int k = b * 5 + d;
                if (k >= K) break;
                if (row_empty[k]) continue; /* leave as 0 */
                uint32_t r = xs() % 1000000u;
                uint8_t u;
                if (r < (uint32_t)(zero_prob * 1e6)) u = 0;
                else u = (xs() & 1u) ? 1u : 2u;
                byte += (uint8_t)(u * p3[d]);
            }
            W_packed[j * Kp + b] = byte;
        }
    }
    free(row_empty);
}

static void rand_x(int8_t* X, int n, int range) {
    for (int i = 0; i < n; i++) {
        int v = (int)(xs() % (uint32_t)(2 * range + 1)) - range;
        X[i] = (int8_t)v;
    }
}

/* Manually count empty K-rows by direct decode of W_packed. */
static int count_empty_rows(const uint8_t* W_packed, int K, int N) {
    int Kp = (K + 4) / 5;
    static const uint8_t POW3[5] = {1u, 3u, 9u, 27u, 81u};
    int n_empty = 0;
    for (int k = 0; k < K; k++) {
        int b = k / 5, d = k % 5;
        int found_nz = 0;
        for (int j = 0; j < N; j++) {
            uint8_t byte = W_packed[(size_t)j * Kp + b];
            uint8_t u = (uint8_t)((byte / POW3[d]) % 3u);
            if (u != 0) { found_nz = 1; break; }
        }
        if (!found_nz) n_empty++;
    }
    return n_empty;
}

static void test_one(int M, int K, int N, double zp, double erp,
                     const char* label) {
    int Kp = M4T_TRIT_PACKED5_BYTES(K);
    uint8_t* W = (uint8_t*)calloc((size_t)N * Kp, 1);
    int8_t*  X = (int8_t*) calloc((size_t)M * K, 1);
    int32_t* Y_dense = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));
    int32_t* Y_rs    = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));

    int expected_empty;
    rand_packed_with_empties(W, K, N, zp, erp, &expected_empty);
    rand_x(X, M * K, 100);

    /* Sanity: encoder should count the same empties we injected
     * (modulo natural-occurring empties from zp itself). */
    int observed_empty = count_empty_rows(W, K, N);

    /* Pack rowskip + dense baseline run. */
    m4t_ternary_rowskip_packed_t* P = m4t_ternary_rowskip_pack(W, K, N);
    if (!P) {
        fprintf(stderr, "  FAIL [%s] pack returned NULL\n", label);
        g_fails++;
        free(W); free(X); free(Y_dense); free(Y_rs);
        return;
    }

    int K_c = m4t_ternary_rowskip_packed_K_compressed(P);
    int packed_empty = K - K_c;
    if (packed_empty != observed_empty) {
        fprintf(stderr,
            "  FAIL [%s] encoder empty count: packed=%d observed=%d expected≥%d\n",
            label, packed_empty, observed_empty, expected_empty);
        g_fails++;
    }

    /* Bit-exact verify. */
    m4t_ternary_5in8_matmul_bt(Y_dense, X, W, M, K, N);
    m4t_ternary_rowskip_matmul_bt(Y_rs, X, P, M, K, N);

    int n_diff = 0;
    int64_t max_d = 0;
    for (int i = 0; i < M * N; i++) {
        int64_t d = (int64_t)Y_dense[i] - (int64_t)Y_rs[i];
        if (d < 0) d = -d;
        if (d > max_d) max_d = d;
        if (Y_dense[i] != Y_rs[i]) {
            if (n_diff < 5) {
                fprintf(stderr, "  FAIL[%s] i=%d dense=%d rs=%d\n",
                        label, i, Y_dense[i], Y_rs[i]);
            }
            n_diff++;
        }
    }
    fprintf(stderr,
        "  [%s M=%d K=%d N=%d zp=%.0f%% erp=%.0f%%] "
        "K_c=%d (skip=%d, %.1f%%) diff=%d (max=%lld)\n",
        label, M, K, N, zp * 100, erp * 100,
        K_c, packed_empty,
        100.0 * packed_empty / (K > 0 ? K : 1),
        n_diff, (long long)max_d);
    g_fails += n_diff;

    m4t_ternary_rowskip_packed_free(P);
    free(W); free(X); free(Y_dense); free(Y_rs);
}

int main(void) {
    fprintf(stderr, "test_m4t_ternary_rowskip: bit-exact verification\n");

    /* Empty-row fraction sweep on BitNet shapes. */
    test_one(1, 2560, 2560, 0.50, 0.00, "q-shape erp=0%");
    test_one(1, 2560, 2560, 0.50, 0.05, "q-shape erp=5%");
    test_one(1, 2560, 2560, 0.50, 0.15, "q-shape erp=15% (~o_proj L0)");
    test_one(1, 2560, 2560, 0.50, 0.44, "q-shape erp=44% (~down_proj L1 magnitude)");
    test_one(1, 2560, 2560, 0.50, 0.90, "q-shape erp=90%");
    test_one(1, 2560, 2560, 0.50, 1.00, "q-shape erp=100% (all empty)");

    /* FFN shapes. */
    test_one(1, 2560, 6912, 0.40, 0.10, "gate-shape K=2560 N=6912 erp=10%");
    test_one(1, 6912, 2560, 0.38, 0.44, "down-shape K=6912 N=2560 erp=44%");

    /* Boundary K. */
    test_one(1, 80,  16,  0.40, 0.20, "K=80 N=16 erp=20%");
    test_one(1, 33,  17,  0.50, 0.30, "K%5=3 N=17 erp=30%");
    test_one(1, 5,   3,   0.40, 0.40, "K=5 N=3 erp=40%");
    test_one(1, 1,   1,   0.00, 0.00, "K=1 N=1 dense");

    /* M > 1. */
    test_one(2, 2560, 2560, 0.50, 0.15, "M=2 q-shape erp=15%");
    test_one(8, 512,  256,  0.40, 0.20, "M=8 erp=20%");
    test_one(32, 256, 128,  0.40, 0.10, "M=32 erp=10%");
    test_one(0, 2560, 2560, 0.50, 0.15, "M=0 (empty)");

    /* Realistic BitNet sparsity patterns: per-cell ~50% AND empty rows. */
    test_one(1, 6912, 2560, 0.40, 0.44, "M=1 down-proj L1 simulation");
    test_one(2, 6912, 2560, 0.40, 0.44, "M=2 down-proj L1 simulation");

    if (g_fails > 0) {
        fprintf(stderr, "test_m4t_ternary_rowskip: %d failures\n", g_fails);
        return 1;
    }
    fprintf(stderr, "test_m4t_ternary_rowskip: all tests passed\n");
    return 0;
}
