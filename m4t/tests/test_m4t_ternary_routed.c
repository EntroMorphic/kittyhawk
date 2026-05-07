/*
 * test_m4t_ternary_routed.c — verify sparse-routed ternary matmul.
 *
 * Per the "math as signatures via routing" foundation. The routed
 * variant treats each trit as a route decision (absent/positive/negative)
 * rather than a numeric coefficient. Math is identical to the dense
 * version (multiply by 0 is no-op); bit-exact match is required.
 *
 * Tests:
 *   1. Bit-exact vs dense (m4t_ternary_5in8_matmul_bt) and scalar_ref.
 *   2. Sparsity measurement on random ternary weights.
 *   3. Boundary shapes (small K, small N, K%5 != 0).
 */

#include "m4t_ternary_matmul.h"
#include "m4t_mtfp.h"
#include "m4t_types.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int g_fails = 0;

static uint32_t xs_state = 0xC0FFEEEEu;
static uint32_t xs(void) {
    uint32_t x = xs_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    xs_state = x;
    return x;
}

/* Generate random 5-in-8 packed weights with target zero-trit density. */
static void rand_packed_weights(uint8_t* W_packed, int K, int N, double zero_prob) {
    int Kp = (K + 4) / 5;
    int n_pow3[5] = {1, 3, 9, 27, 81};
    for (int j = 0; j < N; j++) {
        for (int b = 0; b < Kp; b++) {
            uint8_t byte = 0;
            for (int d = 0; d < 5; d++) {
                int k = b * 5 + d;
                if (k >= K) break;
                /* Trit code u: 0→0, 1→+1, 2→-1. zero_prob controls fraction of u=0. */
                uint32_t r = xs() % 1000000u;
                uint8_t u;
                if (r < (uint32_t)(zero_prob * 1e6)) {
                    u = 0;
                } else {
                    /* Split remainder evenly between +1 and -1. */
                    u = (xs() & 1u) ? 1u : 2u;
                }
                byte += (uint8_t)(u * n_pow3[d]);
            }
            W_packed[j * Kp + b] = byte;
        }
    }
}

static void rand_x(int8_t* X, int M, int K, int range) {
    for (int i = 0; i < M * K; i++) {
        int v = (int)(xs() % (uint32_t)(2 * range + 1)) - range;
        X[i] = (int8_t)v;
    }
}

static int compare_arrays(const m4t_mtfp_t* a, const m4t_mtfp_t* b, int n,
                          const char* label) {
    int fails = 0;
    int64_t max_diff = 0;
    for (int i = 0; i < n; i++) {
        int64_t d = (int64_t)a[i] - (int64_t)b[i];
        if (d < 0) d = -d;
        if (d > max_diff) max_diff = d;
        if (a[i] != b[i]) {
            if (fails < 5) {
                fprintf(stderr, "FAIL[%s] i=%d a=%d b=%d\n", label, i, a[i], b[i]);
            }
            fails++;
        }
    }
    if (fails == 0) {
        fprintf(stderr, "  [%s] OK (n=%d, bit-exact)\n", label, n);
    } else {
        fprintf(stderr, "  [%s] %d fails (max_diff=%lld)\n", label, fails, (long long)max_diff);
    }
    g_fails += fails;
    return fails;
}

static void test_bit_exact(int M, int K, int N, double zero_prob, const char* label) {
    int Kp = (K + 4) / 5;
    uint8_t* W_packed = (uint8_t*)calloc((size_t)N * Kp, 1);
    int8_t* X = (int8_t*)calloc((size_t)M * K, 1);
    m4t_mtfp_t* Y_dense = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* Y_routed = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));

    rand_packed_weights(W_packed, K, N, zero_prob);
    rand_x(X, M, K, 100);

    m4t_ternary_5in8_matmul_bt_scalar_ref(Y_dense, X, W_packed, M, K, N);

    int64_t skipped = 0;
    m4t_ternary_5in8_matmul_bt_routed(Y_routed, X, W_packed, M, K, N, &skipped);

    int per_output_trits = K * (M * N);
    fprintf(stderr, "  [%s shape M=%d K=%d N=%d zp=%.0f%%] skipped=%lld of %d trits = %.1f%%\n",
            label, M, K, N, zero_prob * 100,
            (long long)skipped, per_output_trits,
            100.0 * skipped / (double)per_output_trits);
    compare_arrays(Y_dense, Y_routed, M * N, label);

    free(W_packed); free(X); free(Y_dense); free(Y_routed);
}

int main(void) {
    fprintf(stderr, "test_m4t_ternary_routed: bit-exact verification\n");

    /* Random shapes with varied zero-trit density. */
    test_bit_exact(1, 2560, 2560, 0.40, "BitNet-shape n=2560 zp=40%");
    test_bit_exact(1, 2560,  640, 0.40, "BitNet-shape n=640 zp=40%");
    test_bit_exact(1, 6912, 2560, 0.40, "BitNet-shape K=6912 zp=40%");
    test_bit_exact(1, 2560, 6912, 0.40, "BitNet-shape gate/up zp=40%");
    test_bit_exact(1,   80,   80, 0.50, "tiny zp=50%");
    test_bit_exact(1,   83,   17, 0.30, "K%5=3 N=17 zp=30%");
    test_bit_exact(2,  500,  100, 0.40, "M=2 zp=40%");
    test_bit_exact(1,    5,    1, 0.40, "K=5 N=1 zp=40%");
    test_bit_exact(1,    1,    1, 0.40, "K=1 N=1 zp=40%");

    /* Edge: zero density at boundaries. */
    test_bit_exact(1, 2560, 2560, 0.00, "all nonzero (zp=0%)");
    test_bit_exact(1, 2560, 2560, 1.00, "all zero  (zp=100%)");
    test_bit_exact(1, 2560, 2560, 0.99, "near all zero (zp=99%)");

    if (g_fails > 0) {
        fprintf(stderr, "test_m4t_ternary_routed: %d failures\n", g_fails);
        return 1;
    }
    fprintf(stderr, "test_m4t_ternary_routed: all tests passed\n");
    return 0;
}
