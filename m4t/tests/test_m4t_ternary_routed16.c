/*
 * test_m4t_ternary_routed16.c — bit-exact verification of the
 * sparse-routed NEON kernel against the routed_ref oracle.
 *
 * Coverage:
 *   1. Encoder round-trip: every nonzero in W appears in exactly one tile
 *      with the correct sign and position; every zero is absent.
 *   2. Bit-exact match vs m4t_ternary_5in8_matmul_bt_routed_ref on:
 *      - BitNet-shape K=N=2560 at sparsities {0%, 25%, 40%, 50%, 60%, 75%, 90%, 95%, 99%, 100%}
 *      - Boundary shapes (small K, K%32 != 0, single-output, single-input, K<32)
 *      - Wider shapes (K=6912)
 */

#include "m4t_ternary_routed16.h"
#include "m4t_ternary_matmul.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int g_fails = 0;

static uint32_t xs_state = 0xC0FFEEEEu;
static uint32_t xs(void) {
    uint32_t x = xs_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    xs_state = x;
    return x;
}

/* Random 5-in-8-packed weights with target zero density. */
static void rand_packed_weights_5in8(uint8_t* W_packed, int K, int N, double zero_prob) {
    int Kp = (K + 4) / 5;
    static const int n_pow3[5] = {1, 3, 9, 27, 81};
    for (int j = 0; j < N; j++) {
        for (int b = 0; b < Kp; b++) {
            uint8_t byte = 0;
            for (int d = 0; d < 5; d++) {
                int k = b * 5 + d;
                if (k >= K) break;
                uint32_t r = xs() % 1000000u;
                uint8_t u;
                if (r < (uint32_t)(zero_prob * 1e6)) {
                    u = 0;
                } else {
                    u = (xs() & 1u) ? 1u : 2u;
                }
                byte += (uint8_t)(u * n_pow3[d]);
            }
            W_packed[j * Kp + b] = byte;
        }
    }
}

static void rand_x(int8_t* X, int K, int range) {
    for (int i = 0; i < K; i++) {
        int v = (int)(xs() % (uint32_t)(2 * range + 1)) - range;
        X[i] = (int8_t)v;
    }
}

static int8_t decode_trit_5in8(const uint8_t* W, int Kp, int j, int k) {
    static const uint8_t POW3[5] = { 1u, 3u, 9u, 27u, 81u };
    int b = k / 5, d = k % 5;
    uint8_t byte = W[(size_t)j * Kp + b];
    uint8_t u = (uint8_t)((byte / POW3[d]) % 3u);
    if (u == 1u) return 1;
    if (u == 2u) return -1;
    return 0;
}

/* Round-trip: build dense W from packed, build dense W from routed16 tiles,
 * verify equality. Also: every position covered by exactly one tile if
 * it's nonzero, zero positions covered by zero tiles. */
static int verify_encoder_roundtrip(const uint8_t* W_5in8,
                                    const m4t_routed16_packed_t* P,
                                    int K, int N) {
    int Kp = (K + 4) / 5;
    int8_t* W_dense_from_packed = (int8_t*)calloc((size_t)K * N, 1);
    int8_t* W_dense_from_tiles  = (int8_t*)calloc((size_t)K * N, 1);
    int8_t* coverage            = (int8_t*)calloc((size_t)K * N, 1);

    for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
            W_dense_from_packed[j * K + k] = decode_trit_5in8(W_5in8, Kp, j, k);
        }
    }

    int total_tiles = (int)m4t_routed16_packed_total_tiles(P);
    int* col_offset_from_introspection = NULL;
    /* Walk tiles via the public _N + introspection: rebuild via the
     * tile array using internal pointer? The packed handle is opaque,
     * so we expand the dense from tiles using the kernel itself: feed
     * X = e_k for each k and check Y[j] = W[k, j]. */

    /* Direct-method: project a unit vector e_k through routed16 to read column. */
    int8_t* X = (int8_t*)calloc((size_t)K, 1);
    int32_t* Y = (int32_t*)calloc((size_t)N, sizeof(int32_t));

    int max_diff = 0;
    int n_diff = 0;
    for (int k = 0; k < K; k++) {
        memset(X, 0, (size_t)K);
        X[k] = 1;
        memset(Y, 0, (size_t)N * sizeof(int32_t));
        m4t_ternary_routed16_matmul_bt(Y, X, P, 1, K, N);
        for (int j = 0; j < N; j++) {
            int8_t expected = W_dense_from_packed[j * K + k];
            int actual = Y[j];
            if (actual != expected) {
                if (n_diff < 5) {
                    fprintf(stderr, "  FAIL roundtrip k=%d j=%d expected=%d actual=%d\n",
                            k, j, expected, actual);
                }
                n_diff++;
                int d = actual - expected; if (d < 0) d = -d;
                if (d > max_diff) max_diff = d;
            }
            W_dense_from_tiles[j * K + k] = (int8_t)actual;
        }
    }

    free(W_dense_from_packed);
    free(W_dense_from_tiles);
    free(coverage);
    free(X); free(Y);
    (void)col_offset_from_introspection; (void)total_tiles;
    return n_diff;
}

static int test_one(int K, int N, double zero_prob, const char* label,
                    int x_range) {
    int Kp = M4T_TRIT_PACKED5_BYTES(K);
    uint8_t* W_packed = (uint8_t*)calloc((size_t)N * Kp, 1);
    int8_t* X = (int8_t*)calloc((size_t)K, 1);
    int32_t* Y_ref = (int32_t*)calloc((size_t)N, sizeof(int32_t));
    int32_t* Y_neon = (int32_t*)calloc((size_t)N, sizeof(int32_t));

    rand_packed_weights_5in8(W_packed, K, N, zero_prob);
    rand_x(X, K, x_range);

    int n_diff = 0;
    int64_t max_d = 0;

    /* Oracle */
    int64_t skipped = 0;
    m4t_ternary_5in8_matmul_bt_routed_ref(Y_ref, X, W_packed, 1, K, N, &skipped);

    /* Encode */
    m4t_routed16_packed_t* P = m4t_ternary_routed16_pack(W_packed, K, N);
    if (!P) { fprintf(stderr, "  FAIL [%s] encoder returned NULL\n", label); g_fails++; goto done; }

    /* NEON kernel */
    m4t_ternary_routed16_matmul_bt(Y_neon, X, P, 1, K, N);
    for (int j = 0; j < N; j++) {
        int64_t d = (int64_t)Y_ref[j] - (int64_t)Y_neon[j];
        if (d < 0) d = -d;
        if (d > max_d) max_d = d;
        if (Y_ref[j] != Y_neon[j]) {
            if (n_diff < 5) {
                fprintf(stderr, "  FAIL[%s] j=%d ref=%d neon=%d\n",
                        label, j, Y_ref[j], Y_neon[j]);
            }
            n_diff++;
        }
    }

    int total_trits = K * N;
    fprintf(stderr, "  [%s K=%d N=%d zp=%.0f%%] tiles=%zu sparsity=%.1f%% diff=%d (max=%lld)\n",
            label, K, N, zero_prob * 100,
            m4t_routed16_packed_total_tiles(P),
            100.0 * (double)skipped / (double)total_trits,
            n_diff, (long long)max_d);

    g_fails += n_diff;
    m4t_ternary_routed16_packed_free(P);
done:
    free(W_packed); free(X); free(Y_ref); free(Y_neon);
    return n_diff;
}

/* M>1 bit-exact test. Build random X[M, K]; run oracle (which loops i)
 * and routed16 (which loops i) and compare every cell of Y[M, N]. */
static void test_M_gt_1(int M, int K, int N, double zero_prob, const char* label) {
    int Kp = M4T_TRIT_PACKED5_BYTES(K);
    uint8_t* W_packed = (uint8_t*)calloc((size_t)N * Kp, 1);
    int8_t*  X        = (int8_t*) calloc((size_t)M * K, 1);
    int32_t* Y_ref    = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));
    int32_t* Y_neon   = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));

    rand_packed_weights_5in8(W_packed, K, N, zero_prob);
    rand_x(X, M * K, 100);

    /* Oracle: feed all M rows together (it iterates i internally). */
    m4t_ternary_5in8_matmul_bt_routed_ref(Y_ref, X, W_packed, M, K, N, NULL);

    m4t_routed16_packed_t* P = m4t_ternary_routed16_pack(W_packed, K, N);
    if (!P) { fprintf(stderr, "  FAIL [%s] pack NULL\n", label); g_fails++; goto cleanup; }

    m4t_ternary_routed16_matmul_bt(Y_neon, X, P, M, K, N);

    int n_diff = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (Y_ref[i * N + j] != Y_neon[i * N + j]) {
                if (n_diff < 5) {
                    fprintf(stderr, "  FAIL[%s] i=%d j=%d ref=%d neon=%d\n",
                            label, i, j, Y_ref[i*N+j], Y_neon[i*N+j]);
                }
                n_diff++;
            }
        }
    }
    fprintf(stderr, "  [%s M=%d K=%d N=%d zp=%.0f%%] diff=%d\n",
            label, M, K, N, zero_prob * 100, n_diff);
    g_fails += n_diff;
    m4t_ternary_routed16_packed_free(P);
cleanup:
    free(W_packed); free(X); free(Y_ref); free(Y_neon);
}

/* Encoder round-trip: build a small W, encode, query columns via e_k. */
static void test_roundtrip(int K, int N, double zero_prob, const char* label) {
    int Kp = M4T_TRIT_PACKED5_BYTES(K);
    uint8_t* W_packed = (uint8_t*)calloc((size_t)N * Kp, 1);
    rand_packed_weights_5in8(W_packed, K, N, zero_prob);

    m4t_routed16_packed_t* P = m4t_ternary_routed16_pack(W_packed, K, N);
    if (!P) { fprintf(stderr, "  FAIL [%s] roundtrip pack failed\n", label); g_fails++; free(W_packed); return; }

    int n_diff = verify_encoder_roundtrip(W_packed, P, K, N);
    fprintf(stderr, "  [%s roundtrip K=%d N=%d zp=%.0f%%] diff=%d\n",
            label, K, N, zero_prob * 100, n_diff);
    g_fails += n_diff;

    m4t_ternary_routed16_packed_free(P);
    free(W_packed);
}

int main(void) {
    fprintf(stderr, "test_m4t_ternary_routed16: bit-exact verification\n");

    /* Encoder round-trip on small shapes (verify every k contributes correctly). */
    test_roundtrip(64, 32, 0.40, "small dense");
    test_roundtrip(64, 32, 0.95, "small sparse");
    test_roundtrip(33, 17, 0.50, "K%5=3 N=17");
    test_roundtrip(7,  3,  0.30, "tiny K<WINDOW");
    test_roundtrip(31, 5,  0.60, "K<WINDOW");

    /* Bit-exact at varied sparsity, BitNet shapes. */
    test_one(2560, 2560, 0.00, "BitNet 2560 zp=0%",   100);
    test_one(2560, 2560, 0.25, "BitNet 2560 zp=25%",  100);
    test_one(2560, 2560, 0.40, "BitNet 2560 zp=40%",  100);
    test_one(2560, 2560, 0.50, "BitNet 2560 zp=50%",  100);
    test_one(2560, 2560, 0.60, "BitNet 2560 zp=60%",  100);
    test_one(2560, 2560, 0.75, "BitNet 2560 zp=75%",  100);
    test_one(2560, 2560, 0.90, "BitNet 2560 zp=90%",  100);
    test_one(2560, 2560, 0.95, "BitNet 2560 zp=95%",  100);
    test_one(2560, 2560, 0.99, "BitNet 2560 zp=99%",  100);
    test_one(2560, 2560, 1.00, "BitNet 2560 zp=100%", 100);

    /* Wider K (FFN). */
    test_one(2560, 6912, 0.40, "FFN-shape gate",      100);
    test_one(6912, 2560, 0.40, "FFN-shape down",      100);

    /* Boundary cases. */
    test_one(33,    17, 0.40, "K%5=3 N=17",            50);
    test_one(80,     1, 0.40, "single output",         50);
    test_one(32,     5, 0.40, "K==WINDOW",             50);
    test_one(31,     5, 0.40, "K<WINDOW",              50);
    test_one(7,      3, 0.30, "tiny K<<WINDOW",        50);

    /* Single-trit edge. */
    test_one(1, 1, 0.0, "K=1 N=1 dense",              50);

    /* M>1 bit-exact (regression for the silent-truncation bug fix). */
    test_M_gt_1(1,    2560, 2560, 0.40, "M=1");
    test_M_gt_1(2,     512,  256, 0.40, "M=2");
    test_M_gt_1(8,     512,  256, 0.40, "M=8");
    test_M_gt_1(32,    256,  128, 0.40, "M=32");
    test_M_gt_1(4,    2560, 2560, 0.50, "M=4 BitNet shape");
    test_M_gt_1(2,    2560, 6912, 0.40, "M=2 FFN-up");
    test_M_gt_1(0,    2560, 2560, 0.40, "M=0 (empty)");
    test_M_gt_1(3,      31,   17, 0.40, "M=3 K<WINDOW");
    test_M_gt_1(2,      33,   17, 0.50, "M=2 K%5=3");
    test_M_gt_1(2,    6912, 2560, 0.38, "M=2 FFN-down");

    if (g_fails > 0) {
        fprintf(stderr, "test_m4t_ternary_routed16: %d failures\n", g_fails);
        return 1;
    }
    fprintf(stderr, "test_m4t_ternary_routed16: all tests passed\n");
    return 0;
}
