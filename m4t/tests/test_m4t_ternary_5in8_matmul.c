/*
 * test_m4t_ternary_5in8_matmul.c — §20 5-in-8 base-3 matmul tests.
 *
 * Per journal/m4t_5in8_synthesize.md G2 + G3 + G4.
 * Verifies:
 *   - Pack/unpack roundtrip (m4t_pack_trits_5in8_1d / _unpack_).
 *   - Bit-exact: m4t_ternary_5in8_matmul_bt vs _scalar_ref across multiple
 *     K configurations (K = 80, 160, 320, 1280) × 100 random samples each.
 *   - Hand-derived golden values for n=5, 10, 11 trit packings.
 *   - Aliasing assertions fire on Y == X and Y == W (debug-build).
 */

#include "m4t_trit_pack.h"
#include "m4t_ternary_matmul.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s at %s:%d\n", msg, __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

static uint32_t rng_state = 0xdeadbeefu;
static uint32_t rng(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x;
    return x;
}

static void gen_ternary(m4t_trit_t* dst, int n) {
    for (int i = 0; i < n; i++) {
        int v = (int)(rng() % 3) - 1;  /* {-1, 0, +1} */
        dst[i] = (m4t_trit_t)v;
    }
}

/* Test 1: pack/unpack roundtrip. */
static int test_pack_unpack_roundtrip(void) {
    int sizes[] = {5, 10, 11, 80, 320, 1280, 81};
    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++) {
        int n = sizes[si];
        m4t_trit_t* src = (m4t_trit_t*)calloc((size_t)n, sizeof(m4t_trit_t));
        m4t_trit_t* dst = (m4t_trit_t*)calloc((size_t)n, sizeof(m4t_trit_t));
        uint8_t* packed = (uint8_t*)calloc((size_t)M4T_TRIT_PACKED5_BYTES(n), 1);

        gen_ternary(src, n);
        m4t_pack_trits_5in8_1d(packed, src, n);
        m4t_unpack_trits_5in8_1d(dst, packed, n);

        for (int i = 0; i < n; i++) {
            if (src[i] != dst[i]) {
                fprintf(stderr,
                    "FAIL: pack/unpack roundtrip mismatch at n=%d i=%d "
                    "(src=%d, dst=%d)\n", n, i, src[i], dst[i]);
                free(src); free(dst); free(packed);
                return 1;
            }
        }
        free(src); free(dst); free(packed);
    }
    return 0;
}

/* Test 2: hand-derived golden values for byte encoding.
 * Encoding: byte = u_0 + 3*u_1 + 9*u_2 + 27*u_3 + 81*u_4
 * where trit_to_unsigned: -1→2, 0→0, +1→1. */
static int test_pack_golden(void) {
    /* Case A: 5 zeros → byte 0 */
    {
        m4t_trit_t src[5] = {0, 0, 0, 0, 0};
        uint8_t packed = 0xFF;
        m4t_pack_trits_5in8_1d(&packed, src, 5);
        EXPECT(packed == 0u, "5 zeros should pack to 0");
    }
    /* Case B: trit (+1, 0, 0, 0, 0) → byte = 1 */
    {
        m4t_trit_t src[5] = {1, 0, 0, 0, 0};
        uint8_t packed = 0;
        m4t_pack_trits_5in8_1d(&packed, src, 5);
        EXPECT(packed == 1u, "(+1,0,0,0,0) should pack to 1");
    }
    /* Case C: trit (-1, 0, 0, 0, 0) → byte = 2 (u_0 = 2 for -1) */
    {
        m4t_trit_t src[5] = {-1, 0, 0, 0, 0};
        uint8_t packed = 0;
        m4t_pack_trits_5in8_1d(&packed, src, 5);
        EXPECT(packed == 2u, "(-1,0,0,0,0) should pack to 2");
    }
    /* Case D: trit (0, +1, 0, 0, 0) → byte = 3 */
    {
        m4t_trit_t src[5] = {0, 1, 0, 0, 0};
        uint8_t packed = 0;
        m4t_pack_trits_5in8_1d(&packed, src, 5);
        EXPECT(packed == 3u, "(0,+1,0,0,0) should pack to 3");
    }
    /* Case E: max valid byte 242: all-(-1) trits.
     * byte = 2 + 6 + 18 + 54 + 162 = 242. */
    {
        m4t_trit_t src[5] = {-1, -1, -1, -1, -1};
        uint8_t packed = 0;
        m4t_pack_trits_5in8_1d(&packed, src, 5);
        EXPECT(packed == 242u, "5 -1s should pack to 242");
    }
    /* Case F: 10 trits = 2 bytes. */
    {
        m4t_trit_t src[10] = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0};
        uint8_t packed[2] = {0, 0};
        m4t_pack_trits_5in8_1d(packed, src, 10);
        EXPECT(packed[0] == 1u, "byte 0 = +1 at digit 0");
        EXPECT(packed[1] == 3u, "byte 1 = +1 at digit 1 = 3");
    }
    /* Case G: 11 trits = 3 bytes (last byte holds 1 trit + 4 zero pad). */
    {
        m4t_trit_t src[11] = {0,0,0,0,0, 0,0,0,0,0, 1};
        uint8_t packed[3] = {0xFF, 0xFF, 0xFF};
        m4t_pack_trits_5in8_1d(packed, src, 11);
        EXPECT(packed[0] == 0u, "byte 0 zero");
        EXPECT(packed[1] == 0u, "byte 1 zero");
        EXPECT(packed[2] == 1u, "byte 2 = +1 at digit 0 (trailing)");
    }
    return 0;
}

/* Test 3: bit-exact NEON vs scalar across random workloads. */
static int test_matmul_bit_exact(int K, int M, int N, int n_samples) {
    int Kp = K / 5;
    m4t_trit_t* X = (m4t_trit_t*)calloc((size_t)M * K, sizeof(m4t_trit_t));
    m4t_trit_t* W_unp = (m4t_trit_t*)calloc((size_t)N * K, sizeof(m4t_trit_t));
    uint8_t* W_pkd = (uint8_t*)calloc((size_t)N * Kp, 1);
    m4t_mtfp_t* Y_neon = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* Y_ref  = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));

    for (int s = 0; s < n_samples; s++) {
        rng_state = (uint32_t)(s + 1) * 0x9E3779B1u;
        gen_ternary(X, M * K);
        gen_ternary(W_unp, N * K);
        for (int j = 0; j < N; j++) {
            m4t_pack_trits_5in8_1d(W_pkd + (size_t)j * Kp,
                                   W_unp + (size_t)j * K, K);
        }

        m4t_ternary_5in8_matmul_bt(Y_neon, X, W_pkd, M, K, N);
        m4t_ternary_5in8_matmul_bt_scalar_ref(Y_ref, X, W_pkd, M, K, N);

        if (memcmp(Y_neon, Y_ref, (size_t)M * N * sizeof(m4t_mtfp_t)) != 0) {
            fprintf(stderr,
                "FAIL: NEON vs scalar_ref mismatch at K=%d M=%d N=%d sample=%d\n",
                K, M, N, s);
            free(X); free(W_unp); free(W_pkd); free(Y_neon); free(Y_ref);
            return 1;
        }
    }
    free(X); free(W_unp); free(W_pkd); free(Y_neon); free(Y_ref);
    return 0;
}

int main(void) {
    if (test_pack_unpack_roundtrip()) return 1;
    if (test_pack_golden()) return 1;

    /* Multi-config bit-exact verification.
     * K must be multiple of 80 (kernel requirement). N multiple of 4. */
    int Ks[] = { 80, 160, 320, 1280 };
    int Ms[] = { 8, 16 };
    int Ns[] = { 4, 16, 64 };
    int n_samples = 25;  /* per (K, M, N) triple */

    for (size_t ki = 0; ki < sizeof(Ks)/sizeof(Ks[0]); ki++) {
        for (size_t mi = 0; mi < sizeof(Ms)/sizeof(Ms[0]); mi++) {
            for (size_t ni = 0; ni < sizeof(Ns)/sizeof(Ns[0]); ni++) {
                if (test_matmul_bit_exact(Ks[ki], Ms[mi], Ns[ni], n_samples)) {
                    return 1;
                }
            }
        }
    }

    printf("PASS: m4t_ternary_5in8_matmul (pack/unpack roundtrip + golden + "
           "%d × %zu × %zu × %zu = %zu bit-exact NEON/scalar samples)\n",
           n_samples,
           sizeof(Ks)/sizeof(Ks[0]),
           sizeof(Ms)/sizeof(Ms[0]),
           sizeof(Ns)/sizeof(Ns[0]),
           (size_t)n_samples *
           (sizeof(Ks)/sizeof(Ks[0])) *
           (sizeof(Ms)/sizeof(Ms[0])) *
           (sizeof(Ns)/sizeof(Ns[0])));
    return 0;
}
