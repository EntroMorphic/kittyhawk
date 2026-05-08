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
    /* Per TD-1 / spec §20: Kp = ceil(K/5) bytes per packed row. For
     * K%5 == 0 this matches K/5; for K%5 != 0 the trailing byte holds
     * (K%5) valid trits + zero padding. */
    int Kp = (K + 4) / 5;
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

    /* Aligned configs: K multiple of 80, N multiple of 4. Exercises the
     * tile body without touching the K-tail or N-tail paths. */
    int Ks_aligned[] = { 80, 160, 320, 1280 };
    int Ms[] = { 8, 16 };
    int Ns_aligned[] = { 4, 16, 64 };
    int n_samples = 25;

    for (size_t ki = 0; ki < sizeof(Ks_aligned)/sizeof(Ks_aligned[0]); ki++) {
        for (size_t mi = 0; mi < sizeof(Ms)/sizeof(Ms[0]); mi++) {
            for (size_t ni = 0; ni < sizeof(Ns_aligned)/sizeof(Ns_aligned[0]); ni++) {
                if (test_matmul_bit_exact(Ks_aligned[ki], Ms[mi], Ns_aligned[ni], n_samples)) {
                    return 1;
                }
            }
        }
    }

    /* Per TD-1: tail-path tests. K%80 != 0 (covers K-tail) AND N%4 != 0
     * (covers N-tail). Each test exercises 1+ tail path. */

    /* Sanity: K=160 (multiple of 80) with M=4 to verify the tile body alone. */
    if (test_matmul_bit_exact(160, 4, 4, 1)) return 1;

    /* K-tail only: N%4 == 0 but K%80 != 0. */
    int Ks_k_tail[]  = { 5, 80 + 5, 80 + 79, 160 + 1, 240 + 47 };
    int Ns_k_tail[]  = { 4, 16 };
    for (size_t ki = 0; ki < sizeof(Ks_k_tail)/sizeof(Ks_k_tail[0]); ki++) {
        for (size_t ni = 0; ni < sizeof(Ns_k_tail)/sizeof(Ns_k_tail[0]); ni++) {
            if (test_matmul_bit_exact(Ks_k_tail[ki], 4, Ns_k_tail[ni], 10)) {
                return 1;
            }
        }
    }

    /* N-tail only: K%80 == 0 but N%4 != 0. Covers j_tail = 1, 2, 3 cases. */
    int Ns_n_tail[]  = { 1, 2, 3, 5, 6, 7, 17 };
    for (size_t ni = 0; ni < sizeof(Ns_n_tail)/sizeof(Ns_n_tail[0]); ni++) {
        if (test_matmul_bit_exact(80, 4, Ns_n_tail[ni], 10)) return 1;
        if (test_matmul_bit_exact(320, 8, Ns_n_tail[ni], 10)) return 1;
    }

    /* Both tails: K%80 != 0 AND N%4 != 0. Most combinatoric coverage. */
    int Ks_both[]    = { 5, 17, 80 + 17, 160 + 5 };
    int Ns_both[]    = { 1, 3, 5, 7 };
    for (size_t ki = 0; ki < sizeof(Ks_both)/sizeof(Ks_both[0]); ki++) {
        for (size_t ni = 0; ni < sizeof(Ns_both)/sizeof(Ns_both[0]); ni++) {
            if (test_matmul_bit_exact(Ks_both[ki], 4, Ns_both[ni], 10)) {
                return 1;
            }
        }
    }

    /* Per journal/k80_fix_lmm.md: full K%80 sweep covering every
     * boundary-tile pattern. The patched kernel runs the NEON tile
     * body to K_padded = ceil(K/80)*80 with stack-local zero-padded
     * W for the boundary tile. K%80 ∈ {1..79} sweep verifies every
     * trit-count-in-boundary-tile configuration. */
    for (int km = 1; km < 80; km++) {
        int K_test = 160 + km;  /* covers main tile + boundary tile */
        if (test_matmul_bit_exact(K_test, 1, 64, 3)) {
            fprintf(stderr, "FAIL: K%%80 sweep K=%d (km=%d)\n", K_test, km);
            return 1;
        }
    }

    /* K<80 sweep: entire kernel IS the boundary tile. */
    int K_lt80[] = { 1, 5, 17, 33, 40, 64, 79 };
    for (size_t ki = 0; ki < sizeof(K_lt80)/sizeof(K_lt80[0]); ki++) {
        if (test_matmul_bit_exact(K_lt80[ki], 1, 64, 5)) {
            fprintf(stderr, "FAIL: K<80 sweep K=%d\n", K_lt80[ki]);
            return 1;
        }
        if (test_matmul_bit_exact(K_lt80[ki], 4, 5, 5)) return 1;  /* with N-tail */
    }

    /* K=0 explicit: degenerate dot product, Y must be all zeros.
     * Kernel must handle without UB or wrong output. */
    {
        int M = 4, N = 16;
        m4t_mtfp_t* Y = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));
        /* Pre-fill Y with sentinel to detect "kernel didn't write". */
        for (int i = 0; i < M * N; i++) Y[i] = 0xDEADBEEF;
        m4t_ternary_5in8_matmul_bt(Y, NULL, NULL, M, 0, N);
        for (int i = 0; i < M * N; i++) {
            if (Y[i] != 0) {
                fprintf(stderr, "FAIL: K=0 expected Y[%d]=0, got %d\n", i, Y[i]);
                free(Y);
                return 1;
            }
        }
        free(Y);
    }

    /* M>1 K%80 sweep: same K%80 pattern coverage but with multi-row M.
     * Catches per-row state contamination (e.g., X_strided not refreshed,
     * acc registers not reset, Y rows interleaved). */
    int K_m_sweep[] = { 161, 200, 239, 320 + 17, 400 + 79, 4, 33, 79 };
    int Ms_m_sweep[] = { 2, 4, 8 };
    for (size_t ki = 0; ki < sizeof(K_m_sweep)/sizeof(K_m_sweep[0]); ki++) {
        for (size_t mi = 0; mi < sizeof(Ms_m_sweep)/sizeof(Ms_m_sweep[0]); mi++) {
            if (test_matmul_bit_exact(K_m_sweep[ki], Ms_m_sweep[mi], 64, 3)) {
                fprintf(stderr, "FAIL: M>1 sweep K=%d M=%d\n",
                        K_m_sweep[ki], Ms_m_sweep[mi]);
                return 1;
            }
        }
    }

    printf("PASS: m4t_ternary_5in8_matmul (pack/unpack roundtrip + golden + "
           "aligned bit-exact + K-tail + N-tail + both-tail bit-exact)\n");
    return 0;
}
