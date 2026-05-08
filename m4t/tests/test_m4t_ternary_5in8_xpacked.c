/*
 * test_m4t_ternary_5in8_xpacked.c — TD-7 X-packed §20 sibling tests.
 *
 * Two verification gates:
 *   G1: NEON vs scalar_ref bit-exact across multi-config × random samples.
 *   G2: cross-equivalence with §20 — when X_packed unpacks to X, the
 *       xpacked kernel produces the same Y as m4t_ternary_5in8_matmul_bt.
 *       (Strong cross-check: any decode-side bug in the xpacked X path
 *       is caught by comparing against the canonical X-unpacked kernel.)
 */

#include "m4t_trit_pack.h"
#include "m4t_ternary_matmul.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static uint32_t rng_state = 0xc0ffee01u;
static uint32_t rng(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x;
    return x;
}

static void gen_ternary(m4t_trit_t* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = (m4t_trit_t)((int)(rng() % 3) - 1);
    }
}

/* G1: NEON vs scalar_ref. */
static int test_xpacked_bit_exact(int K, int M, int N, int n_samples) {
    int Kp = (K + 4) / 5;
    m4t_trit_t* X_unp = (m4t_trit_t*)calloc((size_t)M * K, sizeof(m4t_trit_t));
    m4t_trit_t* W_unp = (m4t_trit_t*)calloc((size_t)N * K, sizeof(m4t_trit_t));
    uint8_t*    X_pkd = (uint8_t*)calloc((size_t)M * Kp, 1);
    uint8_t*    W_pkd = (uint8_t*)calloc((size_t)N * Kp, 1);
    m4t_mtfp_t* Y_neon = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* Y_ref  = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));

    for (int s = 0; s < n_samples; s++) {
        rng_state = (uint32_t)(s + 1) * 0xC2B2AE3Du;
        gen_ternary(X_unp, M * K);
        gen_ternary(W_unp, N * K);
        for (int i = 0; i < M; i++) {
            m4t_pack_trits_5in8_1d(X_pkd + (size_t)i * Kp,
                                   X_unp + (size_t)i * K, K);
        }
        for (int j = 0; j < N; j++) {
            m4t_pack_trits_5in8_1d(W_pkd + (size_t)j * Kp,
                                   W_unp + (size_t)j * K, K);
        }

        m4t_ternary_5in8_matmul_xpacked_bt(Y_neon, X_pkd, W_pkd, M, K, N);
        m4t_ternary_5in8_matmul_xpacked_bt_scalar_ref(Y_ref, X_pkd, W_pkd, M, K, N);

        if (memcmp(Y_neon, Y_ref, (size_t)M * N * sizeof(m4t_mtfp_t)) != 0) {
            fprintf(stderr,
                "FAIL G1: xpacked NEON vs scalar_ref mismatch K=%d M=%d N=%d sample=%d\n",
                K, M, N, s);
            free(X_unp); free(W_unp); free(X_pkd); free(W_pkd);
            free(Y_neon); free(Y_ref);
            return 1;
        }
    }
    free(X_unp); free(W_unp); free(X_pkd); free(W_pkd);
    free(Y_neon); free(Y_ref);
    return 0;
}

/* G3: route variant bit-exact vs scalar_ref. */
static int test_xpacked_route_bit_exact(int K, int M, int N, int n_samples) {
    int Kp = (K + 4) / 5;
    m4t_trit_t* X_unp = (m4t_trit_t*)calloc((size_t)M * K, sizeof(m4t_trit_t));
    m4t_trit_t* W_unp = (m4t_trit_t*)calloc((size_t)N * K, sizeof(m4t_trit_t));
    uint8_t*    X_pkd = (uint8_t*)calloc((size_t)M * Kp, 1);
    uint8_t*    W_pkd = (uint8_t*)calloc((size_t)N * Kp, 1);
    m4t_mtfp_t* Y_route = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* Y_ref   = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));

    for (int s = 0; s < n_samples; s++) {
        rng_state = (uint32_t)(s + 7) * 0xCC9E2D51u;
        gen_ternary(X_unp, M * K);
        gen_ternary(W_unp, N * K);
        for (int i = 0; i < M; i++) {
            m4t_pack_trits_5in8_1d(X_pkd + (size_t)i * Kp,
                                   X_unp + (size_t)i * K, K);
        }
        for (int j = 0; j < N; j++) {
            m4t_pack_trits_5in8_1d(W_pkd + (size_t)j * Kp,
                                   W_unp + (size_t)j * K, K);
        }

        m4t_ternary_5in8_matmul_xpacked_bt_route(Y_route, X_pkd, W_pkd, M, K, N);
        m4t_ternary_5in8_matmul_xpacked_bt_scalar_ref(Y_ref, X_pkd, W_pkd, M, K, N);

        if (memcmp(Y_route, Y_ref, (size_t)M * N * sizeof(m4t_mtfp_t)) != 0) {
            fprintf(stderr,
                "FAIL G3: xpacked route vs scalar_ref K=%d M=%d N=%d sample=%d\n",
                K, M, N, s);
            free(X_unp); free(W_unp); free(X_pkd); free(W_pkd);
            free(Y_route); free(Y_ref);
            return 1;
        }
    }
    free(X_unp); free(W_unp); free(X_pkd); free(W_pkd);
    free(Y_route); free(Y_ref);
    return 0;
}

/* G2: cross-equivalence with §20. */
static int test_xpacked_vs_bt(int K, int M, int N, int n_samples) {
    int Kp = (K + 4) / 5;
    m4t_trit_t* X_unp = (m4t_trit_t*)calloc((size_t)M * K, sizeof(m4t_trit_t));
    m4t_trit_t* W_unp = (m4t_trit_t*)calloc((size_t)N * K, sizeof(m4t_trit_t));
    uint8_t*    X_pkd = (uint8_t*)calloc((size_t)M * Kp, 1);
    uint8_t*    W_pkd = (uint8_t*)calloc((size_t)N * Kp, 1);
    m4t_mtfp_t* Y_xp = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* Y_bt = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));

    for (int s = 0; s < n_samples; s++) {
        rng_state = (uint32_t)(s + 17) * 0x85EBCA6Bu;
        gen_ternary(X_unp, M * K);
        gen_ternary(W_unp, N * K);
        for (int i = 0; i < M; i++) {
            m4t_pack_trits_5in8_1d(X_pkd + (size_t)i * Kp,
                                   X_unp + (size_t)i * K, K);
        }
        for (int j = 0; j < N; j++) {
            m4t_pack_trits_5in8_1d(W_pkd + (size_t)j * Kp,
                                   W_unp + (size_t)j * K, K);
        }

        m4t_ternary_5in8_matmul_xpacked_bt(Y_xp, X_pkd, W_pkd, M, K, N);
        m4t_ternary_5in8_matmul_bt(Y_bt, X_unp, W_pkd, M, K, N);

        if (memcmp(Y_xp, Y_bt, (size_t)M * N * sizeof(m4t_mtfp_t)) != 0) {
            fprintf(stderr,
                "FAIL G2: xpacked vs §20 bt mismatch K=%d M=%d N=%d sample=%d\n",
                K, M, N, s);
            free(X_unp); free(W_unp); free(X_pkd); free(W_pkd);
            free(Y_xp); free(Y_bt);
            return 1;
        }
    }
    free(X_unp); free(W_unp); free(X_pkd); free(W_pkd);
    free(Y_xp); free(Y_bt);
    return 0;
}

int main(void) {
    /* G1: bit-exact across regimes — aligned + tail paths. */
    int Ks_aligned[] = { 80, 160, 320, 640 };
    int Ms[] = { 4, 8, 16 };
    int Ns_aligned[] = { 4, 16, 64 };
    for (size_t ki = 0; ki < sizeof(Ks_aligned)/sizeof(Ks_aligned[0]); ki++) {
        for (size_t mi = 0; mi < sizeof(Ms)/sizeof(Ms[0]); mi++) {
            for (size_t ni = 0; ni < sizeof(Ns_aligned)/sizeof(Ns_aligned[0]); ni++) {
                if (test_xpacked_bit_exact(
                        Ks_aligned[ki], Ms[mi], Ns_aligned[ni], 10)) {
                    return 1;
                }
            }
        }
    }

    /* Tail paths (mirrors §20 tail coverage from TD-1). */
    int Ks_tail[] = { 5, 17, 85, 159, 161, 287 };
    int Ns_tail[] = { 1, 2, 3, 4, 5, 7, 16 };
    for (size_t ki = 0; ki < sizeof(Ks_tail)/sizeof(Ks_tail[0]); ki++) {
        for (size_t ni = 0; ni < sizeof(Ns_tail)/sizeof(Ns_tail[0]); ni++) {
            if (test_xpacked_bit_exact(Ks_tail[ki], 4, Ns_tail[ni], 5)) return 1;
        }
    }

    /* G2: cross-equivalence with §20 across same regimes. */
    for (size_t ki = 0; ki < sizeof(Ks_aligned)/sizeof(Ks_aligned[0]); ki++) {
        for (size_t mi = 0; mi < sizeof(Ms)/sizeof(Ms[0]); mi++) {
            for (size_t ni = 0; ni < sizeof(Ns_aligned)/sizeof(Ns_aligned[0]); ni++) {
                if (test_xpacked_vs_bt(
                        Ks_aligned[ki], Ms[mi], Ns_aligned[ni], 10)) {
                    return 1;
                }
            }
        }
    }
    for (size_t ki = 0; ki < sizeof(Ks_tail)/sizeof(Ks_tail[0]); ki++) {
        for (size_t ni = 0; ni < sizeof(Ns_tail)/sizeof(Ns_tail[0]); ni++) {
            if (test_xpacked_vs_bt(Ks_tail[ki], 4, Ns_tail[ni], 5)) return 1;
        }
    }

    /* Per journal/k80_fix_lmm.md: full K%80 sweep covering every
     * boundary-tile pattern after the K%80 fix landed in xpacked. */
    for (int km = 1; km < 80; km++) {
        int K_test = 160 + km;
        if (test_xpacked_bit_exact(K_test, 1, 64, 3)) {
            fprintf(stderr, "FAIL: K%%80 sweep K=%d (km=%d)\n", K_test, km);
            return 1;
        }
        if (test_xpacked_vs_bt(K_test, 1, 64, 3)) {
            fprintf(stderr, "FAIL: K%%80 vs §20 sweep K=%d\n", K_test);
            return 1;
        }
    }

    /* K=0 explicit. */
    {
        int M = 4, N = 16;
        m4t_mtfp_t* Y = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));
        for (int i = 0; i < M * N; i++) Y[i] = 0xDEADBEEF;
        m4t_ternary_5in8_matmul_xpacked_bt(Y, NULL, NULL, M, 0, N);
        for (int i = 0; i < M * N; i++) {
            if (Y[i] != 0) {
                fprintf(stderr, "FAIL: xpacked K=0 expected 0, got %d\n", Y[i]);
                free(Y);
                return 1;
            }
        }
        free(Y);
    }

    /* G3: route variant bit-exact (V2 of pure-ternary audit). */
    int Ks_route[] = { 80, 160, 320 };
    int Ms_route[] = { 1, 4 };
    int Ns_route[] = { 4, 16, 64 };
    for (size_t ki = 0; ki < sizeof(Ks_route)/sizeof(Ks_route[0]); ki++) {
        for (size_t mi = 0; mi < sizeof(Ms_route)/sizeof(Ms_route[0]); mi++) {
            for (size_t ni = 0; ni < sizeof(Ns_route)/sizeof(Ns_route[0]); ni++) {
                if (test_xpacked_route_bit_exact(
                        Ks_route[ki], Ms_route[mi], Ns_route[ni], 5)) {
                    return 1;
                }
            }
        }
    }
    /* K%80 sweep on route. */
    for (int km = 1; km < 80; km++) {
        int K_test = 160 + km;
        if (test_xpacked_route_bit_exact(K_test, 1, 64, 2)) {
            fprintf(stderr, "FAIL G3: xpacked route K%%80 sweep K=%d\n", K_test);
            return 1;
        }
    }
    /* K<80 on route. */
    int K_lt80_route[] = { 1, 5, 17, 33, 79 };
    for (size_t ki = 0; ki < sizeof(K_lt80_route)/sizeof(K_lt80_route[0]); ki++) {
        if (test_xpacked_route_bit_exact(K_lt80_route[ki], 1, 64, 5)) return 1;
        if (test_xpacked_route_bit_exact(K_lt80_route[ki], 4, 5, 5)) return 1;
    }
    /* M>1 + K%80 != 0 on route. */
    if (test_xpacked_route_bit_exact(167, 4, 17, 3)) return 1;
    if (test_xpacked_route_bit_exact(287, 8, 5, 3)) return 1;
    /* K=0 on route. */
    {
        int M = 4, N = 16;
        m4t_mtfp_t* Y = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));
        for (int i = 0; i < M * N; i++) Y[i] = 0xCAFEBABE;
        m4t_ternary_5in8_matmul_xpacked_bt_route(Y, NULL, NULL, M, 0, N);
        for (int i = 0; i < M * N; i++) {
            if (Y[i] != 0) {
                fprintf(stderr, "FAIL G3: xpacked route K=0 expected 0\n");
                free(Y); return 1;
            }
        }
        free(Y);
    }

    printf("PASS: m4t_ternary_5in8_xpacked (G1 NEON==scalar_ref + G2 xpacked==§20 + K%%80 sweep + K=0 + G3 route bit-exact)\n");
    return 0;
}
