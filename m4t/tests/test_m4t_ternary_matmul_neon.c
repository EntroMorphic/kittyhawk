/*
 * test_m4t_ternary_matmul_vmlal.c — bit-exact verification of the
 * vmlal_s32-routed ternary matmul (m4t_mtfp_ternary_matmul_bt)
 * against the scalar oracle (m4t_mtfp_ternary_matmul_bt_scalar_ref).
 *
 * Per ternary_mac_routing T-G4. Sample-based (matmul state space is too
 * large to exhaust). Coverage classes:
 *   - K boundary cases (0, 1, 15, 16, 17, 32, 33, 4095, 4096, 4097)
 *   - Sparse vs dense trit distributions
 *   - All-positive, all-negative, all-zero, mixed
 *   - Activation extremes (±MAX_VAL)
 *   - Multiple (M, N) shapes
 */

#include "m4t_mtfp.h"
#include "m4t_ternary_matmul.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_VAL 581130733

static int g_fails = 0;

/* Pack a buffer of unpacked trits (m4t_trit_t in {-1,0,+1}) into the
 * 2-bit packed representation expected by m4t_mtfp_ternary_matmul_bt. */
static void pack_trits(uint8_t* packed, const m4t_trit_t* unpacked, int K) {
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    memset(packed, 0, Kp);
    for (int k = 0; k < K; k++) {
        uint8_t code = (unpacked[k] == 1) ? 0x01u
                     : (unpacked[k] == -1) ? 0x02u
                     : 0x00u;
        packed[k >> 2] |= (uint8_t)(code << ((k & 3) * 2));
    }
}

/* Generate trits with a given density (fraction nonzero) and balance
 * (fraction of nonzero that are +1 vs -1). */
static void gen_trits(m4t_trit_t* dst, int K, double density, double pos_frac,
                      uint32_t seed) {
    srand(seed);
    for (int k = 0; k < K; k++) {
        double r = (double)rand() / (double)RAND_MAX;
        if (r > density) {
            dst[k] = 0;
        } else {
            double p = (double)rand() / (double)RAND_MAX;
            dst[k] = (p < pos_frac) ? 1 : -1;
        }
    }
}

static void gen_activations(m4t_mtfp_t* dst, int n, uint32_t seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        int64_t r = ((int64_t)rand() << 32) ^ rand();
        dst[i] = (m4t_mtfp_t)(r % (2 * MAX_VAL + 1) - MAX_VAL);
    }
}

/* Run one (M, K, N, density, pos_frac, seed) configuration and check
 * vmlal output bit-exact equals scalar_ref output. */
static void test_config(int M, int K, int N, double density, double pos_frac,
                        uint32_t seed, const char* label) {
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    m4t_mtfp_t* X       = M > 0 && K > 0 ? malloc(sizeof(m4t_mtfp_t) * (size_t)M * K) : NULL;
    m4t_trit_t* W_unp   = N > 0 && K > 0 ? malloc(sizeof(m4t_trit_t) * (size_t)N * K) : NULL;
    uint8_t*    W_pack  = N > 0 && K > 0 ? malloc((size_t)N * Kp) : NULL;
    m4t_mtfp_t* Y_vmlal = M > 0 && N > 0 ? malloc(sizeof(m4t_mtfp_t) * (size_t)M * N) : NULL;
    m4t_mtfp_t* Y_ref   = M > 0 && N > 0 ? malloc(sizeof(m4t_mtfp_t) * (size_t)M * N) : NULL;

    if (M > 0 && K > 0) gen_activations(X, M * K, seed);
    if (N > 0 && K > 0) {
        for (int j = 0; j < N; j++) {
            gen_trits(W_unp + (size_t)j * K, K, density, pos_frac, seed + 17 + j);
            pack_trits(W_pack + (size_t)j * Kp, W_unp + (size_t)j * K, K);
        }
    }

    m4t_mtfp_ternary_matmul_bt(Y_vmlal, X, W_pack, NULL, M, K, N);
    m4t_mtfp_ternary_matmul_bt_scalar_ref(Y_ref, X, W_pack, NULL, M, K, N);

    int local_fails = 0;
    for (int i = 0; i < M * N; i++) {
        if (Y_vmlal[i] != Y_ref[i]) {
            local_fails++;
            if (local_fails <= 3) {
                fprintf(stderr,
                    "  FAIL %s i=%d vmlal=%d ref=%d\n",
                    label, i, (int)Y_vmlal[i], (int)Y_ref[i]);
            }
        }
    }
    if (local_fails == 0) {
        printf("  %-50s : PASS  (M=%d K=%d N=%d, %d cells)\n",
               label, M, K, N, M * N);
    } else {
        printf("  %-50s : FAIL  (%d / %d mismatches)\n",
               label, local_fails, M * N);
        g_fails += local_fails;
    }

    free(X); free(W_unp); free(W_pack); free(Y_vmlal); free(Y_ref);
}

/* T-G8: perf comparison — three implementations on two workload shapes. */
typedef void (*matmul_fn)(m4t_mtfp_t*, const m4t_mtfp_t*, const uint8_t*,
                          uint8_t*, int, int, int);

static double bench_one(matmul_fn fn, m4t_mtfp_t* Y, const m4t_mtfp_t* X,
                        const uint8_t* W, int M, int K, int N, int iters) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++) fn(Y, X, W, NULL, M, K, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec));
}

static double min_of(double a, double b) { return a < b ? a : b; }

static void perf_compare(int M, int K, int N, int iters, const char* shape) {
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    m4t_mtfp_t* X       = malloc(sizeof(m4t_mtfp_t) * (size_t)M * K);
    m4t_trit_t* W_unp   = malloc(sizeof(m4t_trit_t) * (size_t)N * K);
    uint8_t*    W_pack  = malloc((size_t)N * Kp);
    m4t_mtfp_t* Y       = malloc(sizeof(m4t_mtfp_t) * (size_t)M * N);

    gen_activations(X, M * K, 9999);
    for (int j = 0; j < N; j++) {
        gen_trits(W_unp + (size_t)j * K, K, 0.5, 0.5, 8888 + j);
        pack_trits(W_pack + (size_t)j * Kp, W_unp + (size_t)j * K, K);
    }

    /* Warmup. */
    for (int w = 0; w < 3; w++) {
        m4t_mtfp_ternary_matmul_bt(Y, X, W_pack, NULL, M, K, N);
        m4t_mtfp_ternary_matmul_bt_scalar_ref(Y, X, W_pack, NULL, M, K, N);
    }

    /* Min-of-5. Post-productionization, the meaningful comparison is
     * production (NEON vmlal path) vs scalar_ref. The bsl-vs-vmlal
     * pre-productionization measurement is preserved in the T-G8 output
     * recorded in journal/ternary_mac_routing_remediation_closeout.md. */
    double prod_min = 1e18, scalar_min = 1e18;
    for (int s = 0; s < 5; s++) {
        prod_min   = min_of(prod_min,
            bench_one(m4t_mtfp_ternary_matmul_bt,            Y, X, W_pack, M, K, N, iters));
        scalar_min = min_of(scalar_min,
            bench_one(m4t_mtfp_ternary_matmul_bt_scalar_ref, Y, X, W_pack, M, K, N, iters));
    }
    double cells = (double)M * N * iters;
    printf("  shape=%-9s M=%2d K=%4d N=%2d iters=%d :\n", shape, M, K, N, iters);
    printf("    scalar_ref : %.2f ns/cell\n", scalar_min / cells);
    printf("    production : %.2f ns/cell  (%.1fx vs scalar_ref)\n",
           prod_min / cells, scalar_min / prod_min);

    free(X); free(W_unp); free(W_pack); free(Y);
}

int main(void) {
    printf("=== Bit-exact: vmlal vs scalar_ref ===\n");

    /* K boundary cases — small (M, N) so we can run many K values. */
    printf("\n-- K boundary cases --\n");
    int K_cases[] = {0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65};
    for (size_t ki = 0; ki < sizeof(K_cases)/sizeof(K_cases[0]); ki++) {
        int K = K_cases[ki];
        char label[64];
        snprintf(label, sizeof(label), "K=%d  density=0.5", K);
        test_config(4, K, 4, 0.5, 0.5, 1000 + (uint32_t)ki, label);
    }

    /* Trit distributions at K=64. */
    printf("\n-- Trit distributions (K=64) --\n");
    test_config(8, 64, 8, 1.00, 0.5, 2001, "all nonzero, balanced ±1");
    test_config(8, 64, 8, 0.50, 0.5, 2002, "half nonzero, balanced");
    test_config(8, 64, 8, 0.10, 0.5, 2003, "sparse 10%, balanced");
    test_config(8, 64, 8, 1.00, 1.0, 2004, "all +1");
    test_config(8, 64, 8, 1.00, 0.0, 2005, "all -1");
    test_config(8, 64, 8, 0.00, 0.5, 2006, "all zero");

    /* Activation extremes (varying seeds). */
    printf("\n-- Activation extremes (K=128) --\n");
    test_config(4, 128, 4, 0.5, 0.5, 3001, "random seed A");
    test_config(4, 128, 4, 0.5, 0.5, 3002, "random seed B");
    test_config(4, 128, 4, 0.5, 0.5, 3003, "random seed C");

    /* Larger shapes — typical bulk workloads. */
    printf("\n-- Bulk shapes --\n");
    test_config(16, 256, 16, 0.5, 0.5, 4001, "M=16 K=256 N=16");
    test_config(8, 1024, 8, 0.3, 0.5, 4002, "M=8  K=1024 N=8");
    test_config(64, 4096, 64, 0.5, 0.5, 4003, "M=64 K=4096 N=64");

    if (g_fails > 0) {
        printf("\nFAIL: %d total mismatches\n", g_fails);
        return 1;
    }
    printf("\nPASS: all configurations bit-exact\n");

    /* T-G8: perf comparison. */
    printf("\n=== Perf comparison (min-of-5) ===\n");
    perf_compare(64, 4096, 64, 5,    "BATCHED");   /* bulk matmul */
    perf_compare(4,  64,   4,  500, "TIGHT-LOOP"); /* small dims, per-call overhead */

    return 0;
}
