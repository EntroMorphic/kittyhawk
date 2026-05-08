/*
 * test_m4t_ternary_matmul_neon.c — bit-exact verification of the
 * production NEON path of m4t_mtfp_ternary_matmul_bt against
 * m4t_mtfp_ternary_matmul_bt_scalar_ref (always-scalar oracle).
 *
 * Original cycle (T-G4): 23 hand-curated configurations.
 * Red-team remediation (R-G1, R-G2, R-G3, R-G4):
 *   R-G1: 1000 random (M, K, N, density, seed) configurations across
 *         a representative shape space.
 *   R-G2: saturation-edge configs — dot products that exceed MAX_VAL,
 *         verifying both paths produce the same clamped output AND
 *         the same SATURATED flag bits.
 *   R-G3: multi-shape BATCHED bench (5 (M, K, N) tuples instead of 1).
 *   R-G4: aliasing assertions for both forbidden cases (Y==X, Y==W_packed).
 */

#include "m4t_mtfp.h"
#include "m4t_ternary_matmul.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

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
static void test_config_v(int M, int K, int N, double density, double pos_frac,
                          uint32_t seed, const char* label, int verbose) {
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

    /* V4 (pure-ternary audit): verify route variant is also bit-exact. */
    m4t_mtfp_t* Y_route = M > 0 && N > 0 ? malloc(sizeof(m4t_mtfp_t) * (size_t)M * N) : NULL;
    if (Y_route) {
        m4t_mtfp_ternary_matmul_bt_route(Y_route, X, W_pack, NULL, M, K, N);
    }

    int local_fails = 0;
    int route_fails = 0;
    for (int i = 0; i < M * N; i++) {
        if (Y_vmlal[i] != Y_ref[i]) {
            local_fails++;
            if (local_fails <= 3) {
                fprintf(stderr,
                    "  FAIL %s i=%d vmlal=%d ref=%d\n",
                    label, i, (int)Y_vmlal[i], (int)Y_ref[i]);
            }
        }
        if (Y_route && Y_route[i] != Y_ref[i]) {
            route_fails++;
            if (route_fails <= 3) {
                fprintf(stderr,
                    "  FAIL %s i=%d route=%d ref=%d\n",
                    label, i, (int)Y_route[i], (int)Y_ref[i]);
            }
        }
    }
    if (local_fails == 0 && route_fails == 0) {
        if (verbose) printf("  %-50s : PASS  (M=%d K=%d N=%d, %d cells, vmlal+route)\n",
                            label, M, K, N, M * N);
    } else {
        printf("  %-50s : FAIL  (vmlal=%d, route=%d / %d mismatches)\n",
               label, local_fails, route_fails, M * N);
        g_fails += local_fails + route_fails;
    }

    free(X); free(W_unp); free(W_pack); free(Y_vmlal); free(Y_ref);
    free(Y_route);
}

/* Convenience wrapper preserving the original verbose-on-pass behavior. */
static void test_config(int M, int K, int N, double density, double pos_frac,
                        uint32_t seed, const char* label) {
    test_config_v(M, K, N, density, pos_frac, seed, label, 1);
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

/* R-G1: stochastic bit-exact stress. Generates N_RANDOM random
 * configurations spanning a representative shape space and verifies
 * each is bit-exact. Per red-team C1 (sample was thin at 23 configs). */
static void test_random_stress(int n_random) {
    printf("\n-- R-G1: Random stress (%d configs) --\n", n_random);
    int M_choices[] = {1, 2, 4, 8, 16, 32};
    int N_choices[] = {1, 2, 4, 8, 16, 32};
    int K_choices[] = {1, 5, 16, 17, 31, 64, 100, 256, 1024};
    double dens_choices[] = {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0};
    double pos_choices[]  = {0.0, 0.3, 0.5, 0.7, 1.0};

    srand(7777);
    int initial_fails = g_fails;
    for (int t = 0; t < n_random; t++) {
        int M = M_choices[rand() % (int)(sizeof(M_choices)/sizeof(int))];
        int N = N_choices[rand() % (int)(sizeof(N_choices)/sizeof(int))];
        int K = K_choices[rand() % (int)(sizeof(K_choices)/sizeof(int))];
        double dens = dens_choices[rand() % (int)(sizeof(dens_choices)/sizeof(double))];
        double pos  = pos_choices[rand() % (int)(sizeof(pos_choices)/sizeof(double))];
        uint32_t seed = (uint32_t)(rand() * 100 + t);
        char label[80];
        snprintf(label, sizeof(label), "rand[%4d] M=%d K=%d N=%d d=%.1f p=%.1f",
                 t, M, K, N, dens, pos);
        test_config_v(M, K, N, dens, pos, seed, label, /*verbose=*/0);
    }
    int new_fails = g_fails - initial_fails;
    if (new_fails == 0) {
        printf("  R-G1 PASS (%d/%d configs bit-exact)\n", n_random, n_random);
    } else {
        printf("  R-G1 FAIL (%d / %d configs failed)\n", new_fails, n_random);
    }
}

/* R-G2: saturation-edge cases. Construct W and X such that the dot
 * product magnitude exceeds MAX_VAL — output gets clamped + flag set.
 * Verify both vmlal and scalar paths produce the same clamped output
 * AND the same SATURATED flag bit pattern. Per red-team C1 (no
 * saturation-edge inputs in original sample). */
static void test_saturation_edge(void) {
    printf("\n-- R-G2: Saturation-edge cases --\n");
    /* For K=64, all trits=+1, all activations=+MAX_VAL:
     *   acc = 64 * 5.81e8 = 3.7e10, clamps to MAX_VAL with SATURATED flag. */
    int K = 64;
    int M = 4;
    int N = 4;
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    int n_flag_bytes = M4T_FLAG_BYTES(M * N);

    m4t_mtfp_t* X       = malloc(sizeof(m4t_mtfp_t) * (size_t)M * K);
    m4t_trit_t* W_unp   = malloc(sizeof(m4t_trit_t) * (size_t)N * K);
    uint8_t*    W_pack  = malloc((size_t)N * Kp);
    m4t_mtfp_t* Y_v     = malloc(sizeof(m4t_mtfp_t) * (size_t)M * N);
    m4t_mtfp_t* Y_s     = malloc(sizeof(m4t_mtfp_t) * (size_t)M * N);
    uint8_t* flags_v    = calloc(n_flag_bytes, 1);
    uint8_t* flags_s    = calloc(n_flag_bytes, 1);

    /* Three saturation-edge configs. */
    struct { const char* label; m4t_mtfp_t x_val; m4t_trit_t w_val; } cases[] = {
        { "+MAX_VAL × +1 → +sat",  M4T_MTFP_MAX_VAL,  1 },
        { "+MAX_VAL × -1 → -sat", M4T_MTFP_MAX_VAL, -1 },
        { "-MAX_VAL × +1 → -sat", -M4T_MTFP_MAX_VAL, 1 },
    };
    int local_fails = 0;
    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        for (int i = 0; i < M * K; i++) X[i] = cases[c].x_val;
        for (int j = 0; j < N * K; j++) W_unp[j] = cases[c].w_val;
        for (int j = 0; j < N; j++) {
            pack_trits(W_pack + (size_t)j * Kp, W_unp + (size_t)j * K, K);
        }
        memset(flags_v, 0, n_flag_bytes);
        memset(flags_s, 0, n_flag_bytes);
        m4t_mtfp_ternary_matmul_bt           (Y_v, X, W_pack, flags_v, M, K, N);
        m4t_mtfp_ternary_matmul_bt_scalar_ref(Y_s, X, W_pack, flags_s, M, K, N);
        int out_fails = 0;
        for (int i = 0; i < M * N; i++) if (Y_v[i] != Y_s[i]) out_fails++;
        int flag_fails = memcmp(flags_v, flags_s, n_flag_bytes) == 0 ? 0 : 1;
        if (out_fails == 0 && flag_fails == 0) {
            /* Verify saturation actually happened. */
            int sat_set = 0;
            for (int i = 0; i < M * N; i++) {
                if (Y_v[i] == M4T_MTFP_MAX_VAL || Y_v[i] == -M4T_MTFP_MAX_VAL) {
                    sat_set = 1; break;
                }
            }
            printf("  %-30s : PASS (clamp matches, flags match%s)\n",
                   cases[c].label, sat_set ? ", saturation triggered" : "");
        } else {
            printf("  %-30s : FAIL (out_fails=%d flag_mismatch=%d)\n",
                   cases[c].label, out_fails, flag_fails);
            local_fails++;
        }
    }
    if (local_fails > 0) g_fails += local_fails;

    free(X); free(W_unp); free(W_pack); free(Y_v); free(Y_s);
    free(flags_v); free(flags_s);
}

/* R-G4: aliasing assertions. Both Y==X and Y==W_packed are forbidden
 * by the kernel's asserts. Test BOTH cases via fork-and-verify-SIGABRT.
 * Per red-team M1 (only Y==X was tested in original cycle). */
static int run_alias(int alias_w) {
    enum { K = 16 };
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    /* Allocate one buffer big enough to use as Y, X, OR W_packed. */
    size_t bytes = sizeof(m4t_mtfp_t) * K > (size_t)Kp ? sizeof(m4t_mtfp_t) * K : (size_t)Kp;
    m4t_mtfp_t* shared = malloc(bytes);
    memset(shared, 0, bytes);
    m4t_mtfp_t X[K] = {0};
    uint8_t W_pack[Kp]; memset(W_pack, 0, Kp);
    if (alias_w) {
        /* Y == W_packed */
        m4t_mtfp_ternary_matmul_bt(shared, X, (uint8_t*)shared, NULL, 1, K, 1);
    } else {
        /* Y == X */
        m4t_mtfp_ternary_matmul_bt(shared, shared, W_pack, NULL, 1, K, 1);
    }
    fprintf(stderr, "FAIL: kernel did not abort on alias\n");
    free(shared);
    return 1;
}

static void test_aliasing(void) {
    printf("\n-- R-G4: Aliasing assertions --\n");
    const char* labels[] = { "Y == X       ", "Y == W_packed" };
    int alias_fails = 0;
    for (int kind = 0; kind < 2; kind++) {
        pid_t pid = fork();
        if (pid == 0) { exit(run_alias(kind)); }
        int status; waitpid(pid, &status, 0);
        if (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT) {
            printf("  alias %s : PASS (SIGABRT)\n", labels[kind]);
        } else {
            printf("  alias %s : FAIL\n", labels[kind]);
            alias_fails++;
        }
    }
    if (alias_fails > 0) g_fails += alias_fails;
}

int main(void) {
    printf("=== Bit-exact: production NEON vs scalar_ref ===\n");

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

    /* R-G1: 1000 random configurations. */
    test_random_stress(1000);
    if (g_fails > 0) { printf("\nFAIL: random stress\n"); return 1; }

    /* R-G2: saturation-edge cases. */
    test_saturation_edge();
    if (g_fails > 0) { printf("\nFAIL: saturation-edge\n"); return 1; }

    /* R-G4: aliasing assertions. */
    test_aliasing();
    if (g_fails > 0) { printf("\nFAIL: aliasing\n"); return 1; }

    printf("\nPASS: all configurations bit-exact (curated + 1000 random + sat + alias)\n");

    /* T-G8 + R-G3: multi-shape perf comparison. */
    printf("\n=== Perf comparison: production NEON vs scalar_ref (min-of-5) ===\n");
    printf("BATCHED shape sweep — verify speedup is shape-stable (R-G3):\n");
    perf_compare(64,  4096, 64,  5,   "BATCHED-A");  /* original */
    perf_compare(8,   4096, 8,   25,  "BATCHED-B");  /* slim aspect */
    perf_compare(128, 1024, 128, 5,   "BATCHED-C");  /* wide aspect */
    perf_compare(32,  512,  32,  20,  "BATCHED-D");  /* mid */
    perf_compare(16,  256,  16,  100, "BATCHED-E");  /* small bulk */
    printf("\nTIGHT-LOOP:\n");
    perf_compare(4,   64,   4,   500, "TIGHT-LOOP"); /* small dims, per-call overhead */

    return 0;
}
