/*
 * bench_m4t_matmul_tile.c — wall-clock probe for the matmul retile.
 *
 * Per journal/m4t_matmul_tile_synthesize.md G3.
 *
 * Times the two retiled libm4t kernels:
 *   - m4t_mtfp_ternary_matmul_bt   (vmlal_s32 route, MTFP19 X)
 *   - m4t_ternary_dot_matmul_bt    (SDOT route, ternary X)
 *
 * Workload: M=8, N=64, K ∈ {1280, 12800, 51200}. Captures L1-resident
 * through L2-resident regimes. Min-of-N=5 sampling per measurement,
 * per CONTRIBUTING throughput-microbench discipline.
 *
 * NOT in ctest. Run manually for measurement evidence.
 */

#include "m4t_ternary_matmul.h"
#include "m4t_trit_pack.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void gen_ternary(m4t_trit_t* dst, int n, uint32_t* state) {
    /* Trivial xorshift32; specifics don't matter for throughput timing. */
    for (int i = 0; i < n; i++) {
        uint32_t x = *state;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        *state = x;
        int v = (int)(x % 3) - 1;  /* {-1, 0, +1} */
        dst[i] = (m4t_trit_t)v;
    }
}

static void gen_mtfp19(m4t_mtfp_t* dst, int n, uint32_t* state) {
    for (int i = 0; i < n; i++) {
        uint32_t x = *state;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        *state = x;
        int v = (int)(x % 1024) - 512;
        dst[i] = (m4t_mtfp_t)v;
    }
}

#define N_REPS 5
#define M 8
#define N 64

static double measure_mtfp_ternary(int K, int reps_per_run) {
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    m4t_mtfp_t* X     = (m4t_mtfp_t*)calloc((size_t)M * K, sizeof(m4t_mtfp_t));
    m4t_trit_t* W_unp = (m4t_trit_t*)calloc((size_t)N * K, sizeof(m4t_trit_t));
    uint8_t*    W_pkd = (uint8_t*)   calloc((size_t)N * Kp, 1);
    m4t_mtfp_t* Y     = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));

    uint32_t state = 0xdeadbeefu;
    gen_mtfp19(X, M * K, &state);
    gen_ternary(W_unp, N * K, &state);
    for (int j = 0; j < N; j++) {
        m4t_pack_trits_1d(W_pkd + (size_t)j * Kp, W_unp + (size_t)j * K, K);
    }

    double min_ms = 1e30;
    for (int trial = 0; trial < N_REPS; trial++) {
        double t0 = monotonic_ms();
        for (int r = 0; r < reps_per_run; r++) {
            m4t_mtfp_ternary_matmul_bt(Y, X, W_pkd, NULL, M, K, N);
        }
        double t1 = monotonic_ms();
        if (t1 - t0 < min_ms) min_ms = t1 - t0;
    }

    free(X); free(W_unp); free(W_pkd); free(Y);
    return min_ms / reps_per_run;
}

static double measure_ternary_dot(int K, int reps_per_run) {
    m4t_trit_t* X = (m4t_trit_t*)calloc((size_t)M * K, sizeof(m4t_trit_t));
    m4t_trit_t* W = (m4t_trit_t*)calloc((size_t)N * K, sizeof(m4t_trit_t));
    m4t_mtfp_t* Y = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));

    uint32_t state = 0xdeadbeefu;
    gen_ternary(X, M * K, &state);
    gen_ternary(W, N * K, &state);

    double min_ms = 1e30;
    for (int trial = 0; trial < N_REPS; trial++) {
        double t0 = monotonic_ms();
        for (int r = 0; r < reps_per_run; r++) {
            m4t_ternary_dot_matmul_bt(Y, X, W, M, K, N);
        }
        double t1 = monotonic_ms();
        if (t1 - t0 < min_ms) min_ms = t1 - t0;
    }

    free(X); free(W); free(Y);
    return min_ms / reps_per_run;
}

int main(void) {
    printf("# bench_m4t_matmul_tile — post-retile wall-clock at M=8, N=64\n");
    printf("# min-of-%d trials, time per call (ms)\n\n", N_REPS);
    printf("K\tmtfp_ternary_ms\tternary_dot_ms\n");

    int Ks[] = { 1280, 12800, 51200 };
    int reps[] = { 1000, 100, 25 };  /* scale to keep runtime bounded */

    for (int i = 0; i < (int)(sizeof(Ks)/sizeof(Ks[0])); i++) {
        int K = Ks[i];
        double t1 = measure_mtfp_ternary(K, reps[i]);
        double t2 = measure_ternary_dot(K, reps[i]);
        printf("%d\t%.4f\t\t%.4f\n", K, t1, t2);
    }

    return 0;
}
