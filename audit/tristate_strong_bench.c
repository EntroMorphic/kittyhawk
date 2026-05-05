/*
 * audit/tristate_strong_bench.c — strong-claim measurement harness
 *
 * Per journal/tristate_strong_synthesize.md. Runs the three NEON kernels
 * (Path A base-3, Path B honest, Path B' skip) on the audit's workload
 * and verifies bit-exact output equivalence. Prints per-config wall-clock
 * timings as informational sanity check (gate is op count via objdump,
 * NOT throughput).
 *
 * NEON-only end-to-end. No scalar reference. Verification is NEON-vs-NEON
 * cross-check: all three kernels must produce identical Y.
 *
 * Output:
 *   stdout:  CSV per (config, seed): K, w_zero, a_zero, seed,
 *            verified_a_eq_b, verified_a_eq_skip, ms_a, ms_b, ms_skip
 *   stderr:  per-config summary
 */

#include "b2b_matmul.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

/* Deterministic xorshift32 PRNG (mirrors tristate_audit.c). */
typedef struct { uint32_t state; } rng_t;

static uint32_t rng_next(rng_t* r) {
    uint32_t x = r->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    r->state = x;
    return x;
}
static void rng_init(rng_t* r, uint32_t seed) {
    r->state = seed ? seed : 0xdeadbeefu;
    for (int i = 0; i < 8; i++) (void)rng_next(r);
}
static double rng_uniform(rng_t* r) {
    return (double)(rng_next(r) >> 8) / (double)(1u << 24);
}
static int rng_sign(rng_t* r) {
    return (rng_next(r) & 1u) ? 1 : -1;
}

static void gen_ternary(int8_t* dst, int n, double p_zero, rng_t* r) {
    for (int i = 0; i < n; i++) {
        double u = rng_uniform(r);
        if (u < p_zero) dst[i] = 0;
        else            dst[i] = (int8_t)rng_sign(r);
    }
}

/* Wall-clock helper. */
static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

typedef struct {
    int K;
    double w_zero;
    double a_zero;
} Config;

static const Config CONFIGS[] = {
    {   64,  0.20,   0.20 },
    {   64,  0.20,   0.60 },
    {   64,  0.60,   0.20 },
    {   64,  0.60,   0.60 },
    {  256,  0.20,   0.20 },
    {  256,  0.20,   0.60 },
    {  256,  0.60,   0.20 },
    {  256,  0.60,   0.60 },
    { 1024,  0.20,   0.20 },
    { 1024,  0.20,   0.60 },
    { 1024,  0.60,   0.20 },
    { 1024,  0.60,   0.60 },
};
#define N_CONFIGS (int)(sizeof(CONFIGS)/sizeof(CONFIGS[0]))
#define N_SEEDS  5

/* Workload dimensions: fixed M=8 outer batch, N=64 hidden, P=8 output
 * (matches audit's 2-layer shape). The strong-claim test only exercises
 * the L1 layer, so we run a single matmul Y = X @ W^T with shape
 * X[M, K] × W[N, K] → Y[M, N]. */
#define M_BATCH 8
#define N_HIDDEN 64

/* Repeat each kernel to get a meaningful wall-clock signal on small K. */
#define REPS 2000

int main(void) {
    printf("config_idx,K,w_zero,a_zero,seed,"
           "ok_a_eq_b,ok_a_eq_skip,"
           "ms_a,ms_b,ms_skip\n");

    int total_runs = 0;
    int total_a_eq_b = 0;
    int total_a_eq_skip = 0;

    for (int c = 0; c < N_CONFIGS; c++) {
        const Config* cfg = &CONFIGS[c];
        int K = cfg->K, M = M_BATCH, N = N_HIDDEN;
        int Kp = (K + 3) / 4;

        double sum_ms_a = 0, sum_ms_b = 0, sum_ms_skip = 0;
        int sum_ok_a_eq_b = 0, sum_ok_a_eq_skip = 0;

        for (int s = 0; s < N_SEEDS; s++) {
            uint32_t seed = (uint32_t)(c * 1000 + s + 1);
            rng_t rng;
            rng_init(&rng, seed);

            int8_t*  X   = (int8_t*)calloc((size_t)M * K, sizeof(int8_t));
            int8_t*  W   = (int8_t*)calloc((size_t)N * K, sizeof(int8_t));
            uint8_t* Wp_a = (uint8_t*)calloc((size_t)N * Kp, sizeof(uint8_t));
            uint8_t* Wp_b = (uint8_t*)calloc((size_t)N * Kp, sizeof(uint8_t));
            int32_t* Ya  = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));
            int32_t* Yb  = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));
            int32_t* Yk  = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));

            gen_ternary(X, M * K, cfg->a_zero, &rng);
            gen_ternary(W, N * K, cfg->w_zero, &rng);

            /* Pack W in both formats. */
            for (int j = 0; j < N; j++) {
                base3_pack(Wp_a + (size_t)j * Kp, W + (size_t)j * K, K);
                b2b_pack  (Wp_b + (size_t)j * Kp, W + (size_t)j * K, K);
            }

            /* Path A — base-3 packed via SDOT. */
            double t0 = monotonic_ms();
            for (int r = 0; r < REPS; r++) {
                base3_packed_matmul_neon(Ya, X, Wp_a, M, K, N);
            }
            double t1 = monotonic_ms();
            double ms_a = t1 - t0;

            /* Path B — B2-B honest. */
            t0 = monotonic_ms();
            for (int r = 0; r < REPS; r++) {
                b2b_honest_matmul_neon(Yb, X, Wp_b, M, K, N);
            }
            t1 = monotonic_ms();
            double ms_b = t1 - t0;

            /* Path B' — B2-B with skip. */
            t0 = monotonic_ms();
            for (int r = 0; r < REPS; r++) {
                b2b_skip_matmul_neon(Yk, X, Wp_b, M, K, N);
            }
            t1 = monotonic_ms();
            double ms_skip = t1 - t0;

            /* Bit-exact verification: all three must produce identical Y. */
            int ok_a_eq_b = (memcmp(Ya, Yb, (size_t)M * N * sizeof(int32_t)) == 0) ? 1 : 0;
            int ok_a_eq_skip = (memcmp(Ya, Yk, (size_t)M * N * sizeof(int32_t)) == 0) ? 1 : 0;

            sum_ms_a += ms_a;
            sum_ms_b += ms_b;
            sum_ms_skip += ms_skip;
            sum_ok_a_eq_b += ok_a_eq_b;
            sum_ok_a_eq_skip += ok_a_eq_skip;

            printf("%d,%d,%.2f,%.2f,%u,%d,%d,%.3f,%.3f,%.3f\n",
                c, K, cfg->w_zero, cfg->a_zero, seed,
                ok_a_eq_b, ok_a_eq_skip,
                ms_a, ms_b, ms_skip);

            total_runs++;
            total_a_eq_b += ok_a_eq_b;
            total_a_eq_skip += ok_a_eq_skip;

            free(X); free(W); free(Wp_a); free(Wp_b);
            free(Ya); free(Yb); free(Yk);
        }

        fprintf(stderr,
            "[summary] cfg %d K=%d w_z=%.2f a_z=%.2f | "
            "verify a==b: %d/%d  a==skip: %d/%d | "
            "mean ms (over %d reps × %d seeds): A=%.3f  B=%.3f  Bskip=%.3f | "
            "B/A=%.2fx Bskip/A=%.2fx\n",
            c, K, cfg->w_zero, cfg->a_zero,
            sum_ok_a_eq_b, N_SEEDS, sum_ok_a_eq_skip, N_SEEDS,
            REPS, N_SEEDS,
            sum_ms_a / N_SEEDS, sum_ms_b / N_SEEDS, sum_ms_skip / N_SEEDS,
            sum_ms_b / sum_ms_a, sum_ms_skip / sum_ms_a);
    }

    fprintf(stderr,
        "[overall] %d runs total; verify a==b: %d/%d  a==skip: %d/%d\n",
        total_runs, total_a_eq_b, total_runs,
        total_a_eq_skip, total_runs);

    /* Cycle gate: bit-exact verification must be 100%. */
    if (total_a_eq_b != total_runs || total_a_eq_skip != total_runs) {
        fprintf(stderr, "[FAIL] precision gate failed — kernels disagree\n");
        return 1;
    }
    fprintf(stderr, "[PASS] precision gate: all kernels bit-exact equivalent\n");
    return 0;
}
