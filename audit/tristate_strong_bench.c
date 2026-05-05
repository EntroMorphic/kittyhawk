/*
 * audit/tristate_strong_bench.c — strong-claim measurement harness
 *
 * Per journal/tristate_strong_synthesize.md + journal/tristate_strong_redteam.md
 * + post-redteam forward pointer to test sub-2-bits/cell base-3 packing.
 *
 * Runs five NEON-only kernels:
 *   Path A  base3_packed_matmul_neon         — base-3 ternary 4-in-8 via SDOT
 *   Path B  b2b_honest_matmul_neon           — B2-B sign+mask separate decode
 *   Path B' b2b_skip_matmul_neon             — B2-B with all-masked-block skip
 *   Path C  b2b_optimal_matmul_neon          — B2-B unified TBL decode
 *   Path D  base3_5in8_matmul_neon           — base-3 5-trits-in-8-bits (1.6 b/c)
 *
 * Plus external grounding via the substrate's
 *   m4t_ternary_dot_matmul_bt — externally validated NEON ternary matmul.
 *
 * K values must be multiples of both 16 (Path A/B/C inner-loop alignment)
 * AND 80 (Path D inner-loop alignment): {80, 320, 1280}.
 *
 * Verification (NEON-only, no scalar reference):
 *   - All 4 audit kernels must produce bit-exact identical Y.
 *   - All 4 must match the substrate's m4t_ternary_dot_matmul_bt output.
 *   - This gives EXTERNAL grounding (substrate has its own scalar-ref
 *     test oracle inside libm4t; we just trust the substrate's verified
 *     NEON kernel as ground truth here).
 *
 * Reports:
 *   - Per-config wall-clock (informational; not gating).
 *   - Skip firing rate (R-G3): empirically count blocks skipped by Path B'.
 */

#include "b2b_matmul.h"

/* External grounding: substrate's verified ternary matmul. */
#include "m4t_ternary_matmul.h"
#include "m4t_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

/* Deterministic xorshift32 PRNG. */
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

/* K values must be multiples of 80 (Path D inner-loop alignment).
 * 80 = 16 × 5: also aligns to Path A/B/C inner block (16-cell).
 * 320, 1280: same alignment. */
static const Config CONFIGS[] = {
    {   80,  0.20,   0.20 },
    {   80,  0.20,   0.60 },
    {   80,  0.60,   0.20 },
    {   80,  0.60,   0.60 },
    {  320,  0.20,   0.20 },
    {  320,  0.20,   0.60 },
    {  320,  0.60,   0.20 },
    {  320,  0.60,   0.60 },
    { 1280,  0.20,   0.20 },
    { 1280,  0.20,   0.60 },
    { 1280,  0.60,   0.20 },
    { 1280,  0.60,   0.60 },
};
#define N_CONFIGS (int)(sizeof(CONFIGS)/sizeof(CONFIGS[0]))
#define N_SEEDS  5
#define M_BATCH 8
#define N_HIDDEN 64
#define REPS 2000

int main(void) {
    printf("config_idx,K,w_zero,a_zero,seed,"
           "ok_a_eq_b,ok_a_eq_skip,ok_a_eq_optimal,ok_a_eq_substrate,ok_a_eq_5in8,"
           "skip_rate,"
           "ms_a,ms_b,ms_skip,ms_optimal,ms_substrate,ms_5in8\n");

    int total_runs = 0;
    int total_a_eq_b = 0, total_a_eq_skip = 0;
    int total_a_eq_optimal = 0, total_a_eq_substrate = 0;
    int total_a_eq_5in8 = 0;
    long long total_skip_blocks = 0, total_blocks = 0;

    for (int c = 0; c < N_CONFIGS; c++) {
        const Config* cfg = &CONFIGS[c];
        int K = cfg->K, M = M_BATCH, N = N_HIDDEN;
        int Kp = (K + 3) / 4;

        double sum_ms_a = 0, sum_ms_b = 0, sum_ms_skip = 0;
        double sum_ms_optimal = 0, sum_ms_substrate = 0, sum_ms_5in8 = 0;
        int sum_ok_a_eq_b = 0, sum_ok_a_eq_skip = 0;
        int sum_ok_a_eq_optimal = 0, sum_ok_a_eq_substrate = 0;
        int sum_ok_a_eq_5in8 = 0;
        long long cfg_skip_blocks = 0, cfg_blocks = 0;

        for (int s = 0; s < N_SEEDS; s++) {
            uint32_t seed = (uint32_t)(c * 1000 + s + 1);
            rng_t rng;
            rng_init(&rng, seed);

            int Kp5 = K / 5;  /* 5-in-8 packed bytes per row */
            int8_t*  X    = (int8_t*)calloc((size_t)M * K, sizeof(int8_t));
            int8_t*  W    = (int8_t*)calloc((size_t)N * K, sizeof(int8_t));
            uint8_t* Wp_a = (uint8_t*)calloc((size_t)N * Kp,  sizeof(uint8_t));
            uint8_t* Wp_b = (uint8_t*)calloc((size_t)N * Kp,  sizeof(uint8_t));
            uint8_t* Wp_d = (uint8_t*)calloc((size_t)N * Kp5, sizeof(uint8_t));
            int32_t* Ya   = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));
            int32_t* Yb   = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));
            int32_t* Yk   = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));
            int32_t* Yo   = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));
            int32_t* Yd   = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));
            m4t_mtfp_t* Ys = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));

            gen_ternary(X, M * K, cfg->a_zero, &rng);
            gen_ternary(W, N * K, cfg->w_zero, &rng);

            for (int j = 0; j < N; j++) {
                base3_pack    (Wp_a + (size_t)j * Kp,  W + (size_t)j * K, K);
                b2b_pack      (Wp_b + (size_t)j * Kp,  W + (size_t)j * K, K);
                base3_5in8_pack(Wp_d + (size_t)j * Kp5, W + (size_t)j * K, K);
            }

            int skip_count = 0;

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

            /* Path B' — B2-B with skip. Capture skip count from one rep. */
            b2b_skip_matmul_neon(Yk, X, Wp_b, M, K, N, &skip_count);
            t0 = monotonic_ms();
            for (int r = 0; r < REPS; r++) {
                b2b_skip_matmul_neon(Yk, X, Wp_b, M, K, N, NULL);
            }
            t1 = monotonic_ms();
            double ms_skip = t1 - t0;

            /* Path C — B2-B optimal (unified TBL). */
            t0 = monotonic_ms();
            for (int r = 0; r < REPS; r++) {
                b2b_optimal_matmul_neon(Yo, X, Wp_b, M, K, N);
            }
            t1 = monotonic_ms();
            double ms_optimal = t1 - t0;

            /* Substrate cross-check (R-G2 external grounding). */
            t0 = monotonic_ms();
            for (int r = 0; r < REPS; r++) {
                /* m4t_ternary_dot_matmul_bt takes UNPACKED ternary X and W.
                 * X stride is K (already unpacked); W stride is K (also
                 * unpacked here — substrate kernel doesn't take packed). */
                m4t_ternary_dot_matmul_bt(Ys, X, W, M, K, N);
            }
            t1 = monotonic_ms();
            double ms_substrate = t1 - t0;

            /* Path D — base-3 5-in-8 (sub-2-bit) packed. */
            t0 = monotonic_ms();
            for (int r = 0; r < REPS; r++) {
                base3_5in8_matmul_neon(Yd, X, Wp_d, M, K, N);
            }
            t1 = monotonic_ms();
            double ms_5in8 = t1 - t0;

            /* Bit-exact verification: all four audit kernels match. */
            int ok_a_eq_b = (memcmp(Ya, Yb, (size_t)M * N * sizeof(int32_t)) == 0) ? 1 : 0;
            int ok_a_eq_skip = (memcmp(Ya, Yk, (size_t)M * N * sizeof(int32_t)) == 0) ? 1 : 0;
            int ok_a_eq_optimal = (memcmp(Ya, Yo, (size_t)M * N * sizeof(int32_t)) == 0) ? 1 : 0;

            /* External cross-check vs substrate. Y_substrate is m4t_mtfp_t
             * (int32, same as int32_t for this build). Compare lane-wise. */
            int ok_a_eq_substrate = 1;
            for (int i = 0; i < M * N; i++) {
                if ((int32_t)Ys[i] != Ya[i]) { ok_a_eq_substrate = 0; break; }
            }

            int ok_a_eq_5in8 = (memcmp(Ya, Yd, (size_t)M * N * sizeof(int32_t)) == 0) ? 1 : 0;

            /* Skip rate */
            int total_blocks_per_run = M * N * (K / 16);
            double skip_rate = (double)skip_count / (double)total_blocks_per_run;
            cfg_skip_blocks += skip_count;
            cfg_blocks      += total_blocks_per_run;

            sum_ms_a += ms_a;
            sum_ms_b += ms_b;
            sum_ms_skip += ms_skip;
            sum_ms_optimal += ms_optimal;
            sum_ms_substrate += ms_substrate;
            sum_ms_5in8 += ms_5in8;
            sum_ok_a_eq_b += ok_a_eq_b;
            sum_ok_a_eq_skip += ok_a_eq_skip;
            sum_ok_a_eq_optimal += ok_a_eq_optimal;
            sum_ok_a_eq_substrate += ok_a_eq_substrate;
            sum_ok_a_eq_5in8 += ok_a_eq_5in8;

            printf("%d,%d,%.2f,%.2f,%u,%d,%d,%d,%d,%d,%.6f,"
                   "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                c, K, cfg->w_zero, cfg->a_zero, seed,
                ok_a_eq_b, ok_a_eq_skip, ok_a_eq_optimal, ok_a_eq_substrate, ok_a_eq_5in8,
                skip_rate,
                ms_a, ms_b, ms_skip, ms_optimal, ms_substrate, ms_5in8);

            total_runs++;
            total_a_eq_b += ok_a_eq_b;
            total_a_eq_skip += ok_a_eq_skip;
            total_a_eq_optimal += ok_a_eq_optimal;
            total_a_eq_substrate += ok_a_eq_substrate;
            total_a_eq_5in8 += ok_a_eq_5in8;
            total_skip_blocks += skip_count;
            total_blocks += total_blocks_per_run;

            free(X); free(W); free(Wp_a); free(Wp_b); free(Wp_d);
            free(Ya); free(Yb); free(Yk); free(Yo); free(Yd); free(Ys);
        }

        double cfg_skip_rate = (double)cfg_skip_blocks / (double)cfg_blocks;
        fprintf(stderr,
            "[summary] cfg %d K=%d w_z=%.2f a_z=%.2f | "
            "verify(/5): a==b:%d skip:%d opt:%d sub:%d 5in8:%d | "
            "skip_rate=%.6f | "
            "ms (mean over %d reps × %d seeds): "
            "A=%.2f B=%.2f Bskip=%.2f Bopt=%.2f Sub=%.2f 5in8=%.2f | "
            "B/A=%.2fx Bskip/A=%.2fx Bopt/A=%.2fx Sub/A=%.2fx 5in8/A=%.2fx\n",
            c, K, cfg->w_zero, cfg->a_zero,
            sum_ok_a_eq_b, sum_ok_a_eq_skip, sum_ok_a_eq_optimal,
            sum_ok_a_eq_substrate, sum_ok_a_eq_5in8,
            cfg_skip_rate,
            REPS, N_SEEDS,
            sum_ms_a / N_SEEDS, sum_ms_b / N_SEEDS, sum_ms_skip / N_SEEDS,
            sum_ms_optimal / N_SEEDS, sum_ms_substrate / N_SEEDS, sum_ms_5in8 / N_SEEDS,
            sum_ms_b / sum_ms_a, sum_ms_skip / sum_ms_a,
            sum_ms_optimal / sum_ms_a, sum_ms_substrate / sum_ms_a,
            sum_ms_5in8 / sum_ms_a);
    }

    fprintf(stderr,
        "[overall] %d runs | "
        "verify a==b:%d/%d skip:%d/%d opt:%d/%d sub:%d/%d 5in8:%d/%d | "
        "skip rate: %lld/%lld = %.6f\n",
        total_runs,
        total_a_eq_b, total_runs, total_a_eq_skip, total_runs,
        total_a_eq_optimal, total_runs, total_a_eq_substrate, total_runs,
        total_a_eq_5in8, total_runs,
        total_skip_blocks, total_blocks,
        (double)total_skip_blocks / (double)total_blocks);

    int all_pass = (total_a_eq_b == total_runs)
                && (total_a_eq_skip == total_runs)
                && (total_a_eq_optimal == total_runs)
                && (total_a_eq_substrate == total_runs)
                && (total_a_eq_5in8 == total_runs);

    if (!all_pass) {
        fprintf(stderr, "[FAIL] verification failed — kernels disagree\n");
        return 1;
    }
    fprintf(stderr,
        "[PASS] all four audit kernels + substrate cross-check bit-exact equivalent\n");
    return 0;
}
