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
#include <math.h>

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

/* R-G1 cache flush: walk a 32 MB buffer to evict the prior kernel's W and X
 * from L1/L2. M-series L2 is 12-16 MB; 32 MB exceeds it.
 *
 * volatile prevents the compiler from optimizing out the read. The xor
 * accumulator is returned via the global to keep the read live without
 * returning. */
static volatile uint8_t flush_sink;

static void cache_flush(uint8_t* flush_buf, size_t flush_size) {
    uint8_t s = 0;
    for (size_t i = 0; i < flush_size; i += 64) {  /* one byte per cache line */
        s ^= flush_buf[i];
    }
    flush_sink = s;
}

/* R-G3: per-config standard deviation alongside mean. */
static double stddev_arr(const double* xs, int n, double mean) {
    if (n < 2) return 0.0;
    double sumsq = 0.0;
    for (int i = 0; i < n; i++) {
        double d = xs[i] - mean;
        sumsq += d * d;
    }
    return sqrt(sumsq / (n - 1));
}

typedef struct {
    int K;
    int N;          /* per-config N (varies for DRAM-bound test) */
    double w_zero;
    double a_zero;
    int reps;
} Config;

/* K values must be multiples of 80 (Path D inner-loop alignment).
 * 80 = 16 × 5: also aligns to Path A/B/C inner block (16-cell).
 *
 * Cache analysis (with 2 bits/cell Path A packing, W bytes = N*K/4):
 *   N=64,K=80..1280:    W ≤ 20KB     fits in L1 (192KB on M-series).
 *   N=64,K=12800:       W ≈ 200KB    exceeds L1, fits in L2.
 *   N=64,K=51200:       W ≈ 800KB    exceeds L1, fits in L2.
 *   N=8192,K=12800:     W ≈ 25.6MB   EXCEEDS L2 (12-16MB) → DRAM-bound.
 *
 * Per-config reps scaled to keep total runtime ~bounded. */
static const Config CONFIGS[] = {
    /* L1-resident regime (compute-bound). Multi-distribution sweep, N=64. */
    {     80,   64,  0.20,   0.20, 2000 },
    {     80,   64,  0.20,   0.60, 2000 },
    {     80,   64,  0.60,   0.20, 2000 },
    {     80,   64,  0.60,   0.60, 2000 },
    {    320,   64,  0.20,   0.20, 2000 },
    {    320,   64,  0.20,   0.60, 2000 },
    {    320,   64,  0.60,   0.20, 2000 },
    {    320,   64,  0.60,   0.60, 2000 },
    {   1280,   64,  0.20,   0.20, 2000 },
    {   1280,   64,  0.20,   0.60, 2000 },
    {   1280,   64,  0.60,   0.20, 2000 },
    {   1280,   64,  0.60,   0.60, 2000 },
    /* L2-resident regime (W exceeds L1). N=64; BitNet-typical distribution. */
    {  12800,   64,  0.60,   0.60,  200 },
    {  25600,   64,  0.60,   0.60,  100 },
    {  51200,   64,  0.60,   0.60,   50 },
    /* R-G2: DRAM-bound regime (W exceeds L2). N=8192 pushes W well beyond
     * L2 capacity. REPS=3 per runtime budget; warm/cold per-call dominates. */
    {  12800, 8192,  0.60,   0.60,    3 },
};
#define N_CONFIGS (int)(sizeof(CONFIGS)/sizeof(CONFIGS[0]))
#define N_SEEDS  5
#define M_BATCH 8

int main(void) {
    /* R-G1: cache-flush buffer. 32 MB exceeds M-series L2 (12-16 MB);
     * walking it between kernels evicts the prior kernel's W and X. */
    const size_t FLUSH_SIZE = 32 * 1024 * 1024;
    uint8_t* flush_buf = (uint8_t*)calloc(FLUSH_SIZE, 1);
    if (!flush_buf) {
        fprintf(stderr, "[FAIL] could not allocate cache-flush buffer\n");
        return 1;
    }

    printf("config_idx,K,N,w_zero,a_zero,seed,"
           "ok_a_eq_b,ok_a_eq_skip,ok_a_eq_optimal,ok_a_eq_substrate,ok_a_eq_5in8,ok_a_eq_e,"
           "skip_rate,"
           "ms_a,ms_b,ms_skip,ms_optimal,ms_substrate,ms_5in8,ms_e\n");

    int total_runs = 0;
    int total_a_eq_b = 0, total_a_eq_skip = 0;
    int total_a_eq_optimal = 0, total_a_eq_substrate = 0;
    int total_a_eq_5in8 = 0, total_a_eq_e = 0;
    long long total_skip_blocks = 0, total_blocks = 0;

    for (int c = 0; c < N_CONFIGS; c++) {
        const Config* cfg = &CONFIGS[c];
        int K = cfg->K, M = M_BATCH, N = cfg->N;
        int Kp = (K + 3) / 4;

        double sum_ms_a = 0, sum_ms_b = 0, sum_ms_skip = 0;
        double sum_ms_optimal = 0, sum_ms_substrate = 0, sum_ms_5in8 = 0;
        double sum_ms_e = 0;
        int sum_ok_a_eq_b = 0, sum_ok_a_eq_skip = 0;
        int sum_ok_a_eq_optimal = 0, sum_ok_a_eq_substrate = 0;
        int sum_ok_a_eq_5in8 = 0, sum_ok_a_eq_e = 0;
        long long cfg_skip_blocks = 0, cfg_blocks = 0;
        /* R-G3: per-seed arrays for SD computation */
        double seeds_ms_a[N_SEEDS], seeds_ms_b[N_SEEDS], seeds_ms_skip[N_SEEDS];
        double seeds_ms_opt[N_SEEDS], seeds_ms_sub[N_SEEDS], seeds_ms_5in8[N_SEEDS];
        double seeds_ms_e[N_SEEDS];
        (void)seeds_ms_b; (void)seeds_ms_skip; (void)seeds_ms_opt;
        (void)seeds_ms_sub; (void)seeds_ms_e;

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
            uint8_t* Xp_e = (uint8_t*)calloc((size_t)M * Kp,  sizeof(uint8_t));
            uint8_t* Xp_f = (uint8_t*)calloc((size_t)M * Kp,  sizeof(uint8_t));
            int32_t* Ya   = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));
            int32_t* Yb   = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));
            int32_t* Yk   = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));
            int32_t* Yo   = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));
            int32_t* Yd   = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));
            int32_t* Ye   = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));
            int32_t* Yf   = (int32_t*)calloc((size_t)M * N, sizeof(int32_t));
            m4t_mtfp_t* Ys = (m4t_mtfp_t*)calloc((size_t)M * N, sizeof(m4t_mtfp_t));

            gen_ternary(X, M * K, cfg->a_zero, &rng);
            gen_ternary(W, N * K, cfg->w_zero, &rng);

            for (int j = 0; j < N; j++) {
                base3_pack    (Wp_a + (size_t)j * Kp,  W + (size_t)j * K, K);
                b2b_pack      (Wp_b + (size_t)j * Kp,  W + (size_t)j * K, K);
                base3_5in8_pack(Wp_d + (size_t)j * Kp5, W + (size_t)j * K, K);
            }
            /* Pack X for Path E (base-3 4-in-8) and Path F (B2-B 4-in-8). */
            for (int i = 0; i < M; i++) {
                base3_pack(Xp_e + (size_t)i * Kp, X + (size_t)i * K, K);
                b2b_pack  (Xp_f + (size_t)i * Kp, X + (size_t)i * K, K);
            }

            int skip_count = 0;

            /* R-G1: cache-flush before each kernel run isolates cold-cache
             * timings. Without this, kernel n+1 finds W warm in L1/L2 from
             * kernel n's run, biasing memory-bandwidth-bound measurements. */

            cache_flush(flush_buf, FLUSH_SIZE);
            /* Path A — base-3 packed via SDOT. */
            double t0 = monotonic_ms();
            for (int r = 0; r < cfg->reps; r++) {
                base3_packed_matmul_neon(Ya, X, Wp_a, M, K, N);
            }
            double t1 = monotonic_ms();
            double ms_a = t1 - t0;

            cache_flush(flush_buf, FLUSH_SIZE);
            /* Path B — B2-B honest. */
            t0 = monotonic_ms();
            for (int r = 0; r < cfg->reps; r++) {
                b2b_honest_matmul_neon(Yb, X, Wp_b, M, K, N);
            }
            t1 = monotonic_ms();
            double ms_b = t1 - t0;

            cache_flush(flush_buf, FLUSH_SIZE);
            /* Path B' — B2-B with skip. Capture skip count from one rep. */
            b2b_skip_matmul_neon(Yk, X, Wp_b, M, K, N, &skip_count);
            t0 = monotonic_ms();
            for (int r = 0; r < cfg->reps; r++) {
                b2b_skip_matmul_neon(Yk, X, Wp_b, M, K, N, NULL);
            }
            t1 = monotonic_ms();
            double ms_skip = t1 - t0;

            cache_flush(flush_buf, FLUSH_SIZE);
            /* Path C — B2-B optimal (unified TBL). */
            t0 = monotonic_ms();
            for (int r = 0; r < cfg->reps; r++) {
                b2b_optimal_matmul_neon(Yo, X, Wp_b, M, K, N);
            }
            t1 = monotonic_ms();
            double ms_optimal = t1 - t0;

            cache_flush(flush_buf, FLUSH_SIZE);
            /* Substrate cross-check (external grounding). */
            t0 = monotonic_ms();
            for (int r = 0; r < cfg->reps; r++) {
                /* m4t_ternary_dot_matmul_bt takes UNPACKED ternary X and W.
                 * X stride is K (already unpacked); W stride is K (also
                 * unpacked here — substrate kernel doesn't take packed). */
                m4t_ternary_dot_matmul_bt(Ys, X, W, M, K, N);
            }
            t1 = monotonic_ms();
            double ms_substrate = t1 - t0;

            cache_flush(flush_buf, FLUSH_SIZE);
            /* Path D — base-3 5-in-8 (sub-2-bit) packed. */
            t0 = monotonic_ms();
            for (int r = 0; r < cfg->reps; r++) {
                base3_5in8_matmul_neon(Yd, X, Wp_d, M, K, N);
            }
            t1 = monotonic_ms();
            double ms_5in8 = t1 - t0;

            cache_flush(flush_buf, FLUSH_SIZE);
            /* Path E — Concern-2 L2: packed X + packed W (both 4-in-8 base-3). */
            t0 = monotonic_ms();
            for (int r = 0; r < cfg->reps; r++) {
                path_e_packed_x_matmul_neon(Ye, Xp_e, Wp_a, M, K, N);
            }
            t1 = monotonic_ms();
            double ms_e = t1 - t0;

            cache_flush(flush_buf, FLUSH_SIZE);
            /* Path F — Concern-2 L2 companion: B2-B packed X + W (both 4-in-8). */
            t0 = monotonic_ms();
            for (int r = 0; r < cfg->reps; r++) {
                path_f_packed_x_b2b_matmul_neon(Yf, Xp_f, Wp_b, M, K, N);
            }
            t1 = monotonic_ms();
            double ms_f = t1 - t0;

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
            int ok_a_eq_e = (memcmp(Ya, Ye, (size_t)M * N * sizeof(int32_t)) == 0) ? 1 : 0;
            int ok_a_eq_f = (memcmp(Ya, Yf, (size_t)M * N * sizeof(int32_t)) == 0) ? 1 : 0;
            /* Track but don't add a CSV column for Path F (kept terse). */
            (void)ms_f;
            if (!ok_a_eq_f) {
                fprintf(stderr, "[FAIL] Path F mismatch at cfg %d seed %u\n", c, seed);
                return 1;
            }

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
            sum_ms_e += ms_e;
            sum_ok_a_eq_b += ok_a_eq_b;
            sum_ok_a_eq_skip += ok_a_eq_skip;
            sum_ok_a_eq_optimal += ok_a_eq_optimal;
            sum_ok_a_eq_substrate += ok_a_eq_substrate;
            sum_ok_a_eq_5in8 += ok_a_eq_5in8;
            sum_ok_a_eq_e += ok_a_eq_e;
            seeds_ms_a[s] = ms_a;
            seeds_ms_b[s] = ms_b;
            seeds_ms_skip[s] = ms_skip;
            seeds_ms_opt[s] = ms_optimal;
            seeds_ms_sub[s] = ms_substrate;
            seeds_ms_5in8[s] = ms_5in8;
            seeds_ms_e[s] = ms_e;

            printf("%d,%d,%d,%.2f,%.2f,%u,%d,%d,%d,%d,%d,%d,%.6f,"
                   "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                c, K, N, cfg->w_zero, cfg->a_zero, seed,
                ok_a_eq_b, ok_a_eq_skip, ok_a_eq_optimal, ok_a_eq_substrate, ok_a_eq_5in8, ok_a_eq_e,
                skip_rate,
                ms_a, ms_b, ms_skip, ms_optimal, ms_substrate, ms_5in8, ms_e);

            total_runs++;
            total_a_eq_b += ok_a_eq_b;
            total_a_eq_skip += ok_a_eq_skip;
            total_a_eq_optimal += ok_a_eq_optimal;
            total_a_eq_substrate += ok_a_eq_substrate;
            total_a_eq_5in8 += ok_a_eq_5in8;
            total_a_eq_e += ok_a_eq_e;
            total_skip_blocks += skip_count;
            total_blocks += total_blocks_per_run;

            free(X); free(W); free(Wp_a); free(Wp_b); free(Wp_d); free(Xp_e); free(Xp_f);
            free(Ya); free(Yb); free(Yk); free(Yo); free(Yd); free(Ye); free(Yf); free(Ys);
        }

        double cfg_skip_rate = (double)cfg_skip_blocks / (double)cfg_blocks;
        double mean_a = sum_ms_a / N_SEEDS;
        double mean_d = sum_ms_5in8 / N_SEEDS;
        double sd_a   = stddev_arr(seeds_ms_a, N_SEEDS, mean_a);
        double sd_d   = stddev_arr(seeds_ms_5in8, N_SEEDS, mean_d);
        fprintf(stderr,
            "[summary] cfg %d K=%d N=%d w_z=%.2f a_z=%.2f | "
            "verify(/5): a==b:%d skip:%d opt:%d sub:%d 5in8:%d e:%d | "
            "skip_rate=%.6f | "
            "ms (mean ± sd over %d reps × %d seeds): "
            "A=%.2f±%.2f B=%.2f Bskip=%.2f Bopt=%.2f Sub=%.2f 5in8=%.2f±%.2f E=%.2f | "
            "B/A=%.2fx Bskip/A=%.2fx Bopt/A=%.2fx Sub/A=%.2fx 5in8/A=%.2fx E/A=%.2fx\n",
            c, K, N, cfg->w_zero, cfg->a_zero,
            sum_ok_a_eq_b, sum_ok_a_eq_skip, sum_ok_a_eq_optimal,
            sum_ok_a_eq_substrate, sum_ok_a_eq_5in8, sum_ok_a_eq_e,
            cfg_skip_rate,
            cfg->reps, N_SEEDS,
            mean_a, sd_a,
            sum_ms_b / N_SEEDS, sum_ms_skip / N_SEEDS,
            sum_ms_optimal / N_SEEDS, sum_ms_substrate / N_SEEDS,
            mean_d, sd_d,
            sum_ms_e / N_SEEDS,
            sum_ms_b / sum_ms_a, sum_ms_skip / sum_ms_a,
            sum_ms_optimal / sum_ms_a, sum_ms_substrate / sum_ms_a,
            sum_ms_5in8 / sum_ms_a,
            sum_ms_e / sum_ms_a);
    }

    fprintf(stderr,
        "[overall] %d runs | "
        "verify a==b:%d/%d skip:%d/%d opt:%d/%d sub:%d/%d 5in8:%d/%d e:%d/%d | "
        "skip rate: %lld/%lld = %.6f\n",
        total_runs,
        total_a_eq_b, total_runs, total_a_eq_skip, total_runs,
        total_a_eq_optimal, total_runs, total_a_eq_substrate, total_runs,
        total_a_eq_5in8, total_runs,
        total_a_eq_e, total_runs,
        total_skip_blocks, total_blocks,
        (double)total_skip_blocks / (double)total_blocks);

    int all_pass = (total_a_eq_b == total_runs)
                && (total_a_eq_skip == total_runs)
                && (total_a_eq_optimal == total_runs)
                && (total_a_eq_substrate == total_runs)
                && (total_a_eq_5in8 == total_runs)
                && (total_a_eq_e == total_runs);

    free(flush_buf);

    if (!all_pass) {
        fprintf(stderr, "[FAIL] verification failed — kernels disagree\n");
        return 1;
    }
    fprintf(stderr,
        "[PASS] all four audit kernels + substrate cross-check bit-exact equivalent\n");
    return 0;
}
