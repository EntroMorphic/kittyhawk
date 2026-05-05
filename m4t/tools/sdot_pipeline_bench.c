/*
 * m4t/tools/sdot_pipeline_bench.c — characterize SDOT throughput on the
 *                                    target M-series hardware.
 *
 * Per project tools convention (see bench_vmlal_throughput.c):
 *   Build:  cc -O3 -mcpu=native m4t/tools/sdot_pipeline_bench.c -o /tmp/sdb
 *   Run:    /tmp/sdb
 *
 * Originally lived in audit/ as the direct mechanism test for the strong-
 * claim Path D vs Path A 1.8× wall-clock win (per
 * journal/p0_concern1_mechanism.md). Moved here 2026-05-05 as enduring
 * hardware-characterization infrastructure (per Item 3 of the production-
 * shoring plan; see journal/m4t_5in8_closeout.md and the prior journal/
 * p0_concern1_mechanism.md for context).
 *
 * Per concern raised in the strong-claim post-P0 review: the 1.8× wall-clock
 * advantage was ATTRIBUTED to "SDOT amortization" (Path D dispatches SDOTs
 * at ~0.82/cycle vs Path A's ~0.46/cycle on M-series, inferred from total
 * SDOT count / wall-clock). The hypothesis fits but was never directly
 * measured against the SDOT throughput ceiling.
 *
 * This bench measures SDOT throughput in three controlled scenarios:
 *
 *   T1 — Pure SDOT, single acc chain (latency-bound).
 *        Each SDOT depends on the prior. Should dispatch at 1 SDOT every
 *        L cycles, where L is SDOT latency. On M-series, SDOT has ~3-4
 *        cycle latency → 0.25-0.33 SDOTs/cycle.
 *
 *   T2 — Pure SDOT, 4 independent acc chains (throughput-bound).
 *        4 chains in parallel saturate the SDOT pipeline. Should
 *        approach M-series SDOT throughput limit (~1-4 SDOTs/cycle
 *        depending on uarch).
 *
 *   T3 — Pure SDOT, 8 independent acc chains.
 *        Tests if N=4 saturates throughput or if more chains help.
 *
 * Comparing the measured pure-SDOT ceiling to Path A's and Path D's
 * actual SDOT dispatch rates tells us how close each kernel is to peak.
 *
 * If Path A is far from peak AND Path D is closer, the SDOT-amortization
 * mechanism is empirically supported (less non-SDOT work → more SDOT
 * dispatch slots used). If both are equally far from peak, the mechanism
 * story is incomplete and we'd need to look elsewhere.
 *
 * NEON only. K%16==0 enforced. Simple wall-clock + SDOT count → throughput.
 */

#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* T1: single acc chain. SDOT depends on prior SDOT. Latency-bound. */
__attribute__((noinline))
static int32_t bench_t1_single_chain(const int8_t* x, const int8_t* w, int n_iters) {
    int8x16_t xv = vld1q_s8(x);
    int8x16_t wv = vld1q_s8(w);
    int32x4_t acc = vdupq_n_s32(0);
    for (int i = 0; i < n_iters; i++) {
        acc = vdotq_s32(acc, xv, wv);
    }
    return vaddvq_s32(acc);
}

/* T2: 4 independent acc chains. Should saturate SDOT throughput. */
__attribute__((noinline))
static int32_t bench_t2_four_chains(const int8_t* x, const int8_t* w, int n_iters) {
    int8x16_t xv = vld1q_s8(x);
    int8x16_t wv = vld1q_s8(w);
    int32x4_t a0 = vdupq_n_s32(0);
    int32x4_t a1 = vdupq_n_s32(0);
    int32x4_t a2 = vdupq_n_s32(0);
    int32x4_t a3 = vdupq_n_s32(0);
    for (int i = 0; i < n_iters; i += 4) {
        a0 = vdotq_s32(a0, xv, wv);
        a1 = vdotq_s32(a1, xv, wv);
        a2 = vdotq_s32(a2, xv, wv);
        a3 = vdotq_s32(a3, xv, wv);
    }
    return vaddvq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)));
}

/* T3: 8 independent acc chains. Tests if 4 saturates or if more helps. */
__attribute__((noinline))
static int32_t bench_t3_eight_chains(const int8_t* x, const int8_t* w, int n_iters) {
    int8x16_t xv = vld1q_s8(x);
    int8x16_t wv = vld1q_s8(w);
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    int32x4_t a4 = vdupq_n_s32(0), a5 = vdupq_n_s32(0);
    int32x4_t a6 = vdupq_n_s32(0), a7 = vdupq_n_s32(0);
    for (int i = 0; i < n_iters; i += 8) {
        a0 = vdotq_s32(a0, xv, wv); a1 = vdotq_s32(a1, xv, wv);
        a2 = vdotq_s32(a2, xv, wv); a3 = vdotq_s32(a3, xv, wv);
        a4 = vdotq_s32(a4, xv, wv); a5 = vdotq_s32(a5, xv, wv);
        a6 = vdotq_s32(a6, xv, wv); a7 = vdotq_s32(a7, xv, wv);
    }
    int32x4_t s01 = vaddq_s32(a0, a1);
    int32x4_t s23 = vaddq_s32(a2, a3);
    int32x4_t s45 = vaddq_s32(a4, a5);
    int32x4_t s67 = vaddq_s32(a6, a7);
    return vaddvq_s32(vaddq_s32(vaddq_s32(s01, s23), vaddq_s32(s45, s67)));
}

int main(void) {
    /* Set up inputs (just non-zero data; specifics don't matter for throughput). */
    int8_t x[16], w[16];
    for (int i = 0; i < 16; i++) {
        x[i] = (int8_t)(i % 3 - 1);
        w[i] = (int8_t)((i + 1) % 3 - 1);
    }

#define N_ITERS 8000000  /* 8M SDOTs per test */
#define N_REPS 5

    printf("test,iters_per_run,mean_ns,sdots_per_ns,sdots_per_cycle_at_3GHz\n");

    /* Warm up. */
    (void)bench_t2_four_chains(x, w, 1024);

    struct {
        const char* name;
        int32_t (*fn)(const int8_t*, const int8_t*, int);
    } tests[] = {
        { "T1_single_chain",   bench_t1_single_chain },
        { "T2_four_chains",    bench_t2_four_chains  },
        { "T3_eight_chains",   bench_t3_eight_chains },
    };
    int n_tests = sizeof(tests) / sizeof(tests[0]);

    for (int t = 0; t < n_tests; t++) {
        double samples[N_REPS];
        int32_t sink = 0;
        for (int r = 0; r < N_REPS; r++) {
            double t0 = monotonic_ns();
            sink ^= tests[t].fn(x, w, N_ITERS);
            double t1 = monotonic_ns();
            samples[r] = t1 - t0;
        }
        /* Use min-of-N for thermal/scheduling noise resistance. */
        double min_ns = samples[0];
        for (int r = 1; r < N_REPS; r++) {
            if (samples[r] < min_ns) min_ns = samples[r];
        }
        double sdots_per_ns = (double)N_ITERS / min_ns;
        /* Convert to SDOTs/cycle at assumed 3 GHz P-core frequency. */
        double sdots_per_cycle = sdots_per_ns / 3.0;
        printf("%s,%d,%.3f,%.4f,%.4f\n",
            tests[t].name, N_ITERS, min_ns, sdots_per_ns, sdots_per_cycle);
        /* Defeat dead code elimination */
        if (sink == (int32_t)0xCAFEBABE) printf("# anti-DCE\n");
    }

    /* Path A and Path D measured rates from strong-claim bench (K=51200,
     * apples-to-apples both tiled). Print them for direct comparison. */
    fprintf(stderr, "\n# Reference: measured SDOT dispatch rate in production kernels\n");
    fprintf(stderr, "# (from audit/strong_results.csv at K=51200, all kernels tiled):\n");
    fprintf(stderr, "#   Path A:  0.46 SDOTs/cycle  (8*64*51200/16 SDOTs in 29.85ms at 3GHz)\n");
    fprintf(stderr, "#   Path D:  0.82 SDOTs/cycle  (same SDOT count in 16.60ms at 3GHz)\n");
    fprintf(stderr, "# If T2/T3 ceiling is X SDOTs/cycle:\n");
    fprintf(stderr, "#   Path A is at (0.46/X)*100%% of peak.\n");
    fprintf(stderr, "#   Path D is at (0.82/X)*100%% of peak.\n");

    return 0;
}
