/*
 * bench_m4t_lto.c — V4-residual-3 microbench targeting LTO cross-TU
 * inlining specifically. Two workload variants — same target function,
 * different data-dependency structure — to surface LTO's effect (or
 * lack thereof) under different bottlenecks.
 *
 * Background: V4-G5 measured LTO impact on bench_m4t_tier2_perf and
 * found no delta. That tells us LTO doesn't help on workloads that are
 * already aggressively per-TU-optimized at -O3 -mcpu=native — but it
 * does NOT tell us whether LTO is doing useful work anywhere.
 *
 * Target function: m4t_mtfp_block_add. Why:
 *   - Small body (~6 NEON instructions). Well below LTO inlining limits.
 *   - Cross-TU: declared in m4t_mtfp.h, defined in m4t_mtfp.c. The
 *     bench's TU sees only the declaration unless LTO inlines.
 *   - NOT static inline — call goes through `bl` unless LTO acts.
 *
 * Two workload variants:
 *
 *   (A) Carry-dependent: each iter's dst depends on the previous iter's
 *       dst (single buffer reused). Tests whether LTO helps when the
 *       inner-loop bottleneck is the data dependency.
 *
 *   (B) Pipelined: round-robin across N independent dst buffers, so
 *       consecutive iters have no data dependency. Exposes call
 *       overhead as the bottleneck — the regime where LTO inlining
 *       should shine.
 *
 * The two variants together form a 2x2 with LTO-on/LTO-off (built by
 * configuring with -DGESH_LTO=ON or =OFF). Reading the matrix:
 *
 *   variant A LTO ≈ variant A no-LTO  → workload is data-dep bound;
 *                                       LTO has nothing to fix.
 *   variant B LTO < variant B no-LTO  → LTO is doing useful work;
 *                                       it just happened not to surface
 *                                       on (A)-shaped workloads.
 *
 * The substrate's real consumers are mostly (A)-shaped (accumulating
 * into a state). That's why the V4-G5 perf bench showed no LTO delta.
 */

#include "m4t_mtfp.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WARMUP_ITERS 200000
#define BENCH_ITERS  10000000   /* 10M */
#define N_INDEP      64         /* independent dsts for variant B */

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

/* Variant A: carry-dependent. Single dst, accumulated. */
static double bench_carry_dep(void) {
    m4t_mtfp_t dst[M4T_MTFP_CELLS_PER_BLOCK] = {0};
    m4t_mtfp_t a[M4T_MTFP_CELLS_PER_BLOCK]   = {1, 0, 0, 0};

    for (int i = 0; i < WARMUP_ITERS; i++) m4t_mtfp_block_add(dst, a);
    dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0;

    double t0 = now_ns();
    for (int i = 0; i < BENCH_ITERS; i++) m4t_mtfp_block_add(dst, a);
    double t1 = now_ns();

    /* Sink to prevent dead-store elimination. */
    volatile int32_t sink = (int32_t)dst[0];
    (void)sink;
    if ((int32_t)dst[0] != BENCH_ITERS) {
        fprintf(stderr, "FAIL variant A: dst[0] = %d, expected %d\n",
                (int)dst[0], BENCH_ITERS);
        return -1.0;
    }
    return (t1 - t0) / (double)BENCH_ITERS;
}

/* Variant B: pipelined. N_INDEP independent dsts, round-robin. */
static double bench_pipelined(void) {
    static m4t_mtfp_t dst[N_INDEP][M4T_MTFP_CELLS_PER_BLOCK];
    m4t_mtfp_t a[M4T_MTFP_CELLS_PER_BLOCK] = {1, 0, 0, 0};
    for (int i = 0; i < N_INDEP; i++)
        for (int j = 0; j < M4T_MTFP_CELLS_PER_BLOCK; j++) dst[i][j] = 0;

    for (int i = 0; i < WARMUP_ITERS; i++)
        m4t_mtfp_block_add(dst[i % N_INDEP], a);
    for (int i = 0; i < N_INDEP; i++)
        for (int j = 0; j < M4T_MTFP_CELLS_PER_BLOCK; j++) dst[i][j] = 0;

    double t0 = now_ns();
    for (int i = 0; i < BENCH_ITERS; i++)
        m4t_mtfp_block_add(dst[i % N_INDEP], a);
    double t1 = now_ns();

    /* Verify: total adds across all dsts equals iters. */
    int64_t total = 0;
    for (int i = 0; i < N_INDEP; i++) total += dst[i][0];
    if (total != BENCH_ITERS) {
        fprintf(stderr, "FAIL variant B: total = %lld, expected %d\n",
                (long long)total, BENCH_ITERS);
        return -1.0;
    }
    return (t1 - t0) / (double)BENCH_ITERS;
}

static void report(const char* label, double ns_per_call) {
    if (ns_per_call < 0) { puts("FAIL"); return; }
    double cycles = ns_per_call * 3.5;   /* ~3.5 GHz on M-series; display only */
    printf("  %-40s ns/call=%.3f cycles=%.1f\n", label, ns_per_call, cycles);
}

int main(void) {
    printf("bench_m4t_lto: m4t_mtfp_block_add x %d iters per variant\n",
           BENCH_ITERS);
    /* Three runs, take the min of each variant. Filters CPU-frequency
     * scaling and other noise. */
    double a_min = 1e18, b_min = 1e18;
    for (int i = 0; i < 3; i++) {
        double a = bench_carry_dep();
        double b = bench_pipelined();
        if (a >= 0 && a < a_min) a_min = a;
        if (b >= 0 && b < b_min) b_min = b;
    }
    report("variant A (carry-dependent)", a_min);
    report("variant B (pipelined, N_INDEP=64)", b_min);
    if (a_min < 0 || b_min < 0) return 1;
    return 0;
}
