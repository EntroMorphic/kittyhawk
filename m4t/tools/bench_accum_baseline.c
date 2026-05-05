/*
 * bench_accum_baseline.c — pre-cycle baseline measurement of
 * m4t_mtfp_vec_accum_aligning_scalar_ref. Per A-G2 of the
 * cross-exp accum routing cycle. INFORMATIONAL only — does NOT gate
 * the cycle (per user directive: function over speed).
 *
 * Build: cc -O3 -mcpu=native -I m4t/src m4t/tools/bench_accum_baseline.c \
 *           build/m4t/libm4t.a -o /tmp/bench_accum_baseline
 * Run:   /tmp/bench_accum_baseline
 *
 * Applies CONTRIBUTING throughput-microbench discipline:
 *   - Distinct inputs per call (heap pool with non-constant addressing)
 *   - Min-of-5 sampling
 *   - Workload shape declared in output
 */

#include "m4t_mtfp.h"
#include "m4t_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_VAL 581130733

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

static double bench_one(int n, int delta, int with_flags, int iters) {
    /* Allocate per-iter scratch to defeat constant-folding. */
    m4t_mtfp_t* running_pool = malloc(sizeof(m4t_mtfp_t) * (size_t)n * (size_t)iters);
    m4t_mtfp_t* addend       = malloc(sizeof(m4t_mtfp_t) * (size_t)n);
    uint8_t* flags = with_flags ? calloc(M4T_FLAG_BYTES(n), 1) : NULL;
    if (!running_pool || !addend) { perror("malloc"); exit(1); }

    /* Seed with pid-derived random data. */
    srand(42);
    for (size_t i = 0; i < (size_t)n * iters; i++) {
        running_pool[i] = (m4t_mtfp_t)(rand() % (2 * MAX_VAL + 1) - MAX_VAL);
    }
    for (int i = 0; i < n; i++) {
        addend[i] = (m4t_mtfp_t)(rand() % (2 * MAX_VAL + 1) - MAX_VAL);
    }
    int8_t addend_exp = 5;
    int8_t base_running_exp = (int8_t)(addend_exp - delta);

    /* Warmup. */
    for (int w = 0; w < 5; w++) {
        int8_t e = base_running_exp;
        m4t_mtfp_vec_accum_aligning_scalar_ref(
            running_pool, &e, addend, addend_exp, flags, n);
    }

    /* Min-of-5. */
    double best_ns = 1e18;
    for (int s = 0; s < 5; s++) {
        double t0 = now_ns();
        for (int it = 0; it < iters; it++) {
            int8_t e = base_running_exp;  /* reset each iter */
            m4t_mtfp_vec_accum_aligning_scalar_ref(
                running_pool + (size_t)it * n, &e, addend, addend_exp, flags, n);
        }
        double t1 = now_ns();
        double ns = t1 - t0;
        if (ns < best_ns) best_ns = ns;
    }
    free(running_pool); free(addend); free(flags);
    return best_ns / ((double)iters * (double)n);  /* ns per cell */
}

int main(void) {
    printf("=== Baseline: m4t_mtfp_vec_accum_aligning_scalar_ref ===\n");
    printf("Pre-cycle scalar perf, INFORMATIONAL only (does not gate cycle).\n\n");

    struct { int n; int delta; int with_flags; int iters; const char* label; } cases[] = {
        { 64,    1, 0, 1000, "n=64    delta=1    no-flags" },
        { 64,    1, 1, 1000, "n=64    delta=1    with-flags" },
        { 64,   10, 1, 1000, "n=64    delta=10   with-flags" },
        { 64,   19, 1, 1000, "n=64    delta=19   with-flags" },
        { 4096,  5, 1,   30, "n=4096  delta=5    with-flags" },
        { 16,    5, 1, 4000, "n=16    delta=5    with-flags (per-call overhead bound)" },
    };
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); i++) {
        double ns = bench_one(cases[i].n, cases[i].delta, cases[i].with_flags, cases[i].iters);
        printf("  %-50s  %.2f ns/cell  (~%.1f cycles/cell @3.5GHz)\n",
               cases[i].label, ns, ns * 3.5);
    }
    return 0;
}
