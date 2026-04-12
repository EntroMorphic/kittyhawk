/*
 * m4t_bench.c — per-opcode cycle-count harness for M4T.
 *
 * Runs each opcode 10^6 times in a tight hot loop, measures via
 * mach_absolute_time, reports mean and p99 cycles per call and per element.
 *
 * Contract gating clause: p99 <= 1.5 * mean. Anything wider means a
 * cold-cache path is leaking through.
 *
 * Build: linked against libm4t.a (see CMakeLists.txt)
 * Run:   ./m4t_bench
 */

#include "m4t_types.h"
#include "m4t_mtfp.h"
#include "m4t_trit_pack.h"
#include "m4t_ternary_matmul.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

/* ── Timing helpers ────────────────────────────────────────────────────── */

static double ticks_to_ns;

static void timing_init(void) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    ticks_to_ns = (double)info.numer / (double)info.denom;
}

static inline uint64_t now_ticks(void) {
    return mach_absolute_time();
}

/* Assumed clock rate in GHz. M4 P-core boost is ~4.4 GHz; E-cores run
 * at ~2.7 GHz. If this bench runs on an E-core, cycle estimates will be
 * inflated by ~60%. Pin to a P-core via `taskpolicy -b deny` or check
 * results against known instruction counts. */
#define M4T_BENCH_CLOCK_GHZ 4.4

static double ticks_to_cycles(uint64_t ticks) {
    double ns = (double)ticks * ticks_to_ns;
    return ns * M4T_BENCH_CLOCK_GHZ;
}

/* ── Bench infrastructure ──────────────────────────────────────────────── */

#define WARMUP_ITERS   1000
#define BENCH_ITERS    1000000
#define SAMPLE_BATCHES 100
#define ITERS_PER_BATCH (BENCH_ITERS / SAMPLE_BATCHES)

static double samples[SAMPLE_BATCHES];

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

typedef struct {
    double mean_cycles;
    double p99_cycles;
    double cycles_per_elem;
    int pass;  /* p99 <= 1.5 * mean */
} bench_result_t;

static void report(const char* name, int n_elems __attribute__((unused)), bench_result_t r) {
    printf("  %-30s  mean=%8.1f cy  p99=%8.1f cy  cy/elem=%6.2f  %s\n",
           name, r.mean_cycles, r.p99_cycles, r.cycles_per_elem,
           r.pass ? "PASS" : "FAIL (p99 > 1.5*mean)");
}

/* ── Benchmarks ────────────────────────────────────────────────────────── */

static bench_result_t bench_vec_add(int n) {
    m4t_mtfp_t* a = calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* b = calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* c = calloc((size_t)n, sizeof(m4t_mtfp_t));
    for (int i = 0; i < n; i++) { a[i] = (m4t_mtfp_t)(i % 1000); b[i] = (m4t_mtfp_t)(i % 500); }

    for (int i = 0; i < WARMUP_ITERS; i++) m4t_mtfp_vec_add(c, a, b, n);

    for (int s = 0; s < SAMPLE_BATCHES; s++) {
        uint64_t t0 = now_ticks();
        for (int i = 0; i < ITERS_PER_BATCH; i++) m4t_mtfp_vec_add(c, a, b, n);
        uint64_t t1 = now_ticks();
        samples[s] = ticks_to_cycles(t1 - t0) / ITERS_PER_BATCH;
    }

    qsort(samples, SAMPLE_BATCHES, sizeof(double), cmp_double);
    bench_result_t r;
    double sum = 0; for (int i = 0; i < SAMPLE_BATCHES; i++) sum += samples[i];
    r.mean_cycles = sum / SAMPLE_BATCHES;
    r.p99_cycles = samples[(int)(SAMPLE_BATCHES * 0.99)];
    r.cycles_per_elem = r.mean_cycles / n;
    r.pass = (r.p99_cycles <= 1.5 * r.mean_cycles);

    free(a); free(b); free(c);
    return r;
}

static bench_result_t bench_matmul_bt(int M, int K, int N) {
    m4t_mtfp_t* X = calloc((size_t)M * K, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* W = calloc((size_t)N * K, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* Y = calloc((size_t)M * N, sizeof(m4t_mtfp_t));
    for (int i = 0; i < M * K; i++) X[i] = (m4t_mtfp_t)(i % 100);
    for (int i = 0; i < N * K; i++) W[i] = (m4t_mtfp_t)(i % 50);

    int iters = BENCH_ITERS / 100;  /* matmul is heavier; fewer iters */
    int batches = SAMPLE_BATCHES;
    int per_batch = iters / batches;
    if (per_batch < 1) per_batch = 1;

    for (int i = 0; i < 10; i++) m4t_mtfp_matmul_bt(Y, X, W, M, K, N);

    for (int s = 0; s < batches; s++) {
        uint64_t t0 = now_ticks();
        for (int i = 0; i < per_batch; i++) m4t_mtfp_matmul_bt(Y, X, W, M, K, N);
        uint64_t t1 = now_ticks();
        samples[s] = ticks_to_cycles(t1 - t0) / per_batch;
    }

    qsort(samples, (size_t)batches, sizeof(double), cmp_double);
    bench_result_t r;
    double sum = 0; for (int i = 0; i < batches; i++) sum += samples[i];
    r.mean_cycles = sum / batches;
    r.p99_cycles = samples[(int)(batches * 0.99)];
    r.cycles_per_elem = r.mean_cycles / (M * N);
    r.pass = (r.p99_cycles <= 1.5 * r.mean_cycles);

    free(X); free(W); free(Y);
    return r;
}

static bench_result_t bench_ternary_matmul_bt(int M, int K, int N) {
    m4t_mtfp_t* X = calloc((size_t)M * K, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* Y = calloc((size_t)M * N, sizeof(m4t_mtfp_t));
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    uint8_t* W = calloc((size_t)N * Kp, 1);
    for (int i = 0; i < M * K; i++) X[i] = (m4t_mtfp_t)(i % 100);
    for (int i = 0; i < N * Kp; i++) W[i] = (uint8_t)(0x55);  /* all +1 trits */

    int iters = BENCH_ITERS / 100;
    int batches = SAMPLE_BATCHES;
    int per_batch = iters / batches;
    if (per_batch < 1) per_batch = 1;

    for (int i = 0; i < 10; i++) m4t_mtfp_ternary_matmul_bt(Y, X, W, M, K, N);

    for (int s = 0; s < batches; s++) {
        uint64_t t0 = now_ticks();
        for (int i = 0; i < per_batch; i++) m4t_mtfp_ternary_matmul_bt(Y, X, W, M, K, N);
        uint64_t t1 = now_ticks();
        samples[s] = ticks_to_cycles(t1 - t0) / per_batch;
    }

    qsort(samples, (size_t)batches, sizeof(double), cmp_double);
    bench_result_t r;
    double sum = 0; for (int i = 0; i < batches; i++) sum += samples[i];
    r.mean_cycles = sum / batches;
    r.p99_cycles = samples[(int)(batches * 0.99)];
    r.cycles_per_elem = r.mean_cycles / (M * N);
    r.pass = (r.p99_cycles <= 1.5 * r.mean_cycles);

    free(X); free(W); free(Y);
    return r;
}

static bench_result_t bench_popcount_dist(int n_trits) {
    int nb = M4T_TRIT_PACKED_BYTES(n_trits);
    uint8_t* a = calloc((size_t)nb, 1);
    uint8_t* b = calloc((size_t)nb, 1);
    uint8_t* mask = malloc((size_t)nb);
    memset(mask, 0xFF, (size_t)nb);
    for (int i = 0; i < nb; i++) { a[i] = (uint8_t)(i & 0xFF); b[i] = (uint8_t)((i * 7) & 0xFF); }

    volatile int32_t sink = 0;
    for (int i = 0; i < WARMUP_ITERS; i++) sink = m4t_popcount_dist(a, b, mask, nb);

    for (int s = 0; s < SAMPLE_BATCHES; s++) {
        uint64_t t0 = now_ticks();
        for (int i = 0; i < ITERS_PER_BATCH; i++) sink = m4t_popcount_dist(a, b, mask, nb);
        uint64_t t1 = now_ticks();
        samples[s] = ticks_to_cycles(t1 - t0) / ITERS_PER_BATCH;
    }

    qsort(samples, SAMPLE_BATCHES, sizeof(double), cmp_double);
    bench_result_t r;
    double sum = 0; for (int i = 0; i < SAMPLE_BATCHES; i++) sum += samples[i];
    r.mean_cycles = sum / SAMPLE_BATCHES;
    r.p99_cycles = samples[(int)(SAMPLE_BATCHES * 0.99)];
    r.cycles_per_elem = r.mean_cycles / n_trits;
    r.pass = (r.p99_cycles <= 1.5 * r.mean_cycles);

    free(a); free(b); free(mask);
    (void)sink;
    return r;
}

static bench_result_t bench_layernorm(int rows, int cols) {
    m4t_mtfp_t* x = calloc((size_t)rows * cols, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* y = calloc((size_t)rows * cols, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* w = calloc((size_t)cols, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* b = calloc((size_t)cols, sizeof(m4t_mtfp_t));
    for (int i = 0; i < rows * cols; i++) x[i] = (m4t_mtfp_t)(i % 1000 - 500);
    for (int i = 0; i < cols; i++) { w[i] = M4T_MTFP_SCALE; b[i] = 0; }

    int iters = BENCH_ITERS / 100;
    int batches = SAMPLE_BATCHES;
    int per_batch = iters / batches;
    if (per_batch < 1) per_batch = 1;

    for (int i = 0; i < 10; i++) m4t_mtfp_layernorm(y, x, w, b, 1, rows, cols);

    for (int s = 0; s < batches; s++) {
        uint64_t t0 = now_ticks();
        for (int i = 0; i < per_batch; i++) m4t_mtfp_layernorm(y, x, w, b, 1, rows, cols);
        uint64_t t1 = now_ticks();
        samples[s] = ticks_to_cycles(t1 - t0) / per_batch;
    }

    qsort(samples, (size_t)batches, sizeof(double), cmp_double);
    bench_result_t r;
    double sum = 0; for (int i = 0; i < batches; i++) sum += samples[i];
    r.mean_cycles = sum / batches;
    r.p99_cycles = samples[(int)(batches * 0.99)];
    r.cycles_per_elem = r.mean_cycles / (rows * cols);
    r.pass = (r.p99_cycles <= 1.5 * r.mean_cycles);

    free(x); free(y); free(w); free(b);
    return r;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    timing_init();

    printf("M4T Opcode Benchmark\n");
    printf("====================\n\n");

    printf("Vector ops (n=256):\n");
    report("m4t_mtfp_vec_add", 256, bench_vec_add(256));

    printf("\nVector ops (n=1024):\n");
    report("m4t_mtfp_vec_add", 1024, bench_vec_add(1024));

    printf("\nPopcount distance:\n");
    report("popcount_dist (64 trits)", 64, bench_popcount_dist(64));
    report("popcount_dist (256 trits)", 256, bench_popcount_dist(256));

    printf("\nDense MTFP matmul_bt:\n");
    report("matmul_bt (1x64x64)", 64, bench_matmul_bt(1, 64, 64));
    report("matmul_bt (4x64x64)", 256, bench_matmul_bt(4, 64, 64));

    printf("\nTernary matmul_bt (MTFP x packed trits):\n");
    report("ternary_matmul_bt (1x64x8)", 8, bench_ternary_matmul_bt(1, 64, 8));
    report("ternary_matmul_bt (1x128x8)", 8, bench_ternary_matmul_bt(1, 128, 8));
    report("ternary_matmul_bt (4x64x8)", 32, bench_ternary_matmul_bt(4, 64, 8));

    printf("\nLayerNorm:\n");
    report("layernorm (1x64)", 64, bench_layernorm(1, 64));
    report("layernorm (1x256)", 256, bench_layernorm(1, 256));

    printf("\n");
    return 0;
}
