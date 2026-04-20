/*
 * m4t_profile.c — microbenchmark for M4T hot-path primitives.
 *
 * Measures throughput (ops/sec, ns/op) of the three hardware-anchored
 * instruction classes that THESIS.md §5 claims:
 *
 *   1. SDOT (vdotq_s32)  — m4t_mtfp4_sdot_matmul_bt
 *   2. TBL  (vqtbl1q_s8) — m4t_trit_mul (representative trit op)
 *   3. VCNT (vcntq_u8)   — m4t_popcount_dist
 *
 * Each benchmark runs a tight loop of N_ITER calls, measures wall time
 * via mach_absolute_time (nanosecond resolution on Apple Silicon), and
 * reports throughput. Run on an idle system for clean numbers.
 *
 * Build:
 *   cmake -S m4t -B build-tools -DM4T_BUILD_TOOLS=ON
 *   cmake --build build-tools -j
 *   ./build-tools/m4t_profile
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 */

#include "m4t_mtfp4.h"
#include "m4t_trit_ops.h"
#include "m4t_trit_pack.h"
#include "m4t_trit_reducers.h"
#include "m4t_ternary_matmul.h"

#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double ns_per_tick(void) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    return (double)info.numer / (double)info.denom;
}

static void fill_random_trits(uint8_t* buf, int packed_bytes) {
    for (int i = 0; i < packed_bytes; i++)
        buf[i] = (uint8_t)(rand() & 0xFF);
}

static void fill_random_mtfp4(m4t_mtfp4_t* buf, int n) {
    for (int i = 0; i < n; i++)
        buf[i] = (m4t_mtfp4_t)((rand() % 81) - 40);
}

static void fill_random_trit_values(m4t_trit_t* buf, int n) {
    for (int i = 0; i < n; i++)
        buf[i] = (m4t_trit_t)((rand() % 3) - 1);
}

/* ── VCNT benchmark: popcount_dist ────────────────────────────────────── */

static void bench_popcount_dist(int n_proj, int n_iter, double ns_tick) {
    int pb = M4T_TRIT_PACKED_BYTES(n_proj);
    uint8_t* a = calloc(pb, 1);
    uint8_t* b = calloc(pb, 1);
    uint8_t* mask = malloc(pb);
    fill_random_trits(a, pb);
    fill_random_trits(b, pb);
    memset(mask, 0xFF, pb);

    volatile int32_t sink = 0;
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < n_iter; i++)
        sink += m4t_popcount_dist(a, b, mask, pb);
    uint64_t t1 = mach_absolute_time();
    (void)sink;

    double ns = (double)(t1 - t0) * ns_tick;
    double ns_per = ns / n_iter;
    double mops = (double)n_iter / (ns / 1e9) / 1e6;
    printf("  popcount_dist  N_PROJ=%-4d  %7.1f ns/op  %7.1f Mops/s  (%d bytes)\n",
           n_proj, ns_per, mops, pb);
    free(a); free(b); free(mask);
}

/* ── SDOT benchmark: mtfp4_sdot_matmul_bt ─────────────────────────────── */

static void bench_sdot_matmul(int M, int K, int N, int n_iter, double ns_tick) {
    m4t_mtfp4_t* X = malloc((size_t)M * K * sizeof(m4t_mtfp4_t));
    m4t_trit_t*  W = malloc((size_t)N * K * sizeof(m4t_trit_t));
    m4t_mtfp4_t* Y = malloc((size_t)M * N * sizeof(m4t_mtfp4_t));
    fill_random_mtfp4(X, M * K);
    fill_random_trit_values(W, N * K);
    memset(Y, 0, (size_t)M * N * sizeof(m4t_mtfp4_t));

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < n_iter; i++)
        m4t_mtfp4_sdot_matmul_bt(Y, X, W, M, K, N);
    uint64_t t1 = mach_absolute_time();

    double ns = (double)(t1 - t0) * ns_tick;
    double ns_per = ns / n_iter;
    long long ops_per = (long long)M * K * N;
    double gops = (double)ops_per * n_iter / (ns / 1e9) / 1e9;
    printf("  sdot_matmul    M=%-2d K=%-4d N=%-2d  %8.1f ns/call  %5.2f Gops/s  (%lld MACs/call)\n",
           M, K, N, ns_per, gops, ops_per);
    free(X); free(W); free(Y);
}

/* ── TBL benchmark: trit_mul ──────────────────────────────────────────── */

static void bench_trit_mul(int n_trits, int n_iter, double ns_tick) {
    int pb = M4T_TRIT_PACKED_BYTES(n_trits);
    uint8_t* a = malloc(pb);
    uint8_t* b = malloc(pb);
    uint8_t* dst = malloc(pb);
    fill_random_trits(a, pb);
    fill_random_trits(b, pb);

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < n_iter; i++)
        m4t_trit_mul(dst, a, b, n_trits);
    uint64_t t1 = mach_absolute_time();

    double ns = (double)(t1 - t0) * ns_tick;
    double ns_per = ns / n_iter;
    double mtrits = (double)n_trits * n_iter / (ns / 1e9) / 1e6;
    printf("  trit_mul       n_trits=%-5d  %7.1f ns/op  %7.1f Mtrits/s\n",
           n_trits, ns_per, mtrits);
    free(a); free(b); free(dst);
}

/* ── VCNT benchmark: trit_counts (masked-VCNT reduction) ─────────────── */

static void bench_trit_counts(int n_trits, int n_iter, double ns_tick) {
    int pb = M4T_TRIT_PACKED_BYTES(n_trits);
    uint8_t* packed = malloc(pb);
    fill_random_trits(packed, pb);
    volatile int64_t sink_p = 0, sink_n = 0;

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < n_iter; i++) {
        int64_t p, n;
        m4t_trit_counts(packed, n_trits, &p, &n);
        sink_p += p; sink_n += n;
    }
    uint64_t t1 = mach_absolute_time();
    (void)sink_p; (void)sink_n;

    double ns = (double)(t1 - t0) * ns_tick;
    double ns_per = ns / n_iter;
    double mtrits = (double)n_trits * n_iter / (ns / 1e9) / 1e6;
    printf("  trit_counts    n_trits=%-5d  %7.1f ns/op  %7.1f Mtrits/s\n",
           n_trits, ns_per, mtrits);
    free(packed);
}

/* ── MTFP19 ternary matmul benchmark ──────────────────────────────────── */

static void bench_ternary_matmul(int M, int K, int N, int n_iter, double ns_tick) {
    m4t_mtfp_t* X = malloc((size_t)M * K * sizeof(m4t_mtfp_t));
    int Kp = M4T_TRIT_PACKED_BYTES(K);
    uint8_t* W = malloc((size_t)N * Kp);
    m4t_mtfp_t* Y = malloc((size_t)M * N * sizeof(m4t_mtfp_t));
    for (int i = 0; i < M * K; i++) X[i] = (m4t_mtfp_t)(rand() % 59049);
    fill_random_trits(W, N * Kp);
    memset(Y, 0, (size_t)M * N * sizeof(m4t_mtfp_t));

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < n_iter; i++)
        m4t_mtfp_ternary_matmul_bt(Y, X, W, M, K, N);
    uint64_t t1 = mach_absolute_time();

    double ns = (double)(t1 - t0) * ns_tick;
    double ns_per = ns / n_iter;
    long long ops_per = (long long)M * K * N;
    double gops = (double)ops_per * n_iter / (ns / 1e9) / 1e9;
    printf("  ternary_matmul M=%-2d K=%-4d N=%-2d  %8.1f ns/call  %5.2f Gops/s  (%lld MACs/call)\n",
           M, K, N, ns_per, gops, ops_per);
    free(X); free(W); free(Y);
}

int main(void) {
    double ns_tick = ns_per_tick();
    srand(42);

    printf("M4T substrate profiling — hot-path primitive throughput\n");
    printf("Timer resolution: %.2f ns/tick\n\n", ns_tick);

    /* VCNT path: popcount_dist across all three code tiers. */
    printf("VCNT (popcount_dist) — ternary Hamming distance:\n");
    bench_popcount_dist(16,   5000000, ns_tick);  /* 4B:  scalar */
    bench_popcount_dist(32,   5000000, ns_tick);  /* 8B:  64-bit popcount */
    bench_popcount_dist(64,   2000000, ns_tick);  /* 16B: NEON 1 iter */
    bench_popcount_dist(128,  2000000, ns_tick);  /* 32B: NEON 2 iter */
    bench_popcount_dist(256,  1000000, ns_tick);  /* 64B: NEON 4 iter */
    bench_popcount_dist(512,  1000000, ns_tick);  /* 128B: NEON 8 iter */
    bench_popcount_dist(1024,  500000, ns_tick);  /* 256B: sustained */
    printf("\n");

    /* SDOT path: int8 ternary matmul. */
    printf("SDOT (mtfp4_sdot_matmul_bt) — int8 ternary matmul:\n");
    bench_sdot_matmul(1, 128,  16, 500000, ns_tick);
    bench_sdot_matmul(1, 256,  16, 500000, ns_tick);
    bench_sdot_matmul(1, 512,  16, 200000, ns_tick);
    bench_sdot_matmul(1, 1024, 16, 100000, ns_tick);
    bench_sdot_matmul(4, 512,  32, 100000, ns_tick);
    bench_sdot_matmul(16, 256, 32,  50000, ns_tick);
    printf("\n");

    /* TBL path: trit element-wise ops. */
    printf("TBL (trit_mul) — element-wise ternary operations:\n");
    bench_trit_mul(64,    5000000, ns_tick);  /* 16B: NEON 1 iter */
    bench_trit_mul(256,   2000000, ns_tick);
    bench_trit_mul(1024,  1000000, ns_tick);
    bench_trit_mul(4096,   500000, ns_tick);  /* sustained */
    printf("\n");

    /* Masked-VCNT: trit counting reduction. */
    printf("Masked-VCNT (trit_counts) — trit distribution reduction:\n");
    bench_trit_counts(64,    5000000, ns_tick);
    bench_trit_counts(256,   2000000, ns_tick);
    bench_trit_counts(1024,  1000000, ns_tick);
    bench_trit_counts(4096,   500000, ns_tick);
    printf("\n");

    /* MTFP19 ternary matmul (TBL-based trit decode + scalar MAC). */
    printf("TBL+MAC (ternary_matmul_bt) — MTFP19 ternary matmul:\n");
    bench_ternary_matmul(1, 128,  16, 500000, ns_tick);
    bench_ternary_matmul(1, 512,  16, 200000, ns_tick);
    bench_ternary_matmul(1, 1024, 16, 100000, ns_tick);
    bench_ternary_matmul(4, 512,  32, 100000, ns_tick);
    printf("\n");

    printf("Done. Run on idle system for reliable numbers.\n");
    return 0;
}
