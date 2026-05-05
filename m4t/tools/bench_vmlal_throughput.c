/*
 * bench_vmlal_throughput.c — characterize vmlal_s32 throughput on the
 * target M-series. Per ternary_mac_routing T-G1.
 *
 * Defenses against constant-folding (round 3 — round 1 was naive
 * constants, round 2 was noinline+asm barrier; both got folded because
 * the compiler factored a*b out of the loop. This round forces inputs
 * to be data-dependent by reading from a non-constant array per iter,
 * mirroring the realistic kernel context where activations stream from
 * memory):
 *   - inputs read from a large heap-allocated array filled at runtime
 *   - bench fns are __attribute__((noinline))
 *   - the array's contents are NOT visible to the compiler at compile time
 *
 * Build:  cc -O3 -mcpu=native m4t/tools/bench_vmlal_throughput.c -o /tmp/bvt
 * Run:    /tmp/bvt
 */

#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define POOL_BYTES (1024 * 1024)  /* 1 MB pool, fits L2 */
#define ITERS 5000000             /* 5M outer iters; each does 8 vmlal calls */

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

/* All patterns load 8 DIFFERENT (a, b) pairs per iter — one per vmlal call.
 * This mirrors the real ternary-matmul kernel where each of the 8 calls
 * processes a different element pair. With identical inputs across the 8
 * calls the compiler factors `acc += 8*(a*b)` and we measure 1 smull/iter
 * instead of 8 vmlals. Defeating that requires distinct inputs per call. */

#define LOAD_8_PAIRS(a_pool, b_pool, idx, mask)                            \
    int32x2_t _a0 = vld1_s32(a_pool + ((idx +  0) & mask));                \
    int32x2_t _a1 = vld1_s32(a_pool + ((idx +  2) & mask));                \
    int32x2_t _a2 = vld1_s32(a_pool + ((idx +  4) & mask));                \
    int32x2_t _a3 = vld1_s32(a_pool + ((idx +  6) & mask));                \
    int32x2_t _a4 = vld1_s32(a_pool + ((idx +  8) & mask));                \
    int32x2_t _a5 = vld1_s32(a_pool + ((idx + 10) & mask));                \
    int32x2_t _a6 = vld1_s32(a_pool + ((idx + 12) & mask));                \
    int32x2_t _a7 = vld1_s32(a_pool + ((idx + 14) & mask));                \
    int32x2_t _b0 = vld1_s32(b_pool + ((idx +  0) & mask));                \
    int32x2_t _b1 = vld1_s32(b_pool + ((idx +  2) & mask));                \
    int32x2_t _b2 = vld1_s32(b_pool + ((idx +  4) & mask));                \
    int32x2_t _b3 = vld1_s32(b_pool + ((idx +  6) & mask));                \
    int32x2_t _b4 = vld1_s32(b_pool + ((idx +  8) & mask));                \
    int32x2_t _b5 = vld1_s32(b_pool + ((idx + 10) & mask));                \
    int32x2_t _b6 = vld1_s32(b_pool + ((idx + 12) & mask));                \
    int32x2_t _b7 = vld1_s32(b_pool + ((idx + 14) & mask));

/* Pattern A: 8 independent accumulators; max throughput. */
__attribute__((noinline))
static int64_t bench_independent(const int32_t* a_pool, const int32_t* b_pool,
                                  int32_t pool_mask) {
    int64x2_t acc0 = vdupq_n_s64(0);
    int64x2_t acc1 = vdupq_n_s64(0);
    int64x2_t acc2 = vdupq_n_s64(0);
    int64x2_t acc3 = vdupq_n_s64(0);
    int64x2_t acc4 = vdupq_n_s64(0);
    int64x2_t acc5 = vdupq_n_s64(0);
    int64x2_t acc6 = vdupq_n_s64(0);
    int64x2_t acc7 = vdupq_n_s64(0);
    int32_t idx = 0;
    for (int i = 0; i < ITERS; i++) {
        LOAD_8_PAIRS(a_pool, b_pool, idx, pool_mask)
        acc0 = vmlal_s32(acc0, _a0, _b0);
        acc1 = vmlal_s32(acc1, _a1, _b1);
        acc2 = vmlal_s32(acc2, _a2, _b2);
        acc3 = vmlal_s32(acc3, _a3, _b3);
        acc4 = vmlal_s32(acc4, _a4, _b4);
        acc5 = vmlal_s32(acc5, _a5, _b5);
        acc6 = vmlal_s32(acc6, _a6, _b6);
        acc7 = vmlal_s32(acc7, _a7, _b7);
        idx += 16;
    }
    int64x2_t sum = vaddq_s64(vaddq_s64(vaddq_s64(acc0, acc1), vaddq_s64(acc2, acc3)),
                               vaddq_s64(vaddq_s64(acc4, acc5), vaddq_s64(acc6, acc7)));
    return vgetq_lane_s64(sum, 0) + vgetq_lane_s64(sum, 1);
}

/* Pattern B: 8 chained into one accumulator; dependency-bound. */
__attribute__((noinline))
static int64_t bench_chained(const int32_t* a_pool, const int32_t* b_pool,
                              int32_t pool_mask) {
    int64x2_t acc = vdupq_n_s64(0);
    int32_t idx = 0;
    for (int i = 0; i < ITERS; i++) {
        LOAD_8_PAIRS(a_pool, b_pool, idx, pool_mask)
        acc = vmlal_s32(acc, _a0, _b0);
        acc = vmlal_s32(acc, _a1, _b1);
        acc = vmlal_s32(acc, _a2, _b2);
        acc = vmlal_s32(acc, _a3, _b3);
        acc = vmlal_s32(acc, _a4, _b4);
        acc = vmlal_s32(acc, _a5, _b5);
        acc = vmlal_s32(acc, _a6, _b6);
        acc = vmlal_s32(acc, _a7, _b7);
        idx += 16;
    }
    return vgetq_lane_s64(acc, 0) + vgetq_lane_s64(acc, 1);
}

/* Pattern C: TWO chained pairs (matches the kernel's acc0+acc1). */
__attribute__((noinline))
static int64_t bench_two_chains(const int32_t* a_pool, const int32_t* b_pool,
                                 int32_t pool_mask) {
    int64x2_t acc0 = vdupq_n_s64(0);
    int64x2_t acc1 = vdupq_n_s64(0);
    int32_t idx = 0;
    for (int i = 0; i < ITERS; i++) {
        LOAD_8_PAIRS(a_pool, b_pool, idx, pool_mask)
        acc0 = vmlal_s32(acc0, _a0, _b0);
        acc1 = vmlal_s32(acc1, _a1, _b1);
        acc0 = vmlal_s32(acc0, _a2, _b2);
        acc1 = vmlal_s32(acc1, _a3, _b3);
        acc0 = vmlal_s32(acc0, _a4, _b4);
        acc1 = vmlal_s32(acc1, _a5, _b5);
        acc0 = vmlal_s32(acc0, _a6, _b6);
        acc1 = vmlal_s32(acc1, _a7, _b7);
        idx += 16;
    }
    int64x2_t sum = vaddq_s64(acc0, acc1);
    return vgetq_lane_s64(sum, 0) + vgetq_lane_s64(sum, 1);
}

typedef int64_t (*bench_fn)(const int32_t*, const int32_t*, int32_t);

static void run(const char* label, bench_fn fn,
                const int32_t* a, const int32_t* b, int32_t mask) {
    /* Warmup. */
    volatile int64_t sink_w = fn(a, b, mask);
    (void)sink_w;
    /* Min-of-5 sample. */
    double best_ns = 1e18;
    for (int s = 0; s < 5; s++) {
        double t0 = now_ns();
        volatile int64_t sink = fn(a, b, mask);
        double t1 = now_ns();
        (void)sink;
        double ns = t1 - t0;
        if (ns < best_ns) best_ns = ns;
    }
    double calls = (double)ITERS * 8.0;
    double ns_per_call = best_ns / calls;
    double cycles_per_call = ns_per_call * 3.5;
    double calls_per_cycle = 1.0 / cycles_per_call;
    printf("  %-30s : %.3f ns/call  ≈ %.2f cycles/call  ≈ %.2f calls/cycle\n",
           label, ns_per_call, cycles_per_call, calls_per_cycle);
}

int main(void) {
    int n = POOL_BYTES / sizeof(int32_t);
    /* Use pow2 for fast mask wrap. */
    int32_t mask = (int32_t)(n / 2 - 1) & ~(int32_t)1;
    int32_t* a_pool = malloc(POOL_BYTES);
    int32_t* b_pool = malloc(POOL_BYTES);
    if (!a_pool || !b_pool) { perror("malloc"); return 1; }
    /* Fill with pid-derived data; compiler can't see the values. */
    int32_t pid = (int32_t)getpid();
    for (int i = 0; i < n; i++) {
        a_pool[i] = pid + i * 17;
        b_pool[i] = pid + i * 23;
    }

    printf("vmlal_s32 throughput (%d outer iters × 8 calls; pool=%d bytes):\n",
           ITERS, POOL_BYTES);
    run("A. independent (8 accs)",   bench_independent, a_pool, b_pool, mask);
    run("C. two chains (acc0,acc1)", bench_two_chains,  a_pool, b_pool, mask);
    run("B. chained (1 acc)",         bench_chained,     a_pool, b_pool, mask);

    free(a_pool); free(b_pool);
    return 0;
}
