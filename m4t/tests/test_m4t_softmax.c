/*
 * test_m4t_softmax.c — verification of m4t_int32_recip + m4t_mtfp_softmax.
 *
 * Per work-unit 4 of bitnet_phase1_synthesize.
 *
 * Gates:
 * 1. Recip: tolerance vs scalar_ref across boundary + random.
 * 2. Softmax: tolerance vs FP scalar_ref.
 * 3. Property: Σ y[i] ≈ 2^30 (probabilities sum to 1).
 * 4. Boundaries: n=1, uniform input, one-hot input.
 */

#include "m4t_mtfp.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static int g_fails = 0;

static uint32_t xs_state = 0xFEEDFACEu;
static uint32_t xs(void) {
    uint32_t x = xs_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    xs_state = x;
    return x;
}

/* ── Recip tests ─────────────────────────────────────────────────────── */

static int recip_check_one(m4t_mtfp_t src) {
    m4t_mtfp_t prod = m4t_int32_recip(src);
    m4t_mtfp_t ref  = m4t_int32_recip_scalar_ref(src);
    int32_t diff = prod > ref ? prod - ref : ref - prod;
    int32_t tol  = (int32_t)((double)ref * 1e-5) + 2;
    if (diff > tol) {
        if (g_fails < 5) {
            fprintf(stderr, "RECIP FAIL: src=%d prod=%d ref=%d diff=%d tol=%d\n",
                    src, prod, ref, diff, tol);
        }
        g_fails++;
        return 1;
    }
    return 0;
}

static void test_recip_boundaries(void) {
    if (m4t_int32_recip(0) != 0) { fprintf(stderr, "FAIL: recip(0)\n"); g_fails++; }
    recip_check_one(1);
    recip_check_one(2);
    recip_check_one(1024);
    recip_check_one(1073741824);
    recip_check_one(2147483647);
    for (int p = 0; p < 31; p++) recip_check_one(1 << p);
    fprintf(stderr, "  [recip boundaries] checked\n");
}

static void test_recip_random(int n) {
    int start_fails = g_fails;
    for (int i = 0; i < n; i++) {
        uint32_t r = xs();
        int shift = (int)(xs() % 31u);
        m4t_mtfp_t src = (m4t_mtfp_t)((r >> shift) & 0x7FFFFFFFu);
        if (src == 0) src = 1;
        recip_check_one(src);
    }
    fprintf(stderr, "  [recip random] %d samples, %d fails\n",
            n, g_fails - start_fails);
}

/* ── Softmax tests ───────────────────────────────────────────────────── */

static int softmax_compare(const m4t_mtfp_t* prod, const m4t_mtfp_t* ref,
                           int n, const char* label) {
    int fails = 0;
    int64_t max_diff = 0;
    int64_t sum = 0;
    for (int i = 0; i < n; i++) {
        int64_t diff = (int64_t)prod[i] - (int64_t)ref[i];
        if (diff < 0) diff = -diff;
        if (diff > max_diff) max_diff = diff;
        sum += prod[i];
        /* Tolerance: 1e-2 relative + 64 LSB (LUT quantization at 4096
         * entries × 30 range gives ~7 bits of LUT error per cell). */
        int64_t aref = ref[i];
        int64_t tol = (int64_t)((double)aref * 1e-2) + 64;
        if (diff > tol) {
            if (fails < 3) {
                fprintf(stderr, "SOFTMAX FAIL[%s] i=%d prod=%d ref=%d diff=%lld tol=%lld\n",
                        label, i, prod[i], ref[i], (long long)diff, (long long)tol);
            }
            fails++;
        }
    }
    /* Property: sum ≈ 2^30 (allow ±0.5% drift from quantization). */
    int64_t expected = (int64_t)1 << 30;
    int64_t sum_diff = sum > expected ? sum - expected : expected - sum;
    int64_t sum_tol  = expected / 200;  /* 0.5% */
    int sum_fail = (sum_diff > sum_tol) ? 1 : 0;
    if (fails == 0 && !sum_fail) {
        fprintf(stderr, "  [%s] OK (max_cell_diff=%lld, sum_off=%lld over %d cells)\n",
                label, (long long)max_diff, (long long)sum_diff, n);
    } else {
        fprintf(stderr, "  [%s] %d cell fails, sum_off=%lld (tol=%lld)\n",
                label, fails, (long long)sum_diff, (long long)sum_tol);
    }
    g_fails += fails + sum_fail;
    return fails;
}

static void test_softmax_uniform(void) {
    /* All x[i] = 5: softmax → uniform 1/n. y[i] = 2^30 / n. */
    int n = 32;
    m4t_mtfp_t x[32];
    for (int i = 0; i < n; i++) x[i] = 5;
    m4t_mtfp_t y[32];
    m4t_mtfp_softmax(y, x, n);
    int64_t expected = ((int64_t)1 << 30) / n;
    int fails = 0;
    int64_t max_diff = 0;
    for (int i = 0; i < n; i++) {
        int64_t diff = (int64_t)y[i] - expected;
        if (diff < 0) diff = -diff;
        if (diff > max_diff) max_diff = diff;
        if (diff > expected / 100) fails++;
    }
    if (fails == 0) {
        fprintf(stderr, "  [uniform n=%d] OK (max_diff=%lld vs expected=%lld)\n",
                n, (long long)max_diff, (long long)expected);
    } else {
        fprintf(stderr, "  [uniform n=%d] %d fails (max_diff=%lld)\n",
                n, fails, (long long)max_diff);
        g_fails += fails;
    }
}

static void test_softmax_one_hot(void) {
    /* x[0] = 50 (way above LUT range from others), x[1..n-1] = 0.
     * y[0] should be ≈ 2^30, others ≈ 0. */
    int n = 16;
    m4t_mtfp_t x[16] = {0};
    x[0] = 50;
    m4t_mtfp_t y[16];
    m4t_mtfp_softmax(y, x, n);
    int64_t y0 = y[0];
    int64_t target = (int64_t)1 << 30;
    int fails = 0;
    if (y0 < target - target / 100) {
        fprintf(stderr, "ONE-HOT FAIL: y[0]=%lld < %lld\n", (long long)y0, (long long)target);
        fails++;
    }
    for (int i = 1; i < n; i++) {
        if (y[i] > target / 1000) {
            fprintf(stderr, "ONE-HOT FAIL: y[%d]=%d (want ≈ 0)\n", i, y[i]);
            fails++;
        }
    }
    if (fails == 0) fprintf(stderr, "  [one-hot] OK (y[0]=%lld)\n", (long long)y0);
    else g_fails += fails;
}

static void test_softmax_n_one(void) {
    m4t_mtfp_t x[1] = {42};
    m4t_mtfp_t y[1];
    m4t_mtfp_softmax(y, x, 1);
    if (y[0] == ((int32_t)1 << 30)) {
        fprintf(stderr, "  [n=1] OK\n");
    } else {
        fprintf(stderr, "  [n=1] FAIL: y[0]=%d (want %d)\n", y[0], (int32_t)1 << 30);
        g_fails++;
    }
}

static void test_softmax_random(int n) {
    m4t_mtfp_t* x = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* y_prod = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* y_ref  = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    /* Generate scores in roughly [-15, +15] (representative of attention). */
    for (int i = 0; i < n; i++) {
        int v = (int)(xs() % 31u) - 15;
        x[i] = (m4t_mtfp_t)v;
    }
    m4t_mtfp_softmax(y_prod, x, n);
    m4t_mtfp_softmax_scalar_ref(y_ref, x, n);
    char label[64];
    snprintf(label, sizeof(label), "random n=%d", n);
    softmax_compare(y_prod, y_ref, n, label);
    free(x); free(y_prod); free(y_ref);
}

static void test_softmax_underflow(void) {
    /* x[0] = 0, x[1] = -100 (way below LUT range → exp underflows to 0).
     * y[0] should be ≈ 2^30, y[1] ≈ 0. */
    m4t_mtfp_t x[2] = {0, -100};
    m4t_mtfp_t y[2];
    m4t_mtfp_softmax(y, x, 2);
    if (y[0] >= ((int32_t)1 << 30) - 16 && y[1] <= 16) {
        fprintf(stderr, "  [underflow] OK (y[0]=%d, y[1]=%d)\n", y[0], y[1]);
    } else {
        fprintf(stderr, "  [underflow] FAIL (y[0]=%d, y[1]=%d)\n", y[0], y[1]);
        g_fails++;
    }
}

int main(void) {
    fprintf(stderr, "test_m4t_softmax: recip boundaries...\n");
    test_recip_boundaries();
    fprintf(stderr, "test_m4t_softmax: recip random 5000...\n");
    test_recip_random(5000);

    fprintf(stderr, "test_m4t_softmax: n=1...\n");
    test_softmax_n_one();
    fprintf(stderr, "test_m4t_softmax: uniform...\n");
    test_softmax_uniform();
    fprintf(stderr, "test_m4t_softmax: one-hot...\n");
    test_softmax_one_hot();
    fprintf(stderr, "test_m4t_softmax: underflow...\n");
    test_softmax_underflow();
    fprintf(stderr, "test_m4t_softmax: random vs FP scalar_ref...\n");
    test_softmax_random(8);
    test_softmax_random(64);
    test_softmax_random(2048);

    if (g_fails > 0) {
        fprintf(stderr, "test_m4t_softmax: %d failures\n", g_fails);
        return 1;
    }
    fprintf(stderr, "test_m4t_softmax: all tests passed\n");
    return 0;
}
