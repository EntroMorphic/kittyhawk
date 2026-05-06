/*
 * test_m4t_rmsnorm.c — verification of m4t_mtfp_rmsnorm.
 *
 * Per work-unit 2 of the bitnet_phase1 LMM cycle.
 *
 * Gates:
 * 1. Tolerance vs FP scalar_ref. NR-based pure-int rsqrt is not bit-exact
 *    vs libm sqrt; the per-cell γ × x × inv pipeline accumulates a few
 *    LSBs of rounding too. Tolerance bounds the divergence.
 * 2. Boundary behavior: n=0 (noop), n=1, n=4, n=2560 (BitNet's hidden).
 * 3. Aliasing: y == x and y == γ both supported.
 *
 * Inputs are random MTFP19 mantissas. γ values stay in a moderate band
 * (|γ| ≤ 2^20) because BitNet's RMSNorm γ is small after bf16→MTFP19
 * conversion; testing extreme γ × x × inv to saturation is a separate
 * concern (clamp is exercised by the saturation test).
 */

#include "m4t_mtfp.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

static int g_fails = 0;

static uint32_t xs_state = 0xBADBEEF1u;
static uint32_t xs(void) {
    uint32_t x = xs_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    xs_state = x;
    return x;
}

/* Random MTFP19 mantissa in [-cap, cap]. */
static m4t_mtfp_t rand_mantissa(int32_t cap) {
    int32_t v = (int32_t)(xs() % (uint32_t)(2 * cap + 1)) - cap;
    return (m4t_mtfp_t)v;
}

/* Compare prod[i] vs ref[i] with tolerance. Returns failure count.
 * Tolerance: 1e-3 relative (rsqrt's NR loses ~5e-7 relative; the per-cell
 * 3-way multiply adds ~1 LSB rounding; cumulative ~1e-5. We use 1e-3 as a
 * generous bound that catches genuine algorithm bugs while ignoring the
 * fixed-point rounding floor). Floor of 4 LSB for tiny refs. */
static int compare_arrays(const m4t_mtfp_t* prod, const m4t_mtfp_t* ref,
                          int n, const char* label) {
    int fails = 0;
    int64_t max_diff = 0;
    for (int i = 0; i < n; i++) {
        int64_t diff = (int64_t)prod[i] - (int64_t)ref[i];
        if (diff < 0) diff = -diff;
        if (diff > max_diff) max_diff = diff;
        int64_t aref = ref[i] < 0 ? -(int64_t)ref[i] : (int64_t)ref[i];
        int64_t tol = (int64_t)((double)aref * 1e-3) + 4;
        if (diff > tol) {
            if (fails < 5) {
                fprintf(stderr, "FAIL[%s] i=%d prod=%d ref=%d diff=%lld tol=%lld\n",
                        label, i, prod[i], ref[i], (long long)diff, (long long)tol);
            }
            fails++;
        }
    }
    if (fails == 0) {
        fprintf(stderr, "  [%s] n=%d OK (max_diff=%lld)\n",
                label, n, (long long)max_diff);
    } else {
        fprintf(stderr, "  [%s] n=%d %d fails\n", label, n, fails);
    }
    g_fails += fails;
    return fails;
}

static void run_random_n(int n, int x_cap, int gamma_cap, m4t_mtfp_t eps,
                         const char* label) {
    m4t_mtfp_t* x = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* g = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* y_prod = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* y_ref  = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    for (int i = 0; i < n; i++) {
        x[i] = rand_mantissa(x_cap);
        g[i] = rand_mantissa(gamma_cap);
        if (x[i] == 0) x[i] = 1;
    }
    m4t_mtfp_rmsnorm(y_prod, x, g, eps, n);
    m4t_mtfp_rmsnorm_scalar_ref(y_ref, x, g, eps, n);
    compare_arrays(y_prod, y_ref, n, label);
    free(x); free(g); free(y_prod); free(y_ref);
}

static void test_n_zero(void) {
    /* n=0 is a noop — no read or write. Pass NULL deliberately to verify. */
    m4t_mtfp_rmsnorm(NULL, NULL, NULL, 1, 0);
    m4t_mtfp_rmsnorm_scalar_ref(NULL, NULL, NULL, 1, 0);
    fprintf(stderr, "  [n=0 noop] OK\n");
}

static void test_aliasing_y_eq_x(void) {
    /* y == x: y is computed cell-by-cell, but each cell only reads x[i] in
     * the same iteration, so writing y[i] = ... after the read is safe. */
    int n = 64;
    m4t_mtfp_t* x_orig = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* g = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* y_separate = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* y_aliased  = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    for (int i = 0; i < n; i++) {
        x_orig[i] = rand_mantissa(1 << 24);
        if (x_orig[i] == 0) x_orig[i] = 1;
        g[i] = rand_mantissa(1 << 18);
        y_aliased[i] = x_orig[i];  /* alias buffer starts as x */
    }
    m4t_mtfp_rmsnorm(y_separate, x_orig, g, 1, n);
    m4t_mtfp_rmsnorm(y_aliased,  y_aliased, g, 1, n);
    compare_arrays(y_aliased, y_separate, n, "y==x alias");
    free(x_orig); free(g); free(y_separate); free(y_aliased);
}

static void test_aliasing_y_eq_gamma(void) {
    /* y == γ: same cell-locality argument applies. */
    int n = 64;
    m4t_mtfp_t* x = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* g_orig = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* y_separate = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* y_aliased  = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    for (int i = 0; i < n; i++) {
        x[i] = rand_mantissa(1 << 24);
        if (x[i] == 0) x[i] = 1;
        g_orig[i] = rand_mantissa(1 << 18);
        y_aliased[i] = g_orig[i];
    }
    m4t_mtfp_rmsnorm(y_separate, x, g_orig, 1, n);
    m4t_mtfp_rmsnorm(y_aliased,  x, y_aliased, 1, n);
    compare_arrays(y_aliased, y_separate, n, "y==γ alias");
    free(x); free(g_orig); free(y_separate); free(y_aliased);
}

static void test_zero_input(void) {
    /* All-zero x → mean_sq=0, mean_shifted = ε. With ε=1, inv ≈ 2^30,
     * inv_at_30 = inv >> 4 ≈ 2^26. y = γ × 0 × inv = 0 for all cells. */
    int n = 16;
    m4t_mtfp_t x[16] = {0};
    m4t_mtfp_t g[16];
    for (int i = 0; i < n; i++) g[i] = rand_mantissa(1 << 18);
    m4t_mtfp_t y[16];
    m4t_mtfp_rmsnorm(y, x, g, 1, n);
    for (int i = 0; i < n; i++) {
        if (y[i] != 0) {
            fprintf(stderr, "FAIL[zero_x]: y[%d]=%d (want 0)\n", i, y[i]);
            g_fails++;
        }
    }
    fprintf(stderr, "  [zero x] OK\n");
}

int main(void) {
    fprintf(stderr, "test_m4t_rmsnorm: n=0 noop...\n");
    test_n_zero();

    fprintf(stderr, "test_m4t_rmsnorm: zero input...\n");
    test_zero_input();

    fprintf(stderr, "test_m4t_rmsnorm: random n boundaries...\n");
    /* x_cap = 2^24 (well under MTFP19_MAX); γ small (~2^18, BitNet-like). */
    run_random_n(1,    1 << 24, 1 << 18, 1, "n=1");
    run_random_n(4,    1 << 24, 1 << 18, 1, "n=4");
    run_random_n(64,   1 << 24, 1 << 18, 1, "n=64");
    run_random_n(2560, 1 << 24, 1 << 18, 1, "n=2560 (BitNet hidden)");

    fprintf(stderr, "test_m4t_rmsnorm: aliasing...\n");
    test_aliasing_y_eq_x();
    test_aliasing_y_eq_gamma();
    {
        /* γ == x full aliasing: same buffer for both. Cell-local reads
         * happen before the write, so safe. */
        int n = 64;
        m4t_mtfp_t* shared = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* x_copy = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* y_aliased = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        m4t_mtfp_t* y_separate = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
        for (int i = 0; i < n; i++) {
            shared[i] = rand_mantissa(1 << 22);
            if (shared[i] == 0) shared[i] = 1;
            x_copy[i] = shared[i];
        }
        m4t_mtfp_rmsnorm(y_separate, x_copy, x_copy, 1, n);
        m4t_mtfp_rmsnorm(y_aliased,  shared, shared, 1, n);
        compare_arrays(y_aliased, y_separate, n, "γ==x alias");
        free(shared); free(x_copy); free(y_aliased); free(y_separate);
    }

    fprintf(stderr, "test_m4t_rmsnorm: stress (large x, large γ)...\n");
    run_random_n(2560, 1 << 28, 1 << 22, 1, "n=2560 large");

    fprintf(stderr, "test_m4t_rmsnorm: small ε variants...\n");
    run_random_n(2560, 1 << 24, 1 << 18, 4, "ε=4");
    run_random_n(2560, 1 << 24, 1 << 18, 100, "ε=100");

    if (g_fails > 0) {
        fprintf(stderr, "test_m4t_rmsnorm: %d failures\n", g_fails);
        return 1;
    }
    fprintf(stderr, "test_m4t_rmsnorm: all tests passed\n");
    return 0;
}
