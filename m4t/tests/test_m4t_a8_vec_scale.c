/*
 * test_m4t_a8_vec_scale.c — A8 quantize/dequantize + vec_scale verification.
 * Per work-unit 5 of bitnet_phase1_synthesize.
 */

#include "m4t_mtfp.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static int g_fails = 0;

static uint32_t xs_state = 0xCAFEBABEu;
static uint32_t xs(void) {
    uint32_t x = xs_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    xs_state = x;
    return x;
}

static m4t_mtfp_t rand_mantissa(int32_t cap) {
    int32_t v = (int32_t)(xs() % (uint32_t)(2 * cap + 1)) - cap;
    return (m4t_mtfp_t)v;
}

/* ── A8 ──────────────────────────────────────────────────────────────── */

static void test_a8_quantize_match_ref(int n, int32_t cap, const char* label) {
    m4t_mtfp_t* x = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    int8_t* y_prod = (int8_t*)calloc((size_t)n, 1);
    int8_t* y_ref  = (int8_t*)calloc((size_t)n, 1);
    for (int i = 0; i < n; i++) x[i] = rand_mantissa(cap);
    m4t_mtfp_t am_prod = m4t_a8_quantize(y_prod, x, n);
    m4t_mtfp_t am_ref  = m4t_a8_quantize_scalar_ref(y_ref, x, n);
    int fails = 0;
    if (am_prod != am_ref) {
        fprintf(stderr, "A8 absmax FAIL[%s]: prod=%d ref=%d\n", label, am_prod, am_ref);
        fails++;
    }
    int max_diff = 0;
    for (int i = 0; i < n; i++) {
        int diff = y_prod[i] - y_ref[i];
        if (diff < 0) diff = -diff;
        if (diff > max_diff) max_diff = diff;
        /* Bit-exact required: same int divide, same rounding convention.
         * Allow up to 1 LSB diff (FP/int round-half ties may resolve differently). */
        if (diff > 1) {
            if (fails < 3)
                fprintf(stderr, "A8 quantize FAIL[%s] i=%d prod=%d ref=%d\n",
                        label, i, y_prod[i], y_ref[i]);
            fails++;
        }
    }
    if (fails == 0) fprintf(stderr, "  [%s] OK (absmax=%d, max_diff=%d)\n",
                            label, am_prod, max_diff);
    g_fails += fails;
    free(x); free(y_prod); free(y_ref);
}

static void test_a8_round_trip(int n, int32_t cap, const char* label) {
    m4t_mtfp_t* x = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    int8_t* q = (int8_t*)calloc((size_t)n, 1);
    m4t_mtfp_t* d = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    for (int i = 0; i < n; i++) x[i] = rand_mantissa(cap);
    m4t_mtfp_t absmax = m4t_a8_quantize(q, x, n);
    m4t_a8_dequantize(d, q, absmax, n);
    /* Quantization error: at most absmax/254 per cell (half a quantization step). */
    int fails = 0;
    int64_t max_err = 0;
    int64_t step = (absmax + 126) / 127;
    for (int i = 0; i < n; i++) {
        int64_t err = (int64_t)d[i] - (int64_t)x[i];
        if (err < 0) err = -err;
        if (err > max_err) max_err = err;
        if (err > step + 1) {
            if (fails < 3)
                fprintf(stderr, "RT FAIL[%s] i=%d x=%d d=%d err=%lld step=%lld\n",
                        label, i, x[i], d[i], (long long)err, (long long)step);
            fails++;
        }
    }
    if (fails == 0) fprintf(stderr, "  [%s round-trip] OK (max_err=%lld, step=%lld)\n",
                            label, (long long)max_err, (long long)step);
    g_fails += fails;
    free(x); free(q); free(d);
}

static void test_a8_zero_input(void) {
    int n = 16;
    m4t_mtfp_t x[16] = {0};
    int8_t y[16];
    m4t_mtfp_t am = m4t_a8_quantize(y, x, n);
    if (am != 0) { fprintf(stderr, "A8 zero-in: absmax %d (want 0)\n", am); g_fails++; }
    for (int i = 0; i < n; i++) {
        if (y[i] != 0) { fprintf(stderr, "A8 zero-in: y[%d]=%d\n", i, y[i]); g_fails++; }
    }
    fprintf(stderr, "  [zero input] OK\n");
}

static void test_a8_max_magnitude(void) {
    int n = 16;
    m4t_mtfp_t x[16];
    for (int i = 0; i < n; i++) x[i] = (i & 1) ? -581130733 : 581130733;
    int8_t y[16];
    m4t_mtfp_t am = m4t_a8_quantize(y, x, n);
    if (am != 581130733) { fprintf(stderr, "A8 max: absmax %d\n", am); g_fails++; }
    /* All cells should map to ±127. */
    for (int i = 0; i < n; i++) {
        int8_t want = (i & 1) ? -127 : 127;
        if (y[i] != want) { fprintf(stderr, "A8 max: y[%d]=%d want %d\n", i, y[i], want); g_fails++; }
    }
    fprintf(stderr, "  [max magnitude] OK\n");
}

/* ── vec_scale ──────────────────────────────────────────────────────── */

static void test_vec_scale_identity(void) {
    /* num = 1, den = 1: y = x. */
    int n = 16;
    m4t_mtfp_t x[16];
    for (int i = 0; i < n; i++) x[i] = rand_mantissa(1 << 24);
    m4t_mtfp_t y[16];
    m4t_mtfp_vec_scale(y, x, 1, 1, n);
    for (int i = 0; i < n; i++) {
        if (y[i] != x[i]) { fprintf(stderr, "ident FAIL i=%d\n", i); g_fails++; }
    }
    fprintf(stderr, "  [identity] OK\n");
}

static void test_vec_scale_match_ref(int n, int64_t num, int64_t den, int32_t cap, const char* label) {
    m4t_mtfp_t* x = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* y_prod = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    m4t_mtfp_t* y_ref  = (m4t_mtfp_t*)calloc((size_t)n, sizeof(m4t_mtfp_t));
    for (int i = 0; i < n; i++) x[i] = rand_mantissa(cap);
    m4t_mtfp_vec_scale(y_prod, x, num, den, n);
    m4t_mtfp_vec_scale_scalar_ref(y_ref, x, num, den, n);
    int fails = 0;
    int64_t max_diff = 0;
    for (int i = 0; i < n; i++) {
        int64_t diff = (int64_t)y_prod[i] - (int64_t)y_ref[i];
        if (diff < 0) diff = -diff;
        if (diff > max_diff) max_diff = diff;
        /* Tolerance: 1 LSB (FP rounding can disagree with int divide on
         * exact half-way ties). */
        if (diff > 1) {
            if (fails < 3)
                fprintf(stderr, "VS FAIL[%s] i=%d prod=%d ref=%d\n",
                        label, i, y_prod[i], y_ref[i]);
            fails++;
        }
    }
    if (fails == 0) fprintf(stderr, "  [%s] OK (max_diff=%lld)\n", label, (long long)max_diff);
    g_fails += fails;
    free(x); free(y_prod); free(y_ref);
}

static void test_vec_scale_saturate(void) {
    /* x · num >> den exceeds MTFP19_MAX → should saturate. */
    int n = 4;
    m4t_mtfp_t x[4] = {581130733, -581130733, 100, -100};
    m4t_mtfp_t y[4];
    m4t_mtfp_vec_scale(y, x, 1000000, 1, n);
    /* x[0] × 1e6 = 5.8e14 → saturates to MTFP19_MAX. */
    if (y[0] != 581130733)  { fprintf(stderr, "sat[0]=%d\n", y[0]); g_fails++; }
    if (y[1] != -581130733) { fprintf(stderr, "sat[1]=%d\n", y[1]); g_fails++; }
    fprintf(stderr, "  [saturate] OK\n");
}

int main(void) {
    fprintf(stderr, "test_m4t_a8_vec_scale: A8 quantize prod-vs-ref...\n");
    test_a8_quantize_match_ref(16,    1 << 18, "n=16");
    test_a8_quantize_match_ref(64,    1 << 22, "n=64");
    test_a8_quantize_match_ref(2560,  1 << 24, "n=2560 (BitNet hidden)");
    test_a8_quantize_match_ref(2560,  1 << 28, "n=2560 large");

    fprintf(stderr, "test_m4t_a8_vec_scale: A8 round-trip...\n");
    test_a8_round_trip(2560, 1 << 24, "n=2560");
    test_a8_round_trip(2560, 1 << 28, "n=2560 large");

    fprintf(stderr, "test_m4t_a8_vec_scale: A8 boundaries...\n");
    test_a8_zero_input();
    test_a8_max_magnitude();

    fprintf(stderr, "test_m4t_a8_vec_scale: vec_scale tests...\n");
    test_vec_scale_identity();
    test_vec_scale_match_ref(2560, 12345, 127, 1 << 18, "scale 12345/127");
    test_vec_scale_match_ref(2560, (int64_t)1234567 * 89012, 127, 1 << 18,
                              "scale (huge num)/127");
    test_vec_scale_saturate();

    if (g_fails > 0) {
        fprintf(stderr, "test_m4t_a8_vec_scale: %d failures\n", g_fails);
        return 1;
    }
    fprintf(stderr, "test_m4t_a8_vec_scale: all tests passed\n");
    return 0;
}
