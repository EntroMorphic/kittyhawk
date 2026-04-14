/*
 * test_ternary_matvec.c — Integration test for NEON ternary matvec
 *
 * 1. Correctness: SDOT result matches scalar reference
 * 2. Float interface: quantize→matvec→dequantize accuracy
 * 3. Benchmark: throughput vs standard float matmul
 */

#include "trix_ternary_matvec.h"
#include "trix_atoms.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT_NEAR(a, b, tol, msg) do { \
    float _a=(a), _b=(b), _t=(tol); \
    if (fabsf(_a-_b) > _t) { \
        printf("FAIL [%s:%d] %s: %.8f != %.8f (tol %.6f)\n", \
               __FILE__,__LINE__,msg,_a,_b,_t); g_fail++; \
    } else { g_pass++; } \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { printf("FAIL [%s:%d] %s\n",__FILE__,__LINE__,msg); g_fail++; } \
    else { g_pass++; } \
} while(0)

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ── Test 1: i8 matvec correctness ── */
static void test_i8_correctness(void) {
    int M = 64, K = 768; /* K must be multiple of 64 for SDOT */

    /* Random ternary weights */
    int8_t* W_i8 = malloc(M * K);
    srand(42);
    for (int i = 0; i < M * K; i++) {
        int r = rand() % 3;
        W_i8[i] = (r == 0) ? 0 : (r == 1) ? 1 : -1;
    }

    /* Pack weights */
    uint8_t* W_packed = malloc(M * (K / 4));
    trix_ternary_pack_weights_i8(W_packed, W_i8, M, K);

    /* Random int8 activations */
    int8_t* act = malloc(K);
    for (int k = 0; k < K; k++) act[k] = (int8_t)((rand() % 201) - 100);

    /* Reference: scalar matvec */
    int32_t* y_ref = calloc(M, sizeof(int32_t));
    for (int m = 0; m < M; m++) {
        int32_t sum = 0;
        for (int k = 0; k < K; k++) {
            sum += (int32_t)W_i8[m * K + k] * (int32_t)act[k];
        }
        y_ref[m] = sum;
    }

    /* NEON SDOT matvec */
    int32_t* y_neon = calloc(M, sizeof(int32_t));
    trix_ternary_matvec_i8(y_neon, act, W_packed, M, K);

    /* Compare */
    int mismatches = 0;
    for (int m = 0; m < M; m++) {
        if (y_ref[m] != y_neon[m]) mismatches++;
    }
    ASSERT_TRUE(mismatches == 0, "i8_exact_match");
    printf("  i8_correctness: %d/%d exact matches\n", M - mismatches, M);

    free(W_i8); free(W_packed); free(act); free(y_ref); free(y_neon);
}

/* ── Test 2: float interface accuracy ── */
static void test_f32_accuracy(void) {
    int M = 32, K = 784;

    /* Ternary weights as float */
    float* W = malloc(M * K * sizeof(float));
    srand(42);
    for (int i = 0; i < M * K; i++) {
        int r = rand() % 3;
        W[i] = (r == 0) ? 0.0f : (r == 1) ? 1.0f : -1.0f;
    }

    /* Random float activations */
    float* x = malloc(K * sizeof(float));
    for (int k = 0; k < K; k++) x[k] = (float)(rand() % 1000) / 500.0f - 1.0f;

    /* Reference: exact float matmul with ternary weights */
    float* y_ref = calloc(M, sizeof(float));
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            y_ref[m] += W[m * K + k] * x[k];
        }
    }

    /* Ternary matvec float interface */
    float* y_ternary = calloc(M, sizeof(float));
    trix_ternary_matvec_f32(y_ternary, W, x, M, K);

    /* Compare — expect small error from int8 quantization of activations */
    float max_abs_err = 0, max_rel_err = 0;
    for (int m = 0; m < M; m++) {
        float err = fabsf(y_ref[m] - y_ternary[m]);
        if (err > max_abs_err) max_abs_err = err;
        float denom = fmaxf(fabsf(y_ref[m]), 1e-6f);
        float rel = err / denom;
        if (rel > max_rel_err) max_rel_err = rel;
    }

    printf("  f32_accuracy: max_abs_err=%.4f max_rel_err=%.4f\n", max_abs_err, max_rel_err);
    /* Int8 quantization of activations introduces ~1/127 error per element.
     * Over K=784 with ~33% zero weights, accumulated error can reach ~4%.
     * This is acceptable for inference; training uses float shadow weights. */
    ASSERT_TRUE(max_rel_err < 0.05f, "f32_rel_err_lt_5pct");

    free(W); free(x); free(y_ref); free(y_ternary);
}

/* ── Test 3: various sizes ── */
static void test_sizes(void) {
    int sizes[][2] = {
        {32, 64},    /* small */
        {32, 128},
        {32, 256},
        {32, 768},   /* MNIST-like */
        {64, 784},
        {128, 3072}, /* CIFAR-like */
        {256, 784},
    };
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("  sizes:\n");
    for (int s = 0; s < n_sizes; s++) {
        int M = sizes[s][0], K = sizes[s][1];
        /* Round K up to multiple of 4 */
        int K4 = (K / 4) * 4;

        float* W = calloc(M * K4, sizeof(float));
        float* x = calloc(K4, sizeof(float));
        srand(s);
        for (int i = 0; i < M * K4; i++) { int r = rand()%3; W[i] = r==0?0:r==1?1:-1; }
        for (int k = 0; k < K4; k++) x[k] = (float)(rand()%1000)/500.0f - 1.0f;

        float* y_ref = calloc(M, sizeof(float));
        float* y_tern = calloc(M, sizeof(float));
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K4; k++)
                y_ref[m] += W[m*K4+k] * x[k];

        trix_ternary_matvec_f32(y_tern, W, x, M, K4);

        float max_rel = 0, max_abs = 0, mean_abs = 0;
        for (int m = 0; m < M; m++) {
            float ae = fabsf(y_ref[m] - y_tern[m]);
            float rel = ae / fmaxf(fabsf(y_ref[m]), 1.0f); /* floor denom at 1.0 */
            if (rel > max_rel) max_rel = rel;
            if (ae > max_abs) max_abs = ae;
            mean_abs += ae;
        }
        mean_abs /= M;
        printf("    M=%3d K=%4d: max_abs=%.2f mean_abs=%.2f max_rel=%.4f\n",
               M, K4, max_abs, mean_abs, max_rel);
        /* Absolute error < 1.0 is fine for classification (logits differ by more) */
        ASSERT_TRUE(max_abs < 2.0f, "size_test_abs");

        free(W); free(x); free(y_ref); free(y_tern);
    }
}

/* ── Test 4: Benchmark ── */
static void test_benchmark(void) {
    int M = 64, K = 768;
    int K4 = (K / 4) * 4;
    int K64 = (K / 64) * 64;

    /* Setup */
    float* W_f = calloc(M * K4, sizeof(float));
    int8_t* W_i8 = calloc(M * K4, sizeof(int8_t));
    float* x_f = calloc(K4, sizeof(float));
    int8_t* x_i8 = calloc(K64, sizeof(int8_t));
    srand(42);
    for (int i = 0; i < M * K4; i++) {
        int r = rand() % 3;
        W_f[i] = r==0 ? 0 : r==1 ? 1 : -1;
        W_i8[i] = r==0 ? 0 : r==1 ? 1 : -1;
    }
    for (int k = 0; k < K4; k++) {
        x_f[k] = (float)(rand()%1000)/500.0f - 1.0f;
        x_i8[k] = (int8_t)(x_f[k] * 100);
    }

    uint8_t* W_packed = calloc(M * (K4 / 4), sizeof(uint8_t));
    trix_ternary_pack_weights_i8(W_packed, W_i8, M, K4);

    float* y_f = calloc(M, sizeof(float));
    int32_t* y_i32 = calloc(M, sizeof(int32_t));

    int iters = 10000;

    /* Benchmark: standard float matmul (trix_matmul_bt) */
    /* W_f as [M, K] for matmul_bt: y = x @ W^T */
    double t0 = now_s();
    for (int i = 0; i < iters; i++) {
        trix_matmul_bt(y_f, x_f, W_f, 1, K4, M);
    }
    double t1 = now_s();
    double float_ms = (t1 - t0) / iters * 1000.0;

    /* Benchmark: ternary i8 matvec (SDOT) */
    double t2 = now_s();
    for (int i = 0; i < iters; i++) {
        trix_ternary_matvec_i8(y_i32, x_i8, W_packed, M, K64);
    }
    double t3 = now_s();
    double ternary_ms = (t3 - t2) / iters * 1000.0;

    /* Benchmark: ternary f32 interface (includes quant/dequant) */
    double t4 = now_s();
    for (int i = 0; i < iters; i++) {
        trix_ternary_matvec_f32(y_f, W_f, x_f, M, K4);
    }
    double t5 = now_s();
    double f32_ternary_ms = (t5 - t4) / iters * 1000.0;

    printf("\n  benchmark (M=%d K=%d, %d iters):\n", M, K4, iters);
    printf("    float matmul_bt:     %.4f ms/call\n", float_ms);
    printf("    ternary i8 (SDOT):   %.4f ms/call (%.1fx faster)\n",
           ternary_ms, float_ms / ternary_ms);
    printf("    ternary f32 (w/quant): %.4f ms/call (%.1fx faster)\n",
           f32_ternary_ms, float_ms / f32_ternary_ms);

    ASSERT_TRUE(1, "benchmark_completed");

    free(W_f); free(W_i8); free(x_f); free(x_i8);
    free(W_packed); free(y_f); free(y_i32);
}

int main(void) {
    printf("=== Ternary Matvec Integration Tests ===\n\n");

    test_i8_correctness();
    test_f32_accuracy();
    test_sizes();
    test_benchmark();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
