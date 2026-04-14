/*
 * trix_neon.h — NEON SIMD intrinsics for trix-ffn-v2.
 *
 * Provides optimized operations for:
 * - LayerNorm (mean, variance, normalize)
 * - Activations (GELU, Tanh, Sigmoid, ReLU)
 * - Dot products (float32, int8 ternary)
 * - Popcount distance (uint8 packed)
 * - Matrix multiply (float32)
 * - Utilities (load, store, reduce)
 *
 * Uses scalar fallbacks - works on any platform.
 */

#ifndef TRIX_NEON_H
#define TRIX_NEON_H

#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Configuration
 * ============================================================================ */

#define TRIX_NEON_INLINE static inline
#define TRIX_NEON_UNUSED __attribute__((unused))

/* Detect NEON availability */
#if defined(__ARM_NEON) || defined(__aarch64__)
#define TRIX_HAS_NEON 1
#include <arm_neon.h>
#else
#define TRIX_HAS_NEON 0
#endif

/* ============================================================================
 * Constants
 * ============================================================================ */

TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_f32_const(float v) { return v; }

/* ============================================================================
 * Basic Arithmetic (Scalar Fallbacks - work everywhere)
 * ============================================================================ */

TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_f32_add(float a, float b) { return a + b; }
TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_f32_sub(float a, float b) { return a - b; }
TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_f32_mul(float a, float b) { return a * b; }
TRIX_NEON_INLINE float trix_f32_div(float a, float b) { return a / b; }
TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_f32_neg(float a) { return -a; }
TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_f32_abs(float a) { return fabsf(a); }

TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_f32_max(float a, float b) { return a > b ? a : b; }
TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_f32_min(float a, float b) { return a < b ? a : b; }

/* ============================================================================
 * Activations (Scalar)
 * ============================================================================ */

TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_f32_relu(float x) { return x > 0.0f ? x : 0.0f; }

TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_f32_gelu(float x) {
    /* GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    float c0 = 0.7978845608028654f;  /* sqrt(2/pi) */
    float c1 = 0.044715f;
    float x3 = x * x * x;
    float t = c0 * (x + c1 * x3);
    return 0.5f * x * (1.0f + tanhf(t));
}

TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_f32_gelu_fast(float x) {
    return x * (1.0f / (1.0f + expf(-1.702f * x)));
}

TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_f32_tanh(float x) { return tanhf(x); }

TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_f32_sigmoid(float x) {
    /* Clamp to prevent overflow */
    if (x < -6.0f) x = -6.0f;
    if (x > 6.0f) x = 6.0f;
    return 1.0f / (1.0f + expf(-x));
}

TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_f32_silu(float x) {
    return x * trix_f32_sigmoid(x);
}

/* ============================================================================
 * Vector Math (Array-based for any size)
 * ============================================================================ */

/* ReLU: y = max(0, x) */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_vec_relu(float* x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

/* GELU in-place */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_vec_gelu(float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = trix_f32_gelu(x[i]);
    }
}

/* Tanh in-place */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_vec_tanh(float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = tanhf(x[i]);
    }
}

/* Sigmoid in-place */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_vec_sigmoid(float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = trix_f32_sigmoid(x[i]);
    }
}

/* ============================================================================
 * LayerNorm
 * ============================================================================ */

/* Compute mean and variance for a single vector */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_mean_var_f32(
    const float* x,     /* [d] */
    int d,
    float* mean_out,
    float* var_out)
{
    double sum = 0.0;
    for (int i = 0; i < d; i++) {
        sum += x[i];
    }
    float mean = (float)(sum / d);
    
    double m2 = 0.0;
    for (int i = 0; i < d; i++) {
        float diff = x[i] - mean;
        m2 += diff * diff;
    }
    
    *mean_out = mean;
    *var_out = (float)(m2 / d);
}

/* NEON-optimized mean and variance */
#if TRIX_HAS_NEON
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_mean_var_f32_neon(
    const float* x,     /* [d] */
    int d,
    float* mean_out,
    float* var_out)
{
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    float32x4_t sum_sq_vec = vdupq_n_f32(0.0f);
    
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        float32x4_t x0 = vld1q_f32(x + i);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t x2 = vld1q_f32(x + i + 8);
        float32x4_t x3 = vld1q_f32(x + i + 12);
        
        sum_vec = vaddq_f32(sum_vec, x0);
        sum_vec = vaddq_f32(sum_vec, x1);
        sum_vec = vaddq_f32(sum_vec, x2);
        sum_vec = vaddq_f32(sum_vec, x3);
        
        sum_sq_vec = vmlaq_f32(sum_sq_vec, x0, x0);
        sum_sq_vec = vmlaq_f32(sum_sq_vec, x1, x1);
        sum_sq_vec = vmlaq_f32(sum_sq_vec, x2, x2);
        sum_sq_vec = vmlaq_f32(sum_sq_vec, x3, x3);
    }
    for (; i + 8 <= d; i += 8) {
        float32x4_t x0 = vld1q_f32(x + i);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        sum_vec = vaddq_f32(sum_vec, x0);
        sum_vec = vaddq_f32(sum_vec, x1);
        sum_sq_vec = vmlaq_f32(sum_sq_vec, x0, x0);
        sum_sq_vec = vmlaq_f32(sum_sq_vec, x1, x1);
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t x0 = vld1q_f32(x + i);
        sum_vec = vaddq_f32(sum_vec, x0);
        sum_sq_vec = vmlaq_f32(sum_sq_vec, x0, x0);
    }
    
    float sum = vaddvq_f32(sum_vec);
    float sum_sq = vaddvq_f32(sum_sq_vec);
    
    for (; i < d; i++) {
        sum += x[i];
        sum_sq += x[i] * x[i];
    }
    
    float mean = sum / d;
    *mean_out = mean;
    *var_out = (sum_sq / d) - (mean * mean);
}
#endif

/* LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_layernorm_f32(
    const float* input,     /* [d] */
    float* output,         /* [d] */
    const float* weight,   /* [d] */
    const float* bias,     /* [d] */
    float eps,
    int d)
{
    float mean, var;
#if TRIX_HAS_NEON
    trix_mean_var_f32_neon(input, d, &mean, &var);
#else
    trix_mean_var_f32(input, d, &mean, &var);
#endif
    
    float inv_std = 1.0f / sqrtf(var + eps);
    
    /* NEON-optimized normalize + scale + bias */
#if TRIX_HAS_NEON
    float32x4_t mean_vec = vdupq_n_f32(mean);
    float32x4_t inv_std_vec = vdupq_n_f32(inv_std);
    
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        float32x4_t x0 = vld1q_f32(input + i);
        float32x4_t x1 = vld1q_f32(input + i + 4);
        float32x4_t x2 = vld1q_f32(input + i + 8);
        float32x4_t x3 = vld1q_f32(input + i + 12);
        
        float32x4_t w0 = vld1q_f32(weight + i);
        float32x4_t w1 = vld1q_f32(weight + i + 4);
        float32x4_t w2 = vld1q_f32(weight + i + 8);
        float32x4_t w3 = vld1q_f32(weight + i + 12);
        
        float32x4_t b0 = vld1q_f32(bias + i);
        float32x4_t b1 = vld1q_f32(bias + i + 4);
        float32x4_t b2 = vld1q_f32(bias + i + 8);
        float32x4_t b3 = vld1q_f32(bias + i + 12);
        
        x0 = vsubq_f32(x0, mean_vec);
        x1 = vsubq_f32(x1, mean_vec);
        x2 = vsubq_f32(x2, mean_vec);
        x3 = vsubq_f32(x3, mean_vec);
        
        x0 = vmulq_f32(x0, inv_std_vec);
        x1 = vmulq_f32(x1, inv_std_vec);
        x2 = vmulq_f32(x2, inv_std_vec);
        x3 = vmulq_f32(x3, inv_std_vec);
        
        x0 = vmlaq_f32(b0, x0, w0);
        x1 = vmlaq_f32(b1, x1, w1);
        x2 = vmlaq_f32(b2, x2, w2);
        x3 = vmlaq_f32(b3, x3, w3);
        
        vst1q_f32(output + i, x0);
        vst1q_f32(output + i + 4, x1);
        vst1q_f32(output + i + 8, x2);
        vst1q_f32(output + i + 12, x3);
    }
    for (; i + 8 <= d; i += 8) {
        float32x4_t x0 = vld1q_f32(input + i);
        float32x4_t x1 = vld1q_f32(input + i + 4);
        float32x4_t w0 = vld1q_f32(weight + i);
        float32x4_t w1 = vld1q_f32(weight + i + 4);
        float32x4_t b0 = vld1q_f32(bias + i);
        float32x4_t b1 = vld1q_f32(bias + i + 4);
        
        x0 = vsubq_f32(x0, mean_vec);
        x1 = vsubq_f32(x1, mean_vec);
        x0 = vmulq_f32(x0, inv_std_vec);
        x1 = vmulq_f32(x1, inv_std_vec);
        x0 = vmlaq_f32(b0, x0, w0);
        x1 = vmlaq_f32(b1, x1, w1);
        
        vst1q_f32(output + i, x0);
        vst1q_f32(output + i + 4, x1);
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t x0 = vld1q_f32(input + i);
        float32x4_t w0 = vld1q_f32(weight + i);
        float32x4_t b0 = vld1q_f32(bias + i);
        
        x0 = vsubq_f32(x0, mean_vec);
        x0 = vmulq_f32(x0, inv_std_vec);
        x0 = vmlaq_f32(b0, x0, w0);
        
        vst1q_f32(output + i, x0);
    }
    for (; i < d; i++) {
        float norm = (input[i] - mean) * inv_std;
        output[i] = norm * weight[i] + bias[i];
    }
#else
    for (int i = 0; i < d; i++) {
        float norm = (input[i] - mean) * inv_std;
        output[i] = norm * weight[i] + bias[i];
    }
#endif
}

/* Batch LayerNorm: apply layernorm to N rows */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_layernorm_batch_f32(
    const float* input,     /* [N, d] row-major */
    float* output,          /* [N, d] row-major */
    const float* weight,    /* [d] */
    const float* bias,      /* [d] */
    float eps,
    int N, int d)
{
    for (int row = 0; row < N; row++) {
        trix_layernorm_f32(
            input + row * d,
            output + row * d,
            weight, bias, eps, d
        );
    }
}

/* ============================================================================
 * Dot Products (Float32)
 * ============================================================================ */

/* Dot product of two float32 vectors, any length */
TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_dot_f32(
    const float* a, 
    const float* b, 
    int n)
{
#if TRIX_HAS_NEON
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);
        
        sum_vec = vmlaq_f32(sum_vec, a0, b0);
        sum_vec = vmlaq_f32(sum_vec, a1, b1);
        sum_vec = vmlaq_f32(sum_vec, a2, b2);
        sum_vec = vmlaq_f32(sum_vec, a3, b3);
    }
    for (; i + 8 <= n; i += 8) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        
        sum_vec = vmlaq_f32(sum_vec, a0, b0);
        sum_vec = vmlaq_f32(sum_vec, a1, b1);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        sum_vec = vmlaq_f32(sum_vec, a0, b0);
    }
    
    float sum = vaddvq_f32(sum_vec);
    
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)a[i] * (double)b[i];
    }
    return (float)sum;
#endif
}

/* ============================================================================
 * Dot Products (Int8 Ternary)
 * ============================================================================ */

#if TRIX_HAS_NEON && defined(__ARM_FEATURE_DOTPROD)

/* Dot product of two int8 ternary vectors using dot product instruction */
TRIX_NEON_UNUSED TRIX_NEON_INLINE int32_t trix_dot_i8_ternary(
    const int8_t* a, 
    const int8_t* b, 
    int n) 
{
    int i = 0;
    int32x4_t acc = vdupq_n_s32(0);
    
    for (; i + 16 <= n; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);
        acc = vdotq_s32(acc, va, vb);
    }
    
    int32_t sum = vaddvq_s32(acc);
    
    for (; i < n; i++) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }
    
    return sum;
}

#else

/* Fallback without dot product instruction */
TRIX_NEON_UNUSED TRIX_NEON_INLINE int32_t trix_dot_i8_ternary(
    const int8_t* a, 
    const int8_t* b, 
    int n) 
{
    int32_t sum = 0;
    for (int i = 0; i < n; i++) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }
    return sum;
}

#endif

/* ============================================================================
 * Popcount Distance (Packed Ternary)
 * ============================================================================ */

/* Popcount distance between two packed ternary vectors */
TRIX_NEON_UNUSED TRIX_NEON_INLINE uint32_t trix_popcount_dist(
    const uint8_t* a,
    const uint8_t* b,
    const uint8_t* mask,
    int packed_dim)
{
    uint32_t sum = 0;
    for (int i = 0; i < packed_dim; i++) {
        uint8_t x = (a[i] ^ b[i]) & mask[i];
        sum += __builtin_popcount((int)x);
    }
    return sum;
}

/* Without mask (all ones) */
TRIX_NEON_UNUSED TRIX_NEON_INLINE uint32_t trix_popcount_dist_nomask(
    const uint8_t* a,
    const uint8_t* b,
    int packed_dim)
{
    uint32_t sum = 0;
    for (int i = 0; i < packed_dim; i++) {
        sum += __builtin_popcount((int)(a[i] ^ b[i]));
    }
    return sum;
}

/* ============================================================================
 * Batch Popcount Distance
 * ============================================================================ */

/* Compute popcount distance from one query to N signatures */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_batch_popcount_dist(
    const uint8_t* query_packed,      /* [packed_dim] */
    const uint8_t* signatures_packed, /* [num_sigs, packed_dim] */
    uint32_t* dist_out,              /* [num_sigs] */
    int num_sigs,
    int packed_dim)
{
    for (int sig = 0; sig < num_sigs; sig++) {
        dist_out[sig] = trix_popcount_dist_nomask(
            query_packed,
            signatures_packed + sig * packed_dim,
            packed_dim
        );
    }
}

/* ============================================================================
 * Ternary Quantization
 * ============================================================================ */

/* Quantize float32 to ternary {-1, 0, +1} */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_quantize_ternary(
    const float* input,    /* [d] */
    int8_t* output,        /* [d] ternary */
    int d,
    float threshold)      /* default 0.3f */
{
    for (int i = 0; i < d; i++) {
        if (input[i] > threshold) output[i] = 1;
        else if (input[i] < -threshold) output[i] = -1;
        else output[i] = 0;
    }
}

/* ============================================================================
 * Ternary Packing
 * ============================================================================ */

/* Pack ternary int8 to 2-bit uint8 */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_pack_ternary(
    const int8_t* src,    /* [d] ternary */
    uint8_t* dst,         /* [packed_dim = (d+3)/4] */
    int d)
{
    int packed_dim = (d + 3) / 4;
    
    for (int i = 0; i < packed_dim; i++) {
        uint8_t code = 0;
        for (int j = 0; j < 4 && i * 4 + j < d; j++) {
            int8_t v = src[i * 4 + j];
            uint8_t c;
            if (v == 1) c = 0x01;
            else if (v == -1) c = 0x02;
            else c = 0x00;
            code |= (c << (j * 2));
        }
        dst[i] = code;
    }
}

/* Unpack 2-bit uint8 to ternary int8 */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_unpack_ternary(
    const uint8_t* src,    /* [packed_dim] */
    int8_t* dst,          /* [d] */
    int d)
{
    for (int i = 0; i < d; i++) {
        uint8_t code = (src[i / 4] >> ((i % 4) * 2)) & 0x03;
        if (code == 0x01) dst[i] = 1;
        else if (code == 0x02) dst[i] = -1;
        else dst[i] = 0;
    }
}

/* ============================================================================
 * Argmax / Argmin
 * ============================================================================ */

/* Argmax of float array */
TRIX_NEON_UNUSED TRIX_NEON_INLINE int trix_argmax_f32(const float* arr, int n) {
    if (n == 0) return -1;
    
    float max_val = arr[0];
    int max_idx = 0;
    
    for (int i = 1; i < n; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

/* Argmin of float array */
TRIX_NEON_UNUSED TRIX_NEON_INLINE int trix_argmin_f32(const float* arr, int n) {
    if (n == 0) return -1;
    
    float min_val = arr[0];
    int min_idx = 0;
    
    for (int i = 1; i < n; i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
            min_idx = i;
        }
    }
    
    return min_idx;
}

/* Argmax with value output */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_argmax_f32_with_val(
    const float* arr, 
    int n, 
    int* idx_out, 
    float* val_out)
{
    if (n == 0) {
        *idx_out = -1;
        *val_out = 0.0f;
        return;
    }
    
    float max_val = arr[0];
    int max_idx = 0;
    
    for (int i = 1; i < n; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    
    *idx_out = max_idx;
    *val_out = max_val;
}

/* ============================================================================
 * Matrix Multiply (Float32)
 * ============================================================================ */

/* Matrix multiply: C = alpha * A * B + beta * C
 * A: [m, k], B: [k, n], C: [m, n]
 */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_gemm_f32(
    const float* A,   /* [m, k] */
    const float* B,   /* [k, n] */
    float* C,         /* [m, n] */
    int m, int k, int n,
    float alpha,
    float beta)
{
#if TRIX_HAS_NEON
    /* NEON-optimized GEMM: 4x4 block tiling */
    for (int i = 0; i < m; i += 4) {
        for (int j = 0; j < n; j += 4) {
            float32x4_t c0 = vdupq_n_f32(0.0f);
            float32x4_t c1 = vdupq_n_f32(0.0f);
            float32x4_t c2 = vdupq_n_f32(0.0f);
            float32x4_t c3 = vdupq_n_f32(0.0f);
            
            for (int p = 0; p < k; p++) {
                float32x4_t a = vdupq_n_f32(A[i * k + p]);
                float32x4_t b = vld1q_f32(B + p * n + j);
                c0 = vmlaq_f32(c0, a, b);
                
                if (i + 1 < m) {
                    a = vdupq_n_f32(A[(i + 1) * k + p]);
                    c1 = vmlaq_f32(c1, a, b);
                }
                if (i + 2 < m) {
                    a = vdupq_n_f32(A[(i + 2) * k + p]);
                    c2 = vmlaq_f32(c2, a, b);
                }
                if (i + 3 < m) {
                    a = vdupq_n_f32(A[(i + 3) * k + p]);
                    c3 = vmlaq_f32(c3, a, b);
                }
            }
            
            if (beta == 0.0f) {
                if (i < m) vst1q_f32(C + i * n + j, c0 * alpha);
                if (i + 1 < m) vst1q_f32(C + (i + 1) * n + j, c1 * alpha);
                if (i + 2 < m) vst1q_f32(C + (i + 2) * n + j, c2 * alpha);
                if (i + 3 < m) vst1q_f32(C + (i + 3) * n + j, c3 * alpha);
            } else {
                if (i < m) vst1q_f32(C + i * n + j, vmlaq_n_f32(c0, vld1q_f32(C + i * n + j), beta) * alpha);
                if (i + 1 < m) vst1q_f32(C + (i + 1) * n + j, vmlaq_n_f32(c1, vld1q_f32(C + (i + 1) * n + j), beta) * alpha);
                if (i + 2 < m) vst1q_f32(C + (i + 2) * n + j, vmlaq_n_f32(c2, vld1q_f32(C + (i + 2) * n + j), beta) * alpha);
                if (i + 3 < m) vst1q_f32(C + (i + 3) * n + j, vmlaq_n_f32(c3, vld1q_f32(C + (i + 3) * n + j), beta) * alpha);
            }
        }
    }
#else
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            
            if (beta == 0.0f) {
                C[i * n + j] = alpha * sum;
            } else {
                C[i * n + j] = alpha * sum + beta * C[i * n + j];
            }
        }
    }
#endif
}

/* ============================================================================
 * Softmax
 * ============================================================================ */

/* Softmax: exp(x_i) / sum(exp(x_j)) */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_softmax_f32(
    float* input_output,  /* [n] */
    int n,
    float temperature)    /* default 1.0f */
{
    if (n == 0) return;
    
    /* Find max for numerical stability */
    float max_val = input_output[0];
    for (int i = 1; i < n; i++) {
        if (input_output[i] > max_val) max_val = input_output[i];
    }
    
    /* Compute exp(x - max) and sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        input_output[i] = expf((input_output[i] - max_val) / temperature);
        sum += input_output[i];
    }
    
    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        input_output[i] *= inv_sum;
    }
}

/* ============================================================================
 * Spline Interpolation (for Score Calibration)
 * ============================================================================ */

/* Linear interpolation between two knots */
TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_spline_interp_linear(
    float x,
    const float* knots,
    int num_knots)
{
    float idx_float = x * (num_knots - 1);
    int idx_low = (int)idx_float;
    if (idx_low >= num_knots - 1) idx_low = num_knots - 2;
    if (idx_low < 0) idx_low = 0;
    
    float t = idx_float - idx_low;
    
    return knots[idx_low] + t * (knots[idx_low + 1] - knots[idx_low]);
}

/* Apply score calibration spline */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_calibrate_scores(
    const float* raw_scores,  /* [batch] */
    float* calibrated_out,    /* [batch] */
    const float* knots,      /* [num_knots] */
    int num_knots,
    float temperature,
    int batch)
{
    for (int i = 0; i < batch; i++) {
        float normalized = 1.0f / (1.0f + expf(-raw_scores[i] / temperature));
        calibrated_out[i] = trix_spline_interp_linear(normalized, knots, num_knots);
    }
}

/* ============================================================================
 * Regularizers
 * ============================================================================ */

/* Balance loss: encourage uniform tile usage */
TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_balance_loss(
    const int32_t* tile_counts,
    int32_t num_tiles,
    int64_t total_count)
{
    if (total_count == 0) return 0.0f;
    
    float ideal = (float)total_count / num_tiles;
    float sum_sq_diff = 0.0f;
    
    for (int i = 0; i < num_tiles; i++) {
        float diff = tile_counts[i] - ideal;
        sum_sq_diff += diff * diff;
    }
    
    return (sum_sq_diff / num_tiles) / (ideal * ideal + 1e-8f);
}

/* Ternary loss: distance to nearest ternary value */
TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_ternary_loss(
    const int8_t* signatures,
    int32_t num_tiles,
    int32_t d_model)
{
    float total_dist = 0.0f;
    
    for (int i = 0; i < num_tiles; i++) {
        for (int j = 0; j < d_model; j++) {
            int8_t v = signatures[i * d_model + j];
            float dist;
            if (v == 1) dist = 0.0f;
            else if (v == -1) dist = 0.0f;
            else dist = 1.0f;  /* distance to nearest ternary (1 or -1) */
            total_dist += dist;
        }
    }
    
    return total_dist / (num_tiles * d_model);
}

/* Sparsity loss: penalize too many non-zeros */
TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_sparsity_loss(
    const int8_t* signatures,
    int32_t num_tiles,
    int32_t d_model,
    float target_nonzero_frac)
{
    int32_t total_nonzero = 0;
    int32_t target_nonzero = (int32_t)(target_nonzero_frac * d_model);
    
    for (int i = 0; i < num_tiles; i++) {
        for (int j = 0; j < d_model; j++) {
            if (signatures[i * d_model + j] != 0) {
                total_nonzero++;
            }
        }
    }
    
    float avg_nonzero = (float)total_nonzero / num_tiles;
    float excess = avg_nonzero - target_nonzero;
    if (excess < 0) excess = 0;
    
    return excess / (d_model + 1e-8f);
}

/* Diversity loss: penalize high similarity between signatures */
TRIX_NEON_UNUSED TRIX_NEON_INLINE float trix_diversity_loss(
    const int8_t* signatures,
    int32_t num_tiles,
    int32_t d_model)
{
    float total_sim = 0.0f;
    int32_t num_pairs = 0;
    
    for (int i = 0; i < num_tiles; i++) {
        for (int j = i + 1; j < num_tiles; j++) {
            int32_t dot = 0;
            for (int d = 0; d < d_model; d++) {
                dot += (int32_t)signatures[i * d_model + d] * (int32_t)signatures[j * d_model + d];
            }
            float sim = (float)dot / d_model;
            if (sim > 0.5f) {
                total_sim += (sim - 0.5f);
            }
            num_pairs++;
        }
    }
    
    if (num_pairs == 0) return 0.0f;
    return total_sim / num_pairs;
}

/* ============================================================================
 * Policy Enforcement
 * ============================================================================ */

/* Check if tile is allowed by policy */
TRIX_NEON_UNUSED TRIX_NEON_INLINE bool trix_policy_is_allowed(
    const int32_t* allow_tiles,    /* [num_allowed] or NULL */
    int32_t num_allowed,
    const int32_t* deny_tiles,     /* [num_denied] */
    int32_t num_denied,
    int32_t tile_idx)
{
    /* Check deny list first */
    for (int i = 0; i < num_denied; i++) {
        if (deny_tiles[i] == tile_idx) return false;
    }
    
    /* If allow list exists, check it */
    if (allow_tiles != NULL) {
        for (int i = 0; i < num_allowed; i++) {
            if (allow_tiles[i] == tile_idx) return true;
        }
        return false;
    }
    
    /* No allow list means allow everything not denied */
    return true;
}

/* Apply policy: set disallowed tiles to -inf in scores */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_policy_apply(
    float* scores,                 /* [num_tiles] */
    const int32_t* allow_tiles,   /* [num_allowed] or NULL */
    int32_t num_allowed,
    const int32_t* deny_tiles,    /* [num_denied] */
    int32_t num_denied,
    int32_t num_tiles)
{
    for (int i = 0; i < num_tiles; i++) {
        if (!trix_policy_is_allowed(allow_tiles, num_allowed, deny_tiles, num_denied, i)) {
            scores[i] = -INFINITY;
        }
    }
}

/* ============================================================================
 * Batch Operations
 * ============================================================================ */

/* Batch argmax: find argmax for each row */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_batch_argmax(
    const float* scores_batch,   /* [batch, num_tiles] */
    int* indices_out,            /* [batch] */
    int batch,
    int num_tiles)
{
    for (int b = 0; b < batch; b++) {
        indices_out[b] = trix_argmax_f32(scores_batch + b * num_tiles, num_tiles);
    }
}

/* Batch routing: find argmax for each sample */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_batch_route(
    const float* x_flat,         /* [batch, d_model] */
    const int8_t* signatures,    /* [num_tiles, d_model] */
    int* tile_indices_out,      /* [batch] */
    int batch,
    int num_tiles,
    int d_model,
    bool use_ternary)           /* true = signatures are ternary */
{
    for (int b = 0; b < batch; b++) {
        float best_score = -INFINITY;
        int best_tile = 0;
        
        for (int t = 0; t < num_tiles; t++) {
            float score;
            if (use_ternary) {
                score = (float)trix_dot_i8_ternary(
                    (const int8_t*)(x_flat + b * d_model),
                    signatures + t * d_model,
                    d_model
                );
            } else {
                score = trix_dot_f32(
                    x_flat + b * d_model,
                    (const float*)(signatures + t * d_model),
                    d_model
                );
            }
            if (score > best_score) {
                best_score = score;
                best_tile = t;
            }
        }
        
        tile_indices_out[b] = best_tile;
    }
}

/* ============================================================================
 * Memory Operations
 * ============================================================================ */

/* Zero array */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_zero_f32(float* arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = 0.0f;
}

TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_zero_i32(int32_t* arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = 0;
}

/* Copy array */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_copy_f32(const float* src, float* dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = src[i];
}

TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_copy_i32(const int32_t* src, int32_t* dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = src[i];
}

/* Scale array */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_scale_f32(float* arr, float scale, int n) {
    for (int i = 0; i < n; i++) arr[i] *= scale;
}

/* Add arrays: dst += src */
TRIX_NEON_UNUSED TRIX_NEON_INLINE void trix_add_f32(const float* src, float* dst, int n) {
    for (int i = 0; i < n; i++) dst[i] += src[i];
}

#ifdef __cplusplus
}
#endif

#endif /* TRIX_NEON_H */
