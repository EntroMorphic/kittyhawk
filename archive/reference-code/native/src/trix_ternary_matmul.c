/*
 * trix_ternary_matmul.c — Float activations × ternary weights, multiply-free
 *
 * Every output element is computed by adding or subtracting float activations
 * based on ternary weight values {-1, 0, +1}. Zero multiplications.
 *
 * NEON implementation: branch-free using vbslq_f32 (bitwise select) to choose
 * between add, subtract, or zero for each weight element. 4 weights processed
 * per NEON iteration.
 */

#include "trix_ternary_matmul.h"
#include <string.h>

#ifdef APPLE
#include <dispatch/dispatch.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define TRIX_HAS_NEON 1
#else
#define TRIX_HAS_NEON 0
#endif

/*
 * Y[M, N] = X[M, K] @ W[N, K]^T
 *
 * Inner loop: for each (i, j), dot X[i,:] with W[j,:] using add/sub/skip.
 * W[j,:] is contiguous — good for streaming.
 * X[i,:] is contiguous — good for streaming.
 * Both operands are contiguous per (i,j) pair: ideal for NEON.
 */
void trix_ternary_matmul_bt(
    float* Y, const float* X, const int8_t* W,
    int M, int K, int N)
{
#if TRIX_HAS_NEON
    for (int i = 0; i < M; i++) {
        const float* xi = X + i * K;
        for (int j = 0; j < N; j++) {
            const int8_t* wj = W + j * K;
            float32x4_t vacc = vdupq_n_f32(0);
            int k = 0;
            for (; k + 4 <= K; k += 4) {
                float32x4_t vx = vld1q_f32(xi + k);
                int8_t w0 = wj[k], w1 = wj[k+1], w2 = wj[k+2], w3 = wj[k+3];
                float32x4_t vw = {(float)w0, (float)w1, (float)w2, (float)w3};
                uint32x4_t pos = vcgtq_f32(vw, vdupq_n_f32(0));
                uint32x4_t neg = vcltq_f32(vw, vdupq_n_f32(0));
                vacc = vaddq_f32(vacc, vbslq_f32(pos, vx, vdupq_n_f32(0)));
                vacc = vsubq_f32(vacc, vbslq_f32(neg, vx, vdupq_n_f32(0)));
            }
            float sum = vaddvq_f32(vacc);
            for (; k < K; k++) {
                if (wj[k] == 1) sum += xi[k];
                else if (wj[k] == -1) sum -= xi[k];
            }
            Y[i * N + j] = sum;
        }
    }
#else
    /* Scalar fallback */
    for (int i = 0; i < M; i++) {
        const float* xi = X + i * K;
        for (int j = 0; j < N; j++) {
            const int8_t* wj = W + j * K;
            float sum = 0;
            for (int k = 0; k < K; k++) {
                if (wj[k] == 1) sum += xi[k];
                else if (wj[k] == -1) sum -= xi[k];
            }
            Y[i * N + j] = sum;
        }
    }
#endif
}

/*
 * Y[M, N] = X[M, K] @ W[K, N]
 *
 * W is [K, N] row-major. For each output (i, j):
 *   y[i,j] = sum_k x[i,k] * W[k,j]
 *
 * W column j is strided (stride = N). Use outer product accumulation instead:
 *   for each k: Y[i,:] += x[i,k] * W[k,:] where W[k,:] is contiguous.
 */
void trix_ternary_matmul(
    float* Y, const float* X, const int8_t* W,
    int M, int K, int N)
{
    memset(Y, 0, (size_t)M * N * sizeof(float));
#if TRIX_HAS_NEON
    for (int i = 0; i < M; i++) {
        const float* xi = X + i * K;
        float* yi = Y + i * N;
        for (int k = 0; k < K; k++) {
            float xik = xi[k];
            if (xik == 0.0f) continue;
            const int8_t* wk = W + k * N;
            float32x4_t vx = vdupq_n_f32(xik);
            int j = 0;
            for (; j + 4 <= N; j += 4) {
                int8_t w0 = wk[j], w1 = wk[j+1], w2 = wk[j+2], w3 = wk[j+3];
                float32x4_t vw = {(float)w0, (float)w1, (float)w2, (float)w3};
                uint32x4_t pos = vcgtq_f32(vw, vdupq_n_f32(0));
                uint32x4_t neg = vcltq_f32(vw, vdupq_n_f32(0));
                float32x4_t vy = vld1q_f32(yi + j);
                vy = vaddq_f32(vy, vbslq_f32(pos, vx, vdupq_n_f32(0)));
                vy = vsubq_f32(vy, vbslq_f32(neg, vx, vdupq_n_f32(0)));
                vst1q_f32(yi + j, vy);
            }
            for (; j < N; j++) {
                if (wk[j] == 1) yi[j] += xik;
                else if (wk[j] == -1) yi[j] -= xik;
            }
        }
    }
#else
    for (int i = 0; i < M; i++) {
        const float* xi = X + i * K;
        float* yi = Y + i * N;
        for (int k = 0; k < K; k++) {
            float xik = xi[k];
            const int8_t* wk = W + k * N;
            for (int j = 0; j < N; j++) {
                if (wk[j] == 1) yi[j] += xik;
                else if (wk[j] == -1) yi[j] -= xik;
            }
        }
    }
#endif
}
