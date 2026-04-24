/*
 * backward_linear.c — scalar implementation.
 *
 * These are the straight-line reference kernels. NEON ports come later
 * once gradient checks pass on the scalar path. The scalar path remains
 * as the correctness oracle for every future vectorized variant.
 */

#include "backward_linear.h"
#include <stddef.h>

void tlinear_forward(
    float* Y,
    const float* X,
    const int8_t* W,
    int M, int K, int N)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            const float*  x_row = X + (size_t)m * K;
            const int8_t* w_row = W + (size_t)n * K;
            for (int k = 0; k < K; k++) {
                acc += x_row[k] * (float)w_row[k];
            }
            Y[(size_t)m * N + n] = acc;
        }
    }
}

void tlinear_backward_dX(
    float* dX,
    const float* dY,
    const int8_t* W,
    int M, int K, int N)
{
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            float acc = 0.0f;
            for (int n = 0; n < N; n++) {
                acc += dY[(size_t)m * N + n] * (float)W[(size_t)n * K + k];
            }
            dX[(size_t)m * K + k] = acc;
        }
    }
}

void tlinear_backward_dW(
    float* dW_latent,
    const float* dY,
    const float* X,
    int M, int K, int N)
{
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            float acc = 0.0f;
            for (int m = 0; m < M; m++) {
                acc += dY[(size_t)m * N + n] * X[(size_t)m * K + k];
            }
            dW_latent[(size_t)n * K + k] += acc;
        }
    }
}
