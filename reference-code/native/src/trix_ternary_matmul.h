/*
 * trix_ternary_matmul.h — Float activations × ternary weights, multiply-free
 *
 * Y[M,N] = X[M,K] @ W_tern[N,K]^T
 *
 * W_tern entries are {-1, 0, +1} stored as int8_t.
 * Computation: add, subtract, or skip. Zero multiplies. Zero quantization.
 * Activations stay float32 throughout.
 */

#ifndef TRIX_TERNARY_MATMUL_H
#define TRIX_TERNARY_MATMUL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Y[M, N] = X[M, K] @ W[N, K]^T
 *
 * X: [M, K] float32 activations (row-major)
 * W: [N, K] int8_t ternary {-1, 0, +1} (row-major)
 * Y: [M, N] float32 output (row-major)
 *
 * Each output element: y[i,j] = sum_k(x[i,k] * w[j,k])
 *                             = sum of x[i,k] where w[j,k]=+1
 *                             - sum of x[i,k] where w[j,k]=-1
 */
void trix_ternary_matmul_bt(
    float* Y, const float* X, const int8_t* W,
    int M, int K, int N
);

/*
 * Y[M, N] = X[M, K] @ W[K, N]
 *
 * For backward dx: dy[batch, D] @ W1_tern[H, D]^T → not needed if we use _bt
 * For backward dx through W1: dz[batch, H] @ W1[H, D] → Y = dz @ W1
 *   M=batch, K=H, N=D, W is [H, D] = [K, N]
 */
void trix_ternary_matmul(
    float* Y, const float* X, const int8_t* W,
    int M, int K, int N
);

#ifdef __cplusplus
}
#endif

#endif /* TRIX_TERNARY_MATMUL_H */
