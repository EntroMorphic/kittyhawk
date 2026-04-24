/*
 * backward_linear.h — ternary linear forward + backward (scalar).
 *
 * This is the first kernel pair of the routed autodiff MVP. Intentionally
 * minimal: dense ternary matmul with float activations, scalar loops, no
 * routing yet. Once this pair passes its gradient check, the rest of the
 * MVP (routing dispatch, STE through top-k) layers on top of it.
 *
 * Y[m,n] = Σ_k X[m,k] · W[n,k]        (forward)
 *   where X ∈ float^{M×K}, W ∈ {-1, 0, +1}^{N×K} stored as int8,
 *         Y ∈ float^{M×N}.
 *
 * Backward pass (chain rule under MSE loss shape; the caller supplies dY):
 *   dX[m,k] = Σ_n dY[m,n] · W[n,k]
 *   dW_latent[n,k] += Σ_m dY[m,n] · X[m,k]   (accumulate into float latent)
 *
 * The "latent" naming signals the substrate discipline: W the ternary is
 * what the forward kernel sees; W_latent is the float shadow that SGD
 * updates. A periodic re-quantization step (not in this file) takes
 * W_latent → W via per-dim density-τ thresholding.
 *
 * Substrate position: train/ is consumer-layer per NORTH_STAR §13
 * (training artifacts live in the consumer). Not linked into libm4t.
 */

#ifndef GLYPH_TRAIN_BACKWARD_LINEAR_H
#define GLYPH_TRAIN_BACKWARD_LINEAR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Y[M,N] = X[M,K] @ W^T[N,K] where W is ternary int8 in {-1, 0, +1}. */
void tlinear_forward(
    float*        Y,
    const float*  X,
    const int8_t* W,
    int M, int K, int N);

/* dX[m,k] = Σ_n dY[m,n] · W[n,k]. Writes dX (not accumulates). */
void tlinear_backward_dX(
    float*        dX,
    const float*  dY,
    const int8_t* W,
    int M, int K, int N);

/* dW_latent[n,k] += Σ_m dY[m,n] · X[m,k]. Accumulates into float latent. */
void tlinear_backward_dW(
    float*       dW_latent,
    const float* dY,
    const float* X,
    int M, int K, int N);

#ifdef __cplusplus
}
#endif

#endif /* GLYPH_TRAIN_BACKWARD_LINEAR_H */
