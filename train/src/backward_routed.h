/*
 * backward_routed.h — routed dispatch forward + backward.
 *
 * The MVP's routing layer. Conceptually:
 *
 *   Forward per token m:
 *     1. scores[m,t] = Σ_h X[m,h] · U[t,h]       // U is per-tile gating weight
 *     2. decisions[m, 0..k-1] = topk_abs(scores[m,:], k)
 *     3. for each selected tile (t_i, sign_i):
 *          contrib[m, :] += sign_i · X[m,:] @ W[t_i,:,:]
 *     4. Y[m,n] = contrib[m,n]
 *
 * Forward kernels composed from tlinear_forward (per-tile) plus scalar
 * top-k and signed-accumulation logic.
 *
 * Backward STE choices:
 *   - Through top-k: gradient flows only to the k SELECTED tiles'
 *     contribution slots. Unselected tiles receive zero gradient.
 *     Straight-through: pretend selection is identity on selected indices.
 *   - Through the signed multiplier: gradient scales by sign_i (transparent).
 *   - dU (gating weights): zero in this MVP — the gating scores are
 *     discrete (they determine WHICH tiles fire). To train U we would
 *     need soft routing (softmax-over-scores differentiable relaxation).
 *     For the MVP, U is FROZEN and hand-initialized; only tile weights W
 *     receive gradient updates.
 *
 * What that means mechanically: this is "fixed routing, trained tile
 * weights." A later cycle can unfreeze U via soft routing.
 *
 * Shape conventions:
 *   X : float [M × H]             per-token activations
 *   U : int8  [T × H]             tile-gating weights, ternary, frozen
 *   W : int8  [T × H × N]         per-tile ternary matmul weights (int8, ternary)
 *   Y : float [M × N]             output
 *   decisions : int32 [M × k]     selected tile indices (-1 sentinel)
 *   signs     : int8  [M × k]     selected signs in {-1, +1}
 *
 * The total parameter count is T·H·N trits for W (trainable) plus T·H
 * trits for U (frozen).
 */

#ifndef GLYPH_TRAIN_BACKWARD_ROUTED_H
#define GLYPH_TRAIN_BACKWARD_ROUTED_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compute scores[m,t] = Σ_h X[m,h] · U[t,h] and choose top-k |score|
 * selections per row. Returns per-row decisions (tile_idx, sign).
 * decisions[m,i] = tile index of i-th selection, -1 if fewer than k
 * non-zero scores. signs[m,i] ∈ {-1, +1, 0}.
 *
 * The top-k is |abs| — same as m4t_route_topk_abs. Unused slots fill
 * with (-1, 0). */
void rroute_forward_select(
    int32_t*      decisions,
    int8_t*       signs,
    const float*  X,
    const int8_t* U,
    int M, int H, int T, int k);

/* Forward tile dispatch: accumulate sign-weighted per-tile matmul
 * contributions into Y for each selected tile of each token.
 *
 *   Y[m, n] = Σ_{i : decisions[m,i] ≥ 0} signs[m,i] · (X[m,:] @ W[decisions[m,i],:,n])
 *
 * Y is written (not accumulated), zero-initialized internally. */
void rroute_forward_dispatch(
    float*        Y,
    const float*  X,
    const int8_t* W,
    const int32_t* decisions,
    const int8_t*  signs,
    int M, int H, int T, int N, int k);

/* Backward dX through dispatch (STE top-k: selected tiles pass through).
 *
 *   dX[m, h] = Σ_{i : decisions[m,i] ≥ 0} signs[m,i] · Σ_n dY[m,n] · W[decisions[m,i],h,n]
 *
 * dX is written (not accumulated). */
void rroute_backward_dX(
    float*        dX,
    const float*  dY,
    const int8_t* W,
    const int32_t* decisions,
    const int8_t*  signs,
    int M, int H, int T, int N, int k);

/* Backward dW into latent accumulators, only for selected tiles.
 *
 *   dW_latent[t, h, n] += Σ_{m : tile t was selected for m with sign s}
 *                         s · X[m,h] · dY[m,n]
 *
 * dW_latent is ACCUMULATED INTO (not written). */
void rroute_backward_dW(
    float*        dW_latent,
    const float*  dY,
    const float*  X,
    const int32_t* decisions,
    const int8_t*  signs,
    int M, int H, int T, int N, int k);

#ifdef __cplusplus
}
#endif

#endif /* GLYPH_TRAIN_BACKWARD_ROUTED_H */
