/*
 * backward_routed.c — scalar forward + backward for routed dispatch.
 *
 * Scalar reference. NEON later, once gradient checks pass.
 *
 * The central design choice is STE through top-k: the backward treats
 * the k selected tiles as if selection were identity on those slots,
 * and zero elsewhere. Non-selected tiles receive no gradient this step.
 * When the selection CHANGES on a later forward pass (because latents
 * drifted enough to reorder scores), a different subset of tiles
 * receives gradient. Over training, every tile that's ever selected
 * gets updates proportional to its participation.
 */

#include "backward_routed.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

void rroute_forward_select(
    int32_t* decisions, int8_t* signs,
    const float* X, const int8_t* U,
    int M, int H, int T, int k)
{
    /* For each token, compute all T scores then pick top-k by |score|. */
    for (int m = 0; m < M; m++) {
        /* Compute scores[T]. */
        float scores[64]; /* MVP cap: T ≤ 64. Enlarge when needed. */
        for (int t = 0; t < T; t++) {
            float acc = 0.0f;
            const float*  x_row = X + (size_t)m * H;
            const int8_t* u_row = U + (size_t)t * H;
            for (int h = 0; h < H; h++) {
                acc += x_row[h] * (float)u_row[h];
            }
            scores[t] = acc;
        }
        /* Top-k |score| with straight-selection. */
        uint64_t used = 0; /* T ≤ 64 bitmask */
        for (int sel = 0; sel < k; sel++) {
            int best_t = -1;
            float best_abs = 0.0f;
            for (int t = 0; t < T; t++) {
                if (used & ((uint64_t)1 << t)) continue;
                float a = scores[t];
                if (a < 0.0f) a = -a;
                if (a > best_abs) { best_abs = a; best_t = t; }
            }
            if (best_t < 0 || best_abs == 0.0f) {
                decisions[(size_t)m * k + sel] = -1;
                signs[(size_t)m * k + sel]     = 0;
            } else {
                used |= (uint64_t)1 << best_t;
                decisions[(size_t)m * k + sel] = best_t;
                signs[(size_t)m * k + sel]     = (scores[best_t] > 0.0f) ? 1 : -1;
            }
        }
    }
}

/* MVP design choice: the `signs` output of rroute_forward_select records
 * each selected tile's score sign but the dispatch always contributes
 * with +1 weight. Mainstream learned-gating (Switch Transformer, GShard)
 * does this because the selection is which-tile, not how-tile-contributes.
 *
 * The substrate's m4t_route_apply_signed uses signed weights because its
 * tiles are trained prototypes with meaningful anti-correlation structure.
 * For a trainer bootstrapping from random init, applying the gate's sign
 * randomizes the forward and destabilizes SGD. A follow-up cycle can
 * re-introduce learned-sign routing once prototypes are discriminative. */
void rroute_forward_dispatch(
    float* Y, const float* X, const int8_t* W,
    const int32_t* decisions, const int8_t* signs,
    int M, int H, int T, int N, int k)
{
    (void)T; (void)signs;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            Y[(size_t)m * N + n] = 0.0f;
        }
        for (int sel = 0; sel < k; sel++) {
            int32_t t = decisions[(size_t)m * k + sel];
            if (t < 0) continue;
            const float*  x_row = X + (size_t)m * H;
            const int8_t* w_tile = W + ((size_t)t * H) * N;
            for (int n = 0; n < N; n++) {
                float acc = 0.0f;
                for (int h = 0; h < H; h++) {
                    acc += x_row[h] * (float)w_tile[(size_t)h * N + n];
                }
                Y[(size_t)m * N + n] += acc;
            }
        }
    }
}

void rroute_backward_dX(
    float* dX, const float* dY, const int8_t* W,
    const int32_t* decisions, const int8_t* signs,
    int M, int H, int T, int N, int k)
{
    (void)T; (void)signs;
    for (int m = 0; m < M; m++) {
        for (int h = 0; h < H; h++) {
            dX[(size_t)m * H + h] = 0.0f;
        }
        for (int sel = 0; sel < k; sel++) {
            int32_t t = decisions[(size_t)m * k + sel];
            if (t < 0) continue;
            const int8_t* w_tile = W + ((size_t)t * H) * N;
            for (int h = 0; h < H; h++) {
                float acc = 0.0f;
                for (int n = 0; n < N; n++) {
                    acc += dY[(size_t)m * N + n] * (float)w_tile[(size_t)h * N + n];
                }
                dX[(size_t)m * H + h] += acc;
            }
        }
    }
}

void rroute_backward_dW(
    float* dW_latent, const float* dY, const float* X,
    const int32_t* decisions, const int8_t* signs,
    int M, int H, int T, int N, int k)
{
    (void)T; (void)signs;
    for (int m = 0; m < M; m++) {
        for (int sel = 0; sel < k; sel++) {
            int32_t t = decisions[(size_t)m * k + sel];
            if (t < 0) continue;
            const float* x_row = X + (size_t)m * H;
            float* wt = dW_latent + ((size_t)t * H) * N;
            for (int h = 0; h < H; h++) {
                float xh = x_row[h];
                for (int n = 0; n < N; n++) {
                    wt[(size_t)h * N + n] += xh * dY[(size_t)m * N + n];
                }
            }
        }
    }
}
