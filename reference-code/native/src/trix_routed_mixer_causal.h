/*
 * trix_routed_mixer_causal.h — Causal variants of scatter and gather
 *
 * For autoregressive language modeling: position i only sees positions 0..i-1.
 *
 * Causal scatter-gather processes positions left to right, maintaining
 * running pools per tile. Each position gathers from the pool accumulated
 * by all prior positions, then scatters its own contribution for future
 * positions to see.
 *
 * Complexity: O(seq × T × D) — same as bidirectional.
 */

#ifndef TRIX_ROUTED_MIXER_CAUSAL_H
#define TRIX_ROUTED_MIXER_CAUSAL_H

#include "trix_routed_mixer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Causal forward: each position sees only prior positions through the mixer.
 * Same API as trix_routed_mixer_forward but with causal masking. */
void trix_routed_mixer_forward_causal(TrixRoutedMixer* rm,
                                       const float* x, float* out, int seq_len);

/* MTFP causal forward: entire path in integer arithmetic.
 * x and out are mtfp_t*. No float conversions. */
void trix_routed_mixer_forward_causal_mtfp(TrixRoutedMixer* rm,
                                            const mtfp_t* x, mtfp_t* out, int seq_len);

#ifdef __cplusplus
}
#endif

#endif /* TRIX_ROUTED_MIXER_CAUSAL_H */
