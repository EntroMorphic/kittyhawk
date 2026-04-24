/*
 * requantize.h — hysteresis-aware re-quantization of float latents to trits.
 *
 * Without hysteresis, latents that cluster near τ flip trit state on every
 * small SGD update, so the deployed integer weights jump between nearby
 * ternary configurations. The trainer's accuracy then oscillates wildly
 * between re-quantize calls (seen in the MVP's first convergence test:
 * 100 % → 1.6 % → 100 % across successive epochs). The underlying
 * optimization IS converging — it's the snapping that's noisy.
 *
 * Fix: the current trit is sticky. Crossing into a new trit state
 * requires the latent to exceed τ by a margin (hysteresis_frac × τ).
 * Standard sign-hysteresis from physical systems; here a scalar per-tensor.
 *
 * Parameters:
 *   W          — int8 trits, in/out. Current state read for stickiness.
 *   W_latent   — float latents, read-only.
 *   n          — element count.
 *   density    — fraction of trits that should be zero (0 state). τ is
 *                computed as the density-th percentile of |W_latent|.
 *   hysteresis — dead-zone half-width as fraction of τ. 0 recovers
 *                the non-sticky threshold. 0.1 is a reasonable default.
 *
 * Returns number of trits that flipped state on this call. High flip
 * count during training is a sign of instability; should decrease as
 * training converges.
 */

#ifndef GLYPH_TRAIN_REQUANTIZE_H
#define GLYPH_TRAIN_REQUANTIZE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int requantize_hysteresis(
    int8_t*      W,
    const float* W_latent,
    int          n,
    double       density,
    double       hysteresis);

#ifdef __cplusplus
}
#endif

#endif /* GLYPH_TRAIN_REQUANTIZE_H */
