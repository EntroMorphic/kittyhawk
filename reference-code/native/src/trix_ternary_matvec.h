/*
 * trix_ternary_matvec.h — NEON multiply-free ternary matvec
 *
 * Adapted from yinsen/neon/neon_ternary.c for integration with trix_atoms.
 *
 * Computes y = W_ternary @ x where W ∈ {-1, 0, +1}^{M×K}
 * using only add/subtract — zero multiplications.
 *
 * Two variants:
 *   1. Float interface: float x → quantize to int8 → ternary matvec → float y
 *   2. Int8 interface:  int8 x → ternary matvec → int32 y (raw, needs scaling)
 *
 * Weight packing: K-vertical format
 *   4 trits per byte: byte = t0 | (t1<<2) | (t2<<4) | (t3<<6)
 *   where ti ∈ {0=zero, 1=+1, 2=-1, 3=zero}
 */

#ifndef TRIX_TERNARY_MATVEC_H
#define TRIX_TERNARY_MATVEC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Weight packing ── */

/*
 * Pack float ternary weights into K-vertical byte format.
 * weights: [M, K] float values in {-1.0, 0.0, +1.0}
 * packed:  [M, K/4] bytes (caller allocates)
 * K must be multiple of 4.
 */
void trix_ternary_pack_weights(
    uint8_t* packed, const float* weights, int M, int K
);

/*
 * Pack int8 ternary weights.
 * weights: [M, K] int8 values in {-1, 0, +1}
 */
void trix_ternary_pack_weights_i8(
    uint8_t* packed, const int8_t* weights, int M, int K
);

/* ── Matvec: int8 activations, packed ternary weights ── */

/*
 * y[M] = W_packed[M, K/4] @ act[K]
 *
 * Uses NEON SDOT with VLD4 deinterleaving on ARM.
 * Scalar fallback on other platforms.
 * K must be multiple of 64 for SDOT path, 4 for scalar.
 */
void trix_ternary_matvec_i8(
    int32_t* y, const int8_t* act, const uint8_t* W_packed,
    int M, int K
);

/* ── Matvec: float interface ── */

/*
 * y[M] = W_ternary[M, K] @ x[K]
 *
 * Internally quantizes x to int8, runs ternary matvec, converts back to float.
 * W_ternary: [M, K] float values in {-1.0, 0.0, +1.0} (NOT packed)
 *
 * This is the drop-in replacement for trix_matmul when weights are ternary.
 * Output is EXACT (no quantization error from weights — they're already ternary).
 * Input quantization to int8 introduces at most 1/127 ≈ 0.8% relative error.
 */
void trix_ternary_matvec_f32(
    float* y, const float* W_ternary, const float* x,
    int M, int K
);

#ifdef __cplusplus
}
#endif

#endif /* TRIX_TERNARY_MATVEC_H */
