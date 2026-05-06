/*
 * gesh/bitnet/bitnet_stubs.h — temporary scalar stubs for substrate
 * primitives that work-units 2-5 will replace with NEON.
 *
 * These signatures are the contract the eventual NEON primitives must
 * honor. Substituting in libm4t versions in work-unit 6 requires
 * changing only the .c file's call sites (or, ideally, the link-time
 * symbol resolution).
 *
 * Per project rule: stubs in this file are NOT "production scalar
 * fallbacks" — they're test-harness scaffolding, comparable to the
 * `_scalar_ref` pattern in libm4t but consumer-side. Marked clearly so
 * they don't drift into production.
 *
 * Buffer types use m4t_mtfp_t (== int32_t) to make the substrate-native
 * semantic explicit; this is RC-4 from the 2026-05-06 red-team.
 */

#ifndef GESH_BITNET_STUBS_H
#define GESH_BITNET_STUBS_H

#include "m4t_types.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* RMSNorm: replaced by m4t_mtfp_rmsnorm in m4t_mtfp.h (work-unit 2).
 * The stub remains in bitnet_stubs.c temporarily; it will be removed
 * after the full bring-up converges. New callers MUST use the substrate
 * primitive. */

/* RoPE: replaced by m4t_mtfp_rope_apply in m4t_mtfp.h (work-unit 3).
 * Note: the original stub used the adjacent-pair convention; BitNet
 * actually uses Llama's rotate_half convention. The substrate primitive
 * is the only correct path. */

/* Softmax (numerically stable: subtract max, exp, normalize).
 * y[i] = exp(x[i] - max(x)) / sum_j exp(x[j] - max(x))
 * n: vector length (= seq_len for attention). */
void bitnet_stub_softmax(m4t_mtfp_t* y, const m4t_mtfp_t* x, int n);

/* A8 quantize: per-token absmax → int8.
 * Computes scale = max(|x|) / 127, then x_int8 = round(x · 127 / max(|x|)).
 *
 * Returns the per-token ABSMAX (NOT the scale; per RC-3 rename).
 * Dequantize divides by 127 to recover.
 *
 * x: input MTFP19, length n.
 * y: output int8, length n. */
m4t_mtfp_t bitnet_stub_a8_quantize(int8_t* y, const m4t_mtfp_t* x, int n);

/* A8 dequantize: int8 × absmax / 127 → MTFP19.
 * y[i] = x_int8[i] · absmax_mtfp19 / 127
 * Caller passes the absmax stored from quantize. */
void bitnet_stub_a8_dequantize(
    m4t_mtfp_t* y, const int8_t* x, m4t_mtfp_t absmax_mtfp19, int n
);

/* ReLU² activation in-place. y[i] = (max(0, x[i]))²
 * MTFP19 throughout. Saturating clamp on result if it exceeds MTFP19_MAX. */
void bitnet_stub_relu2_inplace(m4t_mtfp_t* x, int n);

/* Element-wise multiply: y[i] = a[i] * b[i] (MTFP19 × MTFP19 → MTFP19,
 * with saturating clamp on overflow). Used by FFN's gate*up.
 * Substrate has m4t_mtfp_block_add but no block_mul as of this writing —
 * if work-unit 1 confirms this primitive is needed, work-unit 5+ adds it
 * to libm4t alongside the A8 family. */
void bitnet_stub_elementwise_mul(
    m4t_mtfp_t* y, const m4t_mtfp_t* a, const m4t_mtfp_t* b, int n
);

#ifdef __cplusplus
}
#endif

#endif /* GESH_BITNET_STUBS_H */
