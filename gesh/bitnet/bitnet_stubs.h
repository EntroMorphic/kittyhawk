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

/* RMSNorm: y[i] = γ[i] · x[i] · rsqrt(mean(x²) + ε)
 * The x and y buffers are MTFP19 mantissas. γ is also MTFP19. ε is
 * passed as MTFP19 (caller pre-converts the float epsilon).
 *
 * Note (RC-3 from 2026-05-06 red-team): the canonical eps = 1e-5 from
 * BitNet's config doesn't have a clean MTFP19 mantissa-only
 * representation (substrate uses int mantissa × 3^exp). Work-unit 2
 * resolves the eps representation question. For now: stubs accept
 * eps as MTFP19 directly and callers pass small-but-nonzero values.
 *
 * n is the per-token vector length (= hidden_size or intermediate_size). */
void bitnet_stub_rmsnorm(
    m4t_mtfp_t* y, const m4t_mtfp_t* x, const m4t_mtfp_t* gamma,
    m4t_mtfp_t eps_mtfp19, int n
);

/* RoPE: applies rotary position embedding to (q, k) in-place.
 * q: [num_q_heads × head_dim]
 * k: [num_kv_heads × head_dim]
 * position: token's position index (0-based). Stub uses the formula
 * directly; production version uses precomputed cos/sin LUT. */
void bitnet_stub_rope_apply(
    m4t_mtfp_t* q, m4t_mtfp_t* k,
    int position,
    int num_q_heads, int num_kv_heads, int head_dim,
    double theta_base
);

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
