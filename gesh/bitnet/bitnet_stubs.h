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
 */

#ifndef GESH_BITNET_STUBS_H
#define GESH_BITNET_STUBS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* RMSNorm: y[i] = γ[i] · x[i] · rsqrt(mean(x²) + ε)
 * The x and y buffers are MTFP19 mantissas. γ is also MTFP19. ε is
 * passed as MTFP19 (caller pre-converts the float epsilon).
 * n is the per-token vector length (= hidden_size or intermediate_size). */
void bitnet_stub_rmsnorm(
    int32_t* y, const int32_t* x, const int32_t* gamma,
    int32_t eps_mtfp19, int n
);

/* RoPE: applies rotary position embedding to (q, k) in-place.
 * q: [num_q_heads × head_dim]
 * k: [num_kv_heads × head_dim]
 * position: token's position index (0-based). Stub uses the formula
 * directly; production version uses precomputed cos/sin LUT. */
void bitnet_stub_rope_apply(
    int32_t* q, int32_t* k,
    int position,
    int num_q_heads, int num_kv_heads, int head_dim,
    double theta_base
);

/* Softmax (numerically stable: subtract max, exp, normalize).
 * y[i] = exp(x[i] - max(x)) / sum_j exp(x[j] - max(x))
 * n: vector length (= seq_len for attention). */
void bitnet_stub_softmax(int32_t* y, const int32_t* x, int n);

/* A8 quantize: per-token absmax → int8.
 * Computes scale = max(|x|) / 127, then x_int8 = round(x / scale).
 * Returns the scale (as MTFP19) so dequantize can recover.
 * x: input MTFP19, length n.
 * y: output int8, length n. */
int32_t bitnet_stub_a8_quantize(int8_t* y, const int32_t* x, int n);

/* A8 dequantize: int8 × scale → MTFP19.
 * y[i] = x_int8[i] · scale_mtfp / 127
 * Caller passes the scale stored from quantize. */
void bitnet_stub_a8_dequantize(
    int32_t* y, const int8_t* x, int32_t scale_mtfp19, int n
);

/* ReLU² activation in-place. y[i] = (max(0, x[i]))²
 * MTFP19 throughout. Saturating clamp on result if it exceeds MTFP19_MAX. */
void bitnet_stub_relu2_inplace(int32_t* x, int n);

/* Element-wise multiply: y[i] = a[i] * b[i] (MTFP19 × MTFP19 → MTFP19,
 * with saturating clamp on overflow). Used by FFN's gate*up.
 * Substrate has m4t_mtfp_block_add but no block_mul as of this writing —
 * if work-unit 1 confirms this primitive is needed, work-unit 5+ adds it
 * to libm4t alongside the A8 family. */
void bitnet_stub_elementwise_mul(
    int32_t* y, const int32_t* a, const int32_t* b, int n
);

#ifdef __cplusplus
}
#endif

#endif /* GESH_BITNET_STUBS_H */
