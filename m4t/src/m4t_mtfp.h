/*
 * m4t_mtfp.h — Multi-Trit Floating Point arithmetic
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * This header declares the MTFP19 (int32, 19-trit) numeric core of m4t.
 * Functions here operate on m4t_mtfp_t cells. No float path. int8/int16
 * appear only as MTFP cell containers at clean trit boundaries (MTFP4,
 * MTFP9) — never as binary quantization. Softmax and GELU LUTs are
 * deferred to a later pipeline pass. See m4t/docs/M4T_PIPELINE.md.
 */

#ifndef M4T_MTFP_H
#define M4T_MTFP_H

#include "m4t_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Scalar arithmetic ─────────────────────────────────────────────────── */

/* All scalar arithmetic saturates at ±M4T_MTFP_MAX_VAL. This is a hard
 * API contract: the output of every MTFP operation is a valid MTFP cell,
 * even when the true mathematical result would exceed the range. */

static inline m4t_mtfp_t m4t_mtfp_clamp64(int64_t v) {
    if (v >  (int64_t)M4T_MTFP_MAX_VAL) return  M4T_MTFP_MAX_VAL;
    if (v < -(int64_t)M4T_MTFP_MAX_VAL) return -M4T_MTFP_MAX_VAL;
    return (m4t_mtfp_t)v;
}

static inline m4t_mtfp_t m4t_mtfp_add(m4t_mtfp_t a, m4t_mtfp_t b) {
    return m4t_mtfp_clamp64((int64_t)a + (int64_t)b);
}

static inline m4t_mtfp_t m4t_mtfp_sub(m4t_mtfp_t a, m4t_mtfp_t b) {
    return m4t_mtfp_clamp64((int64_t)a - (int64_t)b);
}

static inline m4t_mtfp_t m4t_mtfp_neg(m4t_mtfp_t a) {
    /* Safe for in-range inputs: |-MAX_VAL| = MAX_VAL. */
    return -a;
}

/* Multi-trit × multi-trit multiply. int64 product, rescale by SCALE,
 * round to nearest with ties away from zero, saturate at ±MAX_VAL.
 *
 * Saturation is a hard requirement: the product of two in-range MTFP cells
 * can legitimately exceed ±MAX_VAL (e.g. 2.0 * 2.0 = 4.0, well below
 * ±9842). Callers must still only pass in-range operands; this routine
 * keeps the output in range even when the true mathematical product
 * wouldn't fit. */
static inline m4t_mtfp_t m4t_mtfp_mul(m4t_mtfp_t a, m4t_mtfp_t b) {
    int64_t prod = (int64_t)a * (int64_t)b;
    if (prod >= 0) prod += M4T_MTFP_SCALE / 2;
    else           prod -= M4T_MTFP_SCALE / 2;
    int64_t q = prod / M4T_MTFP_SCALE;
    if (q >  (int64_t)M4T_MTFP_MAX_VAL) q =  M4T_MTFP_MAX_VAL;
    if (q < -(int64_t)M4T_MTFP_MAX_VAL) q = -M4T_MTFP_MAX_VAL;
    return (m4t_mtfp_t)q;
}

/* Multi-trit × single trit. Zero multiplies: the compiler reduces a
 * multiply by {-1,0,+1} to a negate or a drop. Saturates at ±MAX_VAL
 * for API consistency with add/sub/mul. */
static inline m4t_mtfp_t m4t_mtfp_mul_trit(m4t_mtfp_t a, m4t_trit_t t) {
    return m4t_mtfp_clamp64((int64_t)a * (int64_t)t);
}

/* ── Vector arithmetic ─────────────────────────────────────────────────── */

void m4t_mtfp_vec_zero(m4t_mtfp_t* x, int n);
void m4t_mtfp_vec_add(m4t_mtfp_t* dst, const m4t_mtfp_t* a, const m4t_mtfp_t* b, int n);
void m4t_mtfp_vec_add_inplace(m4t_mtfp_t* dst, const m4t_mtfp_t* a, int n);
void m4t_mtfp_vec_scale(m4t_mtfp_t* dst, const m4t_mtfp_t* src, m4t_mtfp_t scale, int n);

/* ── Dense MTFP × MTFP matmul (int64 accumulator, rescale by SCALE) ──────
 *
 * m4t_mtfp_matmul:     Y[M,N] = X[M,K] @ W[K,N]
 * m4t_mtfp_matmul_bt:  Y[M,N] = X[M,K] @ W[N,K]^T   (W row-major with K stride)
 */
void m4t_mtfp_matmul(
    m4t_mtfp_t* Y,
    const m4t_mtfp_t* X,
    const m4t_mtfp_t* W,
    int M, int K, int N
);

void m4t_mtfp_matmul_bt(
    m4t_mtfp_t* Y,
    const m4t_mtfp_t* X,
    const m4t_mtfp_t* W,
    int M, int K, int N
);

/* ── Bias / normalization ──────────────────────────────────────────────── */

void m4t_mtfp_bias_add(m4t_mtfp_t* x, const m4t_mtfp_t* b, int batch, int dim);

/* Divide each element by isqrt(fan_in). Used before nonlinearities to
 * keep values in the table-lookup range. */
void m4t_mtfp_fan_in_normalize(m4t_mtfp_t* x, int n, int fan_in);

/* LayerNorm, forward only (inference-only v0).
 *
 * `eps` is an MTFP cell in real units (e.g. m4t_mtfp_from_real_hostside(1e-5)
 * if you generated weights offline). The function squares it internally for
 * comparison against variance. */
void m4t_mtfp_layernorm(
    m4t_mtfp_t* dst,
    const m4t_mtfp_t* src,
    const m4t_mtfp_t* weight,
    const m4t_mtfp_t* bias,
    m4t_mtfp_t eps,
    int rows, int cols
);

/* ── Integer square-root helpers ───────────────────────────────────────── */

/* floor(sqrt(x)) for x >= 0, via Newton-Raphson. Pure integer. */
int64_t m4t_isqrt64(int64_t x);

/* Returns SCALE^2 / sqrt(x). x is a variance in SCALE^2 units; result is
 * an rstd in SCALE units. Pure integer, used by layernorm. */
int64_t m4t_mtfp_isqrt_inv(int64_t x);

#ifdef __cplusplus
}
#endif

#endif /* M4T_MTFP_H */
