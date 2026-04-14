/*
 * m4t_mtfp_w.h — MTFP39 wide-cell arithmetic (int64, 39 trits)
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * Parallel to m4t_mtfp.h but for the wide cell type m4t_mtfp_w_t (int64).
 * Roughly 2.5× slower than MTFP19 per element: int64x2 = 2 lanes vs
 * int32x4 = 4 lanes, plus the int64 clamp is 4 instructions (compare +
 * select × 2) vs 2 for int32 (vminq + vmaxq). The wider range is the
 * tradeoff.
 * but 39 trits of range (±3.43e13 real) vs 19 trits (±9842 real).
 *
 * Multiply uses __int128 for the full product before rescale. Supported
 * on aarch64 by both GCC and Clang.
 *
 * v0 scope: arithmetic, vector ops, matmul. Higher-level ops (bias_add,
 * fan_in_normalize, layernorm) are MTFP19-only for now — the wide path
 * targets accumulation and high-precision matmul, not full transformer
 * layers. Add wide-path higher-level ops when a consumer needs them.
 */

#ifndef M4T_MTFP_W_H
#define M4T_MTFP_W_H

#include "m4t_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Scalar arithmetic ─────────────────────────────────────────────────── */

static inline m4t_mtfp_w_t m4t_mtfp_w_clamp128(__int128 v) {
    if (v >  (__int128)M4T_MTFPW_MAX_VAL) return  M4T_MTFPW_MAX_VAL;
    if (v < -(__int128)M4T_MTFPW_MAX_VAL) return -M4T_MTFPW_MAX_VAL;
    return (m4t_mtfp_w_t)v;
}

static inline m4t_mtfp_w_t m4t_mtfp_w_clamp(int64_t v) {
    if (v >  (int64_t)M4T_MTFPW_MAX_VAL) return  M4T_MTFPW_MAX_VAL;
    if (v < -(int64_t)M4T_MTFPW_MAX_VAL) return -M4T_MTFPW_MAX_VAL;
    return v;
}

static inline m4t_mtfp_w_t m4t_mtfp_w_add(m4t_mtfp_w_t a, m4t_mtfp_w_t b) {
    __int128 s = (__int128)a + (__int128)b;
    return m4t_mtfp_w_clamp128(s);
}

static inline m4t_mtfp_w_t m4t_mtfp_w_sub(m4t_mtfp_w_t a, m4t_mtfp_w_t b) {
    __int128 s = (__int128)a - (__int128)b;
    return m4t_mtfp_w_clamp128(s);
}

static inline m4t_mtfp_w_t m4t_mtfp_w_neg(m4t_mtfp_w_t a) {
    return -a;
}

static inline m4t_mtfp_w_t m4t_mtfp_w_mul(m4t_mtfp_w_t a, m4t_mtfp_w_t b) {
    __int128 prod = (__int128)a * (__int128)b;
    if (prod >= 0) prod += M4T_MTFPW_SCALE / 2;
    else           prod -= M4T_MTFPW_SCALE / 2;
    __int128 q = prod / M4T_MTFPW_SCALE;
    return m4t_mtfp_w_clamp128(q);
}

static inline m4t_mtfp_w_t m4t_mtfp_w_mul_trit(m4t_mtfp_w_t a, m4t_trit_t t) {
    return m4t_mtfp_w_clamp(a * (int64_t)t);
}

/* ── Vector arithmetic ─────────────────────────────────────────────────── */

void m4t_mtfp_w_vec_zero(m4t_mtfp_w_t* x, int n);
void m4t_mtfp_w_vec_add(m4t_mtfp_w_t* dst, const m4t_mtfp_w_t* a, const m4t_mtfp_w_t* b, int n);
void m4t_mtfp_w_vec_add_inplace(m4t_mtfp_w_t* dst, const m4t_mtfp_w_t* a, int n);
void m4t_mtfp_w_vec_sub_inplace(m4t_mtfp_w_t* dst, const m4t_mtfp_w_t* a, int n);
void m4t_mtfp_w_vec_scale(m4t_mtfp_w_t* dst, const m4t_mtfp_w_t* src, m4t_mtfp_w_t scale, int n);

/* ── Matmul ────────────────────────────────────────────────────────────── */

/* Y[M,N] = X[M,K] @ W[N,K]^T. Both operands are MTFP39.
 * Accumulates in __int128, rescales by SCALE, clamps to ±MAX_VAL.
 *
 * Constraint: K ≤ 41 with worst-case (MAX_VAL) cell values.
 * MAX_VAL^2 ≈ 4.12e36; __int128 max ≈ 1.7e38; 1.7e38/4.12e36 ≈ 41.
 * For typical cells (≪ MAX_VAL), K can be much larger. No wider hardware
 * accumulator exists; applications needing K > 41 at full range should
 * use the MTFP19 dense matmul (which uses __int128 with K_max ≈ 5e20). */
void m4t_mtfp_w_matmul_bt(
    m4t_mtfp_w_t* Y,
    const m4t_mtfp_w_t* X,
    const m4t_mtfp_w_t* W,
    int M, int K, int N
);

/* Y[M,N] = X[M,K] @ W_packed^T. X is MTFP39, W is packed trits.
 * No multiply — just add/sub int64 cells. Accumulates in __int128
 * for safety at large K, clamps on store. */
void m4t_mtfp_w_ternary_matmul_bt(
    m4t_mtfp_w_t* Y,
    const m4t_mtfp_w_t* X,
    const uint8_t* W_packed,
    int M, int K, int N
);

#ifdef __cplusplus
}
#endif

#endif /* M4T_MTFP_W_H */
