/*
 * m4t_mtfp_w.h — MTFP39 wide-cell arithmetic (int64, 39 trits)
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * Parallel to m4t_mtfp.h but for the wide cell type m4t_mtfp_w_t (int64).
 * Half the NEON throughput (int64x2 = 2 lanes vs int32x4 = 4 lanes),
 * but 39 trits of range (±3.43e13 real) vs 19 trits (±9842 real).
 *
 * Multiply uses __int128 for the full product before rescale. Supported
 * on aarch64 by both GCC and Clang.
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

/* ── Matmul ────────────────────────────────────────────────────────────── */

/* Y[M,N] = X[M,K] @ W[N,K]^T. Both operands are MTFP39.
 * Accumulates in __int128, rescales by SCALE, clamps to ±MAX_VAL. */
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
