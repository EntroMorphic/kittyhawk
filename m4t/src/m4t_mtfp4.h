/*
 * m4t_mtfp4.h — MTFP4 routing cell (int8, 4 trits)
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * The narrowest MTFP cell. 16 cells per NEON register. Designed for
 * routing: signature scores, distance normalization, and the SDOT-native
 * ternary matmul where int8 MTFP4 activations × int8 trit weights feed
 * directly into vdotq_s32 — 16 multiply-accumulates per cycle.
 *
 * radix=2, scale=3^2=9. real = cell / 9.
 * range: ±(3^4-1)/2 / 9 = ±40/9 ≈ ±4.44.
 * resolution: 1/9 ≈ 0.111.
 *
 * The SDOT matmul is the fastest ternary operation M4 silicon can do.
 * It exists because ternary weights {-1, 0, +1} stored as int8 are
 * valid SDOT operands, and MTFP4 activations in int8 are also valid.
 * The int32 accumulator holds the sum without rescale (ternary × MTFP4
 * stays in MTFP4 units; no SCALE division needed).
 */

#ifndef M4T_MTFP4_H
#define M4T_MTFP4_H

#include "m4t_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Scalar arithmetic ─────────────────────────────────────────────────── */

static inline m4t_mtfp4_t m4t_mtfp4_clamp(int32_t v) {
    if (v >  (int32_t)M4T_MTFP4_MAX_VAL) return  M4T_MTFP4_MAX_VAL;
    if (v < -(int32_t)M4T_MTFP4_MAX_VAL) return -M4T_MTFP4_MAX_VAL;
    return (m4t_mtfp4_t)v;
}

static inline m4t_mtfp4_t m4t_mtfp4_add(m4t_mtfp4_t a, m4t_mtfp4_t b) {
    return m4t_mtfp4_clamp((int32_t)a + (int32_t)b);
}

static inline m4t_mtfp4_t m4t_mtfp4_sub(m4t_mtfp4_t a, m4t_mtfp4_t b) {
    return m4t_mtfp4_clamp((int32_t)a - (int32_t)b);
}

static inline m4t_mtfp4_t m4t_mtfp4_neg(m4t_mtfp4_t a) {
    return (m4t_mtfp4_t)(-a);
}

/* MTFP4 × MTFP4 multiply: int16 product, rescale by SCALE=9, clamp. */
static inline m4t_mtfp4_t m4t_mtfp4_mul(m4t_mtfp4_t a, m4t_mtfp4_t b) {
    int16_t prod = (int16_t)a * (int16_t)b;
    if (prod >= 0) prod += M4T_MTFP4_SCALE / 2;
    else           prod -= M4T_MTFP4_SCALE / 2;
    return m4t_mtfp4_clamp((int32_t)(prod / M4T_MTFP4_SCALE));
}

/* MTFP4 × trit: no rescale needed. */
static inline m4t_mtfp4_t m4t_mtfp4_mul_trit(m4t_mtfp4_t a, m4t_trit_t t) {
    return m4t_mtfp4_clamp((int32_t)a * (int32_t)t);
}

/* ── SDOT ternary matmul ───────────────────────────────────────────────── */

/* Y[M,N] = X[M,K] @ W^T where X is MTFP4 (int8) and W is int8 trits.
 *
 * This is the hardware-native ternary matmul: vdotq_s32 computes a
 * 4-element dot product of int8 vectors, accumulating into int32.
 * 16 multiply-accumulates per SDOT instruction.
 *
 * The int32 accumulator holds MTFP4-scale values (scale=9). No rescale
 * by SCALE is needed because trit weights are dimensionless {-1,0,+1}.
 * The result is clamped to ±M4T_MTFP4_MAX_VAL on store.
 *
 * W layout: [N, K] row-major int8, each element in {-1, 0, +1}.
 * NOT packed trits — raw int8 values, because SDOT needs int8 operands.
 */
void m4t_mtfp4_sdot_matmul_bt(
    m4t_mtfp4_t* Y,
    const m4t_mtfp4_t* X,
    const m4t_trit_t* W,
    int M, int K, int N
);

/* ── Conversion ────────────────────────────────────────────────────────── */

/* Convert MTFP19 cells to MTFP4 cells. Rescales from scale=59049 to
 * scale=9 (divide by 6561 = 59049/9), then clamps to ±MAX_VAL_4. */
void m4t_mtfp19_to_mtfp4(
    m4t_mtfp4_t* dst,
    const m4t_mtfp_t* src,
    int n
);

/* Convert MTFP4 cells to MTFP19 cells. Rescales from scale=9 to
 * scale=59049 (multiply by 6561). No clamp needed — MTFP4 max (40)
 * times 6561 = 262440, well within MTFP19 range. */
void m4t_mtfp4_to_mtfp19(
    m4t_mtfp_t* dst,
    const m4t_mtfp4_t* src,
    int n
);

#ifdef __cplusplus
}
#endif

#endif /* M4T_MTFP4_H */
