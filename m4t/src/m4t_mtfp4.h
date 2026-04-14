/*
 * m4t_mtfp4.h — MTFP4 routing cell (int8 mantissa, 4 trits)
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * The narrowest MTFP cell. 16 mantissas per NEON register = one block
 * (16 bytes). Designed for routing: signature scores, distance
 * normalization, and the SDOT-native ternary matmul.
 *
 * Mantissa range: |m| ≤ M4T_MTFP4_MAX_VAL = 40 = (3^4 - 1) / 2.
 * Cells-per-block: M4T_MTFP4_CELLS_PER_BLOCK = 16.
 * Block exponent: sidecar metadata, per the substrate spec (§7). Under
 * the legacy default-block-exponent convention, M4T_MTFP4_SCALE = 3^2
 * specifies `block_exp = -2`; new consumers should track the exponent
 * explicitly rather than reading SCALE as a type property.
 *
 * SDOT is the hardware-native ternary matmul (§8.4 Case W — the output
 * widens to MTFP19 mantissa exactly; the 16-byte MTFP4 block maps to
 * one SDOT input, the 16-byte MTFP19 block to one SDOT output).
 *
 * This header exposes only what live consumers demand:
 *   - m4t_mtfp4_clamp — saturating narrow of an int32 accumulator to
 *     MTFP4 mantissa. Used by the SDOT matmul store and by the
 *     conversion routines. Case S (§8.5) — fixed-output saturation.
 *   - m4t_mtfp4_sdot_matmul_bt — the SDOT primitive.
 *   - m4t_mtfp19_to_mtfp4 / m4t_mtfp4_to_mtfp19 — cell-width narrow/
 *     widen conversions under the default-block-exponent convention.
 *     The widen (to MTFP19) is exact. The narrow (to MTFP4) rounds at
 *     the cell boundary and is a named lossy op — consumers that need
 *     exactness should stay in MTFP19 or request a Case R opt-in (not
 *     provided until a consumer drives it; see §14.2).
 *
 * Scalar arithmetic primitives (add/sub/neg/mul) are intentionally
 * absent from this header. Under the substrate discipline ("no
 * primitive without named consumer demand"), they re-emerge when a
 * consumer asks — and when they do, they land with §8.5-compliant
 * semantics (mul widens to MTFP9 mantissa, not a silent-rounding MTFP4).
 */

#ifndef M4T_MTFP4_H
#define M4T_MTFP4_H

#include "m4t_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Saturating clamp (Case S per §8.5) ───────────────────────────────── */

/* Narrow an int32 accumulator to an MTFP4 mantissa cell. Exact when
 * |v| ≤ M4T_MTFP4_MAX_VAL; saturates at ±MAX_VAL otherwise. Used by the
 * SDOT matmul store and by the cell-width conversion routines. */
static inline m4t_mtfp4_t m4t_mtfp4_clamp(int32_t v) {
    if (v >  (int32_t)M4T_MTFP4_MAX_VAL) return  M4T_MTFP4_MAX_VAL;
    if (v < -(int32_t)M4T_MTFP4_MAX_VAL) return -M4T_MTFP4_MAX_VAL;
    return (m4t_mtfp4_t)v;
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
 * Precondition: W elements MUST be in {-1, 0, +1}. The int32 accumulator
 * is safe for any K with valid trit weights (max per lane = K/4 × 40,
 * trivially below INT32_MAX). Non-trit weights violate this bound.
 *
 * W layout: [N, K] row-major int8, each element in {-1, 0, +1}.
 * NOT packed trits — raw int8 values, because SDOT needs int8 operands.
 * This uses 4× the memory of the packed representation; the tradeoff is
 * that SDOT runs at full throughput with zero decode overhead.
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
