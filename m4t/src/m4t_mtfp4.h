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
 * Block exponent: sidecar metadata, per the substrate spec (§7).
 *
 * SDOT is the hardware-native ternary matmul (§8.4 Case W — output widens
 * to MTFP19 mantissa exactly; the 16-byte MTFP4 block maps to one SDOT
 * input lane, the 16-byte MTFP19 block to one SDOT output lane).
 *
 * This header exposes:
 *   - m4t_mtfp4_clamp — saturating narrow of an int32 to MTFP4. Used by
 *     the narrowing conversion routine. Case S (§8.5) — fixed-output
 *     saturation on the narrow.
 *   - m4t_mtfp4_sdot_matmul_bt — the SDOT primitive. MTFP4 × ternary →
 *     MTFP19 (Case W per §8.4); exact by construction.
 *   - m4t_mtfp19_to_mtfp4 / m4t_mtfp4_to_mtfp19 — cell-width narrow/widen
 *     conversions under the default-block-exponent convention.
 *     - widen (mtfp4_to_mtfp19): Case W, exact (×6561 fits MTFP19 by
 *       static assert).
 *     - narrow (mtfp19_to_mtfp4): rounds-to-nearest (base-3 rounding,
 *       odd divisor 6561, ties impossible) then saturates to ±MAX_VAL.
 *       This is the lossy direction; consumers needing exactness must
 *       stay in MTFP19 or pre-scale into MTFP4 range.
 *
 * Scalar arithmetic primitives (add/sub/neg/mul) are intentionally
 * absent. Under the substrate discipline ("no primitive without named
 * consumer demand"), they re-emerge when a consumer asks — and when
 * they do, they land with §8.5-compliant semantics (mul widens to
 * MTFP9 mantissa per §8.3, not a silent-rounding MTFP4).
 */

#ifndef M4T_MTFP4_H
#define M4T_MTFP4_H

#include "m4t_types.h"
#include "m4t_mtfp.h"   /* m4t_mtfp_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ── Saturating clamp (Case S per §8.5) ─────────────────────────────────── */

/* Narrow an int32 to an MTFP4 mantissa cell. Exact when |v| ≤ MAX_VAL_4;
 * saturates at ±MAX_VAL_4 otherwise. Used by the narrowing conversion. */
static inline m4t_mtfp4_t m4t_mtfp4_clamp(int32_t v) {
    if (v >  (int32_t)M4T_MTFP4_MAX_VAL) return  M4T_MTFP4_MAX_VAL;
    if (v < -(int32_t)M4T_MTFP4_MAX_VAL) return -M4T_MTFP4_MAX_VAL;
    return (m4t_mtfp4_t)v;
}

/* ── SDOT ternary matmul (§8.4 — Case W, exact) ─────────────────────────── */

/* Y[M,N] = X[M,K] @ W^T where X is MTFP4 and W is unpacked ternary (int8
 * in {-1, 0, +1}). Output Y is MTFP19 (int32 mantissas).
 *
 * This is Case W per §8.4: output type widens to MTFP19. Exact by
 * construction — for any |X[i,k]| ≤ MAX_VAL_4 = 40 and W[j,k] ∈ {-1,0,+1},
 *   |Y[i,j]| = |Σ_k X[i,k] · W[j,k]|
 *           ≤ K · 40
 * which fits MTFP19 (max 581 130 733) for K up to ~14.5 million.
 *
 * NEON path uses vdotq_s32 (16 int8 multiply-accumulates per instruction);
 * scalar tail handles K not divisible by 16.
 *
 * W layout: [N, K] row-major int8, each element in {-1, 0, +1}. NOT
 * packed trits — raw int8, because SDOT needs int8 operands. This uses
 * 4× the memory of the packed representation; the tradeoff is that SDOT
 * runs at full throughput with zero decode overhead.
 *
 * Preconditions (asserted in debug):
 *   M >= 0, N >= 0, K >= 0
 *   Y, X, W non-NULL when M·N·K > 0
 *   |X[i,k]| <= MAX_VAL_4
 *   W[j,k] in {-1, 0, +1}  (caller's responsibility; substrate trusts
 *                            at the boundary, samples in debug builds)
 *   Y must not alias X or W
 *
 * No flags parameter: by §8.4 contract this kernel does not saturate or
 * round under valid inputs. Consumers that violate the K bound get
 * undefined behavior at the int32 accumulator overflow point. */
void m4t_mtfp4_sdot_matmul_bt(
    m4t_mtfp_t*       Y,
    const m4t_mtfp4_t* X,
    const m4t_trit_t* W,
    int M, int K, int N
);

/* ── Cell-width conversion ──────────────────────────────────────────────── */

/* MTFP19 → MTFP4. Narrowing conversion under the default-block-exponent
 * convention: divides by SCALE_RATIO = 6561 = 3^(MTFP_RADIX − MTFP4_RADIX)
 * with base-3 round-to-nearest-even (the divisor is odd, so ties cannot
 * occur with integer mantissas — see §14.2 / m4t_mtfp.c rounding rule),
 * then clamps to ±MAX_VAL_4.
 *
 * Lossy in general; exact only when |src[i]| ≤ MAX_VAL_4 · 6561 = 262 440
 * AND src[i] is a multiple of 6561.
 *
 * Optional flag tracking via per-block layout (§14.4). NULL flags
 * disables tracking. ROUNDED bit set when the divide produces a non-zero
 * remainder; SATURATED bit set when the post-rescale magnitude exceeds
 * MAX_VAL_4. */
void m4t_mtfp19_to_mtfp4(
    m4t_mtfp4_t*      dst,
    const m4t_mtfp_t* src,
    uint8_t*          flags,    /* nullable, M4T_FLAG_BYTES(n) bytes */
    int               n
);

/* MTFP4 → MTFP19. Widening conversion. Multiplies each mantissa by
 * SCALE_RATIO = 6561; exact by construction (max output magnitude
 * 40 × 6561 = 262 440 ≪ MTFP19 max). No flags, no saturation. */
void m4t_mtfp4_to_mtfp19(
    m4t_mtfp_t*       dst,
    const m4t_mtfp4_t* src,
    int               n
);

#ifdef __cplusplus
}
#endif

#endif /* M4T_MTFP4_H */
