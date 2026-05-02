/*
 * m4t_ternary_matmul.h — MTFP19 activations × packed ternary weights → MTFP19
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * The routing-native matmul for MTFP19 activations. Law #7 in action:
 * ternary projections (weights) applied to MTFP data (activations).
 * Not a dense matmul — the weights are {-1, 0, +1}, so 1/3 of every row
 * is zero by construction and the inner loop is conditional negate-and-add.
 *
 * Contract:
 *   - Activations X are m4t_mtfp_t mantissas (MTFP19).
 *   - Weights W are 2-bit packed trits in {-1, 0, +1}.
 *   - Output Y is m4t_mtfp_t mantissas (MTFP19).
 *   - Accumulation is int64; the final store is saturating clamp to
 *     ±MAX_VAL_MTFP19 (Case S per §8.5 — fixed-output type, the result
 *     cannot widen without changing the caller's buffer).
 *   - Output mantissa lives at the same block_exp as the input
 *     activation mantissa (weights are pure ternary, carry no scale).
 *
 * Case S vs §8.4's Case W: this kernel is NOT SDOT-shaped (SDOT is
 * int8 × int8 → int32). The SDOT-native path is MTFP4 × ternary →
 * MTFP19, in m4t_mtfp4_sdot_matmul_bt. THIS kernel is for consumers
 * that need full MTFP19 precision on activations.
 *
 * Hardware shape:
 *   - Inner loop decodes 16 trits per iteration, conditionally negates
 *     MTFP19 activations via vbslq_s32 + vnegq_s32, widen-accumulates
 *     into int64.
 *
 * Saturation tracking (§14.4): optional per-block status array. If the
 * int64 accumulator overflows MTFP19 mantissa range, the post-clamp
 * stores ±MAX_VAL and the corresponding cell's SATURATED bit is set
 * (sticky-OR). ROUNDED is never set by this kernel (no rescale). NULL
 * disables tracking. */

#ifndef M4T_TERNARY_MATMUL_H
#define M4T_TERNARY_MATMUL_H

#include "m4t_types.h"
#include "m4t_mtfp.h"   /* M4T_FLAG_BYTES, M4T_FLAG_SATURATED */

#ifdef __cplusplus
extern "C" {
#endif

/* Y[M,N] = X[M,K] @ W^T[K,N]
 *
 * W is stored row-major [N, K] as packed trits: W_packed[j] is a row of
 * Kp = M4T_TRIT_PACKED_BYTES(K) bytes holding K trits LSB-first.
 *
 * Output is MTFP19 (int32 cells), accumulated internally in int64 and
 * clamped to ±MAX_VAL_MTFP19 on store.
 *
 * Optional per-cell saturation tracking via flags. Layout: per-block
 * (1 byte per 4-cell MTFP19 block) over the flattened M·N output. Pass
 * NULL to disable. Sticky-OR'd; caller initializes via memset.
 *
 * Preconditions (asserted in debug):
 *   M >= 0, K >= 0, N >= 0
 *   Y, X, W_packed non-NULL when M·N·K > 0
 *   |X[i,k]| <= MAX_VAL_MTFP19
 *   W contains only valid trit codes (0b00, 0b01, 0b10) — the 0b11 code
 *   is undefined per m4t_trit_pack.h
 *   Y must not alias X or W_packed
 */
void m4t_mtfp_ternary_matmul_bt(
    m4t_mtfp_t*     Y,
    const m4t_mtfp_t* X,
    const uint8_t*  W_packed,
    uint8_t*        flags,        /* nullable, M4T_FLAG_BYTES(M*N) bytes */
    int M, int K, int N
);

#ifdef __cplusplus
}
#endif

#endif /* M4T_TERNARY_MATMUL_H */
