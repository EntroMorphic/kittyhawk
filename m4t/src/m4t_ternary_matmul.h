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

/* Scalar-only reference. Same semantics as m4t_mtfp_ternary_matmul_bt
 * above; never dispatches to NEON. Exposed for tests so the bit-exact
 * verification gate has a stable oracle even after productionization
 * replaces the production kernel's inner loop. Per
 * journal/shift3_neon_remediation_closeout.md (cycle-level lesson lifted)
 * + journal/ternary_mac_routing_synthesize.md T-G2.
 *
 * Production code MUST NOT call this — intentionally slower than the
 * NEON path. Verification only. */
void m4t_mtfp_ternary_matmul_bt_scalar_ref(
    m4t_mtfp_t*     Y,
    const m4t_mtfp_t* X,
    const uint8_t*  W_packed,
    uint8_t*        flags,
    int M, int K, int N
);

/* Ternary × ternary → MTFP19 via SDOT.
 *
 * Both activations and weights are unpacked ternary trits. This is the
 * canonical kernel for substrate consumers projecting ternary signatures
 * through ternary projection matrices (gesh's hot path).
 *
 * Implementation: delegates to `m4t_mtfp4_sdot_matmul_bt`. Ternary
 * mantissas {-1, 0, +1} are a strict subset of MTFP4's range ±40, so
 * the SDOT path applies without scope reach. Both inputs are `int8_t`
 * at the bit level; the SDOT instruction (`vdotq_s32`) processes
 * 16-lane int8 × int8 → int32 accumulate per cycle on Apple Silicon.
 *
 * Use this rather than `m4t_mtfp_ternary_matmul_bt` when activations
 * are ternary; that kernel's vmlal_s32-routed inner loop (~18 NEON ops
 * per 16-trit block, post the ternary_mac_routing cycle) still pays
 * for int32×int32 multiply-accumulate, while SDOT does 16 int8×int8
 * MACs in a single 1-cycle instruction. ~17× more throughput when
 * activations fit in int8.
 *
 * Y[M, N] = X[M, K] @ W^T[K, N], all ternary.
 *
 * Output is `m4t_mtfp_t` (int32). Per-row max |acc| ≤ K, so for
 * K ≤ M4T_SDOT_K_MAX_EXACT no clamp is possible — the output is
 * exact integer dot product.
 *
 * Preconditions:
 *   K ≤ M4T_SDOT_K_MAX_EXACT (HARD: bound from m4t_mtfp4.h)
 *   X, W contain only valid trit codes {-1, 0, +1}
 *   Y must not alias X or W
 */
void m4t_ternary_dot_matmul_bt(
    m4t_mtfp_t* Y,
    const m4t_trit_t* X,
    const m4t_trit_t* W,
    int M, int K, int N
);

/* §20 sub-2-bit packed ternary × ternary → MTFP19 matmul.
 *
 * Y[M, N] = X[M, K] @ W^T[N, K], where:
 *   X is unpacked ternary (m4t_trit_t = int8, values in {-1, 0, +1});
 *   W is 5-trits-in-8-bits packed (1.6 bits/cell), per §20 encoding;
 *   Y is MTFP19 (int32).
 *
 * Per M4T_SUBSTRATE.md §20: sub-2-bit base-3 packing as opt-in dense
 * format. Storage savings ~5× over unpacked, ~1.25× over the default
 * 4-in-8 packing.
 *
 * Packed W layout: row-major. Each row holds K trits in
 * M4T_TRIT_PACKED5_BYTES(K) = (K+4)/5 bytes (5 trits per byte).
 *
 * Implementation: NEON-only, register-tile by 4 j cells, split-LUT decode
 * (1× div-by-9 magic-multiply + 5× vqtbl1q/vqtbl2q lookups per byte).
 * Per audit Path D + journal/m4t_5in8_synthesize.md.
 *
 * Output: per-cell |acc| ≤ K (each MAC is in {-1, 0, +1}). For
 * K ≤ M4T_SDOT_K_MAX_EXACT, output fits MTFP19 by construction (Case W).
 *
 * Preconditions (asserted in debug):
 *   M >= 0, K >= 0, N >= 0;
 *   K % 80 == 0 (NEON inner-block alignment; no scalar tail per project rule);
 *   N % 4 == 0 (register-tile alignment; no untiled tail);
 *   K <= M4T_SDOT_K_MAX_EXACT (no overflow into int32 output);
 *   X and W contain only valid trit codes;
 *   Y, X, W non-NULL when M*N*K > 0;
 *   Y must not alias X or W.
 *
 * Strict alignment is intentional and matches the audit's verified shape.
 * Real consumers with non-aligned (K, N) should pad to the next multiple
 * of 80 / 4 (the trailing trits/cells contribute 0 since pack zero-pads).
 * Future work: K%80 + N%4 tail handling for non-aligned shapes — would
 * mirror Item 1's tile-with-tail pattern, deferred until a consumer
 * demands it. */
void m4t_ternary_5in8_matmul_bt(
    m4t_mtfp_t* Y,
    const m4t_trit_t* X,
    const uint8_t* W_packed,
    int M, int K, int N
);

/* Scalar-only reference. Same semantics as m4t_ternary_5in8_matmul_bt;
 * never dispatches to NEON. Test-only verification oracle.
 * Production code MUST NOT call this — intentionally slower.
 * Per project pattern + journal/m4t_5in8_synthesize.md. */
void m4t_ternary_5in8_matmul_bt_scalar_ref(
    m4t_mtfp_t* Y,
    const m4t_trit_t* X,
    const uint8_t* W_packed,
    int M, int K, int N
);

#ifdef __cplusplus
}
#endif

#endif /* M4T_TERNARY_MATMUL_H */
