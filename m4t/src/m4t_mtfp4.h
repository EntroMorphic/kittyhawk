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

/* Maximum K for which m4t_mtfp4_sdot_matmul_bt produces in-range MTFP19
 * output. Derivation: the worst-case absolute output is
 *   max|Y[i,j]| = max|Σ_k X[i,k] · W[j,k]| = K · MAX_VAL_MTFP4 · 1 = 40K
 * which must satisfy 40K ≤ MAX_VAL_MTFP19 = 581 130 733. So:
 *   K_max_exact = MAX_VAL_MTFP19 / MAX_VAL_MTFP4 = 14 528 268
 *
 * Beyond this bound, the kernel produces mantissas that exceed MTFP19's
 * documented range. The substrate-invariant |cell| ≤ MAX_VAL is violated;
 * downstream kernels reading those cells will produce garbage. Callers
 * exceeding K_max_exact must partition into K-sized chunks, accumulate
 * across chunks via the cross-exponent accumulator, OR use a wider-output
 * SDOT variant (lands MTFP4 × ternary → MTFP39; not yet built). */
#define M4T_SDOT_K_MAX_EXACT \
    ((int)(M4T_MTFP_MAX_VAL / M4T_MTFP4_MAX_VAL))

/* Y[M,N] = X[M,K] @ W^T where X is MTFP4 and W is unpacked ternary (int8
 * in {-1, 0, +1}). Output Y is MTFP19 (int32 mantissas).
 *
 * This is Case W per §8.4: output type widens to MTFP19. Exact by
 * construction for K ≤ M4T_SDOT_K_MAX_EXACT (= 14 528 268).
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
 *   K <= M4T_SDOT_K_MAX_EXACT  (HARD: substrate-invariant violation if exceeded)
 *   Y, X, W non-NULL when M·N·K > 0
 *   |X[i,k]| <= MAX_VAL_4
 *
 *     **Accepted-input-class extension:** ternary activations (X[i,k] ∈
 *     {-1, 0, +1}, semantically `m4t_trit_t`) are also accepted, since
 *     `m4t_trit_t` and `m4t_mtfp4_t` share the underlying int8_t and
 *     ternary values are a strict subset of MTFP4's mantissa range.
 *     Callers in the ternary × ternary regime should prefer the
 *     dedicated wrapper `m4t_ternary_dot_matmul_bt` (m4t_ternary_matmul.h)
 *     which exposes ternary × ternary as a first-class API with a
 *     compile-time bit-compatibility static_assert. The SDOT path
 *     itself is identical for both input classes.
 *
 *   W[j,k] in {-1, 0, +1}
 *     The substrate trusts the caller at the boundary; debug builds
 *     spot-check W[0] and the last cell of W[N-1] only — exhaustive
 *     validation would scan O(N·K) per call, too expensive for the
 *     hot path. Consumers that need exhaustive validation should run
 *     it once at W setup time, outside the matmul loop.
 *   Y must not alias X or W
 *
 * No flags parameter: by §8.4 contract this kernel does not saturate or
 * round under valid inputs (where "valid" includes K ≤ M4T_SDOT_K_MAX_EXACT). */
void m4t_mtfp4_sdot_matmul_bt(
    m4t_mtfp_t*       Y,
    const m4t_mtfp4_t* X,
    const m4t_trit_t* W,
    int M, int K, int N
);

/* Routing-shaped sibling (V3 of the pure-ternary audit). Same I/O and
 * bit-exact output as m4t_mtfp4_sdot_matmul_bt; inner compute is
 * dispatch-shaped (mask + select), no SDOT.
 *
 * Per cell: dispatch on W trit value via vceqq + vandq + vsubq; reduce
 * via vaddlvq. K%16 boundary tile uses stack-local zero-padded W and X.
 *
 * Architecture compliance per
 * memory/feedback_pure_ternary_routed_architecture.md:
 *   (1) pure ternary, (2) routed, (3) non-dense, (4) no binary
 *   structures, (5) no scalar ops.
 *
 * X = m4t_mtfp4_t (int8, range [-40, +40] per MTFP4 contract). W is
 * ternary {-1, 0, +1}. Per-lane diff = pos_sel - neg_sel ∈ [-40, +40]
 * (mutually exclusive masks), so int8 is sufficient — no INT8_MIN
 * concern as in the 5-in-8 _bt_route case.
 *
 * Speed: bench-uncertain — the K%16 fix in the original kernel showed
 * the boundary-tile rewrite was wash for mtfp4_sdot. Whether the
 * dispatch path delivers proportional cost depends on the same
 * factors. Bench in journal/v3_mtfp4_sdot_route_bench.txt. */
void m4t_mtfp4_sdot_matmul_bt_route(
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

/* Scalar-only references for the conversion functions. Same semantics
 * as the public dispatchers; never dispatch to NEON. Test-only oracles
 * for bit-exact verification. Production code MUST NOT call these. */
void m4t_mtfp19_to_mtfp4_scalar_ref(
    m4t_mtfp4_t*      dst,
    const m4t_mtfp_t* src,
    uint8_t*          flags,
    int               n
);
void m4t_mtfp4_to_mtfp19_scalar_ref(
    m4t_mtfp_t*       dst,
    const m4t_mtfp4_t* src,
    int               n
);

#ifdef __cplusplus
}
#endif

#endif /* M4T_MTFP4_H */
