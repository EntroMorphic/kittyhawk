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
 *   K <= M4T_SDOT_K_MAX_EXACT (no overflow into int32 output);
 *   X and W contain only valid trit codes;
 *   Y, X, W non-NULL when M*N*K > 0;
 *   Y must not alias X or W.
 *
 * Alignment recommendation (NOT a precondition): K % 80 == 0 and N % 4 == 0
 * keep the entire computation in the SDOT tile body (5 SDOTs × 4 j cells per
 * 80-trit chunk). For K % 80 != 0, the trailing K%80 trits are processed by
 * a per-trit scalar tail (geometric scalar tail per project rule — sub-block
 * tails are allowed; main path remains NEON-only). For N % 4 != 0, the
 * trailing 1-3 j cells are processed by a single-acc NEON inner loop.
 * Both tails are bit-exact vs the scalar reference; non-aligned shapes are
 * functionally correct but pay a small per-call overhead proportional to
 * the tail size. */
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

/* Sparse-routed variant — first explicitly-routed compute primitive in
 * the substrate. Same math as m4t_ternary_5in8_matmul_bt; same bit-exact
 * output. The difference: this walks the 5-in-8 byte stream and treats
 * each trit as a *route decision* rather than a numeric coefficient:
 *
 *   trit = 0   → connection is absent; skip X[i, k] load entirely.
 *   trit = +1  → forward X[i, k] to accumulator with positive sign.
 *   trit = -1  → forward X[i, k] to accumulator with negated sign.
 *
 * No multiplications are performed. The substrate's existing primitives
 * already encoded routing structure in their data (the W matrix); this
 * primitive recognizes and uses that structure rather than computing
 * through it densely.
 *
 * For BitNet's weight distribution (~40% zero trits), this skips ~40%
 * of the work — fewer X loads, fewer adds, zero multiplies. Bit-exact
 * vs dense because integer addition of zeros is a no-op.
 *
 * Optional output: if `skipped_zeros` != NULL, writes the count of
 * zero trits encountered (across all M × N output cells). Useful for
 * sparsity measurement.
 *
 * Per the project memory's "math as signatures via routing" foundation. */
void m4t_ternary_5in8_matmul_bt_routed(
    m4t_mtfp_t* Y,
    const m4t_trit_t* X,
    const uint8_t* W_packed,
    int M, int K, int N,
    int64_t* skipped_zeros  /* optional; pass NULL to skip counting */
);

/* Per TD-7: §20 sibling with X also packed 5-in-8 (sub-2-bit X-packing).
 *   Y = X @ W^T  where both X and W are §20-packed.
 *
 * X_packed: M rows × M4T_TRIT_PACKED5_BYTES(K) = (K+4)/5 bytes (row-major).
 * W_packed: N rows × M4T_TRIT_PACKED5_BYTES(K) bytes (row-major; W^T layout
 *   so each row j holds K trits of W's column j — same convention as
 *   m4t_ternary_5in8_matmul_bt).
 * Y: M × N int32 outputs.
 *
 * Implementation: NEON-only. Per i, X_packed[i, :] is decoded into 5
 * stride-aligned int8 arrays via the same split-LUT pattern used for W
 * (1× div-by-9 magic-multiply + 5× vqtbl1q/vqtbl2q lookups per byte).
 * Then the tile body runs identically to §20.
 *
 * Same arbitrary-(K,N) support as §20 (TD-1 relaxation): K%80 trailing
 * trits handled by per-trit scalar geometric tail; N%4 trailing j cells
 * handled by single-acc NEON inner loop.
 *
 * Preconditions identical to §20 (modulo the X type difference).
 *
 * **Production guidance (per `journal/td7_xpacked_bench.md`):**
 * §20-xp BEATS §20 at every tested (M, K) — wall-clock ratio xp/§20 in
 * [0.74, 0.86] across M ∈ [1, 4096], K ∈ [1280, 12800]. The mechanism
 * (post-bench analysis): §20-xp's NEON-vectorized X permutation is
 * faster than §20's scalar X-permute. Recommend §20-xp as the default
 * packed kernel; §20 (W-only-packed) is dominated and kept for
 * backwards compatibility only.
 *
 * Vs unpacked dot (`m4t_ternary_dot_matmul_bt`): regime-dependent.
 *   M=1 (single-token inference), K ≥ 4480: §20-xp WINS (xp/dot 0.47-0.86).
 *   M ≥ 8 (batched): unpacked dot wins (xp/dot 1.05-1.5).
 *   Storage / bandwidth bound: §20-xp (5× X savings × 5× W savings
 *     = 25× total bandwidth reduction). */
void m4t_ternary_5in8_matmul_xpacked_bt(
    m4t_mtfp_t* Y,
    const uint8_t* X_packed,
    const uint8_t* W_packed,
    int M, int K, int N
);

/* Scalar-only reference oracle for the X-packed variant. Test-only. */
void m4t_ternary_5in8_matmul_xpacked_bt_scalar_ref(
    m4t_mtfp_t* Y,
    const uint8_t* X_packed,
    const uint8_t* W_packed,
    int M, int K, int N
);

#ifdef __cplusplus
}
#endif

#endif /* M4T_TERNARY_MATMUL_H */
