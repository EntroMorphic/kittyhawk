/*
 * m4t_mtfp.h — MTFP19 mantissa-layer primitives
 *
 * M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
 *
 * The substrate's atomic unit is the BLOCK: exactly one NEON vector,
 * 16 bytes = 4 MTFP19 mantissa cells = one SDOT input lane. Every
 * operation here either operates on a single block (block-native) or
 * composes block operations over an aligned tensor (vec-native).
 *
 * Overflow resolution (per substrate §8.5):
 *   - These are fixed-output-type operations (dst cell is MTFP19;
 *     widening to MTFP39 would change the caller's buffer type).
 *   - Therefore they fall into Case S — SATURATE. Not widen, not
 *     round. Saturation at ±M4T_MTFP_MAX_VAL is informative (not
 *     silent) and flags can be tracked by consumers under §14.4.
 *
 * Same-block contract:
 *   - All cells passed to a single call are interpreted at one
 *     (unspecified) block exponent.
 *   - `m4t_mtfp_block_*` ops enforce this by construction: the
 *     signature is one block in, one block out.
 *   - `m4t_mtfp_vec_*` ops extend the contract to multiple blocks:
 *     the caller asserts the entire vector is a single logical
 *     tensor at one shared block exponent. The substrate cannot
 *     detect a violation; it trusts the caller at the boundary.
 *
 * Cross-block arithmetic across different block exponents is NOT
 * provided here. See M4T_SUBSTRATE.md §14.2 for the deferred
 * `m4t_mtfp_vec_add_aligning` opt-in variant.
 *
 * Input precondition:
 *   - Every cell argument must satisfy |mantissa| ≤ M4T_MTFP_MAX_VAL.
 *   - The substrate trusts this at the boundary; it does not range-check.
 *   - A compile-time assertion guarantees that the non-saturating SIMD
 *     add used internally is safe for in-range inputs (2·MAX_VAL fits
 *     comfortably in int32).
 *
 * Aliasing:
 *   - `block_add(dst, dst)` → dst = 2·dst, saturated per cell.
 *   - `block_sub(dst, dst)` → dst = 0.
 *   - Same for vec variants. Aliasing is well-defined.
 *
 * Consumer-demand trace:
 *   - `block_add` / `vec_add_inplace` : accumulator edge of
 *     m4t_route_apply_signed (signed tile accumulation).
 *   - `block_sub` / `vec_sub_inplace` : signed-minus branch of
 *     the same routing pass.
 *   - `vec_zero` : test harness + routing result pre-zeroing.
 *   - `clamp64`  : ternary matmul store (int64 accumulator → MTFP19).
 */

#ifndef M4T_MTFP_H
#define M4T_MTFP_H

#include "m4t_types.h"
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Substrate invariants. Static asserts catch config drift at compile time
 * so the non-saturating SIMD add used internally stays provably safe. */

_Static_assert(sizeof(m4t_mtfp_t) * M4T_MTFP_CELLS_PER_BLOCK == M4T_BLOCK_BYTES,
               "MTFP19 block must be exactly one NEON vector (16 bytes, 4 int32 cells)");
_Static_assert((int64_t)M4T_MTFP_MAX_VAL * 2 < (int64_t)0x7FFFFFFF,
               "Two in-range MTFP19 mantissas must sum within int32 without wrapping "
               "(so non-saturating SIMD add + min/max clamp is exact per §8.5 Case S)");

/* ── Scalar primitive ─────────────────────────────────────────────────────
 *
 * Saturating clamp of an int64 accumulator to an MTFP19 mantissa cell.
 * Exact when |v| ≤ M4T_MTFP_MAX_VAL; saturates at ±MAX_VAL otherwise
 * (§8.5 Case S). Used by ternary matmul to store its widened accumulator. */

static inline m4t_mtfp_t m4t_mtfp_clamp64(int64_t v) {
    if (v >  (int64_t)M4T_MTFP_MAX_VAL) return  M4T_MTFP_MAX_VAL;
    if (v < -(int64_t)M4T_MTFP_MAX_VAL) return -M4T_MTFP_MAX_VAL;
    return (m4t_mtfp_t)v;
}

/* ── Block-native primitives ──────────────────────────────────────────────
 *
 * Operate on exactly one MTFP19 block (M4T_MTFP_CELLS_PER_BLOCK = 4 cells).
 * The substrate's atomic unit. Every vec op is a composition of these. */

/* dst[0..4) += a[0..4), per cell, saturated at ±M4T_MTFP_MAX_VAL.
 * Same-block contract: dst and a share one block exponent. */
void m4t_mtfp_block_add(
    m4t_mtfp_t dst[M4T_MTFP_CELLS_PER_BLOCK],
    const m4t_mtfp_t a[M4T_MTFP_CELLS_PER_BLOCK]
);

/* dst[0..4) -= a[0..4), per cell, saturated at ±M4T_MTFP_MAX_VAL.
 * Same-block contract. */
void m4t_mtfp_block_sub(
    m4t_mtfp_t dst[M4T_MTFP_CELLS_PER_BLOCK],
    const m4t_mtfp_t a[M4T_MTFP_CELLS_PER_BLOCK]
);

/* ── Vec-native primitives ────────────────────────────────────────────────
 *
 * Compositions of block ops over an aligned tensor. Whole blocks are
 * processed via block ops; a scalar tail (fewer than 4 cells) is handled
 * with identical saturation semantics.
 *
 * Single-tensor contract: all n cells share one block exponent. The
 * substrate cannot verify this; the caller asserts it at the boundary. */

/* dst[0..n) = 0. */
void m4t_mtfp_vec_zero(m4t_mtfp_t* dst, int n);

/* dst[i] += a[i] for i in [0, n), saturated per cell. */
void m4t_mtfp_vec_add_inplace(m4t_mtfp_t* dst, const m4t_mtfp_t* a, int n);

/* dst[i] -= a[i] for i in [0, n), saturated per cell. */
void m4t_mtfp_vec_sub_inplace(m4t_mtfp_t* dst, const m4t_mtfp_t* a, int n);

/* ── Cross-exponent accumulator (§14.2 named opt-in) ──────────────────────
 *
 * The cross-block-exponent add policy from §14.2 of the substrate spec,
 * implemented as a stateful accumulator. This is the substrate's one
 * legitimate lossy path (§8.5 invariant: widen, don't round; this kernel
 * is the named exception that rounds, by explicit caller request).
 *
 * Accumulator semantics:
 *   running   — in-out mantissa buffer at *running_exp.
 *   addend    — new contribution mantissas at addend_exp.
 *   On return: running has accumulated decode(addend, addend_exp),
 *              re-encoded at *running_exp (which may have grown upward).
 *
 * Invariant maintained across calls: |running[i]| <= MAX_VAL at *running_exp.
 *
 * Three live cases plus two degenerate edges:
 *   addend_exp == *running_exp:
 *     Same-block accumulation. Reduces to vec_add_inplace semantics.
 *   addend_exp >  *running_exp:
 *     Running mantissas rescale DOWN by 3^Δ (Δ = addend_exp - *running_exp);
 *     *running_exp updates to addend_exp.
 *   addend_exp <  *running_exp:
 *     Addend mantissas rescale DOWN by 3^Δ; *running_exp unchanged.
 *   |Δ| >= 20 (degenerate, well-defined):
 *     The smaller-exponent side truncates to zero by the math; the kernel
 *     produces the larger side passed through. Not an error.
 *
 * Rounding rule (§8.2): base-3 round-to-nearest-even. Because the divisor
 * s = 3^Δ is always odd (proven invariant of the M4T_POW3_TABLE), the
 * halfway point s/2 cannot occur as an integer remainder; ties are
 * impossible and round-to-nearest is unambiguous. The "even" tie-break
 * specified in §8.2 is satisfied vacuously. Worst-case per-call rounding
 * error in real-number space is bounded by (s-1)/(2s) · 3^result_exp,
 * strictly less than (1/2) · 3^result_exp.
 *
 * Saturation: per-cell, post-add. Path A alignment (max-exponent target)
 * preserves the dominant magnitude; smaller operand vanishes when |Δ| is
 * large.
 *
 * Flag tracking (§14.4 status array, opt-in via non-NULL flags pointer):
 *   Layout: ONE BYTE PER MTFP19 BLOCK (4 cells per block per §7).
 *   For an n-cell tensor, flags has M4T_FLAG_BYTES(n) bytes. Each byte
 *   encodes two events × four cells:
 *
 *     bits 0-1: cell 0 of block — bit 0 SATURATED, bit 1 ROUNDED
 *     bits 2-3: cell 1 of block — bit 2 SATURATED, bit 3 ROUNDED
 *     bits 4-5: cell 2 of block — bit 4 SATURATED, bit 5 ROUNDED
 *     bits 6-7: cell 3 of block — bit 6 SATURATED, bit 7 ROUNDED
 *
 *   Cells beyond n in the trailing partial block are unused and the
 *   kernel does not touch their flag bits.
 *
 *   Sticky-OR semantics: bits set during any call are preserved across
 *   subsequent calls. Caller initializes via
 *   memset(flags, 0, M4T_FLAG_BYTES(n)) and clears manually as needed.
 *
 * Aliasing: running and addend MUST NOT alias each other. The pairwise
 * wrapper additionally forbids dst aliasing b (only dst==a is permitted
 * via the wrapper's internal copy).
 *
 * Preconditions (asserted in debug):
 *   n >= 0
 *   running, addend, running_exp non-NULL when n > 0
 *   |running[i]|, |addend[i]| <= M4T_MTFP_MAX_VAL (MTFP19 substrate invariant)
 */

/* Per-cell event masks. Used by the accessor macros below to test which
 * events fired for a particular cell in a per-block flag byte. */
#define M4T_FLAG_SATURATED  ((uint8_t)0x01)
#define M4T_FLAG_ROUNDED    ((uint8_t)0x02)

/* Number of bytes in a per-block flag array for a tensor of n cells.
 * Rounds up so partial trailing blocks get one byte of storage. */
#define M4T_FLAG_BYTES(n) \
    (((n) + M4T_MTFP_CELLS_PER_BLOCK - 1) / M4T_MTFP_CELLS_PER_BLOCK)

/* Test whether `event` fired for cell `cell_index` in the per-block
 * flag array. Non-zero return means yes. */
static inline int m4t_flag_test(
    const uint8_t* flags, int cell_index, uint8_t event)
{
    int block = cell_index / M4T_MTFP_CELLS_PER_BLOCK;
    int slot  = cell_index % M4T_MTFP_CELLS_PER_BLOCK;
    return (flags[block] >> (slot * 2)) & event;
}

void m4t_mtfp_vec_accum_aligning(
    m4t_mtfp_t*       running,
    int8_t*           running_exp,    /* in-out */
    const m4t_mtfp_t* addend,
    int8_t            addend_exp,
    uint8_t*          flags,          /* nullable, M4T_FLAG_BYTES(n) bytes */
    int               n
);

/* Scalar-only reference. Same semantics as m4t_mtfp_vec_accum_aligning;
 * never dispatches to NEON. Test-only verification oracle exposed so the
 * bit-exact gate has a stable reference even after the production path
 * is replaced with NEON. Per shift3 remediation methodology lifted to
 * project rule.
 *
 * Production code MUST NOT call this — intentionally slower than the
 * production NEON path. Verification only. Per
 * journal/cross_exp_accum_routing_synthesize.md A-G1. */
void m4t_mtfp_vec_accum_aligning_scalar_ref(
    m4t_mtfp_t*       running,
    int8_t*           running_exp,
    const m4t_mtfp_t* addend,
    int8_t            addend_exp,
    uint8_t*          flags,
    int               n
);

/* (R-G3 remediation: removed m4t_mtfp_vec_accum_aligning_neon prototype
 * wrapper. Its body is now inlined directly into the production
 * m4t_mtfp_vec_accum_aligning dispatcher.) */

/* Convenience pairwise wrapper. dst gets a + b at exponent
 * max(e_a, e_b), with rounding/saturation flags.
 *
 * The accumulator is the canonical primitive; this wrapper exists for
 * call sites that genuinely have two distinct buffers and one shot.
 *
 * Aliasing: dst may alias a (the wrapper handles the copy internally);
 * dst MUST NOT alias b. dst==b is asserted-against in debug builds.
 *
 * out_e is nullable — pass NULL if the caller does not need the result
 * exponent (it is deterministic from inputs, max(e_a, e_b)). */
void m4t_mtfp_vec_add_aligning(
    m4t_mtfp_t*       dst,
    int8_t*           out_e,           /* nullable */
    const m4t_mtfp_t* a, int8_t        e_a,
    const m4t_mtfp_t* b, int8_t        e_b,
    uint8_t*          flags,
    int               n
);

/* Pairwise subtract wrapper. Equivalent to add_aligning(dst, &e, a, e_a,
 * neg(b), e_b, flags, n) without the temporary buffer for neg(b). */
void m4t_mtfp_vec_sub_aligning(
    m4t_mtfp_t*       dst,
    int8_t*           out_e,           /* nullable */
    const m4t_mtfp_t* a, int8_t        e_a,
    const m4t_mtfp_t* b, int8_t        e_b,
    uint8_t*          flags,
    int               n
);

/* ── shift3: base-3 positional shift (elemental floor primitive) ─────────
 *
 * Per journal/elemental_floor_synthesize.md. shift3 is one of the ~5
 * elemental ops that can't be built from the others (along with add, neg,
 * sign, select). It's the natural base-3 scaling primitive.
 *
 * Semantics: dst[i] = src[i] * 3^k, with saturation on positive k overflow
 * and base-3 round-to-nearest-even on negative k.
 *
 *   k > 0  : multiply mantissa by 3^k. Saturates at ±MAX_VAL_MTFP19 if the
 *            scaled value would exceed it.
 *   k < 0  : divide mantissa by 3^|k|. Round-to-nearest-even; 3^|k| is odd
 *            so halfway ties cannot occur at integer mantissas (same
 *            invariant as the cross-exponent accumulator's m4t_pow3_round_div).
 *   k = 0  : identity copy.
 *   |k| > 19: clamps. Positive: collapses to ±MAX_VAL or 0. Negative: 0.
 *
 * Substrate-discipline: this is a positional/scaling operation, not
 * arithmetic. Implementation uses the M4T_POW3_TABLE constants directly.
 *
 * Aliasing: dst may alias src.
 *
 * Preconditions:
 *   dst, src non-NULL when n > 0.
 *   |src[i]| ≤ M4T_MTFP_MAX_VAL (substrate invariant).
 */
void m4t_mtfp_shift3(
    m4t_mtfp_t* dst,
    const m4t_mtfp_t* src,
    int k,
    int n
);

/* Scalar-only reference. Same semantics as m4t_mtfp_shift3 above; never
 * dispatches to the NEON path even when M4T_HAS_NEON. Exposed for tests:
 * after productionization, m4t_mtfp_shift3 runs NEON for the divide
 * direction, so a bit-exact gate needs an independent scalar oracle.
 *
 * Production code MUST NOT call this — it is intentionally slower than
 * m4t_mtfp_shift3 and exists solely for verification. Per
 * journal/shift3_neon_redteam.md C1/C2/C3 + remediation R-G1. */
void m4t_mtfp_shift3_scalar_ref(
    m4t_mtfp_t* dst,
    const m4t_mtfp_t* src,
    int k,
    int n
);

/* ── Integer rsqrt (Newton-Raphson, fixed-point) ────────────────────────
 *
 * Per journal/rsqrt_design_lmm.md (compressed LMM cycle 2026-05-06).
 *
 * Computes `dst = round(2^30 / sqrt(src))` for src ≥ 1.
 * Output range: roughly [23170, 2^30]. Special case: src ≤ 0 returns 0
 * (caller's responsibility to add an ε to prevent division by zero).
 *
 * Algorithm: Newton-Raphson with integer initial guess derived from
 * __builtin_clz (count leading zeros gives floor(log2(src))). 3 iterations
 * from this guess deliver bit-exact int32 precision.
 *
 * Bit-exact NEON-vs-scalar_ref: both paths share the same initial-guess
 * formula and iteration. The output is integer-rounded each step
 * identically. NEON path is structured for future per-vector use but
 * currently single-value (rsqrt of a single positive int doesn't
 * naturally vectorize at the kernel level).
 *
 * Caller usage pattern (e.g., RMSNorm):
 *   int64_t sum_sq = ...;            // sum of x² values
 *   int32_t mean = (int32_t)(sum_sq / n);
 *   m4t_mtfp_t inv = m4t_int32_rsqrt(mean + eps);
 *   for each i:
 *     y[i] = clamp_mtfp((int64_t)gamma[i] * x[i] * inv >> 30); */
m4t_mtfp_t m4t_int32_rsqrt(m4t_mtfp_t src);

/* Scalar-only reference. Bit-exact identical output to m4t_int32_rsqrt;
 * exists for the verification gate. Production code MUST NOT call this. */
m4t_mtfp_t m4t_int32_rsqrt_scalar_ref(m4t_mtfp_t src);

/* ── RMSNorm (BitNet's normalization, Llama-family standard) ────────────
 *
 * y[i] = γ[i] · x[i] · rsqrt(mean(x²) + ε)
 *
 * All buffers are MTFP19 mantissas (int32). γ length n, x length n,
 * y length n. eps_mantissa is added to the (shifted-down) mean of
 * squares — caller manages units.
 *
 * Implementation: int64 sum-of-squares with right-shift to avoid
 * overflow at large MTFP19 inputs (n=2560 cells × MTFP19_MAX² ~ 2^70
 * exceeds int64 max). Shift compensated when applying rsqrt.
 *
 * Per-cell γ × x × rsqrt uses __int128 intermediate (the 3-way product
 * can exceed int64 for adversarial inputs). This is a per-cell scalar
 * loop — full NEON SIMD vectorization is deferred because int128
 * exceeds NEON int lane width. Per the cross-exp accum's degenerate-case
 * precedent, scalar-with-documented-reasoning is acceptable here.
 *
 * Saturating clamp on output (Case S; mantissa fits MTFP19).
 *
 * Aliasing: y == x and y == γ are both supported (read-modify on each
 * cell does not depend on later cells).
 *
 * Constraint: n ≤ 7100 with full-MTFP19_MAX-magnitude cells (sum_sq
 * overflows int64 above that). BitNet's max n is 6912 (FFN
 * intermediate), within bounds. Larger n requires a shifted-fold
 * accumulator — out of scope for Phase 1.
 *
 * Sign bias: `x >> SOS_SHIFT` is arithmetic shift, biased toward -∞ for
 * negative `x`. For symmetric random inputs the bias is bounded by
 * tolerance; for adversarial all-negative inputs the SoS sum is
 * inflated by ≤ 2× per cell. Acceptable for normalization; not
 * acceptable if the caller needs bit-exact symmetric behavior.
 *
 * Per journal/rsqrt_design_lmm.md (work-unit 2 of bitnet_phase1). */
void m4t_mtfp_rmsnorm(
    m4t_mtfp_t* y,
    const m4t_mtfp_t* x,
    const m4t_mtfp_t* gamma,
    m4t_mtfp_t eps_mantissa,
    int n
);

/* Scalar reference using libm sqrt for the rsqrt step. FP allowed in
 * scaffolding. Not bit-exact vs production (NR vs FP rounding); used
 * as a precision gate via tolerance comparison. */
void m4t_mtfp_rmsnorm_scalar_ref(
    m4t_mtfp_t* y,
    const m4t_mtfp_t* x,
    const m4t_mtfp_t* gamma,
    m4t_mtfp_t eps_mantissa,
    int n
);

#ifdef __cplusplus
}
#endif

#endif /* M4T_MTFP_H */
