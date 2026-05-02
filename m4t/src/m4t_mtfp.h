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
 * Rounding rule (§8.2): base-3 round-to-nearest. Powers of 3 are odd, so
 * "halfway" cases (remainder == s/2) are impossible with integer mantissas;
 * round-to-nearest is unambiguous. Worst-case per-call truncation in real-
 * number space is bounded by (s-1)/(2s) * 3^result_exp < (1/2) * 3^result_exp.
 *
 * Saturation: per-cell, post-add. Path A alignment (max-exponent target)
 * preserves the dominant magnitude; smaller operand vanishes when |Δ| is
 * large. This is the §8.2 "rounding required" path made explicit.
 *
 * Flag tracking (§14.4 status array, opt-in via non-NULL flags pointer):
 *   bit 0 (M4T_FLAG_SATURATED) — saturation occurred for this cell in any
 *                                call so far. Sticky.
 *   bit 1 (M4T_FLAG_ROUNDED)   — rescale produced a non-zero remainder for
 *                                this cell in any call so far. Sticky.
 *   Bits 2-7: reserved, must be zero on input from caller.
 *
 * Sticky semantics: the kernel never clears flag bits; consumer is
 * responsible for initialization (memset to zero) and any clearing.
 *
 * Aliasing: running and addend MUST NOT alias. running may equal flags's
 * underlying buffer iff the consumer passes flags as NULL.
 *
 * Preconditions (asserted in debug):
 *   n >= 0
 *   running, addend, running_exp non-NULL when n > 0
 *   |running[i]|, |addend[i]| <= M4T_MTFP_MAX_VAL (MTFP19 substrate invariant)
 */

#define M4T_FLAG_SATURATED  ((uint8_t)0x01)
#define M4T_FLAG_ROUNDED    ((uint8_t)0x02)

void m4t_mtfp_vec_accum_aligning(
    m4t_mtfp_t*       running,
    int8_t*           running_exp,    /* in-out */
    const m4t_mtfp_t* addend,
    int8_t            addend_exp,
    uint8_t*          flags,          /* nullable, length n */
    int               n
);

/* Convenience pairwise wrapper. dst gets a + b at exponent
 * max(e_a, e_b), with rounding/saturation flags. Equivalent to:
 *   if (dst != a) memcpy(dst, a, n * sizeof(m4t_mtfp_t));
 *   *out_e = e_a;
 *   m4t_mtfp_vec_accum_aligning(dst, out_e, b, e_b, flags, n);
 *
 * The accumulator is the canonical primitive; this wrapper exists for
 * call sites that genuinely have two distinct buffers and one shot.
 *
 * Aliasing: dst may alias a; dst MUST NOT alias b. */
void m4t_mtfp_vec_add_aligning(
    m4t_mtfp_t*       dst,
    int8_t*           out_e,           /* result exponent, written */
    const m4t_mtfp_t* a, int8_t        e_a,
    const m4t_mtfp_t* b, int8_t        e_b,
    uint8_t*          flags,
    int               n
);

#ifdef __cplusplus
}
#endif

#endif /* M4T_MTFP_H */
