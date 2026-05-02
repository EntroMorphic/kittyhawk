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

#ifdef __cplusplus
}
#endif

#endif /* M4T_MTFP_H */
