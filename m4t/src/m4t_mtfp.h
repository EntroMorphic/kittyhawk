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

/* int32 × int32 → int64 dot product. NEON-only production
 * (vmlal_s32 chain). Returns sum_{i=0}^{n-1} x[i] * y[i] as int64.
 *
 * V13.B of pure-ternary audit: replaces scalar dot loops in
 * bitnet_lm_head and bitnet_argmax_full_vocab. No scalar tail —
 * boundary handled via stack-local zero-padded 4-element buffers.
 *
 * Bound analysis: per product fits int64 (max |MTFP19_MAX|² ≈ 3.4e17).
 * Sum fits int64 for n × 3.4e17 < 9.2e18, i.e. n ≤ 27 in the
 * worst-case-saturated sense. In practice activations are well
 * below MTFP19 max so larger n is fine; the substrate doesn't
 * verify this. Caller responsible if both operands hover near
 * MTFP19_MAX over very large n (matches the prior scalar
 * implementation's overflow behavior).
 *
 * Preconditions:
 *   n >= 0 (n==0 returns 0)
 *   x, y non-NULL when n > 0
 *   x and y point to disjoint or aliasing memory (no aliasing
 *     constraint — operation is read-only on both inputs) */
int64_t m4t_mtfp_vec_dot_i64(const m4t_mtfp_t* x, const m4t_mtfp_t* y, int n);

/* Scalar-only reference. Test oracle for m4t_mtfp_vec_dot_i64.
 * Production code MUST NOT call this. */
int64_t m4t_mtfp_vec_dot_i64_scalar_ref(const m4t_mtfp_t* x, const m4t_mtfp_t* y, int n);

/* Attention V output projection:
 *   y[d] = clamp64(sum_t w[t] * V[t][d] >> shift)
 * where V[t][d] = V_base[(size_t)t * v_stride + d].
 *
 * V14.B of pure-ternary audit: replaces the scalar (d, t) loop in
 * bitnet_forward_block's attention path. NEON-only production. The
 * inner loop is over d (contiguous in V); w[t] is broadcast (vdup_n_s32)
 * and accumulated into int64x2 lanes via vmlal_s32. Shift+clamp pass
 * uses vshlq_s64 + vminq_s64/vmaxq_s64 + vmovn_s64.
 *
 * Stack-local int64 accumulator (head_dim ≤ 256 in BitNet → ≤ 2 KB).
 *
 * Bound analysis: per product fits int64 (≤ 2^60); summed over seq_k
 * activations may approach int64 limits at very long contexts —
 * matches the prior scalar implementation's behavior. Caller responsible.
 *
 * Preconditions:
 *   seq_k >= 0, head_dim >= 0
 *   shift >= 0 and shift <= 62
 *   y, w, V_base non-NULL when seq_k > 0 and head_dim > 0
 *   v_stride >= head_dim (rows do not overlap inside the [head_dim] slice)
 */
void m4t_mtfp_attn_v_combine(
    m4t_mtfp_t* y, int shift,
    const m4t_mtfp_t* w,
    const m4t_mtfp_t* V_base, size_t v_stride,
    int seq_k, int head_dim);

/* Scalar-only reference. Test oracle. Production must not call. */
void m4t_mtfp_attn_v_combine_scalar_ref(
    m4t_mtfp_t* y, int shift,
    const m4t_mtfp_t* w,
    const m4t_mtfp_t* V_base, size_t v_stride,
    int seq_k, int head_dim);

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

/* ── RoPE (rotary position embedding) ───────────────────────────────────
 *
 * Llama-family rotate_half convention. For each head and freq_idx
 * i ∈ [0, head_dim/2):
 *   q'[h, i]            = q[h, i]            · cos_i − q[h, i+d/2] · sin_i
 *   q'[h, i + d/2]      = q[h, i + d/2]      · cos_i + q[h, i]     · sin_i
 *
 * cos/sin are looked up from a precomputed LUT indexed by (position,
 * freq_idx). The LUT is built lazily on first call using libm cos/sin
 * (init-time FP allowed; same precedent as bf16→MTFP19 weight loading).
 *
 * Constraints:
 *   - position < M4T_ROPE_MAX_POSITION (4096; BitNet's
 *     max_position_embeddings).
 *   - head_dim ≤ M4T_ROPE_MAX_HEAD_DIM (256; comfortably above
 *     BitNet's 128) and even.
 *   - Single-threaded init (Phase 1 inference is single-threaded).
 *
 * Convention assumption (RC-10 of work-unit 3 red-team): BitNet
 * b1.58-2B-4T uses Llama's rotate_half convention. The HF model card
 * has no custom modeling_*.py — `trust_remote_code=True` loads via
 * transformers.LlamaForCausalLM-derived class. Final verification
 * happens at work-unit 6 (HF-vs-substrate per-layer comparison) — if
 * the convention is wrong, Q/K post-RoPE outputs would diverge.
 *
 * Saturating clamp on output. RoPE is a rotation — preserves L2 norm —
 * so saturation is rare in practice for valid MTFP19 inputs.
 *
 * Per journal/rope_design_lmm.md (work-unit 3 of bitnet_phase1). */
#define M4T_ROPE_MAX_POSITION   4096
#define M4T_ROPE_MAX_HEAD_DIM   256
#define M4T_ROPE_COS_SIN_SCALE  ((int32_t)1 << 29)  /* Q = 2^29 */

void m4t_mtfp_rope_apply(
    m4t_mtfp_t* q,
    m4t_mtfp_t* k,
    int position,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    double theta_base
);

/* Scalar test oracle. Same LUT, same apply pipeline; bit-exact
 * vs production. */
void m4t_mtfp_rope_apply_scalar_ref(
    m4t_mtfp_t* q,
    m4t_mtfp_t* k,
    int position,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    double theta_base
);

/* ── Integer reciprocal (Newton-Raphson) ────────────────────────────────
 *
 * dst = round(2^30 / src) for src ∈ [1, INT32_MAX].
 *
 * Output range: [1, 2^30] (clamped). For src = 1 → 2^30, for src = 2^30
 * → 1.
 *
 * Pure-int Newton-Raphson:
 *   y_{n+1} = y_n · (2·Q − src · y_n) / Q,  Q = 2^30
 * 5 iterations from a good initial guess give full int32 precision.
 *
 * Used by softmax for the 1/Σ exp step. */
m4t_mtfp_t m4t_int32_recip(m4t_mtfp_t src);
m4t_mtfp_t m4t_int32_recip_scalar_ref(m4t_mtfp_t src);

/* ── Softmax ───────────────────────────────────────────────────────────
 *
 * y[i] = exp(x[i] - max(x)) / Σ exp(x[j] - max(x))
 *
 * Input contract: x[i] is int32 representing natural-log units —
 * 1 LSB = 1 nat. Caller pre-rescales their score to this form.
 *
 * Output: y[i] ∈ [0, 2^30] with Σ y[i] ≈ 2^30 (probabilities at
 * scale 2^30).
 *
 * Implementation: exp LUT (init-time libm) covers z ∈ [-30, 0]; values
 * below underflow to 0. Reciprocal of sum via m4t_int32_recip with
 * pre-shift to fit int31. Per-cell scalar loop.
 *
 * n must be ≥ 1. */
#define M4T_SOFTMAX_LUT_RANGE  30
#define M4T_SOFTMAX_LUT_RES    4096
#define M4T_SOFTMAX_OUT_SCALE  ((int32_t)1 << 30)

void m4t_mtfp_softmax(m4t_mtfp_t* y, const m4t_mtfp_t* x, int n);

/* Independent FP test oracle. Runtime libm exp; same algorithm
 * shape as production. Tolerance comparison. */
void m4t_mtfp_softmax_scalar_ref(m4t_mtfp_t* y, const m4t_mtfp_t* x, int n);

/* ── A8 quantize / dequantize ──────────────────────────────────────────
 *
 * Per-tensor absmax + int8 quantization, matching BitNet's W1.58A8 spec.
 *
 *   y_int8[i] = round(x[i] · 127 / absmax),  clamped to [-127, 127]
 *   absmax    = max_i |x[i]|
 *
 * Quantize returns absmax (caller stores; needed for dequant or
 * vec_scale). For all-zero input, returns 0 and zeros y.
 *
 * Round-half-away-from-zero (matches torch.round / HF reference).
 * Per-cell scalar loop (no NEON int-divide intrinsic; documented per
 * the cross-exp accum's degenerate-case precedent).
 *
 * Per journal/a8_vec_scale_design_lmm.md (work-unit 5 of bitnet_phase1). */
m4t_mtfp_t m4t_a8_quantize(int8_t* y, const m4t_mtfp_t* x, int n);
m4t_mtfp_t m4t_a8_quantize_scalar_ref(int8_t* y, const m4t_mtfp_t* x, int n);

void m4t_a8_dequantize(
    m4t_mtfp_t* y, const int8_t* x, m4t_mtfp_t absmax, int n);
void m4t_a8_dequantize_scalar_ref(
    m4t_mtfp_t* y, const int8_t* x, m4t_mtfp_t absmax, int n);

/* ── Vector scale by num/den ratio ─────────────────────────────────────
 *
 * y[i] = round(x[i] · num / den), saturating to ±M4T_MTFP_MAX_VAL.
 *
 * For BitNet's BitLinear scale apply:
 *   num = α_mantissa × activation_absmax × 3^α_block_exp
 *   den = 127
 * Caller composes num/den; the substrate doesn't reach into α's
 * block_exp encoding.
 *
 * num and den are int64. Per-cell uses __int128 for x · num product
 * (can reach 2^92 for max-magnitude inputs). Per-cell scalar.
 *
 * den must be > 0. Round-half-away-from-zero. */
void m4t_mtfp_vec_scale(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    int64_t num, int64_t den, int n);
void m4t_mtfp_vec_scale_scalar_ref(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    int64_t num, int64_t den, int n);

/* ── bx-aware variants (Phase 2 work-unit 1) ────────────────────────────
 *
 * These variants accept explicit per-tensor block_exp parameters and
 * produce output at a caller-chosen target bx. They exist alongside the
 * implicit-bx variants above so existing tests keep working.
 *
 * Convention: real_value = mantissa × 3^(-block_exp). All bxes are
 * non-negative ints in [0, 35] (above 35 the 3^bx multiplier overflows
 * int64 in some intermediate computations).
 */

/* Rescale a vector between two block_exps:
 *   y_real = x_real, but represented at to_bx instead of from_bx
 *   y_m = x_m × 3^(from_bx - to_bx)   (if from > to, magnify mantissa;
 *                                     if from < to, divide)
 * Saturating clamp on overflow to ±M4T_MTFP_MAX_VAL. */
void m4t_mtfp_rescale_bx(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    int from_bx, int to_bx, int n);

/* RMSNorm with explicit bx for input, γ, and target output:
 *   y_real[i] = γ_real[i] · x_real[i] · rsqrt(mean(x_real²) + ε_real)
 * Output mantissas are at target_bx (not γ_bx as in the implicit variant).
 *
 * eps_mantissa is interpreted at the SAME scale as the SoS-shifted mean
 * (same contract as the existing m4t_mtfp_rmsnorm) — caller picks a
 * small positive value. */
void m4t_mtfp_rmsnorm_bx(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    const m4t_mtfp_t* gamma,
    int x_bx, int gamma_bx, int target_bx,
    m4t_mtfp_t eps_mantissa, int n);

/* Scalar test oracle for m4t_mtfp_rmsnorm_bx (V14.C). Production must not call. */
void m4t_mtfp_rmsnorm_bx_scalar_ref(
    m4t_mtfp_t* y, const m4t_mtfp_t* x,
    const m4t_mtfp_t* gamma,
    int x_bx, int gamma_bx, int target_bx,
    m4t_mtfp_t eps_mantissa, int n);

/* relu² with explicit input bx → target bx rescale. Squaring doubles
 * the bx; the rescale brings it back. y_real[i] = max(0, x_real[i])²
 * preserved; mantissas land at target_bx.
 *
 * V14.D: NEON-only production. ReLU + squared (vmull_s32) + iterated
 * /3 (mulhi by 0xAAAA...AAAB) + clamp + saturating narrow. */
void m4t_mtfp_relu2_inplace_bx(
    m4t_mtfp_t* x, int x_bx, int target_bx, int n);

/* Scalar test oracle for m4t_mtfp_relu2_inplace_bx. Production must not call. */
void m4t_mtfp_relu2_inplace_bx_scalar_ref(
    m4t_mtfp_t* x, int x_bx, int target_bx, int n);

/* Elementwise multiply with bx tracking. y_m_target = a_m × b_m / 3^(a_bx+b_bx-target_bx).
 *
 * V14.E: NEON-only production. vmull_s32 + sign-aware divide
 * (vabsq_s64 / vbslq_s64 re-sign around the iterated /3). */
void m4t_mtfp_elementwise_mul_bx(
    m4t_mtfp_t* y,
    const m4t_mtfp_t* a, int a_bx,
    const m4t_mtfp_t* b, int b_bx,
    int target_bx, int n);

/* Scalar test oracle for m4t_mtfp_elementwise_mul_bx. Production must not call. */
void m4t_mtfp_elementwise_mul_bx_scalar_ref(
    m4t_mtfp_t* y,
    const m4t_mtfp_t* a, int a_bx,
    const m4t_mtfp_t* b, int b_bx,
    int target_bx, int n);

/* BitLinear scale apply with explicit input bx and target output bx:
 *   y_real[i] = y_raw[i] · α_real · absmax_real / 127
 * Output at target_bx, computed as:
 *   y_m_target = y_raw[i] × α_m × absmax_m / (127 × 3^(α_bx + x_bx - target_bx))
 *
 * Constraint: α_bx + x_bx - target_bx ≤ 35 to keep den in int64.
 * For BitNet (α_bx ≤ 18, x_bx ≤ 21, target_bx = 14): max shift = 25.
 *
 * V14.F: NEON-only production. Per-cell |y_raw| × |num| is uint96
 * (3 × uint32 limbs); long-divided by 127 (magic mul with extension)
 * then iterated /3 (same div3 magic as V14.D/E). Sign re-applied via
 * vbslq_s64. Clamp to MTFP19. */
void m4t_mtfp_bitlinear_scale_bx(
    m4t_mtfp_t* y, const m4t_mtfp_t* y_raw,
    const m4t_mtfp_t* alpha_ptr, int alpha_bx,
    m4t_mtfp_t absmax_m, int x_bx, int target_bx,
    int n);

/* Scalar test oracle for m4t_mtfp_bitlinear_scale_bx. Production must not call. */
void m4t_mtfp_bitlinear_scale_bx_scalar_ref(
    m4t_mtfp_t* y, const m4t_mtfp_t* y_raw,
    const m4t_mtfp_t* alpha_ptr, int alpha_bx,
    m4t_mtfp_t absmax_m, int x_bx, int target_bx,
    int n);

/* Bit-faithful BitLinear scale apply (no a8 quantization).
 *
 *   y_m_target = y_raw_i64 × α_m / 3^(α_bx + x_bx - target_bx)
 *
 * y_raw_i64 comes from m4t_mtfp_ternary_matmul_bt_route_i64 (int64,
 * unclamped int32 × ternary sum). |y_raw_i64| ≤ K × MAX_VAL ≈ 2^40
 * for K ≤ 6912. Multiplied by α (int32 mantissa), the int128 product
 * is decomposed as uint96 (3 × uint32 limbs) like V14.F. Sign is
 * sign(y_raw_i64) XOR sign(α_m). Long-divided by 3^shift_exp using
 * the combined-divisor magic; for shift_exp ≤ 19 single pass.
 *
 * Constraint: α_bx + x_bx - target_bx ≤ 35. */
void m4t_mtfp_bitlinear_scale_no_a8_bx(
    m4t_mtfp_t* y, const int64_t* y_raw_i64,
    const m4t_mtfp_t* alpha_ptr, int alpha_bx,
    int x_bx, int target_bx,
    int n);

/* Scalar test oracle. Production must not call. */
void m4t_mtfp_bitlinear_scale_no_a8_bx_scalar_ref(
    m4t_mtfp_t* y, const int64_t* y_raw_i64,
    const m4t_mtfp_t* alpha_ptr, int alpha_bx,
    int x_bx, int target_bx,
    int n);

/* ── ReLU² ──────────────────────────────────────────────────────────────
 *
 * In-place: x[i] = (max(0, x[i]))². Used by BitNet's FFN gated path.
 * Saturating clamp (squaring large MTFP19 mantissas exceeds MTFP19_MAX
 * by a factor of |x|; the clamp pins the result at MAX_VAL). This loses
 * dynamic range, but the downstream RMSNorm normalizes magnitude away.
 *
 * Per-cell scalar — 64-bit multiply doesn't naturally vectorize beyond
 * the existing block ops; documented per the cross-exp accum precedent. */
void m4t_mtfp_relu2_inplace(m4t_mtfp_t* x, int n);
void m4t_mtfp_relu2_inplace_scalar_ref(m4t_mtfp_t* x, int n);

/* ── Element-wise multiply ─────────────────────────────────────────────
 *
 * y[i] = a[i] · b[i] with saturating clamp. Used by BitNet's FFN gated
 * path: gate_act × up. Same precision concern as relu² (squared-magnitude
 * range exceeds MTFP19); same mitigation (followed by RMSNorm). */
void m4t_mtfp_elementwise_mul(
    m4t_mtfp_t* y, const m4t_mtfp_t* a, const m4t_mtfp_t* b, int n);
void m4t_mtfp_elementwise_mul_scalar_ref(
    m4t_mtfp_t* y, const m4t_mtfp_t* a, const m4t_mtfp_t* b, int n);

#ifdef __cplusplus
}
#endif

#endif /* M4T_MTFP_H */
