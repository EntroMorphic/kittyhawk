---
title: Cross-exponent MTFP add — design
date: 2026-05-01 (revised 2026-05-01 per xexpo_design_closeout.md)
status: DESIGN EXPLORATION (pre-cycle). Not a commitment to build. The discipline (no primitive without measured consumer demand) gates implementation behind tier 3a + 3b in `REMEDIATION_PLAN.md`. This document specifies what the kernel *would* be if a consumer qualifies — and what the consumer-discovery cycle needs to test against. Revised after external review surfaced (a) a too-tight error bound and (b) the accumulator-vs-pairwise API question; both folded in below.
companions: REMEDIATION_PLAN.md · m4t/docs/M4T_SUBSTRATE.md (§14.2) · NORTH_STAR.md · journal/xexpo_design_closeout.md
---

# `m4t_mtfp_vec_accum_aligning` — design

## Premise

The substrate's missing kernel: accumulate MTFP19 contributions into a running buffer when contributions may carry **different `block_exp` values** than the running buffer.

Without this, every consumer that combines MTFP tensors of different scale must:
- Force one shared `block_exp` upstream (current state — the substrate is fixed-point-with-conversions), or
- Bypass the substrate and do the rescale in scalar/binary-float code (discipline violation), or
- Avoid the combination entirely (architectural distortion).

This kernel is the one piece between "MTFP-capable" and "genuinely floating in base 3."

## Architectural insight — `apply_signed` is the degenerate case

The tier-2 primitive `m4t_route_apply_signed` is *already an accumulator*, restricted to the trivial `e_running == e_new` case:

```c
/* Excerpt from current m4t_route.c apply_signed. */
for (int sel = 0; sel < k; sel++) {
    if (sign == 1)       m4t_mtfp_vec_add_inplace(result, tile, dim);
    else if (sign == -1) m4t_mtfp_vec_sub_inplace(result, tile, dim);
}
```

The body accumulates `k` tile contributions into `result`. Today every contribution must share the running's `block_exp`. The cross-exponent kernel designed here is what `apply_signed` *becomes* when block_exp drift is allowed across contributions — the same primitive shape, with the same-block-exp constraint dropped.

This reframes the kernel's role from "new primitive" to "generalization of a primitive that already ships." The cycle's hypothesis is therefore: **does removing the same-block-exp constraint on apply_signed-shaped accumulation pay measurable returns?**

## Two paths, only one defensible

Given a running mantissa `running[]` at exponent `e_running` and a new contribution `new[]` at exponent `e_new`, the result lives at exponent `e_result`. Two natural choices:

### Path A — `e_result = max(e_running, e_new)` (align up)

The smaller-exponent operand is rescaled by **dividing** its mantissa by `3^Δ` where `Δ = e_result − e_smaller`. The result lives at the larger scale.

- Pre-rescale overflow: impossible. Division shrinks.
- Post-add overflow: possible if both operands are near `MAX_VAL` after alignment. Saturate; flag.
- Precision behavior: smaller operand loses low-order trits to integer truncation. When `Δ` is large, the smaller operand silently zeroes — preserving the dominant magnitude.
- Dynamic range: large. Result can represent any value in MTFP19 at `e_result`.

### Path B — `e_result = min(e_running, e_new)` (align down)

The larger-exponent operand is rescaled by **multiplying** its mantissa by `3^Δ`.

- Pre-rescale overflow: very likely. `3^Δ × |larger_operand|` exceeds `MAX_VAL` for any operand near full scale at `Δ ≥ 4`. The larger operand saturates *before* the add — catastrophic, because the larger operand carries the dominant magnitude.
- Post-add overflow: possible.
- Precision behavior: preserves smaller operand's precision when result fits — but at the cost of the larger operand frequently saturating to noise.
- Dynamic range: collapsed.

### Decision: **Path A**

Path A is forced by geometry, not idiom: alignment-to-larger is the only positional-arithmetic choice that does not catastrophically saturate the larger operand. The larger operand carries the dominant magnitude; corrupting it is unrecoverable. Path B is a non-starter for any base — the substrate's choice would be the same in base-2, base-3, or any positional system.

(Historical anchor: this is the same choice IEEE-754 makes for binary float alignment. The shared answer comes from shared geometry, not from importing a base-2 idiom.)

The named cost — smaller operand vanishes when scale gap is large — is inherent. Consumers wanting to preserve both magnitudes simultaneously must request a Case-W output type (MTFP39) in a future kernel; that is *not* this kernel.

## Algorithm — accumulator-shaped primary

The accumulator's invariant: at any moment between calls, `|running[i]| ≤ MAX_VAL at running_exp` for every cell. The kernel maintains this across calls by potentially growing `running_exp` upward when contributions push the result out of range at the current exponent.

Four case branches per call:

```
Case e_new == e_running:
    /* Same-block-exp add. Degenerates to vec_add_inplace semantics. */
    for i: running[i] = clamp(running[i] + new[i])

Case e_new > e_running:
    /* Grow running_exp upward to e_new. Running mantissa rescales down. */
    Δ = e_new - e_running
    s = pow3(Δ)
    for i: running[i] = clamp((running[i] / s) + new[i])
    running_exp = e_new

Case e_new < e_running:
    /* New contribution rescales down to running_exp. running_exp unchanged. */
    Δ = e_running - e_new
    s = pow3(Δ)
    for i: running[i] = clamp(running[i] + (new[i] / s))

Pre-add saturation (any case):
    /* If after rescale the to-be-added side already overflows, the add
     * cannot recover. Flag and saturate at the rescaled-but-unadded value. */
```

Exactly one rescale happens per call (or zero in the same-block case). The integer division uses C truncate-toward-zero — substrate-consistent with `m4t_mtfp_clamp64` and `signature_update`'s `means /= T`.

The pairwise primitive `vec_add_aligning(dst, a, e_a, b, e_b)` is implemented as a thin wrapper that initializes `dst = a, e_running = e_a` and then calls accumulate once with `(b, e_b)`. It ships only as a convenience; the accumulator is the canonical primitive.

## Saturation semantics (precise)

Two places saturation can happen in the accumulator: (1) post-add overflow at the current `running_exp`, (2) the pre-add edge where the rescaled smaller side already saturates to ±MAX_VAL even before the add (rare but possible at extreme Δ).

For the accumulator's per-call contract, with C truncate-toward-zero division of any rescaled operand:

```
if !saturated[i] for this call:
    require |decode(running_after[i], e_result)
             − (decode(running_before[i], e_running)
                + decode(new[i], e_new))|
             < 3^e_result                     /* strict */
    /* The error comes entirely from C integer truncation when dividing
     * the smaller-exponent side's mantissa by 3^Δ. Truncation toward
     * zero loses at most (3^Δ − 1)/3^Δ mantissa units at e_result, which
     * is strictly less than 1 mantissa unit, which decodes to strictly
     * less than 3^e_result in real numbers. */

if saturated[i] for this call:
    require sign(running_after[i]) ==
            sign(decode(running_before[i], e_running)
                 + decode(new[i], e_new))
    require running_after[i] ∈ {+MAX_VAL, −MAX_VAL}
    require sat_flags[i] == 1   /* if sat_flags non-NULL */
```

The bound is **strictly** less than `3^e_result`, paired with the truncate-toward-zero rule. Round-to-nearest would give `≤ ½ · 3^e_result`, but the substrate uses truncate everywhere else; consistency wins.

This is the **substrate's saturation contract for cross-exponent accumulation**. The substrate spec amendment (post-cycle, if the kernel ships) records this formally as Case S extended to accumulator semantics.

## API

```c
/* Accumulate a new contribution into a running MTFP19 buffer when the
 * two may carry different block exponents. The running buffer's exponent
 * may grow upward across calls; it never shrinks.
 *
 * Path A alignment with C truncate-toward-zero rescale. The smaller-
 * exponent side loses precision; the larger preserves dynamic range.
 *
 * Saturation: per-cell, post-add (and rare pre-add) per Case S.
 * sat_flags is per-cell (1 byte per cell, 0 or 1); pass NULL to skip.
 *
 * Aliasing: running and new must NOT alias each other (the kernel may
 * read new[i] after writing running[i]). running may equal sat_flags's
 * underlying buffer iff the consumer does not need the flags — but the
 * recommended usage is distinct buffers.
 *
 * Preconditions:
 *   n >= 0
 *   running, new non-NULL when n > 0
 *   running_exp non-NULL
 *   |running[i]|, |new[i]| <= M4T_MTFP_MAX_VAL  (MTFP19 substrate invariant)
 *
 * Documented degenerate behavior (NOT a precondition violation):
 *   |*running_exp − new_exp| > 19 — the smaller-exponent side truncates
 *   to zero by the math; the result is the larger side passed through
 *   (modulo saturation). The kernel does not error; this is well-defined,
 *   if uninformative.
 */
void m4t_mtfp_vec_accum_aligning(
    m4t_mtfp_t*       running,
    int8_t*           running_exp,    /* in-out */
    const m4t_mtfp_t* new,
    int8_t            new_exp,
    uint8_t*          sat_flags,      /* nullable, length n */
    int               n);

/* Convenience pairwise wrapper. Equivalent to:
 *   memcpy(dst, a, n * sizeof(m4t_mtfp_t));
 *   int8_t e = e_a;
 *   m4t_mtfp_vec_accum_aligning(dst, &e, b, e_b, sat_flags, n);
 *   *out_e = e;
 *
 * The accumulator is the canonical primitive; this wrapper exists for
 * call sites that genuinely have two distinct buffers and one shot. */
void m4t_mtfp_vec_add_aligning(
    m4t_mtfp_t*       dst,
    int8_t*           out_e,           /* result exponent, written */
    const m4t_mtfp_t* a, int8_t        e_a,
    const m4t_mtfp_t* b, int8_t        e_b,
    uint8_t*          sat_flags,
    int               n);
```

### Why the running exponent is in-out

Unlike the pairwise design's deterministic `e_d = max(e_a, e_b)`, the accumulator's exponent is *stateful*: it grows across calls and the consumer needs to know its current value to decode the running buffer. The in-out parameter is unavoidable and load-bearing — there is no derivation rule the consumer can apply locally.

This is the *renormalize* operation embedded in the primitive at the natural boundary: when a contribution's scale exceeds the running's representable range, the running rescales itself. Consumers don't call a separate `renormalize()`; the accumulator does it.

### Why `int8_t` for exponent

`int8_t` covers exponents in `[−128, 127]`. MTFP19's full mantissa range corresponds to `3^19 ≈ 1.16e9`; the practical exponent range fits comfortably. Larger types add storage cost without representable benefit.

## Storage granularity (per-tensor, not per-block)

Two options were considered:

1. **Per-tensor** (one `int8_t` exponent for the whole `running[]` or `new[]` array — what the API specifies).
2. **Per-block** (one `int8_t` per 4-cell MTFP19 block, stored in a sidecar array).

Per-block is the substrate spec's stated intent (§7). But the consumers identified for the discovery cycle all emit *one tensor per logical scale*:

- Multi-table SUM resolver: one distance vector per table; all distances in one table share the projection scale.
- Multi-tile routed accumulation: one MTFP19 output per tile; all cells of one tile share the matmul scale.
- Routed autodiff gradients: one gradient tensor per parameter; all cells share the optimizer's per-parameter scale.

Within each logical tensor, scale is uniform. **Per-block sidecar exponents would carry zero information** for these consumers and add 25% storage overhead.

**Decision:** MVP is per-tensor. Per-block becomes a separate kernel only if the consumer-discovery cycle surfaces a tensor whose internal scale legitimately varies across blocks. The plan's open question is provisionally answered.

## `sat_flags` layout

For MVP: **one `uint8_t` per cell**. Values `0` or `1`. Total length `n` bytes.

For accumulator semantics: a cell's flag is set if saturation occurred during *any* call in the sequence reaching that cell. The flag is sticky — the consumer either clears it deliberately or accepts that one saturating call contaminates that cell's history.

Pros: simplest to read, simplest to test.

Cons: 4× the storage of a bit-packed alternative. Acceptable for tier 3 (this kernel runs in accumulation phases, not signature-distance loops).

If the cycle measures saturation rate <0.1%, layout migrates to a single `uint64_t` aggregate counter in a future cycle. Not now.

## Aliasing

`running` and `new` **must not alias** in the accumulator API. The kernel may read `new[i]` after writing `running[i]` (depending on which operand needs rescale), so aliasing is unsafe.

The pairwise wrapper enforces aliasing safety internally via the `dst = a` initialization step. If the consumer needs `dst == a`, the wrapper handles it; if `dst == b`, it does not (a separate wrapper variant could, but is not built until requested).

The property test exercises:
- Distinct `running` and `new` buffers (canonical case).
- `running == sat_flags` underlying buffer with `sat_flags` declared NULL (degenerate but permitted).

## Implementation sketch (scalar MVP)

```c
#include "m4t_mtfp.h"
#include "m4t_internal.h"

/* Powers of 3 up to 3^19. Indexed by Δ ∈ [0, 19].
 * Δ > 19 → degenerate behavior; smaller side truncates to zero. */
static const int32_t M4T_POW3_TABLE[20] = {
    1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683,
    59049, 177147, 531441, 1594323, 4782969, 14348907,
    43046721, 129140163, 387420489, 1162261467
};

void m4t_mtfp_vec_accum_aligning(
    m4t_mtfp_t* running, int8_t* running_exp,
    const m4t_mtfp_t* new, int8_t new_exp,
    uint8_t* sat_flags,
    int n)
{
    assert(n >= 0);
    assert(n == 0 || (running && new));
    assert(running_exp);

    int8_t e_run = *running_exp;

    if (new_exp == e_run) {
        /* Same-block-exp accumulation. */
        for (int i = 0; i < n; i++) {
            int64_t sum = (int64_t)running[i] + (int64_t)new[i];
            m4t_mtfp_t out = m4t_mtfp_clamp64(sum);
            if (sat_flags && sum != (int64_t)out) sat_flags[i] = 1;
            running[i] = out;
        }
        return;
    }

    if (new_exp > e_run) {
        /* Grow running_exp upward; rescale running's mantissas down. */
        int delta = (int)new_exp - (int)e_run;
        if (delta >= 20) {
            /* Degenerate: running side truncates to zero; result is `new`. */
            for (int i = 0; i < n; i++) running[i] = new[i];
            *running_exp = new_exp;
            return;
        }
        int32_t s = M4T_POW3_TABLE[delta];
        for (int i = 0; i < n; i++) {
            int64_t aa = (int64_t)running[i] / s;        /* truncate toward 0 */
            int64_t sum = aa + (int64_t)new[i];
            m4t_mtfp_t out = m4t_mtfp_clamp64(sum);
            if (sat_flags && sum != (int64_t)out) sat_flags[i] = 1;
            running[i] = out;
        }
        *running_exp = new_exp;
        return;
    }

    /* new_exp < e_run: rescale new down; running_exp unchanged. */
    int delta = (int)e_run - (int)new_exp;
    if (delta >= 20) {
        /* Degenerate: new side truncates to zero; running unchanged. */
        return;
    }
    int32_t s = M4T_POW3_TABLE[delta];
    for (int i = 0; i < n; i++) {
        int64_t bb = (int64_t)new[i] / s;                 /* truncate toward 0 */
        int64_t sum = (int64_t)running[i] + bb;
        m4t_mtfp_t out = m4t_mtfp_clamp64(sum);
        if (sat_flags && sum != (int64_t)out) sat_flags[i] = 1;
        running[i] = out;
    }
}

/* Pairwise wrapper. */
void m4t_mtfp_vec_add_aligning(
    m4t_mtfp_t* dst, int8_t* out_e,
    const m4t_mtfp_t* a, int8_t e_a,
    const m4t_mtfp_t* b, int8_t e_b,
    uint8_t* sat_flags,
    int n)
{
    if (dst != a) {
        for (int i = 0; i < n; i++) dst[i] = a[i];
    }
    int8_t e = e_a;
    m4t_mtfp_vec_accum_aligning(dst, &e, b, e_b, sat_flags, n);
    if (out_e) *out_e = e;
}
```

### NEON consideration (deferred)

ARM NEON has no integer-divide instruction. A vectorized version would either:

1. **Multiply by reciprocal.** Precompute `recip[Δ] ≈ 2^32 / 3^Δ`, then `aligned = (mantissa * recip) >> 32`. Standard libdivide pattern. Adds rounding subtlety the property test must verify against the scalar reference.
2. **Scalar inner loop.** Accept that the kernel runs in accumulation phases, not distance loops — no profile evidence yet that it's a hot path.

**MVP is scalar.** Vectorization is its own cycle, gated on profile evidence from a real consumer.

## Property tests (sequence-shaped)

The accumulator's contract is across-call invariant maintenance, not single-call output. Tests must therefore be sequence-shaped: drive the kernel with a sequence of `(new, e_new)` calls and verify properties hold throughout.

Built on the property-based harness specified in `REMEDIATION_PLAN.md` "Test infrastructure":

### `prop_accum_aligning_correctness`

- **Sample:** 10 000 random sequences. Each sequence has `n ∈ [1, 64]` cells and `K ∈ [1, 32]` accumulation calls. Per call: random `new[]` with `|new[i]| ≤ MAX_VAL`, random `e_new` with `|e_new − running_exp_at_call_time| ≤ 19`. Initial running buffer random within MAX_VAL.
- **Reference:** parallel "oracle" running sum maintained as `double[]` (sanctioned per `M4T_SUBSTRATE.md` §12 — test path, not runtime kernel) using `mtfp_decode_to_double`.
- **Check:** at each call's completion, for every non-saturated cell `i`,
  `|decode(running[i], running_exp) − oracle[i]| < 3^running_exp`. Strict bound. For saturated cells, the saturation contract holds.
- **Pass:** all checks pass at every call across 10 000 sequences.

### `prop_accum_aligning_invariant`

- **Sample:** 10 000 random sequences as above.
- **Check:** at every call's completion, `|running[i]| ≤ MAX_VAL` for all `i`. The accumulator's defining invariant must hold across the entire sequence — not just the first call.
- **Pass:** invariant holds at every call across 10 000 sequences.

### `prop_accum_aligning_aliasing`

- **Sample:** 10 000 random sequences.
- **Check:** running aliasing the underlying memory of `sat_flags` (with `sat_flags` declared NULL) produces results bit-identical to the non-aliased case. Distinct `running` and `new` is required (precondition); the test does not exercise the forbidden case.
- **Pass:** identical results across 10 000 sequences.

### `prop_accum_aligning_sat_flags`

- **Sample:** 10 000 sequences drawn from a *saturation-targeted* distribution — operands deliberately near MAX_VAL at compatible exponents, plus the boundary case `|running[i]| + |rescaled_new[i]| = MAX_VAL` (no saturation expected).
- **Check:** for each cell, `sat_flags[i] == 1` if and only if at least one call in the sequence saturated that cell. Sticky semantics; once set, remains set.
- **Pass:** false positives and false negatives both fail; 10 000 / 10 000.

### Pairwise wrapper tests (one property each)

- **`prop_add_aligning_correctness_via_wrapper`** — 10 000 random `(a, b, e_a, e_b)`; pairwise wrapper produces same `dst[]` and `out_e` as a scalar oracle would.
- **`prop_add_aligning_roundtrip`** — `add_aligning(dst, &e, x, e0, neg(x), e0)` produces `dst[i] == 0` for all `i`.

## Open questions for the consumer-discovery cycle

The design assumes consumers will benefit from cross-exponent accumulation. The cycle's instrumentation must verify this on real data. The questions are now sharper than the original design's three:

1. **Does any consumer's accumulation site see `e_new ≠ e_running`?** If multi-table SUM only ever combines distances at one shared exponent, the kernel reduces to the existing `vec_add_inplace` and earns nothing. **Cycle's per-call exponent log directly tests this.**
2. **What `Δ` distribution do real consumers produce?** `Δ ≤ 1` in 99% of calls means precision loss is small (at most one trit). `Δ ≥ 5` regularly means the smaller side genuinely vanishes — and the design's framing matters.
3. **Is the saturation rate non-trivial?** If <0.1%, `sat_flags` becomes dead infrastructure and may downgrade to a counter.
4. **NEW: Does the consumer's natural call pattern match the accumulator API, or is it pairwise?** This is the question the closeout review surfaced. Evidence sources (per `REMEDIATION_PLAN.md`):
   - Static analysis of archived `mnist_routed_bucket_multi.c`.
   - API-shape sketch under both designs at each identified call site.
   - Verdict by majority + hot-path-site weight.
5. **NEW: Does any consumer's `running_exp` legitimately need to decrease across calls?** The accumulator design specifies upward-only growth. If a consumer would benefit from downward migration (e.g., to recover precision after a temporary spike), that's a different primitive.

The cycle's RAW phase records these five measurements directly. The SYNTHESIZE phase decides which design hypotheses hold and which need amending before the kernel ships.

## What this design does not decide

- **Per-block exponent variant.** Deferred until a consumer asks. MVP is per-tensor.
- **Case-W variant.** A separate kernel `m4t_mtfp_vec_accum_aligning_widening` that lands the running buffer in MTFP39 and avoids saturation entirely. Only if the cycle surfaces a consumer that legitimately needs both precision and dynamic range simultaneously.
- **Downward exponent migration.** If a consumer needs it, that's a separate cycle and a separate primitive. The design here is upward-only.
- **NEON vectorization.** Scalar MVP; vectorize only on profile evidence.
- **MTFP4 or MTFP9 variants.** Tier 3 also-deferred surfaces.
- **Subtraction.** `vec_sub_aligning` mirrors the design with `new[i] = -new[i]` substituted. Trivial extension; not built until asked.

## Spec amendment (post-cycle, if kernel ships)

`m4t/docs/M4T_SUBSTRATE.md` §14.2 is updated to:

- Replace named-but-unbuilt status with the kernel's actual semantics (the saturation contract above).
- Note Path A as the chosen alignment strategy and Path B as explicitly rejected.
- Document the per-tensor exponent granularity and the truncate-toward-zero rounding rule.
- Document the accumulator-as-primary, pairwise-as-wrapper API choice (a departure from the original §14.2 sketch).
- Cross-reference both `journal/xexpo_design_*` and the consumer-discovery cycle's findings as the discipline gate.

If the cycle does not qualify a consumer, §14.2 is updated with the cycle's null result and this design (along with the LMM cycle) is preserved as a vetted artifact: a design that did not earn its implementation. Per discipline, that is a real outcome, not a failure.
