---
cycle: xexpo_design
phase: CLOSEOUT
date: 2026-05-01
scope: external review of the synthesize phase; refinements folded in; superseded action plan
companions: xexpo_design_{raw,nodes,reflect,synthesize}.md · docs/DESIGN_X-EXPO.md · docs/REMEDIATION_PLAN.md
status: SUPERSEDES the action plan in xexpo_design_synthesize.md (the RAW/NODES/REFLECT/SYNTHESIZE files are preserved as-is — the trail is the value)
---

# Closeout — xexpo_design

## What happened

External design review on the synthesize-phase output. Two refinements landed that the cycle had partially captured but not sharpened, and one architectural insight that the cycle missed entirely. This closeout supersedes the prior synthesis's action plan with the corrected version.

## What survives the review

- **Path A (max-exponent alignment) as the only defensible choice.** Mathematically forced.
- **The bucket-decomposition methodology.** The boundary between "mathematical necessity" (Bucket A) and "consumer-shape contingency" (Bucket B) is where the original design's mistakes hid — and the LMM partition surfaced them. The reviewer notes this is "diagnostic of a healthy process" — the methodology found mistakes in the predicted place.
- **The trail value of RAW.** The reviewer specifically called out that the unfiltered RAW phase made Finding 2 visible. Cleaned-up prose hides the assumption ("I assumed callers would..."); raw notes surface it. Discipline confirmed.

## What sharpens

### Refinement 1 — Rounding rule and bound are a package

The cycle corrected `≤ 3^(e_d − 1)` to `≤ 3^e_d`. The reviewer points out this is incomplete — the bound depends on the rounding rule:

- **Truncate-toward-zero** (C integer division): bound is `< 3^e_d` (strict). Loss is at most `(3^Δ − 1)/3^Δ` mantissa units at e_d, which is strictly less than 1 mantissa unit, which decodes to strictly less than `3^e_d`.
- **Round-to-nearest**: bound is `≤ ½ · 3^e_d`. Loss is at most half a mantissa unit at e_d.
- **Round-half-to-even** (banker's rounding): same `≤ ½ · 3^e_d` bound, slightly different distributional properties.

**Decision: lock to truncate-toward-zero.** Matches the C semantics already used in libm4t (`m4t_mtfp_clamp64`, `signature_update`'s `means[d] /= T`). Substrate consistency. The contract states both the rule and the bound together as one sentence.

The corrected saturation contract:

```
For Path A alignment (e_d = max(e_a, e_b)) with truncate-toward-zero
division of the smaller operand's mantissa by 3^Δ:

If !saturated:
    require |decode(dst[i], e_d)
             − (decode(a[i], e_a) + decode(b[i], e_b))|
             < 3^e_d                  /* strict */
    /* The error comes entirely from C integer truncation. Loss is at
     * most (3^Δ − 1) / 3^Δ mantissa units at e_d, < 1 mantissa unit,
     * which decodes to < 3^e_d. */

If saturated:
    /* unchanged from prior synthesis */
```

### Refinement 2 — Accumulator is a *different* primitive, not a different signature

The cycle's synthesize phase noted "accumulator API becomes primary if cycle reveals it" but treated it as an API substitution. The reviewer's deeper read: an accumulator primitive has *different invariants*, not just a different shape:

> The running accumulator carries both a mantissa block *and* its current exponent, and the exponent can change across calls (it should grow when contributions exceed the current accumulator's representable range). That's a stateful primitive in a way pairwise isn't, and the contract has to specify when/how the accumulator's exponent can shift. This is actually the geometric `renormalize` operation showing up where it naturally belongs — at the accumulator boundary, not as a separate post-processing step.

This is correct. The accumulator's invariant `|running_mantissa[i]| ≤ MAX_VAL at running_exp` is maintained *across* the call sequence by the primitive itself. The cases:

- `e_new > e_running`: rescale running by `3^(e_new − e_running)` (truncate), set `running_exp = e_new`, add new contribution. Running's precision shrinks; this is the cost of growing dynamic range.
- `e_new < e_running`: rescale new by `3^(e_running − e_new)` (truncate), add into running, exponent unchanged. New contribution loses precision; this is Path A's "smaller operand vanishes" property at the per-call level.
- `e_new == e_running`: same-block-exp add (degenerates to `vec_add_inplace`).
- Pre-add saturation: if the to-be-rescaled side would saturate *before* the add, flag is set. Path A's structural-truncation behavior holds.

The contract grows from "produce dst" to "maintain invariant across the call sequence." Different testing surface entirely.

### Architectural insight the cycle missed

`apply_signed` (already shipped at tier 2) is *already an accumulator*, restricted to the trivial `e_new == e_running` case. Its loop body calls `vec_add_inplace` / `vec_sub_inplace` `k` times into a single `result` buffer. That's the accumulator pattern with one shared exponent.

**The cross-exponent accumulator is what `apply_signed` becomes when block_exp drift is allowed.** They're the same primitive shape; tier 2 shipped the constrained version. Tier 3's contribution is dropping the constraint.

This reframes the kernel's role from "new primitive" to "generalization of a primitive that already works." The cycle's hypothesis is no longer "do consumers benefit from cross-exp arithmetic?" but rather "does removing the same-block-exp constraint on apply_signed-shaped accumulation pay measurable returns?" Same question, sharper framing.

## Updated action plan (supersedes synthesize phase)

### Action 1 — Patch `docs/DESIGN_X-EXPO.md` (next session, ~45 min)

Targeted edits, expanded from the original five:

1. **Saturation-contract section.** Replace `≤ 3^(e_d − 1)` with `< 3^e_d` (strict), state truncate-toward-zero as the rounding rule, derive the bound from the loss model.
2. **Path A justification.** Replace "matches IEEE-754" with "preserves dominant magnitude — alignment-to-larger is the only positional-arithmetic choice that does not catastrophically saturate the larger operand." Move IEEE-754 to a parenthetical.
3. **Δ ≥ 19 precondition.** Soften assertion to documented degenerate behavior.
4. **API section — primary kernel becomes accumulator.** Specify the four cases (e_new >, <, ==, pre-add saturation). Specify the maintained invariant. Pairwise becomes a thin wrapper for the `n=1`-add special case.
5. **Property tests — refit to accumulator.** Tests are now sequence-shaped (multiple calls per property), not single-shot. The four properties become:
   - `prop_accum_aligning_correctness`: across N=1..64 random accumulations per sample, decoded result is within `< 3^e_running` of the real-number sum.
   - `prop_accum_aligning_invariant`: at every step in the sequence, `|running[i]| ≤ MAX_VAL` at `running_exp`.
   - `prop_accum_aligning_aliasing`: `running` aliasing `new` produces same result as non-aliased.
   - `prop_accum_aligning_sat_flags`: flag set iff saturation occurred in *any* step of the sequence reaching that cell.
6. **Add architectural-insight section.** State that `apply_signed` is the same-block-exp special case of this primitive.

### Action 2 — Extend `docs/REMEDIATION_PLAN.md` cycle protocol (next session, ~20 min)

The cycle's call-pattern measurement gains a precise evidence criterion:

> **Evidence sources for pairwise vs accumulator decision:**
> 1. **Static analysis of archived consumers.** Read `01MAY26_archived/tools/mnist_routed_bucket_multi.c` and trace every site that combines two or more MTFP19 buffers. Categorize each: pairwise (one-shot, two distinct buffers, no temporal dependency) or accumulator (running buffer reused across iterations, exponent could legitimately drift across calls).
> 2. **API-shape sketch.** For each identified site, write the call expression under both APIs (pairwise and accumulator). The more natural reading wins; the criterion is "which API requires fewer working-buffer manipulations to express the consumer's natural computation."
> 3. **Verdict:** if accumulator is the more natural reading at >50% of sites (or at *any* hot-path site), accumulator is the primary API. Pairwise becomes the n=1-add convenience wrapper.

The protocol commits to BOTH evidence sources; one alone is insufficient.

### Action 3 — Generalize the spec-as-constraint principle (next session, ~10 min)

`CONTRIBUTING.md` gains a principle (post the existing six invariants):

> **7. Substrate-level specs are upstream of kernel designs.** A kernel design that does not trace back to constraints in `m4t/docs/M4T_SUBSTRATE.md` is suspect. Re-read the relevant spec section before any design memo phase. Spec amendments require a journal cycle; kernel designs that contradict the spec without amending it are a discipline violation.

This is the §14.2 re-read lifted to a general rule. Applies to every kernel design, not just this one.

### Action 4 — Tier 3a + 3b proceed with the sharper hypothesis (next session, then ongoing)

No timeline change. The cycle now tests:
- Whether consumers' call patterns favor pairwise or accumulator (Action 2's criterion).
- Whether `apply_signed`-shaped accumulation pays measurable returns from cross-exp generalization (the architectural-insight reframing).
- The original three measurements (heterogeneity probability, Δ distribution, saturation rate) still apply.

## Success criteria for this closeout

- [ ] `docs/DESIGN_X-EXPO.md` is patched with all six items in Action 1.
- [ ] `docs/REMEDIATION_PLAN.md` cycle protocol gains the explicit evidence-source criterion.
- [ ] `CONTRIBUTING.md` gains the spec-as-constraint principle.
- [ ] `CHANGELOG.md` records the cycle landing and the closeout's key findings.
- [ ] No code changes. The closeout is documentation work only; tier 3a (consumer rebuild) is the next code-touching step.

## What this closeout does not change

- The cycle's findings remain valid. Both Finding 1 (error bound) and Finding 2 (pairwise vs accumulator) are correct as identified; the closeout sharpens them, doesn't reverse them.
- The discipline framing remains valid. Designing ahead of measurement is exploration, not violation, *if* the cycle tests the design's hypotheses rather than confirming them. The reviewer's principle (substrate spec upstream of kernel design) tightens this without contradicting it.
- The MTFP-capable-vs-genuinely-MTFP framing remains valid. The substrate is still fixed-point-in-practice until a consumer drives the cross-exp kernel and the cycle measures real benefit.

## Loop-back triggers from here

- **Back to NODES** if tier 3a's static analysis of `mnist_routed_bucket_multi.c` surfaces a third call pattern (neither pairwise nor accumulator). Worth flagging because it would mean the design space has a missed dimension.
- **Back to REFLECT** if §14.2's actual prose (still un-re-read) constrains the design in ways neither the cycle nor the reviewer anticipated.
- **No loop-back** if tier 3b confirms accumulator at >50% of sites and the saturation/heterogeneity measurements come in within their pre-committed bands. That's the wood cutting itself.

## A note on methodology

The reviewer's confirmation that the bucket-decomposition found mistakes "in the predicted place" is worth recording as a methodology data point. The Laundry Method's "the delta is where mistakes hide — check the boundaries" is not just a heuristic; in this cycle it was load-bearing. The pairwise-vs-accumulator question sat exactly on the bucket-A/bucket-B boundary, and that's where the assumption hid. One data point doesn't establish the methodology, but it's consistent with the LMM doc's claim about boundary-checking.

The reviewer also noted: "The 'raw notes on what I was actually thinking' phase is underrated. Most design retrospectives skip directly to the cleaned-up reflection, which loses the unfiltered self-knowledge that surfaces actual confusions and assumptions."

This is the second cycle in this project (counting prior-cycle archives) where RAW phase content surfaced findings that REFLECT alone would have missed. Worth promoting to memory: the RAW phase's discipline is doing real work; do not treat it as scaffolding.
