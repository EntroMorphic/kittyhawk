---
date: 2026-04-14
phase: REFLECT
topic: What is the right criterion for "base-3 native," and what replaces sign_extract?
---

# Reflect

## Core insight

**C-final is a consumer-relative criterion, not a primitive-absolute one.** This reframing changes the answer to the replacement question.

I was trying to label each primitive in the substrate as "base-3 native" or "base-2 shaped" in isolation. The sign_extract audit worked at first because that primitive had a *specific* failure on MTFP projections. But when I looked at signature_update (N9 in nodes), I realized sign_extract *inside* signature_update — where the zero state is produced by integer col_sum-minus-mean and can genuinely occur — doesn't fail C-final at all.

So the question isn't "should sign_extract exist?" It's "does this consumer's input distribution at this primitive realize all three states?" And that's not a property of the primitive — it's a property of the **primitive + consumer pair**.

This matches the substrate/consumer throughline cleanly:
- Substrate: provides primitives that CAN use three states.
- Consumer: asserts preconditions that its inputs WILL realize all three.

sign_extract doesn't have to be deleted. It has to be renamed and its contract has to be explicit: "this primitive is three-state-capable iff the caller's input distribution produces zero with non-trivial probability. For inputs where zero is measure-zero, prefer `threshold_extract`." That's a documentation fix plus a naming fix plus a new primitive — not a deletion.

---

## Resolved tensions

### T1 (extraction required vs unneeded) → RESOLVED: both are valid; they're different architectures
Option 3 (signature = mantissa, no extraction) is a different CONSUMER pipeline, not a substitute for extraction primitives. Both can coexist in the substrate. A consumer that produces MTFP4 projections and packs them directly doesn't need an extractor. A consumer that produces MTFP19 projections and wants to route on a signature does. The substrate provides the building blocks for both; consumers pick the pipeline shape.

### T2 (per-primitive vs per-use) → RESOLVED: per-use, formally
C-final is a **consumer-level precondition**, not a substrate-level property. The substrate's spec-level claim is: "this primitive has three-way semantic capacity; its zero state is not measure-zero by construction." The consumer's claim is: "my input distribution realizes all three states, so C-final holds for my use."

### T3 (elegance vs migration cost) → DISSOLVED: it was a false choice
Once T1 is resolved (both can coexist), we don't have to pick Option 2 vs Option 3. We add `threshold_extract` (Option 2) for the extraction-based consumers. Option 3 (no extraction) is a consumer-level architectural choice that doesn't require substrate changes — any consumer can already produce MTFP4 projections and pack them via existing primitives.

### T4 (criterion rigor vs usability) → RESOLVED: C-final + consumer precondition
C-final at the substrate level is sharp: "does this primitive have three-way semantic capacity?" Sign_extract passes this (the operation DOES distinguish three states in its type). The consumer-level check is: "does my input distribution activate the zero state non-trivially?" Both are mechanical. Together they cover what C-final was trying to do in one step.

### T5 (sign_extract might not be wrong everywhere) → RESOLVED by the reframe
Sign_extract's zero state is structurally reachable (C-final C1/C2 pass). Whether it's reached in practice depends on input distribution (C3). So sign_extract stays in the substrate with a *named contract* about what kinds of inputs make it a well-typed routing primitive. The bug was using it with MTFP-projection inputs where C3 fails silently, not using sign_extract itself.

---

## Hidden assumptions challenged

1. **"A primitive is either base-3 native or it isn't."** False. Primitives have three-way *capacity*; whether that capacity is *realized* is a consumer-level property.
2. **"sign_extract is base-2-shaped."** More precise: sign_extract is three-state-capable but fails C3 on MTFP-projection inputs. It would pass C3 on integer-difference inputs (like inside signature_update). The shape depends on the input distribution.
3. **"We need to replace sign_extract."** Partial. We need to ADD threshold_extract (for extract-from-MTFP-projections use cases) AND formalize the substrate/consumer contract around C-final. We don't need to DELETE sign_extract.
4. **"Option 2 and Option 3 are alternatives."** False. Option 2 is a substrate addition; Option 3 is a consumer pipeline change. Different layers.
5. **"C-final is the right criterion."** Refined. C-final was trying to be an absolute property of a primitive. The correct form is: the substrate ensures C1/C2 (three-way capacity is real), the consumer asserts C3 (three-way capacity is realized).
6. **"Architectural correctness means replacing primitives."** Sometimes it means adding primitives + documenting contracts, not replacing.

---

## What I now understand

The architectural question "is sign_extract base-3 native?" was wrongly framed. The correct framing has two parts:

1. **Does the primitive have three-way semantic capacity at the substrate level?**
   Test: collapse any one state in the primitive's definition; does the semantic change? YES → three-way capacity exists.
   sign_extract: yes (if you remove the zero case, the semantic is ill-defined for input = 0).
   distance_batch: yes (removing zero-vs-sign asymmetry changes the metric).
   apply_signed: yes (removing sentinel changes decision semantics).
   vec_add: yes at the mantissa level (removing zero trits changes arithmetic).

2. **Does the consumer's input distribution realize all three states at this primitive?**
   Test: sample realistic inputs, observe that all three states occur with non-trivial frequency.
   sign_extract + MTFP projections: NO (zero is measure-zero).
   sign_extract + integer col_sum-minus-mean: YES.
   etc.

The substrate's responsibility stops at (1). The consumer's responsibility is (2). The bug in the current repo is that we used sign_extract with inputs from (1) that don't realize its three-way capacity — and we never wrote down that this was the consumer's job.

---

## What the cycle surfaced that I missed at first

- **The reframe from primitive-absolute to primitive-consumer-pair**. I was trying to audit primitives individually; the real audit is primitive-plus-input-distribution.
- **sign_extract is the τ=0 degenerate of threshold_extract**. This means the relationship between the primitives is mathematical, not rival. The substrate already had a generalized primitive (threshold_extract) in degenerate form; we just never exposed the parameter.
- **Options 2 and 3 are at different layers**. Option 2 is a substrate addition. Option 3 is a consumer pipeline choice. Conflating them confused the scope.
- **Deletion is probably wrong**. The instinct "get rid of sign_extract" was based on the false belief that it's structurally binary. It's not — it's a special case of a generalized primitive, and in its correct use case it's fine.

---

## What remains uncertain

1. **How strict should the substrate's own contract be about C3?** One answer: the substrate names which primitives have three-way capacity; consumers assert C3 when they call. Another: the substrate's primitive docstrings explicitly call out "zero state is measure-zero for input class X" where X is known. The latter is more defensive.
2. **Does signature_update need any change?** It calls sign_extract internally. If col_sum-minus-mean genuinely realizes all three states on typical weight matrices, the call is correct. If not, signature_update needs threshold_extract. This is a measurable consumer-level question.
3. **Does the routed MNIST consumer need threshold_extract or Option 3 pipeline?** Open. Either could close the 23-point gap. Option 3 (MTFP4 signatures) would preserve 4× more information per dim. Threshold_extract with tuned τ would add a sparsity feature per dim. Different experimental directions.

---

## Next steps implied by the cycle

Not: delete sign_extract.
Not: commit to Option 2 vs Option 3.

Instead:
1. Formalize C-final as a **two-part criterion**: substrate-side (three-way capacity) + consumer-side (realized in input distribution). Write it into `m4t/docs/M4T_SUBSTRATE.md` as a new section.
2. Add `m4t_route_threshold_extract` as a new substrate primitive. Don't delete sign_extract.
3. Rename or relabel `sign_extract` to make its contract explicit — e.g., amend its docstring to name the C3 precondition: "three-state output iff input values realize zero with non-trivial probability (typically integer-arithmetic inputs, not MTFP projection outputs)."
4. Consumers choose: sign_extract if their input naturally realizes zero; threshold_extract otherwise.
5. Document Option 3 (MTFP4-native signatures) as a separate pipeline pattern, not a substrate change.
