---
cycle: xexpo_design
phase: SYNTHESIZE
date: 2026-05-01
scope: actionable output from RAW + NODES + REFLECT
companions: docs/DESIGN_X-EXPO.md · docs/REMEDIATION_PLAN.md
---

# Synthesize — xexpo_design

## What this synthesis is

A revision plan for `docs/DESIGN_X-EXPO.md` and an extended specification for what tier 3b (the consumer-discovery cycle) must measure. The reflection surfaced one outright bug in the design and one load-bearing hypothesis the design quietly assumed. Both must be addressed before the design serves as the cycle's input.

## Key decisions

### D1 — Two design components survive the cycle regardless

These are mathematical, not consumer-dependent:

1. **Path A alignment** (max-exponent target). Stays. Path B is a non-starter.
2. **The "preserves dominant magnitude" framing**. Stays, replacing the IEEE-754 reference. Geometry, not idiom.

### D2 — One design component is wrong as written

**The error bound `3^(e_d − 1)` is too tight by a factor of 3.** Correct bound, derived from integer truncation in Path A's smaller-operand divide:

```
error_per_cell ≤ (3^Δ - 1) / 3^Δ  mantissa units at e_d
            = at most 1 mantissa unit at e_d
            = at most 3^e_d  in real-number terms
```

The property test as written would have rejected correct kernels. **Update the design's saturation contract section to use the correct bound** before any property-test infrastructure is built.

### D3 — One load-bearing decision is provisional, not made

The design specifies pairwise `vec_add_aligning(dst, a, e_a, b, e_b)`. The cited consumers naturally accumulate; the right primitive may be `vec_accum_aligning(running, &e_running, new, e_new)` instead. **Until the cycle measures the call pattern, the design is provisional on this axis.**

The cycle's instrumentation must record:
- Per consumer: pairwise calls vs running-accumulator calls. (Operationalized as: are operands distinct buffers each call, or is one operand the result-buffer-from-the-previous-call?)
- If accumulator: how does the running exponent migrate? Always upward to the new max, or sometimes downward when a new operand is much smaller? (Downward migration would be unusual and worth flagging.)

**Decision endpoint:** if the cycle reveals accumulator semantics, the design is *redesigned*, not just *implemented*. The pairwise kernel ships only as a special case of the accumulator, not as the primary API.

### D4 — Several decisions are consumer-shape hypotheses awaiting the cycle

Listed for the cycle's measurement protocol:

| Hypothesis | Cycle measurement | Disposition if false |
|---|---|---|
| Consumers produce heterogeneous block_exp | Per-call `(e_a, e_b)` log; report P(e_a ≠ e_b) | If <0.5%, kernel earns nothing; substrate stays MTFP-capable, fixed-point-in-practice |
| Δ distribution is meaningful | Histogram of \|e_a − e_b\| | If Δ ≤ 1 in 99% of calls, kernel is mechanical; consumer doesn't materially benefit |
| Saturation rate at post-add clamp | Counter incremented per saturating cell | If <0.1%, sat_flags becomes a single counter; per-cell layout is dropped |
| Kernel is NOT in a hot path | Profile callsite frequency | If >1% of consumer wall time, NEON design is its own cycle |
| Per-tensor exponent is sufficient | Per-tensor probe: do all blocks within one tensor share an effective scale? | If no, per-block aligning add is a separate kernel |

### D5 — Discipline-question resolutions

- **Δ ≥ 19 case:** soften to "well-defined but degenerate" — drop the hard assertion. Consumer gets a correct (if uninformative) result.
- **§14.2 must be re-read** before the tier 3c design memo phase. Add to the cycle's prep checklist.
- **The design as exploration:** legitimate, *but* the cycle must test the design's hypotheses, not just verify the kernel's correctness against them. This is the meaningful discipline check — does the design lead the cycle, or does the cycle lead the design?

## Action plan

### Action 1 — Patch the design (today, before any cycle work begins)

Targeted edits to `docs/DESIGN_X-EXPO.md`:

1. **Saturation-contract section.** Replace `≤ 3^(e_d − 1)` with `≤ 3^e_d` and rewrite the derivation to show the integer-truncation source. The looser bound is the *correct* one.
2. **Path A justification section.** Replace "matches IEEE-754" with "preserves dominant magnitude — the only positional-arithmetic choice that does not catastrophically saturate the larger operand." Move IEEE-754 to a parenthetical historical anchor.
3. **Δ ≥ 19 precondition section.** Replace assertion with documented degenerate behavior. Update the API doc accordingly.
4. **Add a new section: "Provisional API choice — pairwise vs accumulator."** State explicitly that the pairwise design is conditional on the cycle's call-pattern measurement; sketch the accumulator API as the alternative; note the decision endpoint.
5. **Property test sample-count gate.** Update the correctness property to use the corrected bound.

These five edits are entirely on the existing design document. No new code. ~30 minutes of work.

### Action 2 — Extend the remediation plan's cycle protocol (today)

Add D4's measurement table to `docs/REMEDIATION_PLAN.md`'s "Consumer-discovery cycle — measurement protocol" section. The cycle's instrumentation list grows from three items to five. The pass thresholds gain a row for "call pattern." The new row's threshold:

| Pattern | Verdict |
|---|---|
| Consumers naturally call pairwise | Pairwise design stands; implement as written |
| Consumers naturally accumulate | Redesign to accumulator API; pairwise becomes special case |
| Mixed (some pairwise, some accumulator) | Implement accumulator; pairwise is sugar over accumulator |

### Action 3 — Add §14.2 re-read to the cycle's RAW-phase checklist (today)

The substrate spec's existing prose may constrain the design in ways the design didn't honor. The cycle's RAW phase records what §14.2 says, what assumptions it makes, and which of those assumptions the design violated.

### Action 4 — Tier 3a (consumer rebuild) and Tier 3b (cycle) proceed as planned (next session)

With the corrected design and extended protocol, Tier 3a's lift and Tier 3b's measurements become a tighter test of a sharper hypothesis. No timeline change.

## Specification — corrected design saturation contract

Replace the current section in `docs/DESIGN_X-EXPO.md` with:

```
For Path A alignment (e_d = max(e_a, e_b)) with integer truncation of the
smaller operand:

If !saturated:
    require |decode(dst[i], e_d)
             − (decode(a[i], e_a) + decode(b[i], e_b))|
             ≤ 3^e_d
    /* One-trit precision at the result exponent. The error comes
     * entirely from integer truncation when dividing the smaller
     * operand's mantissa by 3^Δ. Truncation toward zero loses at most
     * (3^Δ − 1) / 3^Δ mantissa units at e_d, which is bounded by
     * 1 mantissa unit, which decodes to 3^e_d in real numbers. */

If saturated:
    require sign(dst[i]) == sign(decode(a[i], e_a) + decode(b[i], e_b))
    require dst[i] ∈ {+MAX_VAL, −MAX_VAL}
    require sat_flags[i] == 1   /* if sat_flags non-NULL */
```

## Specification — provisional API decision endpoint

Insert into `docs/DESIGN_X-EXPO.md` after the "API" section:

```
This API is provisional. The cycle's call-pattern measurement (tier 3b)
decides whether the kernel ships as pairwise (this signature), as an
accumulator (running, &e_running, new, e_new), or as both with one
implemented in terms of the other.

If the cycle reports >50% of consumer call sites use the kernel in a
running-accumulator pattern, the accumulator API becomes primary. The
design memo (tier 3c) records the choice; the kernel implementation
(tier 3d) ships the chosen API.

Pairwise is the simpler design and the simpler test. Accumulator is the
shape consumers may actually need. The cycle decides.
```

## Success criteria

This synthesis is successful if:

- [ ] The design's correctness bound is corrected before any property test is written.
- [ ] The pairwise-vs-accumulator question is explicit in the design, not buried in "future work."
- [ ] The cycle's measurement protocol gains the call-pattern measurement.
- [ ] §14.2 is on the cycle's RAW-phase checklist.
- [ ] Tier 3a and 3b proceed without a fresh discipline violation (the design's exploration status remains exploration, and the cycle tests the design's hypotheses rather than confirming them).

## What this synthesis surprised me with

I expected the LMM pass to confirm the design with minor refinements. It instead found one outright bug (the error bound) and one load-bearing missed question (pairwise vs accumulator). Neither would have been caught by reading the design once more; the bucket-based decomposition (mathematical / consumer-shape / discipline) is what surfaced them. The Laundry Method was load-bearing — the bug sat exactly at the boundary between Bucket A and Bucket B, which is where mistakes hide.

The cleaner this synthesis ends up looking, the more it confirms the method did its job. Six hours; four sharpening; the wood cuts itself when the grain is read correctly.

## Loop-back triggers (when to re-run this cycle)

- If the consumer-discovery cycle (tier 3b) surfaces a fourth consumer beyond the three currently named, re-enter REFLECT to re-test the design's hypotheses against it.
- If §14.2 turns out to specify semantics the design contradicts (rather than supersedes), re-enter NODES to remap the constraints.
- If after tier 3b the cycle's verdict is "all hypotheses confirmed; pairwise is right; bound is right," do NOT re-enter — that's the wood cutting itself, not a signal of missed grain.
