---
date: 2026-04-14
phase: SYNTHESIZE
topic: What is the right criterion for "base-3 native," and what replaces sign_extract?
---

# Synthesize

## The criterion

**A primitive is base-3 native by the substrate/consumer contract.** Not as a single property.

Two-part test:

**C-sub (substrate side).** The primitive's semantic definition distinguishes three states structurally. Test: collapse any one of the three states in the definition; if the semantic becomes ill-defined or trivially equivalent to a binary operation, fail. The substrate audit applies C-sub to each primitive it exposes.

**C-con (consumer side).** The consumer's input distribution at this primitive realizes all three states with non-trivial probability. Test: sample realistic inputs; observe that each of the three states occurs often enough that the primitive's three-way semantic is actually exercised. The consumer asserts C-con as a precondition when it calls the primitive.

Together: a primitive IS base-3 native in a given deployment iff C-sub holds (substrate guarantee) AND C-con holds (consumer precondition).

This matches the throughline. Substrate guarantees three-way capacity; consumer asserts three-way realization. Neither side is absolute.

## The replacement (and non-replacement)

**sign_extract is not deleted.** It passes C-sub (semantic is genuinely three-way: positive → +1, negative → -1, zero → 0). It fails C-con for MTFP projection inputs (zero is measure-zero). The fix is naming the contract, not removing the primitive.

**`m4t_route_threshold_extract` is added** as a new substrate primitive: `|v| < τ → 0, v ≥ τ → +1, v ≤ -τ → -1`. This primitive passes C-con unconditionally for any τ > 0 — the zero state is guaranteed by an explicit band, not a measure-zero coincidence. Consumers whose input is MTFP projections (where sign_extract would fail C-con) use this.

**sign_extract is the τ=0 degenerate of threshold_extract.** We document this relationship in the substrate spec. A future clean-up could unify them; for now, the rename-and-add approach preserves backward compatibility with any consumer whose input distribution does realize zero naturally (e.g., signature_update's internal call, where col_sum − mean is an integer that genuinely hits zero).

**Option 3 (MTFP4-native signatures, no extraction)** is NOT a substrate change. It is a consumer pipeline pattern: a consumer that produces MTFP4 projections and packs them as signatures directly — no extract step, no primitive needed. The substrate already provides what's needed (`m4t_mtfp19_to_mtfp4` or MTFP4-native SDOT). Document this as a distinct pattern in THESIS.md or consumer guides.

## Concrete substrate changes (scope)

**Add:**
- `m4t_route_threshold_extract(uint8_t* dst_packed, const int64_t* values, int64_t tau, int n)`.
- Tests in `test_m4t_route.c`: τ=0 reduces to sign_extract exactly; positive τ produces all three states on realistic inputs; negative τ is rejected (precondition).

**Rename (or annotate in place):**
- `m4t_route_sign_extract` — not the symbol, but its docstring. Name C-con explicitly: "produces the zero trit ONLY when the input value is exactly zero. For inputs where `v == 0` is a measure-zero event (MTFP projections, most floating-pointy-shaped continuous outputs), this primitive is a binary sign-extractor in practice. Use `threshold_extract` for thresholded three-state extraction."

**Do not delete:**
- `m4t_route_sign_extract` — has legitimate consumers (signature_update's internal call).
- `m4t_route_signature_update` — its internal sign_extract call is correct for the integer col_sum-minus-mean inputs; unchanged.

**Add spec section:**
- `m4t/docs/M4T_SUBSTRATE.md` gains §18 (or appended section): "Base-3 native: substrate and consumer contracts." Articulates C-sub and C-con. Becomes the criterion any future primitive review applies.

**Update existing docs:**
- `docs/REMEDIATION_PLAN.md`: close the "sign_extract is binary" finding as a misdiagnosis — the primitive is three-state-capable; the bug was using it with a consumer that violated C-con.
- `NORTH_STAR.md`: no changes. The compass was right; I applied it imprecisely.

## Scope (revised from the prior estimate)

Previously I estimated ~250 lines of code to rewrite. The corrected scope is:

- ~60 lines added: `m4t_route_threshold_extract` implementation + header.
- ~40 lines added: tests for the new primitive.
- ~30 lines changed: docstring on `sign_extract` naming C-con explicitly.
- ~40 lines added: §18 "base-3 native contract" in substrate spec.
- ~10 lines added/changed in `docs/REMEDIATION_PLAN.md` (closing the finding as misdiagnosis).
- 0 lines deleted from the existing substrate.
- 0 lines changed in `m4t_route_signature_update`.
- Consumer impact: `tools/mnist_routed_lattice.c` can optionally switch to `threshold_extract` for re-experimentation. Not required by this change.

Approximately **180 lines**, most of them additive. Not an everything-rewrite. Not even the 250-line rewrite I estimated before the cycle.

## What the cycle changed about my mental model

Before: sign_extract is binary-shape; replace it.
After: sign_extract is three-state-capable; add a sibling for use cases where sign_extract's zero-state is measure-zero; document the contract.

Before: the criterion is a property of a primitive.
After: the criterion is a two-part contract — substrate claims capacity, consumer asserts realization.

Before: the replacement question is "pick Option 1, 2, or 3."
After: Option 2 is a substrate addition, Option 3 is a consumer pipeline choice; Option 1 is subsumed by Option 2; we do Option 2 now and document Option 3 as a pattern.

Before: the fix is a code change.
After: the fix is mostly documentation of a contract we never wrote down, plus one new primitive.

## Success criteria for the cycle

- [x] Criterion articulated in testable form (C-sub, C-con).
- [x] Criterion is mechanical (counterfactual collapse test on the substrate side; input-distribution test on the consumer side).
- [x] Criterion maps onto the existing substrate/consumer throughline.
- [x] Replacement question resolved: add rather than replace; name the contract rather than delete.
- [x] Scope quantified (~180 lines, mostly additive, mostly documentation).
- [x] No consumer forced to change unless it wants to experiment.

## What to do next

1. Pause here. User validates C-sub + C-con before any code.
2. If validated: add `m4t_route_threshold_extract` + tests; annotate `m4t_route_sign_extract` docstring; append §18 to `M4T_SUBSTRATE.md`; close the finding in `REMEDIATION_PLAN.md`.
3. Optionally (separate step): modify `mnist_routed_lattice.c` to use `threshold_extract` with a tuned τ; re-run and measure whether genuinely-three-state routing changes the 58.37% accuracy.
4. Continue with other thesis-level work (benchmark bed, hardware utilization discharge, etc.) as separate tracks.

## What I'm explicitly NOT doing

- Not deleting sign_extract.
- Not changing signature_update.
- Not building Option 3 (MTFP4-native signatures) as a substrate change.
- Not running experiments based on the cycle's conclusions until the criterion is validated.
