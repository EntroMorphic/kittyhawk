# Nodes of Interest: Path Forward After R1 Remediation FAIL

## Node 1: R1's failure mode is specific, not general
The dual rule does what it claims (kernels run, conf bits encode magnitude bands, predictable rule-difference cases route correctly). What it FAILED at was producing operational benefit: partition change is small, arity-1 discrimination got worse, cost is 5.68x.
**Why it matters:** the failure isn't "the rule is broken." The failure is "the rule's intended benefit doesn't materialize at this scale." Different next moves are appropriate for different framings of the failure.

## Node 2: The concerns R1 was supposed to address are still mostly open
Concerns 2 (substrate kernels unused), 3 (third state as binary), 7 (compose-equivalence), 8 (low headroom) are nominally addressed but not substantively. Concern 8 actually got WORSE.
**Why it matters:** picking a path forward means choosing how (or whether) to address these concerns by other means.

## Node 3: Three structural framings of the failure
- **F1 — "Wrong rule":** signature richness was the right axis, dual is the wrong implementation. Fix → R1 v2 (Option B).
- **F2 — "Wrong axis":** signature richness was never going to address the concerns; the consumer needs MORE CELLS, not RICHER CELLS. Fix → R3 sig_dim sweep with sign-only as primary (Option D).
- **F3 — "Wrong layer":** the concerns are substrate-level, not consumer-level. Expression-routing should accept sign-only and move on; vision claim #3 manifests elsewhere. Fix → revert and move to P1-1 (Option F).
**Why it matters:** F1 vs F2 vs F3 imply very different next cycles. The R1 evidence doesn't distinguish them yet.

## Node 4: Sig_dim sweep is the cheapest test that distinguishes the framings
R3 (sweep sig_dim ∈ {16, 32, 64, 128}) was already in the R2 plan. Running it with BOTH rules (not just one) would generate evidence about whether more cells beats richer cells.
- If sign-only at sig_dim=64 has min inter-class distance ≥ 6 → F2 wins (more cells is the right axis); revert R1.
- If sign-only and dual both plateau at sig_dim=64 with similar discrimination → F3 wins (the consumer is dim-saturated; pivot to P1-1).
- If dual at sig_dim=64 outperforms sign-only at same dim → F1 wins; rescue R1 v2.
**Why it matters:** ~3 days of work resolves a fork in the road that would otherwise be picked by intuition.

## Node 5: Concern 1 (scope gap) is the only "vision-level" concern in our list
Concerns 2/3/7/8 are about HOW the consumer uses the substrate. Concern 1 is about WHETHER expression-routing scales beyond toy. They're different categories.
**Why it matters:** R1 attempted to address consumer-level concerns. R2 was meant to address the vision-level concern. Failing at R1 doesn't tell us anything about R2's prospects — those are independent.

## Node 6: P1-1 (close primitives floor) is foundational
Without exp/log, the expression-routing work is structurally limited. Many natural mathematical equivalence classes (e.g., `exp(x)*exp(y) ≡ exp(x+y)`) can't even be EXPRESSED in the current vocabulary, let alone routed. P1-1 might unlock more than R-track work would.
**Why it matters:** if the goal is "show vision claim #2 at non-toy scale," P1-1 might be a more direct path than R3/R2 with the existing vocabulary.

## Node 7: Per-arity rules feel ad hoc
Option A (revert arity-1, keep dual for arity-2) is pragmatic but creates a code-level split. Two rules. Different per-arity behavior. Hard to reason about generalizing to arity-3.
**Why it matters:** even if Option A is the cheapest move, its long-term cost is fragmentation. Should be temporary if adopted at all.

## Node 8: Sunk cost is a real psychological pressure
We've invested ~3 weeks in R1 (planning, LMM cycle, code, red-team, remediation). The instinct is to extract value from that investment by patching the rule. But the most honest read of the data may be "the rule's premise was wrong; move on."
**Why it matters:** sunk-cost reasoning is the project's biggest immediate bias risk. Worth naming explicitly so it doesn't drive the choice.

## Node 9: Adversarial probes are cheap but don't change the verdict
Could spawn another subagent to design "break-the-rule" probes. Result would either reinforce the FAIL (more evidence the rule doesn't work) or surface a single failure mode. Either way the path forward isn't unblocked.
**Why it matters:** Option E is gating, not generative. Skip unless the user wants more evidence before committing.

## Node 10: The R2 plan's three-track structure (R1, R3, R2) was sequenced for the success case
R1 → R3 → R2 made sense if R1 PASSed: the sig_dim sweep would calibrate the new rule, then the scale experiment would test it at non-toy size. With R1 FAILing, the sequencing breaks. R3 with the failed rule is questionable; R2 even more so.
**Why it matters:** the plan needs replanning, not just rerouting through it.

---

## Tension Summary

- **T1 (closure cost vs depth):** Cheap moves (revert, accept FAIL) preserve momentum but leave concerns open. Deeper moves (R1 v2, sig_dim with both rules) cost more cycles but actually resolve the question.
- **T2 (consumer-level vs substrate-level fix):** R1 attempted a consumer-level fix for what may be a substrate-level problem. Trying again with a different consumer-level rule may repeat the mistake.
- **T3 (vision claim #2 scale gap vs vision claim #3 substrate-distinctness):** R1's failures are about claim #3. R2's promise is about claim #1 (scope gap) framed via claim #2. Different work; can't be conflated.
- **T4 (sunk cost vs honest pivot):** the temptation to "save R1" by patching it could waste another 2 weeks. The temptation to "abandon R1" could throw away real learning. Both are real biases.

## Dependencies

- **D1:** Choosing Option D (sig_dim sweep with both rules) before B/F/H is the falsification move that distinguishes F1/F2/F3. Other choices commit to a framing prematurely.
- **D2:** P1-1 (Option F) is independent of R-track outcomes. Could proceed in parallel with any other choice.
- **D3:** Per-arity rules (Option H) require a maintenance commitment that grows with each new arity. Treat as temporary scaffolding, not architecture.
