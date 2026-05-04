# Red-Team: R1 (Per-Expression-Tau Dual-Threshold Signature Rule)

Adversarial pass on R1's PASS verdict from `journal/r1_signature_rule_closeout.md`. The closeout itself flagged several limits in its "honest concerns" section; this document sharpens the framing and adds findings the closeout didn't.

Findings: 2 critical, 4 high, 4 medium, 3 low.

---

## C1 — R1-B is satisfied by ANY use of the conf channel, not by useful information

The gate measures "did the signature bytes differ from sign-only?" The dual signature has a NEW channel (conf bits) that didn't exist in sign-only. *Any* expression with non-trivial values will set at least one conf bit, making the dual signature byte-different from the sign-only signature by construction. The 92% PASS rate proves "we added a new channel," not "the new channel carries useful information."

Concrete attack: a rule that emits `m4t_route_threshold_extract_dual` with `tau_weak = max_abs / 1000` (almost everything is "strong") would still produce dual signatures with conf bits all set, byte-different from sign-only on ~100% of random expressions. R1-B would PASS at higher than 92%. But that rule would carry no useful information — every expression would be "all strong-positive or all strong-negative."

The honest gate would measure **partition change**: does the new rule produce a meaningfully different equivalence-class partition? The closeout reported this informally (+8 / +17 classes per 100 random expressions) but didn't gate it. Without the gated measure, the PASS verdict is weaker than its name suggests.

**Severity:** critical. The gate as written doesn't measure what its label claims.

**Recommended:** strengthen R1-B in any follow-on cycle to "≥30% of random-expression PAIRS that merged under sign-only now split under new rule, OR vice versa." That measures actual partition change, not byte change.

---

## C2 — R1-A backward-compat measured on a probe set that doesn't exercise the new rule

The 30 subagent probes are mostly clean algebraic equivalents (`x+0 → x`, `y+x → x+y`, `min+max → x+y`, etc.). For these probes, both the sign-only rule and the dual rule produce equivalent routings — they're CLEAN equivalences that any sane signature rule would recognize.

The 96.7% match under the new rule is therefore *not* evidence that the new rule preserves backward-compat. It's evidence that the probe set is dominated by easy cases that both rules handle identically.

The cases where the rules genuinely differ are exactly the cases the probe set didn't include — magnitude-distinct equivalents like `x²-1` (which routes to `x*x`'s class under the new rule but to its own class under sign-only). The probes don't test those transitions.

**Severity:** critical. The verdict "rule preserves what worked" is unsupported by this probe set.

**Recommended:** construct a "rule-difference probe set" — probes specifically designed to differ between sign-only and dual-rule routings. If subagent intuition matches the dual-rule's routing on those probes, that's real backward-compat evidence. If intuition splits, the new rule may be making mathematically wrong choices.

---

## H1 — The new merger `x²-1 ≡ x*x` was reported but not interrogated

Under per-expression tau, the curated arity-1 bank now has `(x-1)*(x+1)` (i.e., `x²-1`) merged into the `x*x` class. Mathematically these are different functions: `x²` and `x²-1` differ by a constant. The merger happens because their tiny absolute difference (1 unit) is below the per-expression weak threshold (`max_abs/4 = 225` for `x²` on our test inputs).

The closeout calls this "defensible" because the magnitude profiles are similar. But "defensible" isn't "correct." If a downstream consumer uses the bank to recognize symbolic equivalence (where adding 1 matters), the dual rule's merger is a wrong answer.

The closeout flagged this honestly but didn't propose a test for whether such downstream consumers would be hurt. The merger is being accepted on aesthetics ("similar magnitude profiles → similar enough"), not on a test that proves the merger is harmless.

**Severity:** high.

**Recommended:** add a probe class specifically targeting "constant-offset equivalents" (`x*x` vs `x*x + 1` vs `x*x - 5`). If they all merge, the rule has chosen magnitude-profile equivalence; if they don't, the rule discriminates by offset. Either result is informative; the current state (assumed-defensible without test) is not.

---

## H2 — Integer division in tau computation creates granularity discontinuities

`tau_weak = max_abs / 4` and `tau_strong = max_abs / 2` use C integer division. For `max_abs ∈ [0, 3]`, `tau_weak = 0`. For `max_abs ∈ [4, 7]`, `tau_weak = 1`. Etc. Two near-equivalent expressions whose `max_abs` values straddle a granularity step (say one has max=7, other has max=8) get tau values that differ by 1, which can flip cell classifications and produce noticeably different signatures.

This discontinuity is most pronounced at small max_abs values, where the integer division steps are largest as a fraction. For expressions whose max output happens to be small (e.g., `min(x, 2) - max(x, -2)` for x in [-3, 3]), the tau values can swing dramatically with small input changes.

Untested in R1's verification.

**Severity:** high. Could cause near-equivalent expressions to route to different classes for arithmetic reasons unrelated to the rule's design intent.

**Recommended:** sweep small-max expressions and report the rate of granularity-induced signature flips. Or: switch to a tau computation that's robust to integer-division granularity (e.g., shift the threshold by a small fixed offset, or use a continuous formula).

---

## H3 — Information-gain test was single-seed

`info_gain_count` was called with seeds `0xa1u` (arity-1) and `0xa2u` (arity-2). One seed each. The remediation cycle's M2 discipline (multi-seed) wasn't applied here.

The 92% PASS could be a single-seed lucky draw. Without multi-seed runs, we don't know the variance. A different seed might produce 50%, or 99%, or anywhere in between.

**Severity:** high. Single-seed results don't generalize per the project's own discipline rule.

**Recommended:** rerun with 5 seeds, report mean ± stddev. If stddev is > 10pp, the gate threshold needs to account for variance.

---

## H4 — Confidence-weighted distance kernel cost wasn't measured

`m4t_route_confidence_weighted_dist` has a per-position scan loop with branches (per `m4t_route.c`). Slower than `m4t_popcount_dist`'s pure SIMD popcount. At toy scale (16-trit signatures, 11-class banks), irrelevant. At R2 scale (sig_dim 64+, 1000+ classes), could be the bottleneck.

R1 ships with no measurement of this cost. R2 will discover it the hard way.

**Severity:** high (for downstream R2). Low for R1's own scope.

**Recommended:** add a simple timing report to the R1 probe — milliseconds per probe routing under each rule. Establishes a baseline for R2's scaling work to track against.

---

## M1 — Bank inter-class distance under new rule wasn't re-measured

Concern 8 was the original red-team's flag that arity-1 inter-class min distance was 3 (below 4-trit headroom threshold) under sign-only. The R1 closeout claims concern 8 is "partially addressed" because the new rule produces +8/+17 more classes per 100 random expressions.

But: more classes doesn't mean better inter-class spread. Two new classes that are 1 trit apart from each other or from existing classes is *worse* than fewer classes that are well-separated.

The §5 inter-class diagnostic from the original remediation binary wasn't re-run under the new rule. We don't actually know whether the new rule improved or worsened the headroom.

**Severity:** medium.

**Recommended:** R3 (sig_dim sweep) will measure this. But R1's "partially addressed" claim is premature.

---

## M2 — R1-C is satisfied "by construction" via grep, not by runtime check

The gate "new rule's call path includes ≥1 previously-unused substrate kernel" is verified by code review (grep). A future refactor that removes the kernel call (e.g., inlining the dual-threshold logic in C) would silently break the gate without any runtime signal.

The gate doesn't verify that the kernel is doing useful work either — it only verifies the kernel is invoked. A rule that called the kernel and then overwrote its output with sign-only would still pass R1-C.

**Severity:** medium. The gate is real but mechanical.

**Recommended:** add a runtime probe that compares dual-signature output to a sign-only output and asserts they're not bit-identical (when they shouldn't be). Catches silent regression.

---

## M3 — For typical expressions, the zero band dominates

Per-expression tau means many cells land in the "zero" band. Concrete: `x` evaluated at our 16 inputs has max_abs=30. `tau_weak = 7`. Cells where `|x| ≤ 7` (positions 3-10, i.e., x ∈ {-5,-3,-2,-1,0,1,2,3,5}) all classify as zero — that's 9 of 16 cells.

So `x`'s dual signature is approximately:
- Strong-neg at positions 0,1 (x = -30, -15, both well above strong threshold of 15)
- Weak-neg at position 2 (x = -10)
- Zero at positions 3-10 (9 cells)
- Weak-pos at position 11 (x = 5? no wait, 5 < 7 so zero)

Actually let me recompute. x at our 16 inputs:
{-30, -15, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 15, 20, 30}
max_abs = 30, tau_weak = 7, tau_strong = 15.

- pos 0 (x=-30): -30 < -15 → strong-neg, conf=1 since |-30| > 15
- pos 1 (x=-15): -15 < -7 → trit=-1; |-15| > 15? Strict, so no → conf=0. Weak-neg.
- pos 2 (x=-10): trit=-1, conf=0. Weak-neg.
- pos 3 (x=-5): -5 > -7? Yes → trit=0. Zero.
- pos 4 (x=-3): zero
- pos 5 (x=-2): zero
- pos 6 (x=-1): zero
- pos 7 (x=0): zero
- pos 8 (x=1): zero
- pos 9 (x=2): zero
- pos 10 (x=3): zero
- pos 11 (x=5): 5 > 7? No → zero
- pos 12 (x=10): 10 > 7 → trit=1, conf=0. Weak-pos.
- pos 13 (x=15): 15 > 7 → trit=1; 15 > 15? Strict, no → conf=0. Weak-pos.
- pos 14 (x=20): trit=1, conf=1. Strong-pos.
- pos 15 (x=30): trit=1, conf=1. Strong-pos.

So `x`'s dual signature: 1 strong-neg, 2 weak-neg, 9 zeros, 2 weak-pos, 2 strong-pos. **9 of 16 cells (56%) are zero.**

The "5-state encoding" undersells how dominated the zero state is for self-normalized signatures of monotone expressions. Discrimination capacity is concentrated in the few non-zero cells.

**Severity:** medium. Affects how much information the rule actually carries vs. its theoretical capacity.

**Recommended:** consider asymmetric tau (e.g., `tau_weak = max_abs / 8`, `tau_strong = max_abs / 2`) to give the weak band more room and reduce zero-state dominance. Or: report per-band cell-count distribution as a diagnostic.

---

## M4 — `info_gain_count` and `partition_diff` use different seeds

`info_gain_count` for arity-1 used seed `0xa1u`. `partition_diff` for arity-1 used seed `0xb1u`. Different random expression sets. The 85% information-gain rate and the +8 partition delta describe different samples, can't be cross-referenced or jointly analyzed.

**Severity:** medium. Reports look comparable but aren't.

**Recommended:** use the same seed for both, OR run both metrics over a shared sample set per seed, with multi-seed reporting (per H3).

---

## L1 — R1-C verifiable only by grep

If a future refactor inlines the dual-threshold logic and removes the explicit `m4t_route_threshold_extract_dual` call, R1-C silently still PASSes "by construction" because the construction was a one-time grep. No runtime guard prevents this regression.

**Severity:** low. A real risk only if someone attempts that refactor without re-running R1.

**Recommended:** add a CI check that `expr_to_signature_dual` body contains a call to the kernel. Trivial regex check.

---

## L2 — "Concern 7 partially closed" rests on `x ≢ x³` analogy

The closeout claims concern 7 (compose-equivalence collapses precision-distinct expressions) is partially addressed because the new rule splits `x ≢ x³` (different magnitude profiles). But the actual case the concern was about (`exp(x)` vs Taylor truncation) cannot be tested in the current expression vocabulary (no exp).

The analogy might or might not hold for transcendentals. The "partial closure" is weaker than it sounds.

**Severity:** low. The analogy is plausible; no false claim, just an overstated closure.

**Recommended:** re-evaluate concern 7 after P1-1 (exp/log primitives) lands. Until then, mark concern 7 as "mechanism plausible, not yet tested in target case."

---

## L3 — Doubled per-tile storage

The dual rule stores trit signature + conf bitmap per tile. At sig_dim=16, that's 4 bytes + 2 bytes = 6 bytes per tile (vs 4 bytes for sign-only). 1.5x storage. Trivial at toy scale; matters at R2 scale where bank size could be 2000+.

Not a verdict-changer; worth tracking.

**Severity:** low.

**Recommended:** R2's scaling experiment should report bank memory footprint as one of its diagnostics.

---

## Summary

| ID | Severity | Status |
|----|----------|--------|
| C1 | Critical | R1-B gate doesn't measure information gain in any meaningful sense |
| C2 | Critical | R1-A probe set doesn't exercise the rule's new behaviors |
| H1 | High | New merger `x²-1 ≡ x*x` accepted on aesthetics, not test |
| H2 | High | Integer-division granularity in tau creates discontinuities |
| H3 | High | Single-seed information-gain test, no multi-seed |
| H4 | High | Confidence-weighted distance kernel cost unmeasured |
| M1 | Medium | Bank inter-class distance not re-measured under new rule |
| M2 | Medium | R1-C grep-verified, no runtime check |
| M3 | Medium | Zero state dominates signatures (9/16 cells for `x`) |
| M4 | Medium | Seed mismatch between info-gain and partition-diff reports |
| L1 | Low | Future refactor could silently bypass R1-C |
| L2 | Low | Concern 7 closure analogous, not tested |
| L3 | Low | 1.5x per-tile storage; matters at R2 scale |

## What this red-team changes about the verdict

R1 still PASSes its gates as written. But the verdict is **weaker than the gate names suggest**:

- "Backward compat" is established only on cases where both rules agree (C2). The rule's new behaviors are untested.
- "Information gain" is established as "the rule produces different bytes" not "the rule produces more useful information" (C1). The honest gate would measure partition change.
- "Substrate-kernel use" is established mechanically, not behaviorally (M2).

The rule is real, the new substrate kernels are exercised, the partition does change at scale. But the *evidence* for these claims is structurally weaker than the verdict's PASS/PASS/PASS table implies.

**Recommendation:** before R3 (sig_dim sweep) or R2 (scale experiment), address C1 and C2 in a remediation pass. C1 by replacing R1-B with a partition-change gate. C2 by constructing rule-difference probes specifically designed to surface where the dual rule diverges from sign-only. If those tests still PASS, the R1 verdict is robust. If they FAIL, R1 needs a redesign before R3/R2 build on it.
