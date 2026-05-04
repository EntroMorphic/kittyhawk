# Closeout: R1 Red-Team Remediation (100/100 — VERDICT: FAIL)

Per `journal/r1_remediation_precommit.md` against the 13 findings in `journal/r1_signature_rule_redteam.md`.

## Overall verdict: FAIL

```
§1 partition-change   : FAIL (mean 4.2% sd 0.9pp)         gate >=30%, fail <15%
§2 rule-difference    : PASS (7/7 = 100.0%)               gate >=70%
§3 constant-offset    : reported (4/6 pairs merge)        diagnostic
§4 granularity        : WEAK (76.7% combined)             gate >=80%
§5 timing             : reported (dual 5.68x slower)      diagnostic
§6 inter-class M1     : FAIL (arity-1 min=1)              gate min>=4, fail min<2 → arity-1 hit fail floor
§7 runtime regression : PASS                              gate any conf set
§8 per-band dist      : reported (arity-1 zero 66.5%)     diagnostic + flag
```

Two FAILs (§1, §6); one WEAK (§4); two PASS (§2, §7); three diagnostics (§3, §5, §8). Per pre-committed gate semantics: **OVERALL FAIL**.

## Per-finding disposition (13/13 addressed)

| ID | Severity | Disposition | Outcome |
|----|----------|-------------|---------|
| **C1** | Critical | FIX — partition-change gate | **FAIL.** Mean 4.2% pair-change across 5 seeds; gate required ≥30%, fail-floor <15%. The dual rule does NOT meaningfully change the equivalence partition. |
| **C2** | Critical | FIX — rule-difference probes | **PASS.** 7/7 hand-designed probes route to dual-expected class. Rule does differ in specific predictable cases. |
| **H1** | High | FIX — constant-offset diagnostic | **REPORTED.** 4/6 pairs merge under dual; mixed behavior. `x*x` and `x*x+1` route to DIFFERENT classes (the +1 shift puts the shifted version in `x*(x-3)`'s class — accidental neighbor, not principled). `x*x` and `x*x-1` SAME (the documented merger). |
| **H2** | High | FIX — granularity sweep | **WEAK.** Arity-1 93.3%, arity-2 60.0%. Combined 76.7% below 80% gate. Granularity discontinuity is real for 2-var expressions where small constant shifts cross multiple band boundaries. |
| **H3** | High | FIX — multi-seed | **DONE.** All measurements under 5 seeds with mean ± stddev. Stddev 0.9pp shows results are stable across seeds (the FAIL is consistent, not luck-of-draw). |
| **H4** | High | FIX — timing diagnostic | **REPORTED.** Dual confidence-weighted distance is **5.68x slower** than sign-only popcount per route (0.278μs vs 0.049μs). Real cost for R2 scaling. |
| **M1** | Medium | FIX — inter-class distance under new rule | **FAIL.** Arity-1 minimum inter-class distance = 1 under dual rule (vs 3 under sign-only). Dual rule made discrimination WORSE, not better. Arity-2 OK at min=8. |
| **M2** | Medium | FIX — runtime regression check | **PASS.** All curated candidates have at least one conf bit set; the kernel is producing structured output, not silently degenerate. |
| **M3** | Medium | FIX — per-band distribution | **REPORTED + FLAG.** Arity-1 zero band 66.5% of cells (above 60% flag threshold). Arity-2 zero band 22.2% (OK). Confirms the red-team's M3 concern about zero-state dominance for monotone arity-1 expressions. |
| **M4** | Medium | FIX — unified seed list | **DONE.** All multi-seed measurements use `{0xa1..0xa5}`. Reports cross-comparable. |
| **L1** | Low | DEFER with rationale | DONE. M2 runtime check supersedes the CI-check need. |
| **L2** | Low | DEFER for P1-1 | DONE. Concern 7 re-evaluation requires exp/log primitives. |
| **L3** | Low | DEFER for R2-watch | DONE. Storage cost (1.5x) noted; matters only at R2 scale. |

**13/13 addressed. 10 measurements run, 3 explicitly deferred. The PASS/FAIL outcomes are the data.**

## What this FAIL teaches

The R1 verdict (`PASS PASS PASS`) was structurally weak because its gates measured the wrong things. The remediation's gates measure what matters — and the dual rule fails them.

Specifically:

**The dual rule's "information" is largely cosmetic.** Adding the conf channel makes signatures byte-different from sign-only (R1-B's 92%) but doesn't materially change which expressions get grouped together (§1's 4.2%). It's a new channel that mostly carries redundant information.

**The dual rule made arity-1 discrimination WORSE.** Inter-class min distance dropped from 3 to 1. The conf channel adds noise that brings tiles closer together rather than spreading them apart. This is the opposite of what concern 8 wanted.

**The new merger `x²-1 ≡ x*x` (H1's concern) is real and consequential.** Three classes that were distinct under sign-only (`|x|`, `x*x`, `(x-1)*(x+1)`) collapsed to one under dual. The closeout called this "defensible" on aesthetic grounds; the actual measurement shows it tightened the bank's discrimination headroom (M1's FAIL).

**Per-expression tau over-aggressive zero-banding.** For monotone arity-1 expressions, 9 of 16 cells routinely land in the zero band (M3's confirmed flag). Discrimination capacity is concentrated in 7 cells, half of which are typically the strong-band extremes that all monotone-shape expressions share.

**The new rule is 5.68x slower.** Even if it worked, this would be a real cost at scale. R2 scaling would feel it.

## Honest reframing

**R1's PASS verdict from `journal/r1_signature_rule_closeout.md` should be considered functionally OVERTURNED for arity-1.** The rule passes the original easy gates but fails the gates that actually measure what the rule was supposed to do (carry more information, improve discrimination, exercise substrate-distinctness in a way that matters operationally).

For arity-2, the picture is mixed: §1 still FAILs (only 3.4% pair change) but §6 inter-class distance is fine (min=8). The arity-2 zero-band distribution is healthy (22.2%). Arity-2 may be salvageable with the dual rule; arity-1 is not.

## Substrate-discipline notes

- All measurements ran through substrate kernels. No regression in kernel use (R1-C's claim survives).
- The new code (`expr_routing_r1_remediation.c`) builds clean under `-Werror`.
- All prior tests still PASS (verified separately).
- The ship-with-FAIL discipline (per the project's earlier P0-4 negative result) means the dual rule remains in the codebase — its FAILing measurements are documented at the verdict level, not by removing the code.

## Proposed next moves

This is honest data, not catastrophic. The project has shipped negative results before (P0-4). Three options:

**Option A: Revert to sign-only for arity-1; keep dual for arity-2.** The arity-2 case may benefit from the dual rule's richer encoding (zero band 22%, inter-class min 8). Arity-1 should revert. Compose the choice per arity. ~2 days work.

**Option B: Redesign the rule (R1 v2).** Per-expression tau over-aggressively zero-bands. Try alternatives: smaller zero band (e.g., `tau_weak = max_abs / 16`), or a hybrid (sign-only for monotone, dual for non-monotone), or fixed tau scaled to the test-input range rather than expression-specific. ~1 week design + 1 week implementation + verification.

**Option C: Accept the FAIL and proceed to R3/R2 anyway.** R3 (sig_dim sweep) might surface that with sig_dim=64 the dual rule's discrimination recovers. R2 (scale experiment) might show the rule works at scale even with arity-1 weakness. Risk: building on a known-broken foundation.

My read: Option A is cheapest and honest. Option B is the right research move if the goal is to actually solve concerns 2/3/7/8 properly. Option C compounds risk.

## What R1's overall track now looks like

| Cycle | Verdict |
|-------|---------|
| R1 original | PASS (gates didn't bite) |
| R1 red-team | 13 findings, 2 critical |
| R1 remediation (this) | FAIL on 2 of 6 gated sections; the rule is structurally weaker than its original PASS suggested |
| R1 next | open — needs Option A, B, or C decision before R3 / R2 build on top |

The discipline pattern (PASS → red-team → remediation → honest FAIL) worked exactly as designed. The original PASS was wrong; the remediation surfaced it; the verdict is honest.
