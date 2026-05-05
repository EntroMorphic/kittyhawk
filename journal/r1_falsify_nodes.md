# NODES: methodical falsification of the R1 claim

Atomic claims extracted from `r1_falsify_raw.md`.

## What R1 was, restated for falsification

- **N1.** R1 = per-expression-tau dual-threshold signature rule for expression-routing signatures. Each cell encodes sign + strength relative to the per-expression τ, instead of just sign.
- **N2.** R1's underlying CLAIM (in falsifiable form): "For an expression set E and test inputs X, dual-threshold signatures discriminate equivalence classes strictly better than sign-only signatures on at least one axis: (a) more classes, (b) higher min inter-class distance, (c) more meaningful partition (≥30% partition change), (d) better held-out routing accuracy."
- **N3.** R1's substrate-claim framing (vision claim 3): the dual rule should exercise base-3's third state as a load-bearing information carrier, not as a default sink.

## Prior verdicts (verified by reading closeouts)

- **N4.** R1 initial cycle (`r1_signature_rule_closeout.md`): VERDICT PASS on R1-A/B/C — but red-team caught these gates didn't bite.
- **N5.** R1 100/100 remediation (`r1_remediation_closeout.md`): VERDICT FAIL.
  - §1 partition-change: 4.2% (gate ≥30%, fail-floor <15%) → FAIL
  - §6 inter-class distance arity-1: dual min=1 vs sign-only min=3 → DUAL WORSE → FAIL
  - §4 granularity: 76.7% (gate ≥80%) → WEAK
- **N6.** R1 fork experiment (`r1_fork_closeout.md`): F3 wins. Sign-only and dual both saturate at arity-1 regardless of sig_dim. Bottleneck is upstream of the rule.
- **N7.** Per-band distribution measured: arity-1 zero-band 66.5% (M3 flag) — third state OVER-DOMINATES rather than carrying information.

## Falsification axes (existing data + one new)

- **N8.** F-A1 (equivalence-class count): dual ~38, sign-only ~27 (random-bank). +40% more classes — WEAK SUPPORT for R1. But class count alone isn't quality; could be arbitrary fragmentation.
- **N9.** F-A2 (inter-class minimum distance, arity-1): dual=1, sign-only=3 → DUAL WORSE → R1 FALSIFYING.
- **N10.** F-A3 (partition-change rate): 4.2% → 96% agreement → dual carries near-zero unique partition information → R1 FALSIFYING.
- **N11.** F-A4 (substrate-novelty / third-state utilization): arity-1 zero-band 66.5% → third state is a default sink, not a load-bearing carrier → R1 FALSIFYING ON CLAIM-3 AXIS.
- **N12.** F-A5 (held-out routing accuracy): NOT YET RUN. The missing axis. If dual > sign-only by ≥5pp on held-out classification, R1 supported on routing. If equal/worse, R1 falsified on routing.

## Cycle scope

- **N13.** Replicate F-A1 through F-A4 against current substrate code. Algorithm logic unchanged from prior cycle (R1 expression routing infrastructure in `gesh/src/expr_signature.{h,c}`); should produce same numbers. Verifies prior data still holds.
- **N14.** Construct F-A5 test: train a class-mean bank on K labeled expressions, classify K' held-out expressions, measure routing accuracy. Compare dual vs sign-only.
- **N15.** Final verdict: R1 methodically falsified iff F-A2, F-A3, F-A4, F-A5 all fail to support; F-A1 may weakly support (more classes) but doesn't constitute SUPPORT for the underlying claim.

## Methodology constraints

- **N16.** This is a SCIENCE cycle, not an engineering cycle. Output is a journal verdict + falsification matrix, not a code change. The dual-threshold infrastructure already exists (`gesh/src/expr_signature.{h,c}::expr_to_signature_dual`).
- **N17.** R1's status moves from "FAIL per remediation" to "METHODICALLY FALSIFIED across 5 axes, archived." Same outcome, more rigorous evidence trail.
- **N18.** Per memory: no consumer-demand framing; no speedup-gating. Vision-claim-3 evidence is the directive.

## Concerns

- **N19.** Replication of prior measurements is hygiene, not new evidence. If the prior FAILs were robust, replication just confirms what we know. Worth doing but not the load-bearing piece.
- **N20.** F-A5 (held-out routing accuracy) is the missing axis. Constructing a fair test requires: (a) labeled expression equivalence classes for training, (b) DIFFERENT held-out expressions with KNOWN equivalence to a training class. The prior expression sets in `gesh/bench/expr_routing_*.c` may already provide this structure.
- **N21.** Claim 3 is BIGGER than R1. R1 is one specific test of claim 3. Falsifying R1 doesn't falsify claim 3 broadly. The cycle should be clear about this distinction.
- **N22.** Results may be ambiguous. F-A1 supports weakly; F-A5 hasn't run. The cycle should report each axis honestly even if the cumulative verdict is mixed.

## Open questions

- **N23.** Is there ANY expression-set + test-input combination where dual > sign-only on routing accuracy? If yes, R1's claim has limited validity (weak claim 3 support). If no, R1 is robustly falsified.
- **N24.** Does R1's failure say anything about claim 3 generally? Or just about THIS rule? The fork cycle hinted: bottleneck is upstream of the rule. If rules don't matter, claim 3's "third state is load-bearing" needs different operationalization (not at the SIGNATURE-RULE layer).
- **N25.** What expression-set / signature-derivation pattern WOULD test claim 3 cleanly? Open question; not for this cycle, but for the journal record.

## What this cycle is NOT

- **N26.** NOT a re-implementation of R1. Existing code stays.
- **N27.** NOT a falsification of vision claim 3 broadly. Only of the specific R1 expression of it.
- **N28.** NOT an engineering cycle. No production substrate change.
- **N29.** NOT a perf cycle. No speedup measurements.
