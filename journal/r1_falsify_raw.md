# RAW: methodical falsification of the R1 claim

Stream of consciousness on what the user means by "falsify the R1 claim, methodically" and what the cycle should look like.

## What R1 was

The original R1 (per `journal/r1_signature_rule_synthesize.md`): per-expression-tau dual-threshold signature rule for expression-routing signatures. Instead of a sign-only ternary code (each cell is just sign(expr_value)), use TWO thresholds per expression to derive a richer code (each cell encodes both sign AND strength relative to the per-expression τ).

**The claim, in plain terms:** the dual rule produces more discriminative / more informative expression signatures than the sign-only rule, on at least one meaningful axis.

The substrate-novelty framing: dual rule supposedly exercises the third state (zero) more meaningfully — the strength channel adds information sign-only can't carry. This was R1's bid for vision claim 3 (base-3's third state is load-bearing, not absence-of-signal).

## What prior cycles found

**R1 initial cycle** (`journal/r1_signature_rule_closeout.md`): VERDICT PASS on three easy gates (R1-A 96.7% backward-compat probes, R1-B 92% byte-different signatures, R1-C by construction). The PASS was structurally weak — gates measured "did the rule do something different" not "did it do something better."

**R1 red-team** (`journal/r1_signature_rule_redteam.md`): 13 findings. Critical: R1-B's "information gain" gate is satisfied by ANY use of the conf channel; R1-A's backward-compat probe set doesn't exercise the rule's new behaviors. Gates didn't bite.

**R1 100/100 remediation** (`journal/r1_remediation_closeout.md`): VERDICT FAIL on substantive gates:
- §1 partition-change: 4.2% mean (gate ≥30%, fail-floor <15%) — FAIL
- §6 inter-class distance arity-1: dual min=1 vs sign-only min=3 — DUAL WORSE — FAIL
- §4 granularity: 76.7% (gate ≥80%) — WEAK
- §2 rule-difference: 7/7 — PASS (but this is C2 — "the rule does differ in specific cases", which is necessary but not sufficient)
- §7 runtime regression: PASS (but this is just "kernel produces structured output", trivial)
- 3 diagnostic measurements

**R1 fork experiment** (`journal/r1_fork_closeout.md`): F3 wins. The expression set is signature-saturated at arity-1 regardless of rule (random-bank class count plateaus ~27 for sign-only and ~38 for dual). Adding sig_dim from 16 → 64 doesn't help either rule meaningfully. Conclusion: the bottleneck is the EXPRESSION SET (or the embedding it generates), not the RULE. Pivoted to P1-1 (exp/log primitives).

## Where R1 stands

**Functionally falsified** by the remediation cycle. Two FAIL gates, one WEAK, multiple diagnostic measurements showing the dual rule's "information" is largely cosmetic (96% partition agreement with sign-only) and arity-1 discrimination got WORSE (min distance dropped from 3 to 1).

The fork cycle doubled down: even sig_dim doesn't help; the bottleneck is upstream of the rule entirely.

## What "methodical falsification" adds beyond the existing FAIL

The prior cycle's gates were chosen ahead of time and they failed — that IS Popperian falsification. What could be more methodical?

A few candidates:

**1. Replicate the measurements.** The prior tests were run weeks ago against pre-NEON substrate code. The substrate has been heavily reworked since (shift3, ternary MAC, cross-exp accum all productionized). The ALGORITHM logic of expression routing is unchanged but a methodical re-test confirms the prior FAIL holds under current code.

**2. Add the missing axis.** Existing tests measure properties of the signature space (class count, distance, partition agreement). They do NOT measure routing accuracy on held-out test data. A "real-data" axis (synthetic but held-out) would round out the falsification: if dual is better at ROUTING (not just at being byte-different), R1 has at least one supporting axis. If it's the same or worse at routing, the falsification is robust.

**3. Substrate-novelty axis.** Per CONTRIBUTING rule 6 ("Substrate-novelty audit"). Does the dual rule exercise base-3's third state in a way sign-only structurally cannot? The remediation's per-band measurement showed arity-1 zero-band at 66.5% — the third state DOMINATES, which is the opposite of "load-bearing additional information." Substrate-novelty FAIL: dual uses the zero state but uses it as a default sink, not an information carrier.

**4. Final verdict statement.** A clean "R1 is METHODICALLY FALSIFIED across N independent axes" statement, with each axis named and its evidence cited. Replaces the prior fragmented "FAIL on gates 1, 6 + WEAK on 4 + PASS on 2, 7" framing with a single coherent verdict.

## Falsification axes I want to test methodically

**F-A1: Equivalence-class count.** Does dual produce SIGNIFICANTLY more equivalence classes than sign-only? Prior data: dual ~38, sign-only ~27 on random-bank. ~40% more — meaningful. THIS axis actually supports R1 weakly. But more classes doesn't mean BETTER routing — could just mean more arbitrary fragmentation.

**F-A2: Inter-class minimum distance.** Does dual have HIGHER min inter-class distance than sign-only? Prior: arity-1 dual=1, sign-only=3 → DUAL WORSE. R1 falsifying.

**F-A3: Partition-change rate.** Do the rules produce MEANINGFULLY different partitions? Prior: 4.2% partition change → 96% agreement. Dual carries near-zero unique information. R1 falsifying.

**F-A4: Substrate-novelty (third-state utilization).** Does the third state carry information, or dominate as a sink? Prior: arity-1 zero-band 66.5% → the third state is over-used, not informationally rich. R1 falsifying on the substrate-claim axis.

**F-A5: Held-out routing accuracy (NEW).** Train a bank on N expressions, classify M held-out expressions to their nearest bank class. Compare dual vs sign-only accuracy. If dual > sign-only by a meaningful margin (say ≥5pp), R1 supported on the routing axis. If equal or worse, R1 falsified on the routing axis too.

## What I'd land on

Run all 5 axes. Report each verdict honestly. If F-A5 also falsifies, then R1 is comprehensively falsified across signature properties (F-A2, F-A3, F-A4) AND routing performance (F-A5), with one weak supporting axis (F-A1: more classes — but this could be noise, not signal).

The verdict should distinguish:
- "R1's specific implementation FAILed" (already established)
- "R1's underlying CLAIM is structurally wrong" (what methodical falsification proves)

The prior cycle did the former. The methodical cycle should do the latter.

## Concerns

**1. Replicating prior measurements doesn't add new evidence.** If the prior FAILs were robust, replication just confirms what we know. Worth doing for hygiene but doesn't change the verdict.

**2. F-A5 (held-out routing accuracy) requires building a test that doesn't exist yet.** The expression-routing-probe test infrastructure exists but doesn't have a held-out test set. Need to construct: (a) train bank on K expressions, (b) classify K' held-out expressions, (c) measure correctness against ground truth equivalence classes.

**3. "Methodical" is a slippery word.** The prior cycle WAS methodical — it specified gates and ran them. What I'm adding is more axes + cleaner verdict structure. The user may want something else; should clarify if the cycle gets stuck.

**4. R1 is already archived per the fork cycle.** The codebase has the dual-threshold infrastructure but no current cycle uses it. Falsifying it more rigorously doesn't change current code. It's an EVIDENCE AUDIT for vision claim 2.

## What feels right

This is a SCIENCE cycle, not an engineering cycle. The artifacts are journal docs and probe binaries (mostly already exist). The output is a CONFIDENT VERDICT on R1's claim, not a code change.

Smaller cycle than the kernel cycles. Probably 1-2 hours of work:
- 30 min: replicate prior measurements (probably unchanged but verifying)
- 30-60 min: construct + run F-A5 held-out routing accuracy
- 30 min: write the methodical-falsification closeout

The deliverable: `journal/r1_falsify_closeout.md` with a clean verdict.

## Where I'd land if running on instinct

Sketch the 5-axis falsification matrix in SYNTHESIZE. Pre-commit gates per axis. Execute. Report verdict.

The R1 status would then move from "FAIL per remediation cycle" to "METHODICALLY FALSIFIED per 5-axis test, archived." Same ultimate state, but the audit trail is more rigorous.
