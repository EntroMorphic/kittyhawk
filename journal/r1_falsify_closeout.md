# CLOSEOUT: methodical falsification of the R1 claim

Per `journal/r1_falsify_synthesize.md`. Five-axis falsification matrix executed: 4 substantive axes verified against current substrate; 1 (F-G5) designed but deferred.

## Verdict: R1 METHODICALLY FALSIFIED

**On 3 of 4 substantive axes that measure discrimination quality, R1's dual-threshold rule fails to outperform sign-only. The one weakly-supporting axis (F-G1, equivalence-class count) is a non-quality metric — more classes alone doesn't constitute better discrimination.**

```
F-G1 (class count + intra-class consistency)  : WEAK SUPPORT — non-quality metric
F-G2 (inter-class minimum distance)            : FAIL — dual WORSE than sign-only
F-G3 (partition-change rate)                   : FAIL — 96% partition agreement
F-G4 (substrate-novelty / third-state usage)   : FAIL — third state OVER-DOMINANT
F-G5 (held-out routing accuracy)               : DEFERRED — see below
```

## Per-axis disposition

### F-G1: Equivalence-class count + intra-class consistency

**Measurement (replicated, current substrate):**
- arity-1: sign-only 22 classes, dual 30 classes → +36% more
- arity-2: sign-only 41 classes, dual 58 classes → +41% more

Both deltas exceed the 20% threshold for class count.

**Intra-class consistency (curated):** §2 "rule-difference" probe verifies 7/7 hand-designed equivalence pairs route to the same class under dual. Limited curated set, but consistency holds where measured.

**Verdict: WEAK SUPPORT for R1.** The dual rule produces more equivalence classes AND respects curated equivalences. But "more classes" doesn't intrinsically mean "better discrimination" — could be arbitrary fragmentation. Without an axis measuring whether the additional classes are MEANINGFUL, F-G1 is consistent with R1's claim but doesn't validate it.

### F-G2: Inter-class minimum distance

**Measurement (replicated):**
- arity-1: sign-only min=3, dual min=1 → DUAL WORSE
- arity-2: sign-only min ~8, dual min=8 → SAME

Pre-committed gate: dual ≥ sign-only AND dual ≥ 4 trits.

**Verdict: FAIL. Dual is strictly worse on arity-1.** Discrimination headroom decreased from 3 trits to 1 trit. The conf channel adds noise that brings tiles closer together — opposite of R1's claim that dual carries MORE discriminative information.

### F-G3: Partition-change rate

**Measurement (replicated):** mean 4.2% partition change between dual and sign-only across 5 seeds (sd 0.9pp). 96% of expressions are placed in the same equivalence class by both rules.

Pre-committed gate: ≥30% partition change (with correlation to mathematical intuition for the changes).

**Verdict: FAIL. The rules disagree on only ~4% of expressions.** Dual carries near-zero unique partition information — what it does differently is essentially noise relative to sign-only. R1's claim requires the rules produce MEANINGFULLY different partitions; the data shows them as functionally equivalent.

### F-G4: Substrate-novelty (third-state utilization)

**Measurement (replicated):**
- arity-1 zero-band: 66.5% of cells (FLAG: zero-dominated)
- arity-2 zero-band: 22.2% of cells (OK)

Pre-committed gate: zero-band fraction in [20%, 60%] for BOTH arities (third state load-bearing, neither absent nor dominant).

**Verdict: FAIL on arity-1.** The third state DOMINATES (66.5%) — it acts as a default sink, not as an information carrier. This contradicts R1's substrate-novelty framing (the dual rule was supposed to use the third state more meaningfully than sign-only). At arity-1, the dual rule uses the third state EXCESSIVELY to the point of dominance, which is the opposite of "load-bearing additional information."

### F-G5: Held-out routing accuracy — DEFERRED

**Design (per SYNTHESIZE):** train a class-mean bank on K labeled expressions; classify K' held-out expressions; measure routing accuracy against ground-truth equivalence labels; compare dual vs sign-only.

**Why deferred:** rigorous construction requires external equivalence ground truth — for two random expressions, knowing they're "actually equivalent" requires either (a) human curation (small scale only) or (b) algebraic-equivalence detection (substantial engineering). Neither is in scope for this science cycle, which is testing R1's claim against existing infrastructure.

**Possible proxies considered:**
- Routing stability under symmetry variants (commutativity / identity transforms): would test whether the rules respect known algebraic equivalences in random expressions. The §2 rule-difference probe does this on 7 hand-curated cases (7/7 PASS dual). Random-scale extension would require generating algebraic variants programmatically — non-trivial.
- Cross-rule routing agreement on held-out: tautological — F-G3 already measures this as 96% agreement.

**Honest framing:** F-G5 would round out the falsification with a routing-USE axis. The 4 axes already verified give a strong verdict on discrimination quality; routing accuracy would test whether the discrimination quality matters for actual classification performance. Without F-G5, the verdict rests on 4 axes; with F-G5, it would rest on 5. Both are methodical; the 4-axis verdict is sufficient but not maximally rigorous.

## Cumulative verdict

**R1 methodically falsified on every axis that measures discrimination quality.** F-G2 (dual worse on inter-class), F-G3 (rules functionally equivalent on partition), F-G4 (third state dominant rather than load-bearing) — all fail.

The weak support from F-G1 (more classes) does not redeem the claim. More classes without better discrimination is fragmentation, not refinement. The substrate's purpose for these signatures is **routing**, not class-counting; rule-induced over-fragmentation that doesn't improve routing is noise.

**The dual rule's only structural difference from sign-only is the conf channel, and the conf channel:**
1. Adds noise that decreases inter-class distance (F-G2)
2. Doesn't change partition assignment for ~96% of expressions (F-G3)
3. Pushes the third state into over-dominance at arity-1 (F-G4)

R1's underlying claim ("dual signatures discriminate better than sign-only") is **unsupported by every axis except a non-quality count metric.**

## What this falsifies — and what it does NOT

**Falsified:** the specific R1 implementation. Per-expression-tau dual-threshold rule does not produce better signatures than sign-only on the discrimination-quality axes that matter.

**NOT falsified — outside scope:**
- **Vision claim 3 broadly.** R1 was one specific operationalization of "third state is load-bearing." Other operationalizations (different test-input strategies, different signature derivation, different consumer-side use of the third state) remain testable. R1's failure is evidence that the dual-threshold approach doesn't work; it's not evidence that base-3's third state is uninformative.
- **Held-out routing performance** (F-G5). The cycle didn't measure routing accuracy under the dual rule on independent test sets. If a future cycle constructs F-G5 cleanly and the dual rule unexpectedly performs better on routing, the verdict shifts.
- **Future expressions of claim 3.** A different signature rule, a different consumer pattern, or a different evidentiary axis could all produce different verdicts.

## Methodology lifted

**1. Multi-axis falsification with pre-committed gates is the Popperian discipline pattern for substrate research claims.** The prior R1 cycle reached FAIL on 2 of 8 measured gates; this cycle restructured the same evidence around 4 substantive axes with explicit numerical thresholds, plus replicated all data under current substrate code. The restructured framing produces a cleaner verdict than the gate-by-gate FAIL/PASS/WEAK report.

**2. Replication is hygiene, not new evidence — but worth doing when the substrate has changed.** All 3 replicated measurements (F-G2, F-G3, F-G4) produced byte-identical numbers vs prior cycle. R1 algorithm logic was unchanged by the recent NEON kernel work. Confirmed.

**3. "More" doesn't equal "better" without a quality metric.** F-G1's class count is a quantity measure; without intra-class consistency or routing accuracy, more classes alone is consistent with arbitrary fragmentation. Future falsifications should pair quantity metrics with quality metrics.

**4. F-G5 (held-out test) requires external ground truth.** Constructing it rigorously for randomly-generated expressions requires algebraic equivalence detection or human curation — substantial engineering. Future cycles considering routing-accuracy tests should budget accordingly.

## Status of R1 in the project

R1 status moves from "FAIL per remediation cycle" to **"METHODICALLY FALSIFIED across 4 axes — discrimination quality, partition information, third-state utilization, and inter-class distance — with one non-quality axis (class count) weakly consistent."**

The dual-threshold infrastructure (`gesh/src/expr_signature.{h,c}::expr_to_signature_dual`, `expr_bank_dual_t`) remains in the codebase per project rule "DELETE = never." It's archived, not active.

Per the fork experiment, R-track was already closed; pivoted to P1-1 (exp/log primitives). The vision claim 2 work (math as routing signatures) is paused awaiting either:
- A different signature rule that addresses what dual missed
- A different bottleneck (the fork experiment found expression-set saturation, not rule choice, was the limiter at arity-1)
- Or work on vision claim 3 more directly (different operationalization)

## Honest concerns from this cycle

**1. The 4-axis verdict is robust but not maximally rigorous.** F-G5 would have added held-out routing accuracy. Without it, "R1 methodically falsified" rests on signature-property and substrate-novelty axes; routing-performance hasn't been directly measured. Documented as a follow-on if R1's status ever needs re-evaluation.

**2. F-G1's "weak support" interpretation is judgment.** Class count of +36-41% is not nothing. The verdict rests on the framing that "more classes alone isn't quality" — defensible but not unique. A different framing could call F-G1 "supports R1 weakly," which would make the cumulative verdict 1 weak / 3 fail rather than 0/4. Either way, R1 doesn't earn a "supported" verdict, but the strength of the falsification is interpretation-dependent at the F-G1 axis.

**3. The cycle didn't add new substrate evidence — it restructured existing evidence.** Replication of F-G2/F-G3/F-G4 against current substrate confirmed prior data; no new measurements (other than F-G1's intra-class consistency check via §2 probe). For users wanting "fresh" evidence, this cycle is light.

**4. Vision claim 3's broader status remains untested directly.** R1's falsification doesn't tell us whether claim 3 is true or false in general. The "third state is load-bearing" assertion isn't refuted by R1's failure; only the specific dual-threshold operationalization is. Future work testing claim 3 should explicitly note this scope.

## Status

CLOSED — R1 methodically falsified across 4 substantive axes. The one axis weakly consistent (F-G1 class count) is a non-quality metric and doesn't redeem the claim. The dual-threshold rule's specific implementation is documented as not delivering better signatures than sign-only on any axis that measures discrimination quality. R1 stays archived; vision claim 2 stays paused; vision claim 3's broader status remains untested directly.

**UPDATE 2026-05-05 — F-G5 closed.** Per `journal/r1_falsify_f_g5.md` (TD-8 closeout): F-G5 implemented with int64-evaluation behavioral-equivalence ground truth. On the canonical (tight-input, no-overflow) configuration, dual UNDERPERFORMS sign-only by 2.58 pp on held-out routing accuracy. R1 status now **methodically falsified across 5 substantive axes** (was 4). Apparent +8.23 pp dual win at wide input range was an int64 overflow artifact (caught by RC-1 red-team and remediated).

Followups (deferred):
- (none — F-G5 closed)
- A direct claim-3 falsification cycle that's NOT rule-specific (tests the third-state-as-load-bearing claim independent of any signature rule).
- A new signature rule attempt that addresses what dual missed (different consumer pattern, different test-input strategy).
