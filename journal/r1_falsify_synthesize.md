# SYNTHESIZE: methodical falsification of the R1 claim

Pre-committed plan + gates derived from `r1_falsify_reflect.md`.

## Decision

**Methodically falsify R1 by running 5 independent axes, each with a pre-committed numerical gate.** Replicate the 3 axes already measured (verify prior data still holds under current substrate); add the 2 missing/weak axes (F-A1 paired with intra-class consistency; F-A5 held-out routing accuracy). Final verdict: "R1 methodically falsified across 4-of-5 axes" or whatever the data actually shows.

This is a science cycle. Output is a journal verdict + falsification matrix, not a code change. Existing R1 infrastructure (`gesh/src/expr_signature.{h,c}::expr_to_signature_dual`, the `expr_routing_*.c` probe binaries) stays as-is.

## R1's claim, restated for falsification

**For an expression set E and test inputs X, the per-expression-tau dual-threshold signature rule discriminates expression equivalence classes strictly better than the sign-only rule on at least one of the 5 axes below.**

If R1's claim is true, at least 1 of {F-A1, F-A2, F-A3, F-A4, F-A5} should support it.

If 0/5 support: **R1 methodically falsified.**
If 1/5 supports (specifically F-A1 — class count alone): **R1 weakly falsified — class count alone doesn't constitute discrimination quality.**
If 2+/5 support: **R1 not falsified by this cycle; mixed verdict; need more axes.**

## Pre-committed gates

### F-G1 — Equivalence-class count + intra-class consistency (replaces N8/F-A1)

**Measurement:** for both rules, compute the number of distinct equivalence classes on the standard expression set (random-bank). Plus measure intra-class consistency: are expressions within one equivalence class actually mathematically equivalent?

**R1 supports iff:** dual produces ≥20% more classes than sign-only AND intra-class equivalence rate is ≥80% (i.e., expressions grouped together are in fact mathematically equivalent).

**R1 fails to support iff:** EITHER dual class count is within 20% of sign-only OR intra-class equivalence is <80% (more classes from arbitrary fragmentation, not principled).

**Source:** existing probe binaries in `gesh/bench/expr_routing_r1*.c`. If intra-class consistency isn't already measured, add it.

### F-G2 — Inter-class minimum distance (replicates N9/F-A2)

**Measurement:** minimum Hamming distance between expressions in DIFFERENT equivalence classes, both rules, both arity-1 and arity-2.

**R1 supports iff:** dual min ≥ sign-only min AND dual min ≥ 4 trits.

**R1 fails to support iff:** dual min < sign-only min OR dual min < 4.

**Prior data:** arity-1 dual=1, sign-only=3 → dual WORSE → FAIL.

**Replication:** re-run; verify the same data emerges.

### F-G3 — Partition-change rate (replicates N10/F-A3)

**Measurement:** what fraction of expressions are placed in DIFFERENT equivalence classes by dual vs sign-only? Random-bank, multi-seed.

**R1 supports iff:** mean partition change ≥ 30% AND the changes correlate with mathematical equivalence intuition (verifiable via spot-check on a curated set).

**R1 fails to support iff:** mean partition change < 30% OR changes are random (no correlation with intuition).

**Prior data:** mean 4.2% (gate ≥30%, fail-floor <15%) → FAIL.

**Replication:** re-run; verify.

### F-G4 — Substrate-novelty (third-state utilization) (replicates N11/F-A4)

**Measurement:** per-band cell distribution. What fraction of cells land in each band (negative / zero / positive)?

**R1 supports iff:** zero-band fraction is between 20% and 60% (load-bearing third state, neither dominant nor absent) for BOTH arity-1 AND arity-2.

**R1 fails to support iff:** zero-band fraction <20% (third state under-used) OR >60% (third state DOMINATES, becomes a default sink — the prior arity-1 finding).

**Prior data:** arity-1 zero-band 66.5% → FAIL (substrate-novelty failure).

**Replication:** re-run; verify.

### F-G5 — Held-out routing accuracy (NEW)

**Measurement:** train a class-mean bank on K labeled expressions; classify K' held-out expressions to their nearest bank class; measure accuracy against ground-truth equivalence labels. Compare dual vs sign-only.

**R1 supports iff:** dual accuracy > sign-only accuracy by ≥5pp on at least one configuration (specific arity + sig_dim).

**R1 fails to support iff:** dual accuracy ≤ sign-only accuracy on EVERY configuration tested.

**Construction:** reuse the existing labeled expression sets in `gesh/bench/expr_routing_*.c`. Split into 80% train / 20% held-out by RANDOM SEED (not by hand-picking, which could introduce selection bias). Multi-seed (5 seeds) for robustness.

**Configurations to test:** at minimum {arity=1, sig_dim=64} and {arity=2, sig_dim=64}. Per CONTRIBUTING multi-config rule.

## Order of execution

F-G1 → F-G2 → F-G3 → F-G4 (replicate prior; quick) → F-G5 (new construction; longer) → final verdict.

If any replicated axis (F-G2, F-G3, F-G4) shows DIFFERENT data than the prior cycle, STOP and investigate. The substrate change shouldn't affect R1 logic, but a discrepancy is informative.

## Risk register

- **R1 (replicated data differs from prior):** would mean the substrate change subtly affected R1 logic, OR my replication is misconfigured, OR prior data was wrong. Mitigation: investigate before proceeding; report finding.
- **R2 (F-G5 unexpectedly supports R1):** dual could win on routing accuracy even though it loses on signature-property axes. Outcome would be: signature properties fail but routing succeeds → R1 has limited validity. Honest verdict; surfaces an interesting finding.
- **R3 (F-G5 construction fail):** existing expression sets may not have a clean train/held-out split. Mitigation: construct a held-out set explicitly.

## What this cycle is NOT

- Not falsifying vision claim 3 broadly. Only the R1-specific expression of it.
- Not modifying R1 production code. Existing infrastructure stays.
- Not a perf cycle. No speedup measurements.

## Done when

5 axes measured with pre-committed gates. Final verdict tabulated. CLOSEOUT records:
- Per-axis verdict + numerical evidence
- Cumulative verdict (R1 falsified / weakly falsified / not falsified / mixed)
- Methodology lifted (if any)
- Forward pointer (does this affect vision claim 3 broadly? what's next?)

## Status

Pre-committed. Beginning F-G1 next.
