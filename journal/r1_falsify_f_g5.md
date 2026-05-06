# R1 falsification F-G5: held-out routing accuracy

Closes TD-8 from `docs/TECHNICAL_DEBT.md`. Per `journal/r1_falsify_closeout.md` "Followups (deferred)" — the 5th axis of the R1 falsification matrix that was deferred in the original closeout for "external equivalence ground truth requires substantial engineering."

## Method

For each input range (band):
1. Generate K_TOTAL = 8000 random arity-1 expressions, max_depth = 4.
2. Compute "behavioral fingerprint" per expression: int64 evaluation on N_FP = 32 fixed inputs.
3. Group expressions by fingerprint into equivalence classes — two expressions with byte-identical fingerprints are declared equivalent.
4. Filter to classes with ≥ 4 members (need 1 anchor + ≥ 3 held-out test exprs).
5. Per kept class: first member = bank anchor; rest = held-out test set.
6. Build TWO banks from anchors: sign-only (`expr_to_signature`) and dual (`expr_to_signature_dual`). One tile per class.
7. Route each held-out test expr → predicted class = nearest bank tile (Hamming for sign-only, confidence-weighted for dual).
8. Routing accuracy = (predicted_class == true_class) / total_test.

**Pre-committed gate:** dual must beat sign-only by ≥ 2 pp absolute to count as a verdict shift.

## Red-team RC-1 (caught BEFORE finalizing the verdict)

The first run used the "wide" {−30..30} input band per `expr_routing_r1.c`'s convention. Result: dual beats sign-only by +8.23 pp — apparent verdict shift.

**RC-1 caught the artifact.** Depth-4 random expressions on inputs in {−30..30} can produce mul-heavy chains that overflow int64 (max |x|^16 ≈ 10^23, vs int64 ceiling ≈ 9.2 × 10^18). Overflow fragments otherwise-equivalent expressions into distinct fingerprints, biasing the equivalence-class set toward "trivial" classes (constants, all-zero outputs) where dual's magnitude information happens to help.

**Remediation:** rerun with the tight {−3..3} band (max |x|^16 = 3^16 ≈ 43 M, well within int64). Both bands now run; tight is canonical, wide is reported as a sanity-check showing how much the verdict depends on input range.

## Results

```
# Input band: tight  {−3..3} — canonical (no int64 overflow)
Generated 8000 expressions → 1526 distinct fingerprints
Kept 195 classes (≥ 4 members each) → 195 anchors + 6116 held-out tests

  sign-only : 1448 / 6116  =  23.68%
  R1 dual   : 1290 / 6116  =  21.09%
  gap (dual − sign-only) : −2.58 pp

  Per-class breakdown over 195 classes:
    classes where dual > sign-only : 18
    classes where dual = sign-only : 157
    classes where dual < sign-only : 20
```

```
# Input band: wide   {−30..30} — sanity-check (some overflow expected at depth 4)
Generated 8000 expressions → 1978 distinct fingerprints
Kept 195 classes → 195 anchors + 5592 held-out tests

  sign-only :  822 / 5592 = 14.70%
  R1 dual   : 1282 / 5592 = 22.93%
  gap (dual − sign-only) : +8.23 pp

  Per-class breakdown over 195 classes:
    classes where dual > sign-only : 44
    classes where dual = sign-only : 148
    classes where dual < sign-only :  3
```

## Verdict

**F-G5: R1 STILL FALSIFIED. Now 5-axis verdict.**

On the canonical (tight, no-overflow) configuration, the dual rule UNDERPERFORMS sign-only by 2.58 pp — passing the pre-committed gate in the *opposite* direction (gap = −2.58 pp, |gap| > 2). Per-class breakdown is essentially symmetric (18 dual-better vs 20 dual-worse out of 195) — no systematic dual advantage.

The apparent +8.23 pp win on the wide band is an artifact: int64 overflow fragments equivalent expressions into spurious distinct classes, biasing the kept-class set toward "trivial" classes where dual happens to win.

**R1 status update:**
- Was: "methodically falsified across 4 axes — discrimination quality, partition information, third-state utilization, inter-class distance — with one non-quality axis (class count) weakly consistent."
- Now: same 4 axes plus F-G5 confirms FAIL on the routing-USE axis under behavioral-equivalence ground truth.

## Honest concerns about the F-G5 method

**1. Equivalence-ground-truth proxy.** Behavioral fingerprint on N_FP = 32 fixed inputs is a proxy for true algebraic equivalence. False NON-equivalence (algebraically-equivalent expressions with different fingerprints) can occur from input-set inadequacy or from int64 overflow (RC-1). Tight-band mitigates overflow but doesn't address sample inadequacy. A larger N_FP (say 128) and adversarially-chosen inputs would harden the ground truth; not in scope for this closeout.

**2. Single anchor per class.** The bank uses ONE anchor (the first member encountered) per class. If the anchor's signature is non-representative of its class, accuracy is depressed. A multi-anchor or class-mean bank would likely improve absolute accuracy for both rules, but the *relative comparison* (dual vs sign-only) should be insensitive to this.

**3. Absolute accuracy is low (15–24%).** Both rules are far from the ~100% ceiling. This means the signatures don't capture algebraic equivalence well at this scale (16 signature inputs, depth-4 random expressions). The result is a *relative* statement ("dual is no better than sign-only"), not an absolute statement about routing quality.

**4. Pre-commit hygiene.** The gate (≥ 2 pp absolute) was set in this same session, before running. RC-1 was caught BEFORE finalizing. No iteration on K_TOTAL / N_FP / MIN_PER_CLASS / VERDICT_GAP_PP after seeing the result.

## What this confirms — and what it does NOT

**Confirmed:** The R1 dual-threshold rule does not provide a measurable routing-accuracy advantage over sign-only under behavioral-equivalence ground truth at clean (no-overflow) configurations.

**NOT confirmed:**
- A different signature rule could still beat sign-only on routing — F-G5's verdict is rule-specific, not class-of-rules.
- Vision claim 3 broadly is unaffected. R1 was one operationalization; F-G5's failure refines the falsification of *that specific operationalization*.

## Status

CLOSED. R1 methodically falsified across 5 substantive axes (was 4). Closes TD-8.

## Cross-references

- Original 4-axis closeout: `journal/r1_falsify_closeout.md`
- Original synthesize: `journal/r1_falsify_synthesize.md`
- F-G5 bench source: `gesh/bench/expr_routing_r1_f_g5.c`
- Project-tracking: `docs/TECHNICAL_DEBT.md` TD-8 (now removed)
