# Phase β red-team — VALIDATED 3/3 collapses to MIXED 1/3 under alternative normalization

This is the red-team on commit `299a421` (Phase β VALIDATED 3/3). The
pre-registered verdict depended on a normalization choice (`d̂/Dmax`)
that I picked instrumentally in the FROZEN spec but did not defend
rigorously. Under the alternative (and arguably more natural)
normalization (`d̂/D_ambient`), **two of three P-rules fail or
reverse**, leaving only P3 as a robust finding.

This is the **second time in this Phase α/β arc that a normalization
choice was the load-bearing methodology gap.** Phase α had the same
shape: pre-registered absolute-d̂ rule passed; ambient-D-normalized
analysis reversed the verdict. I encoded the lesson in memory
(`feedback_spot_check_before_verdict.md` last bullet) AND THEN
walked into a sibling version of the same trap in Phase β by
choosing Dmax-normalization without examining the alternative.

## The normalization problem

Macocco's d̂ is an **intrinsic dimension** — a count of effective
manifold dimensions, with units of "number of cells worth of
variation." For the same data, d̂ is comparable across metrics
because both measure the same thing (effective dim count).

What ISN'T directly comparable across reps is the *absolute*
distance range:
- Hamming on D=128 cells: max distance = 128
- L1 on D=128 trits: max distance = 2D = 256
- Hamming on D=203 bits: max distance = 203

The pre-reg FROZE `d̂/Dmax` as the comparison. But `Dmax` is a
**distance**, not a **dimension count**. Dividing d̂ by Dmax mixes
units. The alternative — `d̂/D_ambient` (intrinsic dim divided by
ambient cell count) — keeps the units consistent: both numerator
and denominator are in cells.

## The numbers under both normalizations

Same bootstrap output, two ways:

| representation | d̂ (B=200 mean) | d̂/Dmax | d̂/D_ambient |
|---|---|---|---|
| substrate_L1     |  98.3 | 0.384 [0.370, 0.397] | **0.768** [0.741, 0.795] |
| B0_Hamming_sub   | 104.7 | 0.818 [0.783, 0.850] | 0.818 [0.783, 0.850] |
| B2_sign (random) | 134.6 | 0.663 [0.661, 0.664] | 0.663 [0.661, 0.664] |
| B3_sign          | 134.6 | 0.663 [0.661, 0.665] | 0.663 [0.661, 0.665] |
| B4_pca           | 141.4 | 0.697 [0.695, 0.699] | 0.697 [0.695, 0.699] |
| B5_scrambled_sub | 106.4 | 0.416 [0.399, 0.430] | 0.831 [0.799, 0.860] |

Note: B0, B2, B3, B4 are unchanged (Dmax = D_ambient for Hamming).
Only substrate_L1 and B5_scrambled differ between normalizations,
because L1 max distance = 2 × ambient.

### P1 under both normalizations

> P1: L1-substrate < B0_Hamming_sub (does the L1 metric reveal
> structure Hamming hid?)

- **Dmax:** 0.384 vs 0.818 → gap +0.434, CIs disjoint → **PASS**
- **D_ambient:** 0.768 vs 0.818 → gap +0.050, CIs *overlap*
  (substrate hi 0.795 > B0 lo 0.783) → **FAIL on CI rule**

The 43-percentage-point "gap" under Dmax shrinks to 5 percentage
points under D_ambient. The real story: the L1 metric does change
d̂ on the same data — from 103.2 (Hamming) to 98.3 (L1), a ~5%
reduction. Real but modest. The "dramatic" claim was the
normalization, not the data.

### P2 under both normalizations

> P2: L1-substrate ≤ B4_pca (does substrate beat structured binary
> at equal capacity?)

- **Dmax:** 0.384 vs 0.697 → gap +0.313, CIs disjoint → **PASS**
- **D_ambient:** 0.768 vs 0.697 → gap **−0.071** (substrate
  HIGHER), CIs overlap (B4 hi 0.699 < substrate lo 0.741, wait —
  let me re-check) → substrate's lo (0.741) > B4's hi (0.699) →
  CIs are disjoint **in the wrong direction**.

The P2 PASS reverses entirely. Under D_ambient, substrate
*fills more of its cells* than structured binary at equal capacity.
This is the same Phase α remediation finding (commit `c10bd39`)
restated. The substrate's larger Dmax under L1 was making it
look more "compressed" via the normalization.

### P3 under both normalizations

> P3: L1-substrate < B5_scrambled (is centrality of 0
> load-bearing?)

- **Dmax:** 0.384 vs 0.416 → gap +0.032, CIs disjoint → **PASS**
- **D_ambient:** 0.768 vs 0.831 → gap +0.063, CIs disjoint
  (substrate hi 0.795 < B5 lo 0.799) → **PASS**

P3 survives both normalizations because the comparison is
**within the same metric family** (both substrate_L1 and B5 use
L1-style distances on 128 trits with Dmax=256). The Dmax-vs-Dambient
ambiguity doesn't apply here; the comparison is apples-to-apples.

## Honest revised verdict

| P-rule | Dmax normalization | D_ambient normalization | Robust? |
|---|---|---|---|
| P1 | PASS (+43pp) | FAIL (CI overlap) | **NO** |
| P2 | PASS (+31pp) | FAIL (reverses) | **NO** |
| P3 | PASS (+3pp)  | PASS (+6pp) | **YES** |

**Verdict robustly clear: 1/3 P-rules → MIXED.**

Under the FROZEN pre-reg rule, Phase β was VALIDATED 3/3. Under a
defensible alternative rule, Phase β is MIXED 1/3. The pre-reg
should have specified the normalization with first-principles
justification; instead it took an instrumental choice and the
result tracks that choice.

## What's actually load-bearing in the data

Three findings ARE robust across red-team:

**1. The substrate has fewer absolute effective dimensions than
binary baselines.** d̂ ≈ 98-105 (substrate under either metric)
vs 134-141 (binary). The substrate's smaller D=128 ambient ceiling
is part of this; you can't have more effective dims than ambient
cells. Whether "fewer absolute dims" counts as a substrate-
distinctive property depends on what you're optimizing.

**2. The L1 metric on the same substrate signatures gives slightly
lower d̂ than categorical Hamming.** 98.3 vs 103.2, a ~5% reduction.
Real but small. The metric does reveal *some* additional compactness
(the path-graph structure does some work), but it's not a dramatic
effect. The pre-reg's framing of "metric choice does most of the
work" was an artifact of normalization.

**3. The centrality of 0 in the cell-graph IS load-bearing.** P3
passes under both normalizations. d̂/D_ambient for substrate_L1
(0-as-center) = 0.768; same data, +1-as-center metric = 0.831.
The 6-percentage-point reduction is statistically real (CI-disjoint
under both normalizations) and confirms the user's "0 is geometric,
not arbitrary" claim. **This is the cleanest survival from Phase β.**

## What it means for the vision claim

The vision (refined in `project_vision.md`): "base-3 IS the graph"
— trits live on a 3-vertex path with 0 as the natural center.

What Phase β + red-team actually showed:

- **The path-graph structure does some real work** (P1 honest +5pp
  effect of L1 over Hamming).
- **0-as-center is genuinely special** (P3 +6pp effect of 0-center
  over +1-center, both normalizations).
- **Substrate under L1 does NOT clearly beat structured binary** at
  equal capacity (P2 reverses under D_ambient).

Cleaned-up claim: **base-3 with path-graph metric anchored at 0
captures geometric structure that categorical-Hamming-on-ternary
does not.** Modest effect size (5-6pp). The third state has a
measurable geometric role, the path-graph structure has a smaller
measurable effect, and the absolute "substrate beats binary"
direction depends on a normalization choice that doesn't have a
single right answer.

## Memory update

The lesson encoded twice already (`feedback_spot_check_before_verdict.md`
last bullet — comparisons across mismatched ambient spaces need
normalization rules pre-registered with justification) needs to be
sharper:

- It applies not only when ambient spaces differ in cell count, but
  also when the **metric's distance range** differs (Hamming vs L1).
- When choosing a normalization for the FROZEN spec, run the
  alternative normalization too and check both. If they disagree,
  the result is normalization-sensitive and should be reported as
  such, not buried.
- Pre-register **multiple normalizations** when more than one is
  defensible, and require pass under all of them for a full PASS.

## Discipline log

This is the **13th caught misalignment** and the *same shape* as
catches #10 and #11. The pattern is so consistent it's almost
embarrassing: pre-register a comparison rule → test → claim victory
→ user (or red-team) catches that an alternative defensible rule
gives a different verdict → revise.

The substrate-distinctive claim has now been measured 4 times:
- Phase α original (commit 309fed0): "VALIDATED 2/3" under absolute d̂.
- Phase α red-team + remediation (e569f79, c10bd39): "MIXED → reversed"
  under normalized d̂/D_ambient.
- Phase β under L1 + Dmax-normalization (299a421): "VALIDATED 3/3."
- Phase β red-team (this journal): "MIXED 1/3" under L1 + D_ambient-
  normalization.

Across these, only P3 (centrality of 0) is robust. The substrate's
geometric distinctness is real but smaller than any single verdict
has claimed.

## Status of downstream applications

Phase β synthesis (commit 299a421) said: "Round 2 spline operations
have justification restored." Under the red-team, that's an
overclaim too. Honest reading:

- **Soft routing under L1**: the manifold-structure premise is
  *weakly* supported (P1 honest gap is 5pp, not 43pp). Worth
  exploring but with realistic expectations.
- **Bank interpolation**: P3 confirms 0-as-center is real; geometric
  interpolation through 0 IS meaningful. Could work.
- **Nyström compression**: substrate's d̂/D_ambient at 0.77 is HIGHER
  than binary at 0.70. Landmark-sparse coverage isn't supported.
  *Compression target is NOT justified.*

The downstream-impact section of Phase β synthesis is also revised.

## Sign-off

The substrate has measurable geometric distinctness, but smaller
than I claimed. The "0 as center" property is real (P3 robust).
The "L1 metric reveals dramatic hidden structure" framing was a
normalization artifact. The "substrate beats structured binary at
equal capacity" claim doesn't survive normalization-sensitivity
analysis.

What we know: there is a real signal in the path-graph structure,
load-bearing in 0's centrality, modest in size. The vision claim
has a measurement-grounded foundation, but a narrower one than
either Phase α or pre-redteam Phase β suggested.
