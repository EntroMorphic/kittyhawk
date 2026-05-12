# Phase α REMEDIATED — substrate-distinctive claim downgrades from VALIDATED → MIXED

This is the comprehensive remediation of every red-team finding in
`td27_phase_alpha_redteam_2026-05-12.md` (commit `e569f79`). All 8
methodology gaps were addressed in a single rebuilt run.

## TL;DR — the M1 claim REVERSES under remediation

The original "VALIDATED 2/3" was based on a 41% absolute d̂ gap that
the red-team flagged as unit-of-measure inflation. Under the stricter
remediated rule (absolute gap ≥ 20% **AND** normalized d̂/D gap ≥ 5pp
**AND** disjoint CIs **AND** vs structured baseline), M1 **fails on
the normalized direction at every checkpoint**:

| representation | d̂ (mean) | d̂/D (mean) | 95% CI on d̂/D |
|---|---|---|---|
| substrate    | **104.73** | **0.818** | [0.783, 0.850] |
| B2 (random sign, 203b) | 134.56 | 0.663 | [0.661, 0.664] |
| B3 (random sign, 203b) | 134.56 | 0.663 | [0.661, 0.665] |
| B4 (PCA+sign, 203b)    | 141.43 | 0.697 | [0.695, 0.699] |

**Substrate occupies MORE of its (smaller) capacity than the binary
baselines occupy of their (larger) capacity** — by 12-15 percentage
points, CI-disjoint in the wrong direction.

The "fewer absolute effective dimensions" finding from the original
verdict was a unit-of-measure artifact at mismatched ambient D. Once
you compare like-to-like (fraction of capacity used), substrate is
the LEAST manifold-compressed of the four representations tested.

**Final remediated verdict: MIXED (1/3 measures clear).** Only the
new M3 Wasserstein test (a topological-shape diagnostic, not a
dimensionality diagnostic) clears. M1 and M2 both fail.

## Remediations applied (per red-team gap)

### [#1] M3 verdict rule replaced

Old rule: "substrate longest_bar > 2× B3.bar_p95." Degenerate at
integer-Hamming pooled scale (B3.p95 = 0, so 2× = 0, any positive
value passes trivially).

New rule: **W₁ Wasserstein distance between bar-length distributions**.
Substrate's bar distribution differs from B2's by `W(substrate, B2)
= 0.007`. The null baseline `W(B3, B2) = 0.003` (two random
projections of the same data) — substrate is `2.3×` the null, just
over the 2× threshold. **PASS.** But this is a *shape* result (bar
distributions differ), not a *dimensionality* result.

### [#2] M1 verdict rule strengthened

Old rule: "absolute relative gap ≥ 20%, CIs disjoint."
New rule: **absolute gap ≥ 20% AND normalized d̂/D gap ≥ 5pp AND CIs
disjoint on BOTH metrics AND vs BOTH B2 (random) and B4 (structured)
baselines.**

Result:

| comparison | abs_gap | norm_pp | abs_pass | norm_pass |
|---|---|---|---|---|
| vs B2 (random) | +22.2% | **−0.155** | True | **False** |
| vs B4 (structured) | +26.0% | **−0.121** | True | **False** |

Normalized direction is reversed (substrate is HIGHER d̂/D, not
lower). **M1 FAIL** under the new rule.

### [#3] τ sensitivity sweep

The original used τ=5000 hard-coded. The sweep:

| τ | nonzero | d̂_sub | abs_gap | norm_pp_gap |
|---|---|---|---|---|
|  2,000 | 81.7% | 110.40 | +18.0% | **−0.199** |
|  5,000 | 61.2% | 103.24 | +23.3% | **−0.143** |
| 10,000 | 36.2% |  60.39 | +55.1% | +0.191 |
| 20,000 |  9.3% |  11.01 | +91.8% | +0.577 |

The absolute gap monotonically grows with τ — but only because
substrate becomes increasingly sparse-and-trivial. At τ=20000,
substrate is 91% zeros; calling its d̂≈11 "lower intrinsic dim than
B2's 134" is meaningless — it's just a degenerate signature.

The **normalized** gap stays in the wrong direction (substrate
higher d̂/D) for sensible τ values (≤5000, where nonzero rate is
≥61%). Substrate-distinctive claim is **τ-fragile**: only crosses
into "favorable" normalized direction when substrate is destroyed.

### [#4] B4 structured baseline added

B4 = sign of top-203 PCA projection of K. This is the substrate-
relevant comparison: at equal information capacity, with a
*structured* binary code (not random Gaussian hash). B4 outperforms
B2 (random) on every metric tested, and substrate fails to beat B4
on the normalized M1 metric just as it fails vs B2.

### [#5] Bootstrap CIs for M1, M2, M3

200 resamples × N_sub=500 from the full N=12300 corpus. Every metric
now has 95% CIs. The d̂/D CIs are tight (B2: [0.661, 0.664]) — the
direction-of-effect on the wrong-direction M1 is highly significant.

### [#6] (t1, t2) sensitivity

Three quantile-pair choices:

| qpair | substrate | B2 | B3 | B4 |
|---|---|---|---|---|
| (3%, 7%)  |  92.4 | 130.4 | 130.3 | 137.5 |
| (5%, 15%) | 103.2 | 134.6 | 134.6 | 140.7 |
| (10%, 30%) | 114.8 | 139.2 | 139.1 | 143.7 |

Direction of effect is **invariant to (t1, t2) choice**. Substrate is
absolute-lower at every choice; the gap magnitude varies modestly
(~25%). Estimator is robust.

### [#7] Calibration on K-cache-matched synthetic

The original calibration was on uniform-random ternary
(p_nonzero=67%). K-cache has p_nonzero ≈ 62% (or 81% at lower τ).
Re-running calibration with **structured ternary** (p_nonzero=0.62,
matching K-cache):

```
d_true  d_hat   err
    10   9.91  0.89%
    20  19.95  0.24%
    50  49.74  0.52%
   100  99.19  0.81%
```

All within 1%. **Estimator is sound on K-cache-like distributions.**
This rules out "the off-distribution use invalidates the estimate"
as an explanation for the wrong-direction normalized result.

### [#8] More diverse prompts

Original corpus: ~2 prompts, N=2400. Remediated corpus: 7 prompts
(`dump`, `multitoken`, `p1..p5`) of varying lengths (1, 6, 4, 4, 11,
8 positions), **N=12300**. Direction-of-effect is unchanged from
N=2400 — substrate fills *more* of its capacity, not less. **Not a
small-sample artifact.**

## What the original "41% gap" was actually measuring

The original `td27_phase_alpha_synthesis_2026-05-12.md` reported
substrate d̂=78.6 vs B2 d̂=134.1 → "41% relative gap." This was
arithmetically correct under the FROZEN absolute-d̂ rule, but
conceptually a **unit-of-measure inflation**:

- Substrate's ambient ceiling is 128 (HEAD_DIM trits).
- B2's ambient ceiling is 203 (FROZEN equal-bits design).
- d̂ scales with the ambient ceiling.

The proper "equal-bits" comparison is **what fraction of the ambient
ceiling does each representation occupy**. That's d̂/D. Under that
metric:

- substrate: 0.617 (original, N=2400) → 0.818 (remediated, N=12300)
- B2: 0.659 → 0.663
- B4 (added): 0.697

At the original N=2400 the substrate was *near-equal* to B2 in d̂/D
(0.617 vs 0.659 — substrate even slightly *lower*, the only
defensible reading of "manifold-compressed"). At the remediated
N=12300, substrate jumps to 0.818 d̂/D — substantially HIGHER than
both B2 and B4.

The N=2400 finding was a small-sample bias toward substrate. At
larger N, the estimator pulls substrate's d̂/D up toward 0.82,
revealing that substrate is **as dimensional or MORE dimensional**
than equal-bit binary codes at the same K data.

## Honest interpretation

The substrate-claim's strongest form — "base-3 carries information
base-2 collapses" — predicts substrate would occupy a *lower*-
dimensional manifold than equal-bits binary. **The remediated
measurement contradicts this.** Substrate occupies a HIGHER fraction
of its ambient capacity than either random or PCA-structured binary.

What the substrate DOES have that survives remediation:

1. **Different absolute d̂.** Substrate's 128-trit ambient ceiling
   means it caps at d̂ ≈ 128, while binary caps higher (203). For
   downstream applications where total compute or memory scales with
   d̂ in some absolute way, substrate's smaller cap is a feature.
2. **Different topological signature.** M3 Wasserstein test passes:
   substrate's bar-length distribution differs from random binary
   projections by more than baseline projection variability. This
   says the substrate metric has a *distinctive* clustering shape,
   not a *better* one.
3. **Calibration-robust estimator.** v2 works correctly on the
   K-cache distribution. The reversal isn't a measurement bug.

What it does NOT have:

- "Lower intrinsic dimensionality" — the central claim of the strong
  vision. Falsified at N=12300 across τ sweep and B4 comparison.
- "More reciprocal kNN structure" — already known from the red-team
  pass; substrate is more hub-dominated.
- "More topologically clustered" — at per-layer scale, substrate has
  shorter persistence bars than B2.

## What this means for Round 2 (downstream spline operations)

From `td27_spline_explorations_2026-05-12.md`, the three operations
predicated on substrate manifold structure:

- **Soft routing (Idea C):** the manifold-compression premise is
  contradicted by the d̂/D measurement. Soft routing's expected win
  was modest already (~1-2% total cost); now its theoretical
  justification is missing. **Move from "weakly supported" to
  "speculative."**
- **Bank interpolation (Idea D):** depends on substrate manifold
  preserving semantic similarity. With substrate d̂/D higher than B4,
  no reason to expect interpolation in substrate space is more
  semantically meaningful than interpolation in PCA-binary space.
  **Move from "speculative" to "not justified."**
- **Nyström compression (Idea E):** ~6.7× compression target assumed
  substrate had landmark-sparse manifold structure. Substrate's d̂/D
  ≈ 0.82 means landmarks would need to be 82% of N to cover the
  manifold — not the sparse coverage Nyström assumes. **The compression target is not supported.**

The Round 2 operations were always downstream of Phase α. The
remediated Phase α says: **do not pursue them as currently framed.**
If pursued, the motivation must be re-derived from a different
substrate property than manifold-compression.

## What the substrate IS, after remediation

The substrate is:
- A **ternary representation with 128 cells** (smaller ambient than
  equal-bits binary).
- Holds **fewer absolute effective dimensions** (~104 vs 134 for B2)
  but **as a higher fraction of its ambient space**.
- Has **distinctive (different, not better) topology** in persistence.
- Has **more hub-dominated and asymmetric kNN graph** than binary —
  not a "smoother" manifold.
- Useful as a **low-resolution proxy for K** at 1/16th the bits of
  fp16, with bounded fidelity loss in downstream inference (covered
  by other journals: Phase A, work-unit 10).

This is a **useful primitive with characterizable trade-offs**, not
a representation that "carries information binary collapses." The
strong vision claim is **falsified at the M1 level under proper
controls.**

## Discipline log

This is the **11th caught overclaim** of the session sequence. The
sequence: original commit → user-prompted red-team → remediation →
self-inflicted verdict downgrade. Each pass tightened the rules and
revealed a different problem with the previous verdict.

The pre-registration discipline worked at the calibration-stop level
(v1 calibration failed, implementation halted, math was re-derived).
It also worked at the verdict-rule level — but only because the user
forced a red-team after the initial verdict.

The lesson encoded in memory (`feedback_spot_check_before_verdict.md`):
- Originally: spot-check the cases the metric scores worst before
  declaring.
- Updated by red-team #1: pre-registered verdict rules themselves
  need red-teaming (hidden edge cases).
- **Updated by this remediation: comparison metrics across mismatched
  ambient spaces need normalization rules pre-registered too.** The
  "absolute d̂ gap" rule was a hidden trap built into FROZEN v2.

## What stays valid from prior commits

The corrected M1 estimator (`m1_estimator_v2.py`, both ARCH-A and
ARCH-B) is sound; structured-marginal calibration confirms it. The
v1 calibration-fail and the v2 calibration-pass are both unchanged.

What changes: the **interpretation** of the K-cache results. The
"VALIDATED 2/3" label from commit `309fed0` and the "MIXED 1/3"
label from commit `e569f79` are both superseded by this commit's
**MIXED 1/3 (with M1 PASS reversed under stricter rules)**.

## Files committed

- `experiments/phase_alpha/run_phase_alpha_v2.py` — remediated
  pipeline with all 8 fixes.
- `experiments/phase_alpha/results/phase_alpha_v2_results.json` —
  full numeric results.
- `experiments/phase_alpha/results/v2_run_log.txt` — archived log.
- `data/c_dump_v2/` — new dump corpus (5 diverse prompts; ~150MB).
- `experiments/phase_alpha/load_k_signatures.py` — updated to handle
  multi-directory dump loading and new prompt prefixes.

The original `run_phase_alpha.py`, `phase_alpha_results.json`, and
`run_log.txt` remain as the "uncorrected" reference for direct
comparison.

## Sign-off

Phase α now has three deliverables: the corrected estimator (sound),
the K-cache measurement (well-controlled), and the honest verdict
(MIXED with the M1-fail reversal documented). Strong-form
substrate-distinctive claim is falsified. The substrate retains
characterizable utility as a quantization primitive but does not
demonstrate the manifold-compression property the vision claim
predicted.
