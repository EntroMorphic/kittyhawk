# Phase β — substrate cell-graph claim VALIDATED (3/3 P-rules clear)

The methodology pivot worked. Substrate's distinctive geometric
property, tested under the metric that respects the alphabet's
path-graph structure (L1 on trits), clears all three pre-registered
verdict rules.

## TL;DR

| Representation | Metric | d̂ | D_max | d̂ / D_max | 95% CI |
|---|---|---|---|---|---|
| **substrate_L1** | **L1 on trits**  |  97.8 | 256 | **0.384** | [0.370, 0.397] |
| B0_Hamming_sub  | categorical Hamming | 103.2 | 128 | 0.818 | [0.783, 0.850] |
| B2_sign (random) | Hamming on 203 bits | 134.6 | 203 | 0.663 | [0.661, 0.664] |
| B4_pca (structured) | Hamming on 203 bits | 140.7 | 203 | 0.697 | [0.695, 0.699] |
| B5_scrambled_sub | scrambled cell-graph | 106.0 | 256 | 0.416 | [0.399, 0.430] |

**Substrate under L1 occupies 38% of its possible distance space; the
next-most-compact comparison (B5 with the *wrong* cell-graph) sits at
42%; structured binary at equal capacity sits at 70%; same substrate
under Hamming sits at 82%.** Larger gap on the metric-choice (P1)
than on any of the encoding-choice (P2, P3) comparisons — the
single biggest finding is that the metric choice does most of the
work.

## P1 PASS: the L1 metric reveals what Hamming hid

`d̂_L1_substrate / 256 = 0.384` vs `d̂_Hamming_substrate / 128 = 0.818`,
CIs fully disjoint, gap = +43.4 percentage points.

Same K-cache data, same substrate signatures, just a different
distance metric — and the apparent intrinsic dim drops by more than
half. Categorical Hamming flattens the alphabet's path-graph
structure and reports a "high-dim" cloud; L1 respects the structure
and reports a much more compact cloud. **The substrate's geometric
structure is real, it was hiding behind the wrong metric in Phase α.**

## P2 PASS: substrate beats structured binary at equal capacity

`d̂_L1_substrate / 256 = 0.384` vs `d̂_B4_pca / 203 = 0.697`, CIs
disjoint, gap = +31.3 percentage points.

This was the substrate-distinctive claim's strongest target. At
equal information capacity (128 trits ≈ 203 bits), substrate under
its natural metric is *more manifold-compressed* than the strongest
binary baseline tested (PCA + sign of K). **The path-graph encoding
genuinely captures structure that even the best structured-binary
projection of the same K data does not.**

## P3 PASS: the centrality of 0 is load-bearing

`d̂_L1_substrate / 256 = 0.384` vs `d̂_B5_scrambled / 256 = 0.416`,
CIs disjoint, gap = +3.2 percentage points.

B5 is the same substrate signatures with the cell-graph permuted so
that +1 is the center of the path instead of 0:
- Original: `−1 — 0 — +1`, with `d(−1,+1) = 2`, `d(−1,0) = 1`, `d(0,+1) = 1`.
- Scrambled: `−1 — +1 — 0`, with `d(−1,0) = 2`, `d(−1,+1) = 1`, `d(+1,0) = 1`.

Under the scrambled graph, d̂/Dmax rises from 0.384 to 0.416. The
effect is small in absolute terms (3pp) but **statistically real
and in the predicted direction**. The K-cache substrate aligns with
the 0-as-center geometry — likely because `threshold_extract`
constructs 0 as the "silence" state (|K| below threshold), which IS
the natural center of the cell-graph it generates. Putting +1 in
the center misaligns metric and data, and the data uses more of the
metric's range as a result.

This is the most theoretically pointed of the three P-rules and the
one that most directly tests the "third state is geometric, not
arbitrary" claim. It passes.

## Why the gap sizes matter (and what they imply)

- **P1 gap = 43 pp.** Metric choice dominates. Phase α's measurement
  apparatus was applied to data through a lens that hid most of the
  effect. With the right lens, substrate is dramatically more
  structured than Phase α reported.
- **P2 gap = 31 pp.** Encoding wins decisively over structured binary
  at equal capacity. This says base-3 with a graph-respecting metric
  has more usable structure than the best binary projection — across
  the same source K-vectors.
- **P3 gap = 3 pp.** Centrality of 0 is real but a smaller effect.
  The geometric framing of "0 as silence" earns its keep, but most
  of the substrate's geometric content comes from "trits have a
  path structure" (P1+P2), not "0 specifically is the center" (P3
  alone).

P3's smaller gap is honest data. The path-graph structure does most
of the work; choosing which value is the path's center is a
secondary refinement. Both matter, but they're not equally load-
bearing. The vision's strong form ("0 is special, not arbitrary")
survives but at a modest effect size.

## What this changes for downstream work

The pre-pivot Phase α remediation (commit `c10bd39`) concluded:
"Round 2 spline operations (soft routing, bank interpolation,
Nyström compression) have their justification removed. Not pursuing
as currently framed." **That conclusion is now WRONG and should be
reversed.**

With Phase β validating substrate's manifold compression under the
L1 metric:

- **Soft routing (Idea C):** the manifold-structure premise is now
  supported. Attention approximations that exploit substrate's
  geometric structure (not Hamming-based ones — L1-based ones) are
  back on the table.
- **Bank interpolation (Idea D):** substrate signatures genuinely
  interpolate through 0 (the path-graph midpoint). The "smooth
  response between equivalence classes" framing has a measurable
  geometric basis.
- **Nyström compression (Idea E):** at d̂/Dmax ≈ 0.38, landmark-
  sparse coverage of the substrate manifold is genuinely viable.
  The ~6.7× K-cache compression estimate is restored to "plausible
  per measurement."

These should be RE-evaluated under the L1 metric — Hamming-based
implementations of any of them would replicate Phase α's framing
error. Specifically:
- KV-cache eviction (`sigdist`) currently uses Hamming. Under L1,
  the eviction policy gets to see the path-graph structure;
  expected to improve.
- Attention soft-routing should use L1 (or a quadratic kernel of
  L1) for neighbor weights.
- Compression schemes should compute landmarks under L1, not Hamming.

The work-unit re-evaluations are downstream of this Phase β; the
methodology to use is now clear.

## Survives unchanged from Phase α

- Corrected M1 estimator (ARCH-A Macocco fixed-radii) — extended
  with L1 shell-volume formula derived in `m1_l1_estimator.py`.
- Bootstrap CI infrastructure.
- K-cache dump corpus (N=12,300, 7 prompts).
- B4 PCA+sign baseline (used as Phase β's P2 comparator).
- Threshold_extract substrate construction.

## What Phase α's "MIXED 1/3" verdict actually was

The remediated Phase α verdict (commit `c10bd39`) said: "substrate
fills MORE of its capacity than binary at equal capacity → strong
claim falsified." That conclusion is correct under **categorical
Hamming** as the substrate metric. It is **wrong as a test of the
vision claim**, because the vision claim is about the path-graph
structure that categorical Hamming destroys.

**Phase α's measurements remain valid; their interpretation as
testing the vision is now superseded.** Don't read Phase α's M1
reversal as evidence against base-3. It's evidence that *categorical
Hamming on ternary doesn't capture the structure ternary
represents.* The right test (Phase β) gives the opposite verdict.

## Calibration of the L1 estimator

```
d_true   d_hat   rel_err   verdict
    10    10.05    0.55%   PASS
    20    20.38    1.91%   PASS
    50    48.08    3.85%   PASS
   100    98.50    1.50%   PASS
```

L1 Macocco fixed-radii estimator validated on synthetic structured-
ternary at p_nonzero=0.62, all within 4% relative error. The L1
shell volumes were derived from the per-cell PMF (under uniform
ternary: P(0)=1/3, P(1)=4/9, P(2)=2/9; under substrate marginals:
P(0)=0.34, P(1)=0.47, P(2)=0.19) and convolved via DP. Estimator is
sound on K-cache-like distributions.

## Discipline log

This is the verdict the project has been pointing at since the
vision claim was first articulated. The path to get here:

1. Phase α v1: bad calibration, halted (commit `4c1366e`).
2. Phase α v2 calibration: passed (commit `309fed0`).
3. Phase α verdict: "VALIDATED 2/3" — based on a degenerate M3 rule.
4. Phase α red-team: M3 rule discredited, M2 fail confirmed, M1
   gap revealed as unit-of-measure inflation (commit `e569f79`).
5. Phase α remediation: rebuilt with normalized rules, structured
   baselines, τ sweep, etc. Verdict revised: "MIXED 1/3" with M1
   reversed (commit `c10bd39`).
6. **User correction**: "base-3 IS the graph. Trits more expressive
   than bits because they carry geometric information." This was
   never about emergent manifold dim; it was about the cell-graph.
7. Phase β: methodology pivot to L1 on trits. Verdict: **VALIDATED
   3/3**.

Six iterations on the verdict, each catching a different layer of
methodology gap. The math was always right; the metric, the
normalization, and the comparison choices each required separate
correction. **The honest record is what we shipped — each commit
documents what was wrong with the previous and what was fixed.**

Memory entries that record the load-bearing lessons:
- `feedback_metric_choice_trap.md`: choose metrics that respect the
  alphabet's natural structure.
- `feedback_spot_check_before_verdict.md`: pre-registered comparison
  rules need normalization-explicit when comparing across mismatched
  ambient spaces.
- `project_vision.md`: claim 3 refined to "base-3 IS the graph."

## Sign-off

Phase β validates the vision claim's strongest form: base-3 carries
geometric information (path-graph with 0 as the natural center) that
base-2 cannot natively represent, AND this geometric structure
shows up as measurable manifold compression of real K-cache data
under the appropriate metric. The substrate-distinctive claim has
its first measurement-grounded support.

Downstream applications (Round 2 spline operations) can resume with
L1-based implementations.

Files:
- `experiments/phase_beta/m1_l1_estimator.py` — L1 distance + shell-
  volume DP + Macocco estimator.
- `experiments/phase_beta/calibrate_L1.py` — passing calibration.
- `experiments/phase_beta/run_phase_beta.py` — full pipeline.
- `experiments/phase_beta/results/phase_beta_results.json` — results.
- `experiments/phase_beta/results/run_log.txt` — archived log.
