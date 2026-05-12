# Methodology pivot: Phase α tested the wrong metric — supersedes verdict, salvages infrastructure

This journal supersedes the interpretive framing of all prior Phase α
journals (`td27_geometric_prereg_v2`, `td27_phase_alpha_synthesis`,
`td27_phase_alpha_redteam`, `td27_phase_alpha_remediation`). The
**math is intact**; the **measurement infrastructure is intact**; the
**verdicts I drew about the substrate-distinctive claim are not** —
they were verdicts about a different claim than the one the vision
makes.

## The framing error

The vision claim (project_vision memory, refined 2026-05-12):
**Base-3 IS the graph.** A binary cell {−1, +1} is two endpoints with
nothing between. A ternary cell {−1, 0, +1} is a path graph of length
2 with **0 as the natural center**. Substrate signatures live in the
D-fold product of this path graph — a structured discrete object
with native adjacency, native distance (one-step transitions),
native interpolation through 0.

What I tested (Phase α, all four prior journals): whether the K-cloud
in substrate space has lower emergent intrinsic dimensionality than
binary baselines, using **categorical Hamming** as the substrate
distance.

Under categorical Hamming, `d(a, b) = count of cells that disagree`.
The three cell values {−1, 0, +1} are treated as **arbitrary labels**
— equivalent to {red, green, blue}. The path-graph structure is
invisible to the metric. For binary {−1, +1}, categorical Hamming IS
the natural path-graph distance (cells differ by 0 or 1; no richer
structure exists). So my "substrate-under-Hamming vs B2-under-
Hamming" comparison was comparing two representations that BOTH had
been stripped of any alphabet-graph structure. The substrate-
distinctive property — that the alphabet has graph structure with 0
as the natural center — was destroyed by the metric choice before
the comparison started.

The right substrate distance is **L1 on trits**:
```
d(a, b) = Σᵢ |aᵢ − bᵢ|
```
Under L1: d(+1, −1) = 2, d(+1, 0) = 1, d(0, −1) = 1. The metric
encodes that the zero is between — passing through 0 costs less than
crossing it. For binary, L1 = Hamming (no difference), so the
asymmetry is captured exactly where it should be: substrate gets
richer geometry, binary doesn't pretend to have what it lacks.

## What survives from the prior work

The math, calibration, infrastructure, and dataset all transfer to
the corrected framing:

| asset | reusable? | how |
|---|---|---|
| Corrected M1 estimator (ARCH-A Macocco, ARCH-B TwoNN) | YES | Both work on any pairwise distance matrix — change input distances, math is unchanged. |
| Structured-marginal calibration (p_nonzero=0.62) | YES — but must re-validate | The estimator was calibrated for Hamming-shell-volumes; under L1 the shell volumes are different. Need new closed form for L1-ternary shell volumes (DP/FFT over single-cell PMF). |
| Bootstrap CI infrastructure | YES | Generic over the distance computation. |
| (t1, t2) sensitivity, τ sweep, prompt diversity | YES | Operational scaffolding, metric-agnostic. |
| B4 PCA+sign structured-binary baseline | YES | Comparison target for L1-substrate, since B4's native metric is also Hamming-on-binary = L1-on-binary. |
| M2 kNN reciprocity / Gini | YES | Distance-agnostic graph metric. |
| M3 Wasserstein on persistence bars | YES | Distance-agnostic shape metric. |
| K-cache dump corpus (N=12300, 7 prompts) | YES | Raw data. Re-derive signatures + L1 distances. |

## What gets re-derived

- **Shell-volume formula under L1-ternary.** For two i.i.d. uniform
  ternary cells, P(|aᵢ − bᵢ| = k):
  - P(k=0) = 1/3
  - P(k=1) = 4/9
  - P(k=2) = 2/9
  Pairwise L1 distance over D cells is the sum of D i.i.d. samples
  from this distribution. The CDF (cumulative shell volume) is the
  D-fold convolution — computable via DP or FFT.
  Under non-uniform marginals (substrate has 62% nonzero — sign roughly
  balanced, so ~31% −1, ~38% 0, ~31% +1), the per-cell PMF shifts
  slightly: more 0s in the marginals → more 1-step distances, fewer
  2-step distances. Re-derive with empirical marginals.

- **Macocco fixed-radii under L1.** The Binomial(k_i, p(d)) likelihood
  carries over. p(d) = V_L1(t1, d, marginals) / V_L1(t2, d, marginals).
  MLE via Brent on log-ratio.

- **Calibration on L1 synthetic.** Same synthetic-data generator
  (structured ternary, p_nonzero=0.62), measure pairwise L1, recover
  known d. Pass criterion unchanged: |d̂ − d| / d < 0.20 for d ≥ 10.

## What gets verdict-revised

The prior verdicts on M1, M2, M3 under categorical Hamming are
**not retracted as numbers** — they are correct measurements of
substrate-under-categorical-Hamming. They are **retracted as
evidence for or against the vision claim**, because the vision
claim is about the cell-graph and categorical Hamming flattens it.

In particular, the **"VALIDATED → MIXED → falsified" narrative on M1
across the three commits (309fed0, e569f79, c10bd39) does not speak
to the vision claim**. It speaks to a claim I made up about
emergent manifold compression under a metric that destroys the
substrate's distinguishing alphabet structure. Useful as
methodology history; not load-bearing for the vision.

## What Phase β tests instead

The Phase β pre-registration (next journal:
`td28_phase_beta_prereg_2026-05-12.md`) replaces categorical Hamming
with L1 on trits and adds new tests targeted at the cell-graph claim:

1. **L1-substrate vs Hamming-substrate.** Same data, different metric.
   If L1 reveals lower intrinsic dim than Hamming does, the path-graph
   structure is doing real work that categorical Hamming hid.

2. **L1-substrate vs binary baselines.** B2 (random sign) and B4
   (PCA+sign) under their native Hamming metric. Substrate under L1.
   At equal information capacity, does the path-graph structure give
   substrate a geometric advantage?

3. **Scrambled-ternary control.** Permute the three labels so 0 is
   NOT the center of the path graph (e.g., reassign labels: −1 → −1,
   0 → +1, +1 → 0, so the path becomes −1 → +1 → 0). Re-run L1.
   If the centrality of 0 is load-bearing, scrambling should hurt.
   If scrambling doesn't matter, the cell-graph framing isn't doing
   what the vision says.

4. **Zero-state utility.** Marginal entropy per cell, and a downstream
   test: does the zero state correlate with low |K| in the source
   data (the silence signal)? If yes, the 0 is a real geometric
   anchor.

## Discipline log

This is the **12th caught misalignment** of the session sequence —
and the most consequential. The math was right, the rigor was real,
the verdict was *honestly recorded under the metric I chose*. But
the metric was the wrong test for the vision claim, and only the
user catching it kept Phase α from becoming a journal of conclusions
that didn't match the question.

Memory updated with two entries:
- `project_vision.md` — refined claim 3 to "base-3 IS the graph,"
  with the cell-level path structure explicit.
- `feedback_metric_choice_trap.md` — categorical Hamming on ternary
  is not a default; choose the metric that respects the alphabet's
  native graph structure.

## What's next

Phase β pre-registration → L1 distance implementation → calibration
on L1 synthetic → Phase β run → honest verdict against the cell-graph
claim. The infrastructure carries over; the question changes.
