# Phase β pre-registration — substrate geometric structure under L1 cell-graph metric

This is the pre-registration for Phase β, supersedes Phase α's
FROZEN spec (`td27_geometric_prereg_v2_2026-05-12.md`) for the
substrate-distinctive claim. Phase α tested categorical-Hamming
intrinsic dimensionality, which flattens the ternary alphabet's
path-graph structure. Phase β tests the **path-graph structure
itself** as the vision claims it.

Sections marked **FROZEN** are locked; modifications require explicit
justification in the modifying commit.

## Motivation (revised per pivot journal)

Vision claim 3: **Base-3 IS the graph.** Trits live on a 3-vertex
path graph (−1 — 0 — +1) with 0 as the natural center. Bits live on
a 2-vertex graph (−1 — +1) with no middle. The substrate's
distinguishing property is the graph structure of its alphabet, not
emergent manifold structure of its data clouds.

The right substrate distance is **L1 on trits**:
`d(a, b) = Σᵢ |aᵢ − bᵢ|`.
Under L1: d(+1, −1) = 2, d(+1, 0) = 1, d(0, −1) = 1. The metric
encodes "passing through 0 costs less than crossing it." For binary,
L1 = categorical Hamming exactly, so the asymmetry is captured only
where substrate genuinely differs from binary.

## FROZEN — Substrate distance metric

**Substrate L1 distance.** For ternary signatures s1, s2 ∈
{−1, 0, +1}^D:
```
d_L1(s1, s2) = Σᵢ |s1ᵢ − s2ᵢ|
```
Integer-valued, range [0, 2D].

**Binary baseline distance.** For sign signatures b1, b2 ∈
{−1, +1}^D_b: standard Hamming `d_H(b1, b2) = Σᵢ [b1ᵢ ≠ b2ᵢ]`.
Equivalent to L1 on {−1, +1}, range [0, D_b]. Note: I'll report
d_H / 2 below to match the L1 unit (since under L1 each
disagreement costs 2 not 1 — actually no, for binary {−1, +1} a
disagreement is |(+1) − (−1)| = 2, so binary L1 distance ranges
[0, 2D_b]. To match scales, divide binary L1 by 2 to recover
"number of cells that differ.").

Conceptually: substrate L1 measures **total transition cost** to
convert one signature to the other along the cell-graph; binary L1
measures the same on a less-rich graph.

## FROZEN — Baselines (revised)

**B0 (NEW): Categorical-Hamming substrate.** Same substrate
signatures, but with categorical-Hamming distance instead of L1.
Tests whether the L1 metric reveals structure that categorical
Hamming was hiding. This is the substrate-as-Phase-α-saw-it.

**B1: Raw K float32, continuous TwoNN.** Unchanged from Phase α.

**B2: Equal-bits random sign.** D=203 sign bits via random Gaussian
projection of K. Native metric: Hamming on 203 bits.

**B3: Equal-bits random sign, different seed.** Same as B2, different
seed; null-distribution baseline for B2.

**B4: PCA + sign.** Top-203 PC projection of K, then sign. Structured
binary baseline at equal capacity.

**B5 (NEW): Scrambled-ternary control.** Same substrate signatures,
but label the cells with a permuted alphabet so 0 is NOT at the
center of the path graph. Specifically, apply the permutation
−1 → −1, 0 → +1, +1 → 0 to every cell. The resulting "scrambled"
signatures have the same Hamming distances to themselves as
substrate does, but the L1 distance between scrambled signatures
no longer reflects "passes through silence" — it reflects an
arbitrary cell-graph. If the centrality of 0 is load-bearing,
L1-substrate beats L1-scrambled. If not, scrambling doesn't matter
and the vision's "0 as center" claim is decorative.

## FROZEN — Measures (revised)

**M1: Local intrinsic dimensionality under chosen metric.**
Macocco fixed-radii estimator (ARCH-A from Phase α) applied with
L1-substrate shell volumes (re-derived; see Implementation below).
Reported as d̂ and d̂/D_max where D_max is the maximum possible
distance in the metric (substrate L1: D_max = 2D = 256;
Hamming-substrate: D_max = D = 128; binary L1 / 2: D_max = D_b = 203).

**M2: kNN topology divergence.** Unchanged from Phase α: reciprocity
at k ∈ {5, 10, 20, 50}, degree Gini, clustering coefficients. Now
applied to L1-substrate kNN graph.

**M3: Persistent Betti-0 via Wasserstein bar distribution.** Phase α
remediation's new metric. Apply to L1-substrate.

## FROZEN — Verdict rules

The Phase α remediation showed pre-registered comparison rules can
trap by treating mismatched ambient spaces as comparable. Phase β
verdict rules require **normalized comparisons** explicitly and add
the scrambled-ternary control.

**P1: L1-substrate shows lower intrinsic dim than its
Hamming-equivalent (B0).**
Required: d̂_L1_substrate < d̂_B0_Hamming-substrate, normalized by
their respective D_max, with bootstrap CIs disjoint. The
normalization handles that L1 ranges [0, 2D] while Hamming ranges
[0, D]. **If P1 fails, the L1 metric isn't revealing structure that
Hamming was hiding** — substrate's path-graph claim is not earning
its keep.

**P2: L1-substrate matches or beats B4 (PCA+sign) on d̂/D_max.**
Required: d̂_L1_substrate / 2D ≤ d̂_B4 / D_b at the 95% CI level.
**If P2 fails, even with the right metric substrate doesn't compete
with structured binary at equal capacity.**

**P3: Zero centrality matters (substrate beats scrambled-ternary).**
Required: d̂_L1_substrate / 2D < d̂_L1_scrambled / 2D, CI-disjoint.
**If P3 fails, the centrality of 0 is decorative; the substrate's
"0 as silence" framing has no measurable effect.**

**Substrate-claim VALIDATED iff ALL THREE of P1, P2, P3 clear.**
**FALSIFIED iff 0 of 3 clear.**
**MIXED for 1 or 2 of 3.**

This is stricter than Phase α: previously 2-of-3 measures sufficed,
now all three must clear. Justification: each P-rule tests a distinct
load-bearing component of the vision claim. Failing any one is
informative.

## FROZEN — Calibration

The L1-ternary estimator must be recalibrated. Closed-form L1 shell
volumes derived from the per-cell PMF:
- Uniform-ternary: P(|aᵢ−bᵢ|=k) ∈ {1/3, 4/9, 2/9} for k ∈ {0, 1, 2}.
- Structured-ternary (substrate-like, ~62% nonzero, balanced sign):
  ~ {0.38 + 0.31²·2 ≈ 0.57, ..., adjust empirically}.
Cumulative shell volume V_L1(t, d) = D-fold convolution of single-cell
PMF, summed for k ≤ t/2 ... actually no, summed for total ≤ t. Use
DP: shell_cum[d, t] = Σ_{k=0..2} P(k) · shell_cum[d-1, t-k].

Pass criterion: |d̂ − d_true| / d_true < 0.20 for d_true ∈ {10, 20,
50, 100}, on synthetic structured-ternary data with empirical
marginals matching substrate.

## FROZEN — Data

K-cache corpus from `data/c_dump/` + `data/c_dump_v2/`. N = 12,300
K-signatures across 7 prompts × multiple positions × 30 layers × 5
KV heads × 2 K-sites (pre/post-RoPE). Same as Phase α remediation.

Sample N_eff = 1500 per regime for pooled measures. Bootstrap B=200
at N_sub=500.

## NOT FROZEN — Implementation language

Python + NumPy + SciPy. L1 distance via the same one-hot matmul
trick used for Hamming, generalized: encode each trit value at each
cell as 1 + (value+1)/2 — no, simpler — encode trits in {−1, 0, +1}
directly as float, compute pairwise L1 via `|x[:,None,:] − x[None,:,:]|.sum(axis=-1)`.
At N=1500 D=128 this is 0.3GB intermediate — fits.

## Implementation order (FROZEN)

1. **L1 distance.** Implement `pairwise_L1_int8` (vectorized).
2. **L1 shell volumes.** DP for V_L1(t, d) under structured marginals.
3. **L1 Macocco estimator.** Reuses ARCH-A solver from m1_estimator_v2;
   inject L1 shell volumes.
4. **Calibration.** Synthetic structured-ternary at known d ∈ {10, 20,
   50, 100}, verify |d̂ − d|/d < 0.20.
5. **Phase β run.** All P1/P2/P3 tests with bootstrap CIs on full corpus.
6. **Journal results.**

## Estimated effort

Same-day implementation: L1 distance + shell volumes + calibration ~1h,
full run ~1h, journaling ~1h.

## What this commits the project to

If P1+P2+P3 all clear, the substrate's path-graph claim has its first
measurement-grounded support — different question than Phase α
attempted to answer, different methodology, ideally different result.

If P3 fails specifically (centrality of 0 isn't load-bearing),
the "0 as silence" framing is rhetorical; substrate's third state
behaves like an arbitrary label and the vision's strong form is
falsified at the alphabet level.

If P2 fails specifically (substrate doesn't beat B4 even with the
right metric), substrate-as-low-bit-quantization is still valid
but **base-3 doesn't add geometric value over structured binary at
equal capacity**.

Each result is informative; the pre-registered structure ensures we
record what we found, not what we hoped.

## Sign-off

Once this pre-reg is committed, Phase β implementation begins. No
further methodology discussion needed (bug-fixes during implementation
require explicit justification in the fixing commit).
