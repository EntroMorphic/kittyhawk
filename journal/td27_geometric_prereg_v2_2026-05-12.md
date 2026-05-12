# Phase α geometric-structure diagnostic — frozen pre-registration v2

**Supersedes** `td27_geometric_prereg_2026-05-12.md` (commit `9a81745`)
and its red-team amendment in `33b53ce`. This v2 addresses all 10
red-team findings before any implementation. Sections marked
**FROZEN** are locked; modifications require explicit justification in
the modifying commit.

## Motivation (unchanged from v1)

Test whether substrate signature space has GEOMETRIC structure that
scalar/binary representations of the same data lose. This is the
first measurement that could connect the vision memory's strong claim
("base-3 carries information base-2 collapses") to a measurable
property. If validated, Round 2 Nyström operations (soft routing,
bank interpolation, Nyström compression — see
`td27_spline_explorations_2026-05-12.md`) become substantive
downstream applications. If not, the substrate's claim narrows to
"useful primitive, not architecturally distinctive."

## What v2 fixes from v1

| # | v1 issue | v2 fix |
|---|---|---|
| 1 | Macocco estimator is for binary Hamming or L¹ on integer lattice, NOT ternary categorical Hamming | Derive ternary categorical Hamming shell-volume formula from first principles; document the derivation; validate on synthetic data |
| 2 | B2 baseline information-asymmetric (D=128 trits ≈ 202 bits vs D=128 bits = 128 bits) | Equal-bits comparison: D=128 trits (~203 bits with `log₂(3)≈1.585`) vs D=203 bits |
| 3 | Citation hallucinations | Verified: Macocco-Glielmo-Grilli-Laio (PRL 130, 067401, 2023); Ruppik et al. (NeurIPS 2025, arXiv:2506.01034), not "Mattia et al." |
| 4 | Loki unacknowledged | Cited (Singhania et al., NeurIPS 2024, arXiv:2406.02542); strong null hypothesis for M1 |
| 5 | DASH-KV / HashEvict Hamming-on-K precedent unacknowledged | Cited; novelty narrowed to "ternary" not "Hamming" |
| 6 | TwoNN bias 5-20% unaccounted for | Calibration on synthetic known-d data; require effect-size > bias-floor |
| 7 | "Close-prototype regime" doesn't transfer to K-cache | Operationalized per FROZEN section below |
| 8 | N=10K sample arithmetic sloppy | Honest accounting; pre-register actual N from ACTV2 dumps |
| 9 | 1-2 day estimate optimistic | Honest estimate: 1-2 weeks |
| 10 | M3 "longest/second-longest > 2.0" magic | Pre-register threshold computed from B3 (random projection) null distribution |

## FROZEN — Ternary categorical Hamming intrinsic-dimensionality estimator

### Derivation (FROZEN)

For substrate signatures in `{-1, 0, +1}^D` under **categorical Hamming**
(`d(s1, s2) = #{i : s1[i] ≠ s2[i]}`, counts disagreements regardless of
magnitude):

**Shell volume.** Number of signatures at exact distance `r` from a
fixed signature in the ambient space:
```
V(r) = C(D, r) · 2^r
```
Derivation: pick r positions out of D to disagree (`C(D,r)`); at each
disagreeing position, the new signature can take 2 values (the two
not equal to the original) → `2^r`. Total ambient space size: `3^D`.

**TwoNN ratio under ternary categorical Hamming.** Following Macocco
et al. (2023, PRL) for binary Hamming, generalized to alphabet size
A=3:

For two nearest neighbors at integer distances r1 ≤ r2, the ratio
`μ = r2/r1` is NOT continuously distributed (integer support). Use the
**I3D binomial-shell likelihood** adapted to ternary:

```
P(r2 ≤ R2 | r1 = R1, local intrinsic dim = d)
  = 1 - (1 - V_ternary(R2)/V_ternary(R_max))^{N-1}
```
where `V_ternary(R) = Σ_{i=0}^{R} C(D, i) · 2^i` is the cumulative
shell volume under ternary Hamming, `R_max` is the max meaningful
distance, and `N` is the sample size.

The MLE estimate of `d` maximizes the likelihood over observed `(r1,
r2)` pairs. For binary, this reduces to Macocco's formula. For
ternary, the `2^r` factor in the shell volume changes the likelihood
shape — denser shells at intermediate `r`, sparser at the extremes.

**FROZEN constraint**: the implementation must verify this estimator
on synthetic ternary data of known intrinsic dimension (Section
"Calibration") before any LLM K-cache measurement.

### Implementation outline (FROZEN)

```
def ternary_categorical_hamming_id(signatures, N_neighbors=2):
    # 1. Compute all pairwise Hamming distances (popcount-friendly)
    # 2. For each point, sort distances; take r1, r2
    # 3. MLE over (r1, r2) pairs using the binomial-shell likelihood
    # 4. Return d_hat
```

The likelihood evaluation: discretize d over a grid (e.g., 1 to D in
steps of 0.5); compute log-likelihood at each grid point; return
argmax.

## FROZEN — Baselines

### B1: raw K-values in float32 (substrate vs its own input)

Compute L2 intrinsic dimensionality on the same K vectors before
threshold_extract. Uses standard Levina-Bickel / TwoNN (continuous
case). Tests whether substrate extraction ADDS structure or just
preserves what's already there.

### B2: sign-only binarization — EQUAL-BITS (load-bearing falsification)

**Equal-bits requirement.** D=128 substrate trits ≈ 128 · log₂(3) ≈
**202.86 bits**. So B2 must use D_B2 = 203 sign-bits to match
substrate's information capacity.

Implementation: project each 128-trit substrate signature's K source
vector to 203 dimensions via a **fixed random orthogonal projection**
(stable across all comparisons), then take sign of each projected
component. Distance: binary Hamming over 203 bits.

**Why equal-bits matters:** without it, "lower intrinsic
dimensionality for substrate" could be the trit machinery OR the 58%
higher capacity at fixed D.

### B3: random Gaussian projection → sign (with equal-bits also)

Same projection scheme as B2 (203-bit binary Hamming) but with the
projection RE-DRAWN with a different seed. Tests whether substrate's
specific threshold_extract τ matters vs any 203-bit random hash.

## FROZEN — Measures

### M1: Local intrinsic dimensionality

Per derivation above. Reported as `d̂` ± bootstrap 95% CI on the same
N signatures.

### M2: k-NN topology divergence

- Mutual-kNN reciprocity at k ∈ {5, 10, 20, 50}
- Local clustering coefficient distribution (median, IQR, 95th
  percentile)
- Degree distribution in symmetrized k-NN graph (hub-ness)

All metrics reported on substrate AND each baseline.

### M3: Persistent Betti-0

For each integer radius `r ∈ {0, 1, ..., 2·sig_dim}`: union-find of
edges within distance `r`, count components. Curve `r → β_0(r)`.

**Threshold for plateau detection (FROZEN amendment):** the
"longest/second-longest persistence bar > 2.0" criterion from v1 was
arbitrary. v2 replaces it with: longest bar > 2× the 95th-percentile
of bars in **B3's β_0 curve** (B3 is random projection; should have
no real cluster plateaus). If substrate's longest bar exceeds
B3-derived threshold, plateau is significant.

## FROZEN — Calibration (NEW in v2)

**Before** measuring on K-cache signatures, validate the M1 estimator
on synthetic ternary data of known intrinsic dimension.

### Synthetic dataset construction

For each target `d ∈ {2, 5, 10, 20, 50, 100}`:
1. Generate N = 5000 random vectors in {-1, 0, +1}^d (the manifold).
2. Embed into {-1, 0, +1}^128 via random injection (fixed seed): map
   each d-vector to a 128-vector where d cells are the original
   values and 128-d cells are sampled uniformly from {-1, 0, +1}.
3. Apply M1 estimator to the resulting 5000 × 128 substrate
   signatures.
4. Report `d̂` and compare to true `d`.

### Calibration pass criterion (FROZEN)

For each target `d`, `|d̂ - d| / d < 0.20` (within 20% relative).
The Levina-Bickel/TwoNN family has known 5-20% bias at moderate N
and d; this allows for that range.

**If calibration fails** (any synthetic `d` exceeds 20% relative
error): the estimator implementation has a bug, OR the
ternary-Hamming TwoNN formulation is wrong. Phase α implementation
**stops** until resolved.

## FROZEN — Sample size and data source

- **Data source**: existing ACTV2 dumps at `data/c_dump/` (per
  agent 1 inventory, 242 files × 30 layers × 6 positions × 2
  prompts × 12 capture sites).
- **Filter to K signatures**: capture sites `k_pre_rope` and
  `k_post_rope` only. Per file, that's 2 sites × 5 KV heads × HEAD_DIM
  dimensions per position.
- **Honest N**: 30 layers × 6 positions × 2 prompts × 2 K-site types
  × 5 KV heads = **3600 K-signatures per ACTV2 dump variant**. With
  pooling across layers and KV heads, total N ≈ 3600. (NOT 10K as v1
  claimed.)
- **Layer stratification**: report results at layers 0 (early), 14
  (mid), 29 (late) separately AND pooled. Geometry may differ.

## FROZEN — Substrate-claim verdict rules

For each measure M ∈ {M1, M2, M3}, the substrate must clear B2
(equal-bits sign-only) by an effect size LARGER than the calibration-
measured estimator bias on synthetic data, with 95% bootstrap CI
excluding zero.

Specifically:
- **M1**: `d̂_substrate < d̂_B2` by ≥ 20% relative, CI-significant.
  (Larger gap than the calibration bias floor.)
- **M2**: substrate's mutual-kNN reciprocity > B2's by ≥ 5 percentage
  points AND degree distribution less hub-dominated (Gini index
  lower by ≥ 0.05). Both must hold across at least 3 of 4 k values.
- **M3**: substrate's longest β_0 persistence bar > 2× B3 null
  threshold.

**Substrate claim VALIDATED** iff ≥ 2 of 3 measures clear the above.

**Substrate claim FALSIFIED** iff 0 of 3 measures clear.

**Substrate claim MIXED** iff 1 of 3 measures clears.

## FROZEN — Operational definition of "close-prototype regime" for K-cache

v1's "close-prototype regime" was synthesized prototypes 1-9 trits
apart. For trained K-cache, operationalize as follows:

**Close-prototype regime**: K-signatures from the SAME (layer,
kv_head) but DIFFERENT token positions within a prompt.
**Far-prototype regime**: K-signatures from DIFFERENT (layer,
kv_head) combinations.

The hypothesis: substrate's lower intrinsic dimensionality should be
more pronounced in the close regime (where prototypes are
genuinely-similar K's of nearby tokens) than the far regime (where
K's are unrelated by construction).

Pre-register reporting both regimes separately. The substrate-claim
verdict above applies to the CLOSE regime (where the substrate
distinctive claim should be most visible).

## NOT FROZEN — Implementation language and library choices

PyTorch and NumPy for numerical work. SciPy for statistics
(bootstrap CIs, KDE for B3 null distribution). No specific
implementation language constraint; the test is portable to any
language that can read the ACTV2 format and compute the metrics.

## Prior art (verified, citations corrected)

- **Levina & Bickel (2004)** — original MLE intrinsic dim. NIPS.
- **Facco et al. (2017)** — TwoNN. *Scientific Reports*.
  [nature.com/articles/s41598-017-11873-y](https://www.nature.com/articles/s41598-017-11873-y)
- **Macocco, Glielmo, Grilli, Laio (2023)** — *Intrinsic Dimension
  Estimation for Discrete Metrics*. PRL 130, 067401.
  [arxiv.org/abs/2207.09688](https://arxiv.org/abs/2207.09688) — the
  load-bearing methodology for the binary case; ternary derivation
  in this pre-reg builds on this framework.
- **Ansuini, Laio, Macke, Zoccolan (2019)** — TwoNN applied to deep-
  net representations. NeurIPS.
- **Ruppik et al. (2025)** — *Less is More: Local Intrinsic Dimensions
  of Contextual Language Models*. NeurIPS.
  [arxiv.org/abs/2506.01034](https://arxiv.org/abs/2506.01034) —
  closest precedent for local ID on LLM representations.
- **Naitzat, Zhitnikov, Lim (2020)** — Topology of Deep Neural
  Networks via Betti numbers. JMLR 21.
- **Singhania et al. (2024) — Loki** — measured linear ID of attention
  K's at ~80-rank with 90% variance.
  [arxiv.org/abs/2406.02542](https://arxiv.org/abs/2406.02542) —
  **strong null hypothesis for M1**; substrate must clear what's
  already known about K-cache low-rank structure.
- **HashEvict (Liu et al., 2024)** [arxiv.org/abs/2412.16187](https://arxiv.org/abs/2412.16187)
  and **DASH-KV (2026)** — Hamming-distance KV eviction. **Substrate
  novelty narrows to "ternary" not "Hamming."**
- **KV-cache low-dim structure precedents**: ClusterAttn (ACL 2025);
  Thin Keys, Full Values (arXiv:2603.04427).

## Implementation order (FROZEN)

1. **Calibration code first.** Implement M1 estimator + synthetic
   data generator + calibration check. Pass criterion: |d̂-d|/d <
   0.20 across all 6 synthetic d values.
2. **Bake real-data pipeline.** Load ACTV2 dumps, extract K
   signatures, compute substrate Hamming + B1 L2 + B2 binary
   Hamming + B3 random-projection Hamming distance matrices.
3. **Compute M1, M2, M3** on substrate + each baseline + close vs far
   regimes.
4. **Bootstrap CIs.** 1000 resamples; report (mean, 95% CI) per
   measure per baseline per regime.
5. **Apply verdict rules.** Verify the calibration bias was budgeted.
6. **Journal results.** Per-measure tables, plots, honest verdict.

## Estimated effort (FROZEN)

- Step 1 (calibration code): 2-3 days
- Step 2 (data pipeline): 1-2 days
- Step 3 (compute measures): 1 day (estimates fast at N=3600)
- Step 4 (CIs + analysis): 1-2 days
- Step 5-6 (verdict + journal): 1-2 days
- **Total: 1-2 weeks** of focused work (honest, not v1's optimistic
  1-2 days).

## What this pre-reg COMMITS the project to

If implementation results match this pre-reg's verdict rules, the
substrate-claim's strongest form ("base-3 carries information base-2
collapses") earns its first measurement-grounded support. If they
don't, the claim narrows.

**Either outcome is publishable.** A clean negative result here would
be the substrate's most important empirical finding since P0-4.

Per the discipline antibody: success/failure criteria defined BEFORE
data is observed. Implementation begins after this pre-reg is
committed.

## Sign-off

This v2 pre-registration is the immediately-actionable specification.
Once committed, implementation can begin without further methodology
discussion (modulo bug-fixes or specification holes discovered during
implementation, which require explicit justification in their
fixing commit).
