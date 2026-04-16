---
title: Lattice Geometry Resolver — reading the routing pass's own measurements
status: Design proposal (2026-04-16)
companion: docs/DYNAMIC_NPROJ.md · journal/fashion_mnist_atomics.md · journal/rerank_first_light.md
evidence: CIFAR-10 atomics (75% per-table ties), cross-seed overlap (94% fate-invariant),
          dynamic N_PROJ cascade (37% ceiling despite 16→1024 width), M_rr=32 showing
          wider projections saturate
---

# Lattice Geometry Resolver

## The problem

The SUM resolver scores each candidate by summing M Hamming distances:

    score(c) = Σ_m  popcount_dist(q_sig_m, c_sig_m)

This treats every table equally. On CIFAR-10 at N_PROJ=16, 75% of
per-table (query, candidate) distance comparisons are tied — the
table cannot distinguish the correct neighbor from the confuser.
SUM gives those tied tables the same weight as the 25% of tables
that see clear separation. The decisive signal is diluted by noise.

The result: accuracy saturates at ~37% regardless of projection
width (16→1024) or table count (8→32). The ceiling is not the
projection — it's the resolver throwing away geometric information
the routing pass already computed.

## The insight

The routing pass — M independent projections, bucket lookup,
multi-probe, union construction — manifests a rich geometric
structure at every query. The SUM resolver collapses that structure
to a single scalar per candidate. Three categories of geometric
information are currently discarded:

### 1. Per-table confidence (margin)

For each table m, the routing pass produces Hamming distances from
the query to every candidate in the union. The gap between the
nearest candidate and the runner-up at table m is a measure of how
decisively that table's projection discriminates:

    margin(m) = d_runner_up(m) - d_winner(m)

A table with margin > 0 has a clear geometric preference — its
projection carved the lattice in a direction that separates the
winner from the field. A table with margin = 0 is geometrically
blind in that direction — its projection collapsed the winner and
runner-up to the same Hamming distance.

The CIFAR-10 atomics showed 75% of per-table pairs are tied
(margin = 0). SUM gives those 75% of tables equal weight with
the 25% that are decisive. A margin-weighted resolver would
suppress the blind tables and amplify the decisive ones.

### 2. Table-agreement patterns

When tables 0, 12, and 47 all place the same candidate closest
while tables 3, 8, and 22 place a different candidate closest,
the PATTERN of agreement encodes geometric information that SUM
discards. Different agreement patterns correspond to different
geometric relationships between the query and the candidates
in different subspaces of the lattice.

The per-table 1-NN vote (Atom 2) is a coarse summary of this:
on CIFAR-10, the true class wins 15.5% of per-table votes and
the winner wins 15.5%, with 69% going to "other." The table-
agreement structure has high entropy — votes are scattered across
many classes. A resolver that reads agreement PATTERNS (not just
vote counts) could extract structure that the scalar vote count
misses.

### 3. Distance-profile shape

The distribution of distances from the query to all union
candidates at a single table has shape: the winner might be
isolated (clear gap to the field) or embedded in a cluster of
same-distance candidates. That shape is a measurement of the
local lattice geometry around the query under that table's
projection.

A table where the distance profile is sharply peaked (one
candidate much closer than the rest) is highly informative.
A table where the profile is flat (many candidates at similar
distances) carries little discriminative power. The shape
encodes "how much does this projection's geometry resolve the
query's neighborhood."

## Design: margin-weighted SUM resolver

The simplest resolver variant that reads the lattice's own
geometry. Three progressively richer versions:

### V1: global margin-weighted SUM

Weight each table's contribution to the candidate score by the
table's overall decisiveness:

    weight(m) = margin(m) = d_2nd_nearest(m) - d_nearest(m)

    score(c) = Σ_m  weight(m) × dist(q_m, c_m)

Tables with large margin contribute proportionally more. Tables
with margin = 0 (tied at the top) contribute nothing. The winner
is the candidate with the lowest weighted sum.

Cost: one extra pass over the union per table to find d_nearest
and d_2nd_nearest. O(n_hit × M) — same as the existing SUM
resolver. The margin computation adds a constant factor (find
top-2 instead of top-1 per table), not an asymptotic increase.

### V2: per-candidate margin-weighted SUM

Instead of a global per-table weight, compute per-candidate
weights. For candidate c at table m, the weight reflects how
distinctive c's position is in that table's lattice:

    w(c, m) = max(0, d_median(m) - dist(q_m, c_m))

Candidates closer than the median get positive weight; candidates
farther get zero. This amplifies candidates that stand out in
multiple projections and suppresses candidates that are just
"average-close."

Cost: O(n_hit × M) to compute medians (via partial sort or
running estimate). Slightly more expensive than V1 but same
asymptotic complexity.

### V3: agreement-weighted SUM

Weight each table by whether its per-table 1-NN agrees with the
running best candidate. Tables that consistently point at the
same candidate reinforce each other; tables that disagree are
discounted:

    For each candidate c:
      agree_count(c) = number of tables where c is the 1-NN
      score(c) = (Σ_m dist(q_m, c_m)) / (1 + agree_count(c))

This is structurally similar to the vote-weighted resolver
(Phase A, falsified) but applied AT the wider projection
stages where the vote signal is richer. Phase A tested vote-
weighting at N_PROJ=16 where per-table votes were near-chance;
at N_PROJ=512+ the per-table votes should be more discriminative.

## Relationship to the Dynamic N_PROJ cascade

The lattice geometry resolver and the dynamic cascade are
complementary:

- **Dynamic N_PROJ** gives the resolver more measurements
  (wider projections produce finer Hamming distances with
  fewer ties).
- **Lattice geometry resolver** extracts more information
  from the measurements it has (weighted instead of uniform
  aggregation).

The cascade found that wider projections saturate at ~37% on
CIFAR-10 because even at N_PROJ=1024, the SUM resolver still
treats every table equally. The geometry resolver attacks the
same ceiling from the other side: even at N_PROJ=16, IF the
resolver reads the 25% of decisive tables and ignores the 75%
of tied tables, the effective signal-to-noise ratio improves.

The combination — dynamic cascade with geometry-aware resolution
at each stage — is the expected production architecture. The
cascade provides the right resolution per query; the resolver
reads the geometry at that resolution optimally.

## Relationship to prior resolver experiments

| resolver | what it reads | result | why |
|---|---|---|---|
| SUM (scalar) | sum of distances, equal weights | baseline | treats all tables equally |
| SUM (voteweighted) | sum_dist / (1 + votes) | falsified (Phase A) | votes at N_PROJ=16 are near-chance; dividing by noise adds noise |
| SUM (radiusaware) | sum_dist + λ×min_radius | falsified (Phase B.1) | radius is a coarsening of info already in sum_dist |
| SUM (NEON4) | same as scalar, SIMD | bit-exact, faster | same resolver, faster execution |
| **margin-weighted** | per-table confidence × distance | **proposed** | reads the lattice's own geometry to weight signal vs noise |

The key difference from the falsified variants: voteweighted and
radiusaware tried to fold in FILTER-STAGE information (how many
tables found the candidate, at what probe depth). That information
was already captured in sum_dist. Margin-weighting folds in
RESOLVER-STAGE information (how confident is each table's
measurement) which is genuinely new — it's not a summary of
something the unweighted sum already has.

## What this does NOT attempt

- **Learned projections.** The projection matrix stays random
  ternary. The geometry resolver reads the routing pass's own
  measurements — it doesn't change what's projected, only how
  the measurements are aggregated.

- **Pixel-space computation.** No L1/L2/cosine distance on raw
  pixels. No PCA, no centroids, no float at runtime. The resolver
  reads only Hamming distances in trit-lattice space.

- **Cross-candidate distance structure.** V1/V2/V3 all read
  query-to-candidate distances, not candidate-to-candidate. A
  future variant could use pairwise candidate distances to detect
  cluster structure within the union, but that's O(n_hit² × M)
  and deferred.

## Expected outcomes

### CIFAR-10

The 75% per-table tie rate means ~48 of 64 tables contribute
nothing under margin-weighting. The remaining ~16 decisive tables
carry all the weight. If those 16 tables have a consistent
geometric preference for the correct class (which the Atom 3 data
suggests — the mean per-table gap was +0.020 when restricted to
non-tied pairs), margin-weighting should amplify that preference.

Expected: 2-5pp improvement over scalar SUM at matched N_PROJ and
M. If the 25% of decisive tables have a 2:1 ratio of true-closer
to winner-closer (extrapolating from Atom 3: 13.4% vs 11.5%
after removing ties), margin-weighting would surface that ratio
more cleanly.

### Fashion-MNIST

65% per-table tie rate. Same mechanism, smaller expected gain
(1-3pp) because the decisive fraction is larger and the current
SUM already captures more of the signal.

### MNIST

The per-table tie rate is lower (estimated ~40-50%) and accuracy
is already 97%+. Margin-weighting should be neutral to slightly
positive — most of the signal is already captured by equal-weight
SUM.

## Implementation

### In glyph_resolver.{h,c}

New function:

```c
int glyph_resolver_sum_marginweighted(
    const glyph_union_t* u,
    int                  m_active,
    int                  sig_bytes,
    uint8_t* const*      table_train_sigs,
    const uint8_t* const* query_sigs,
    const uint8_t*       mask);
```

Same signature as glyph_resolver_sum. Internally:

1. For each table m, scan the union to find d_1st and d_2nd
   (nearest and second-nearest Hamming distances to the query).
   margin(m) = d_2nd - d_1st.
2. For each candidate c, compute weighted_score(c) = Σ_m
   margin(m) × dist(q_m, c_m).
3. Return the label of the candidate with the lowest
   weighted_score.

### In glyph_config

Add "marginweighted" to the --resolver_sum options.

### In the consumer tools

Wire the new resolver alongside the existing variants. No
structural changes to the query loop.

## Risks

1. **Margin computation doubles the inner-loop work.** Finding
   d_1st and d_2nd per table requires two passes or a single pass
   tracking top-2. At O(n_hit × M) this is 2× the cost of scalar
   SUM. For the re-rank pass where n_hit × M is already bounded,
   this is acceptable.

2. **Margin might correlate with distance.** If tables with large
   margins also happen to produce large distances (because their
   projection spans a high-variance direction), the weighting
   could amplify high-distance tables regardless of their
   discriminative power. Need to measure whether margin correlates
   with accuracy improvement or just with scale.

3. **The decisive 25% of tables might not agree.** If the non-tied
   tables disagree on which candidate is correct (some point at
   true, others at confuser), margin-weighting amplifies conflicting
   signals. The net effect could be no better than uniform.
   Atom 3 showed true-closer (13.4%) vs winner-closer (11.5%) on
   non-tied pairs — a 1.16:1 ratio favoring truth. Margin-weighting
   amplifies this ratio, but 1.16:1 is not a strong edge.

4. **V1 might not be enough.** The global per-table margin treats
   all candidates equally at a given table — it doesn't know which
   candidate the margin is ABOUT. V2 (per-candidate) is richer but
   more complex. Start with V1 to establish whether the mechanism
   works at all before adding per-candidate sophistication.

## Summary

The Trit Lattice LSH is geometric. The routing pass creates
measurements — per-table Hamming distances, vote patterns,
distance profiles — that encode the lattice's local geometry
around each query. The current SUM resolver reads only the sum
of distances, treating all tables as equally informative. The
lattice geometry resolver reads the routing pass's own confidence
signal (per-table margin) to weight decisive tables above blind
ones. The lattice measures itself and tells the resolver where
to listen.
