---
date: 2026-04-16
phase: NODES
topic: Routing the CIFAR-10 gap — from 37.90% to 53%+ without leaving the lattice
---

# Routing the gap — NODES

---

## Node 1 — The compression ratio is the bottleneck, not the projection width

We proved N_PROJ=16 through 1024 all saturate at ~37% on CIFAR-10.
More trits per table doesn't help. But MNIST at N_PROJ=16 gets
97%. What's different?

MNIST: 784 dims → 16 trits = 1:49 compression.
CIFAR-10: 3072 dims → 16 trits = 1:192 compression.

At 1:192, each trit summarizes ~192 pixels. At 1:49, each trit
summarizes ~49 pixels. The per-trit information density is 4×
lower on CIFAR-10. And widening N_PROJ doesn't change the
INPUT dimensionality — it just adds more projections over the
same 3072 dims, each equally washed-out.

The fix: reduce the input dimensionality per table. Give each
table 192 dims instead of 3072. At N_PROJ=16 over 192 dims,
the compression is 1:12 — denser than MNIST.

## Node 2 — Three approaches to per-table dimension reduction

(a) **Random dimension subsets.** Each table projects a random
subset of D dims out of 3072. Subset selected by RNG — no
spatial knowledge. Routing-native by construction.

(b) **Fixed spatial blocks.** Partition the 32×32×3 image into
spatial regions (e.g., 8×8×3 = 192 dims). Each table projects
one region. Imposes spatial structure by design.

(c) **Routing-learned subsets.** Build many random-subset tables,
route training data, measure per-table accuracy, keep the most
discriminative subsets. Routing measures itself to select inputs.

**Dependency:** (a) tests whether input subsetting helps at all.
(b) tests whether spatial coherence adds value beyond random
subsetting. (c) is the routing-pure optimization of (b).

## Node 3 — Random subsets are the minimum viable experiment

Random subsets require zero design choices. The sig_builder
already takes input_dim — just set it to D=192 and pass a
subset of the input vector. The only new code is subset
selection (RNG-based) and per-query subset extraction.

If random subsets don't help, spatial blocks won't either
(they're a constrained version of subsetting). If random
subsets help, spatial blocks might help MORE (because spatial
coherence is real in images). Either way, random subsets are
the cheapest test of the subsetting hypothesis.

## Node 4 — Composition handles the fusion

M tables with different subsets → M independent lattice views
of different input subspaces → union merges candidates found
in ANY subspace → resolver scores by summed distance across
subspaces.

This is EXACTLY the multi-table composition from Axis 6, but
with each table seeing a different subspace instead of the
same space through different projections. The composition
mechanism is unchanged. Only the inputs change.

## Node 5 — Per-table subset overlap controls diversity

If subsets are fully disjoint (each table sees unique dims),
the tables provide maximum diversity but zero redundancy.
If subsets overlap heavily, tables provide less diversity but
more robust voting.

With M=16 tables and D=192 each: 16×192 = 3072 = full image.
A non-overlapping partition covers the whole image exactly once.
This is the most efficient use of table budget.

With M=64 tables and D=192 each: 64×192 = 12288 > 3072.
Subsets must overlap (~4× coverage). Each pixel is seen by
~4 tables. This provides redundancy at the cost of correlation
between tables.

**Tension:** high M with overlapping subsets vs low M with
disjoint subsets. The oracle data (100% at M≥8) suggests we
don't need high M for union completeness — the value of M
is in ranking quality, not finding quality.

## Node 6 — Color channels are a natural subset boundary

CIFAR-10 is 32×32×3. The three color channels (1024 dims each)
are a natural partitioning boundary:

- 3 tables × 1 channel each = 3 tables, D=1024.
- 12 tables × 1 quadrant per channel = 12 tables, D=256.
- 48 tables × 1 block per channel = 48 tables, D=64.

Channel separation means each table sees a single color's spatial
structure without cross-channel interference. A ship's blue sky
dominates the B channel; separating channels lets the B-channel
tables encode "blueness" without R/G noise.

## Node 7 — The re-rank should also use spatial subsets

The re-rank pass currently uses wider N_PROJ over the FULL
3072 dims. If the filter uses spatial subsets, the re-rank
should too — re-rank each table's score using wider signatures
over THAT TABLE'S spatial subset, not over the full image.

At D=192 per table, N_PROJ=32 gives 1:6 compression for
re-rank. N_PROJ=64 gives 1:3 — extremely dense, nearly
full information. Compare to the current re-rank: N_PROJ=32
over 3072 dims = 1:96 compression — still washed out.

## Node 8 — The union overlap problem

If each table sees different dims, candidates found by table 0
(top-left region) may not be scored by table 1 (top-right region)
during the resolve pass. But they ARE scored — the resolver
computes dist(q, c) at every table for every candidate in the
union. The dist at table 1 for a candidate found by table 0
still uses table 1's projection — it's just that the candidate
was FOUND via table 0's projection.

This is fine. The resolver reads all M tables regardless of
which table FOUND the candidate. The finding and scoring paths
are decoupled. A candidate in the union gets scored on every
spatial region.

## Node 9 — This composes with everything we've built

- Spatial subsets + multi-resolution re-rank: filter at N_PROJ=16
  per region, re-rank at N_PROJ=64 per region.
- Spatial subsets + k-NN: top-K by combined spatial score,
  rank-weighted vote.
- Spatial subsets + dynamic cascade: escalate uncertain queries
  to wider per-region projections.
- Spatial subsets + multi-resolution combined scoring: combine
  7 resolution levels, each with spatial subsetting.

The spatial subsetting is a change to the INPUT of the projection,
not to the architecture. Everything downstream works unchanged.

## Node 10 — What "routing the solution" means here

The user wants the routing architecture to DISCOVER the right
input subspaces, not have them imposed. Random subsets are
routing-native but not routing-DISCOVERED. Spatial blocks are
imposed structure.

The routing-discovered version: build 1000 random-subset
tables, route all training queries, measure per-table accuracy,
keep the best 64. The RNG generates candidates; the routing
measurements SELECT. This is analogous to random search in
hyperparameter optimization — random generation + measurement-
based selection.

The measurement is routing-native (Hamming distance, per-table
1-NN accuracy on training data). The selection is a simple sort.
The result is a set of input subsets that the routing itself
validated as discriminative.

## Tensions

**T1:** Random subsets vs spatial blocks vs routing-learned
subsets. Which first?

**T2:** Per-table D: how many dims per table? 192 (matches
MNIST compression ratio), 256 (per-channel quadrant), 1024
(per-channel)?

**T3:** Overlap: disjoint partition (M=16, cover once) vs
overlapping (M=64, cover 4×)?

**T4:** Does input subsetting compose well with the existing
multi-resolution scoring, or does it replace it?
