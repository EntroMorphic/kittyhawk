---
date: 2026-04-17
phase: NODES
topic: Closing the CIFAR-10 gap to >50%
---

# CIFAR-10 >50% — NODES

---

## Node 1 — The 38% ceiling is a projection quality ceiling

Every resolver, cascade, and table-count intervention has hit
the same wall: ~38% on CIFAR-10. Brute-force confirms the filter
is not the bottleneck. The ceiling is in the PROJECTIONS — random
ternary directions over 3072-dim RGB don't capture enough class-
discriminative structure.

## Node 2 — Two paths to better projections

(a) **Import structured features** (SSTT's approach): hand-design
spatial/gradient/intensity features. Proven to work (53%). Not
"routing the solution."

(b) **Select projections by routing measurement**: generate many
random directions, measure their class-discriminability using
routing primitives, keep the best ones. Routing-native: the
selection criterion is routing accuracy, the computation uses
ternary matmul and integer arithmetic.

## Node 3 — Routing-native projection selection

Generate N_candidates >> M random ternary projection directions
(e.g., 1000). For each direction w, compute the mean projection
value μ_c(w) = mean(w⋅x) for each class c over the training set.
Score the direction by how well it separates classes:

    sep(w) = Σ_{c≠c'} |μ_c(w) - μ_c'(w)|

Keep the top M directions. Build tables using these curated
directions instead of random ones.

Computation: N_candidates ternary matmul over N_train vectors.
The matmul is `m4t_ternary_matmul_bt` — existing kernel. The
class means are integer sums. The selection is a sort. Every
step is routing-native.

## Node 4 — The two-layer routing alternative

Layer 1: random projection → per-table vote pattern (64-dim).
Layer 2: project the vote pattern → classify.

Avoids the projection selection step. But requires routing
all training images through Layer 1 (expensive), handling
self-match, and encoding discrete vote patterns as MTFP for
Layer 2's ternary projection.

More complex, less proven. Defer until Node 3 is tested.

## Node 5 — What "routing the solution" means for projection selection

The user's constraint is that the solution must emerge from the
routing architecture, not from external pixel-space computation.
Projection selection satisfies this:

- The CANDIDATE directions are random ternary (RNG-generated).
- The SELECTION uses routing measurements (ternary matmul outputs,
  class-conditional means on the training set).
- The ARCHITECTURE is unchanged (bucket index, multi-probe, SUM/k-NN).

The routing tells us which of its own random experiments were
discriminative. It's not importing structure — it's discovering
which random structure happens to align with class boundaries.

## Node 6 — Expected gain from projection selection

If 64 curated directions out of 1000 random candidates are used
at N_PROJ=1 each (one trit per selected direction), we have 64
tables × 1 trit per table. Each trit is the most discriminative
single trit we could find. This is maximally selective but very
narrow per table.

Alternatively: select the best 64 directions and use them as
the 64 rows of a SINGLE projection matrix. One table at
N_PROJ=64 with curated directions. The signature is 64 trits
chosen for discriminability, not 64 random trits.

At the extreme: select the best 1024 directions and build
N_PROJ=1024 with all curated rows. But the brute-force showed
N_PROJ=64 is optimal, so curating 64 is the target.

## Node 7 — The selection might find gradient-like directions

Random ternary directions over 3072 dims sometimes approximate
spatial gradients by chance — e.g., a direction with +1 on
adjacent pixels in one row and -1 on adjacent pixels in the
next row effectively computes a vertical edge response. If
these gradient-like directions happen to score high on class
separability, selection would keep them. The routing would
DISCOVER gradient features without anyone hand-designing them.

Whether this actually happens depends on the density of
gradient-like directions in the random space. At 3072 dims
with ~1024 non-zero weights per direction, each direction is
a random 1024-pixel-subset sum-or-difference. Some of these
will correlate with edge structure. Out of 1000 candidates,
probably a few dozen approximate useful spatial filters.

## Node 8 — The N_PROJ=64 bucket architecture

Independent of projection selection, we need to run N_PROJ=64
with the full LSH. The plan:

- Compute 64-trit signatures (16 bytes) per prototype.
- Bucket index on the first 16 trits (4 bytes = uint32 key).
- Multi-probe at the 16-trit level.
- Resolver scores on all 64 trits.

This is the filter-ranker decomposition at a fixed architecture.
Bucket key uses a SUBSET of the signature as the routing address.
Resolver uses the FULL signature for scoring.

This works with both random and selected projections. Build
the bucket architecture first, verify it matches brute-force
at random projections, then swap in selected projections.

## Node 9 — k sweep at N_PROJ=64

We haven't swept k at N_PROJ=64 M=64. The brute-force has k=5
hardcoded. At N_PROJ=64, the distance distribution is different
from N_PROJ=16 — larger k might help if the correct-class
neighbors are distributed across a wider distance range.

Quick test: sweep k ∈ {1, 3, 5, 7, 10, 20} at N_PROJ=64 M=64
brute-force. If k=20 beats k=5 by >1pp, larger k is part of
the path to 50%.

## Node 10 — Combine selection + N_PROJ=64 + k sweep + LSH

The full path to 50%:
1. Build N_PROJ=64 LSH (bucket on first 16 trits, score on 64).
2. Verify it matches brute-force on random projections.
3. Generate 1000 random projection candidates.
4. Score each by class separability on training set.
5. Select top 64.
6. Rebuild tables with selected projections.
7. Sweep k.
8. Measure.

## Tensions

**T1:** Projection selection (Node 3) vs two-layer routing
(Node 4). Which first?

**T2:** How many candidate projections to generate? More is
better for selection quality but costs build time. 1000? 5000?

**T3:** Selection criterion — class separability (Fisher-like)
vs per-direction routing accuracy (expensive but more direct)?

**T4:** Should selected projections be at N_PROJ=1 per table
(64 tables, maximally diverse) or N_PROJ=64 in one table (one
table, maximally concentrated)?
