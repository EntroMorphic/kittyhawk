---
date: 2026-04-17
phase: REFLECT
topic: Closing the CIFAR-10 gap to >50%
---

# CIFAR-10 >50% — REFLECT

---

## Core insight

RAW spent a long time circling between two-layer routing, feature
selection, and vote-pattern routing before landing on projection
selection. NODES clarified why: the ceiling is in the projections,
not the architecture. Every architectural improvement (resolver,
cascade, tables) operates DOWNSTREAM of the projection. If the
projection collapses Cat/Dog/Deer into the same Hamming region,
no downstream mechanism can separate them.

But REFLECT reveals something the RAW missed: **projection
selection and the LSH architecture are coupled, not independent.**

The brute-force showed N_PROJ=64 is optimal for RANDOM projections.
But if we SELECT projections for discriminability, the optimal
N_PROJ might shift. Selected projections are higher-quality per
direction — each trit carries more signal. Fewer might suffice
(N_PROJ=32 selected could beat N_PROJ=64 random). Or more might
help (N_PROJ=128 selected might not exhibit the noise-amplification
problem that random N_PROJ=128 showed).

The N_PROJ=64 peak was caused by random directions adding noise
past 64. Selected directions ADD SIGNAL past 64 — the peak could
move to a much higher N_PROJ. This means we should NOT assume
N_PROJ=64 is optimal for selected projections. The N_PROJ sweep
must be re-run after selection.

## T1 resolved: projection selection first, two-layer deferred

Projection selection is simpler (one tool, one pass), directly
attacks the diagnosed bottleneck (projection quality), and
composes with the existing LSH architecture unchanged. Two-layer
routing is interesting but untested and requires significant
infrastructure (Layer 2 training data generation, self-match
handling, vote encoding).

Select projections. Measure. If <50%, THEN consider two-layer
as the next escalation.

## T2 resolved: start with 1000, scale if needed

1000 candidates at 3072 dims: the ternary matmul for one
direction over 50K training vectors takes ~0.1ms. 1000 directions
= ~100ms total. Class mean computation: 50K × 1000 integer sums
grouped by class = negligible. Total selection pass: ~1 second.

If 1000 is insufficient (selected projections don't improve much
over random), scale to 10000 at ~10s cost. The selection is cheap
relative to the index build.

## T3 resolved: class separability first, routing accuracy later

Class separability (|μ_i - μ_j| between class means) is O(N_train)
per direction. Per-direction routing accuracy is O(N_train²) per
direction (requires scoring every pair). Separability is 50000×
cheaper.

More importantly, separability measures the DIRECTION'S
discriminative power, not its interaction with other directions.
Two directions with high separability but measuring the SAME
axis would be redundant. A follow-up step could de-correlate
the selected set (keep direction k only if it adds information
beyond directions 1..k-1). But start with pure separability
ranking — it's the minimum viable selection.

## T4 resolved: M tables × N_PROJ=1 selected trits each

This is the configuration that maximizes diversity. Each table
uses ONE highly discriminative trit. The union is built from M
independent "is this image on the positive or negative side of
this discriminative axis?" routing decisions. The SUM resolver
scores by how many discriminative axes agree on each candidate.

N_PROJ=1 per table means sig_bytes=1 (1 byte, containing 1 trit
in 2 bits). The bucket index has only 3 possible keys per table
(+1, 0, -1). That's extremely coarse — each bucket contains
~N_train/3 prototypes. Multi-probe at radius 1 hits all 3 buckets.
The union would be enormous.

Actually, that's too coarse. N_PROJ=1 per table with multi-probe
would put nearly ALL training prototypes in the union, defeating
the filter.

Better: group selected directions into tables. If we select 256
discriminative directions, group them into M=16 tables of N_PROJ=16
each (4-byte sigs, existing uint32 bucket key). Each table's 16
trit positions are the 16 most discriminative directions assigned
to that table. The bucket index has 3^16 ≈ 43 million possible
keys — fine-grained enough for effective filtering.

Or: group into M=4 tables of N_PROJ=64 each (16-byte sigs).
Requires the wider bucket key or the filter-on-first-16-trits
approach.

The natural configuration: **select 256 directions, group into
M=16 tables × N_PROJ=16.** Uses the existing uint32 bucket
infrastructure. Each table's 16 trits are all discriminative.
Compare to current: M=16 tables × N_PROJ=16 with random trits.

If this beats 38%, scale up: select more directions, group into
M=64 tables × N_PROJ=16 (1024 selected directions total).

## What I now understand

1. **The path to 50% is CURATED projections, not architectural
   complexity.** We've exhausted the downstream interventions
   (resolver, cascade, tables). The upstream intervention
   (projection quality) is the remaining lever.

2. **Routing-native curation:** generate random candidates,
   measure class separability via ternary matmul outputs, keep
   the best. The routing discovers discriminative structure in
   its own random space.

3. **The N_PROJ=64 peak applies to random projections only.**
   Selected projections may have a different optimal width because
   each trit carries more signal.

4. **The implementation composes with existing infrastructure.**
   No bucket changes needed if we use M×N_PROJ=16 tables. The
   projection matrix is different (curated vs random) but the
   packing, indexing, probing, and resolving are identical.

5. **The selection is cheap.** O(N_candidates × N_train) ternary
   matmuls + O(N_candidates × N_classes) class-mean computation.
   At 1000 candidates: ~1 second.

## What remains uncertain

- Whether selected projections actually improve over random. The
  hypothesis is that some random directions align with class-
  discriminative axes and selection finds them. If ALL random
  directions are equally uninformative (uniform ~10% per-class
  accuracy), selection can't help.

- Whether the gain is 2pp or 15pp. If the top 64/1000 directions
  are only marginally better than a random set of 64, we get 2pp.
  If they're dramatically better (some directions genuinely
  approximate gradient or edge detectors), we could get 10-15pp.

- Whether de-correlation of selected directions matters. Two
  directions that both detect "has blue sky" would be redundant.
  Pure separability ranking doesn't prevent redundancy.

- Whether the class-separability criterion (Fisher-like) is the
  right criterion. Separability measures class-mean distance but
  not class-overlap. Two classes with well-separated means but
  high within-class variance would score high on separability
  but low on actual discriminability.
