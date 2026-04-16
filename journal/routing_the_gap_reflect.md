---
date: 2026-04-16
phase: REFLECT
topic: Routing the CIFAR-10 gap — from 37.90% to 53%+ without leaving the lattice
---

# Routing the gap — REFLECT

---

## Core insight

RAW circled around spatial structure vs random subsets for
20 paragraphs. NODES crystallized the real question: does
reducing the per-table input dimensionality break the 37%
ceiling? Everything else is secondary.

The argument is clean:

1. N_PROJ scaling saturates (16→1024 all land at ~37%).
   More trits per table doesn't add information when each
   trit summarizes 192 pixels.

2. Table-count scaling saturates (M=8→64 gives <2pp). More
   tables with the same 1:192 compression ratio are
   redundant.

3. Resolver improvements saturate (+2.58pp from multi-
   resolution combined scoring). Better aggregation can't
   recover signal the projection destroyed.

4. MNIST at 1:49 compression reaches 97%. The compression
   ratio determines the ceiling.

5. Therefore: REDUCE THE INPUT DIMENSIONALITY PER TABLE.
   At D=192, N_PROJ=16 is 1:12 compression — denser than
   MNIST. If the hypothesis is right, per-table accuracy
   should jump dramatically.

**T1 resolved: random subsets first.**

Random subsets test the hypothesis with zero design choices.
If D=192 random subsets don't help, the hypothesis is wrong
and spatial blocks won't help either. If they help, spatial
blocks are the follow-up.

More importantly: random subsets are routing-native. The
subset selection comes from the same RNG that generates
projection weights. No external knowledge enters the system.
The routing architecture is routing through random input
subspaces — a generalization of routing through random
projections.

## T2 resolved: D = 256 (per-channel quadrant)

The right D is set by the compression ratio target. MNIST
works at 1:49. To match that on CIFAR-10:

    D = 16 × 49 = 784   (too large — only 4 tables)
    D = 16 × 16 = 256   (1:16, 12 tables for full coverage)
    D = 16 × 12 = 192   (1:12, 16 tables for full coverage)

D=256 is a natural breakpoint: 3072/256 = 12 disjoint subsets.
With M=16 filter tables, 12 cover the full image and 4 are
random repeats or wider overlapping regions. That's the
right balance between coverage and budget.

But actually, the SIMPLEST test is to just run with random
subsets at D=256 and M=16. No need to be clever about the
partition.

## T3 resolved: disjoint first, overlap later

Disjoint subsets maximize diversity and give cleanest signal
on whether subsetting helps. If M=12 disjoint tables cover
the full image, that's the minimum sufficient set. Additional
tables (M=16..64) add overlapping redundancy.

Start with M_filter=16, D=256, random subsets (some overlap
at M=16 over 12 disjoint blocks). Measure. If it works,
sweep M and overlap ratio.

## T4 resolved: subsetting replaces full-image tables, composes with everything else

Subsetting changes the projection INPUT, not the downstream
architecture. All existing mechanisms (multi-probe, k-NN,
multi-resolution re-rank, dynamic cascade) work unchanged
on top of subsetted projections. No replacement — composition.

The multi-resolution re-rank becomes especially powerful with
subsetting: re-rank at N_PROJ=64 over D=256 is 1:4 compression
— almost full-information per region. Compare to the current
1:96 compression at N_PROJ=32 over 3072 dims. The re-rank
actually HAS information to work with.

## What I now understand

The gap between 37.90% and 53% is not about resolver
sophistication, projection width, table count, or cascade
architecture. It's about **per-table information density**.
Each table's projection must be dense enough relative to its
input space that the resulting signature captures discriminative
structure. At 1:192 (3072 dims → 16 trits), it can't. At
1:12-1:16 (192-256 dims → 16 trits), it should.

This is a routing-native fix. The architecture already composes
M independent views. Making each view see a DIFFERENT, COMPACT
input subspace instead of the SAME, DILUTE full input is a
change to the routing topology, not to the mechanism.

## What remains uncertain

1. Whether random dimension subsets actually improve CIFAR-10
   accuracy. (Measurable — the whole point of the experiment.)

2. Whether the improvement is large enough to close the gap
   to 53%. Even at MNIST-like compression ratios, CIFAR-10's
   input distribution is harder (natural images vs pen strokes).
   The ceiling might be 45% instead of 53%.

3. Whether spatial coherence matters beyond random subsetting.
   If random subsets of 256 dims perform the same as spatial
   8×8×3 blocks of 256 dims, spatial structure doesn't help
   and the projection is purely about compression ratio.

4. Whether the multi-probe works well with subsetted projections.
   Each table's signature space is smaller (16 trits over 256
   dims vs 3072 dims). The bucket distribution may be different
   — more distinct buckets (less collision) because the
   projection is denser. Could be good (more precise routing)
   or bad (fewer multi-probe hits).
