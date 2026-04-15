---
date: 2026-04-15
scope: LMM cycle — can local + global routing with Trit Lattice LSH reach 97% at N_PROJ=16?
phase: REFLECT
---

# REFLECT

## Core insight

**In a routing-only substrate, "local routing" and "global routing" are not two different mechanisms. They are two different roles of the same primitive applied at two different aggregation scales. Local = per-query neighborhood lookup in a single bucket index. Global = union of neighborhood lookups across M independent bucket indexes. The global operator is the union itself — it requires no new primitive, no new data structure, no dense scan. It is literally the set-theoretic merge of per-table results.**

This reframes the 97% question. The question was: "can we break 97% at N_PROJ=16 with local and global routing?" The answer the reflection phase produces is: **yes if the M=? at which M independent 16-trit bucket indexes collectively address >97% of the necessary input-space neighborhoods is a feasible integer.** M is the only free variable. Everything else — the per-table multi-probe budget, the resolver choice, the wall-clock profile — is a consequence of how M interacts with the random-projection family's correlation structure.

The scaling curve at pure single-hash gives a floor: N_PROJ=512 reaches 97.06%, so ~512 trits of information content is the ceiling-of-ceilings we need to approximate. Random 16-trit tables composed via multi-table LSH are **not information-theoretically equivalent** to a single 32× larger hash — they have partially correlated coverage (same projection family, same density) but also benefit from independent multi-probe neighborhoods per table. In LSH theory this usually *beats* single-hash-at-same-bits for recall; we need to measure whether it beats or ties for accuracy-at-1NN.

## Resolved tensions

**T1 (what does "local + global routing" mean?) resolved.** Reading A — classical multi-table LSH with union-merge — is the productive interpretation AND the only one that preserves the no-dense-scan contract. I am committing to it for the synthesize phase. The user can correct if they meant something else; the commitment must be named explicitly, not silently assumed.

**T2 (independence vs correlation of random tables) is an empirical question, not a tension to resolve by reasoning.** The synthesize phase will measure it directly by reporting per-table disagreement rates at representative M values. If independence is high, small M suffices; if correlation is high, large M is needed or we need a different hash construction (density variation, seed-dependent tau).

**T3 (resolver choice) stays open as a secondary sweep.** Vote-count, summed-distance, per-table-1-NN-then-vote. The synthesize phase runs all three at each M. The best resolver at small M may differ from the best resolver at large M — measurement will show.

**T4 (empty-queries floor) predicts a fast early win.** At M=2, empty rate should drop from 1.75% to ~0.03% assuming near-independence. That alone recovers ~1.7 accuracy points over the M=1 bucket consumer. First measurable validation of the multi-table architecture.

**T5 (cost budget) gives the pitch shape.** Dense N_PROJ=512 at 97.06% ≈ 4000 μs/query; multi-table M=32 at 97% projected ≈ 320 μs/query. ~12× wall-time win at matched accuracy. The pitch is not just "97% is reachable" but "97% at 10× less cost than the dense path at matched information content."

**T6 (diminishing returns) affects only the tail.** Going from M=1 to M=4 should produce a large accuracy jump (85% → 93%+ is my rough model). M=4 to M=16 is where diminishing returns kick in. M=16 to M=32 is the 97% crossing. M=32 to M=64 is the ceiling-approach regime. The curve shape is what the experiment measures.

## Prediction

**Multi-table routed bucket LSH can break 97% at N_PROJ=16 with M in {16, 32, 64}, most likely M=32.** Derivation:

- M=1: measured 82.58% (current routed baseline)
- M=2: predicted ~88-90% (empty-query rescue + candidate union grows)
- M=4: predicted ~93% (matches N_PROJ=64 pure single-hash)
- M=8: predicted ~95% (matches N_PROJ=128 pure single-hash)
- **M=16: predicted ~96.5% (matches N_PROJ=256 pure single-hash)**
- **M=32: predicted ~97% (matches N_PROJ=512 pure single-hash) ← crosses target**
- M=64: predicted ~97.5% (matches N_PROJ=1024 pure single-hash)

The predictions track the scaling curve directly. **If multi-table tables are nearly independent, the experiment crosses 97% at M≈32.** If tables are moderately correlated, M=64 may be needed. If tables are strongly correlated (unlikely but possible), we may saturate below 97% and need hash construction changes.

## Cost predictions

At M=32 and per-table cost ~10 μs:

- Bucket lookups: 32 × 9.9 μs = 317 μs
- Candidate union merge: ~50 μs overhead (hash-set insert over ~4000 unique entries)
- Resolver vote-count: ~30 μs (pure counting, no distance)
- **Total: ~400 μs/query**

Dense N_PROJ=512 single-hash scan at 97.06%: ~4000 μs/query (from the scaling curve throughput data).

**Predicted win: 10× faster wall time at matched accuracy.** Memory cost: 32 MB of bucket indexes (32 × 937 KB + overhead). Fits comfortably.

## Architectural framing

This experiment is the direct extension of Axis 5 (signature-as-address) to a multi-hash routed architecture. The Axis 5 bucket consumer was a single-table LSH. This is the same architecture generalized to M tables with union-merge at the global step. The rules from Axis 4d (information leverage is filter-stage-first) apply directly: every hash goes into its own table's filter stage; nothing is spent on a per-table resolver.

The meta-router LMM cycle asked "can we route around a deficient filter?" and the answer was no, you should deepen the filter. This LMM cycle asks a structurally different question: "can we compose many small routed filters to reach what a single big dense scan reaches?" The answer — if my prediction holds — is yes, and the composition is union-merge over independent bucket indexes. Union-merge is the global routing operator the substrate has been waiting for.

## The question for SYNTHESIZE

**"Build multi-table routed bucket LSH at M ∈ {1, 2, 4, 8, 16, 32, 64} and measure the accuracy-vs-M curve. Cross-reference against resolver choice (vote-count, summed-distance, per-table-1-NN vote). Find the minimum M to break 97%. Report wall-clock, memory, and accuracy side-by-side with the dense N_PROJ=512 baseline."**

That's the experiment. ~600-800 lines of C. One-afternoon build, one-evening run.

Two risks named explicitly:

1. **If the M-accuracy curve saturates below 97%**, we've falsified the hypothesis and learned that at N_PROJ=16 random ternary projections have enough correlation to cap multi-table LSH below 97% no matter how many tables we add. That's a genuinely new finding about the projection family. We'd then need density variation, seed-dependent tau, or a structurally different hash construction — which belongs in a separate LMM cycle.

2. **If the curve reaches 97% but the per-query cost is much higher than predicted (say ~1500 μs)**, we've broken the target but not the cost story. The architecture is still a win over dense N_PROJ=512 (>2×) but not the 10× I'm predicting. Honest measurement; adjust the pitch.

Both outcomes are publishable. Neither wastes the cycle.
