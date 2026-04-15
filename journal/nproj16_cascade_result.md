---
date: 2026-04-15
scope: Cascade experiment result at N_PROJ=16
type: LMM synthesize → execute → verify
tool: tools/mnist_cascade_nproj16.c
parent: journal/nproj16_to_90_synthesize.md
---

# Cascade at N_PROJ=16: the hash-as-filter reframe hits 92.72%

LMM-predicted 87-91%. Actual: **90.75% at K=50, 92.72% at K=100.** The filter-vs-classifier reframe is validated. 16-bit signature used as a coarse index plus pixel-L2 1-NN over a small filtered pool beats the pure-signature classifier by **+30 percentage points**.

## Setup

Deskewed MNIST, N_PROJ=16, density=0.33. Seed=42 for primary hash; seed=1337 for the secondary-hash resolver variant (E5). Single seed — the point is mechanism, not ±σ.

Primary pass: compute 60K Hamming distances per query.
Resolver pass: re-rank the top-K candidates by pixel distance (L1 or L2sq).

Inference time: 3.7s for all 10K queries × all 5 K values × all resolvers. Cascade is not expensive.

## Results

| K | pure-hash majority | cascade L1 1-NN | cascade L2 1-NN |
|---|---|---|---|
| 5 | 61.16% | 75.92% | 76.23% |
| 10 | 62.92% | 81.64% | 82.00% |
| 20 | 64.14% | 85.74% | 86.40% |
| 50 | 63.31% | **90.15%** | **90.75%** |
| 100 | 63.00% | 92.27% | **92.72%** |

**Ceiling at top-100: 99.50%** (correct class present). Prior estimate of 91.47% at top-10 was correct, but widening the candidate pool to top-100 expands the ceiling dramatically.

K=20 fixed variants:
- Pixel-L1 3-NN majority: 82.38% (worse than 1-NN — see below)
- Partition-aware (singleton → top-1, else L1 resolve): 78.50%
- Secondary-hash Hamming re-rank (E5): 74.84%

## Mechanism readout

### The filter-classifier reframe is real

At K=20, cascade gains **+21.6 points** over pure-hash majority (64.14 → 85.74). The 16-bit hash was never the limiting factor — the voting step on top of it was. The hash filters 60K → 20 correctly in the sense that ceiling-at-K=20 is already above 90%; voting just throws most of that signal away by collapsing rank and distance into counts.

### K=50-100 is the sweet spot

Contrary to the prediction that K=50+ would "confuse the resolver with far candidates," accuracy keeps climbing through K=100. Reason: MNIST classes are very separable in pixel space on small pools, and even a far candidate that's pixel-distant won't be picked as 1-NN. The resolver is more discriminating than feared. **The correct mental model: primary hash is a loose filter; pixel 1-NN is a strong discriminator that benefits from a wider candidate net.**

### L2sq marginally beats L1

L2 is +0.35 to +0.60 over L1 uniformly. Consistent with pixel-intensity differences being better modeled by squared distance — a typical MNIST observation.

### 1-NN beats 3-NN on filtered pools

At K=20 L1, 1-NN gets 85.74%; 3-NN majority gets only 82.38%. In a filtered pool of already-close candidates, the single nearest-pixel neighbor is usually the correct class; extending to 3 lets wrong-class near-misses vote. **In a cascade, don't re-introduce voting at the resolver stage.**

### Partition-aware hurts

E4 (skip resolver when tied_count==1) got 78.50%, *below* plain L1 cascade at 85.74%. Interpretation: singleton top-1 is wrong often enough that pixel re-rank also helps the singleton cases. The probe showed correct-in-tied-min is 75.85% overall; among singletons, probably similar or lower. Skipping pixel for singletons throws away free signal.

**Correction to the partition-aware hypothesis:** singletons aren't reliable. Apply the resolver uniformly.

### Secondary hash gives about half the cascade gain

E5 (another 16-bit hash as resolver) got 74.84% at K=20 vs cascade-L1 at 85.74%. So:
- Pure hash: 64.14%
- Hash + another hash as resolver: 74.84% (+10.7 from more bits)
- Hash + pixel resolver: 85.74% (+21.6 from pixel signal)

**Roughly half the cascade gain is "more bits" (secondary hash), half is "richer signal" (pixel access).** This is the clearest possible answer to whether N_PROJ=16 can "really" reach 90%: with only signatures (secondary hash), it reaches ~75%; with pixel fallback, ~90%. The architecture must include pixel access to cross 90%.

## Cost accounting

- Primary hash: 60000 × 16 = 960K trit-ops.
- Pixel-L1 resolver at K=50: 50 × 784 = 39K scalar-abs-ops per query.
- Ratio: resolver is 4% of hash cost. Hash still dominates.

Cascade at K=50 = 90.75%, cheaper than N_PROJ=128 dense k-NN (128 × 60K = 7.7M trit-ops) and about as accurate. The 16-bit hash as a *filter* is architecturally efficient; the 16-bit hash as a *classifier* was architecturally wrong.

## Comparison against the scaling curve

From the complete scaling curve:

| Architecture | Accuracy |
|---|---|
| Pure signature k-NN, N_PROJ=16, k=7 | 62.00% |
| Pure signature k-NN, N_PROJ=64, k=7 | ~91% |
| Pure signature k-NN, N_PROJ=128, k=7 | ~94% |
| Pure signature k-NN, N_PROJ=4096, k=5 | 97.99% |
| **Cascade (N_PROJ=16 + pixel L2, K=50)** | **90.75%** |
| **Cascade (N_PROJ=16 + pixel L2, K=100)** | **92.72%** |

The 16-bit cascade is competitive with N_PROJ=64 pure and approaches N_PROJ=128 pure, at a fraction of the hash cost. The LSH cascade pattern beats brute-forcing more projections.

## The architectural insight

**We conflated "how many bits the classifier uses" with "how many bits the filter uses."** The filter can be 16 bits cheap; the classifier can be richer (pixel access, secondary hash, learned comparator) — and the resulting system is named by its primary index, not by its final discriminator. This is standard LSH practice; Glyph had been doing something non-standard (hash-as-classifier) and paying for it.

The routing-first thesis is preserved: the 16-bit hash is the coarse routing substrate. The resolver is a consumer that operates on routed candidates. This is *more* aligned with Glyph's philosophy than the prior pure-signature k-NN — routing narrows, resolver decides.

## Follow-ups

1. **Cascade at larger N_PROJ.** Does the cascade gain shrink as N_PROJ grows (because pure-hash is already accurate)? Probably yes — at N_PROJ=4096 the pure baseline is 97.99%; cascade has little room. Likely crossover where cascade no longer helps is around N_PROJ=256-512.

2. **Learned 16-bit hash.** If a supervised hash separates classes better, pure-hash baseline at N_PROJ=16 might itself reach 80-90%, shrinking the cascade gap. Out of scope for now.

3. **Per-class centroid resolver.** Replace pixel 1-NN with distance to per-class centroid. Cheaper (10 distances instead of K) but may underperform — the probe showed tied sets are not just two-class; rough centroids lose local detail.

4. **Sweep density / seed across cascade.** Verify cascade is seed-robust. Expected: yes, since resolver is deterministic.

5. **Publish the cascade pattern.** This is a general architectural move for routing-first substrates: any coarse-index + local-resolver design. Document in `docs/M4T_SUBSTRATE.md` or similar.

## Pointers

- Tool: `tools/mnist_cascade_nproj16.c`.
- LMM synthesize that predicted this: `journal/nproj16_to_90_synthesize.md`.
- Atomic probe that motivated it: `journal/nproj16_atomic_mechanism.md`.
- Complete scaling curve for comparison: `journal/full_scaling_curve.md`.
- Prior amplification experiment (failed because fallback set was random, not filtered): `journal/amplification_negative_result.md`.
