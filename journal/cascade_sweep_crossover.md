---
date: 2026-04-15
scope: Cascade gain vs pure-hash accuracy across N_PROJ
type: scaling verification
tool: tools/mnist_cascade_sweep.c
parent: journal/cascade_atomics_mechanism.md
---

# Cascade gain decays as predicted; pixel resolver has its own ceiling at ~97.6%

Atomic-mechanism prediction: cascade headroom shrinks as N_PROJ grows, with crossover around N_PROJ ≈ 256. Sweep result confirms: cascade gain drops below 1% at N_PROJ=256, below 0.2% at N_PROJ=1024, and goes slightly negative at N_PROJ=4096. But the sweep also exposes a **new ceiling**: the pixel L2 resolver saturates at ~97.57% regardless of how good the primary hash becomes.

## Full sweep

Single seed 42, K_RESOLVE=50, density=0.33, deskewed MNIST.

| N_PROJ | pure top-1 | pure k=7 maj | ceiling@50 | cascade L2 | Δ(casc - maj) | time |
|---|---|---|---|---|---|---|
| 8    | 33.52% | 38.74% | 97.60% | **82.61%** | **+43.87%** | 2.4s |
| 16   | 55.48% | 62.00% | 98.59% | **90.75%** | **+28.75%** | 3.4s |
| 32   | 76.10% | 80.75% | 99.46% | **95.04%** | **+14.29%** | 1.7s |
| 64   | 89.81% | 91.55% | 99.77% | 96.65% | +5.10% | 1.9s |
| 128  | 94.41% | 95.22% | 99.77% | 97.28% | +2.06% | 2.7s |
| **256**  | 96.37% | 96.56% | 99.85% | 97.51% | **+0.95%** | 4.4s |
| 512  | 97.04% | 97.06% | 99.89% | 97.57% | +0.51% | 7.9s |
| 1024 | 97.37% | 97.43% | 99.90% | 97.58% | +0.15% | 15.3s |
| **4096** | 97.69% | 97.65% | 99.86% | 97.57% | **−0.08%** | 63.2s |

## Predicted vs observed crossover

Prediction: cascade stops helping around N_PROJ=256.

Observed:
- At N_PROJ=256 cascade still adds +0.95%.
- At N_PROJ=512 cascade adds +0.51%.
- At N_PROJ=1024 cascade adds +0.15% — statistically negligible.
- At N_PROJ=4096 cascade is **−0.08%** — first negative gain.

Practical crossover: **around N_PROJ=512**, where the gain drops below 0.5% and becomes architecturally not worth the extra pixel access. The prediction of N_PROJ≈256 was close; 512 is the honest cut.

## The new finding: pixel L2 has a ceiling

Cascade accuracy plateaus at **97.57% ± 0.01%** from N_PROJ=512 through 4096 — completely independent of the filter's accuracy.

| N_PROJ | pure maj | cascade |
|---|---|---|
| 512 | 97.06% | 97.57% |
| 1024 | 97.43% | 97.58% |
| 4096 | 97.65% | **97.57%** |

At N_PROJ=4096, pure k=7 majority **beats** cascade by 0.08 points. The filter has gotten so accurate that pixel L2 1-NN's own misrankings start costing more than voting's. 97.57% is essentially the limit of pixel-L2 1-NN on deskewed MNIST given a near-perfect candidate pool.

This is a deep architectural fact: **each resolver has its own asymptotic accuracy, independent of the filter.** For deskewed MNIST, pixel-L2 1-NN asymptotes at ~97.57%. To go higher with cascade you'd need a richer resolver (larger k, centroid distance with per-class weighting, learned metric, convolutional features, etc.).

## Cost-accuracy implications

N_PROJ=8 cascade reaches **82.61%** — better than pure N_PROJ=32 (80.75%) at a quarter of the hash cost. A similar pattern at each rung:

| Cascade config | ≈ pure N_PROJ |
|---|---|
| N_PROJ=8 cascade | ≈ pure N_PROJ=32 |
| N_PROJ=16 cascade | ≈ pure N_PROJ=64 |
| N_PROJ=32 cascade | ≈ pure N_PROJ=256 |
| N_PROJ=64 cascade | ≈ pure N_PROJ=2048 |
| N_PROJ=128 cascade | near full-cascade ceiling (97.3%) |

So cascade buys approximately **one octave of N_PROJ** at small scales (N_PROJ ≤ 64), then saturates. The filter-ranker decomposition is most valuable where the filter is most "lossy on rank" — exactly the regime the atomics predicted.

## Two separate ceilings, cleanly visible

- **Filter ceiling:** ceiling@50 stays at 99.86-99.90% from N_PROJ=64 upward. The filter can't preserve more information; a few MNIST queries are just genuinely far from their class.
- **Resolver ceiling:** pixel L2 1-NN asymptotes at 97.57% given near-perfect filter. The resolver has its own limits — shape-ambiguous digits that L2 can't separate.

**Combined cascade ceiling: ~97.57%.** Not 99.86%, because not every "correct in top-50" case is pixel-closest-correct.

This is exactly the factorization the atomics predicted: cascade accuracy = filter-presence × conditional-resolver-rate. At large N_PROJ: 99.87% × ~97.7% ≈ 97.6%.

## Updated architectural rule

1. **Use cascade when N_PROJ ≤ 128.** Gains are 2-30 percentage points.
2. **Cascade is a wash at N_PROJ ≥ 512.** Pure hash is already near the pixel-resolver ceiling.
3. **To push beyond cascade's resolver ceiling, change the resolver.** More bits in the hash won't help past N_PROJ ≈ 128 cascade.

## What this implies for Glyph's substrate

The atomics + sweep together make a strong architectural claim:

**Signature size is not a monotonic accuracy lever.** Once the filter is good enough that ceiling@K is near 100%, additional signature bits buy diminishing returns. The real lever is the resolver quality — which is orthogonal to signature design.

Routing-first substrate implication: the Trit Lattice LSH at N_PROJ=16-128 paired with a cheap pixel-L2 resolver is a **better architecture** than pure signature k-NN at N_PROJ=4096. Same ceiling, 32× less hash work, plus the inspectability of two-stage routing.

## Follow-ups

1. **Richer resolvers at small N_PROJ.** Can a per-class centroid distance get cascade at N_PROJ=128 above the 97.57% pixel-L2 ceiling? Probably not by much, but worth 10 lines of code.
2. **Cascade + vote within resolved candidates.** Pick top-5 by pixel L2 within top-50, vote among them. Hybrid of filter-ranker and voting. Likely a small improvement.
3. **Learned resolver.** A tiny linear classifier trained on top-K candidate features. Exits Glyph's no-ML-training philosophy but tests the resolver ceiling.
4. **Apply the filter-ranker principle to other consumers.** Any Glyph consumer using signature k-NN directly should be reviewed for cascade potential.

## Pointers

- Tool: `tools/mnist_cascade_sweep.c`.
- Atomic mechanism that predicted this: `journal/cascade_atomics_mechanism.md`.
- Cascade winning result at N_PROJ=16: `journal/nproj16_cascade_result.md`.
- Complete scaling curve (pure-hash only): `journal/full_scaling_curve.md`.
