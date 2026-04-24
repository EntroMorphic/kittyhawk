---
date: 2026-04-22
scope: step_change cycle — atomics of what worked (multi-scale, region_tau) and what didn't, across MNIST / Fashion-MNIST / CIFAR-10
phase: measurement
---

# Step-change atomics

Tooling: `direct_lsh --dump_preds PATH` emits per-query `(qi y lsh sel pig gsh)` lines. Nine dumps collected — baseline / MS4 / MS4+R4 across all three datasets. Analysis via paste + awk over the 10k-query files.

## Headline selective accuracy (recap)

| Dataset | base | +MS4 | +MS4+R4 |
|---|---|---|---|
| MNIST | 97.18% | 97.24% | 97.30% |
| Fashion-MNIST | 87.95% | **88.66%** | 88.29% |
| CIFAR-10 | 46.63% | 47.73% | **48.05%** |

## Atomic 1 — per-transition prediction churn

Per-query flip counts between configs:

| Transition | w→r | r→w | net | wrong-but-different |
|---|---|---|---|---|
| CIFAR base→MS4 | 889 | 779 | **+110** | 1529 |
| CIFAR MS4→+R4 | 886 | 854 | **+32** | 1578 |
| Fashion base→MS4 | 244 | 173 | **+71** | 91 |
| Fashion MS4→+R4 | 159 | 196 | **−37** | 75 |
| MNIST base→MS4 | 66 | 60 | +6 | 26 |
| MNIST MS4→+R4 | 46 | 40 | +6 | 17 |

**Churn scales with dataset difficulty.** CIFAR sees 1500+ "wrong-but-different" flips per transition — the scorer is deeply unstable on hard queries. Fashion has 75–91. MNIST has 17–26 (near saturation).

**Net gain is a small residual on top of large churn.** MS4 on CIFAR: 889 helpful flips vs 779 harmful → only +110 net. This is not a uniform improvement; it's the result of a ranking perturbation that happens to help slightly more than it hurts. Same mechanism at MS4→+R4.

## Atomic 2 — scorer-component breakdown

Selective = Hamming k-NN when LSH agrees with GSH, else pair-IG re-rank. Decomposing:

### CIFAR-10

| Config | LSH | PIG | GSH | SEL | agree |
|---|---|---|---|---|---|
| base | 44.68% | 45.73% | 36.87% | 46.63% | 50.0% |
| MS4 | 45.71% (+1.03) | 46.77% (+1.04) | 37.39% (+0.52) | 47.73% (+1.10) | 50.8% |
| MS4+R4 | 45.46% (**−0.25**) | 47.47% (+0.70) | 37.52% (+0.13) | 48.05% (+0.32) | 50.2% |

**R4 on CIFAR does not improve Hamming k-NN — it DROPS raw LSH by 0.25pp.** The gain (+0.32pp Selective) comes entirely from pair-IG (+0.70pp). Per-region tau generates signatures where class-pair-weighted distance extracts more signal even though uniform Hamming extracts less.

### Fashion-MNIST

| Config | LSH | PIG | GSH | SEL | agree |
|---|---|---|---|---|---|
| base | 87.78% | 87.21% | 85.26% | 87.95% | 91.0% |
| MS4 | 88.25% (+0.47) | 87.78% (+0.57) | 86.34% (+1.08) | 88.66% (+0.71) | 91.5% |
| MS4+R4 | 88.24% (−0.01) | 87.26% (**−0.52**) | 86.35% (+0.01) | 88.29% (**−0.37**) | 91.5% |

**R4 on Fashion leaves LSH flat and HURTS pair-IG by 0.52pp.** The Selective regression traces entirely to pair-IG collapsing.

### Disagreement-gate quality

Selective routes to pair-IG when LSH disagrees with GSH. `P(PIG correct | disagree)` is the quality of that routing:

| Config | P(LSH\|dis) | P(PIG\|dis) | PIG's edge |
|---|---|---|---|
| CIFAR base | 33.0% | 36.9% | +3.9pp |
| CIFAR MS4+R4 | 32.8% | 38.0% | **+5.2pp** |
| Fashion base | 54.6% | 56.5% | +1.9pp |
| Fashion MS4+R4 | 52.4% | 52.9% | **+0.5pp (collapsed)** |

On CIFAR, R4 widens pair-IG's disagreement-resolution edge by +1.3pp. On Fashion, it collapses the edge from +1.9pp to +0.5pp. **That collapse is the mechanism for Fashion R4 regression.**

## Atomic 3 — per-class breakdown: who gains, who loses?

### CIFAR R4 winners (MS4 → MS4+R4 class net)

| Class | net | reading |
|---|---|---|
| 6 Frog | +13 | subject-dominated; benefits from spatial tau |
| 7 Horse | +13 | subject-dominated |
| 9 Truck | +12 | subject-dominated |
| 4 Deer | +11 | subject-dominated |
| 1 Car | +6 | subject-dominated |
| 5 Dog | +6 | subject-dominated |

### CIFAR R4 losers

| Class | net | reading |
|---|---|---|
| 2 Bird | **−18** | sky-background; R4 confuses with similar-background classes |
| 8 Ship | −8 | sea-background |
| 3 Cat | −1 | complex shape; neutral |

### Fashion R4 losers (MS4 → MS4+R4 class net — all negative)

| Class | net | reading |
|---|---|---|
| 0 T-shirt | **−12** | upper-body-garment cluster |
| 6 Shirt | −7 | upper-body-garment cluster |
| 5 Sandal | −7 | footwear cluster |
| 4 Coat | −6 | upper-body-garment cluster |

### Fashion R4 migration directions (r→w — pairs most confused)

| Migration | count |
|---|---|
| 6 Shirt → 0 T-shirt | 17 |
| 6 Shirt → 2 Pullover | 16 |
| 0 T-shirt → 6 Shirt | 15 |
| 5 Sandal → 9 AnkleBoot | 14 |
| 2 Pullover → 4 Coat | 14 |
| 4 Coat → 6 Shirt | 12 |

Intra-cluster confusion grows under R4. Upper-body garments (T-shirt, Pullover, Coat, Shirt) get mixed with each other; footwear (Sandal, Sneaker, AnkleBoot) likewise.

### CIFAR R4 recovery directions (w→r — hard pairs disambiguated)

| Migration | count | reading |
|---|---|---|
| 5 Dog → 3 Cat | 37 | classic hard pair |
| 8 Ship → 0 Airplane | 35 | vehicle cluster |
| 8 Ship → 9 Truck | 32 | vehicle cluster |
| 3 Cat → 5 Dog | 32 | classic hard pair |
| 4 Deer → 6 Frog | 30 | animal cluster |
| 1 Car → 9 Truck | 28 | vehicle cluster |

R4 on CIFAR specifically disambiguates classical hard pairs (cat/dog, vehicle trio, animal pairs).

## Mechanistic explanation

**Multi-scale (MS4)** adds downsampled intensity + gradient channels. It helps everywhere because it contributes information at coarser spatial scales that fine-grained single-scale features miss — specifically classes with multi-scale structure like natural-image scenes. Churn is real but the net is positive across all three datasets.

**Per-region tau (R4)** calibrates tau separately per 4×4 spatial region. Its effect depends on the class signal's spatial structure:

- **CIFAR-10: class signal IS spatially heterogeneous.** Natural images have different pixel statistics in different spatial regions (sky vs. ground, center subject vs. background). Per-region tau tracks these statistics, producing signatures where **per-class-pair weighted distance** (pair-IG) finds more signal. Raw Hamming k-NN (uniform weighting) slightly loses, but pair-IG more than compensates. Net positive.

- **Fashion-MNIST: class signal is GLOBAL.** Garment identity is defined by overall shape (silhouette, sleeves, hem), not per-region intensity statistics. Per-region tau introduces spatial noise — regions that "should" share tau get different taus, blurring the boundaries that distinguish upper-body garments from each other. Pair-IG weights, which were calibrated for a globally-tau'd signature, now read noise instead of signal. PIG collapses, Selective regresses.

- **MNIST: class signal is centered and saturated.** Digits occupy a near-centered 14×14 region of the 28×28 image. R4 is mostly neutral. Tiny churn.

## Why MS4 works where R4 does not

MS4 is **additive** — it appends new feature channels. The old channels are unchanged. Any signal the baseline extracted is still present; the new channels add (correlated, sometimes redundant) additional signal. Net positive across datasets because "more relevant channels" rarely hurts.

R4 is **substitutive** — it replaces one tau with many. When the spatial structure of class signal matches the region grid, it's a win. When class signal is global, it's a regression. **The effect depends on whether spatial heterogeneity is load-bearing for the class boundary.**

## Implications for future cycles

1. **R4 is dataset-conditional.** Default off; enable only when class signal is known to be spatially heterogeneous.

2. **Auto-detect R4 applicability.** A calibration-time statistic could gate it: measure per-region class-conditional intensity variance. High variance → enable R4. Low variance → leave it off.

3. **The scorer composition matters more than the distance.** On CIFAR, R4 loses on Hamming but wins on pair-IG. The Selective composite architecture (Hamming filter + pair-IG re-rank) benefits from signatures that are slightly worse for uniform distance but richer for weighted distance. This is a non-obvious design tradeoff.

4. **Ceiling on CIFAR from this path looks like ~48%.** MS4+R4 at 48.05% is the best measured. Moving beyond likely requires a different axis — not a deeper multi-scale, not a finer region grid.

5. **Fashion-MNIST's global-signal property is a known class.** If R4's spatial-tau paradigm applies elsewhere, datasets similar to Fashion (centered-object, uniform background) are "don't use R4" datasets. CIFAR-like natural scenes are "do use R4" datasets.

## Residuals / open

- Multi-seed verification of the CIFAR +0.32pp R4 gain not done. Could the CIFAR Selective +0.32pp also be noise? Three seeds would resolve.
- An 8×8 region grid might over-segment on 32×32 images. Would 2×2 or 3×3 be better on Fashion? (My earlier sweep showed 4×4 best on CIFAR; didn't sweep grid for Fashion.)
- A heuristic to auto-choose `--region_tau` based on training-set class-conditional statistics has not been implemented. Would make R4 opt-in-by-default for datasets that benefit, opt-out for the rest.
