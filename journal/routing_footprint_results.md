# Routing Footprint Experiment 1 — Results

**Date:** 2026-04-20
**Prediction:** 60-70% discrimination on CIFAR-10 resolver-miss cases
**Actual:** 16.7% on CIFAR-10, 31.8% on MNIST, 34.2% on Fashion-MNIST

---

## The hypothesis is falsified for Tier 1 co-bucket routing distance.

### Raw data

| Dataset | M | Resolver-miss (oracle hit, wrong k-NN) | Route discriminates | Tied | Route wrong | Avg d_route correct | Avg d_route wrong |
|---|---|---|---|---|---|---|---|
| MNIST | 64 | 318 | 31.8% | 4.1% | 64.2% | 40.84 | 37.31 |
| Fashion-MNIST | 64 | 1221 | 34.2% | 20.3% | 45.5% | 53.58 | 52.66 |
| CIFAR-10 | 64 | 5531 | 16.7% | 64.9% | 18.5% | 63.35 | 63.33 |

### Why it failed

**CIFAR-10:** The 16-trit key from 192 summary trits produces d_route ≈ 63/64 for BOTH correct and wrong prototypes. Almost every table disagrees on the key — the spatial summary is too coarse for 10-class RGB classification. At 4×4 block pooling on 32×32×3 images, the summary trit is a majority vote over 16 RGB pixels. This destroys the texture information that discriminates CIFAR classes. The routing footprint collapses to noise because all keys are different.

**MNIST:** The Hamming-nearest wrong prototype is also routing-closer (37.3 < 40.8 tables). Routing and Hamming rank prototypes THE SAME WAY — the routing footprint is a coarsened proxy for Hamming distance, not an independent signal. This is exactly what the radiusaware failure predicted (reflect concern #1).

**Fashion-MNIST:** Closest to independent signal (34.2% vs 45.5%, 0.9-table gap in d_route). But still below 50% — the routing footprint does not reliably discriminate when Hamming fails.

### What the prediction missed

The SYNTHESIZE predicted 60-70% discrimination. The actual is 17-34%. The error: I assumed the routing footprint would be independent of Hamming because it operates on a different representation (16-trit summary key vs full signature). In reality, the 16-trit key is a COARSENING of the full signature — images with similar full signatures also tend to have similar summary keys. The coarsening loses information without adding any new signal.

The Tier 1 routing distance is not independent of Hamming. It's a lossy compression of Hamming. The information flows one way: full signature → summary key. There's no information in the key that isn't already in the full signature.

### What this means for the routing-footprint thesis

The strongest version of the thesis ("routing footprint is the primary metric") is falsified. The routing decisions (which bucket in which table) are dominated by the same similarity that Hamming already measures.

However, this only tests Tier 1 (co-bucket membership). Tier 2 (probe-depth vectors) and Tier 3 (co-candidate overlap) were not tested. The REFLECT concern was correct: Tier 1 is a coarsening of Hamming, not independent of it. Tiers 2 and 3 may carry different information — but the Tier 1 result is discouraging enough that the burden of proof has shifted.

### What's left

The path forward is NOT routing-topology metrics. It's trit-level improvements:

1. **Block-encoded pattern distance.** The diagnosed gap from the SSTT comparison. 3-trit blocks (27 symbols) with pattern-aware distance. This operates at the trit level, not the routing level — it's a better distance function, not a different metric space.

2. **Sparsity-aware distance (Experiment 4 from SYNTHESIZE).** Shared zeros as positive evidence. This is also trit-level and can be tested independently.

3. **Wider summary keys.** The 16-trit key is too coarse. A wider key (32 or 64 trits) would give the routing footprint more discrimination — but this is an engineering improvement to the filter, not a new metric for scoring.

### LMM verdict

The LMM cycle worked correctly: it made a specific prediction (60-70%), ran the experiment, and measured the answer (17-34%). The prediction was wrong by 2×. The reflect phase identified the correct concern (radiusaware precedent, concern #1) but I overweighted the counter-argument. The data is clear: Tier 1 co-bucket routing distance is a lossy compression of Hamming distance, not an independent signal.

Experiments 2-3 (routing-only k-NN, combined scoring) are not worth running — the gate experiment (Experiment 1) failed below the 50% threshold on all three datasets. Experiment 4 (sparsity-aware distance) remains testable as an independent trit-level hypothesis.
