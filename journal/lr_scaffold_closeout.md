---
date: 2026-04-21
scope: lr_scaffold LMM cycle close-out — what the cycle delivered, what it didn't, and what the next cycle should be
phase: CLOSE
---

# Close-out: the lr_scaffold LMM cycle

## What the cycle was asked to do

Evaluate whether logistic regression benefits Glyph. Produce a falsifiable experiment and run it.

## What the cycle actually delivered

Three things, only one of which was the experiment it was framed around.

**1. A reframe.** LR is a misnamed operation at the substrate level; the base-3 native shape for the per-class-scoring problem is a ternary prototype per class, scored via `popcount_dist` (or SDOT), argmax/argmin over class scores. Named Class-Signature Argmax (CSA). This survived.

**2. A measurement that matters across the whole project.** While preparing CSA, I surfaced the oracle-over-union numbers from `direct_lsh`'s existing sweep output — they had always been there but nobody had read them as a diagnosis:

| Dataset | Oracle @ M=64 | Hamming k-NN | Gap |
|---|---|---|---|
| MNIST | 100.00% | 96.82% | 3.18pp |
| Fashion-MNIST | 99.99% | 87.78% | 12.21pp |
| CIFAR-10 | 99.99% | 44.68% | 55.31pp |

On first read this appeared to be a scorer problem. The follow-up experiments force the correct reading: "correct class is in the union" just means one correct-class sample exists somewhere among ~1,600 candidates — it doesn't place the correct sample near the top under Hamming distance. The 55pp on CIFAR-10 is not reachable by any uniform-Hamming scorer. It's a Hamming-geometry separability limit on the direct-quantized signature.

**3. A new production-useful consumer.** `tools/csa_classifier.c`, rule-compliant, default build. Best configurations:

| Dataset | CSA-k best | Hamming k-NN | Gap | Speedup |
|---|---|---|---|---|
| MNIST | 95.64% (k=1024 top_k=5) | 96.82% | −1.18pp | ~20× |
| Fashion-MNIST | 86.29% (k=4096 top_k=1) | 87.78% | −1.49pp | ~100× |
| CIFAR-10 | 43.34% (k=4096 top_k=10) | 44.68% | −1.34pp | ~22× |

CSA-k at high k converges structurally toward Hamming k-NN (it IS Hamming k-NN restricted to a class-balanced subset). Its contribution is architectural simplicity, not accuracy: no bucket, no probe, no resolver.

## What the cycle tried and did not deliver

E1 (single-centroid CSA) and the 64-epoch perceptron refinement both regress below Hamming k-NN on every dataset:

| Dataset | Centroid | 64-ep perceptron pocket | Hamming k-NN |
|---|---|---|---|
| MNIST | 80.01% | 84.90% (epoch 19) | 96.82% |
| Fashion-MNIST | 65.56% | 76.13% (epoch 46) | 87.78% |
| CIFAR-10 | 29.25% | 31.63% (epoch 52) | 44.68% |

Integer perceptron with batch updates + end-of-epoch re-sign **does not converge**. On MNIST and CIFAR-10 it oscillates; on Fashion-MNIST it tracks meaningfully upward for ~46 epochs then diverges. Best-epoch numbers are pocket snapshots, not convergent training.

## The single diagnosis that unifies the numbers

**Uniform Hamming distance on direct-quantized signatures is the ceiling, regardless of classifier architecture.**

- CSA (centroid) and CSA-k (subset k-NN) both use uniform Hamming → at best they recover most of the Hamming k-NN accuracy, never beat it.
- 64-ep perceptron operates on uniform Hamming argmin → oscillates around the centroid baseline.
- Only pair-IG (per-dim-per-class-pair weighted Hamming) beats baseline, and only by +1.95pp on CIFAR-10.

The 55pp CIFAR-10 headroom is locked behind the distance function, not the classifier. `journal/cifar10_nproj_ceiling.md`'s original "representational" framing was correct; the scoring-ceiling reframe I attempted is wrong once the measurement is read precisely.

## Next-cycle seed

The immediate next LMM cycle should be on **distance-function design for direct ternary signatures**, not classifier architecture. Concrete starter questions:

- What's the cheapest non-uniform Hamming that beats pair-IG on CIFAR-10? Candidates: global per-dim weights (single vector, no per-class-pair matrix), block-level distance (SSTT-style pattern matching at 4×4 spatial or similar), learned Hamming mask.
- Is there a base-3-native pattern-level distance primitive latent in the substrate (similar to how CSA was latent via SDOT)? Worth a substrate-surface audit: TBL for pattern dispatch, masked-VCNT for block-level agreement counting, `m4t_trit_mul` for trit-trit pattern scoring.
- Does moving from "per-trit independence" distance to "per-block correlation" distance recover some of the 55pp oracle gap on CIFAR-10? `docs/FINDINGS.md` Axis 8 already names this as the active frontier.

The cycle name would be something like `distance_function` or `block_distance_design` (noting `tools/block_distance.c` already exists as an analysis harness — the cycle could reuse it as its measurement bed).

## What to commit from this cycle

- `tools/csa_classifier.c` — rule-compliant, default-build, passing all tests. Production-useful as a fast-inference consumer (1.3pp accuracy cost for 20–100× speedup).
- `CMakeLists.txt` — csa_classifier registered as a production tool.
- `journal/lr_scaffold_*.md` — six artifacts (raw, nodes, reflect, synthesize, e1_results, e1_r1_final, closeout). The close-out is the canonical reading; earlier artifacts show the reasoning path.

## What not to commit

Nothing from the cycle deletes, archives, or breaks existing code. Zero changes to libglyph, libm4t, direct_lsh, or any other consumer.

## One-sentence cycle verdict

The cycle found a substrate-native primitive (CSA), shipped it as a useful fast-inference consumer (CSA-k), and surfaced the real lever for CIFAR-10 accuracy (distance-function design) — but did not itself produce a classifier that beats the existing Hamming k-NN baseline, because the limit isn't the classifier.
