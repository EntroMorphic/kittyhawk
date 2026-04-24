---
date: 2026-04-21
scope: E1 + R1 measurement from lr_scaffold LMM cycle — Class-Signature Argmax and its multi-prototype extension on MNIST, Fashion-MNIST, CIFAR-10
phase: measurement
---

# E1 + R1 results: Class-Signature Argmax (CSA) and CSA-k

Implementation: `tools/csa_classifier.c`, default build (rule-compliant, no random projections).
Kernel: `m4t_popcount_dist` against per-class ternary prototypes.

## Summary table — best config per dataset

| Dataset | Hamming k-NN (direct_lsh) | Selective (direct_lsh) | **CSA best** | Config | Time/query | vs Hamming |
|---|---|---|---|---|---|---|
| MNIST | 96.82% @ ~1–2 ms | 97.02% @ ~1–2 ms | **95.64%** | k=1024, top_k=5 | **62 μs** | **−1.18pp, ~20× faster** |
| Fashion-MNIST | 87.78% @ ~60 ms | 87.95% @ ~60 ms | **86.29%** | k=4096, top_k=1 | **589 μs** | **−1.49pp, ~100× faster** |
| CIFAR-10 | 44.68% @ ~65 ms | 46.63% @ ~65 ms | **43.34%** | k=4096, top_k=10 | **2943 μs** | **−1.34pp, ~22× faster** |

## Pre-commit gate finding

`direct_lsh` already reports oracle-over-union in its sweep table. Extracting:

| Dataset | Oracle @ M=64 | Hamming k-NN | Scoring headroom |
|---|---|---|---|
| MNIST | 100.00% | 96.82% | +3.18pp |
| Fashion-MNIST | 99.99% | 87.78% | +12.21pp |
| CIFAR-10 | 99.99% | 44.68% | **+55.31pp** |

The CIFAR-10 candidate union almost always *contains* the correct class; the scorer fails to extract it. This contradicts `journal/cifar10_nproj_ceiling.md`'s claim that the 46% vs SSTT's ~53% gap is representational. **The CIFAR-10 gap is a scoring problem, not a signature problem.** This is the most durable finding of the cycle and survives whether CSA ships or not.

## E1: single-prototype CSA (class centroid + sign-threshold)

Sweep over margin ∈ {0.00, 0.05, …, 0.50}, with and without perceptron refinement (1–10 epochs).

Best-per-dataset E1 results:

| Dataset | Best E1 | Config |
|---|---|---|
| MNIST | 82.96% | perceptron 5 epochs, margin=0.20 |
| Fashion-MNIST | 65.56% | centroid, margin=0.50 |
| CIFAR-10 | 29.25% | centroid, margin=0.00 |

Perceptron oscillates and fails to converge on Fashion-MNIST (34% → 42% train error) and CIFAR-10 (73% → 82% train error). The single-sample ±1 update is drowned out by the centroid's O(n_class) baseline magnitude on high-variance data.

**E1 is an unambiguous loss per the synthesis's pre-declared gate** (CSA >2pp below Hamming k-NN on MNIST). The single-prototype compression reproduces the classical centroid-classifier result: ~80% MNIST, ~65% Fashion, ~30% CIFAR-10. The primitive kernel shape is correct; the data-compression is too aggressive.

## R1: multi-prototype CSA-k

k prototypes per class, selected as the first k class-matching training samples (deterministic). Inference: argmin over 10·k prototypes (`top_k=1`) or rank-weighted k-NN majority vote over the top-K closest prototypes (`top_k>1`).

### k sweep (top_k=1)

MNIST:

| k | accuracy | μs/query |
|---|---|---|
| 1 | 80.01% (centroid) | 0.1 |
| 16 | 82.45% | 1.0 |
| 64 | 88.85% | 3.8 |
| 256 | 93.21% | 15.4 |
| 1024 | 95.48% | 61.5 |

Fashion-MNIST:

| k | accuracy | μs/query |
|---|---|---|
| 1 | 65.56% (centroid) | 0.2 |
| 64 | 76.21% | 9.5 |
| 256 | 80.20% | 60.7 |
| 1024 | 83.05% | 233.2 |
| 4096 | 86.29% | 589.2 |

CIFAR-10:

| k | accuracy | μs/query |
|---|---|---|
| 1 | 29.25% (centroid) | 0.7 |
| 64 | 27.22% | 44.8 |
| 256 | 29.88% | 180.0 |
| 1024 | 34.08% | 719.8 |
| 4096 | 40.64% | 3032.4 |

### top_k refinement

top_k>1 applies rank-weighted majority vote over the top-K closest prototypes across all (class, proto) pairs (same scheme as `direct_lsh`'s KNN resolver). Effect:

- **MNIST** (k=1024): top_k=1→95.48%, top_k=5→95.64% (+0.16pp), top_k=10→95.56%. Marginal.
- **CIFAR-10** (k=1024): top_k=1→34.08%, top_k=5→36.57%, top_k=10→**38.12%** (+4.04pp). Substantial.
- **CIFAR-10** (k=4096): top_k=10→**43.34%** (+2.70pp over top_k=1 at same k).

Top-K voting matters more on noisy data: the single closest prototype can be an outlier; averaging top-K class labels smooths the decision. This mirrors the rank-weighted KNN finding in `direct_lsh`'s main sweep (k=5-rw > 1-NN on every dataset).

## Three takeaways

1. **Single-prototype CSA (E1) reproduces the classical centroid-classifier floor.** Not viable as a standalone classifier. Perceptron doesn't rescue high-variance data.

2. **Multi-prototype CSA-k is a real speed/accuracy tradeoff.** On all three datasets it recovers within 1.2–1.5pp of Hamming k-NN at 17–100× the speed. For a latency-constrained deployment this is a genuinely useful primitive.

3. **CSA-k converges structurally to Hamming k-NN as k → n_class.** At full k it IS Hamming k-NN restricted to a class-balanced subset of training. The distinguishing substrate-level feature is not accuracy — it's the absence of the bucket/probe/resolve pipeline: CSA-k is `N_CLASSES × k` popcount_dist calls, no filter stage, no union. Simpler architecture at comparable accuracy.

## What the cycle learned that survives

The LMM reframe ("LR is the wrong name; the substrate-native shape is class-prototype scoring") was correct at the kernel-primitive level. The first experiment (E1) falsified the specific training rule proposed in the synthesis (single centroid + sign). The follow-up experiment (R1) validated the primitive shape at higher k with top-K voting. Both outcomes are the synthesis working as intended — the reframe produced two falsifiable experiments, each of which measured what it claimed to measure.

The deeper structural finding is orthogonal to CSA: **CIFAR-10 is scorer-bottlenecked, not representation-bottlenecked.** Future effort on CIFAR-10 accuracy should target the scoring stage (better per-dim weights, pattern-level distance, or learned discriminators) rather than the signature building stage.

## Remaining branches

**B1: ship CSA-k as a production-ready speed/accuracy tradeoff consumer.** Adds one-shot argument for "fast-inference" deployments where 1.3pp accuracy cost for 20–100× speedup is desirable. Low cost; tool already exists.

**B2: investigate learned per-class ternary weights via quantized LR.** The original scaffolding option from the synthesis. Now that R1 has proven the single-centroid rule is inadequate and the primitive shape is sound, this becomes: "given the prototype architecture works, can we train the 10 prototypes better than `first-k training samples` by actually optimizing for class discrimination?" Crosses the float-training line at build time; closest integer-only alternative is pocket-perceptron (perceptron with best-so-far snapshot).

**B3: close the cycle.** Document the findings, update the LMM artifacts to reflect actual outcomes, leave CSA-k in place as a substrate-exercising consumer without calling it production. Continue CIFAR-10 work on other paths.

**B4: use the oracle-gap finding to redirect.** The 55pp scoring headroom on CIFAR-10 is the most actionable insight. Better CIFAR-10 accuracy probably comes from better per-dim weights applied at the direct_lsh resolver stage (where candidate filtering already works), not from a different classifier architecture. CSA-k would be a side finding; the main thread becomes "what scorer beats pair-IG's +1.95pp on CIFAR-10?"
