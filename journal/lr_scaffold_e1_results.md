---
date: 2026-04-21
scope: E1 measurement from lr_scaffold LMM cycle — class-centroid CSA on MNIST / Fashion-MNIST / CIFAR-10
phase: measurement
---

# E1 results: class-centroid CSA

Experiment E1 from `lr_scaffold_synthesize.md`: implement a Class-Signature Argmax classifier with one ternary prototype per class, trained by integer centroid + sign-threshold, scored via `m4t_popcount_dist` argmin.

Implementation: `tools/csa_classifier.c`, default build (rule-compliant, no random projections).

## Configuration

- Three datasets, pipeline matches `direct_lsh` prior to the scoring stage (deskew/normalize/gradients per dataset).
- MNIST: deskew on, normalize on, density 0.10, no gradients.
- Fashion-MNIST: deskew off, normalize on, density 0.395, gradients on.
- CIFAR-10: deskew off, normalize on, density 0.395, gradients on.
- Two training modes: `centroid` (one pass, sign-threshold with margin) and `centroid+perceptron` (centroid init then bounded-epoch integer perceptron updates).
- Margin swept from 0.00 to 0.50.

## Headline numbers

| Dataset | Best CSA (config) | direct_lsh Hamming k-NN | direct_lsh Selective | Gap vs k-NN |
|---|---|---|---|---|
| MNIST | **82.96%** (perceptron 5ep, m=0.20) | 96.82% | 97.02% | −13.86pp |
| Fashion-MNIST | **65.56%** (centroid, m=0.50) | 87.78% | 87.95% | −22.22pp |
| CIFAR-10 | **29.25%** (centroid, m=0.00) | 44.68% | 46.63% | −15.43pp |

Inference time: 0.1–0.7 μs/query across datasets — roughly 1000–3000× faster than `direct_lsh`'s probe+union+resolve path (1–2 ms/query).

## Perceptron behavior

| Dataset | Centroid | Ep 1 | Ep 3 | Ep 5 | Ep 10 |
|---|---|---|---|---|---|
| MNIST (m=0.20) | 80.80% | 81.09% | 82.92% | **82.96%** | 74.86% (overfits) |
| Fashion-MNIST (m=0.50) | 65.56% | training-error 34.45% → oscillates to 41.80% → test_acc 43.89% | — | — | — |
| CIFAR-10 (m=0.20) | 27.41% | 72.68% train err → oscillates to 82.16% | — | — | **diverges** |

Perceptron helps MNIST slightly and diverges on Fashion-MNIST / CIFAR-10. The single-sample ±1 update is drowned out by the centroid's thousands-of-samples baseline magnitude on high-variance data — margin needs to be large enough to require many samples to flip, but large margins kill perceptron's expressivity. No margin/epoch combination recovers direct_lsh's k-NN baseline.

## Bug surfaced during the experiment

Initial MNIST run produced 10.51% accuracy because `glyph_sig_quantize` on unsigned-pixel MNIST produces signatures in `{+1, 0}` only (no −1). Class centroids then converge to near-identical "~70% +1 / ~30% 0" vectors and all queries collide on class 9. Fix: `--normalize` centers the pixel values around zero so quantization produces real ternary `{+1, 0, −1}` signatures. direct_lsh's Hamming k-NN is robust to this quirk because it measures distance to specific training samples; CSA's centroid averaging is not. Documented for future reference — any centroid-style consumer on MNIST must normalize first.

## What the synthesis predicted vs what happened

Synthesis claim (from `lr_scaffold_synthesize.md`):
> Success: CSA matches Hamming k-NN within 0.3pp on at least one dataset.
> Unambiguous win: CSA beats Hamming k-NN on any dataset.
> Unambiguous loss: CSA is >2pp below Hamming k-NN on MNIST.

Outcome: CSA is 13.86pp below Hamming k-NN on MNIST. **E1 is an unambiguous loss** per the pre-declared gate. The primitive shape is correct (sub-μs inference, SDOT/popcount-native) but one-centroid-per-class is a lossy compression of the training signal in every dataset tested.

## Why E1 failed in a principled way

E1 effectively implements a classical centroid classifier in ternary-quantized signature space. Classical centroid classifiers produce ~80% MNIST accuracy — E1 reproduces this (82.96%). The dataset-level gap is not a substrate artifact; it is the well-known gap between centroid and instance-based classification on high-intra-class-variance data.

The LMM reflection identified the substrate's per-class prototype shape as the "base-3 answer" to LR's scoring problem. That identification was correct at the kernel-primitive level. What the reflection missed was that **prototype training rule determines accuracy more than prototype shape**. Integer centroid-then-sign is the simplest rule; it is demonstrably too coarse.

## Remaining branch points

Three options remain live after E1:

**R1: multi-prototype CSA (CSA-k).** k prototypes per class via clustering in trit space (k-medoids) or random subset sampling. At k=1 recovers E1. As k→n_train/N_CLASSES converges to Hamming k-NN. At intermediate k (e.g., k=16 or 64), inference is 10·k SDOT calls per query — still fast, may recover accuracy. Cheap to try (adds an outer loop over k prototypes per class in the predict step).

**R2: quantized LR training (the original scaffold).** Train float LR externally or at startup, quantize weights to ternary, deploy. Introduces float at training time (new §12 exception or offload to Python outside repo). Matches classical "ternary transformer" training recipes. Most likely to beat centroid training.

**R3: stop and accept negative finding.** CSA as a standalone classifier is ~15–22pp behind k-NN; not production-viable. Close the LMM cycle with this result documented. The ~1000× speedup would be useful only if accuracy were competitive.

R1 is cheapest and most substrate-aligned. R2 is more powerful but crosses the "no gradient descent training inside the C repo" line. R3 is the clean-stop option.

Independent observation: the pre-commit gate measurement (oracle-over-union ≈ 100% on all three datasets) is the more thesis-relevant finding of this cycle. It falsifies the `cifar10_nproj_ceiling.md` claim that CIFAR-10 is bottlenecked by representation — the candidate union contains the correct class on 99.99% of queries; the scoring stage extracts only 44.68% of it. **The CIFAR-10 gap is a scoring problem, not a signature problem.** That insight survives even if CSA itself is abandoned.
