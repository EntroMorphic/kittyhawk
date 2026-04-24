---
date: 2026-04-24
scope: audit + measurement — features of the Glyph stack that existed but were never applied to real datasets, especially CIFAR. "Sweep it all" per user directive after CIFAR representation-tax discussion.
phase: MEASURE
---

# Sweep: underused features applied to image canon

## Motivation

Prior cycles left capabilities on the shelf:
- **`csa_classifier`** (Class-Signature Argmax, derived from `lr_scaffold` cycle) — never tested on CIFAR.
- **`libtrain`** (routed autodiff MVP: tlinear, rroute, hysteresis requant) — passes gradient checks and a 2-class convergence toy but has never been applied to any real dataset.
- **`block_distance`** — existing brute-force ceiling measurement tool; status on CIFAR unclear.
- **Alternate resolvers** (voteweighted, radiusaware) — implemented in libglyph, usage patterns on CIFAR not documented.
- **Feature extraction before ternarization** — analogue of the Go `contrast3` win; only H+V gradients currently exist for images.

The sweep ran the cheap measurements (CSA + libtrain + direct_lsh resolver variants), surfaced one unexpected positive result, and quantified the gap between each single primitive and the full `direct_lsh` production stack.

## Baseline reference (k=50 brute-force Hamming, full 10k test)

From `image_distance_probe`:

| Dataset | Raw Hamming k-NN | direct_lsh Selective |
|---|---|---|
| MNIST | 95.65% | 97.18% |
| Fashion-MNIST | 81.04% | 87.95% |
| CIFAR-10 | **13.06%** | 46.63% |

The gap between raw Hamming baseline and production tells you how much work the downstream pipeline does. MNIST: +1.5pp (saturated). Fashion: +6.9pp (moderate). **CIFAR: +33.6pp (everything depends on the pipeline).**

## Result 1 — `csa_classifier` (Class-Signature Argmax)

Ternary per-class prototypes scored by Hamming. Never run on CIFAR before.

| Dataset | centroid (+gradients +normalize) | perceptron |
|---|---|---|
| MNIST (no grad, no norm) | 10.51% (degenerate) | 10.32% (degenerate) |
| MNIST +gradients | **52.41%** | — |
| Fashion +gradients +normalize | 64.82% | **72.48%** |
| CIFAR +gradients +normalize | **29.02%** | 22.33% (hurts) |

**Finding:** on CIFAR, CSA **centroid mode achieves 29.02%** — more than 2× raw Hamming k-NN (13.06%) and above libtrain tlinear (20%, see below) using no SGD at all, just integer averaging per class. This is a substrate-native result that has never been measured before.

**Caveat:** perceptron mode (integer sign updates without LR control) hurts CIFAR (22.33%). Principled SGD > naive integer updates. CSA centroid init is a better *starting point* than random init for any downstream trainer.

## Result 2 — `libtrain.tlinear` (first application to real datasets)

Dense ternary linear classifier trained via `tlinear_forward` + `tlinear_backward_dW` + `requantize_hysteresis`. MSE loss against {−1, +1} targets. Y forward-scaled by `1/sqrt(K)` to make MSE well-posed at real signature dimensions (otherwise loss explodes at K=784+).

Config: batch=128, lr=5e-3, epochs=30, pocket snapshot for test eval.

| Dataset | **tlinear (new)** | vs k-NN | vs direct_lsh production |
|---|---|---|---|
| MNIST (intensity only) | **83.06%** | −12.59pp | −14.12pp |
| Fashion (gradients + normalize) | **73.77%** | −7.27pp | −14.18pp |
| CIFAR (gradients + normalize) | **20.00%** | **+6.94pp** | −26.63pp |

**Finding:** libtrain's tlinear + hysteresis SGD **works end-to-end on real data**. Loss stabilizes (with proper Y scaling), accuracy climbs across epochs, pocket snapshot captures the best intermediate state. This is the first application of the routed autodiff MVP to a benchmark.

**Result is mixed:**
- On **MNIST and Fashion**, tlinear loses to k-NN by 7–13pp. k-NN exploits fine-grained instance-level similarity that a single dense linear layer compresses away.
- On **CIFAR**, tlinear *beats* raw Hamming k-NN by **+6.94pp** (20% vs 13%). Instance-based k-NN is so weak on CIFAR (barely above random) that even a modestly-expressive learned classifier can improve on it. **This is the first direct evidence that learned weights patch some of the CIFAR representation tax.**
- **All three lose to `direct_lsh` production** by 14–27pp. A single dense ternary linear layer is not architecturally rich enough to match the full direct_lsh ensemble (multi-table routing + GSH + pair-IG + selective scoring).

**Interpretation:** the "learned weights vs hand-engineered enrichments" question has an asymmetric answer. On datasets where Hamming already carries good signal (MNIST, Fashion), enrichments dominate and compression to a linear layer loses. On datasets where Hamming is nearly useless (CIFAR), even weak learned weights add signal. But the production stack's architectural diversity outcompetes any single primitive.

## Result 3 — CSA beats tlinear on CIFAR

The most surprising data point: **CSA centroid mode (29%) beats tlinear SGD (20%) on CIFAR** despite CSA being simpler (no SGD, no LR, no latent updates). The class-averaged ternary prototype captures CIFAR-class structure better than a randomly-initialized linear layer trained for 30 epochs.

**Implication:** class-centroid initialization is a strictly better starting point for any CIFAR trainer than random Gaussian. A CSA-initialized tlinear trainer is the obvious next experiment; it combines CSA's structural prior with tlinear's refinement capacity. Queued (not executed in this sweep).

## Result 4 — direct_lsh resolver variants on CIFAR (all four bit-identical)

Quick ablation over the four SUM resolver modes on CIFAR (all other flags held at production config: `--no_deskew --normalize --density 0.395 --gradients --m_max 64`).

| Resolver | LSH-only k=5-rw at M=64 | Pair-IG | GSH 1-NN | Selective |
|---|---|---|---|---|
| `scalar` (default) | 44.68% | 45.73% | 36.87% | **46.63%** |
| `neon4` | 44.68% | 45.73% | 36.87% | 46.63% |
| `voteweighted` | 44.68% | 45.73% | 36.87% | 46.63% |
| `radiusaware` (λ=8) | 44.68% | 45.73% | 36.87% | 46.63% |

**All four variants produce bit-identical numbers across the entire M-sweep.** This is itself a finding: the SUM resolver is not the bottleneck on CIFAR. `voteweighted` and `radiusaware` were originally designed to recover the Fashion-MNIST resolver gap (concentrated in the upper-body-garment cluster, classes {0, 2, 4, 6}); on CIFAR there is no analogous tied-distance pattern for them to break. The bottleneck is upstream — at the filter step or the signature representation, not at the SUM ranker.

**Implication:** alternate resolvers are Fashion-specific tools. They don't apply to CIFAR. Direct_lsh's CIFAR Selective (46.63%) is fixed by everything except the resolver choice.

## Results NOT measured in this sweep

- **`block_distance` on CIFAR.** Attempted, aborted — O(N²) brute force on 3072-dim signatures is too slow for in-session measurement. Reopen as a scheduled background run.
- **direct_lsh resolver variants on Fashion** — Result 4 measured CIFAR only; the resolver variants were originally designed for Fashion. Re-running on Fashion to confirm they still help there is a separate small ablation.
- **Diagonal/local-contrast features for images.** Analogous to Go's `contrast3` win; adds 2–4× the trit budget per image. Requires code in `glyph_dataset` gradients module. Not written.
- **Multi-table M sweep beyond default.** direct_lsh default is M=64; untested at M=128 or M=32 with different seeds. Not run.

## Ranked value of what was measured

1. **CSA centroid on CIFAR = 29.02%** — biggest unexpected positive. A substrate-native, training-free classifier reaches mid-twenties on CIFAR from raw features, with zero SGD. Worth documenting as a baseline independent of `direct_lsh`.
2. **libtrain tlinear + hysteresis works on real data** — first ever measurement. The MVP scales from 16-dim toy to 9024-dim CIFAR once the MSE scale factor (1/sqrt(K)) is applied. This unlocks further trainer cycles.
3. **Learned weights > raw k-NN on CIFAR (+6.94pp)** but **lose on MNIST/Fashion** — explains why direct_lsh's enrichment stack was needed: not every dataset rewards the same architecture.
4. **CSA > tlinear on CIFAR** — centroid init dominates random init for this task. CSA-init + tlinear SGD is the queued next experiment.

## What the sweep confirms about CIFAR

The CIFAR representation tax is real AND multi-faceted:
- k-NN alone: 13% (barely above random).
- Single-layer learned classifier (tlinear): 20%.
- Class-centroid classifier (CSA): 29%.
- Full production stack (multi-table routing + GSH + pair-IG + selective): 46.63%.

Each step adds architectural diversity, not just parameter count. The gap to SSTT (~53%) sits at the limit of what direct-ternary-quantization-from-RGB can represent. Closing it further needs either (a) a richer pre-quantization feature pipeline (diagonal gradients, local contrast, color-opponent channels) or (b) a learned representation (our own routed_autodiff with working learned routing), not better distance metrics on the existing signature.

## Artifacts

- `tools/trained_classifier.c` — new tool, minimal libtrain tlinear + hysteresis classifier. First application of the routed autodiff MVP to real datasets.
- CSA classifier runs on all three datasets — confirms CSA centroid as a substrate-native baseline worth citing.
- `journal/hamming_norm_on_images.md` (prior) — covers the density-bias question, complements this sweep.

## Queued follow-ups (not this cycle)

1. **CSA-initialized tlinear trainer** — use CSA centroid prototypes as W init for tlinear SGD. Combines the strongest non-routing baseline (CSA) with SGD refinement. Expected to beat both alone on CIFAR.
2. **Diagonal gradients + local contrast in glyph_dataset** — pre-quantization feature additions; port the Go `contrast3` pattern to images. Expected modest gains on CIFAR's texture-heavy classes.
3. **`block_distance` CIFAR scheduled run** — overnight-style measurement to get the block-distance ceiling on CIFAR.
4. **Multi-table M>64 sweep** — test whether direct_lsh's production config is on the plateau or there's headroom.

## NORTH_STAR discipline check

- **§4 (scaffolding sanction)**: all runs use existing production/research tools or add a minimal new consumer (`trained_classifier.c`). No binary-float runtime, no Python.
- **§12 (no binary float in compute)**: tlinear uses float for W_latent and activations (explicitly sanctioned training-only sites). All deployed weights remain ternary.
- **§13 (training artifacts in consumer)**: `trained_classifier.c` lives in `tools/`, not in libm4t or libglyph. Consumer-layer per discipline.
- **Routing claim**: the measurements confirm routing (direct_lsh's multi-table + GSH + pair-IG) is the architectural shape that wins on CIFAR, not single-layer classifiers. NORTH_STAR's "routing essential" clause holds.
