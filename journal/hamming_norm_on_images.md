---
date: 2026-04-24
scope: measurement — does `hamming_norm` retrofit into image pipelines, or stay Go-specific?
phase: MEASURE
---

# Measurement: `hamming_norm` on MNIST / Fashion-MNIST / CIFAR-10

## Setup

Standalone tool `tools/image_distance_probe.c`. Direct ternary quantization via `glyph_sig_quantize` (same as `direct_lsh`), brute-force k-NN classification under both `hamming` and `hamming_norm`, full test set (10k queries per dataset). No gradients, no multi-scale, no GSH — raw intensity trits only, so the measurement isolates the distance metric from the downstream enrichments.

Per-dataset config mirrors `direct_lsh` defaults: MNIST density 0.10 with deskew, Fashion / CIFAR density 0.33 with `--no_deskew --normalize`.

## Results (full 10k test, k=50)

| Dataset | `hamming` | `hamming_norm` | Δ | Verdict |
|---|---|---|---|---|
| MNIST | 95.65% | 95.49% | −0.16pp | tied (within noise) |
| Fashion-MNIST | **81.04%** | 78.39% | **−2.65pp** | **hurts** |
| CIFAR-10 | 13.06% | 14.04% | +0.98pp | marginal gain (3× noise floor) |
| Go phase-ID (reference, from substrate_distance_refinement) | 40.40% | **88.40%** | **+48.00pp** | **large gain** |

Absolute image numbers are low because this is the *baseline* substrate pipeline without gradients, MS4, pair-IG rerank, or GSH — comparable to the raw-ternary-Go-positions test, not to `direct_lsh` production numbers (which reach 97.18% / 87.95% / 46.63% via those enrichments).

## Per-class density variance (diagnostic)

Mean trit density per class on train (raw `hamming` quantization):

| Dataset | Range | Mean | Range / mean |
|---|---|---|---|
| MNIST (sig=784) | 85.8 (class 1) to 192.0 (class 0) | 149.9 | **71%** |
| Fashion (sig=784) | 246.0 (class 5 sandal) to 508.6 (class 2 pullover) | 388.4 | **68%** |
| CIFAR (sig=3072) | 1012.7 to 1018.5 | 1016.0 | **0.6%** |
| Go (sig=361) | 5 (opening) to 250 (endgame) | ~125 | **~200%** |

CIFAR's density uniformity after `--normalize` is striking: every class sits within ±0.6% of the mean. The normalization pass already equalizes total signal across classes; there is no density variance left for `hamming_norm` to correct.

MNIST and Fashion have large density variance (70%) but `hamming_norm` doesn't help on either. The reason, given both findings together:

**`hamming_norm` is a metric remedy, not an enhancement.** It fixes raw Hamming when the sparse-vector attractor dominates positional discrimination. For Go (density varies 50× across game phases on a small 361-trit signature), the attractor dominates → `hamming_norm` recovers +48pp. For MNIST/Fashion/CIFAR, positional structure dominates density variance — raw Hamming already works correctly — so normalization just adds noise.

Why doesn't Fashion's large density variance (70%) produce the same Go-like attractor problem? Because on images, **density correlates with class in informative ways**. A class-5 sandal query finds other class-5 sandals at low Hamming partly *because* they share similar density AND similar positions. That's real signal, not an artifact. `hamming_norm` removes that useful signal.

On Go, adjacent-phase positions share density by construction (move count → density is monotone), so the density signal is class-uninformative at the phase level — it's the positional similarity *within* a density band that would carry phase-specific information. Raw Hamming can't see that structure through the density gradient.

## Decision

**Do not retrofit `hamming_norm` into `direct_lsh` or any image consumer.** The measurement shows it hurts Fashion-MNIST by 2.65pp and is neutral or marginal on MNIST/CIFAR.

**Keep `hamming_norm` as a Go-specific distance** in `go_probe` and future `routed_go` trainer work. For sparse-discrete domains with large class-uninformative density variance (Go positions, DNA sequences, edit tag vectors, ternary-coded survey responses), `hamming_norm` is the correct substrate distance primitive.

**Retrospective on `step_change` cycle**: the CIFAR representation tax measured there (46.63% vs ~53% SSTT) is a **real representation tax, not a metric artifact**. `hamming_norm` does not meaningfully close it (CIFAR delta +0.98pp on raw intensity; would not measurably change the direct_lsh production number). The demotion of image canon from primary benchmark to regression-guard (see `journal/base3_benchmarks_closeout.md`) is confirmed.

## Mechanism summary

Three conditions must simultaneously hold for `hamming_norm` to improve over raw Hamming:

1. **Class-uninformative density variance.** If density correlates with class in useful ways (as on images), raw Hamming's density sensitivity is load-bearing.
2. **Large density variance relative to signature size.** Go's 5–250 density on 361 trits is an order of magnitude more extreme than Fashion's 246–508 on 784 trits, even though the fractional range is similar.
3. **No prior density decorrelation in the pipeline.** Image pipelines include `--normalize` (per-image zero-mean unit-variance in integer arithmetic), which equalizes class-conditional signal strength upstream of quantization. Go has no analogous step.

Domains that meet all three: raw ternary positional states (Go, Chess, Othello, edit sequences, DNA mutation maps), where variable sparsity carries phase-like information but doesn't discriminate between classes within a phase.

## Artifacts

- `tools/image_distance_probe.c` — standalone measurement tool. Links libglyph (for dataset + sig + popcount_dist), no bucket/GSH/pair-IG dependencies.
- Results recorded above. Raw full-test-set k=50 numbers are deterministic for a given τ calibration.

## Tasks

- #40 (this measurement): **COMPLETE**. Decisive negative: no retrofit.
- #41 (routed_go trainer): **UNCHANGED** — `hamming_norm` is still the right distance for Go; the trainer cycle can proceed whenever user prioritizes it.
- No new task needed. The substrate-wide claim made in `substrate_distance_refinement_closeout.md` is explicitly rescinded in favor of the domain-specific framing above.
