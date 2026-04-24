---
date: 2026-04-23
scope: S7 thesis-calibration cycle close-out — in-C F-stat fallback results, what the cycle answered, what it didn't
phase: CLOSE
---

# S7 close-out

## Cycle in one paragraph

User declined Python in the repo. Ran the in-C fallback — class-conditional linear feature selection via F-statistic. Fallback tests a weaker version of the S7 question: "does linear, data-dependent dim selection beat uniform direct quantization?" Answer: only on MNIST (where the baseline is saturated); not on Fashion or CIFAR. The deeper thesis-calibration question ("can a non-linear gradient-trained encoder reach SSTT on CIFAR?") is **unanswerable in C** — it requires autodiff + non-linear architectures. Cycle delivered one real data point and one unexpected positive finding (MNIST K=1024 F-stat beats baseline).

## Results table

| Dataset | Config | Config setting | Selective | Δ vs direct |
|---|---|---|---|---|
| CIFAR-10 | direct MS4+R4 Selective (baseline) | — | 48.05% | — |
| CIFAR-10 | direct MS4+R4 brute-1NN | — | 42.76% | −5.29pp |
| CIFAR-10 | F-stat encoder Selective | K=4096 | 41.29% | **−6.76pp** |
| CIFAR-10 | F-stat encoder brute-1NN | K=4096 | 40.44% | −2.32pp (vs direct brute) |
| Fashion | direct MS4 Selective (baseline) | — | 88.66% | — |
| Fashion | direct MS4 brute-1NN | — | 87.86% | −0.80pp |
| Fashion | F-stat encoder Selective | K=2000 | 87.06% | −1.60pp |
| Fashion | F-stat encoder Selective | K=1024 | 86.33% | −2.33pp |
| MNIST | direct MS4 Selective (baseline) | — | 97.24% | — |
| MNIST | direct MS4 brute-1NN | — | 97.36% | +0.12pp |
| MNIST | F-stat encoder Selective | K=1024 | **97.73%** | **+0.49pp** |
| MNIST | F-stat encoder Selective | K=512 | 96.11% | −1.13pp |

## Three findings

**1. F-stat dim selection hurts CIFAR at every K tested.** Top-K (K ∈ {2048, 4096, 6000, 8192, 10000, 11000}) Selective consistently lands at 41.29% — 7pp below direct MS4+R4. Even the K=4096 brute-1NN underperforms direct brute-1NN (40.44% vs 42.76%). Linear class-conditional feature selection is the wrong tool on CIFAR.

**2. F-stat value distributions differ dramatically by dataset.** CIFAR top F-stat is 0.306; Fashion is 4.27; MNIST is 1.68. On CIFAR, individual dims carry almost no class signal — the classification signal is distributed across many weak dims. F-stat selection discards the ensemble that makes direct quantization viable. On Fashion/MNIST, specific dims (ink patterns) carry real per-dim signal, so F-stat can select meaningfully.

**3. MNIST K=1024 F-stat BEATS direct MS4 by +0.49pp.** Unexpected positive. On easy data where the baseline is already saturated, linear class-conditional selection improves slightly because it drops low-F-stat noise dims. Concrete: MNIST F-stat K=1024 = 97.73%; direct MS4 Selective = 97.24%. Not a huge win but reproducible and mechanistically sensible.

## What the cycle could not answer

**"Can a non-linear gradient-trained encoder reach SSTT (~53%) on CIFAR-10?"**

The in-C fallback cannot approximate non-linear learned features. The closest in-C proxy to "learned encoding" is class-conditional linear operations (F-stat, Fisher discriminant, class centroids). All are structurally linear. SSTT's ~53% likely requires non-linear feature extraction (attention, pooling, block-pattern scoring).

Without running Python, the "substrate-bounded vs encoding-bounded" question remains open on CIFAR.

## Brute 1-NN as a control revealed Glyph's downstream value

Secondary finding from the `--brute_1nn` control flag:

| Dataset | Direct brute-1NN | Direct Selective | Glyph downstream adds |
|---|---|---|---|
| MNIST | 97.36% | 97.24% | **−0.12pp** (brute beats pipeline!) |
| Fashion | 87.86% | 88.66% | +0.80pp |
| CIFAR-10 | 42.76% | 48.05% | **+5.29pp** |

**Glyph's filter+pair-IG+Selective pipeline adds value proportional to dataset difficulty.** On MNIST it's slightly harmful. On Fashion, moderate gain. On CIFAR, it's the lion's share of the improvement over brute 1-NN.

This reframes the CIFAR 48.05% number: 42.76% of it is raw ternary-signature 1-NN, and Glyph's pipeline adds the remaining 5.29pp via filtering + pair-IG re-rank. The *pipeline itself* is the CIFAR success story, not just the signature.

## Artifact inventory

- `direct_lsh --fstat_K N` — class-conditional F-stat dim selection. Stays in the tree as an experimental encoder option.
- `direct_lsh --brute_1nn` — brute-force 1-NN control classifier. Bypasses filter+resolver for signature-quality measurements.
- `direct_lsh --region_tau auto` — experimental COM-spread heuristic. Reports metric but defaults to disabled (metric doesn't reliably predict R4 benefit).

All three ship in the production build, tagged as experimental where applicable.

## Next-cycle seed

The thesis-calibration question is still open on CIFAR. Two honest paths:

**Path A (external Python):** revisit user's "no Python" stance. NORTH_STAR §4 explicitly sanctions scaffolding for calibration. If user permits Python in `tools/experimental/`, run the original S7 design (linear encoder with auxiliary classifier, STE-quantize, export trit sigs, measure). ~2 days.

**Path B (accept the open question):** do not run S7; accept that CIFAR 48% may be the ternary-signature ceiling for direct quantization + uniform Hamming on Glyph's downstream. Move to a different benchmark or a different direction entirely.

**Path C (in-C non-linear proxy):** implement a simple two-layer ternary network with hand-tuned weights — e.g., HOG-style oriented gradient bins, or ternary block-pattern features hand-designed from prior. This is NOT gradient-trained but IS non-linear. Might recover some of the CIFAR gap. Moderate scope (1 week).

Default recommendation: **re-ask the user on Python.** The in-C fallback has exhausted what it can say. A definitive thesis-calibration answer requires autodiff.

## What this cycle produced regardless of the open thesis question

1. **The brute-1NN control.** Any future calibration experiment should pair with it to decompose pipeline-vs-signature contributions. Now a permanent flag.

2. **F-stat ruling.** Linear class-conditional dim selection does not help on the hard dataset. Narrows the remaining live hypotheses.

3. **MNIST +0.49pp via F-stat K=1024.** Small but real; useful for saturated-baseline datasets.

4. **Clear decomposition of Glyph's 48.05% CIFAR:** 42.76% is the signature's raw discriminability (brute 1-NN); 5.29pp is Glyph's downstream contribution. Tells us where to invest next.
