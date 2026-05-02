---
title: Phase B Gate 2 — H1 mechanism test (implicit denoising via random ternary projection)
date: 2026-05-02
benchmark: synthetic prototype classification (D=64, K=16, C=10)
status: PASS — H1 supported with Pearson r = +0.892, p << 0.001
---

# Phase B Gate 2 — H1 mechanism test

**Pre-committed gate (per `journal/gesh_findings_synthesize.md`):**
- **PASS (H1 supported):** positive correlation, p < 0.01 across N ≥ 100 random R samples.
- **FAIL (H1 falsified):** zero or negative correlation.
- **INCONCLUSIVE:** weak positive correlation, p ∈ [0.01, 0.1].

**Verdict: PASS.** Pearson r = +0.892, t = 157.89 (df = 6398), p << 0.001.

## What H1 says

The C2 finding (random ternary projection at sig_dim = D beats identity by +7pp) was paired with a hypothesis that the docs initially treated as a finding: "random ternary projection mixes the 48 noise dims into incoherent contributions while informative dims preserve class-correlated signal — implicit denoising." The Phase A.2 red-team caught this and demoted it to a hypothesis (H1) with a specific mechanism test.

**Mechanism prediction:** for each output dim *j* of a random R, the class-discrimination that dim provides is positively correlated with how strongly R[j] aligns with the informative subspace's class-distinct structure.

Specifically:
- **x[j]** = stddev across classes of (R[j] · P_c), where P_c is class c's prototype (zero outside informative dims). High x = R[j] separates classes well via prototype alignment.
- **y[j]** = max-min spread of per-class average projection accumulator at output dim j, observed on training data (permille scale to avoid integer truncation). High y = output dim is observed to be class-discriminative.

## Setup

- Synthetic prototype benchmark: D=64, K=16 informative + 48 noise, C=10 classes, 10% per-trit noise on informative dims, n_train=2000.
- 100 independent random R samples (different seeds via `gesh_init_random_projection`).
- Each R has sig_dim = D = 64; per sample, all 64 output dims scored.
- 6,400 (x, y) observations in total.

## Results

### Aggregate
- **Pearson r(x, y) = +0.8921**
- **t-statistic = 157.89** (df = 6398)
- **p << 0.001** (|t| > 3.29 threshold for p < 0.001 is far exceeded)

### Stratification (mean y by tertile of x)

| x range          | bin label         |     n | mean(y) |
|------------------|-------------------|------:|--------:|
| [0.0, 1.5)       | low alignment     |     7 |  3,649  |
| [1.5, 3.0)       | mid alignment     | 1,267 |  7,451  |
| [3.0, +∞)        | high alignment    | 5,126 | 11,404  |

Monotone: low → mid → high alignment maps to increasing class-discrimination spread, with a 3.1× ratio between high and low bins.

## What this confirms

H1's mechanism is **observed**, not just plausible:

1. R[j]'s prototype-alignment score predicts which output dims of the random projection actually carry class-discriminative signal in practice.
2. Output dims with low informative-subspace alignment (low x) yield small inter-class spread (low y) — the noise contributions there are not class-correlated.
3. Output dims with high informative-subspace alignment (high x) yield large inter-class spread (high y) — the informative-dim contributions dominate, even though noise dims are still mathematically present in the sum.

The "implicit denoising" framing is empirically supported on this benchmark: the noise dims contribute incoherently *to class discrimination* (the class-conditional means concentrate to similar values when noise dominates the projection), while informative-dim contributions produce class-distinct projection means.

## What this does NOT show

- **Transfer to non-rigged benchmarks.** This test runs on the synthetic with K=16 / 48-noise structure where informative and noise dims are cleanly separated by index. On real data (e.g., MNIST), there is no clean informative/noise split — every pixel carries some signal, with varying class-correlation. The H1 mechanism's *form* (alignment with class-distinct structure → discrimination) likely generalizes, but the *test* as constructed depends on the synthetic's structure.
- **That random projection is optimal.** H1 says random projection retains class-discriminative structure when alignment is high; it doesn't say random projection is the best you can do. A learned R could potentially align better with the informative subspace, capturing more class-distinct dims at every output position. The Phase A.2 sweep showed lattice update adds +8pp in compression — H1 is consistent with that, since better alignment → more usable output dims.
- **Anything about real-data transfer of C2.** C2's transfer to MNIST was confirmed independently in Gate 1 (random R at sig_dim=128 beats identity at sig_dim=784 by +7pp). Gate 2 is a *mechanism* test, not a *transfer* test.

## Implications for documentation

Per `gesh/docs/sweep_dims_results.md` § Hypotheses:
- H1 was flagged as a hypothesis, not a finding.
- This probe upgrades H1 from hypothesis to **demonstrated mechanism** (within the synthetic benchmark's domain).

The `sweep_dims_results.md` section on H1 should be updated to cite this probe.

## Reproduction

```bash
cmake --build build -j --target gesh_denoise_probe
./build/gesh/gesh_denoise_probe
```

Deterministic. Total runtime ~1 second on Apple Silicon.
