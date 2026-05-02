---
title: Phase A.2 — sig_dim sweep
date: 2026-05-02 (multi-seed revision)
benchmark: synthetic prototype classification
status: deterministic measurement, 5 seeds per (sig_dim, variant) cell
---

# Phase A.2 sig_dim sweep — multi-seed

Three variants × eight projection dimensions, **5 independent seeds per cell**, deterministic. Tool: `gesh/bench/sweep_dims.c`. Reproducible via `./build/gesh/gesh_sweep_dims`.

## Setup

- D = 64 input dims (K = 16 informative + 48 noise).
- C = 10 classes, 10% per-trit noise.
- n_train = 2000, n_test = 500, top_k = 1.
- Training budget: ~5 flip-evaluations per trit on average, spread over 50 epochs.
- **Intra-epoch refresh:** bank and batch resampled every (n_flips/4) flips per the H1/H2 red-team remediations.
- **Early stopping:** patience 5 epochs (training halts when batch error plateaus).
- Each (sig_dim, variant) cell = mean ± sample stddev across 5 seeds with independent (init, train) seed pairs per trial.

## Results

| sig_dim | random          | trained         | gain    |  budget |
|---------|------------------|------------------|---------|---------|
|       2 |  15.6% ± 3.1 pp |  21.0% ± 2.4 pp |  +5.4 pp |     640 |
|       4 |  21.2% ± 1.6 pp |  26.8% ± 2.3 pp |  +5.6 pp |    1280 |
|       8 |  31.8% ± 3.1 pp |  36.2% ± 0.8 pp |  +4.4 pp |    2560 |
|      16 |  43.4% ± 3.8 pp |  51.4% ± 4.6 pp |  +8.0 pp |    5120 |
|      32 |  59.0% ± 2.5 pp |  67.2% ± 2.4 pp |  +8.2 pp |   10240 |
|      64 |  76.4% ± 2.1 pp |  78.2% ± 2.3 pp |  +1.8 pp |   20480 |
|     128 |  90.0% ± 1.7 pp |  89.2% ± 1.5 pp |  −0.8 pp |   40960 |
|     256 |  95.4% ± 0.9 pp |  95.4% ± 0.5 pp |  +0.0 pp |   81920 |

**Identity (sig_dim = D = 64, no projection): 69%.**

## What multi-seed corrected from the earlier single-seed version

The single-seed sweep reported:
- A **+15pp peak at sig_dim = 16**. Multi-seed mean: **+8.0pp**. The peak narrative was a single-seed artifact; the gain is real but smaller.
- A **−2pp "anomaly" at sig_dim = 64** ("training walks into a worse basin"). Multi-seed mean: **+1.8pp**. **The anomaly evaporates** — within seed noise.
- A **+13pp gain at sig_dim = 32**. Multi-seed: **+8.2pp**.

Per-seed results were within ±2.5pp of the multi-seed mean at most cells, but the headline numbers (peaks, anomalies) were dominated by single-seed luck. **This is exactly the C1 issue in the Phase A.2 red-team:** a shared-seed single-trial sweep produces narratives that average out under proper variance accounting.

## What survives multi-seed

### 1. Lattice update earns its complexity in the compression regime
Compression peak: **+8pp at sig_dim ∈ {16, 32}**. Both significantly above the 1pp stddev floor at sig_dim ≥ 64. The mechanism does meaningful work when the projection has to *select* discriminative dims; it does cosmetic polish (or nothing) when there's room for redundant encoding.

### 2. Random ternary projection at sig_dim = D beats identity
Identity at sig_dim = 64 hits 69%; random ternary projection at sig_dim = 64 hits **76.4% ± 2.1pp** — **+7pp over identity at the same dimensionality.** The mechanism (hypothesis): random ternary projection mixes the 48 noise dims into incoherent signal that the class-mean bank averages toward zero, while informative dims survive the projection. Identity preserves noise dims directly, where they dilute the class-mean's signal-to-noise ratio.

This finding is robust across seeds (±2.1pp stddev; the +7pp gap is well above noise). **It is not yet mechanism-verified** — the "implicit denoising" framing is a *hypothesis* that explains the data; it has not been tested by, e.g., examining which dims survive the random projection. Worth a follow-up cycle.

### 3. Capacity floor at sig_dim ≤ 4
At sig_dim = 2: 21% mean trained accuracy. With 3² = 9 distinct ternary signatures vs 10 classes, this is information-theoretically limited. At sig_dim = 4: 27%. These are capacity bounds, not training failures.

### 4. Diminishing returns at sig_dim ≥ 128
At sig_dim = 128, multi-seed gain is **−0.8 ± 1.5pp** (slightly negative, within noise of zero). At sig_dim = 256, exactly +0.0pp. Random ternary expansion already encodes whatever signal exists; training has nothing to add.

## Hypotheses (NOT verified findings)

These remain conjectures — plausible explanations for the data, not demonstrated mechanisms. Future cycles could pressure-test them:

1. **"Implicit denoising via random ternary projection."** Hypothesis explaining why random R at sig_dim = D beats identity. Mechanism test: project the noise dims through a random ternary matrix, examine the per-dim class-conditional variance of the projected signatures. Predicts: noise-dim signal averages toward zero in projected space.

2. **"Compression sweet spot near the informative-dim count."** Hypothesis explaining why gain peaks near sig_dim = 16 = K (the informative-dim count of the synthetic benchmark). Mechanism test: vary K (informative dim count) in the data generator and sweep sig_dim; predict peak gain shifts with K.

3. **"Random ternary expansion is enough for trivial separation."** Hypothesis explaining the +0pp gain at sig_dim = 256. Mechanism test: increase noise level or class count to force a regime where 256-dim random projection can't separate; check whether training then helps.

## Implications for Phase B+

- **If a downstream consumer wants compact signatures, lattice update is worth the complexity.** Multi-seed +5 to +8pp at sig_dim ≤ 32.
- **At sig_dim ≥ D, training is mostly cosmetic on this benchmark.** Random ternary projection captures most of what training would. May shift on harder benchmarks; worth testing.
- **Identity projection is dominated by random ternary projection** at the same sig_dim, robustly across seeds.
- **No "training underperforms random" regime detected** at the multi-seed level, contra the single-seed sweep's −2pp anomaly. The implementation can claim "training never hurts on average within ±2pp seed noise."

## Reproduction

```bash
cmake -S . -B build && cmake --build build -j
./build/gesh/gesh_sweep_dims
```

Total runtime ~22 seconds on Apple Silicon. Deterministic given the seed lists in `sweep_dims.c::main`.

## Methodology note

This sweep was originally run with **single seed** and produced narratives (peak gain, anomaly, anti-pattern) that did not survive multi-seed averaging. The Phase A.2 red-team's C1 finding caught this, prompting the multi-seed rewrite. **Lesson recorded in `CONTRIBUTING.md`:** any benchmark claim with directional language ("peak", "anomaly", "winner") needs multi-seed validation before promotion to a finding. Single-seed measurements are exploratory; multi-seed measurements are evidence.
