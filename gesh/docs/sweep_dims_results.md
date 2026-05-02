---
title: Phase A.2 — sig_dim sweep
date: 2026-05-02
benchmark: synthetic prototype classification
status: deterministic measurement, single seed
---

# Phase A.2 sig_dim sweep

Three variants × eight projection dimensions, deterministic seeds. Tool: `gesh/bench/sweep_dims.c`. Reproducible via `./build/gesh/gesh_sweep_dims`.

## Setup

- D = 64 input dims (K = 16 informative + 48 noise).
- C = 10 classes, 10% per-trit noise.
- n_train = 2000, n_test = 500, top_k = 1.
- Training budget: ~5 flip-evaluations per trit on average, spread over 50 epochs (so larger projections get proportionally more training).

## Results

| sig_dim | random | trained | gain | flip_budget | rand_s | train_s |
|---------|--------|---------|------|-------------|--------|---------|
|       2 |    19% |     23% |  +4  |         640 |   0.00 |    0.02 |
|       4 |    24% |     30% |  +6  |        1280 |   0.00 |    0.03 |
|       8 |    35% |     44% |  +9  |        2560 |   0.00 |    0.05 |
|      16 |    47% |     62% | **+15** |     5120 |   0.00 |    0.10 |
|      32 |    62% |     75% | +13  |       10240 |   0.00 |    0.30 |
|      64 |    79% |     77% |  −2  |       20480 |   0.00 |    1.00 |
|     128 |    89% |     91% |  +2  |       40960 |   0.00 |    3.47 |
|     256 |    95% |     96% |  +1  |       81920 |   0.00 |   12.68 |

**Identity (sig_dim = D = 64, no projection): 69%.**

## What this shows

### 1. Lattice update earns its complexity in the compression regime

The largest gain (+15pp) is at sig_dim = 16 — exactly the number of informative dims in the data. The projection has to *find* the right 16 dims out of 64; lattice update solves a substantial fraction of that problem. At sig_dim = 32 (still compressed), the gain is +13pp.

In the expansion regime (sig_dim > D = 64), random R already encodes almost everything via redundancy; training adds 1–2pp.

### 2. Random ternary projection at sig_dim ≥ D outperforms identity

Identity (sig_dim = 64, raw input → bank) hits 69%. Random ternary projection at sig_dim = 64 hits 79% — **+10pp over identity at the same dimensionality.**

The mechanism: random ternary projection of the 48 noise dims produces incoherent signal that the class-mean bank averages toward zero, while informative dims still carry through (each random projection trit is a weighted sum). Identity preserves noise dims directly, where they dilute the signal-to-noise ratio in Hamming distance.

This is interesting on its own — random ternary projection is doing implicit denoising. The substrate's "no random projections in image pipelines" rule was for production deployment of LSH consumers; here, in a routing-layer setup, random ternary projection has a use case.

### 3. Anomaly at sig_dim = 64: trained −2pp vs random

At sig_dim = 64 (matching D), random R hits 79% but trained R drops to 77%. Within seed noise (test = 500 samples, ±1pp ≈ ±5 samples; ±2pp ≈ ±10 samples) — could be a fluke, or could indicate that lattice update from a random init walks into a worse local basin than random ternary's "implicit regularization" basin.

Worth multi-seed measurement before drawing conclusions. Not a bug; an empirical finding.

### 4. Capacity floor at sig_dim ≤ 4

With sig_dim = 2, the projection space has only 3² = 9 distinct ternary signatures — barely enough for 10 classes. Trained R reaches 23%, well above random chance (10%) but capacity-bounded. At sig_dim = 4, 81 distinct signatures, 30% trained.

These floors aren't training failures; they're information-theoretic capacity limits.

## Phase B+ implications

- **Sub-D compression is where lattice training pays off.** If a downstream consumer wants compact signatures (small sig_dim), lattice update is the discipline-aligned mechanism — it earns its complexity at compression.
- **At sig_dim ≥ D, training is mostly cosmetic on this benchmark.** Random ternary projection captures most of what training would. This may be benchmark-specific (the synthetic task has clear informative-vs-noise dim separation); harder benchmarks may shift the curve.
- **Identity projection is dominated.** Anywhere you'd consider identity, random ternary projection at the same sig_dim does better. Worth testing on richer benchmarks.
- **The sig_dim = 64 anomaly is the most interesting finding to investigate.** Multi-seed sweep would tell us if it's a measurement artifact or a real "training underperforms random" regime that the discipline should know about.

## Curve shape

```
Accuracy by sig_dim (D = 64):

100% |                                          ████ ████
     |                                       ████
 90% |                                  ████
     |                              ████
 80% |                          ████
     |                      ████  ████  ←  random R
 70% |                  ████      ████  ←  trained R
     |               ████             [identity 69% ━━ at sig=64]
 60% |           ████  ████
     |        ████
 50% |     ████
     |   ████
 40% |  ███
     | ██
 30% |█
     |██
 20% |█
     |─────┴──┴───┴────┴────┴────┴─────────┴─────────────┴
       2   4   8   16   32   64       128             256

  random R nearly tracks trained R at high dim;
  trained R dominates at compression;
  +15pp peak gain at sig_dim = 16 (the informative-dim count).
```

## Reproduction

```bash
cmake -S . -B build && cmake --build build -j
./build/gesh/gesh_sweep_dims
```

Total runtime ~17 seconds on Apple Silicon. Deterministic given the seeds in `sweep_dims.c::make_fixture` and `run_random` / `run_trained`.
