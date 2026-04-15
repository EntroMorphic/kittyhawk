---
title: Hyperparameters Reference — Glyph / M4T MNIST Experiments
status: As of 2026-04-14 (through `mnist_routed_amplified.c`)
companion: docs/FINDINGS.md · docs/REMEDIATION_PLAN.md · CHANGELOG.md
---

# Hyperparameters

Permanent reference for every hyperparameter set across the MNIST experiments. Organized by role. Each entry gives the value, the role, and where it's set so future modifications are traceable.

## Headline configuration

The best measured config so far (deskewed MNIST, routed k-NN, 97.86-97.90%):

| Parameter | Value | Notes |
|---|---|---|
| Input | 784-dim MNIST pixels, deskewed | Image-moment-based integer shear |
| N_PROJ | 2048 | Projection dimension |
| Projection type | Random ternary uniform | Pr[−1] = Pr[0] = Pr[+1] = 1/3 |
| Density target | 0.33 | Balanced base-3: 33% zero trits |
| τ calibration | 33rd-percentile of \|projection\| from 1000-image sample | Applied symmetrically to train + test |
| Classifier | k-NN with Hamming popcount | k = 5 with rank-weighted voting |

## Full parameter list by category

### Data preprocessing

| Parameter | Value | Where set | Role |
|---|---|---|---|
| `INPUT_DIM` | 784 | `#define` in every tool | MNIST pixel count (28×28) |
| `N_CLASSES` | 10 | `#define` in every tool | Digit classes |
| Pixel → MTFP | `(byte × 59049 + 127) / 255` | `load_images_mtfp` in every tool | Maps [0, 255] → [0, 59049] under the default block-exponent convention |
| Deskewing | Integer image-moment shear | `deskew_image`, `deskew_all` | Row-by-row shear aligned to principal inertial axis |

### Projection

| Parameter | Value | Where set | Role |
|---|---|---|---|
| `N_PROJ` | 512 or 2048 | `#define` or sweep constant | Output dimension of random projection |
| Weight distribution | {−1, 0, +1} uniform | `rng_next() % 3` in tool mains | Ternary projection matrix values |
| RNG | xoshiro128++ -ish | `rng_s[4]` state, `rng_next()` | Deterministic from 4-element seed |
| Projection kernel | `m4t_mtfp_ternary_matmul_bt` | `m4t/src/m4t_ternary_matmul.c` | MTFP19 × packed-trit → MTFP19; bit-select inner loop |

### Master seeds (reproducibility)

| Index | Seed 4-tuple | Used by |
|---|---|---|
| 0 | `{42, 123, 456, 789}` | all MNIST tools |
| 1 | `{137, 271, 331, 983}` | routed_knn, routed_weighted, routed_amplified |
| 2 | `{1009, 2017, 3041, 5059}` | routed_knn, routed_weighted, routed_amplified |

`mnist_routed_trace.c` uses only seed #0.

For K=5 ensemble in `mnist_routed_amplified.c`, each projection k ∈ {0..4} offsets the master seed by `k × (997, 1009, 1013, 1019)` per coordinate.

### Signature extraction (threshold_extract)

| Parameter | Value | Where set | Role |
|---|---|---|---|
| Primitive | `m4t_route_threshold_extract` | `m4t/src/m4t_route.c` | Produces packed-trit signatures |
| Density target | 0.33 | `DENSITY` in tools | Fraction of signature trits set to 0 |
| τ_q computation | 33rd percentile of \|projection\| from sample | `tau_for_density` helper | Determines the band width |
| Sample size | 1000 images × N_PROJ | `TAU_SAMPLE` constant | Enough for stable percentile estimate; smaller would introduce variance |
| Applied to | Both train and test signatures | Natural symmetry: same distribution | Symmetric deployment required by §18 |

### k-NN classifier

| Parameter | Value | Where set | Role |
|---|---|---|---|
| Top-k buffer | `MAX_K` = 5 (or 7 in full sweep) | Tool-local `#define` | Stores nearest prototypes per query |
| Hamming distance | `m4t_popcount_dist` with all-ones mask | `m4t/src/m4t_trit_pack.c` | Bit-level popcount of XOR |
| L1 baseline (dense) | NEON `vabdq_s32` + `vaddw_s32` widening | `l1_distance_mtfp` helper | Fair-comparison SIMD baseline |
| Top-k insertion | Scalar insertion sort | `topk_insert_i32` / `topk_insert_i64` | k is small (≤ 7); complexity irrelevant |
| Tie-breaking | Lowest class index wins | Majority vote loop | Deterministic but arbitrary |

### Vote rules

| Rule | Formula | Implemented in |
|---|---|---|
| Majority | Count labels in top-k, argmax | `vote_majority` helpers |
| Distance-weighted | `weight = max_dist − d` per neighbor, argmax sum | `vote_distance_weighted` |
| Rank-weighted | `weight = k − rank`, argmax sum | `vote_rank_weighted` |
| Exponential-weighted | `weight = 2^(k − rank − 1)`, argmax sum | Planned for full sweep |

Max Hamming distance = `2 × N_PROJ` (2 bits per trit, all may differ). At N_PROJ=2048, max_dist = 4096.

### Ensemble (`mnist_routed_amplified.c`)

| Parameter | Value | Role |
|---|---|---|
| `K_PROJS` | 5 | Number of independent projections |
| Per-projection vote rule | Rank-weighted k=5 | Inner classifier per projection |
| Aggregation | Majority of K per-projection predictions | Outer vote |
| `N_MASTER_SEEDS` | 3 | Outer-loop reproducibility |
| Fallback trigger | K-agreement < threshold | Detects ensemble disagreement |
| Fallback thresholds tested | 5, 4, 3 | Strict to lenient |
| Fallback classifier | Deskewed-pixel L1 k-NN, k=3 | Known 97.16% baseline |

### Substrate constants (inherited)

| Constant | Value | Defined in |
|---|---|---|
| `M4T_MTFP_SCALE` | 59049 = 3^10 | `m4t/src/m4t_types.h` |
| `M4T_MTFP_MAX_VAL` | 581,130,733 = (3^19 − 1)/2 | `m4t_types.h` |
| `M4T_BLOCK_BYTES` | 16 | `m4t_types.h` |
| `M4T_MTFP_CELLS_PER_BLOCK` | 4 | `m4t_types.h` |
| `M4T_TRIT_PACKED_BYTES(n)` | `(n + 3) / 4` | Macro in `m4t_types.h` |
| `M4T_ROUTE_MAX_T` | 64 | `m4t_route.h` |

### Build flags

| Flag | Value | Purpose |
|---|---|---|
| Compiler | AppleClang 17+ | aarch64 C11 compiler |
| Optimization | `-O3 -mcpu=native` | Full optimization with SIMD |
| Warnings | `-Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes` | Strict warnings |
| Errors | `-Werror` | Warnings fail the build |
| Standard | C11 | Language standard |
| Target | `aarch64 + NEON` | Apple M-series; other targets fail configure |

### Statistical reporting

| Method | Implementation |
|---|---|
| Mean | Arithmetic mean over seeds |
| Stddev | Bessel-corrected sample stddev (n-1) |
| Significance (paired) | Per-seed differences, t-statistic |
| Significance (unpaired) | σ_diff = √(σ_A² + σ_B²), z-statistic |
| Confidence | 3-seed measurements report mean ± stddev; single-run measurements explicitly noted |

## What was swept vs. held fixed

**Swept** (across experiments including the full matrix sweep in `tools/mnist_full_sweep.c`):
- N_PROJ ∈ {256, 512, 1024, 2048, 4096}
- Density ∈ {0.00, 0.10, 0.20, 0.25, 0.33, 0.50, 0.67, 1.00} (at various tools)
- k ∈ {1, 3, 5, 7}
- Vote rule ∈ {majority, distance-weighted, rank-weighted, exponential-weighted}
- Mode ∈ {raw pixels, deskewed pixels}
- Master seeds: 3

**Held fixed**:
- Projection distribution (ternary uniform 1/3 each)
- τ calibration method (percentile from sample)
- τ calibration sample size (1000)
- Hamming distance masking (all-ones mask)
- Signature extraction primitive (`threshold_extract`)
- Deskewing algorithm (image-moment shear)
- Pixel scaling (×59049, integer division /255)

**Not yet swept** (candidates for future experiments):
- N_PROJ beyond 4096 (8192)
- Non-uniform projection distributions
- τ calibration sample size sensitivity
- Per-dim τ (each projection dim gets its own threshold)
- Confusion-pair-specific masks
- k beyond 7

## Parameter sensitivities (from full matrix sweep, 2026-04-15)

From the 81-cell × 3-seed sweep in `journal/full_matrix_sweep.md`:

- **N_PROJ** (at d=0.33, k=5, rank-wt): 1024 → 2048 → 4096 gains ~0.11% then ~0.13%. Clean doubling scaling; not saturated.
- **Density** (at N_PROJ=4096, k=5, rank-wt): 0.25 → 0.33 → 0.50 gives 97.91 → **97.99** → 97.73. Peak at 0.33 (balanced base-3). Tightest stddev also at 0.33.
- **k** (rank-wt at N_PROJ=4096, d=0.33): k=3 → 97.73, k=5 → **97.99**, k=7 → 97.95. k=5 is the sweet spot.
- **Vote rule**: rank-weighted dominates at k≥5 (7 of top-10 configs). Majority peaks at k=3. **Exponential-weighted collapses to top-1 classification** at every k (top-1's weight exceeds sum of remaining weights at k=3, 5, 7) — same accuracy across k values, consistently 0.2-0.4% below rank-wt.
- **Deskewing**: +0.6% accuracy; always worth doing.
- **Master seeds**: Per-seed stddev 0.01-0.07% for 3-seed measurements. Best configuration (N=4096, d=0.33, k=5, rank-wt) achieves ±0.01% — extremely stable.

## Reproducibility command

```bash
# Build
cmake -S . -B build && cmake --build build -j

# Full matrix sweep (3 seeds × 3 N_PROJ × 3 density × 3 k × 3 vote; ~9 min)
./build/mnist_full_sweep <mnist_dir>

# Primary k-NN sweep (3 seeds × 2 N_PROJ × 2 modes; ~20 min)
./build/mnist_routed_knn <mnist_dir>

# Vote-rule adaptation (3 seeds, deskewed N=2048; ~1.5 min)
./build/mnist_routed_weighted <mnist_dir>

# Ensemble + fallback (3 seeds, K=5 projections; ~8 min)
./build/mnist_routed_amplified <mnist_dir>

# Inspectability trace (single seed; ~10 s)
./build/mnist_routed_trace <mnist_dir>
```

Cross-reference: `docs/FINDINGS.md` §Reproducibility for commit hashes and environment.
