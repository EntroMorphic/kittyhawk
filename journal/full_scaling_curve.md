---
date: 2026-04-15
scope: Complete MNIST accuracy scaling curve across 13 N_PROJ values (2 → 8192)
type: experiment (extended matrix probes)
tool: tools/mnist_full_sweep.c (with spot-probe N_PROJ modifications)
---

# The complete scaling curve — N_PROJ from 2 to 8192

Extension of the main 81-cell matrix sweep (`journal/full_matrix_sweep.md`) with spot-probes at N_PROJ ∈ {512, 8192, 256, 128, 64, 32, 16, 8, 4, 2}. The curve now spans four orders of magnitude in projection dimension and half a decade of accuracy, from ~19% to ~98%.

## The full curve

Per-N_PROJ best configuration (3-seed mean ± stddev):

| N_PROJ | Sig bytes | Best (density, k, vote) | Accuracy | Δ from prev |
|---|---|---|---|---|
| 2 | 1 | (0.50, 7, majority) | 18.84 ± 1.44% | — |
| 4 | 1 | (0.33, 7, majority) | 29.76 ± 3.42% | +10.92 |
| 8 | 2 | (0.33, 7, majority) | 42.41 ± 4.84% | +12.65 |
| 16 | 4 | (0.33, 7, majority) | 63.47 ± 2.41% | **+21.06** |
| 32 | 8 | (0.33, 7, rank-wt) | 81.75 ± 1.32% | +18.28 |
| 64 | 16 | (0.33, 7, rank-wt) | 92.01 ± 0.38% | +10.26 |
| 128 | 32 | (0.33, 7, rank-wt) | 95.57 ± 0.40% | +3.56 |
| 256 | 64 | (0.25, 5, rank-wt) | 96.89 ± 0.31% | +1.32 |
| 512 | 128 | (0.33, 5, rank-wt) | 97.43 ± 0.08% | +0.54 |
| 1024 | 256 | (0.33, 5, rank-wt) | 97.75 ± 0.07% | +0.32 |
| 2048 | 512 | (0.33, 5, rank-wt) | 97.86 ± 0.01% | +0.11 |
| 4096 | 1024 | (0.33, 5, rank-wt) | 97.99 ± 0.01% | +0.13 |
| 8192 | 2048 | (0.33, 7, rank-wt) | 98.00 ± 0.04% | +0.01 |

## Shape of the curve

Clean sigmoid when plotted against log2(N_PROJ):

- **Chance floor**: 10% (10-class MNIST). Even N_PROJ=2 clears this decisively at 18.84%.
- **Steepest regime**: N_PROJ=8 → 32 gains +39.3 percentage points. Here each doubling roughly doubles accuracy.
- **Inflection**: around N_PROJ=64-128 where gain-per-doubling starts shrinking.
- **Diminishing returns**: N_PROJ=256 onward, gains drop to 1.3% then 0.5% then 0.3% per doubling.
- **Saturation**: N_PROJ=4096 → 8192 gains only +0.01% — within noise.

## Five readings

### 1. Information-theoretic consistency

The steep-climb region matches information theory:

- 3^2 = 9 possible 2-trit signatures; < 10 MNIST classes. Impossible to assign a unique signature per class. 18.84% reflects partial class-signature overlap.
- 3^4 = 81 signatures; ~8× the class count. Enough buckets to assign at least one per class but lots of overlap. 29.76%.
- 3^8 = 6561 signatures; on 60K prototypes, ~9 prototypes per signature bucket on average. Each bucket has enough samples to classify by majority. 42.41%.
- 3^16 = 43M signatures; most of 60K prototypes are unique in signature-space. But distance quantization still produces ties. 63.47%.
- 3^32 ≈ 1.85 × 10^15 signatures; vastly more than needed. 81.75%.

The steep climb corresponds to the regime where *signature space becomes large enough to distinguish classes*, and the slow climb to where *additional capacity yields diminishing discrimination*.

### 2. The vote-rule inversion at N_PROJ ≤ 16

In the prior mechanism cycle (`journal/mechanism_that_worked_synthesize.md`), rank-weighted k=5 was identified as the winning vote rule because its steep profile (5:1 weight ratio) provides signal-dominance while k=5 provides hedge.

**At N_PROJ ≤ 16, this flips: majority beats rank-weighted.** Per-N_PROJ best at small scales all show "majority" as winner; rank-weighted reclaims dominance at N_PROJ ≥ 32.

The proposed mechanism: at tiny N_PROJ, the Hamming-distance space is small (max = 2×N_PROJ = 4 at N_PROJ=2, or 32 at N_PROJ=16). Distances are integers in [0, max]. With 60K training prototypes, many tie exactly. Top-k is dominated by tied selections from insertion-sort order — arbitrary relative to ground truth.

Rank-weighted amplifies this noise because it assigns disproportionate weight to top-1 (which is effectively random among the tied set). Majority spreads the weight equally, diluting the tie-noise.

At larger N_PROJ, ties are rare (distance space is wide, distances are well-spread), so rank-weighted's amplification works *for* the signal rather than against noise.

This prediction — that rank-weighted SHOULD fail in the highly-tied regime — is consistent with the mechanism cycle's claim that rank-weighting succeeds only when there's enough information density in the top ranks to outvote ties and top-1 errors. At N_PROJ=16, there isn't.

### 3. k=7 dominates the small-N_PROJ regime, k=5 the middle, k=7 returns at the tail

| N_PROJ | Best k |
|---|---|
| 2-32 | k=7 |
| 64-128 | k=7 |
| 256 | k=5 |
| 512-4096 | k=5 |
| 8192 | k=7 |

At small N_PROJ, more hedge compensates for tie-noise and information scarcity. At middle N_PROJ, k=5 is the sweet spot. At very large N_PROJ, k=7's extra hedge shows a marginal edge — consistent with signatures becoming so discriminative that *more* neighbors can be trusted to agree.

### 4. Density 0.33 dominates across four orders of magnitude

From N_PROJ=4 to N_PROJ=4096, balanced base-3 is the empirical optimum. The exceptions:

- **N_PROJ=2**: d=0.50 wins at 18.84% vs d=0.33's 18-ish. With only 2 trits, forcing more zeros preserves more information than distinguishing signs on noisy projections.
- **N_PROJ=256**: d=0.25 wins (96.89) vs d=0.33 (96.86). Within stddev; probably not meaningful.

Balanced base-3 being the peak from 4 to 4096 is a ten-order-of-magnitude span of signature capacity. The sweet spot is remarkably invariant.

### 5. Stability grows with N_PROJ

| N_PROJ regime | Typical stddev |
|---|---|
| 2-16 | 1-5% |
| 32-256 | 0.3-1.5% |
| 512-1024 | 0.07-0.08% |
| 2048+ | 0.01-0.04% |

The substrate becomes "reliable" at N_PROJ ≈ 512 and above. Below that, the choice of random projection matrix significantly affects accuracy. This is a practical consideration for reproducibility: single-seed results at small N_PROJ should not be trusted without multi-seed variance estimation.

## Exponential weighting: collapsed at every scale

The exp-wt vote rule (weights `2^(k−rank−1)`) produces IDENTICAL accuracy across k=3, 5, 7 at every (N_PROJ, density) from 2 to 8192. At every k, top-1's weight exceeds the sum of all other weights combined:
- k=3: weights {4, 2, 1}; top-1 = 57%
- k=5: weights {16, 8, 4, 2, 1}; top-1 = 52%
- k=7: weights {64, 32, 16, 8, 4, 2, 1}; top-1 = 50.4%

Exp-wt is mathematically equivalent to "top-1 only" classification regardless of k. The "too-steep" failure mode predicted in `journal/mechanism_that_worked_*.md` is *scale-independent*: it holds from 2 trits to 8192 trits.

At every N_PROJ tested, exp-wt underperforms both majority and rank-weighted at the optimal k. This is the cleanest validation of a predicted failure mode in the project's history.

## Per-query compute and the efficiency curve

Wall-time measurements during the sweep:

| N_PROJ | k-NN wall / 10K queries | Per-query | Throughput |
|---|---|---|---|
| 2 | 2.1 s | 0.21 ms | 4800/s |
| 4 | 1.5 s | 0.15 ms | 6700/s |
| 8 | 2.1 s | 0.21 ms | 4800/s |
| 16 | 3.0 s | 0.30 ms | 3300/s |
| 32 | 1.2 s | 0.12 ms | 8300/s |
| 64 | 0.9 s | 0.09 ms | 11000/s |
| 128 | 1.0 s | 0.10 ms | 10000/s |
| 256 | 1.3 s | 0.13 ms | 7700/s |
| 512 | 2.4 s | 0.24 ms | 4200/s |
| 1024 | 3.5 s | 0.35 ms | 2900/s |
| 2048 | 7.0 s | 0.70 ms | 1400/s |
| 4096 | 16-27 s | 1.6-2.7 ms | 370-625/s |
| 8192 | 39 s | 3.9 ms | 260/s |

Non-monotonic at the bottom. **N_PROJ=64 is the throughput peak (11 000 queries/sec) at 92% accuracy.** NEON popcount's 16-byte block is the natural unit of the TBL+VCNT instructions; signature sizes at or slightly above 16 bytes hit peak SIMD efficiency.

Below N_PROJ=64, the signature is smaller than one NEON vector, so the scalar tail dominates. Above N_PROJ=64, more vectors-per-signature linearly grow compute.

## Operating points

From the full curve, clean deployment options:

**For minimum-latency applications** (embedded, real-time): N_PROJ=64, rank-wt k=7, d=0.33 → 92.01 ± 0.38%, 0.09 ms/query, 15 KB signature set (fits in L1 cache).

**For stable service workload** (~100-1000 queries/sec): N_PROJ=512 or 1024. 97.43-97.75%, 0.24-0.35 ms/query. Stddev tight (0.07%).

**For reported-accuracy headlines**: N_PROJ=4096, rank-wt k=5, d=0.33 → 97.99 ± 0.01%, 2 ms/query. Knee of the curve.

**Don't use**: N_PROJ=8192 (2× compute for +0.01% over 4096). N_PROJ ≤ 16 (accuracy and variance both too poor).

## What this curve does NOT tell us

- **Generalization**: measurements are only on deskewed MNIST. Another dataset's curve will differ in shape, knee position, and saturation point.
- **Non-deskewed accuracies**: not retested at small N_PROJ. The ~0.6-point deskewing boost is assumed constant but could vary at small scales.
- **Non-power-of-2 points**: the curve is measured at discrete N_PROJ. The true inflection might sit at N_PROJ=24 or 48 rather than exactly at our samples.
- **Dependence on n_train (number of prototypes)**: all measurements use full 60K training set. Scaling n_train is a separate axis not tested here.

## Pointers

- Main 81-cell matrix sweep (N_PROJ ∈ {1024, 2048, 4096}): `journal/full_matrix_sweep.md`.
- Vote-rule mechanism cycle: `journal/mechanism_that_worked_{raw,nodes,reflect,synthesize}.md`.
- Per-classification inspectability (the architectural foundation for mechanism analysis): `journal/routed_inspectability_trace.md`.
- Tool used: `tools/mnist_full_sweep.c`. Spot probes performed by editing `N_PROJ_VALUES[]` constant; canonical version restored to {1024, 2048, 4096}.
