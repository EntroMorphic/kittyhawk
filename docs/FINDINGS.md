---
title: Findings — Glyph / M4T Rebuild
status: As of 2026-04-15 (cascade atomics + resolver-ceiling sweep)
companion-docs: NORTH_STAR.md · docs/THESIS.md · CHANGELOG.md · m4t/docs/M4T_SUBSTRATE.md
---

# Findings

Consolidated results from the ground-zero rebuild through the inspectability demonstration. NORTH_STAR holds the vision; THESIS holds the falsification criteria; CHANGELOG holds the commit-by-commit trail; the `journal/` directory holds the research cycles. This file is the distilled "what did the measurements tell us" layer.

## Summary (30 seconds)

Trit Lattice LSH k-NN on the rebuilt M4T substrate, deskewed MNIST, N_PROJ=2048, k=3, three RNG seeds:

| Axis | Measurement |
|---|---|
| **Accuracy (N_PROJ=2048, majority k=3)** | 97.79 ± 0.05% |
| **Accuracy (N_PROJ=2048, rank-weighted k=5)** | 97.86 ± 0.01% |
| **Accuracy (N_PROJ=4096, rank-weighted k=5 — pure-signature best)** | **97.99 ± 0.01%** |
| **Accuracy (N_PROJ=16 cascade — filter + pixel-L2 1-NN, K=50)** | **90.75% (single seed)** |
| **Accuracy (N_PROJ=16 cascade, K=100)** | **92.72%** — 30-point lift over pure-signature majority at same N_PROJ |
| **Accuracy scaling curve (N_PROJ=2 to 8192)** | Sigmoid in log-space; saturates at 8192 (+0.01% for 2× compute). Throughput peak at N_PROJ=64 (11 000 queries/sec, 92% accuracy). See `journal/full_scaling_curve.md`. |
| **Speed (N_PROJ=2048)** | 7.0 s for 10K × 60K k-NN queries — 20.3× faster than NEON-vectorized dense L1 over the same projections |
| **Inspectability** | Per-classification audit trail — structurally unavailable to dense k-NN |
| **Adaptation** | First gradient-free failure-guided classifier modification on the substrate: rank-weighted voting confirmed empirically optimal via full 81-cell matrix sweep. Exponential weighting empirically confirmed to collapse to top-1 at any k ("too-steep" failure mode). |

Routing wins significantly against both the same-features dense baseline (NEON-vectorized L1 over identical projections: 97.62 ± 0.07%; routing Δ +0.17% at k=3, +0.25% at k=5) and the classical dense pixel k-NN baseline (deskewed-pixel L1: 97.16%; routing Δ +0.63 points).

## Axis 1 — Accuracy

### Full sweep results (3 seeds, mean ± stddev)

| Features | Classifier | N_PROJ | k=1 | k=3 | k=5 |
|---|---|---|---|---|---|
| Raw pixels | L1 k-NN (deterministic) | — | 96.31% | 96.33% | 96.18% |
| **Deskewed pixels** | **L1 k-NN (deterministic)** | — | 96.98% | **97.16%** | 97.09% |
| Raw projections | NEON L1 k-NN | 512 | 96.56 ± 0.06% | 96.70 ± 0.07% | 96.66 ± 0.06% |
| Raw projections | Routed Hamming | 512 | 96.45 ± 0.09% | 96.74 ± 0.08% | **96.77 ± 0.06%** |
| Raw projections | NEON L1 k-NN | 2048 | 96.78 ± 0.09% | 97.00 ± 0.05% | 96.85 ± 0.05% |
| Raw projections | Routed Hamming | 2048 | 96.97 ± 0.05% | **97.30 ± 0.03%** | **97.18 ± 0.04%** |
| Deskewed projections | NEON L1 k-NN | 512 | 97.27 ± 0.13% | 97.41 ± 0.06% | 97.41 ± 0.08% |
| Deskewed projections | Routed Hamming | 512 | 97.14 ± 0.11% | 97.27 ± 0.09% | 97.29 ± 0.10% |
| Deskewed projections | NEON L1 k-NN | 2048 | 97.49 ± 0.10% | 97.62 ± 0.07% | 97.52 ± 0.05% |
| Deskewed projections | **Routed Hamming** | 2048 | 97.54 ± 0.07% | **97.79 ± 0.05%** | **97.77 ± 0.02%** |

### Headline gaps (routed minus same-features dense L1)

| Mode | N_PROJ | k=1 | k=3 | k=5 |
|---|---|---|---|---|
| Raw | 512 | −0.11% (0.9σ) | +0.05% (0.5σ) | +0.12% (1.2σ) |
| Raw | 2048 | +0.19% (1.8σ) | **+0.30% (5.2σ)** | **+0.32% (5.9σ)** |
| Deskewed | 512 | −0.13% (0.8σ) | −0.13% (1.2σ) | −0.11% (0.9σ) |
| Deskewed | 2048 | +0.05% (0.4σ) | +0.17% (2.0σ) | **+0.25% (4.6σ)** |

Bold entries are statistically significant (≥2σ). Routing decisively wins at N_PROJ=2048 for k=3 and k=5; is roughly tied at N_PROJ=2048 k=1; loses slightly at N_PROJ=512 (all modes, all k); and wins or ties everywhere else.

### vs the classical strong dense baseline

Best dense baseline: **deskewed-pixel L1 k-NN, 97.16%** at k=3. This is the single-run classical baseline used in the pre-rebuild journal (97.61% was the prior record; 97.16% is what we re-measured on the rebuilt substrate, using the same algorithm).

Best routed: **97.79 ± 0.05%** at deskewed-proj N_PROJ=2048 k=3.

**Routing wins by 0.63 points** against this baseline. This is the most important number — the projection-L1 comparison is a same-features control; the pixel-k-NN comparison is the "does routing beat a classical baseline" question.

## Axis 2 — Speed

### Wall-time measurements (single-threaded, mean over 3 seeds)

| Path | N_PROJ | Time (N=10 000 × M=60 000 k-NN queries) |
|---|---|---|
| Deskewed-pixel L1 k-NN | — | 45.4 s |
| NEON L1 over projections | 512 | 28.4 s |
| Routed Hamming over signatures | 512 | 2.4 s |
| **Speedup (routed/NEON-L1)** | **512** | **12.0×** |
| NEON L1 over projections | 2048 | 141.4 s |
| Routed Hamming over signatures | 2048 | 7.0 s |
| **Speedup (routed/NEON-L1)** | **2048** | **20.3×** |

### Why the speedup holds up under fair comparison

The L1 baseline is NEON-vectorized (`vabdq_s32` + `vaddw_s32` widening accumulate) — same SIMD shape as the routed popcount path. The speedup is algorithmic, not SIMD-deployment asymmetry.

Two structural reasons routing wins on speed:

1. **Compression.** A signature at N_PROJ=2048 is 512 bytes (4 trits per byte × 2048 trits = 2048/4 = 512 bytes). The MTFP projection is 2048 × 4 = 8192 bytes. 16× compression before the distance calculation even starts.
2. **Cache behavior.** 60 000 signatures × 512 bytes = 30 MB. Fits in M-series L2 (16 MB per cluster, plus L3). 60 000 projections × 8192 bytes = 480 MB — hits DRAM every query. The distance-kernel work per byte is similar; memory bandwidth is what dominates.

The 20.3× isn't a microbenchmark artifact; it's the consequence of the trit compression plus the popcount instruction shape.

## Axis 3 — Inspectability

### What the audit trail contains (per classification)

For each of the 10 000 test queries, the routed k-NN produces by construction:

1. **Top-5 nearest training prototypes** — indices, labels, Hamming distances.
2. **Vote composition at k=3** — counts per class.
3. **Per-trit decomposition of the distance to the top-1:**
   - Agreements by trit value: both +1 / both 0 / both −1.
   - Sign flips (cost 2 each): +1 vs −1 or reverse.
   - Zero-vs-sign mismatches (cost 1 each): 0 vs ±1 or reverse.
4. **Per-class nearest-prototype distance** — for all 10 MNIST classes, the minimum Hamming distance to any prototype of that class.
5. **Failure classification** (when misclassified): derived from integer thresholds on the above numbers. Four categories: NARROW MISS, VISUAL CONFUSION, SEPARATED, OUTLIER.

All of this is available at inference time without additional computation — the numbers exist as intermediate values in the distance calculation. Dense L1 doesn't have them; L1 is a scalar sum with no compositional structure.

### Aggregate findings from the misclassification set

Over 221 misclassifications at 97.79% (deskewed, N=2048, k=3):

| Failure type | Count | Fraction | Criterion |
|---|---|---|---|
| NARROW MISS | 74 | 33.5% | Correct-class prototype within 10 bits of winner |
| VISUAL CONFUSION | 65 | 29.4% | Both classes have near-lattice prototypes |
| SEPARATED | 82 | 37.1% | Correct class genuinely far from query |
| OUTLIER | 0 | 0.0% | No class has a close prototype |

**Absence of OUTLIER** means the lattice covers the test distribution; every query lands somewhere recognizable.

### Structural observation about where errors live

Across the traces, typical near-miss classification has:
- ~70% trit agreement with its top-1 prototype.
- Of the ~30% disagreements: sign-flips (cost 2, full opposition) are **5–10%**; zero-vs-sign mismatches (cost 1, threshold-boundary noise) are **90–95%**.

Errors cluster at the quantization boundary, not at semantic opposition. The router rarely says "+1" where the correct class says "−1"; it more often says "0" where the correct class says ±1, or vice versa. This is a statement about where information loss lives in the signature encoding.

Dense L1 cannot surface this observation. The distance is a sum of absolute magnitudes; "kind of disagreement" isn't preserved through the sum.

## Axis 4 — Cascade architecture (filter-ranker decomposition)

Added 2026-04-15. The atomic probe at N_PROJ=16 exposed that the Trit Lattice signature plays two different roles with two different accuracies, and the pure-signature-k-NN benchmark had been reading only the weaker one.

### The finding in one table

At N_PROJ=16, deskewed MNIST, single seed:

| Signature role | Accuracy |
|---|---|
| Classifier (pure k=7 majority) | 62.00% |
| Filter — correct class in top-50 | **98.59%** |
| Filter + pixel-L2 1-NN resolver (K=50) | **90.75%** |
| Filter + pixel-L2 1-NN resolver (K=100) | **92.72%** |

The 16-bit hash preserves **neighborhood membership** at 98.59% while getting top-1 correct only 55.48% of the time. Voting reads the destroyed rank information; a cascade reads the preserved set membership.

### Rescue/damage at K=50

| | cascade right | cascade wrong |
|---|---|---|
| pure-hash top-1 right | 5407 | 141 |
| pure-hash top-1 wrong | **3668** | 784 |

Rescue : damage ratio = **26 : 1** (3668 rescued vs 141 damaged).

### Hash-rank distribution of cascade's correct picks

| hash-rank of cascade's correct pick | fraction |
|---|---|
| rank 1 | 4.26% |
| ranks 3–5 | 8.89% |
| ranks 6–10 | 12.44% |
| ranks 11–20 | 19.93% |
| **ranks 21–50** | **50.72%** |

Over half of cascade's correct picks live in hash-ranks 21-50. The hash places correct prototypes in the neighborhood; it does not rank them. Pixel L2 does the ranking.

### Sweep across N_PROJ (crossover)

Single seed, K_RESOLVE=50, density=0.33:

| N_PROJ | pure maj | cascade L2 | Δ |
|---|---|---|---|
| 8    | 38.74% | **82.61%** | **+43.87%** |
| 16   | 62.00% | 90.75% | +28.75% |
| 32   | 80.75% | 95.04% | +14.29% |
| 64   | 91.55% | 96.65% | +5.10% |
| 128  | 95.22% | 97.28% | +2.06% |
| 256  | 96.56% | 97.51% | +0.95% |
| 512  | 97.06% | 97.57% | +0.51% |
| 1024 | 97.43% | 97.58% | +0.15% |
| **4096** | **97.65%** | **97.57%** | **−0.08%** |

Cascade gain decays monotonically. Practical crossover at N_PROJ=512. First negative gain at N_PROJ=4096.

### Two independent ceilings

- **Filter ceiling:** correct class in top-50 saturates at 99.87-99.90% by N_PROJ=64.
- **Resolver ceiling:** pixel-L2 1-NN accuracy saturates at **97.57%** regardless of how accurate the filter becomes. From N_PROJ=512 through 4096, cascade is effectively flat at 97.57-97.58%.

Cascade accuracy factorizes cleanly as `filter_presence × conditional_resolver_rate ≈ 99.87% × 97.7% ≈ 97.6%`.

### Cost-accuracy implication

| Cascade at | matches pure at |
|---|---|
| N_PROJ=8 | ≈ N_PROJ=32 |
| N_PROJ=16 | ≈ N_PROJ=64 |
| N_PROJ=32 | ≈ N_PROJ=256 |
| N_PROJ=64 | ≈ N_PROJ=2048 |
| N_PROJ=128 | near full-cascade ceiling |

Cascade buys approximately **one octave of N_PROJ** at small scales before saturating at the resolver ceiling.

### Why earlier amplification failed

The amplification experiment (`journal/amplification_negative_result.md`) ran pixel k-NN over the full 60 000 unfiltered prototypes and gained nothing. The reason is now visible: within a filtered top-50 pool, the correct-class prototype is on average **33% pixel-closer** than the nearest wrong-class prototype (relative margin +0.3255). That margin exists *because* the filter has removed most wrong-class mass. Pixel distance is not a classifier — it is a ranker that needs a filter to feed it.

### Architectural rules (updated)

1. Use cascade whenever N_PROJ ≤ 128 (gain ≥ 2 percentage points).
2. Cascade is a wash at N_PROJ ≥ 512.
3. To exceed the cascade ceiling (97.57% on deskewed MNIST with pixel-L2), change the **resolver**, not the filter.
4. Benchmark a coarse hash by its **ceiling at top-K**, not by its classifier accuracy. Classifier accuracy under-reports what the hash can enable.

## What we got right, what we got wrong

The path from 58% to 97.79% was not a single experiment; it was a sequence of corrections. Recording them for future sessions that might face similar hazards.

### Corrections made along the way

| Iteration | Wrong claim | Correction | Pointer |
|---|---|---|---|
| 1 | "Three-state routing loses to sign-only on MNIST" (based on τ-sweep on `mnist_routed_lattice.c`) | Same-τ on differently-scaled class and query sides produces asymmetric zero density, not the base-3 symmetric deployment the criterion was designed for. Symmetric τ via per-side percentile calibration fixes it. | `journal/tau_sweep_routed_mnist.md` superseded by `journal/routed_knn_mnist.md` |
| 2 | "sign_extract must be deleted" (first LMM cycle on §14) | The LMM scrutiny cycle found that the two-part criterion (C-sub + C-con) collapses into single-part "emission coverage." sign_extract IS the τ=0 degenerate of threshold_extract; the right move is to delete it and parameterize. | `journal/base3_native_criterion_*.md` + `journal/updated_model_scrutiny_*.md` |
| 3 | "Centroid-routed MNIST hits 58%; three-state routing doesn't help" | The centroid architecture throws away within-class variation. MNIST digits are multimodal in projection space. k-NN preserves modality. 58% → 97.79%. | `journal/routed_knn_mnist.md` |
| 4 | "Routed beats dense by 0.26 points, 10.8× faster" (single-run, scalar-L1 baseline) | Single-run at ~1.5σ isn't statistically significant; scalar-L1 vs NEON-popcount is apples-to-oranges. Three-seed runs with NEON-vectorized L1 show the wins strengthening: +0.30% at 5.2σ, 20.3× faster. | `docs/REMEDIATION_PLAN.md` fourth round; `CHANGELOG.md` retracted/revised entry |

### Pattern

Every correction was found by a red-team, a remediation plan, or an LMM cycle explicitly checking whether our prior claim held up. None were found accidentally. The tooling that produced the corrections (LMM, §18 emission-coverage criterion, fair-comparison remediation) is now part of the repo and is expected to catch similar failures in future work.

## Verified claims

These hold on the measurements as recorded:

- **§18 emission-coverage criterion is empirically load-bearing.** Symmetric deployment (+33.4% / 0 32.9% / −33.7% trit distribution across all configurations tested) yields higher accuracy AND higher speed than asymmetric deployments. Not just a documentation rule; a predictor of deployment quality.
- **Routing primitives produce arithmetically correct results end-to-end.** `m4t_route_threshold_extract`, `m4t_popcount_dist` (used via `m4t_route_distance_batch`), `m4t_route_topk_abs`, and `m4t_route_apply_signed` compose into a working k-NN classifier at 97.79% accuracy; the numbers are reproducible to within RNG seed variance (stddev 0.02-0.10%).
- **Hardware-native throughput holds up under fair comparison.** 20.3× speedup at N_PROJ=2048 against NEON-vectorized dense L1 — not a scalar-baseline artifact.
- **Inspectability is structural, not bolted on.** The per-trit decomposition comes free from the popcount primitive's integer sum structure. No extra work was added to the substrate to produce the audit trail.
- **The 58% → 97.79% progression identifies three distinct error classes** (centroid architecture, asymmetric τ, single-RNG single-baseline) — each one's fix is documented and reproducible.
- **The Trit Lattice signature is a lossy locality hash, not a classifier.** Ceiling@50 = 98.6% at N_PROJ=16; top-1 = 55.5%. Voting reads the destroyed rank information; the cascade reads the preserved set membership. Verified by rescue/damage 26:1, by hash-rank distribution of cascade's correct picks (50.7% in ranks 21-50), and by the N_PROJ crossover sweep. See `journal/cascade_atomics_mechanism.md` and `journal/cascade_sweep_crossover.md`.

## Unverified / qualified claims

These have support but aren't decisively established:

- **"Routing will naturally outperform dense in a base-3 environment"** (NORTH_STAR §Claim) — supported on MNIST k-NN but MNIST is a cooperative task for both approaches. The thesis is about base-3-*native* problems where dense should genuinely break, not just match. Such benchmarks have not been identified yet (see `docs/THESIS.md` §4).
- **"Routing wins generally."** Wins at N_PROJ=2048, loses by ~0.12% at N_PROJ=512. The advantage is projection-dimensionality-dependent. Not universal; not independent of configuration.
- **Training paradigm.** All measurements use prototype-based inference. No model was trained from gradients on this substrate — in fact, NORTH_STAR's claim about gradient-free base-3 training is an open research scope (see `journal/ternary_routing_helps_*.md` for the LMM cycle on this).
- **Generalization beyond MNIST.** No other benchmark has been run. The routing surface may or may not win on CIFAR-10, character n-gram classification, or other candidates listed in `docs/THESIS.md` §4.
- **Hardware utilization beyond wall-time.** We measure wall time (45 min vs 7s). We do NOT measure SDOT utilization, TBL throughput, or cache hit rates in detail. The "hardware-native" claim rests on wall-time evidence plus plausible explanations (compression, cache behavior).

## Reproducibility

### Repository state

- Commit `7c4cdac` at time of this document. 
- `git log --oneline` captures the rebuild arc and subsequent experiments.

### Environment

- Apple M-series (aarch64 + NEON). Non-aarch64 targets fail at CMake configure.
- CMake ≥ 3.16, AppleClang 17+ or equivalent.

### Building

```bash
cmake -S . -B build
cmake --build build -j
```

### Running the experiments

All MNIST tools require a directory containing standard MNIST IDX files:

```bash
# Dense L1 baseline, routed k-NN fair sweep (3 seeds, 2 N_PROJ, 2 modes)
./build/mnist_routed_knn <mnist_dir>

# Inspectability demo (single seed, 2048 projections, trace first 8 misclassifications)
./build/mnist_routed_trace <mnist_dir>

# Prior-era baselines (kept for comparison)
./build/mnist_trit_lattice <mnist_dir>     # two-stage LSH (81.40% centroid)
./build/mnist_routed_lattice <mnist_dir>   # symmetric-τ centroid sweep
```

### RNG seeds used

`tools/mnist_routed_knn.c`:
```c
SEEDS[N_SEEDS][4] = {
    { 42,   123,  456,  789  },
    { 137,  271,  331,  983  },
    { 1009, 2017, 3041, 5059 }
};
```

`tools/mnist_routed_trace.c` uses the first seed `{42, 123, 456, 789}` for reproducibility.

### Running times (M-series, single-threaded)

- `mnist_routed_knn`: ~20 minutes for the full 3-seed × 2-N_PROJ × 2-mode × 2-path sweep plus the pixel baselines.
- `mnist_routed_trace`: ~10 seconds.
- `mnist_trit_lattice`: ~1 minute for the full N_PROJ sweep.
- M4T unit tests: ~1 second total, 5 binaries.

## Pointers

### For the vision
- `NORTH_STAR.md` — why base-3, why routing, what the end-game is not.
- `docs/THESIS.md` — what would falsify the thesis; current open empirical questions.

### For the substrate
- `m4t/docs/M4T_SUBSTRATE.md` — canonical 18-section spec. §18 is the base-3-native criterion.
- `m4t/README.md` — live surface inventory.

### For the cascade architecture (Axis 4)
- `journal/nproj16_atomic_mechanism.md` — atomic probe at N_PROJ=16; partition asymmetry explains vote-rule inversion.
- `journal/nproj16_to_90_{raw,nodes,reflect,synthesize}.md` — LMM cycle on "can N_PROJ=16 reach 90%?" — the filter-ranker reframe.
- `journal/nproj16_cascade_result.md` — cascade at N_PROJ=16 hits 92.72%.
- `journal/cascade_atomics_mechanism.md` — decomposition of why cascade works; rescue:damage 26:1; hash-rank distribution.
- `journal/cascade_sweep_crossover.md` — cascade across N_PROJ; crossover at 512; pixel-L2 resolver ceiling at 97.57%.

### For the detailed experimental record
- `journal/routed_knn_mnist.md` — the k-NN wins, with "Revised after fourth red-team" section.
- `journal/routed_inspectability_trace.md` — the audit-trail demonstration.
- `journal/rebuilt_substrate_first_light.md` — the first-light numerical reproduction.
- `journal/tau_sweep_routed_mnist.md` — the centroid τ-sweep (superseded).
- `journal/fully_routed_mnist.md` — the earlier (centroid-era) routed experiment.
- `journal/base3_native_criterion_{raw,nodes,reflect,synthesize}.md` — LMM cycle that produced §18.
- `journal/updated_model_scrutiny_{raw,nodes,reflect,synthesize}.md` — scrutiny meta-cycle.
- `journal/ternary_routing_helps_{raw,nodes,reflect,synthesize}.md` — LMM on how routing helps kernels and training.

### For the process
- `docs/REMEDIATION_PLAN.md` — four rounds of red-team + fixes.
- `CHANGELOG.md` — commit-by-commit arc with measurements.
- `archive/README.md` — what lives in the archive and why.

## What this does not attempt to answer

- Whether the routing thesis generalizes to any other benchmark.
- Whether gradient-free base-3 training is viable at scale.
- Whether the 20× speedup holds on AMD Zen 4 or x86-AVX-512 targets. We run aarch64 + NEON only; other silicon has different instruction shapes.
- Whether the specific configurations tested (N_PROJ ∈ {512, 2048}, density = 0.33, k ∈ {1, 3, 5}) are optimal. Deeper sweeps are cheap and not yet run.

Each of those is an open experiment. The measurements here are the platform for running them, not a declaration that they are run.

---

**Summary of this document's purpose:** if you came to this repo cold and wanted to know "what did they actually measure and what does it mean" without piecing it together from a commit log, this is the one page to read. NORTH_STAR is the compass; this is the logbook.
