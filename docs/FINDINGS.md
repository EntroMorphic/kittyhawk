---
title: Findings — Glyph / M4T Rebuild
status: As of 2026-04-15 (multi-table routed bucket at M=32 reaches 97.24% on deskewed MNIST at N_PROJ=16 — first routed architecture in Glyph to exceed 97%; matches pure N_PROJ=512 scaling curve at equivalent total bits)
companion-docs: NORTH_STAR.md · docs/THESIS.md · CHANGELOG.md · m4t/docs/M4T_SUBSTRATE.md
---

# Findings

Consolidated results from the ground-zero rebuild through the last completed MNIST rerun before the routed-resolver conversion. NORTH_STAR holds the vision; THESIS holds the falsification criteria; CHANGELOG holds the commit-by-commit trail; the `journal/` directory holds the research cycles. This file is the distilled "what did the measurements tell us" layer.

The cascade architecture (Axis 4) has two layers of measurements in this document:
- A historical section covering the dense-resolver cascade that first validated the filter-ranker reframe.
- A routed-rerun section covering the same experiments after the sixth-round remediation converted all resolvers to routing primitives. Both are reported side by side so the mechanism story is legible without smuggling in dense signal.

## Summary (30 seconds)

Trit Lattice LSH k-NN on the rebuilt M4T substrate, deskewed MNIST, N_PROJ=2048, k=3, three RNG seeds:

| Axis | Measurement |
|---|---|
| **Accuracy (N_PROJ=2048, majority k=3)** | 97.79 ± 0.05% |
| **Accuracy (N_PROJ=2048, rank-weighted k=5)** | 97.86 ± 0.01% |
| **Accuracy (N_PROJ=4096, rank-weighted k=5 — pure-signature best)** | **97.99 ± 0.01%** |
| **Accuracy (N_PROJ=16 cascade — filter + pixel-L2 1-NN, K=50) — HISTORICAL dense resolver** | 90.75% (single seed) |
| **Accuracy (N_PROJ=16 cascade, K=100) — HISTORICAL dense resolver** | 92.72% |
| **Accuracy (N_PROJ=16 routed quadruple: H1 filter + H2+H3+H4 fusion)** | 83.86% |
| **Accuracy (N_PROJ=16 routed, fused filter H1+H2 + H2+H3+H4 baseline L50_H1 → L50_H12 lift)** | 83.86% → **88.44%** (+4.58 from a single architectural move) |
| **Accuracy (N_PROJ=16 routed, L200_H12 — fused filter + widened K, best measurement-scaffolding variant)** | **88.87%** (single seed), within 0.59 of global Gq at ~50% of Gq's cost |
| **Accuracy (N_PROJ=16 routed BUCKET consumer — first genuinely routed architecture, O(1) amortized query)** | **82.58%** at **9.9 μs/query** (single seed), −1.28 below dense L50_H1 at **~197× faster wall time** |
| **Accuracy (N_PROJ=16 multi-table routed bucket, M=16 tables, SUM resolver)** | **96.13%** at **~0.67 ms/query** (single seed) |
| **Accuracy (N_PROJ=16 multi-table routed bucket, M=32 tables, SUM resolver — first routed architecture to exceed 97%)** | **97.24%** at **~1.92 ms/query** (single seed); matches pure N_PROJ=512 scan (97.06%) at ~2× faster wall time |
| **Accuracy (N_PROJ=16 multi-table routed bucket, M=64 tables, SUM resolver)** | **97.31%** at **~4.13 ms/query** (single seed); within noise of pure N_PROJ=1024 scan (97.37%) |
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

## Axis 4 — Cascade architecture (historical filter-ranker decomposition)

Added 2026-04-15. Historical note: this axis summarizes the last completed dense-resolver cascade measurements before the sixth remediation round converted the live cascade architecture to routed resolvers. The atomic probe at N_PROJ=16 exposed that the Trit Lattice signature plays two different roles with two different accuracies, and the pure-signature-k-NN benchmark had been reading only the weaker one.

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

### Status after routed-resolver conversion

The live cascade tools (`tools/mnist_cascade_nproj16.c`, `tools/mnist_cascade_sweep.c`, `tools/mnist_cascade_atomics.c`, `tools/mnist_resolver_sweep.c`) no longer use pixel-L2 or any other dense resolver. The tables above remain historically useful because they explain *why* the cascade worked; the tables below are the literal description of the current implementation.

## Axis 4b — Cascade architecture (routed rerun)

Added 2026-04-15 after the sixth-round dense→routed conversion. Same cascade structure, same primary 16-bit hash, but the resolver stage now uses additional routing primitives — a second 16-bit hash with an independent seed (H2), a third (H3), or dual/triple-hash fusions. No pixel or centroid access remains on the classifier path.

### The finding in one table (routed cascade, deskewed MNIST, density=0.33, single seed)

| Signature role | Accuracy |
|---|---|
| Classifier (pure k=7 majority at N_PROJ=16) | 62.00% |
| Filter — correct class in top-50 | 98.59% |
| Filter + routed H2 1-NN resolver (K=50) | **77.33%** |
| Filter + routed H2 1-NN resolver (K=100) | 78.47% |
| Filter + routed H2+H3 1-NN fusion (K=20, best small-N_PROJ resolver) | **81.35%** |

Filter-ranker reframe holds. +15 points over pure-hash majority using nothing but signature operations on the 16-bit lattice. Triple-hash fusion adds another +4 points at small N_PROJ.

### Routed cascade across N_PROJ (H2 1-NN baseline, K_RESOLVE=50)

| N_PROJ | pure maj | routed cascade | Δ |
|---|---|---|---|
| 8 | 38.74% | **54.21%** | **+15.47** |
| 16 | 62.00% | 77.33% | +15.33 |
| 32 | 80.75% | 89.25% | +8.50 |
| 64 | 91.55% | 93.87% | +2.32 |
| 128 | 95.22% | 95.67% | +0.45 |
| 256 | 96.56% | 96.44% | **−0.12** |
| 512 | 97.06% | 96.98% | −0.08 |
| 1024 | 97.43% | 97.08% | −0.35 |
| 4096 | 97.65% | 97.41% | −0.24 |

Practical crossover at **N_PROJ=256** — one step earlier than the dense-resolver crossover at N_PROJ=512. The routed resolver hits its own ceiling sooner.

### Best routed resolver per N_PROJ (resolver sweep)

| N_PROJ | pure maj | best routed resolver | accuracy |
|---|---|---|---|
| 16 | 62.00% | **H2+H3 1-NN (triple-hash fusion)** | **81.35%** |
| 128 | 95.22% | H2 5-NN rank-weighted | 96.50% |
| 1024 | 97.43% | H2 5-NN rank-weighted | 97.53% |

At small N_PROJ, stacking independent routed views pays. At large N_PROJ, resolver choice barely matters — the filter is already near-perfect and all variants sit within 0.4 points.

### Atomic decomposition — same mechanism, smaller margin

Rerun of `mnist_cascade_atomics` on the routed cascade (N_PROJ=16, K=50):

| Metric | Routed | Dense (historical) |
|---|---|---|
| Rescue count | 2743 | 3668 |
| Damage count | 558 | 141 |
| **Rescue:damage ratio** | **5 : 1** | 26 : 1 |
| Conditional resolver rate | 78.44% | 92.05% |
| Relative margin (wrong − correct) / (wrong + correct + 1) | **+0.2056** | +0.3255 |
| Tied-min partition accuracy | 85.90% | 95.96% |
| Elsewhere-in-top-10 accuracy | 64.66% | 86.94% |

Same mechanism in every row; every magnitude is roughly 60-80% of the dense version. The routed resolver still discriminates correctly on average — the "correct is closer" signal has +0.21 relative margin — but the signal is thinner, so the conditional rate is ~13 points below pixel L2 and the rescue:damage ratio drops by 5×.

### New failure modes: correlated hash errors

The routed rerun exposes confusion pairs that were absent from the dense version:

| true | pred | pure err | cascade err | Δ |
|---|---|---|---|---|
| 3 | 8 | 46 | 74 | −28 |
| 3 | 5 | 43 | 67 | −24 |
| 6 | 8 | 21 | 35 | −14 |

A second random 16-bit ternary projection shares more failure modes with the first than an orthogonal modality would. Digits whose trit projections look similar under any random projection (3↔8, 3↔5) regress in the routed cascade. Pixel L2 broke these because shape differences are pixel-visible; secondary hashes do not because they cannot see shapes they didn't already fail to see.

### Two ceilings still cleanly visible

- **Filter ceiling** (unchanged): correct class in top-50 saturates at 99.87-99.90% by N_PROJ=64.
- **Routed resolver ceiling:** routed cascade plateaus at ~97.5% from N_PROJ=512 upward. Specifically, `99.87% × 78.9% ≈ 78.8%` at N_PROJ=16 (where 78.9% is the conditional routed resolver rate), climbing as the filter becomes near-perfect and the conditional rate co-climbs.

Full writeup: `journal/routed_cascade_rerun.md`.

### Architectural rules (routed version)

1. Use cascade when N_PROJ ≤ 64 (gain ≥ 2 points over pure majority).
2. Above N_PROJ ≥ 256 the routed cascade is a wash and trends slightly negative.
3. To exceed the routed cascade ceiling without reintroducing dense signal, stack independent routed views (H2+H3 fusion; quadruple-hash H2+H3+H4 at **83.86%**; possibly learned routed projections).
4. The filter-ranker reframe is substrate-invariant: the hash is always a better filter than a ranker. Only the resolver's ceiling moves.

## Axis 4c — Meta-routing and the observability ceiling at N_PROJ=16

Added 2026-04-15 after the routed quadruple-hash rerun pushed local cascade to 83.86% at N_PROJ=16. The natural next question: can a global observer layer rescue queries that the local cascade fails on, without reintroducing dense signal? LMM cycle `journal/meta_router_online_{raw,nodes,reflect,synthesize}.md` produced a design; P1 gate + atomic decomposition falsified the strong version of it.

### The P1 prerequisite gate (passed)

`tools/mnist_local_vs_global.c` compares three variants at N_PROJ=16, K_RESOLVE=50, single seed:

| variant | accuracy | Δ vs local |
|---|---|---|
| L — local quadruple (H1 top-50 + H2+H3+H4 fusion) | 83.86% | — |
| Gt — global H2+H3+H4 summed over 60K | 86.64% | +2.78 |
| **Gq — global H1+H2+H3+H4 summed over 60K** | **89.46%** | **+5.60** |

2×2 contingency `(L vs Gq)`:

|  | Gq right | Gq wrong |
|---|---|---|
| L right | 8055 | 331 (damage) |
| L wrong | 891 (rescue) | 723 (both wrong) |

- Rescue:damage ratio = 891 : 331 ≈ **2.7 : 1**.
- Conditional `P(Gq correct | L wrong) = 55.2%`.
- **Oracle ceiling** (L right ∪ Gq right) = 9277 = **92.77%**.

Gate: **PASS.** An architecture that escalates only the rescuable queries would reach 92.77% — an additional +8.91 points over local at bounded cost.

### The atomic decomposition (revised P2 scope)

`tools/mnist_lvg_atomics.c` packs seven signals per query and reports distributions per contingency cell. The headline is negative for the original LMM synthesize's ambition:

**On every inference-available signal, rescues and damages have nearly indistinguishable distributions.** Ensemble disagreement cleanly separates easy (21.7%) from hard (~60-68%), but within the hard set — the 1945 queries where escalation even matters — disagreement sees rescue (67.9%), damage (59.5%), and double-fail (58.9%) as essentially the same.

The feature that *would* separate rescue from double-fail is the rank of the correct class in H1's top-50:

| cell | rank=1 | 2-5 | 6-10 | 11-20 | 21-50 | >50 |
|---|---|---|---|---|---|---|
| rescue | 21.1% | 30.5% | 15.0% | 14.5% | 11.5% | **7.4%** |
| both wrong | 18.4% | 27.8% | 16.3% | 14.7% | 12.5% | **10.4%** |

**7.4% of rescues have correct class *outside* H1's top-50 entirely.** These are filter-miss failures — no resolver reading only the filtered pool can fix them; global fusion rescues them precisely because it sees all 60K prototypes. But this feature requires knowing the true label, so it's unavailable at inference.

One modest observable: damages pick fusion winners deeper in H1's ordering (52.9% at ranks 21-50 vs 46.5% for easy, 6.7% at ranks 2-5 vs 13.0% for rescue). A 6-8 point gap — marginal selectivity.

Per-class:

| class | rescues | damages | rescue:damage |
|---|---|---|---|
| 3 | 94 | 65 | **1.4** |
| 4 | 117 | 34 | 3.4 |
| 6 | 77 | 19 | 4.1 |
| 7 | 67 | 17 | 3.9 |

**Class 3 has the worst rescue:damage ratio.** Global fusion has its own 3↔5 and 3↔8 confusions; escalating class-3 queries barely helps. Class 1 is effectively solved (4:3, negligible mass).

### Revised meta-router predictions

Using ensemble disagreement as the primary escalation signal (the only clean observable):

| architecture | predicted / measured |
|---|---|
| pure local cascade | 83.86% |
| **disagreement meta-router (predicted)** | **~88.0%** |
| pure global Gq | 89.46% |
| oracle ceiling (perfect router) | 92.77% |

The meta-router's natural ceiling is ~88%, about 1.5 points below pure global at ~50% of pure-global cost. The ~4.77-point gap between meta-router and oracle lives in queries whose rescue/damage label is hidden in information the substrate structurally cannot observe.

### New verified claim

**The observability ceiling at N_PROJ=16.** Rescue and damage classification from global escalation cannot be predicted from inference-available signals. Any meta-router built on `{H1 min_d, tied_count, ensemble disagreement, fusion pick rank, fusion margin}` is bounded at ~88% aggregate accuracy because rescues and damages share distributions on all these axes. The only separator is correct-class rank in H1's pool, which requires label knowledge. See `journal/lvg_atomics_decomposition.md`.

### What would push above 88%

Reading the atomics, closing the gap requires structural changes, not a better meta-router:

1. **Broaden H1's pool.** 7.4% of rescues have correct outside top-50 — widening K to 100 or 200 increases the filter ceiling. Linear resolver-cost.
2. **Fused filter.** Use H1+H2 (two independent hashes) at the filter stage instead of H1 alone, then resolve locally with H3+H4. Adds information at the filter level where the bottleneck lives.
3. **Per-class policy.** Use local's predicted class as a routing-context feature — skip escalation on predicted class 3 since global damages ≈ global rescues in that region.
4. **More independent hashes.** Pentuple-hash fusion (H2+H3+H4+H5) continues the +2.5/+1.5-point gain trajectory at small N_PROJ.

Of the four, **fix #2 was tested next and the result reshapes the whole cascade line.** See Axis 4d.

## Axis 4d — Fused filter: the fix that deprecated the meta-router

Added 2026-04-15. The Axis 4c atomic decomposition identified two structural failures in the local architecture and suggested four possible fixes. `tools/mnist_local_v2.c` composes Fix #1 (widen K_RESOLVE) and Fix #2 (fused filter) on independent axes and reruns the P1 gate against the same Gq reference. The measurement falsified the meta-router architecture and established a new architectural rule about information leverage in a cascade.

### The fix axes

- **Fix A — widen K_RESOLVE.** Motivated by the 7.4% of rescues whose correct class sits outside H1's top-50 pool entirely (from Axis 4c table D). Test K ∈ {50, 100, 200}. Cost scales linearly in the resolver stage, which is cheap relative to the filter.
- **Fix B — fused filter.** Motivated by H1's 55.5% top-1 rank accuracy (vs 98.59% neighborhood ceiling at K=50): a single 16-trit hash destroys ranking information *inside* its preserved neighborhood. The fused filter takes top-K by `(H1 + H2)` summed distance instead of `H1` alone, then resolves locally with H3+H4 (or H2+H3+H4 when H2 is still available). H2 moves from "one of three resolvers" to "half of the filter" — same four hashes, same arithmetic, different cascade position.

The two fixes are independent axes and compose into a 2×3 grid. All six variants are compared against the same Gq reference (global H1+H2+H3+H4 summed over all 60K prototypes, 89.46% accuracy at N_PROJ=16).

### Full result table (N_PROJ=16, density=0.33, single seed 42, deskewed MNIST, 10K test queries)

| variant | filter stage | resolver stage | K_RESOLVE | accuracy | Δ vs L50_H1 baseline | Δ vs Gq |
|---|---|---|---|---|---|---|
| **L50_H1** (original baseline) | H1 alone over 60K | H2+H3+H4 over top-50 | 50 | 83.86% | — | −5.60 |
| L100_H1 | H1 alone over 60K | H2+H3+H4 over top-100 | 100 | 85.59% | +1.73 | −3.87 |
| L200_H1 | H1 alone over 60K | H2+H3+H4 over top-200 | 200 | 86.79% | +2.93 | −2.67 |
| **L50_H12** | (H1+H2) over 60K | H3+H4 over top-50 | 50 | **88.44%** | **+4.58** | **−1.02** |
| L100_H12 | (H1+H2) over 60K | H3+H4 over top-100 | 100 | 88.73% | +4.87 | −0.73 |
| **L200_H12** | (H1+H2) over 60K | H3+H4 over top-200 | 200 | **88.87%** | **+5.01** | **−0.59** |
| Gq (reference) | — | (H1+H2+H3+H4) over all 60K | — | 89.46% | +5.60 | — |

**Fused filter (Fix B) alone — holding K constant at 50 — lifts accuracy by +4.58 points.** Widening K alone (Fix A) lifts by +2.93 points at its maximum (K=50 → K=200). Fix B provides more than 1.5× the accuracy gain of Fix A from a single change at the filter stage. Composing both fixes (L200_H12) lands at 88.87% — **closing 89% of the original L→Gq gap**.

### Filter ceiling comparison

The filter ceiling is the fraction of test queries whose correct class is present *anywhere* in the filter's top-K output — the upper bound that any resolver reading only the filtered pool can reach.

| K | H1 alone filter | (H1+H2) fused filter | lift |
|---|---|---|---|
| 50 | 98.59% | 99.55% | +0.96% |
| 100 | 99.50% | 99.85% | +0.35% |
| 200 | 99.86% | 99.94% | +0.08% |

The fused filter at K=50 drops the filter-miss rate from 1.41% to 0.45% — roughly a 3× reduction. But note that the ceiling lift at K=50 is 0.96 points while the accuracy lift (L50_H1 → L50_H12) is 4.58 points. **Most of the fused filter's benefit is NOT about adding correct-class prototypes to the pool.** It's about improving the *ranking* of prototypes already in the pool. The fused filter pulls correct-class prototypes from deep ranks (6+) to shallow ranks (1-10) *before* the hard K-cut, so the resolver sees a cleaner starting distribution.

### Contingency against Gq (full 2×2 for each variant)

| variant | LR_GR (both right) | LR_GW (damage) | LW_GR (rescue) | LW_GW (both wrong) | net (rescue−damage) | oracle ceiling |
|---|---|---|---|---|---|---|
| L50_H1 (original P1) | 8055 | 331 | **891** | 723 | **+560** | 92.77% |
| L100_H1 | 8265 | 294 | 681 | 760 | +387 | 92.40% |
| L200_H1 | 8432 | 247 | 514 | 807 | +267 | 91.93% |
| **L50_H12** | 8651 | 193 | **295** | 861 | **+102** | 91.39% |
| L100_H12 | 8681 | 192 | 265 | 862 | +73 | 91.38% |
| **L200_H12** | 8658 | 229 | **288** | 825 | **+59** | 91.75% |

Three observations in this table:

1. **Rescues collapse from 891 to 265-295.** The fused filter removes 67-70% of the queries that global fusion was previously rescuing. Those queries are now handled correctly by local directly.
2. **Damages also drop, from 331 to ~192-229.** The fused filter doesn't just rescue new queries; it also makes local more reliable on queries global previously got wrong.
3. **Oracle ceiling shrinks from 92.77% to 91.38-91.75%.** As local gets stronger, the total room for any meta-router to add value on top shrinks. An oracle meta-router on L200_H12 could reach at most 91.75% — only 2.88 points above L200_H12 itself, compared to 8.91 points of headroom above L50_H1.

### Cost accounting

Cost is dominated by the global passes in each stage. Counting popcount distance operations per query (one popcount over 60K prototypes is 60K ops):

| architecture | global passes | resolver work | total distance ops | relative cost | accuracy |
|---|---|---|---|---|---|
| L50_H1 (baseline) | 1 × 60K (H1) | 50 × 3 (H2+H3+H4) | 60K + 150 ≈ 60K | 1.00× | 83.86% |
| L200_H1 | 1 × 60K (H1) | 200 × 3 | 60K + 600 ≈ 60.6K | 1.01× | 86.79% |
| **L50_H12** | 2 × 60K (H1, H2) | 50 × 2 (H3+H4) | 120K + 100 ≈ 120K | 2.00× | 88.44% |
| **L200_H12** | 2 × 60K (H1, H2) | 200 × 2 (H3+H4) | 120K + 400 ≈ 120K | 2.01× | 88.87% |
| Gq (global reference) | 4 × 60K (H1+H2+H3+H4) | — | 240K | 4.00× | 89.46% |

**L200_H12 at 2× baseline cost captures 99.3% of Gq's accuracy gain over baseline at 50% of Gq's cost.** Resolver-stage cost (widening K from 50 to 200) is negligible — the filter dominates completely because each global pass is 60K operations versus the resolver's 100-400 ops.

### Why the fused filter works — the information-leverage principle

Both L50_H1 and L50_H12 use the exact same four hashes (H1, H2, H3, H4). Same RNG seeds, same density, same prototype set, same Hamming kernel. The only difference is **which stage applies H2**.

- In L50_H1, H2 is one of three resolvers. The filter decides which 50 prototypes survive via H1 alone; H2 contributes ranking information only *after* that commitment.
- In L50_H12, H2 is half of the filter. The filter decides which 50 prototypes survive via (H1+H2); H2 contributes set-membership information *before* the commitment.

Moving H2 across this boundary buys **+4.58 points** of accuracy without adding any new information to the system.

The mechanism: the filter stage is a hard commitment — only top-K candidates reach the resolver. Correct-class prototypes that the filter ranks below position K are gone forever regardless of what the resolver would have said about them. The atomic decomposition showed that 48% of the original rescues lived at H1 ranks 6+ (`rank 6-10`: 15.04%, `11-20`: 14.48%, `21-50`: 11.45%, `>50`: 7.41%), and H1 alone was failing to pull them forward.

The fused filter uses H2's second opinion at the filter stage to *rescue these prototypes before they get cut*. Once correct-class prototypes sit in shallow positions of top-K, the resolver's job becomes easier — even a narrower resolver (H3+H4 instead of H2+H3+H4) is enough to discriminate them from wrong-class neighbors.

Stated as an architectural rule:

> **Information leverage rule:** in a cascade, information applied at the filter stage constrains set membership; information applied at the resolver stage only re-orders an already-committed pool. When the filter is imperfect, spend marginal routing information on the filter first. Information has higher leverage earlier in the cascade.

This is the **dual of the filter-ranker reframe** from Axis 4:
- Axis 4 said: "The hash is a filter, not a classifier — use it as a filter."
- Axis 4d says: "And when you have more than one hash available, put them *all* at the filter stage until the filter saturates. Only spend remaining hashes at the resolver."

Together the two rules give a concrete ordering for how to allocate signature budget in a multi-hash cascade.

### Why meta-routing is now obsolete

The Axis 4c LMM cycle proposed an online, inline meta-router to close the L→Gq gap by routing hard queries to global on a per-query basis. The atomic decomposition predicted a ceiling of ~88% for any meta-router built on inference-available signals. The fused-filter fix produces the same ~88% accuracy without any routing at all — and crucially, it lowers the oracle ceiling from 92.77% to 91.75%, leaving only 2.88 points of theoretical headroom for *any* meta-router to add on top.

A meta-router on L200_H12 would have to:
1. Find 288 rescuable queries among ~1342 imperfect queries (rescue:damage ratio 1.26:1, barely above chance)
2. Without any clean observable separator (rescues and damages remain observationally indistinguishable)
3. Add perhaps +0.2-0.4 points of realistic aggregate accuracy
4. At ~50% additional cost (partial Gq escalation on ~30% of queries)

The cost/benefit does not justify building the routing-context k-NN bank. The meta-router LMM cycle is therefore **closed with a negative architectural verdict on its proposed primary artifact but a positive research verdict on its secondary discovery.** The cycle's P1 gate forced the atomic decomposition; the atomic decomposition exposed the filter-ranking structure of the gap; the fused-filter fix became obvious from that exposure and recovers almost all of the predicted meta-router gain through a simpler architectural change.

Stated plainly: **the meta-router was a proposal to route around a deficient filter. The correct answer is to deepen the filter.**

### Architectural rules (updated with Axis 4d)

1. **Use cascade when N_PROJ ≤ 64.** Gain ≥ 2 points over pure majority on deskewed MNIST.
2. **Above N_PROJ ≥ 256 the routed cascade is a wash.** The filter's own ranking is already near-ceiling and cascade adds little.
3. **To exceed the routed cascade ceiling without reintroducing dense signal, spend information at the filter stage first.** Fused filter (H1+H2) beats adding the same hash as a resolver by ~4.5 points at N_PROJ=16. Triple-filter may continue the trend; this is an open measurement.
4. **The filter-ranker reframe is substrate-invariant.** The hash is always a better filter than a ranker. Only the resolver's ceiling moves.
5. **Information leverage is filter-stage-first.** When allocating multiple independent hashes across a cascade, put them at the filter stage until the filter saturates, *then* use remaining hashes as resolvers.
6. **Meta-routing has an observability ceiling at small N_PROJ.** Rescue and damage share distributions on all inference-available signals; observable-feature meta-routers cannot separate them beyond a ~88% ceiling at N_PROJ=16 given the observability data. This bounds any cycle that proposes query-time escalation based on signature-space features.

### Follow-ups opened by this result

1. **Fused filter across N_PROJ.** Does the +4.58-point lift at N_PROJ=16 hold at N_PROJ=8, 32, 64, 128? Predicted shape: largest gain at small N_PROJ where H1's ranking is weakest; diminishing gain as N_PROJ grows and H1 alone becomes a good ranker.
2. **Triple-filter (H1+H2+H3).** With one hash held back at the resolver stage, does a three-hash filter beat a two-hash filter + one-hash resolver? Per the information-leverage rule, yes — but the effect size depends on marginal returns and may be small if L200_H12 is already close to ceiling.
3. **Quadruple-filter = Gq.** Applying all four hashes at the filter stage with an empty resolver is literally Gq. So the space of routed-cascade architectures is bounded between "one hash at filter" (L50_H1 baseline) and "all hashes at filter" (pure global). L200_H12 sits at (2 filter, 2 resolver) and captures most of the gap. The question is whether (3 filter, 1 resolver) sits meaningfully between (2,2) and (4,0).
4. **Retire the meta-router cycle.** Add a closing synthesize note to `journal/meta_router_online_synthesize.md` acknowledging the deprecation. Reference `journal/fused_filter_fix.md` as the resolution.

## Axis 5 — Routed architecture: the bucket-indexed consumer

Added 2026-04-15 after the user caught a critical framing failure: every cascade tool in the tree (`mnist_cascade_nproj16`, `mnist_cascade_sweep`, `mnist_cascade_atomics`, `mnist_resolver_sweep`, `mnist_local_vs_global`, `mnist_local_v2`, `mnist_lvg_atomics`, `mnist_routed_knn`, `mnist_full_sweep`) runs routing primitives inside an `O(N_train)` dense outer loop. They compute `m4t_popcount_dist(query, train[i])` for every `i` in `[0, n_train)`. That is a dense architectural shape with routed kernels — a substrate-level NORTH_STAR violation even though the per-comparison primitive is routing-native.

The "20.3× speedup over dense L1" headline from the earlier Axis 2 is apples-to-apples on the SAME `O(N)` shape: routed-Hamming kernels vs NEON-L1 kernels, both scanning all 60K prototypes. The speedup is real but it is a *compression* win, not a *routing* win at the architecture level.

This axis documents the first Glyph consumer that respects the contract end-to-end: `tools/mnist_routed_bucket.c`. Training is a one-time sort of `(signature_key, prototype_index)` pairs. Query time is binary search into the sorted table plus ternary-Hamming multi-probe over neighbor codes. **Zero `popcount_dist` calls at the filter stage.** The signature is no longer an operand — it is the address the query dereferences.

### Architecture

```
TRAINING (one-time)
  for each prototype i in 0..N_train:
      sig_key[i] = sig_to_key(threshold_extract(W_H1 @ x_train[i], tau))
      append (sig_key[i], i) to entries[]
  qsort(entries, by sig_key)    // 3 ms at N_train = 60000

QUERY (per test sample)
  q_key = sig_to_key(threshold_extract(W_H1 @ x_query, tau))
  candidate_set = {}
  for radius r = 0 .. MAX_RADIUS:
      if |candidate_set| >= MIN_CANDIDATES: break
      for each probe_key in ternary_neighbors(q_key, r):
          start = lower_bound(entries, probe_key)
          while entries[start].key == probe_key:
              candidate_set.add(entries[start].proto_idx)
              start += 1
  // Resolver — routed, over the candidate set only (never over 60K)
  for each c in candidate_set:
      score[c] = popcount_dist(q_H2, H2[c]) + popcount_dist(q_H3, H3[c]) + popcount_dist(q_H4, H4[c])
  return label(argmin score)
```

### Ternary multi-probe enumeration

At ternary Hamming radius `r`, neighbor codes are enumerated by direct trit manipulation on packed 2-bit fields (no unpacking to int8_t arrays):

- **r=0:** one probe (the query's own code).
- **r=1:** at one trit position, transition 0 ↔ ±1 (cost 1 per position). At density=0.33 this produces ~27 probes per query on average (16 positions × 1.67 outcomes each, weighted by the 0/±1 split).
- **r=2:** either (a) a single sign-flip at one non-zero position (+1 ↔ −1, cost 2 per position), or (b) two distinct cost-1 moves on different positions. Total ~340 probes at r=2 per query.

Enumeration cost is cache-friendly because every probe key is a 4-byte integer derived from bit-field edits to a stack-local scratch buffer. Each probe is then a binary search over a 937 KB sorted table — fits comfortably in L2.

### Index characteristics at N_PROJ=16 (60 000 MNIST training prototypes)

```
60 000 training prototypes → 37 906 distinct buckets (1.58× compression)

bucket size histogram
  size 1        29 616 buckets (78.1% of buckets)
  size 2-3       6 099 buckets
  size 4-7       1 621 buckets
  size 8-15        420 buckets
  size 16-31       112 buckets
  size 32-63        25 buckets
  size 64-127       12 buckets
  size 128+          1 bucket  (the degenerate all-zero / near-zero-sig region)
```

78% of buckets are singletons — most H1 signatures are occupied by exactly one training prototype. The codebook (`3^16 ≈ 43 million` possible codes) is heavily under-saturated by 60K prototypes, so most *codes* stay empty, and the occupied ones usually hold one item. The retrieval behavior is dominated by the handful of non-singleton buckets that queries actually land in.

### Tuning sweep (`MAX_RADIUS × MIN_CANDIDATES` grid)

| MAX_R | MIN_C | accuracy | avg candidates | avg probes | empty | μs/query |
|---|---|---|---|---|---|---|
| 0 | 1 | 36.99% | 9.2 | 1.0 | 4797 | 0.4 |
| 0 | 20 | 36.99% | 9.2 | 1.0 | 4797 | 0.3 |
| 0 | 100 | 36.99% | 9.2 | 1.0 | 4797 | 0.3 |
| 0 | 400 | 36.99% | 9.2 | 1.0 | 4797 | 0.3 |
| 1 | 1 | 61.80% | 8.4 | 9.8 | 1129 | 0.7 |
| 1 | 20 | 68.77% | 31.5 | 20.7 | 1129 | 1.6 |
| 1 | 100 | 68.90% | 46.5 | 22.0 | 1129 | 2.1 |
| 1 | 400 | 68.91% | 49.0 | 22.1 | 1129 | 2.2 |
| 2 | 1 | 67.82% | 8.4 | 33.3 | 175 | 1.3 |
| 2 | 20 | 81.17% | 54.6 | 150.0 | 175 | 5.5 |
| **2** | **100** | **82.58%** | **136.4** | **216.2** | **175** | **9.9** |
| 2 | 400 | 82.60% | 193.8 | 237.1 | 175 | 12.2 |

**Three regimes:**

1. **r=0 (exact-only): 36.99%.** 4 797 queries (48%) are empty because their signature doesn't exactly match any training prototype. The 52% that find an exact match get ~71% accuracy on their own. Matches the Axis 4c atomic probe's exact-match fraction exactly.
2. **r≤1: 68.90%.** Adding cost-1 neighbors catches most queries. 1 129 (11%) remain empty. Matches the probe's min-Hamming ≤ 2-bits histogram.
3. **r≤2: 82.58%.** Only 175 queries (1.75%) remain empty. The consumer has now saturated the radius budget at which ~99% of queries have any reachable neighbor.

Best operating point: `MAX_R=2, MIN_C=100 → 82.58%, 9.9 μs/query, 136 candidates avg, 216 probes avg`. Going to MIN_C=400 adds 0.02 accuracy points for 2.3 μs extra — negligible.

### Cost comparison against the dense baseline

| architecture | popcount_dist calls/query | μs/query | accuracy |
|---|---|---|---|
| dense L50_H1 (scan 60K H1, top-50, H2+H3+H4 resolver) | **60 150** | **~1 950** | 83.86% |
| **routed bucket** (MAX_R=2, MIN_C=100) | **~410** | **9.9** | 82.58% |
| ratio | **~147×** fewer popcount ops | **~197×** faster wall time | −1.28 points |

The routed bucket consumer issues zero `popcount_dist` calls at the filter stage. All filter work is binary search + multi-probe enumeration on the sorted bucket table. The resolver runs `popcount_dist` only over the small candidate set (avg 136 prototypes). Wall time is dominated by the multi-probe enumeration, not the resolver.

### The radius profile matches the Axis 4c atomic probe — exactly

Queries stopped at each radius:

| radius | queries | fraction | predicted by probe (min Hamming ≤ corresponding bits) |
|---|---|---|---|
| r=0 | 5 203 | 52.03% | 52% at min_d = 0 ✓ |
| r=1 | 3 668 | 36.68% | ~90% at min_d ≤ 2 bits (cumulative ~89%) ✓ |
| r=2 | 954 | 9.54% | ~99% at min_d ≤ 4 bits (cumulative ~98.25%) ✓ |
| empty at r=2 | 175 | 1.75% | ~1% at min_d > 4 bits ✓ |

The bucket consumer is literally measuring what the atomic probe predicted. Every query's radius of resolution is a direct consequence of the H1 signature codebook's collision structure, which is exactly what the probe characterized. **The dense outer loop in the cascade tools was buying nothing that ternary multi-probe doesn't buy directly.**

### Where the 1.28-point gap lives

Routed bucket at 82.58% vs dense L50_H1 at 83.86% = −1.28 points. Two contributors:

1. **175 empty queries at r=2 (dominant).** Queries whose nearest training signature is at Hamming ≥ 3 cannot be reached by the current radius budget. At default-to-miss they contribute up to ~1.58 points of accuracy loss vs an architecture that would have scanned them anyway. Closing this requires r=3 (more probes per query) or fused-filter bucket (see below).
2. **Candidate-set shape differences (minor).** Dense L50_H1 always hands the resolver exactly 50 candidates ordered by H1 Hamming; routed bucket hands it a variable-size set (avg 136) that is the *union* of all prototypes within Hamming radius 2. For most queries the bucket set is a superset of dense's top-50, so this contributes very little.

### New architectural rule

Adding to the cascade rules list (rules 1–6 in Axis 4d):

**7. Production k-NN uses the signature as an address, not as an operand.** Build a bucket index keyed on the packed-trit signature; query via binary search + ternary multi-probe; run the resolver only on the candidate set. The `O(N_train)` outer loop is scaffolding for research measurements, not the architecture. This rule *subsumes* the filter-ranker reframe: the filter now does zero distance work — it performs a table lookup. Distance computation is reserved for the resolver, over the small candidate set.

### What this settles, and what it doesn't

**Settled by this measurement:**

- The signature-as-address architecture works at the expected speed: ~200× faster than dense at matched accuracy on deskewed MNIST at N_PROJ=16.
- Ternary multi-probe enumeration is correct — the radius escalation profile exactly matches the atomic probe's min-Hamming histogram.
- The dense outer loop in every prior cascade tool is demoted from "routing architecture" to "measurement scaffolding." The tools remain useful for research experiments but are not the production path.
- The 1.28-point gap at radius budget r≤2 is dominated by 175 filter-miss queries, not by any structural deficit of the bucket approach.

**Not yet settled:**

- **Fused-filter bucket.** Concatenate H1+H2 into 8-byte keys; bucket on the concatenation. Expected ~87–89% tracking the L50_H12 dense result. This applies the Axis 4d information-leverage rule to the routed architecture. Not yet built.
- **N_PROJ scaling.** At larger N_PROJ the collision rate drops and most buckets become singletons, which is efficient for lookup but may require larger radius budgets for recall. Unmeasured.
- **Multi-seed variance.** Single seed only so far.
- **Larger training sets.** At 60K the bucket index is 937 KB. Scaling to 1M prototypes or beyond is unmeasured.
- **Online addition.** The index is static at training time. Online prototype addition requires an append-friendly structure (append-only table with periodic re-sort, or hash map).

### Full writeup

See `journal/routed_bucket_consumer.md` for the complete derivation, code snippets, architectural framing, and follow-up list.

## Axis 6 — Multi-table routed bucket LSH: breaking 97% at N_PROJ=16

Added 2026-04-15. This axis extends the Axis 5 signature-as-address architecture to M independent bucket tables, each keyed on a different-seeded 16-trit ternary random projection. Per-table ternary multi-probe provides local recall; cross-table union-merge provides global coverage; a summed-distance resolver scores the union. **At M=32 the architecture reaches 97.24% accuracy on deskewed MNIST — the first routed architecture in Glyph to exceed 97% at N_PROJ=16.** At M=64 it reaches 97.31%. Zero dense scans anywhere in the pipeline.

### The LMM cycle that produced this

Full four-file LMM cycle in `journal/break_97_nproj16_{raw,nodes,reflect,synthesize}.md`. The user's hypothesis was "local and global routing with the Trit Lattice LSH can break 97% at N_PROJ=16." The cycle committed to Reading A (classical multi-table LSH with union-merge) as the concrete mapping of "local + global" to a routing-only substrate, predicted target crossing around M=32, and gated execution on an oracle-pass prerequisite measurement.

### Architecture

```
TRAINING (one-time, ~11 seconds for M=64 on 60K MNIST)
  for m in 0..M:
      W_m = random_ternary_projection(seed_m, N_PROJ=16)
      tau_m = percentile(|W_m @ x|, density=0.33)       // per-table tau calibration
      for i in 0..N_train:
          sig_m[i] = threshold_extract(W_m @ x_train[i], tau_m)
          entries_m[i] = (sig_to_key(sig_m[i]), i)
      qsort(entries_m, by sig_key)                       // ~3 ms per table

QUERY (per test sample, O(1) amortized in N_train)
  clear candidate union (votes[], hit_list[])
  for m in 0..M:
      q_sig_m = threshold_extract(W_m @ query, tau_m)
      for r = 0..2:
          if per_table_candidates(m) >= MIN_CANDS: break
          for each neighbor_key in ternary_neighbors(q_sig_m, r):
              lookup_bucket(entries_m, neighbor_key)
              append matching proto_idxs to union (votes++, hit_list)
  score each candidate in hit_list via SUM resolver:
      score[c] = Σ_m popcount_dist(q_sig_m, candidate_sig_m)
  return label(argmin score)
```

**Local routing** is the per-table multi-probe: a ternary Hamming neighborhood walk (radius 0, 1, 2) producing candidates that live near the query's sig in that table's projection. **Global routing** is the cross-table union-merge: the candidate set is the union of every table's local neighborhood. The operator is set-theoretic, not a new primitive. The resolver reads the composite distance across all M tables and ranks the union.

### Phase 2 — oracle ceiling pass (go/no-go gate)

For each query and each M checkpoint, Phase 2 measured `P(correct class present anywhere in the cross-table union)`:

| M | oracle ceiling | avg union size | avg probes per query |
|---|---|---|---|
| 1 | 94.30% | 94.3 | 194.4 |
| 2 | **97.90%** | 132.5 | 425.9 |
| 4 | 99.75% | 315.9 | 801.1 |
| 8 | 99.99% | 543.6 | 1657.2 |
| 16 | **100.00%** | 1072.8 | 3274.0 |
| 32 | **100.00%** | 1985.8 | 6538.7 |
| 64 | **100.00%** | 3521.2 | 13064.4 |

Full oracle-only sweep wall clock: 5.43 s for 10K queries. **~0.54 ms/query including all probing across M=64.**

**Two observations from Phase 2 that reshaped the whole cycle's expectations:**

1. **M=1 oracle is already 94.30%**, not the ~60-62% I had predicted from extrapolating the pure-signature scaling curve. I had conflated "classification accuracy" (the scaling curve's metric — which is rank-destroyed) with "set-membership in the multi-probe neighborhood" (the oracle ceiling metric — which is neighborhood-preserved). The Axis 4c atomic probe had already measured the right anchor (52% exact-match, 97% at min Hamming ≤ 2 bits), and the Axis 5 filter ceiling at top-50 was 98.59%. **Anchoring Phase 3 predictions on the scaling curve was a mistake; the correct anchor is the atomic probe's Hamming histogram.**

2. **M=2 oracle is 97.90% — already at target.** Two tables is enough that the candidate union contains the correct class for 97.9% of queries. If any resolver could extract classification from that union within 0.9 points of oracle, we'd break 97% at M=2. This changes the question from "how many tables do we need for oracle to reach 97%" (answered: 2) to "how fast does the resolver gap close."

### Correlation analysis from Phase 2

If tables were fully independent, the miss rate would fall geometrically: at M tables, `miss_M = miss_1^M`. Measured vs predicted-under-independence:

| M | predicted miss (indep) | measured miss | correlation factor |
|---|---|---|---|
| 1 | 5.70% | 5.70% | — (baseline) |
| 2 | 0.32% | 2.10% | ~6.5× |
| 4 | 0.0011% | 0.25% | ~230× |
| 8 | ~0% | 0.01% | — |

**Random ternary projections at matched density are moderately correlated**, showing ~6× more shared miss queries at M=2 than fully-independent LSH theory predicts. This is a structural property of the projection family at density=0.33, not a design choice. The architecture still composes well because the miss rate drops geometrically even with correlation — miss events require ALL M tables to miss, and pairwise correlation doesn't fully defeat that.

Closing the correlation gap would require density variation across tables (τ=0.50, τ=0.20 per the Axis 4 resolver sweep) or structurally different hash generators. Not tested in Phase 3.

### Phase 3 — full resolver sweep

With the oracle gate passing decisively, Phase 3 ran the three resolver variants (VOTE, SUM, PTM) at every M checkpoint:

| M | VOTE | SUM | PTM | oracle |
|---|---|---|---|---|
| 1 | 62.96% | 54.50% | 54.63% | 94.30% |
| 2 | 71.82% | 77.78% | 62.20% | 97.90% |
| 4 | 76.75% | 88.91% | 75.34% | 99.75% |
| 8 | 81.83% | 93.84% | 86.07% | 99.99% |
| 16 | 85.78% | 96.13% | 91.48% | 100.00% |
| **32** | 88.50% | **97.24%** | 94.25% | 100.00% |
| **64** | 89.77% | **97.31%** | 95.36% | 100.00% |

Full-sweep wall clock: 68.42 s. Per-resolver cumulative:
- **VOTE: 0.21 s** (pure set-counting, no distance arithmetic)
- **PTM: 25.12 s**
- **SUM: 34.55 s**

**Target crossing at M=32 with the SUM resolver.**

### Resolver variants explained

- **VOTE (set-membership scoring):** For each class `c`, sum the votes for prototypes of class `c` across the union. Vote count per candidate is "how many tables placed this prototype in the query's neighborhood" (weighted by multi-probe reach). Pick the class with the highest total. O(|union|) time. No distance arithmetic.
- **SUM (summed-distance scoring):** For each candidate in the union, compute `Σ_m popcount_dist(q_sig_m, candidate_sig_m)` across all M active tables. Pick the candidate with the smallest sum. This is structurally equivalent to k=1 nearest neighbor under a composite `M × N_PROJ`-trit signature, restricted to the candidate union. O(|union| × M) time.
- **PTM (per-table 1-NN majority):** For each table `m` independently, find the candidate in the union with the smallest popcount_dist in that table's projection. That yields M candidate labels (one per table). Majority vote across the M labels. Per-table 1-NN is "constrained to the union" — table m might have its true 1-NN outside the union, in which case PTM picks the best-in-union-under-m. O(|union| × M) time.

### Sanity checks (from the red-team)

**Check 1: M=1 VOTE = 62.96% ≈ pure N_PROJ=16 k-NN = 62.00%.**
At M=1 the VOTE resolver reduces to majority voting within the multi-probe neighborhood of a single hash — essentially a small generalization of the pure N_PROJ=16 k-NN classifier. The scaling curve predicted 62.00%; measured is 62.96%. **Within 1 point.** The 0.96-point lift is because multi-probe widens the neighborhood beyond a single bucket. **Sanity passes.**

**Check 2: M=1 SUM (54.50%) is lower than M=1 VOTE (62.96%) — the filter-ranker reframe reappears.**
At M=1 the SUM resolver picks the candidate with the smallest table-0 popcount_dist within table-0's own multi-probe union. That's reading the *rank-destroyed* signal the filter-ranker reframe said would underperform (Axis 4). VOTE at M=1 sidesteps the rank signal entirely — it reads class frequency in the neighborhood instead. The 8.5-point gap (VOTE − SUM at M=1) is the filter-ranker asymmetry manifesting as an internal consistency check within this architecture. **Sanity passes; mechanism is consistent with Axis 4.**

**Check 3: matched-hash-budget comparison against the Axis 5 single-table consumer.**
The Axis 5 single-table consumer (Axis 5) used 4 hashes total: table 0 as filter and independent H2/H3/H4 for the resolver. It reached 82.58%. This tool at M=4 uses 4 hashes total, all as filter-and-resolver tables. SUM at M=4 is **88.91% vs 82.58% — +6.33 points from multi-table union-merge at matched information budget.** This is the multi-table architectural win measured directly, on an apples-to-apples hash budget. **Sanity passes; the union-merge structure adds real value beyond the information content.**

### Comparison to the pure-signature scaling curve at matched total bits

Multi-table LSH with M tables of `N_PROJ=16` each has `M × 16` total signature trits per query. The pure-signature scaling curve (Axis 1) measured dense k-NN accuracy at various single-hash signature sizes. Matched-bits comparison:

| architecture | total signature bits | accuracy |
|---|---|---|
| Pure N_PROJ=128 single-hash | 128 | 95.22% |
| Pure N_PROJ=256 single-hash | 256 | 96.56% |
| Pure N_PROJ=512 single-hash | 512 | 97.06% |
| Pure N_PROJ=1024 single-hash | 1024 | 97.37% |
| Pure N_PROJ=2048 single-hash | 2048 | 97.79% |
| Pure N_PROJ=4096 single-hash | 4096 | 97.99% |
| **Multi-table SUM at M=16** | **256** | **96.13%** (−0.43 vs pure N_PROJ=256) |
| **Multi-table SUM at M=32** | **512** | **97.24%** (+0.18 vs pure N_PROJ=512) |
| **Multi-table SUM at M=64** | **1024** | **97.31%** (−0.06 vs pure N_PROJ=1024) |

**At matched total bits, multi-table routed bucket SUM matches or slightly beats the pure scaling curve.** The small bonus at M=32 (+0.18) comes from the independence structure — 32 independent 16-trit random projections each with multi-probe neighborhoods collectively cover more input-space geometry than a single 512-trit random projection does. The bonus vanishes at higher bits (M=64) because the curve flattens.

**This is the structural result of Axis 6.** Multi-table LSH is *not* equivalent to one big hash at the same bits — it's slightly better due to independence — but the improvement is modest (<1 point in the measured regime) and concentrated in the middle of the curve.

### Resolver gap analysis

The resolver gap is `oracle − best_resolver_accuracy` at each M. It measures how much accuracy is lost because the resolver cannot extract the correct class from the union even though the correct class is present:

| M | oracle | best resolver (SUM) | gap |
|---|---|---|---|
| 1 | 94.30% | 62.96% (VOTE) | 31.34 |
| 2 | 97.90% | 77.78% (SUM) | 20.12 |
| 4 | 99.75% | 88.91% (SUM) | 10.84 |
| 8 | 99.99% | 93.84% (SUM) | 6.15 |
| 16 | 100.00% | 96.13% (SUM) | 3.87 |
| 32 | 100.00% | 97.24% (SUM) | 2.76 |
| 64 | 100.00% | 97.31% (SUM) | **2.69** |

**The gap shrinks from 31.34 (M=1) to 2.76 (M=32), then plateaus.** M=64 barely improves on M=32 (−0.07 gap). This is the **structural ceiling of random-ternary-SUM ranking** on deskewed MNIST: ~2.7% of queries have the correct class in the multi-probe union, but summed popcount-Hamming ranks a wrong-class prototype higher. These are queries whose correct prototype is "far" in composite signature space even though the filter neighborhoods collectively contain it.

Closing this gap requires architectural moves beyond "more tables":
1. **Density variation across tables** — partial success on some confusion pairs (6↔8) per Axis 4 resolver sweep, no effect on others (3↔5, 4↔9).
2. **Structurally different hash generators** (not just different seeds) — untested.
3. **Learned projections** — explicitly out of scope (no supervised training in the routing substrate).
4. **Richer resolver metric** (weighted sum, per-table normalization, calibrated distance) — an interesting micro-experiment I haven't run.

### Resolver behavior — SUM dominates, VOTE plateaus weak, PTM middles

**SUM is the best resolver at every M ≥ 2.** At M=32 SUM (97.24%) beats VOTE (88.50%) by 8.74 points and PTM (94.25%) by 2.99 points. The gap between SUM and VOTE widens with M, not shrinks.

**VOTE's structural ceiling:** set-membership voting counts how many tables placed each candidate in the query's neighborhood, weighted by vote count. It discards the underlying distance information — a candidate at Hamming 0 in every table counts the same as a candidate at Hamming 2 in every table, provided both show up in every union. SUM preserves the distance gradient; VOTE collapses it. Ceiling of VOTE at M=64 is 89.77%, about 7.5 points below SUM.

**PTM's middle position:** per-table 1-NN produces a noisy estimate; majority-voting M noisy estimates loses to summing M distances because summation preserves the gradient while voting collapses it. PTM at M=64 is 95.36%, about 2 points below SUM.

**Unexpected finding (red-team): VOTE does not capture the multi-table architecture's strength.** My synthesize-phase prediction suggested VOTE might beat SUM at low M because "tie-breaking across 32 tables is a strong signal." Empirically it never beats SUM at M ≥ 2 and the gap widens. Set-membership voting is architecturally weaker than summed-distance ranking in this measurement. Naming the surprise because it contradicts the pre-measurement intuition and should recalibrate priors for related future designs.

### Cost-accuracy at operating points

Per-query cost estimates, broken down into probing work (binary-search + multi-probe enumeration) and resolver work (summed-distance computation):

| M | accuracy (SUM) | probing ms | SUM resolver ms | total ms/query | notes |
|---|---|---|---|---|---|
| 1 | 54.50% | 0.02 | 0.002 | ~0.02 | too small for production use |
| 2 | 77.78% | 0.04 | 0.005 | ~0.05 |  |
| 4 | 88.91% | 0.08 | 0.025 | ~0.10 |  |
| 8 | 93.84% | 0.17 | 0.087 | ~0.26 |  |
| **16** | **96.13%** | 0.33 | 0.343 | **~0.67** | **cost sweet spot below target** |
| **32** | **97.24%** | 0.65 | 1.27 | **~1.92** | **target crossing** |
| 64 | 97.31% | 1.31 | 2.82 | ~4.13 | diminishing returns |

**The SUM resolver cost dominates at high M.** Per-query SUM work is `n_hit × M × popcount_dist`. At M=32 with `n_hit≈1986` and ~20 ns per popcount_dist on packed 4-byte trit codes, that's ~1.27 ms resolver work plus ~0.65 ms probing work = ~1.92 ms/query total.

### Reference baselines for context

| architecture | accuracy | ms/query | substrate |
|---|---|---|---|
| Axis 5 single-table bucket (H2+H3+H4 resolver) | 82.58% | ~0.01 | routed |
| Dense L50_H1 (Axis 4a) | 83.86% | ~1.95 | dense scaffolding |
| Dense L50_H12 (Axis 4d) | 88.44% | ~1.95 | dense scaffolding |
| Dense L200_H12 (Axis 4d) | 88.87% | ~1.95 | dense scaffolding |
| Dense Gq (global 4-hash) | 89.46% | ~3.9 | dense scaffolding |
| Pure N_PROJ=512 dense scan | 97.06% | ~4.0 | dense scaffolding |
| Pure N_PROJ=2048 dense scan (Axis 2 headline) | 97.79% | ~7 | dense scaffolding |
| **Multi-table M=16 SUM** | **96.13%** | **~0.67** | **routed** |
| **Multi-table M=32 SUM** | **97.24%** | **~1.92** | **routed** |
| **Multi-table M=64 SUM** | **97.31%** | **~4.13** | **routed** |

**M=32 SUM matches the wall time of dense L200_H12 (~1.95 ms) while being +8.37 accuracy points higher.** It matches pure N_PROJ=512's accuracy while being ~2× faster. All with zero dense paths. **This is the new routed headline at N_PROJ=16.**

### Architectural rule 8 — multi-table composition

Extending the Axis 4d + Axis 5 rule list with a new rule:

**Rule 8. Multi-table composition reproduces the scaling curve at equivalent total bits, with a small independence bonus.** The signature-as-address architecture (Axis 5 rule 7) composes through M independent bucket tables via union-merge at the global step and summed-distance resolver at the scoring step. At matched total bits (`M × N_PROJ` = equivalent single-hash `N_PROJ`), multi-table SUM matches the pure scaling curve within noise and may gain up to ~0.2 points from independence structure. This gives a concrete allocation policy for building routed k-NN consumers at any target accuracy on the scaling curve: pick a base `N_PROJ` small enough for cheap per-table operations (N_PROJ=16 is a sweet spot given 4-byte signatures fit in one uint32 binary-search key), then compose M tables until matched-bits accuracy hits the target. Wall-time cost of the resulting architecture is ~2× faster than the equivalent pure dense scan because the filter stage never touches all N_train prototypes.

Rule 8 subsumes Axis 5's rule 7 (signature-as-address) by showing that the single-table case (M=1) is a special case of the multi-table composition, and the multi-table case is how you reach arbitrary points on the scaling curve without exceeding a fixed per-table signature size.

### What Axis 6 does *not* settle

1. **Per-class accuracy at M=32 SUM.** The 97.24% is an average — some digits may still be weaker. Phase 4 would measure per-class to verify no digit is pathologically bad.
2. **Multi-seed variance.** Single seed only so far. Multi-seed error bars would confirm the ±0.5 point stability assumption.
3. **Direct correlation measurement.** The ~6× correlation factor was derived from oracle miss rates, not from direct per-table disagreement counts. A direct measurement would verify.
4. **Density variation at mid M.** Replacing some density-0.33 tables with density-0.50 / 0.20 variants at M=16 or M=32 — does it push the accuracy up further at matched M?
5. **Radius ablation.** Does r ≤ 1 suffice at large M because the union is already saturated? Cheaper queries if so.
6. **Other datasets.** Deskewed MNIST only. CIFAR-10, FashionMNIST, and other benchmarks are open.
7. **Larger training sets.** At N_train = 60K the bucket indexes fit comfortably; scaling to 1M prototypes changes the bucket density distribution and may change multi-probe recall.

### Full writeups

- LMM cycle: `journal/break_97_nproj16_{raw,nodes,reflect,synthesize}.md`
- Phase 3 results (with full red-team): `journal/break_97_nproj16_phase3_results.md`
- Tool: `tools/mnist_routed_bucket_multi.c`

## Axis 7 — Fashion-MNIST generalization + resolver-gap atomics

Date: 2026-04-15. Tool: `tools/fashion_atomics.c`, `tools/mnist_routed_bucket_multi.c`. Journals: `journal/fashion_mnist_first_light.md`, `journal/fashion_mnist_atomics.md`, `journal/fashion_mnist_density_sweep.md`.

### First light

The Axis 6 architecture generalizes to Fashion-MNIST without code changes — only `--data` and `--no_deskew` differ. M=64 SUM at density 0.33: **85.15%** (matching classical pixel k-NN baselines for this dataset). Oracle is 100% at M≥16 so the correct neighbor is always in the union; the ~15% failure is entirely downstream of filtering.

The resolver gap (oracle minus SUM) is ~6× wider on Fashion-MNIST (14.85pp) than on MNIST (2.69pp at M=64). The gap is concentrated in the upper-body-garment cluster: classes 0 (T-shirt), 2 (Pullover), 4 (Coat), 6 (Shirt), which share similar fabric-filled silhouettes.

### Phase A — vote-weighted SUM (falsified)

`glyph_resolver_sum_voteweighted`: scores each candidate as `sum_dist / (1 + votes[c])`. Hypothesis: folding filter-stage vote count into the resolver ranking should help on datasets where per-table signature distance is noisy. Result: neutral on MNIST, slightly harmful on Fashion-MNIST. Falsified.

However, the per-class instrumentation added for Phase A revealed the concentration of Fashion-MNIST errors in the upper-body cluster — the key diagnostic finding of this axis.

### Phase B.1 — radius-aware SUM (falsified)

`glyph_resolver_sum_radiusaware`: scores each candidate as `sum_dist + λ × min_radius[c]`. Hypothesis: penalizing candidates found only via deep multi-probe expansion should close the gap. Result: monotone degradation as λ increases (Fashion-MNIST 85.15% at λ=2, 85.02% at λ=8, 84.82% at λ=16, 84.49% at λ=32). Multi-probe radius is a coarsening of information already present in sum_dist.

### Atomics — where the lattice fails

Three measurements decompose the 1485 Fashion-MNIST failures at M=64:

**Atom 1 (rank & gap).** 36.8% of failures have the true-class best prototype at rank 1 (runner-up). But only 6.6% are within 1 Hamming unit of the winner. Mean gap: 17.64 Hamming units. Most failures are not tiebreaks — the wrong class is genuinely closer in summed lattice distance.

**Atom 2 (per-table vote).** Mean per failing query: true class 19.5/64 (30.5%), winner 22.5/64 (35.1%), other 22.0/64 (34.4%). The plurality of individual tables already votes for the wrong class. Fusion is not breaking good signal — it is faithfully summing bad signal.

**Atom 3 (per-table sig-distance gap).** Mean (d_winner − d_true) = −0.036 Hamming bits. 65% of (query, table) pairs are tied at the per-table min-Hamming level. The per-table ternary projection cannot discriminate upper-body classes — a fraction-of-a-bit bias accumulates over M=64 tables.

**Magnet audit.** 1361 distinct training prototypes won at least one of 1485 failures. Top-20 share: 4.1%. No pathological prototypes. The gap is structural, not example-driven.

### Phase B.2 — density-varied multi-table sweep (falsified, side win)

Hypothesis: mixing different projection densities across tables should diversify the lattice faces and break the 65% tied-gap rate. Two spreads tested:

| config | Fashion-MNIST M=64 SUM |
|---|---|
| fixed d=0.33 (baseline) | 85.15% |
| mixed wide {0.20, 0.33, 0.50} | 84.57% (−0.58) |
| mixed narrow {0.25, 0.33, 0.40} | 85.13% (−0.02) |
| fixed d=0.25 | **85.54%** (+0.39) |

Both mixes are strictly dominated by the single best density. Atomics under mixed mode confirmed the mechanism: tied-gap rate INCREASED from 65.0% to 67.7%. Density mixing smeared discriminative signal instead of diversifying it — different densities sample the same pixel population so they produce overlapping, not disjoint, projections.

**Multi-seed confirmation (3 seeds × 5 densities × 2 datasets):**

Fashion-MNIST density ranking is perfectly consistent across all 3 seeds: 0.25 > 0.33 > 0.20 > 0.40 > 0.50. The 0.25 > 0.33 gap (+0.30pp mean) survives with p<0.02.

MNIST density ranking: 0.33 > 0.25 ≈ 0.20 > 0.40 > 0.50. MNIST prefers 0.33 (97.35% mean vs 97.16% at 0.25).

**Finding: density is a dataset-dependent hyperparameter.** MNIST's sparse pen-stroke foreground rewards denser projections (0.33); Fashion-MNIST's dense fabric foreground rewards sparser projections (0.25).

### What Axis 7 does *not* settle

1. Whether the upper-body-cluster gap can be closed without preprocessing changes (N_PROJ=32 or block-structured spatial projections are untested).
2. Whether Shirt (class 6) magnetism is a spatial or textural phenomenon.
3. Whether the density optimum shifts at different M values.
4. CIFAR-10 and other non-MNIST benchmarks.

### Full writeups

- Fashion-MNIST first light: `journal/fashion_mnist_first_light.md`
- Atomics diagnosis: `journal/fashion_mnist_atomics.md`
- Density sweep: `journal/fashion_mnist_density_sweep.md`
- Diagnostic tool: `tools/fashion_atomics.c`

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
- **The Trit Lattice signature is a lossy locality hash, not a classifier.** Ceiling@50 = 98.6% at N_PROJ=16; top-1 = 55.5%. Voting reads the destroyed rank information; the cascade reads the preserved set membership. Verified twice: with a dense pixel resolver (rescue/damage 26:1, +30-point lift) and with a routed secondary-hash resolver (rescue/damage 5:1, +15-point lift). Same mechanism, different resolver ceilings. See `journal/cascade_atomics_mechanism.md`, `journal/cascade_sweep_crossover.md`, and `journal/routed_cascade_rerun.md`.
- **Filter-ranker reframe is substrate-invariant.** The hash is a filter regardless of what reads it. Resolver choice sets the absolute ceiling (routed ~97.5%, pixel-L2 ~97.6% on deskewed MNIST); it does not change whether the cascade helps.
- **Observability ceiling at N_PROJ=16.** The local-vs-global contingency at N_PROJ=16 has a 92.77% oracle ceiling, but rescues and damages share distributions on every inference-available signal we measured. Meta-routing based on disagreement/margin/tied-count is bounded at ~88% — ~1.5 below pure global, ~4.8 below the oracle. The gap is not a design problem; it's a structural information limit at this N_PROJ. See `journal/lvg_atomics_decomposition.md`.
- **Information leverage is filter-stage-first.** At N_PROJ=16 on deskewed MNIST, moving H2 from a resolver to half of the filter lifts accuracy from 83.86% to 88.44% — a +4.58 point gain from reallocating *existing* information to the filter stage. No new hashes, no new primitives, same arithmetic. The filter-ranker reframe has a dual: once you're spending information routing-style, allocate it earliest. L200_H12 (fused filter + widened K) reaches 88.87% at 2× baseline cost and within 0.59 points of pure global Gq (89.46%) at 50% of Gq's cost. See `journal/fused_filter_fix.md`. This also deprecates the meta-router cycle as a practical architecture: the fix the LMM cycle was supposed to enable is unnecessary because the filter-stage fix closes 89% of the gap directly.
- **The Trit Lattice signature is an *address*, not an operand.** Every cascade tool in the tree (built 2026-04-14 through early 2026-04-15) runs routing primitives inside an `O(N_train)` dense outer loop: for each query, `popcount_dist` is called against every training signature. That is dense architectural shape with routed kernels — a substrate violation even though the kernels themselves are routing-native. The first genuinely routed consumer, `tools/mnist_routed_bucket.c`, uses the 4-byte H1 signature as a key into a sorted bucket table and looks it up via binary search + ternary multi-probe. At N_PROJ=16 on deskewed MNIST the routed bucket reaches 82.58% at 9.9 μs/query, versus dense L50_H1 at 83.86% / ~1950 μs/query — **~147× fewer `popcount_dist` calls, ~197× faster wall time, −1.28 points of accuracy.** The accuracy gap is dominated by 175 queries (1.75%) whose nearest training signature exceeds the r≤2 radius budget. Radius escalation profile (52% r=0 / 37% r=1 / 10% r=2 / 2% empty) matches the Axis 4c atomic probe's Hamming histogram exactly. See `journal/routed_bucket_consumer.md`. **This reframes the existing cascade tools as research scaffolding, not production architecture.**
- **Multi-table composition reproduces the scaling curve at equivalent total bits.** At N_PROJ=16 on deskewed MNIST, M=32 independent bucket tables with ternary multi-probe and summed-distance resolver reach **97.24%** accuracy — the first routed architecture in Glyph to exceed 97%. At M=64 the accuracy is 97.31%. At matched total bits (`M × N_PROJ`), multi-table matches or slightly beats the pure-signature scaling curve: M=32 (512 total bits) reaches 97.24% vs pure N_PROJ=512 at 97.06% (+0.18 from independence); M=64 (1024 bits) reaches 97.31% vs pure N_PROJ=1024 at 97.37% (within noise). Per-query wall time is ~1.92 ms at M=32 — about 2× faster than pure N_PROJ=512 dense scan at matched accuracy. Random ternary projections at matched density are moderately correlated (~6× miss-rate vs fully-independent LSH theory at M=2), but the miss rate still drops geometrically because correlation is pairwise while miss events require ALL M tables to miss. The resolver gap to oracle plateaus at ~2.7 points at M ≥ 32 — a structural ceiling of random-ternary-SUM ranking. **The signature-as-address rule composes through multi-table union-merge, and the composition provides a direct knob for reaching any target on the scaling curve without exceeding a fixed per-table signature size.** See `journal/break_97_nproj16_phase3_results.md`.

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

### For the cascade architecture (Axis 4 + 4b)
- `journal/nproj16_atomic_mechanism.md` — atomic probe at N_PROJ=16; partition asymmetry explains vote-rule inversion.
- `journal/nproj16_to_90_{raw,nodes,reflect,synthesize}.md` — LMM cycle on "can N_PROJ=16 reach 90%?" — the filter-ranker reframe.
- `journal/nproj16_cascade_result.md` — historical cascade at N_PROJ=16 (dense resolver) hits 92.72%.
- `journal/cascade_atomics_mechanism.md` — historical decomposition with pixel resolver; rescue:damage 26:1.
- `journal/cascade_sweep_crossover.md` — historical cascade across N_PROJ with pixel resolver; crossover at 512.
- `journal/routed_cascade_rerun.md` — routed-only cascade rerun: 77.33% at N_PROJ=16 with H2, 81.35% with H2+H3 fusion; crossover at 256; mechanism preserved.
- `journal/routed_quadruple_decorrelation.md` — H2+H3+H4 quadruple fusion hits 83.86% at N_PROJ=16; density decorrelation partial.
- `journal/meta_router_online_{raw,nodes,reflect,synthesize}.md` — LMM cycle on an online, inline meta-router as k-NN over routing-context signatures.
- `journal/lvg_atomics_decomposition.md` — atomic decomposition of L vs Gq contingency; observability ceiling at ~88% identified.
- `journal/fused_filter_fix.md` — fused-filter fix rerun: L50_H1 (83.86%) → L50_H12 (88.44%) from moving H2 to filter stage; L200_H12 reaches 88.87% within 0.59 of Gq; meta-router deprecated.
- `journal/routed_bucket_consumer.md` — first genuinely routed consumer: signature as address, bucket index + ternary multi-probe at O(1) amortized; 82.58% at 9.9 μs/query; cascade tools reframed as measurement scaffolding.
- `journal/break_97_nproj16_{raw,nodes,reflect,synthesize}.md` — LMM cycle that designed the multi-table architecture.
- `journal/break_97_nproj16_phase3_results.md` — Axis 6 full measurement: multi-table bucket SUM at M=32 reaches 97.24%, at M=64 reaches 97.31%, first routed architecture to exceed 97% at N_PROJ=16.

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

- Whether the routing thesis generalizes beyond Fashion-MNIST to CIFAR-10 or other benchmarks.
- Whether gradient-free base-3 training is viable at scale.
- Whether the 20× speedup holds on AMD Zen 4 or x86-AVX-512 targets. We run aarch64 + NEON only; other silicon has different instruction shapes.
- Whether the specific configurations tested (N_PROJ ∈ {512, 2048}, density = 0.33, k ∈ {1, 3, 5}) are optimal. Deeper sweeps are cheap and not yet run.

Each of those is an open experiment. The measurements here are the platform for running them, not a declaration that they are run.

---

**Summary of this document's purpose:** if you came to this repo cold and wanted to know "what did they actually measure and what does it mean" without piecing it together from a commit log, this is the one page to read. NORTH_STAR is the compass; this is the logbook.
