---
date: 2026-04-15
scope: Routed-only cascade rerun on MNIST after dense-resolver remediation
type: mechanism + sweep
tools: tools/mnist_cascade_nproj16.c, tools/mnist_cascade_sweep.c, tools/mnist_cascade_atomics.c, tools/mnist_resolver_sweep.c
parent: journal/cascade_atomics_mechanism.md (historical, dense resolvers)
---

# Routed cascade on MNIST — filter-ranker reframe survives end-to-end routing

Sixth-round remediation removed the dense pixel resolvers (`pixel_l1`, `pixel_l2sq`, `cosine_sim`, per-class pixel centroids) from the cascade tools and replaced them with routed variants (secondary hash, tertiary hash, dual-hash fusion, per-class routed nearest). This is a rerun of the cascade story on deskewed MNIST with routing maintained end-to-end. Headline: **the filter-ranker reframe holds; the absolute gains are smaller and the ceiling is lower than the dense-resolver version, but the mechanism is identical.**

## Headline numbers (deskewed MNIST, density=0.33, K_RESOLVE=50)

Single primary seed 42, secondary seed 1337, tertiary seed 1009.

### Cascade across N_PROJ (H2 1-NN routed resolver)

| N_PROJ | pure top-1 | pure maj k=7 | ceiling@50 | routed cascade | Δ (casc − maj) | Δ (casc − top-1) |
|---|---|---|---|---|---|---|
| 8 | 33.52% | 38.74% | 97.60% | **54.21%** | **+15.47** | +20.69 |
| 16 | 55.48% | 62.00% | 98.59% | **77.33%** | **+15.33** | +21.85 |
| 32 | 76.10% | 80.75% | 99.46% | 89.25% | +8.50 | +13.15 |
| 64 | 89.81% | 91.55% | 99.77% | 93.87% | +2.32 | +4.06 |
| 128 | 94.41% | 95.22% | 99.77% | 95.67% | +0.45 | +1.26 |
| 256 | 96.37% | 96.56% | 99.85% | 96.44% | **−0.12** | +0.07 |
| 512 | 97.04% | 97.06% | 99.89% | 96.98% | −0.08 | −0.06 |
| 1024 | 97.37% | 97.43% | 99.90% | 97.08% | −0.35 | −0.29 |
| 4096 | 97.69% | 97.65% | 99.86% | 97.41% | −0.24 | −0.28 |

**Practical crossover at N_PROJ≈256** (first negative Δ over pure majority). Earlier than the dense-resolver crossover at N_PROJ=512 — the routed resolver hits its own ceiling sooner because a second 16-bit hash adds less signal than pixel access did.

### Cascade at N_PROJ=16 (detailed per-K, single-seed)

| K | pure maj | cascade H2 1-NN | cascade dual-hash 1-NN |
|---|---|---|---|
| 5 | 61.16% | 69.07% | 68.77% |
| 10 | 62.92% | 72.24% | 72.11% |
| 20 | 64.14% | 74.84% | 74.77% |
| 50 | 63.31% | **77.33%** | 76.70% |
| 100 | 63.00% | 78.47% | 78.03% |

Ceiling at top-100: 99.50%. Routed cascade at K=50 lifts 62% → 77.33%, a +15.33-point gain over pure-hash majority using only routing primitives on the 16-bit lattice.

### Resolver sweep (best routed resolver per N_PROJ)

| Resolver | N_PROJ=16 | N_PROJ=128 | N_PROJ=1024 |
|---|---|---|---|
| R1 H2 1-NN | 77.33% | 95.67% | 97.08% |
| R2 H1+H2 1-NN (dual-hash) | 76.70% | 96.31% | 97.29% |
| R3 H2 3-NN majority | 78.16% | 96.32% | 97.52% |
| R4 H2 5-NN majority | 78.71% | 96.39% | 97.51% |
| R5 H2 7-NN majority | 78.51% | 96.25% | 97.39% |
| R6 H2 5-NN rank-wt | 79.43% | 96.50% | **97.53%** |
| R7 H2 5-NN dist-wt | 79.22% | 96.42% | 97.50% |
| R8 H3 1-NN | 73.21% | 95.60% | 97.40% |
| **R9 H2+H3 1-NN (triple-hash fusion)** | **81.35%** | 96.40% | 97.39% |
| R10 per-class nearest H2 | 76.70% | 95.56% | 97.08% |
| R11 H1+H2 rank hybrid | 71.43% | 95.96% | 97.46% |

**Best routed resolver at N_PROJ=16: triple-hash fusion (H2+H3 1-NN) at 81.35%.** At small N_PROJ, adding independent routed views pays — this is the routed-native analogue of "reach for a richer resolver."

**At large N_PROJ the resolver choice barely matters:** all variants sit within 0.4 points at N_PROJ=1024, and none materially beats pure k=7 majority (97.43%). The routed cascade ceiling on deskewed MNIST is ~97.5%, about 0.1 point below the pure pipeline's 97.65% at N_PROJ=4096.

## Atomic decomposition — mechanism is identical

Rerun of `mnist_cascade_atomics` on routed-only cascade at N_PROJ=16, K=50:

### A. Filter ceiling curve (unchanged from dense run — the filter is the same)

| K | correct in top-K |
|---|---|
| 1 | 55.48% |
| 10 | 90.81% |
| 50 | 98.59% |
| 100 | 99.50% |
| 200 | 99.86% |

### B. Conditional resolver: secondary-hash 1-NN given correct in top-K

| K | conditional rate |
|---|---|
| 1 | 100.00% |
| 10 | 79.55% |
| 50 | **78.44%** |
| 100 | 78.86% |
| 200 | 78.88% |

Compared to pixel-L2's ~91% conditional rate: routed resolver is ~13 points weaker per query conditional on the pool containing the answer. This is where the gap lives.

### C. Rescue/damage matrix at K=50

|  | cascade right | cascade wrong |
|---|---|---|
| pure-hash top-1 right | 4990 | 558 |
| pure-hash top-1 wrong | **2743** | 1709 |

- Rescued: 2743 (27.43%).
- Damaged: 558 (5.58%).
- **Rescue:damage ratio ≈ 5:1** (vs 26:1 with pixel resolver).

Still asymmetric, still net-positive, but four times noisier. The routed resolver misjudges ~14% of the cases where the hash got top-1 right, whereas pixel L2 only misjudged 2.5%. This is the "more signal" gap made quantitative.

### D. Hash-rank distribution of cascade's correct picks

| hash-rank bucket | fraction |
|---|---|
| rank 1 (top-1) | 5.55% |
| rank 2 | 4.67% |
| ranks 3–5 | 11.26% |
| ranks 6–10 | 15.09% |
| ranks 11–20 | 23.76% |
| **ranks 21–50** | **39.67%** |

Same shape as the dense rerun (which had 50.72% in ranks 21–50). The routed resolver still pulls most correct picks from deep in the hash ranking — the hash's filter role is preserved, the resolver's ranking role is weaker.

### E. Per-partition cascade accuracy

| Partition (from top-10) | count | routed cascade | dense cascade (historical) |
|---|---|---|---|
| correct in tied-min set | 7519 | 85.90% | 95.96% |
| correct elsewhere in top-10 | 1562 | 64.66% | 86.94% |
| correct nowhere in top-10 | 919 | 28.73% | 54.62% |

Every partition takes a ~10-20 point hit from the resolver switch. The elsewhere-in-top-10 partition, which needs the most resolver work, shows the biggest drop (−22 points). Predictable: pixel access was doing the heaviest lifting in exactly the cases where hash-rank-1 was wrong.

### F. Class-pair confusion — digit 0 still dominates, but new regressions appear

Top routed-cascade improvements (pure-hash k=7 → routed cascade K=50):

| true | pred | pure err | cascade err | Δ |
|---|---|---|---|---|
| 8 | 0 | 149 | 24 | +125 |
| 5 | 0 | 120 | 58 | +62 |
| 6 | 0 | 100 | 41 | +59 |
| 0 | 6 | 98 | 42 | +56 |
| 0 | 5 | 87 | 36 | +51 |

Digit 0 remains the hash's error sink; secondary hash partially resolves it.

New regressions (absent in the dense cascade):

| true | pred | pure err | cascade err | Δ |
|---|---|---|---|---|
| 3 | 8 | 46 | 74 | −28 |
| 3 | 5 | 43 | 67 | −24 |
| 6 | 8 | 21 | 35 | −14 |

The routed resolver confuses digits whose ternary projections look similar under both H1 and H2 (3↔8, 3↔5, 6↔8). Pixel L2 broke these confusions; secondary hash does not. This is direct evidence that the two views are not fully independent — a second random ternary projection shares more failure modes with the first than an orthogonal modality would.

### G. Secondary-hash margin

Average relative margin `(d_wrong − d_correct) / (d_wrong + d_correct + 1)` within top-50, across 9301 queries: **+0.2056**.

Positive, but smaller than pixel L2's +0.3255. This is the quantitative reason the routed cascade is weaker: the "correct is closer" signal is still there, but thinner.

## Mechanism in one sentence

**The filter-ranker reframe is substrate-invariant — a 16-bit hash is always a better filter than a ranker — but the ranker's upper bound depends on the signal it gets to rank with.** Pixel-L2 on a filtered pool has +0.33 margin and ~91% conditional accuracy; a secondary 16-bit hash on the same filtered pool has +0.21 margin and ~78% conditional accuracy. Fold this into the filter-presence product and the architectural ceilings come out: `99.87% × 97.7% ≈ 97.6%` for pixel, `99.87% × 78.9% ≈ 78.8%` for H2 alone. Triple-hash fusion (H2+H3) raises the effective conditional rate at small N_PROJ.

## Why the routed cascade still matters

1. **It stays inside the routing budget.** No float, no pixel access, no dense array scan on the classifier path. Honors NORTH_STAR.
2. **Filter-ranker reframe survives.** +15 points over pure-hash majority at N_PROJ=16 using nothing but signature operations.
3. **Triple-hash fusion is a clean routed-native lever.** At small N_PROJ, adding independent routed views buys 3-5 points. Stackable in principle.
4. **Large-N_PROJ regime is essentially tied with pure k-NN.** Cascade neither helps nor materially hurts at N_PROJ ≥ 512. This means cascade is an "optional" architectural move at that scale, not a forced trade-off.

## Comparison table: routed vs dense cascade at N_PROJ=16

| Metric | Routed cascade | Dense cascade (historical) |
|---|---|---|
| Cascade accuracy at K=50 | 77.33% | 90.75% |
| Cascade accuracy at K=100 | 78.47% | 92.72% |
| Best variant at K=20 | 81.35% (H2+H3) | 86.40% (pixel L2) |
| Rescue count at K=50 | 2743 | 3668 |
| Damage count at K=50 | 558 | 141 |
| Rescue:damage | 5:1 | 26:1 |
| Conditional resolver rate | 78.44% | 92.05% |
| Margin (relative) | +0.2056 | +0.3255 |
| Tied-min partition accuracy | 85.90% | 95.96% |
| Elsewhere-in-top-10 accuracy | 64.66% | 86.94% |

## Scaling prediction, verified

Atomics-derived prediction: cascade headroom shrinks as N_PROJ grows, crossover around N_PROJ=256. Routed sweep confirms: first negative Δ at N_PROJ=256. The resolver difference shifted the crossover point left by one step.

## Follow-ups

1. **Quadruple-hash fusion.** If H2+H3 bought +4.65 points at N_PROJ=16, does H2+H3+H4 buy more? Cheap to test.
2. **Correlated-failure analysis.** 3↔8 and 3↔5 regressions suggest shared failure modes between H1 and H2. Testing whether H3 (a different RNG seed with the same generator) decorrelates them, vs. a structurally different hash generator (e.g. different density), is the next diagnostic.
3. **Per-class signature bank.** Precompute one routed signature per class (mean of class prototypes after routing) and use as resolver — pure-routing analogue of per-class centroid. Fast.
4. **Learned secondary projections.** Optimize the secondary hash for class separation conditional on the primary filter. Still routed at inference, but with a training phase. Larger move.
5. **Apply the `test_routed_tool_smoke.c` pattern.** Every routed consumer should ship with a smoke test using tiny synthetic IDX data so CI covers the routed path end-to-end.

## Pointers

- Tools: `tools/mnist_cascade_nproj16.c`, `tools/mnist_cascade_sweep.c`, `tools/mnist_cascade_atomics.c`, `tools/mnist_resolver_sweep.c`.
- Historical dense-resolver writeup: `journal/cascade_atomics_mechanism.md`, `journal/cascade_sweep_crossover.md`, `journal/nproj16_cascade_result.md`.
- LMM cycle: `journal/nproj16_to_90_{raw,nodes,reflect,synthesize}.md`.
- Original vote-rule probe: `journal/nproj16_atomic_mechanism.md`.
