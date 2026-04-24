---
date: 2026-04-24
scope: substrate_distance_refinement LMM cycle close-out — results + gate decisions
phase: CLOSE
---

# Close-out: substrate_distance_refinement cycle

## Cycle in one paragraph

`base3_go_probe` closed RED: raw trit Hamming on 19×19 Go positions classified phase at 40.40% vs density-only 98.28%. This cycle tested whether substrate-level, non-trainable fixes — density-normalized Hamming and 3×3 local-contrast encoding — close the gap, and whether either approach supports density-controlled retrieval tasks as well. **All three cycle gates passed**: density-normalized Hamming lifted phase-ID from 40.40% to **85.40%** (+45pp); local-contrast encoding lifted it to **73.60%** (+33pp); and raw Hamming on a density-controlled task (same-game retrieval) already scored **24.48% hit-rate at k=50, a 304.8× lift over random**. The original probe RED was not a substrate failure — it was a task-metric interaction. The substrate sees structure. The metric needs one multiplicative fix to see it alongside density.

## Results table — all configurations

All runs on the same 77,795-position corpus from 1,994 games (featurecat 10k sample). 80/20 train/test, test capped at 500 for brute-force k-NN.

### Task: phase-ID (3-class; random = 33.3%, majority-class = 38.4%, density-only k=200 = 98.80%)

| Encoding | Metric | k=50 | k=100 | k=200 |
|---|---|---|---|---|
| raw | hamming | 40.40% | 39.80% | 38.60% |
| raw | **hamming_norm** | **85.40%** | 84.00% | 82.60% |
| contrast3 | hamming | 73.60% | 72.60% | 71.80% |

### Task: same-game retrieval (k=50; random = 0.08%)

| Encoding | Metric | Mean hit-rate | Lift × random |
|---|---|---|---|
| raw | hamming | 24.48% | **304.8×** |
| contrast3 | hamming | 24.03% | 299.1× |

## Gate decisions

- **Gate A (metric)** — density-normalized Hamming improves phase-ID by ≥10pp over raw Hamming: **+45.00pp**. ✅ **PASS with massive margin.**
- **Gate B (representation)** — 3×3 contrast improves phase-ID by ≥10pp over raw Hamming: **+33.20pp**. ✅ **PASS.**
- **Gate C (structure)** — either variant achieves ≥5% same-game hit rate at k=50: raw Hamming alone hits **24.48%**. ✅ **PASS (300× lift over random).**

All three gates pass. The original `base3_go_probe` RED call stands as the *measured* cycle-1 result, but the *interpretation* is now substantially corrected.

## The corrected interpretation

The probe-1 finding was: "raw trit Hamming phase-ID = 40%, density-only = 98%; Hamming cannot see structure." That was wrong. The correct finding is:

1. **Raw trit Hamming has a density-scaling bias.** Sparse-vs-sparse comparisons produce systematically smaller distances than dense-vs-dense, biasing k-NN neighborhoods toward the sparse class regardless of query density. Fix: divide by sum-of-densities, which is `d_norm(a,b) = H(a,b) · C / (|a|₀ + |b|₀ + ε)`. This is a one-line, trainless, zero-allocation substrate fix.

2. **Raw trit Hamming already sees positional structure.** Same-game retrieval — a density-controlled task — hits 24.48% at k=50 with 300× random lift, on the vanilla representation. The substrate has been able to see structure all along; the probe's phase-ID task just happened to be density-correlated, exposing the scaling bias.

3. **Local-contrast encoding is a real but lesser lever.** Contrast3 improves phase-ID substantially (+33pp) without a metric change, but does NOT improve same-game retrieval. So its value is specifically in de-correlating features from density for density-correlated tasks; it doesn't add structural signal beyond what raw already carries for density-controlled tasks.

4. **Density-normalized Hamming is the biggest bang-per-buck substrate fix we've found.** +45pp on phase-ID from a metric swap, no representation changes, no training, trivially retrofittable into every consumer using Hamming. The next substrate-wide question is whether this same fix improves image pipelines (direct_lsh on MNIST/CIFAR), which may retroactively recover some of the CIFAR representation tax measured in `step_change`.

## Retroactive consequences

### For `base3_go_probe`
The RED call is preserved — those were the numbers on that setup. The *probe recipe* in `base3_benchmarks_synthesize.md` should be updated to include `hamming_norm` as the reference metric for any future probes, since vanilla Hamming carries a known bias.

### For `base3_benchmarks`
The Go-first commitment is **reopened**. With density-normalized Hamming, the substrate demonstrates ~85% phase-ID accuracy using zero-training on raw board state — not at density-only's 98.8%, but within ~14pp of it, using only substrate primitives. That's real substrate-native capability, not a category error.

### For `step_change`
The CIFAR representation tax measured there was at least partly a metric tax. If density-normalized Hamming closes some of the gap on CIFAR too, the "CIFAR is representation-capped" framing needs nuance. The decision to demote image canon to regression-guard still stands (see `project_benchmark_pivot.md`), but this finding is worth re-measuring against.

### For `routed_autodiff`
The expert-collapse finding is unaffected — that was a trainer question, not a metric question. But if `hamming_norm` becomes the substrate's primary distance, future trainer cycles should use it as the routing-score substrate.

## What to ship

1. **`hamming_norm` as a substrate primitive candidate.** Promote to `libm4t` (or `libglyph` if scope is just consumer-level) after one independent measurement on a non-Go dataset. Name: `m4t_hamming_norm` or `glyph_hamming_density_normalized`.
2. **`go_probe.c` keeps its current CLI** (encoding/metric/task toggles). Useful for future Go work.
3. **`substrate_distance_refinement` journal set** (this cycle's five files) is the canonical reference.

## What to NOT ship

- `contrast3` as a separate substrate primitive. It helps phase-ID but not structure-retrieval, so its generality is uncertain. Treat as a task-specific feature encoder for phase-correlated tasks; don't promote.
- A retrofit into direct_lsh yet. Measure first on MNIST/CIFAR to see if `hamming_norm` helps or hurts there before touching production image consumers.

## Next cycles (queued)

1. **`hamming_norm` on images** — measurement cycle, not a decision cycle. Run MNIST/Fashion/CIFAR through direct_lsh with a `--metric hamming_norm` variant. If neutral or positive, retrofit. If negative, document and stop.
2. **`routed_go` trainer** — the Go-first commitment from `base3_benchmarks` is active again. Build a learned-routing Go trainer on top of `hamming_norm` distance. Use the expert-collapse finding from `routed_autodiff` as the design baseline (need learned U, load-balance loss, or class-centroid init).
3. **Retrospective on `step_change`** — short journal entry re-interpreting the CIFAR tax measurement in light of the density bias finding. Not a full cycle, just a correction.

## NORTH_STAR alignment

- `§4 (scaffolding sanction)`: go_probe is explicit probe scaffolding; extensions to it were sanctioned.
- `§12 (no binary float in compute)`: maintained. `hamming_norm` uses integer arithmetic (fixed-point scale by 1024 / (d_a + d_b + 1)). No float.
- `§13 (training in consumer)`: maintained. Zero training. All fixes are substrate primitives.
- **Routing claim check**: the substrate's routing primitives operate over distance scores. If `hamming_norm` becomes the score function, routing-by-learned-gates has a more informative signal. The next cycle's trainer is better-positioned.

## One-line summary

**Raw trit Hamming has a one-line density-scaling bug. Fixing it recovered 45pp of phase-ID accuracy on Go positions, and the fix is trivially retrofittable into every Glyph consumer.**

## Artifacts

- `tools/go_probe.c` — extended with `--encoding/--metric/--task` (5 configs tested).
- `journal/substrate_distance_refinement_{raw,nodes,reflect,synthesize,closeout}.md` — full cycle.
- Memory update (see `project_benchmark_pivot.md`): Go is reopened as a candidate; `hamming_norm` is a substrate-wide fix candidate.

---

## Red-team addendum (same day)

After the above closeout, ran an adversarial pass to find faults. Four concerns surfaced; results below.

### Concern 1 — within-game data leakage (the big one)

**Conjecture:** position-wise 80/20 split puts adjacent-move positions (which differ by ~5 trits) from the same game on both sides of train/test. A query at move 155 has ~31 same-game companions in train; these near-duplicates would be the nearest neighbors by Hamming regardless of metric, inflating phase-ID accuracy.

**Test:** added `--split game` mode that partitions entire games into train vs test (no game-id appears on both sides). Re-ran all 4 phase-ID configs.

**Result — REJECTED:**

| Config | position-split k=50 | game-split k=50 | Δ |
|---|---|---|---|
| raw + hamming | 40.40% | 40.80% | +0.40 |
| raw + hamming_norm | 85.40% | **88.40%** | +3.00 |
| contrast3 + hamming | 73.60% | 73.80% | +0.20 |
| contrast3 + hamming_norm | 74.80% | 79.60% | +4.80 |

Game-split is **marginally stronger** than position-split. Within-game leakage was not inflating results — if anything it may have been slightly depressing them via phase-boundary mixed-votes within same-game neighbors. The substrate's 48pp lift under game-split confirms the metric fix is not an artifact.

### Concern 2 — same-game random baseline was miscalculated

**Bug:** original code printed random-baseline as `k / n_train`, but the correct expected random same-game fraction is `g_q / n_train` (per query, where `g_q` = train companions from query's game). Fixed to compute `mean(g_q) / n_train` over queries.

**Result:** corrected baseline is **0.0592%** (not 0.0803% as originally reported). Measured hit rates were unchanged, so lifts went UP, not down.

| Config | k=50 hit rate | Original lift | Corrected lift |
|---|---|---|---|
| raw + hamming | 24.48% | 304.8× | **413.4×** |
| raw + hamming_norm | 27.32% | — (not run) | **461.4×** |

hamming_norm adds a small but consistent bump on same-game retrieval too (~3pp at k=50, +48× lift).

### Concern 3 — untested combo: hamming_norm + contrast3

**Test:** ran the fourth phase-ID cell that cycle-1 synthesize didn't cover.

**Result:** contrast3 + hamming_norm underperforms raw + hamming_norm on both splits:

| | game-split k=50 |
|---|---|
| raw + hamming_norm | **88.40%** |
| contrast3 + hamming_norm | 79.60% |

contrast3 saturates some of the density information that hamming_norm needs to normalize against. The two fixes are **not additive** — pick one. raw + hamming_norm wins.

### Concern 4 — is hamming_norm secretly a density-difference metric?

**Conjecture:** `d_norm = H · C / (d_a + d_b + 1)` might correlate strongly with `|d_a - d_b|`, in which case the 45pp gain is a density-ordering effect in disguise.

**Test:** compare hamming_norm to pure density-only k-NN (same dataset, same k).

**Result — PARTIALLY CONFIRMED, NOT FATAL:**

| Metric | game-split phase-ID k=50 | game-split phase-ID k=200 |
|---|---|---|
| raw + hamming_norm | 88.40% | 86.40% |
| **density-only** (yardstick, k=200) | — | **98.60%** |

hamming_norm is **not** pure density-difference — if it were, it would match density-only's 98.60%. The 10pp gap shows positional content is present (and adding noise). However, most of the 48pp gain over raw Hamming IS density-recovery; hamming_norm is mostly "density-aware Hamming" rather than a structural leap.

**Honest framing:** hamming_norm unlocks the density signal that raw Hamming was obscuring. It also carries positional content but at a slight cost (noise). For tasks where density alone suffices (phase), density-only beats it. For tasks where density is useless (same-game retrieval with density-controlled queries), hamming_norm adds small positional improvements on top of raw Hamming's already-strong 413× structural lift.

### Red-team verdict

The central claim — **raw Hamming has a density-scaling bias; normalizing by density-sum closes ~48pp on phase-ID** — survives. Nuances that refine the original closeout:

1. The gain is mostly **density-recovery**, not a new structural capability. For density-correlated tasks, density-only is still stronger than hamming_norm.
2. **Raw Hamming already sees structure** on density-controlled retrieval (413× random lift on same-game), which the original closeout noted but undersold. The substrate wasn't structurally blind in cycle 1; it just failed a density-correlated task because the metric had a density bias *in the opposite direction* (sparse-cluster attractor).
3. **Game-wise leakage is not a confound** — results hold (and mildly strengthen) under game-split.
4. **Don't combine contrast3 with hamming_norm.** They're substitutes, not complements. hamming_norm alone is the winner.

### Updated recommendations

- **Ship `hamming_norm` retrofit into direct_lsh** (task #40 unchanged) — but frame it as "density-aware Hamming" not "structural upgrade." Image pipelines already include implicit density decorrelation (per-image contrast normalize, MS4 channels), so the lift on images may be much smaller than on Go. That's still the measurement we need.
- **Reverse-order the next cycles.** Before `routed_go` trainer (task #41), first run the image-pipeline measurement (task #40) to calibrate how much of the density-bias fix is Go-specific. If `hamming_norm` is neutral on images, we're looking at a Go-specific gain and the substrate-wide claim is weaker.
- **For same-game retrieval, raw Hamming is sufficient.** No need to ship hamming_norm for retrieval tasks specifically; the 3pp improvement isn't worth the pipeline change if retrieval is the only goal.
- **Do NOT promote contrast3.** Its phase-ID gain (+33pp) is real but subsumed by hamming_norm (+48pp), and contrast3 hurts when combined with hamming_norm. Treat as an exploratory encoding that didn't earn production status.

### The corrected one-line summary

**Raw trit Hamming has a density-scaling bias that `hamming_norm` corrects; the fix unlocks density-adjacency information, not a new structural axis, and its substrate-wide value is contingent on whether image pipelines already compensate for density elsewhere.**
