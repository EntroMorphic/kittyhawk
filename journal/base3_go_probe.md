---
date: 2026-04-24
scope: base3_benchmarks cycle — Go probe outcome (phase classification via raw Hamming on ternary 19×19 positions)
phase: PROBE
---

# Probe: ternary Go phase classification via raw Hamming

## Outcome — PROBE RED

**Phase-ID k-NN accuracy (Hamming on 361-trit 19×19 boards):**

| k | Accuracy | Confusion (truth × pred) |
|---|---|---|
| 50 | **40.40%** | open 122/122, mid 15/207, end 65/171 |
| 100 | 39.80% | open 122/122, mid 12/207, end 65/171 |
| 200 | 38.60% | open 122/122, mid 8/207, end 63/171 |

**Baselines for comparison:**

| Baseline | Accuracy |
|---|---|
| Random (uniform 3-class) | 33.33% |
| Majority-class (always "mid") | 38.41% |
| **Density-only k-NN (count stones, k=200)** | **98.28%** |

**Decision per `base3_benchmarks_synthesize.md` §Decision rule:** probe result is at the border of RED (40%) but the density-only baseline makes the finding unambiguous — the signal is *present in the representation but not visible to Hamming*.

## Setup

- Corpus: 2000 amateur games from featurecat/go-dataset 10k archive (real pro-free-play games, each ~150–300 moves).
- Parser: minimal main-line SGF parser in C (120 LOC), extracting SZ, HA, and move sequence for B/W nodes.
- Replayer: standard Go rules engine (captures via group flood-fill, suicide rejection). `games ok=1994 / illegal_move=0` → parser and rules are correct.
- Sampling: one position every 5 moves → 77,795 positions across 1,994 games.
- Encoding: 361 int8 trits per position, from current-mover's perspective (own=+1, empty=0, opponent=−1).
- Labels: move number → phase bin (0–59 opening / 60–149 middle / 150+ endgame). Distribution across dataset: open=21449, mid=29397, end=26949.
- Split: 80/20 train/test, test capped at 500 to keep brute-force k-NN tractable.
- Distance: Hamming (count of trit mismatches), NEON-accelerated.

## Interpretation

The phase-ID task is **nearly trivial in the input representation** (density-only gets 98%). Yet raw Hamming *on the same representation* delivers 40% — worse than always-predict-mid.

Why: Hamming on ternary vectors counts position-mismatches. For two Go boards:
- Nearly-empty vs nearly-empty → few nonzero positions on either side → tiny Hamming, regardless of where the stones actually are. **Opening positions form a dense, self-attracting cluster.**
- Nearly-empty vs dense → Hamming dominated by the dense board's nonzero positions.
- Dense vs dense (same phase, different games) → Hamming dominated by positional disagreement — two endgame boards typically have stones in *different* places.

Result: nearly-empty-vs-nearly-empty distances are the *smallest* distances in the dataset, so the k-NN neighborhood of any queried board (regardless of its true phase) is dominated by opening positions. Opening-recall goes to 100%, middle/end recall collapses.

The confusion matrix is textbook: **opening is detected perfectly; middle is almost entirely misclassified as opening; endgame mostly goes to middle or opening.**

## Why the finding matters

This is a substrate-level diagnostic, not a benchmark failure. The probe does exactly what a probe should do: **it rules out a configuration we were about to commit to.**

1. **Ternary-native input is necessary but not sufficient.** Go board state is the purest ternary input we've ever fed the substrate — zero quantization loss, zero representation tax. Yet the substrate's default distance metric (Hamming) doesn't see the signal that distinguishes positions.

2. **The "base-3 native" criterion in `base3_benchmarks_synthesize.md` needs a fourth element.** Current three: (a) ternary-representable input, (b) routing-load-bearing task, (c) inspectability-credited evaluation. Missing: **(d) discriminative signal must be visible at the bit level the substrate's distance metrics operate on.** If aggregate statistics (density, mean, any summary) trivially beat bit-distance, the substrate is carrying less information than a 1-bit summary.

3. **This is the same finding as the image-classification story, dressed differently.** Raw pixel Hamming on MNIST beats chance but loses to gradient+multi-scale enrichment. Raw Go position Hamming loses to stone-count, because board density is a 1-number summary that dominates positional structure. The pattern: **raw trit distance sees low-order statistics, not structure.** The substrate either needs (i) richer per-position features (local patterns, group shapes, liberty maps) to make structure visible in trit space, OR (ii) a distance metric that isn't purely positional (weighted Hamming with learned weights, or something more elaborate).

4. **Density-only 98% on phase is not a Glyph claim.** It's a trivial classifier. The probe's green rule required Hamming-on-trits to succeed, and it didn't. A 98% density-based classifier is not a substrate-native claim — any system that counts stones can do it.

## What this probe does *not* say

- Go is not a bad substrate target in principle. Rich local-pattern encodings (3×3 corner patterns, eye shapes, group liberties) might make positions Hamming-discriminable.
- Learned routing might change the picture. The `routed_autodiff` cycle's expert-collapse finding is relevant: with a learned gate, the routing layer could partition positions by structural features rather than by density.
- Next-move-wins (the second arm of the synthesize decision rule) wasn't tested here. Amateur 10k-level data is too noisy for that label to be meaningful, and the phase finding alone is decisive enough that testing the second arm adds little.

## What this probe *does* say

- **`base3_benchmarks_synthesize.md`'s Go-first commitment is premature.** The synthesize committed to Go on the strength of its substrate-property alignment on paper. The probe falsifies that alignment for the current substrate configuration: ternary input + Hamming distance fails the minimum-viable substrate-task-fit test.
- **Tabular is not the right fallback either**, for a similar reason. Tabular inputs are mixed discrete/continuous, which means more quantization work upstream, and most tabular classifiers already operate on derived features, not raw columns. Handing Hamming raw columns would reproduce the same failure mode.
- **The real gap is substrate-level**: the current distance metric is insufficient for any input representation where discriminative structure isn't position-by-position.

## Recommended next step

**Do not** jump to the tabular probe. Do **not** start a Go-specific trainer. Instead, refine the benchmark criterion:

> A base-3-native benchmark must satisfy (d): **the discriminative signal between classes must be carried by position-by-position trit differences, not by aggregate statistics.**

Candidate benchmarks passing (a)+(b)+(c)+(d):
- Pattern-rich tasks where class identity comes from *which positions* are set, not *how many* (e.g., character recognition with meaningful stroke positions — MNIST-style, but that's back to images).
- Binary/ternary sequence motif tasks (DNA motif detection, pattern-completion games, cryptogram classification).
- Go with **pre-computed local-pattern trits** rather than raw board state — but that's a substrate-extension cycle.

The honest next step is another LMM cycle on the **distance-metric question**: given that raw Hamming misses structural signal, what substrate-native metric or enrichment captures it? Options:
- Local-pattern features (3×3 or 5×5 windows → trit-pattern vocabulary).
- Learned routing (puts the distance question in learnable space).
- Weighted Hamming with per-position weights from IG or pair-IG (we did this for images — did it ever *close* the gap, or just narrow it?).

## Artifacts

- `tools/go_probe.c` — SGF parser + rules engine + ternary encoder + brute-force k-NN classifier. 400 LOC. Standalone (no libglyph dependency). Builds via `cmake --build ... --target go_probe`.
- `data/go/10k.7z` — 42MB featurecat 10k corpus. Kept locally; not committed.
- `data/go/sgf_sample/10k/` — 2000 extracted SGFs used for this probe.

## Status

- Task #38 resolved: probe RED.
- Task #37 (base3_benchmarks LMM cycle) — reopen via a **follow-up cycle** on distance-metric refinement, provisional slug `substrate_distance_refinement`.
- Image canon stays in regression-guard role; no reversion.
