---
date: 2026-04-24
scope: LMM cycle — substrate_distance_refinement
phase: SYNTHESIZE
---

# SYNTHESIZE: substrate_distance_refinement

## Commitment

Extend `tools/go_probe.c` with two orthogonal fixes (one metric, one representation) and one density-controlled task (same-game retrieval). Re-run on the same 77k-position Go corpus from `base3_go_probe`. Produce a results table; pass/fail each cell against three gates.

**No other work in this cycle.** No libm4t changes, no direct_lsh changes, no new benchmark. Close with a written go/no-go on whether the substrate's distance machinery can see structural signal without training.

## Exact additions to `go_probe.c`

### CLI
```
go_probe <sgf_dir> [--max_games N] [--sample_every K]
                    [--encoding {raw, contrast3}]
                    [--metric   {hamming, hamming_norm}]
                    [--task     {phase, same_game}]
```

Defaults reproduce the original probe: `--encoding raw --metric hamming --task phase`.

### Encoding: `contrast3`

For each cell (r, c), emit:
```
s = Σ_{(r', c') ∈ 3×3 neighborhood, clamped to board} board[r',c']
     (own = +1, empty = 0, opp = −1)
trit[r*19 + c] = +1 if s > 0
                 −1 if s < 0
                  0 if s == 0
```

Boundary cells take whatever's inside the board; no padding. Output is 361 trits, same shape as raw. Computation: O(361 · 9) per board, < 10 µs per position.

### Metric: `hamming_norm`

```
d_norm(a, b) = hamming(a, b) · C / (density(a) + density(b) + ε)
```

With C = 100 (scale so d_norm values sit in a comfortable int range) and ε = 1 (avoid div-by-zero on empty boards). Density is count of nonzero trits. Return a fixed-point int so ordering is stable.

Important: normalization is computed per-pair, so the metric is **not** a vector norm — it's a pairwise dissimilarity. That's fine; k-NN only needs ordering of distances from query to each train point.

### Task: `same_game`

Track `source_game_id` per position. For each test position q:
1. Rank all train positions by distance.
2. Take top k.
3. Compute fraction that come from the same source_game_id as q.

Report:
- mean fraction across test positions.
- compare to random baseline = k / n_train.
- "same-game hit rate lift" = mean_fraction / random_baseline.

For k=50, random baseline ≈ 50 / 62236 ≈ 0.08%. Gate C passes if lift ≥ 60× (i.e., mean_fraction ≥ ~5%).

## Gates (from NODES §17)

- **Gate A (metric)**: `hamming_norm` improves phase-ID accuracy by ≥ 10pp over `hamming` on raw encoding.
- **Gate B (representation)**: `contrast3` improves phase-ID by ≥ 10pp over raw encoding at `hamming`.
- **Gate C (structure)**: either variant achieves ≥ 5% same-game hit rate at k=50 (vs 0.08% random = 60× lift).

Outcomes:
- **Pass A only**: cheap metric fix suffices; retrofit into direct_lsh etc. Next cycle: Go trainer on top of normalized Hamming.
- **Pass B only**: representation enrichment is the key; local-feature extraction becomes substrate-wide standard. Next cycle: richer per-cell features.
- **Pass A and B**: both real; prefer combined. Next cycle: Go trainer with contrast3 + normalization.
- **Pass C only**: substrate sees structure but phase-ID is a bad test. Base3_benchmarks synthesis reopens with a different task target.
- **Fail all**: substrate structural failure. Next cycle forced to learned routing or different representation entirely.

## Implementation plan

1. Add `--encoding`, `--metric`, `--task` CLI parsing.
2. Add `contrast3` encoder: `encode_contrast3(const board_t* b, int mover, int8_t* out)`.
3. Refactor emit_position to take an encoding function pointer, populate per-config.
4. Add `hamming_norm` path; store densities alongside trits for efficient pair-distance.
5. Add `same_game` task: store `game_id` per position; k-NN reports per-query top-k source-match count.
6. Rebuild, run 4 informative configurations head-to-head:
    - (raw, hamming, phase) — reproduce original probe.
    - (raw, hamming_norm, phase) — Gate A.
    - (contrast3, hamming, phase) — Gate B.
    - (raw, hamming, same_game) — establish structure-baseline.
    - (contrast3, hamming, same_game) — if Gate B passes, test if it also helps structure.
7. Document in `substrate_distance_refinement_closeout.md`.

## Out of scope

- Weighted Hamming with pair-IG (more than metric swap).
- Learned routing (its own cycle).
- Richer per-cell features beyond 3×3 contrast.
- Direct_lsh retrofit (post-cycle decision).
- Image-pipeline regression on these fixes (separate measurement).

## One-line summary

**Measure whether density-normalization and/or 3×3 local-contrast encoding moves the Go probe's 40.40% phase-ID and 0.08% random-retrieval baseline into substrate-claim territory, and call the result honestly.**

## Deliverable

`journal/substrate_distance_refinement_closeout.md` with a single results table, pass/fail on each of three gates, and a one-paragraph call on what the next cycle is.
