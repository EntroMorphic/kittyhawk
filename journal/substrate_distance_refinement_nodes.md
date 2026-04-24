---
date: 2026-04-24
scope: LMM cycle — substrate_distance_refinement
phase: NODES
---

# NODES: substrate_distance_refinement

## Discrete ideas

1. **The Go probe failure is a general-case warning, not a Go-specific bug.** Int8 trit Hamming has a density-scaling pathology that will recur wherever class correlates with sparsity. Fixing it once fixes it everywhere.

2. **Three axes, three experiments max.** Metric × representation × task = 8 configurations, but only 3 are *informative*: raw Hamming (baseline, re-run to confirm), density-normalized Hamming on raw trits, and Hamming on 3×3 contrast trits. Further variants are diagnostic filler unless the first three disagree.

3. **Density-normalized Hamming is a one-line fix.** `d_norm(a,b) = Hamming(a,b) · C / (density(a) + density(b) + ε)`. No training, no new kernels, no representation change. Should be tried first because if it works, we stop there.

4. **3×3 local-contrast trits is the representation-side one-shot.** For each cell c on a 19×19 board, emit `sign(sum_{c' in 3x3(c)} own(c') - opp(c'))`, clamped to {−1, 0, +1}. Output is still 361 trits; content now encodes local balance. Adjacent cells share windows so the feature is spatially smoothed. Runs in O(361·9) per board — cheap.

5. **Same-game retrieval is the density-controlled task.** Query position → rank train positions by distance → compute what fraction of top-k come from the same SGF source file. This measures whether distance captures *positional* similarity, independent of density (because temporally-adjacent positions in a game have similar density). A good baseline: random retrieval gives `k / n_train` fraction of same-game hits. A substrate that sees structure should beat this by a lot.

6. **Density-only as a yardstick, not a competitor.** The 98.28% phase-ID from density is a diagnostic, not a target. We're not trying to beat density-only at density-proxy tasks; we're trying to find what the substrate sees that density doesn't. Same-game retrieval is the right yardstick because density can't solve it — adjacent positions have similar density, so density-only would rank every adjacent-density position as "close" regardless of game origin.

7. **Local contrast ≠ structural features.** 3×3 contrast encodes who-dominates-locally, which is one slice of positional structure. It misses life/death, connectivity, territory. But it's the minimum step beyond raw stone presence, and it's substrate-native (ternary output, fast to compute). If even this fails, more elaborate features almost certainly won't help either — the metric itself is the problem.

8. **Keep the experiment apples-to-apples.** Same 77k positions, same 80/20 split, same brute-force k-NN, only change the distance function and/or the trit encoding. Anything else muddies attribution.

9. **The cycle can be done in one tool update.** Extend `tools/go_probe.c` with: (a) `--metric {hamming, hamming_norm}`, (b) `--encoding {raw, contrast3}`, (c) `--task {phase, same_game}`. Produce a results table across the 2×2×2 grid, report the four informative cells.

10. **Budget check:** phase-ID probe runs in ~0.8s per k at 62k×500 positions. Doubling the configurations to measure is still under a minute total. Budget is trivial — the limiting factor is cycle discipline, not compute.

11. **Interpreting results — four possibilities:**
    - (A) **density-norm Hamming fixes phase-ID** (≥60%): the metric was the whole problem. Representation is fine. Upstream substrate decisions stand.
    - (B) **density-norm fails, contrast3 fixes phase-ID**: the representation was insufficient. Substrate needs a feature-extraction step before Hamming.
    - (C) **neither fixes phase-ID, but one fixes same-game retrieval**: substrate can see positional structure, but density-proxy tasks just aren't the substrate's wheelhouse. Benchmark-class question reopens.
    - (D) **nothing works on either task**: the probe really is RED at the substrate level; next cycle is learned routing, not metric tweaks.

12. **Anti-pattern: don't pile fixes.** If density-norm Hamming works, stop. Don't also add contrast3 "for completeness" and muddle the attribution. Smallest winning change first.

13. **Inspectability check: weighted Hamming / pair-IG is out of scope here.** It requires train-set scanning for weights, which is more than a metric swap. Tempting but expensive; defer until the minimum fix is shown insufficient.

14. **Not included by design:** learned routing, pattern vocabularies, neural embedding. All are real candidates but each is its own cycle. This cycle buys clarity on whether *any* non-trainable fix suffices before we escalate.

15. **Expected result honestly:** density-normalized Hamming should help phase-ID (it's compensating for exactly the pathology we diagnosed) but may not reach the 60% green threshold by itself — because phase is intrinsically multi-way-density-correlated, and removing one density axis doesn't remove all of them. The more interesting measurement is probably same-game retrieval.

16. **Same-game retrieval threshold calibration:** if there are ~1994 source games and k=50 neighbors, a random baseline is 50/62236 ≈ 0.08% same-game hits. A structure-aware distance should give dramatically more — if we see 5-10%+ same-game-in-top-k, that's a >50× improvement over random, a clear substrate signal.

17. **Closing gates:**
    - Gate A (metric): density-normalized Hamming improves phase-ID by at least 10pp over raw Hamming.
    - Gate B (representation): 3×3 contrast improves phase-ID by at least 10pp over raw Hamming.
    - Gate C (structure): either variant achieves ≥5% same-game-in-top-50 (vs ~0.08% random).
    - Passing any two of three: substrate fix is real. Passing only C: substrate sees structure but phase-ID is a bad test. Passing only A: cheap metric fix suffices.

18. **If all three gates pass:** the base3_benchmarks synthesize's Go-first commitment is reopened with caveats (needs the refined distance). If only C passes: benchmark choice still needs revision; at least we know the substrate can see something.

19. **Lineage note:** this cycle is the natural continuation of the `step_change` cycle's MS4 finding. MS4 added multi-scale trits to image signatures and won consistently. Local-contrast for Go is MS4's analogue — adding structure-aware trits before the distance metric runs. The pattern generalizes: raw trit distance is underpowered; a feature-extraction step before quantization is usually necessary.

20. **Scope hygiene:** single-file tool extension, single journal cycle, single results table. If the cycle closes RED, the NEXT cycle is learned routing; if GREEN, the next cycle is the Go trainer proper. Either way the immediate scope is bounded.
