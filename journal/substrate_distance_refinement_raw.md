---
date: 2026-04-24
scope: LMM cycle — substrate_distance_refinement. Triggered by base3_go_probe RED: raw ternary Hamming is density-dominated and cannot see structural signal on 19×19 board states.
phase: RAW
---

# RAW: substrate_distance_refinement

The Go probe's diagnostic was sharp: Hamming k-NN at 40.40% on phase-ID, density-only baseline at 98.28%. The input representation *contains* the signal (density-only sees it perfectly), but the substrate's metric doesn't. This cycle is about closing that gap at the substrate level — not by changing benchmarks again, but by upgrading the distance machinery so it can see structure.

## What the probe actually measured

Raw trit Hamming over 361-dim vectors has a degenerate property: sparse vectors cluster tightly (few nonzeros, little room for disagreement) and dense vectors scatter (many nonzeros, lots of room for disagreement). Phase in Go correlates monotonically with density. Therefore Hamming-between-sparse-and-sparse is *systematically smaller* than Hamming-between-dense-and-dense, even when the dense-dense pair is phase-matched and positionally similar. Opening positions form an attractor; middle/end positions scatter into the opening attractor's k-nearest neighborhood.

The pattern isn't Go-specific. It's a property of int8-trit Hamming over vectors with variable density. Any dataset with class-correlated density — MNIST (digit stroke density), document frequency histograms, genomic sparsity — will exhibit the same failure.

## What's on the table — candidate fixes

Three orthogonal axes, each independently composable:

### Axis 1: Distance metric
- **Density-normalized Hamming**: `d(a,b) = Hamming(a,b) / (density(a) + density(b) + ε)`. Cheapest fix. Compensates for the sparse-vector attractor by dividing out the baseline. No representation change, no training.
- **Weighted Hamming**: per-position weight from information gain or pair-IG on train set, applied per-bit at distance time. Already used in direct_lsh's `--pair_ig` rerank. Costs weight computation up front.
- **SDOT-like inner-product distance**: `d = -sum(a[i]*b[i])`. Scored rather than counted. Own-own pair contributes +1 (closeness), own-opp pair contributes -1 (distance), own-empty pair contributes 0. Fundamentally different from Hamming on the structural-zero question; direct_lsh's E1 experiment measured it against Hamming on images (lost).
- **Local-window overlap**: don't compare position-by-position, compare 3×3-window-histograms.

### Axis 2: Input representation enrichment
- **3×3 local contrast**: each cell emits sign(own_count - opp_count) over its 3×3 neighborhood. Still 361 trits, still ternary, but now each trit encodes local balance rather than raw stone presence.
- **Per-cell fanout**: each cell emits K trits indicating structural features — center-stone, liberty count bucketed, group-size bucketed, eye-shape indicator. Balloons the signature from 361 → 361·K.
- **Bag-of-patterns**: enumerate K common 3×3 patterns (by frequency in training), encode each position as a K-trit presence vector. Loses positional information, gains compositional structure.
- **Pattern-position vocabulary**: each cell × each pattern-class → one trit. Preserves locality. Very large signature.

### Axis 3: Task reformulation
- **Same-game retrieval**: given a query position, measure whether the top-k nearest are drawn from the same source game (temporally nearby positions). Density-controlled because adjacent positions in a game have similar density. A clean test of whether structure-similarity is captured.
- **Pattern-localization**: given a board, predict where the action is (which quadrant has the most recent activity). Requires positional specificity.
- **Same-player-to-move vs different**: pure symmetry task, density-independent.

## What the substrate already knows how to do

- NEON Hamming int8 — done.
- NEON popcount on packed-trit blocks — done (libm4t).
- Weighted Hamming via pair-IG — done (direct_lsh tools).
- Density computation — trivial.

So the *cheap* path is: density-normalized Hamming on existing representation. The *richer* path is: local-pattern representation + Hamming on that.

## What's conspicuously missing from our toolkit

- A structure-aware ternary distance that doesn't reduce to counting or summing.
- Any per-cell feature beyond the raw value.
- A task in our measurement vocabulary that's density-controlled by construction (all current tasks — digit class, CIFAR class, Go phase — have some degree of density correlation).

The missing piece isn't exotic: it's just the step from "raw trit distance" to "distance after trit-level feature extraction." Images do this via gradient+multi-scale (direct_lsh's MS4). Go needs the equivalent.

## What the cycle must answer

1. **Does density-normalized Hamming close the gap** on the existing phase-ID probe? (Cheapest candidate — checks whether the diagnosis is complete.)
2. **Does 3×3 local contrast encoding close the gap** on the same probe? (Representation-side candidate — most substrate-native.)
3. **Does either metric support same-game retrieval** — a density-controlled positional-similarity task that actually tests substrate structure?

If (1) alone fixes phase-ID: the issue was just the metric. Substrate representation is fine.
If (1) fails but (2) succeeds: the issue was representation. Substrate needs enrichment.
If neither fixes phase-ID: structural failure. Substrate needs learned routing OR a different task domain.
If any fix *also* succeeds on same-game retrieval: evidence that substrate can see positional structure with the right primitive.

## Scope guardrails

- **Stay in C.** No Python anywhere.
- **Stay ternary.** Output of any enrichment must be int8 trits.
- **Stay in the go_probe tool for this cycle.** Don't promote anything to libm4t or libglyph until the distance refinement proves itself on at least two tasks.
- **Re-use the 77k-position corpus from the Go probe.** Same data, new distance — head-to-head is the point.
- **Single-file cycle, minimum viable extension.** Don't build a second benchmark; use the one that already exposed the problem.

## Residue for NODES

- The axes are independent — metric × representation × task. Budget lets us explore 2 × 2 × 2 = 8 configurations but we only need 3-4 informative ones.
- Distance-metric fix is the first experiment *because it tells us whether the representation is sufficient*. If density-normalized Hamming fixes things, no representation work is needed yet.
- Task reformulation (same-game retrieval) is the second critical experiment *because it decouples density from position*. A good result there is a substrate-level win even if phase-ID stays hard.
- All three axes together would be the maximal version; NODES picks what's in scope.
