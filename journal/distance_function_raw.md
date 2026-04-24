---
date: 2026-04-21
scope: LMM cycle — can a better distance function on direct ternary signatures recover some of the 55pp CIFAR-10 oracle gap that uniform Hamming cannot?
phase: RAW
---

# RAW: distance-function design for direct ternary signatures

Seed from `lr_scaffold_closeout.md`: uniform Hamming on direct-quantized signatures is the ceiling across every classifier architecture we've tested. Pair-IG moves it +1.95pp on CIFAR-10 with per-class-pair per-dim weights. The oracle-over-union is 99.99%, meaning correct-class samples exist in the union, but Hamming distance can't rank them near the top. The gap is distance-geometric, not classifier-architectural.

So the question is: what distance function on these signatures would rank the correct-class candidates higher, and is any substrate-native primitive already aligned with it?

## What I think I know

Ternary Hamming, as currently implemented, assigns cost {0, 1, 2} per dim based on trit pair state. It's uniform over dims (every dim has equal weight), symmetric in direction (q=+1/t=0 same as q=0/t=+1), and independent across dims (no cross-dim correlation). Pair-IG breaks uniformity (per-dim per-class-pair weights). It leaves symmetry and independence alone.

SSTT — the published ternary-transformer baseline — reaches ~53% on CIFAR-10. The close-out notes it uses "pattern-level block scoring with correlation-aware scoring." I haven't read the paper in detail, but the phrasing suggests it breaks independence: scores blocks of trits as patterns rather than counting per-trit mismatches. That's the 7pp gap we'd like to close.

## Four candidate distance families

**A. Global per-dim weights (cheap).** One weight vector `w ∈ ℝ^D` (or `ℤ^D`), applied as `Σ_d w[d] × cost(q[d], t[d])`. Simpler than pair-IG (no per-class-pair matrix). Weight derivation: variance of the dim across training set (high-variance dims matter more), or entropy, or inverse-frequency. Integer-quantized weights are natural.

**B. Block-level Hamming.** Split D dims into blocks of B trits each. Per-block: compute Hamming between query and target blocks. Then combine per-block results. Options for combination: sum (degenerate — equals full Hamming), max (single worst block drives cost), threshold-count (block "matches" if cost ≤ threshold; count matches). This introduces locality — nearby trits interact. TBL dispatches 4 trits at a time; block size 4 maps cleanly.

**C. Pattern distance.** For each block of B trits, learn a small set of prototype patterns from training data. Score block = distance to nearest prototype pattern. Sum over blocks. This is a hash-of-hashes structure — two-layer signature. Substrate-native via trit_mul + popcount sum.

**D. Correlation distance.** Score correlations between pairs (or triples) of trits. E.g., for each dim pair (i, j), a co-occurrence weight that fires when both query trits match the target in a specific way. Breaks independence fundamentally. Probably expensive but very expressive.

## First instincts

- **A is the obvious first try.** Cheap, measurable, substrate-native (one extra multiply per popcount term, or a weighted sum after popcount). Low risk. If the 55pp oracle gap has any low-hanging fruit, a global per-dim weight vector from an information-theoretic derivation should catch some.
- **B (block Hamming with max or threshold-count) is the substrate-native move.** Block size 4 matches the 2-bit-per-trit packing (one byte = 4 trits = one block). TBL dispatches over 4-trit pairs directly. Masked-VCNT counts block agreements. This is the most base-3-aligned option.
- **D is too ambitious for a first cycle.** Pairwise correlation over D=9024 dims on CIFAR-10 is O(D²) = 81M features. Feasible but heavy. Later.
- **C (patterns) is what SSTT does and gets to 53%.** Copying SSTT's shape is scaffolding per NORTH_STAR §4 — valid but not where the substrate alignment lives.

## What scares me

- **A may not help much.** If the per-dim discriminative signal is already close to pair-IG's +2pp (which itself is per-dim-per-pair), a simpler global per-dim weight can't exceed it meaningfully. Risk: measure a +0.5pp improvement and be forced to conclude "per-dim alone is not enough."

- **B's max/threshold aggregators might be too aggressive or too blunt.** Max over blocks makes a single bad block dominate. Threshold-count over blocks discards distance magnitude. Sum recovers Hamming. The right aggregator is not obvious; I don't have a clean prior.

- **The oracle gap may be intrinsically out of reach for any local distance.** If correct-class samples are truly in the top-1000-ish of 1600 candidates under Hamming, the correct-class geometry might require global features (image-level structure) that no per-dim or per-block distance can recover. If that's true, **no amount of distance redesign moves us past ~50% on CIFAR-10**; we'd need new input features, not new distance.

- **Reinventing SSTT under a Glyph label.** If B or C ends up being "what SSTT already does", then the cycle is scaffolding (per §4 sanctioned), but the thesis-relevant delta is just "Glyph's kernel runs it faster than SSTT's implementation." Useful, but not the base-3-native discovery the previous cycle pattern suggests is possible.

## What would actually be substrate-native

TBL is first-class dispatch over 4-trit patterns. `m4t_trit_mul` is a TBL-backed LUT op that multiplies pairs of trits and produces a trit. The three-state multiplication table is:

|   | −1 | 0 | +1 |
|---|---|---|---|
| **−1** | +1 | 0 | −1 |
| **0** | 0 | 0 | 0 |
| **+1** | −1 | 0 | +1 |

"Agreement" in the product table is |trit_mul(q, t)| — the product is +1 when q and t are the same non-zero state, −1 when opposite, 0 when either is zero. Sum of `trit_mul(q, t)` over all D dims gives a signed score: positive when q and t align, negative when they disagree, zero when one is a structural zero. This is fundamentally different from Hamming, which conflates alignment and orthogonality into a "cost" integer.

**SDOT computes exactly this sum, using int8 ternary values and 16× parallel dot-accumulate.** `m4t_mtfp4_sdot_matmul_bt` is the kernel. score(q, t) = sdot(q_int8, t_int8) = Σ q[d] × t[d], with q, t ∈ {-1, 0, +1}. Argmax score over candidates → best match. This is already what CSA does for 10 class prototypes. Extending to per-candidate scoring gives a signed inner-product distance (or more precisely: argmax inner-product similarity).

Is that better than Hamming? Inner-product similarity ignores mismatches-at-zero: q=+1, t=0 contributes 0 to the score, same as q=0, t=0. Hamming penalizes the first but not the second. The two distances rank candidates differently — and on average, SDOT similarity is LESS sensitive to the "structural zero" state than Hamming.

That might be the move: **stop penalizing zero-trit structural disagreements as if they were meaningful.** A query trit being zero (noise floor) and a target trit being ±1 probably shouldn't count as a full mismatch — Hamming gives it cost 1, SDOT gives it cost 0 (because q × t = 0). If the density-calibrated quantization puts zeros in genuinely uninformative dims, Hamming is over-penalizing them.

## Open questions

1. Has anyone tested SDOT similarity (inner product, argmax) against the full training set as a baseline, compared to popcount_dist k-NN? This would be a direct A/B on direct_lsh's architecture with a substrate-native distance swap.
2. What does CSA-k look like with SDOT-based scoring instead of popcount_dist? I assumed Hamming was the default kernel; inner-product might rank differently.
3. Is the 55pp CIFAR-10 gap reachable with any *local* distance, or does it require features Hamming/SDOT can't see?
4. If block-distance is the route, what's the right aggregator: sum, max, weighted-sum, threshold-count? The choice may be more important than the block size.

## First instincts for the cycle

- Start with the cheapest measurable experiment: inner-product distance via SDOT in the direct_lsh resolver. If SDOT distance beats Hamming meaningfully, we have a new primitive that's substrate-native with zero invention.
- Then add global per-dim weights (A). Cheapest non-uniform distance; if it helps, great, if not, we've ruled out one simple hypothesis.
- If neither moves the number, pivot to block distance (B) with TBL dispatch.
- Resist the urge to copy SSTT's pattern scheme until A/B/SDOT are exhausted. The thesis is that base-3-native primitives exist; forcing ourselves through them first is the discipline.
