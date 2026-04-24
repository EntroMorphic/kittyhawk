---
date: 2026-04-21
scope: LMM cycle — can a better distance function on direct ternary signatures recover some of the 55pp CIFAR-10 oracle gap that uniform Hamming cannot?
phase: REFLECT
---

# REFLECT

## The core insight

**Hamming distance penalizes the structural zero; SDOT inner-product ignores it. That asymmetry is what makes one base-2-shaped and the other base-3-native.**

Hamming comes from "count disagreements," a base-2 concept where the two states are binary opposites. Ternary Hamming kept the intuition and added a midpoint cost: agreement=0, partial disagreement (0 vs ±1)=1, full opposition (+1 vs −1)=2. This is well-defined but it imports the base-2 framing: *every* trit state difference is a disagreement of some magnitude. The structural zero — "this dim carries no signal for this input" — gets treated as a near-miss, not as "neutral / not applicable."

SDOT inner-product is different. `q[d] × t[d] ∈ {−1, 0, +1}` scores only *signed-agreement*: aligned non-zeros (+1), opposed non-zeros (−1), or one-or-both zero (0). The zero contributes exactly zero to the score — it literally does not participate. This is the NORTH_STAR's geometric-fullness claim: "zero is a structural state on the lattice, as load-bearing as ±1," and load-bearing here means *it tells us the dim is not active for this query/target*, not *there's a cost to pay for the mismatch*.

Under this reading, the right base-3-native distance isn't Hamming at all — it's **negative-SDOT-similarity or a rank-equivalent score**. And the substrate already has SDOT running at 55–60 Gops/s with no current production consumer using it for scoring. The lr_scaffold cycle found CSA as "the shape the substrate has been waiting to use." **This cycle finds SDOT-as-distance — a kernel that's been running at peak throughput with no consumer for the same reason.**

## Why Hamming has been dominant anyway

Because it's the distance function that comes for free from the packed-trit representation. XOR + popcount on 2-bit-per-trit packing, no decoding needed, one NEON instruction path. It's **computationally convenient**, which is a real advantage, but it isn't the same as **semantically correct**.

Pair-IG worked by re-weighting Hamming — essentially, "Hamming is fine, but not all dims should count equally." That's addressing the uniformity axis of Hamming's three structural choices (uniformity, symmetry, independence). It leaves the symmetry (including the cost-1-for-zero-vs-nonzero asymmetry question) and the independence untouched. **The cycle has been treating Hamming's uniformity as the only axis that matters. The zero-handling axis is the one SDOT breaks.**

## Resolved tensions

**T1 (SDOT vs Hamming — is SDOT better?):** the question "is SDOT better" is the wrong question at the cycle-level. The cycle-level question is "which distance's *ranking geometry* aligns with the class manifold in ternary space?" Empirically falsifiable: measure CSA and direct_lsh accuracy under both distances on all three datasets. If SDOT ranks correct-class candidates higher than Hamming, it's a better fit for *this* classification surface. The base-3-native claim earns its keep by winning, or it doesn't.

**T2 (global per-dim weights vs pair-IG):** Family A (global per-dim) can't beat Family pair-IG on expressivity, but that's not why it's interesting. It's interesting because it fits into a much simpler architecture — one weight vector, one multiply-accumulate — and establishes a baseline for "what does per-dim weighting buy at minimum cost." If Family A recovers most of pair-IG's gain, we know the per-class-pair dimension isn't load-bearing. If it doesn't, we know per-pair is doing real work.

**T3 (Family C pattern distance vs substrate-native):** SSTT's pattern scheme isn't itself substrate-native — it's a learned codebook of dense-shape patterns. TBL-based 4-trit block dispatch *is* substrate-native. The right move is NOT to copy SSTT but to test whether TBL pattern dispatch — reading 4 trits at a time as a single entity and scoring its *kind* rather than its *content* — carries more class signal than per-trit independence. That's the same *shape* as pattern distance but the primitive is native, not imported.

**T4 (aggregator choice in Family B):** the base-3-native aggregator is the one the substrate can do in one kernel. Masked-VCNT `m4t_trit_sparsity` gives a block-level count of non-zero trits. `m4t_trit_eq` followed by `m4t_trit_signed_sum` gives block-level agreement counts. Sum-of-blocks reduces to full Hamming; max-of-blocks requires extra scalar logic; threshold-count is natural because VCNT already does the counting. **Threshold-count is the substrate's default aggregator.** Set a per-block threshold T; score = count(blocks where cost ≤ T). Implementable in one TBL + one VCNT pass.

**T5 (ceiling assumption — is the signal there?):** the cycle should spend its first 20 minutes on cheap sanity checks before investing in any Family. Specifically: (a) CSA/CSA-k with SDOT scoring on MNIST; (b) pair-IG scoring with uniform weights (sanity check that pair-IG's structure, not just its weights, matters); (c) brute-force Hamming k-NN on CIFAR-10 with no bucket filter (rules out filter-stage loss). If all three land within 1pp of their current numbers, the scoring-layer has meaningful headroom and Family A/B experiments are worth running.

**T6 (SDOT unused because of architectural path-dependence):** confirmed. `direct_lsh` was built around `glyph_sig_quantize` → packed-trit sigs → `popcount_dist`. That architecture makes Hamming the default. Switching to SDOT needs either: unpacking sigs at query time, or maintaining a parallel int8-ternary representation. Both are cheap; neither was built because nobody questioned the kernel choice.

## Hidden assumptions I was making

- **"More expressive distance = better accuracy."** Not obviously true. If the signature itself doesn't cleanly separate classes, no distance on it will. A better distance on a bad signature is still bounded.

- **"Pattern distance is SSTT-shaped and therefore the natural pattern-level option."** The substrate already has TBL-based 4-trit pattern dispatch as a first-class op. SSTT's implementation might look similar but the underlying primitive on M-series silicon is different. Naming the primitive correctly matters.

- **"Hamming symmetry is structurally neutral."** Might be wrong. If the query's structural zeros cluster in different positions than the target's — e.g., query is a specific MNIST "5" with zeros at its top-right curve, target is a generic "5" with zeros at its bottom-left — Hamming treats both structural-zero positions as equally mismatched. An asymmetric distance could prefer targets whose zero positions subset the query's zero positions. I don't yet know whether this matters for image classification.

- **"The 55pp oracle gap is all scorer headroom."** Re-examining after the lr_scaffold close-out: the 55pp is the *theoretical* gap, but how much of it is reachable by any local distance (per-dim, per-block, per-pair)? The oracle counts "correct class somewhere in the top-1600 of 10,000 candidates" — that's a very weak condition. Realistic reachable gap with a better distance may be 2–10pp, not 55. **Cycle success needs to be re-framed around realistic recovery, not oracle ceiling.**

## What I now understand

1. **The primary experiment is an SDOT-vs-Hamming A/B.** One kernel swap; two distance semantics; three datasets. Resolves T1 and T6 with one pass. Cost is a few hours of implementation + the existing measurement harness. This is E1.

2. **The second experiment is the simplest plausible Family A.** Global per-dim weights derived from per-dim entropy (1 vector, no pair matrix). Swap into `direct_lsh` resolver as a side-by-side scorer. This resolves T2 and calibrates how much per-dim weighting alone buys. This is E2.

3. **The third experiment — if one of the first two shows signal — is substrate-native block distance.** 4-trit blocks via TBL dispatch, threshold-count aggregator via VCNT. One new kernel call (or composition: `trit_eq` then `trit_sparsity` over block-spans). This is E3, conditional on E1 or E2 showing headroom.

4. **The thesis-relevant claim is SDOT-as-distance.** The other experiments are calibrators. If E1 wins, the base-3-native distance story for the substrate is validated with a measured number and no invention. If E1 doesn't win, we've learned that Hamming's symmetry is load-bearing on these signatures and the thesis reframe was wrong. Either way, clean signal.

5. **Pattern distance (Family C) is out of scope for this cycle.** Either TBL block distance (E3) subsumes it structurally, or a future cycle names "prototype pattern codebooks" as a separate question.

6. **The CIFAR-10 oracle gap reframe.** 55pp was headline-misleading. The realistic reachable gap via distance-function alone is probably 2–10pp based on pair-IG's +1.95pp and the Hamming/SDOT structural-zero question. That's still meaningful — beating pair-IG by 2–5pp is a material production improvement. Frame success accordingly.

## Open residuals

- **R1: naming.** "SDOT distance" is accurate but awkward. "Inner-product similarity" is generic. "Ternary alignment score" is descriptive. Use SDOT-distance in the synthesis since it anchors to the specific kernel; revisit when the primitive proves useful.

- **R2: dataset scope.** E1 and E2 should run on all three datasets — cheap, and MNIST may reveal different behavior than CIFAR-10. Fashion-MNIST is the interesting middle case (some pair-IG benefit; gradients are active).

- **R3: what happens if SDOT ties Hamming on all three datasets?** That's not a loss — it means the two distances are roughly equivalent on these signatures, which is still thesis-relevant data. The substrate would then have two equally-valid primitives; path-dependence on Hamming is no longer a limitation.
