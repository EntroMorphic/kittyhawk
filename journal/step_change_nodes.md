---
date: 2026-04-22
scope: LMM cycle — given the lr_scaffold + distance_function cycle results, what's the next step-change worth chasing?
phase: NODES
---

# NODES

## Discrete ideas

1. **Two cycles have cleared the classifier and scorer layers.** Everything tested at those layers lands within ±1.5pp of Hamming k-NN. The remaining axis is *upstream* — the signature itself.

2. **Pair-IG remains the only positive scorer signal.** +1.95pp on CIFAR. The per-class-pair structure (not per-dim weighting in general) is what works. Future scoring-layer work should start from that insight, not re-test it.

3. **Oracle-over-union was misleading; realistic CIFAR-10 headroom is smaller than it implied.** Correct target: +3–7pp over current 46.63%, not the 53pp the oracle suggested. Frame success accordingly.

4. **S1: Multi-scale signature pyramid.** Add intensity + gradients at 1× and 2× downsampled resolutions to the existing channels. Quadruples total_dim. Tests whether the CIFAR ceiling is still climbing with more bits or is at an asymptote.

5. **S2: Pattern codebook signature.** Per-block (4×4 or 8×8) pattern dispatch — emit a multi-trit code naming which pattern the block matches. TBL-native. Closest in shape to SSTT's pattern-level scoring, but constructed via direct dispatch rather than learned codebooks.

6. **S3: Multi-distance ensemble (cheapest test of E3 complementarity).** Run the same filter stage, score candidates with Hamming AND block_threshold, combine by rank fusion or normalized sum. Directly verifies whether the CIFAR E3 +0.56pp is a real compounding axis or a noise blip.

7. **S4: Per-region τ calibration.** Currently global τ per channel type. Spatial quadrant or finer grid could capture that some regions carry more class discrimination than others. Single-file change in direct_lsh's calibration code.

8. **S5: Block-level first-class signatures.** Restructure the signature as (count(+1), count(0), count(-1)) per block or similar summary statistic — smaller sig_bytes, local structure preserved. Halves memory and changes what the distance function operates on.

9. **S6: Signed-value retention (MTFP × trit matmul).** Skip the direct quantization for scoring purposes; use m4t_mtfp_ternary_matmul_bt with MTFP-scale query values against ternary class prototypes. Unpacks vs the current "quantize everything to trit" choice.

10. **S7: Learned signature via gradient-trained ternary encoder (scaffolding).** Train a small quantized network that emits per-image ternary code. Base-2 training path; base-3 inference. Explicitly scaffolding per NORTH_STAR §4.

11. **The bit-budget Pareto frontier is unmapped.** direct_lsh at various total_dim sizes on CIFAR-10 would tell us whether the current ceiling is at an asymptote or still climbing. **Cheapest informative measurement.** Would directly inform whether S1 (more bits) should even work.

12. **The complementarity hint is unverified.** CIFAR E3 Selective +0.56pp is single-seed. Multi-seed run is ~2-3 hours of compute and produces either confirmation or falsification of S3's premise.

13. **Substrate-native vs scaffolding tension.** NORTH_STAR says prefer substrate-native primitives; §4 sanctions scaffolding explicitly as a temporary calibrator. S1, S2, S3, S4, S5 are substrate-native. S7 is scaffolding. S6 is borderline (retains MTFP).

14. **Cost asymmetry between candidates.** S3 (ensemble) and S11 (bit-budget Pareto) are hours of work. S1 (multi-scale) is ~1 day. S2, S5, S6 are ~2-3 days each. S7 (scaffold trainer) is ~1 week minimum. Priority should account for this.

15. **Expected-value per cost.** S3 is cheap and has already-positive preliminary signal (+0.56pp). S11 (Pareto) is cheap and reframes the entire question. S1 is cheap and tests "more bits help" hypothesis. S2/S5 are more costly but could reveal structural wins. S7 is expensive and only informative if all substrate-native options fail first.

16. **The thesis hasn't failed but hasn't decisively won.** MNIST: tied with SSTT. Fashion-MNIST: +1.41pp over SSTT. CIFAR-10: −6.4pp below SSTT. The thesis-relevant task on CIFAR-10 is either to close that gap via substrate-native work, or to discover that CIFAR-10 is a domain where the substrate's primitives are structurally bounded and the thesis only applies to lower-complexity visual tasks.

17. **An orthogonal axis not in my S1-S7 list: training-set leverage.** All cycles so far have used the full training set. What if direct_lsh's filter+score pipeline, given ONLY a 5000-sample-per-class curated subset (class-balanced, instance-balanced across difficulty levels), performed better than full-training? This is a data-selection question, not a signature or scoring question.

18. **Another orthogonal axis: hard-negative mining at the filter stage.** Current filter is bucket-based multi-probe, uniform. If certain wrong-class neighbors consistently beat correct-class neighbors on Hamming distance, explicitly down-weighting those "hard negatives" in the union construction might help. Adjacent to filter-layer design.

## Tensions

- **T1 (S1 cheap-add-bits vs bit-budget Pareto).** If we're not at an asymptote, S1 helps; if we are, S1 wastes cycles. Node 11 (Pareto measurement) is the prerequisite — should run first.

- **T2 (S3 ensemble test vs the single positive signal being noise).** The CIFAR E3 +0.56pp is the only positive distance-function result across 18 comparisons. By chance alone, noise could produce that. S3 stakes effort on the signal being real. Node 12 (multi-seed verification) is the cheap resolver — should run first.

- **T3 (substrate-native vs scaffolding preference).** NORTH_STAR prefers substrate-native (S1-S6). But if all substrate-native options saturate at ~50% CIFAR-10, we never learn whether the gap is substrate-bounded or approach-bounded without running a scaffold (S7) for comparison.

- **T4 (realistic target vs ambitious target).** Cycle success criterion matters. "Beat SSTT on CIFAR-10" (+6.4pp target) is ambitious and drives aggressive S2/S5/S7 experiments. "Find a substrate-native primitive that explains current results and suggests a principled path" is a knowledge target — could close out on S11's Pareto map alone.

- **T5 (parallel exploration vs sequential).** S11 and S12 are both cheap and should definitely run. Then the question is which ONE of S1-S7 to commit to next — or whether to run S3 and S1 in parallel since they're both cheap.

- **T6 (data-selection axis ignored).** Node 17 (training subset selection) is orthogonal to everything in S1-S7. It might be the single biggest lever we haven't touched. But it's also the hardest to reason about without measurement — depends on whether CIFAR-10's class confusion is driven by genuinely ambiguous samples or by dataset noise.

## Dependencies

- S1 (multi-scale): depends on bit-budget Pareto measurement (Node 11) to inform whether added dims help. Should gate S1 behind the Pareto result.
- S3 (ensemble): depends on CIFAR E3 multi-seed verification (Node 12) — if complementarity is noise, S3 is chasing nothing.
- S2 (pattern codebook): depends on codebook-selection approach (designed vs learned vs clustered). Cheapest: K-means over training blocks. More expensive: learned via gradient descent.
- S4 (per-region τ): minimal dependency; direct code edit.
- S5 (block-summary signatures): depends on defining a block summary — simple count of non-zero trits per block is cheapest. More complex: 3-way summary.
- S6 (signed retention): depends on kernel choice. m4t_mtfp_ternary_matmul_bt exists; just needs a new consumer path.
- S7 (learned encoder): depends on a training loop. Would violate "no Python in toolchain" unless training is in C. Scope is large.

## Open questions

- **Q1: Should the cycle output be a ranked priority list, a selected single experiment, or a strategic reframe?** The synthesis deliverable depends on this.

- **Q2: Is it worth measuring the bit-budget Pareto frontier (Node 11) BEFORE picking a step-change?** Strongly yes — it's cheap and falsifies/confirms the "more bits help" hypothesis that underlies S1.

- **Q3: Is multi-seed verification of the CIFAR E3 +0.56pp (Node 12) worth doing first?** Yes if we're considering S3. Cheap. Resolves T2.

- **Q4: Has the lr_scaffold or distance_function cycle told us the thesis is bounded, or just that we haven't found the right primitive?** This is the meta-question. Two cycles landing mostly-null is evidence that the PROBLEM is hard; it's not evidence that the SUBSTRATE is bounded. S11's Pareto result would help distinguish these.

- **Q5: Is "CIFAR-10 specifically" the right target, or is the project's thesis bigger than any one benchmark?** If CIFAR-10 is forcing the wrong comparison (because it's base-2-pixel-native and our substrate is base-3-native), we might be optimizing against the wrong oracle. Harder question.

- **Q6: What about the "block distance complementary to pair-IG" hint?** Even if +0.56pp alone is noise, the mechanism (different distances make different mistakes) is a generalizable finding. Multi-distance Selective as an architectural component might work even if no single distance beats Hamming.
