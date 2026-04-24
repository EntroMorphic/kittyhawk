---
date: 2026-04-22
scope: LMM cycle — given the lr_scaffold + distance_function cycle results, what's the next step-change worth chasing?
phase: RAW
---

# RAW: step-change search after the distance-function close-out

Two closed cycles told us, collectively, where signal isn't. Now I need to honestly survey what remains, name step-change candidates rather than continuations, and write down what my gut says before reasoning structures it.

## What the measurements have actually ruled out

Across direct_lsh, csa_classifier, and everything I've run on MNIST / Fashion-MNIST / CIFAR-10:

1. **Classifier architecture is not the lever.** CSA single-centroid reproduces classical centroid accuracy (~80% / 65% / 29%). CSA-k converges toward Hamming k-NN from below, never surpasses it. 64-epoch perceptron oscillates. None of these break the Hamming-kNN baseline at any k.

2. **Local distance-function variations are not the lever.** E1 SDOT: tied or worse. E2 weighted_hamming: consistently −0.2pp. E3 block_threshold: tied on Fashion/CIFAR, lost on MNIST, tiny +0.56pp on CIFAR Selective (possibly complementarity, possibly noise). Per-trit reweighting, per-block aggregation, inner-product-as-distance — all in the same neighborhood as uniform Hamming.

3. **Pair-IG is still the only scoring improvement.** +1.95pp on CIFAR-10 Selective. Per-class-pair weighted Hamming. Average of pair-IG weights globally doesn't replicate it (E2 confirmed) — per-pair specialization is load-bearing.

4. **Oracle-over-union (99.99% on all three datasets) was a misleading headline.** "Correct sample exists in top-1600 of 10,000" is a weak condition. The real gap is ranking: the correct sample is buried deep in most queries that Hamming loses. Any scorer that operates on per-trit distance of direct-quantized signatures is bounded by how these signatures separate classes under local metrics.

## What we haven't touched

This is the productive axis. The cycles covered classifiers and distances — both downstream of the signature. Upstream untouched:

**Signature construction:**
- Quantization threshold τ is currently global per-channel (intensity τ, gradient τ). Could be spatial — per-region τ that respects image structure.
- The trit state assignment is symmetric around zero: `v > τ → +1, v < -τ → -1, else 0`. Asymmetric assignment per-region could carry more signal.
- Density parameter (--density 0.395 for CIFAR) is tuned globally; per-region density might give a better signature for regions that carry more class info.

**Feature channels:**
- Currently: intensity + horizontal gradient + vertical gradient. Flat concatenation, single-scale.
- Multi-scale: pyramid of intensity + gradient at 1× and 2× downsampled resolutions. Still routing-compatible — all channels are per-dim ternary.
- More channels: second-order derivatives, oriented gradient bins (4 angles?), color channels as separate trit maps (for CIFAR-10: R, G, B separate rather than averaged).
- Pattern channels: for each 4×4 spatial block, encode which of a small set of prototype patterns the block matches. Trit per pattern, pattern set learned or designed.

**Signature SIZE:**
- CIFAR-10 total_dim=9024 at density 0.395 is what we've been testing. What about higher-resolution signatures with lower density (finer spatial detail + sparser trits)? Or lower-resolution with higher density (coarser but denser)?
- There may be a total-bit-budget / class-discriminability Pareto frontier we haven't mapped.

**Multi-signature / ensemble:**
- Block distance + Hamming combined via voting or min. If block distance sees complementary errors (the CIFAR +0.56pp hint), an ensemble might compound.
- Different preprocessing (deskew vs no-deskew, normalize vs raw) could produce signatures that capture different signal. Ensemble over preprocessing variants.

**Substrate primitives not yet consumed:**
- `m4t_trit_mul` as a feature-extraction operator (pointwise ternary product across signatures) has no consumer.
- `m4t_mtfp_ternary_matmul_bt` (MTFP×ternary matmul) is used only by legacy random-projection path. Could build a substrate-native feature layer on top.
- TBL-based 4-trit pattern dispatch — what E3 half-used. A full "pattern codebook" layer (each 4-trit block → pattern ID → trit encoding per pattern) would be a new consumer.

## Step-change candidates (not increments)

Drawing from the above, naming candidates that could produce >5pp improvement if they work:

**S1: Multi-scale signature pyramid.** Signature = intensity @ 1×, intensity @ 2×, gradients @ 1×, gradients @ 2×. Quadruples total_dim for CIFAR-10 (~36k trits). Hamming on this larger signature could capture multi-scale structure. Routing-compatible; simple to implement.

**S2: Pattern codebook signature.** Replace per-pixel quantization with per-block pattern dispatch. For each 4×4 (or 8×8) block, compare to a small codebook of learned or designed patterns, emit a multi-trit code indicating which pattern dominates. Base-3-native via TBL.

**S3: Multi-distance ensemble.** Run LSH filter once, then score candidates with BOTH Hamming and block_threshold; combine by rank-fusion or normalized-sum. Tests whether the CIFAR +0.56pp complementarity hint compounds.

**S4: Per-region τ.** Calibrate a different τ for each spatial quadrant (or finer grid) on training. Directly addresses the "some regions carry more class signal than others" observation that per-dim weighting tries to approximate after the fact.

**S5: Block-level first-class signatures.** Instead of flat per-dim trits, construct the signature as a sequence of block-level codes. E.g., for each 4×4 block: (count(+1), count(0), count(-1)) encoded as a 2-trit code. Halves signature size but preserves local structure.

**S6: Signed-value retention.** Instead of quantizing to {-1, 0, +1}, keep MTFP values and use a quantized similarity (e.g., MTFP × trit matmul). Uses m4t_mtfp_ternary_matmul_bt. More signal per dim, at cost of sig_bytes.

**S7: Learned signatures (scaffold).** Train a small ternary network (or compressed sensing projection trained via gradient descent + quantized) that produces a per-image signature. Crosses the "no random weights" rule technically but the weights would be LEARNED from data — same semantic status as pair-IG's per-dim weights. This is the closest to base-2-SSTT shape. Scaffolding per NORTH_STAR §4.

## First instincts

- **S1 and S4 are the cheapest.** S1 adds channels; S4 varies τ. Both are single-file changes to direct_lsh's preprocessing. Measurable in hours.
- **S3 is the most interesting SHORT-TERM.** The CIFAR complementarity hint is the only positive signal from the distance_function cycle. Testing whether it compounds is cheap and potentially reveals a new axis (distance ensembles).
- **S2 and S5 are the most THESIS-relevant.** Block-level-first-class signatures align with NORTH_STAR's "base-3 primitives are first-class" principle. TBL dispatch is already base-3-native; making the signature itself block-shaped tests whether the whole pipeline benefits.
- **S6 is a hedge.** If the issue is *quantization*, keeping MTFP values might recover some signal. But it doubles memory and partially undoes the direct-quantization story.
- **S7 is scaffolding.** Would probably work (SSTT-ish). Would not be thesis-relevant.

## What scares me

- **We may have reached the frontier of what direct quantization can offer.** If S1-S6 all land within 2pp of current CIFAR Selective (46.63%), the lesson is "direct per-pixel ternary quantization is not sufficient for CIFAR-10-scale visual discrimination." The step-change would then require departing from the direct-quantization shape — back toward pattern signatures (SSTT) or learned encoders.

- **The oracle-gap framing has been misleading for two cycles.** Chasing the 55pp on CIFAR-10 makes us reach for classifier and scorer levers that can't move it. The realistic target on CIFAR-10 may be **+3-7pp over current 46.63%, not +30-50pp**. If we can't face that honestly, we'll keep wasting cycles on architecture changes bounded by the same 50% asymptote.

- **I keep surfacing "substrate-native" candidates (S2, S5) and measuring "scaffold" candidates (S7) as backstops.** The pattern is correct per NORTH_STAR — prefer substrate-native — but I want to be honest that IF the base-3-native primitives turn out to be structurally bounded at ~47% on CIFAR-10 and SSTT-style at ~53%, the thesis has a problem.

- **The thesis hasn't failed yet, but it hasn't decisively won either.** MNIST won, Fashion-MNIST is approximately tied with SSTT, CIFAR-10 is down 6-7pp. If S1-S6 don't close that, S7 (scaffold) becomes important not because it's thesis-aligned but because it probes whether the gap is substrate-limited or approach-limited.

## Open questions

1. **Which step-change has the best expected-value-per-cost?** S1 (multi-scale) and S4 (per-region τ) are cheapest. S3 (ensemble) tests an existing signal. S2 (pattern codebook) is substrate-aligned but more work. Priority?

2. **Should I map the bit-budget Pareto frontier before picking a step-change?** E.g., measure direct_lsh Hamming at total_dim = {3000, 6000, 9000, 12000, 18000} and see whether the CIFAR ceiling is at asymptote or still climbing. If still climbing, S1 (multi-scale, which increases total_dim) is likely to help. If asymptotic, it won't.

3. **Is the complementarity finding (CIFAR E3 Selective +0.56pp) reproducible across seeds?** If yes, that's a real axis; if noise, it's not. Multi-seed verification is cheap.

4. **Is the cycle's success criterion "beat SSTT on CIFAR-10" or "find a substrate-native shape that explains the gap"?** These are different targets and point to different experiments.

5. **What's the cost of a "representation spec" pass?** I.e., freeze the current architecture and run a month-sized project purely on signature construction, without touching classifiers or distances. Would that produce the CIFAR-10 step-change, or hit the same asymptote?
