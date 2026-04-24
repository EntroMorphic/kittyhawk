---
date: 2026-04-22
scope: LMM cycle — S7 thesis-calibration scaffold. Learned ternary encoder vs direct quantization. Substrate-bounded or encoding-bounded?
phase: RAW
---

# RAW: S7 learned ternary encoder — thesis calibration

The step_change closeout recommended S7 as the higher-information next experiment. Three cycles of direct-quantization work have landed the CIFAR-10 Selective at 48.05%, ~5pp below SSTT's ~53%. The remaining gap could be:

- **Substrate-bounded:** ternary signatures at this dimension and resolution are intrinsically limited on CIFAR-10. Any ternary encoding plateaus near 48%.
- **Encoding-bounded:** direct quantization is a specific encoding choice. A different (learned) encoding could reach SSTT.

S7 is designed to distinguish these. Train a small network to produce ternary signatures, measure, compare.

## What I think I know

SSTT reaches ~53% on CIFAR-10 using ternary attention on pattern-level blocks. Its weights are learned via gradient descent + straight-through-estimator quantization. It's a real ternary system, not a float system.

Glyph's current pipeline: directly quantize pixel/gradient intensities to trits via density-calibrated τ. The signature dim = image dim (roughly). No learned transformation.

A learned encoder would sit between input image and final trit signature, producing a fixed-dim signature that could be different (possibly better) than direct quantization. Canonical architectures:
- Linear: W × x → quantize. Weights learned. This is essentially a ternary-LSH where weights are gradient-trained instead of random.
- MLP: one hidden layer, nonlinearity, then quantize. Adds nonlinearity.
- Patch-based CNN: learn spatially-convolved feature maps, quantize. Closer to SSTT.

## Three architecture options, rough cost

**Option A — learned linear encoder.** W ∈ ℝ^{D_in × D_out}, train via cross-entropy over class labels, straight-through-estimator for sign quantization. D_out = desired signature dim (e.g., 4096). Similar shape to direct quantization + quantized learned projection.

Cost: one matmul + sign. Simple. Could be coded as a PyTorch script → quantize to trit → export table → load in direct_lsh.

**Option B — small MLP encoder.** One hidden layer of size H (256?), ReLU/GELU, then linear + sign to D_out trits. Adds nonlinearity; might unlock features direct quantization can't see.

**Option C — patch CNN encoder.** 3×3 conv → pool → conv → sign. Closer to SSTT. More expressive but more complex.

Each option produces a D_out-trit signature per image. Plug into direct_lsh downstream (filter + pair-IG re-rank unchanged). **The encoder replaces the `glyph_sig_quantize` step.**

## Training location — where does the gradient descent live?

Three options:

1. **External Python/PyTorch.** Train offline, export W as trit tables, load in direct_lsh via new CLI flag `--encoder_weights PATH`. Matches the `m4t_lut_gen.c` precedent: float at build/training time, integer at runtime.

2. **In-repo C trainer.** Write a new C tool `tools/s7_trainer.c` that does gradient descent in float, quantizes, exports. Keeps C-only discipline but doubles implementation cost (I'd have to write autodiff from scratch).

3. **Hand-designed fixed transformations.** Specific per-class prototypes learned via class-centroid or discriminant analysis. Avoid gradient descent entirely. Likely underfits CIFAR but clean and integer-only.

Option 1 is the pragmatic one but crosses the "no Python in toolchain" rule. Option 2 is the purist one but has scope explosion risk. Option 3 is the cleanest but might not answer the question (it's not really a "learned encoder" in the SSTT sense).

## What's the question exactly?

**"Can any ternary encoding reach SSTT (~53%) on CIFAR-10 using Glyph's downstream filter/resolver?"**

If yes → encoding matters, direct quantization was suboptimal, future work on CIFAR is in representation.
If no → the substrate or downstream (filter, resolver, pair-IG) is a hard ceiling, CIFAR is maxed out, move to different benchmarks.

The downstream pipeline is held constant: glyph_bucket + glyph_multiprobe + pair-IG re-rank. Only the trit signature generation changes.

## What scares me

- **Scope creep.** Going from "run an experiment" to "build a ternary training pipeline" is a 1–2 week project. I have a history of escalating from narrow experiments to architecture builds. Red-team: keep this TIGHT. External Python, minimum viable trainer, single configuration, one measurement.

- **The answer might be "in between."** S7 could land at 50% — not SSTT-level, not direct-quantization-level. Then the question "substrate-bounded or encoding-bounded" isn't cleanly answered. It says "encoding matters some but not all." I need to accept that outcome and document it.

- **Straight-through-estimator is a specific training choice.** SSTT uses a specific STE variant with a clipping trick. A naive STE might train poorly and give a low number that doesn't reflect the theoretical best. I could under-report what a learned encoder can achieve.

- **Glyph is C-only per memory.** Deploying a Python training script crosses a stated discipline. Needs to be framed as "scaffold experiment, artifacts external, runtime C-only." The trained weights come in as a binary file. Compile-time-fixed after that.

- **Choosing D_out.** Direct quantization uses ~9000 trits on CIFAR with MS4. Should S7 produce the same-dim signature? Or a smaller one (e.g., 512)? Small learned is "can expressive learning beat inefficient direct"; same-dim is "can expressive learning use the same budget better". Different experiments.

## Naive approach

1. PyTorch script: load CIFAR-10, train a small CNN with trit-quantized output, export 10000 test sigs + 50000 train sigs as raw trit files.
2. New direct_lsh flag `--sigs_from_file train_sigs.bin test_sigs.bin` that loads pre-quantized signatures from disk and skips all of its own quantization.
3. Measure Selective accuracy. Compare to MS4+R4 baseline (48.05%).

This reuses the entire Glyph downstream and isolates the "encoder" variable.

## What I'd want to measure

- S7 Selective accuracy on CIFAR-10 at same signature dim as MS4+R4 baseline.
- Also at smaller dim (1024, 2048) to see if expressive learning does more with less.
- Also on Fashion-MNIST and MNIST for reference (both are less encoding-bound).

## Open questions

1. **Is it acceptable to use PyTorch for training?** Per project scope ("C-only, no Python in toolchain"), no. Per NORTH_STAR §4 (scaffolding sanctioned), yes for this specific calibration experiment. User call.

2. **D_out: same dim, smaller, or sweep?** Cheapest: same dim (direct comparison). Biggest-information: sweep.

3. **Straight-through-estimator or a smarter quantization trick?** Influences the ceiling. Default: naive STE for simplicity.

4. **Should we train to classification loss or to signature-prototype loss?** Classification loss is standard; prototype loss (produce sig close to training-set prototype for correct class) is more direct and trainable with less compute. Probably classification loss is cleaner.

5. **If S7 beats MS4+R4, how much of the win is the encoder vs how much is the scaffolding (we're using a powerful training procedure the baseline doesn't have)?** Partial answer: a random ternary projection (Glyph's legacy path) at same dim is a lower bound; if S7 is below that we've learned nothing; if S7 is above both random projection and direct quantization, the encoder adds signal.

6. **Benchmark: what's the MAXIMUM accuracy the downstream pipeline could deliver regardless of signatures?** Oracle says ~100% on all datasets, but oracle-in-union is a weak condition. The realistic ceiling might be 55% on CIFAR even with perfect encoder. Need to think about this upper bound.

## First instincts

- **Start external Python** if the user agrees. Scope: one training script, ~200 lines, two-hour implementation. Export binary trit files.
- **Same-dim comparison first** — most directly answers the question.
- **Naive STE** for simplicity. If S7 clearly wins, more sophisticated quantization can come later.
- **Gate: S7 must beat MS4+R4 by ≥2pp on CIFAR Selective to count as "encoding-bounded confirmation."** Weaker wins are ambiguous.
