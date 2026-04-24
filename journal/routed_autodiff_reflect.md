---
date: 2026-04-23
scope: LMM cycle — routed autodiff in NEON C
phase: REFLECT
---

# REFLECT

## The core insight

**Routed autodiff isn't "autodiff applied to routing." It's a different computation graph primitive set.** Dense autodiff thinks in tensors and matmuls; every op has a backward-tensor shape that mirrors its forward-tensor shape. Routed autodiff thinks in (signature, decision, sparse-dispatch) triples; the backward isn't a tensor reshuffle but a **sparse gradient scatter** conditioned on the forward decision.

If Glyph builds this right, the substrate gains a new op class the way SDOT gave it ternary matmul. Not "we have autodiff now" but "we have routed backward." Every consumer that wants learned routed computation can use it. That's substrate-level, not tool-level.

## Why the MVP framing is correct

The user said "let's autodiff in NEON-optimized C, routed." Two valid readings:

1. **Build a general routed-autograd engine.** Multi-op tape, dynamic graphs, composable ops, tested to death. Matches what PyTorch gives for dense. Thesis-grade. Multi-month scope.

2. **Build the minimum that demonstrates the pattern and answers one open question.** One-layer routed ternary encoder, hand-coded forward + backward, CIFAR measurement. 1-2 weeks.

The reflection strongly prefers reading 2 for this cycle because:

- **The S7 question is still open.** Without a routed trainer, we don't know if a learned ternary encoder beats direct quantization on CIFAR. The MVP answers this.
- **Building a general engine before answering the question is premature.** If routed autodiff doesn't help on CIFAR even in the simplest form, the whole direction is in doubt. MVP tests the premise.
- **The MVP is composable.** Once one forward+backward pair is working, adding more is incremental. The MVP isn't throwaway — it's the first primitive in a future engine.
- **User's working-style memory: confirm large moves.** A multi-month engine build needs explicit sign-off. The MVP is already a 1-2 week commit; worth confirming. A proper engine is ~10× that.

## Why the routing is load-bearing, not incidental

The user's phrase "routed, of course" signals NORTH_STAR alignment. Let me reason from hardware:

- **Dense matmul wastes gradients.** On ternary weights with 1/3 zeros, 33% of dW updates are on weights that contributed zero to forward and will receive meaningless gradient (STE). You spend compute updating weights that can't move the loss.

- **Routed forward already ignores 75% of weight tiles per query** (top-k=4 of T=16). The backward gradient should flow INTO THE SAME 25% only. Dense backward would uniformly update all 16 tiles, corrupting the 12 that didn't fire.

- **SDOT hardware expects packed int8 ternary inputs.** A routing-shaped backward can stay SDOT-native — backward dX is SDOT against the selected tile's ternary weights. Backward dW on selected tiles is a float outer product (accumulated into latents, not ternary). Both are NEON-native.

**The thesis claim:** routed backward is cheaper AND more informative per gradient step than dense backward on ternary. Same reason routing forward beats dense forward.

## Resolved tensions

**T1 (MVP vs engine):** MVP first. Single forward+backward pair, one tool, one measurement. If the MVP validates the thesis (routed autodiff can train a ternary encoder that beats direct quantization on CIFAR), consider an engine in a follow-up cycle.

**T2 (float vs integer latents):** float first. Integer fixed-point latents are a research question in themselves; the MVP shouldn't take that on. Substrate purity restored when the trained weights export as int/trit.

**T3 (STE vs soft routing):** STE first. The user hasn't asked for soft routing; it adds complexity; it may or may not help. MVP uses STE (clipped identity where the trit-boundary is saturated).

**T4 (in-tree vs external):** in-tree, new `train/` directory. Consistent with C-only discipline. Tagged experimental.

**T5 (scalar backward first, NEON after):** yes. Write scalar backward for each kernel, numerically gradient-check, THEN NEON-ify. Premature NEON is debugging hell.

**T6 (gradient check is the first test):** codified. Every new backward kernel gets a numerical-gradient-check test before anything else. Same pattern as PyTorch's gradcheck.

**T7 (output format):** existing `m4t_pack_trits` + the packed-trit file convention. Trainer writes exactly what `glyph_sig_quantize` would have written. `direct_lsh --sigs_from_file` reads it.

## Hidden assumptions I was making

- **"Latent float weights is acceptable for a substrate-aligned repo."** Defensible under NORTH_STAR §4 (scaffolding) and §12 (sanctioned float at build/training time). The output is int/trit; the training is a transient float pass. Same precedent as `m4t_lut_gen`.

- **"STE is the right backward for routing."** Actually untested. STE is the DEFAULT in ternary literature, but it's a hypothesis. The MVP will tell us whether the gradient signal that leaks through STE is enough to train.

- **"Single-layer routed encoder is sufficient for S7."** Also untested. SSTT uses multi-layer attention. A single layer may plateau below 53%. If it does, we've learned that single-layer isn't enough; that's still informative.

- **"The training loss should be classification via an auxiliary head."** Pragmatic choice. Contrastive would be more native to the k-NN downstream. Going with the simpler choice first.

## What I now understand

1. **The MVP is a consumer tool with a clear contract.**
    - Input: training data (CIFAR-10 loaded the same way `direct_lsh` loads it).
    - Output: packed-trit signature files (train + test) in the same format as `glyph_sig_quantize`.
    - Between: one forward+backward pass per training step, STE through routing gates, float latent weights, trit deployment.

2. **Three kernels to write carefully.**
    - `sdot_backward_dX`: NEON SDOT-shape, given dY and ternary W_trit, compute dX.
    - `sdot_backward_dW`: outer product of X and dY on selected tiles, accumulated into float latents.
    - `routed_dispatch_backward`: scatter upstream dY through the k selected tiles, zero elsewhere.

3. **Plus helpers:**
    - STE backward (clipped identity).
    - Signature derivation from X (forward; reuse existing `threshold_extract`).
    - Latent-to-trit quantization (periodic re-pack after updates).

4. **Testing strategy is non-negotiable.**
    - Every new kernel: write forward, write backward, finite-difference gradient check against backward.
    - Build a toy problem (2-class linear separation) where the learned weights should converge to known values; verify they do.
    - THEN run CIFAR.

5. **The cycle's deliverable is two things:**
    - (a) MVP routed trainer tool.
    - (b) Measured CIFAR Selective accuracy with the trained encoder vs direct MS4+R4.

6. **Gate: ≥50% Selective on CIFAR.** Tight, clean, interpretable.

## Open residuals

- **R1 (engineering scope).** 1-2 weeks is my estimate. Could be 2-3 if gradient check reveals bugs. Pre-declared: if by end of week 2 training still doesn't converge on a toy problem, halt and reassess.

- **R2 (architecture choice: how many tiles, how much hidden).** T=16, H=256, D_out=11000 for CIFAR is the MS4+R4-matching config. Smaller first for iteration speed. Pre-declared: iterate on T=4, H=64, D_out=2048 CIFAR toy. When training converges, scale to match MS4+R4.

- **R3 (follow-up engine).** If MVP validates, the natural next cycle is generalizing — adding ops, maybe a simple tape, maybe multiple layers. Not this cycle.

- **R4 (what if MVP fails).** If training diverges or plateaus below 48%, the cycle has failed to answer S7 but has still produced (a) a routed backward kernel set, (b) empirical evidence that single-layer routed is insufficient, (c) a starting point for deeper architectures. Not zero value.
