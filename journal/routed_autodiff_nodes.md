---
date: 2026-04-23
scope: LMM cycle — routed autodiff in NEON C
phase: NODES
---

# NODES

## Discrete ideas

1. **Routed autodiff is a SUBSTRATE EXTENSION.** Not a consumer experiment. Dense autodiff libraries (PyTorch etc.) do not express routing natively. Glyph adding it would be a thesis-relevant primitive class, not an imitation of existing tools.

2. **Forward routing primitives already exist.** `m4t_route_threshold_extract`, `distance_batch`, `topk_abs`, `apply_signed`, `signature_update`. All NEON-native. Backward passes are the missing half.

3. **Backward for each routing primitive needs a sanctioned STE choice.**
    - `threshold_extract` (sign-at-tau): backward = clipped identity on |x| ≤ tau_grad bound. Standard STE.
    - `topk_abs` (hard selection): backward = identity on selected positions, zero on unselected. Straight-through topk.
    - `apply_signed` (gated accumulation): backward = scatter gradient into selected tiles by sign.
    - `distance_batch` (XOR + popcount): backward = zero (pure signature comparison; upstream gradient ends here under STE).
    - `signature_update` (one-shot setup): no backward needed in inner loop.

4. **SDOT backward shape.** Forward `Y[m,n] = sum_k X[m,k] * W[n,k]` with X int8, W ternary int8, Y int32. Backward given dY:
    - `dX[m,k] = sum_n dY[m,n] * W[n,k]` — ternary weight matmul over dY, SDOT-shaped.
    - `dW_float[n,k] += sum_m dY[m,n] * X[m,k]` — accumulates into FLOAT latent, not ternary. Needs int32 or float accumulation.

5. **Weight latents live in float; deployed weights in trit.** Training loop: keep `W_latent ∈ float[T × H × D]`; periodically quantize to `W_trit_packed ∈ uint8[T × H × Dp]` for the forward path. Hardware SDOT still operates on the packed trits. Gradient updates float latents; trit packing refreshed every N steps.

6. **STE clipping is load-bearing.** Naive STE (full identity backward) lets latents drift arbitrarily, destabilizing training. Standard mitigation: clip backward gradient where |W_latent| > 1 (gradient saturates at trit boundary). Per-primitive STE decisions are hypotheses that need empirical validation.

7. **MVP = single routed ternary encoder.** One layer. T tiles × H hidden × D dims. Routing via input-derived signature. Output = per-image ternary code. Classification via auxiliary softmax head on pre-quantized activations. ~150-300 lines of trainer C + ~200 lines of backward NEON kernels.

8. **What NOT to build in the MVP.**
    - Multi-layer compositions.
    - General autograd tape (op recording + replay).
    - Dynamic graphs.
    - Everything beyond one forward+backward pass hand-coded.

9. **STE for top-k is the riskiest design choice.** Even deep-learning practitioners disagree — some use soft top-k (differentiable relaxation via softmax), others use STE. Start with STE for simplicity. If training doesn't converge or quality is poor, try soft top-k as second-pass.

10. **Loss function choice.** Auxiliary softmax classifier head on pre-sign activations gives dense gradient signal. Contrastive loss on signatures gives a more routing-shaped signal but is harder to tune. Start with auxiliary softmax; revisit.

11. **Tree location.** New directory `train/` at repo root. Sibling to `m4t/`, `src/`, `tools/`. Contains a `routed_trainer` tool + its backward NEON kernels. Never linked into libm4t or libglyph (substrate discipline — training is consumer-layer).

12. **Training is float at runtime of trainer; inference is integer.** The trainer tool itself USES float (for latents) but PRODUCES integer/trit artifacts. Consistent with `m4t_lut_gen.c` precedent (float at build-time, int at runtime). Artifact committed: packed-trit tile weights and their signature index.

13. **Gate: MVP trained routed encoder on CIFAR-10 reaches ≥50% Selective.** 48.05% is direct-quantization baseline. 53% is SSTT. ≥50% = routed autodiff materially improves signature quality. <48% = training is broken; debug. 48–49% = ambiguous.

14. **MS4+R4+FSTAT+BRUTE are all downstream of this.** The same trainer can produce signatures that get fed through any existing scorer. The existing CLI flags (`--sigs_from_file` was in the S7 synthesize but not implemented; needs to happen now since the trainer outputs files) become the reload path.

15. **NEON backward kernels to write.**
    - SDOT-backward-dX: ternary weights @ float dY → float dX. NEON via vdotq_s32 on int8 with float accumulation.
    - SDOT-backward-dW: float-outer-product of X and dY on selected tiles. Most expensive hot loop.
    - Signed-accumulation-backward: scatter dY into selected tile dX slots with sign. Simple index loop.
    - STE-sign-backward: identity with clipping. Element-wise NEON.

16. **Memory footprint.** For CIFAR-10 with D_in=3072 pixels, T=16 tiles, H=256 hidden, D_out=11000:
    - Float latents: 16 × 256 × 11000 × 4 bytes = 180 MB.
    - Trit packed deployed weights: 16 × 256 × 11000 × 2 bits = 11 MB.
    - Forward activations (training): batch × (D_in + T hidden-outs + D_out) floats. Moderate.
    - Backward gradients: same shape as weights (180 MB float) + intermediate gradients.
    Fits on modern machines; not free but manageable.

17. **Training compute.** Batch 256, 20 epochs, 50k samples = 4000 steps. Per step: forward (maybe ~1-5 ms), backward (maybe ~5-20 ms), update (fast). Full training ~1-2 hours of CPU time. Feasible. GPU would be faster but GPU is out of scope for this substrate.

## Tensions

- **T1 (Scope: MVP vs general engine).** MVP is 1-2 weeks and answers S7. General engine is 1-3 months and opens a substrate direction. User may want either; propose MVP.

- **T2 (Float latents vs integer latents).** Float is classical, known-convergent. Integer fixed-point latents are substrate-pure but experimental. First pass float; note integer as future work.

- **T3 (STE vs soft routing).** STE is simpler but may train worse. Soft routing (softmax over scores, differentiable) is more complex but proven in e.g. Switch Transformer. Start STE.

- **T4 (In-tree vs external).** Trainer is C, so should be in-tree. But it's experimental scaffolding. Place in new `train/` directory with experimental tag. Not production.

- **T5 (Backward NEON vs backward scalar).** Scalar backward is 10× slower but simpler to debug. Propose: write scalar first, measure, then NEON hot loops once tests pass.

- **T6 (Test strategy).** How do you test an autodiff engine? Standard: numerical gradient check — compute dW via finite differences, compare to analytical backward. Should be the FIRST test after any kernel.

- **T7 (Output format).** Trainer should produce packed-trit signatures in the SAME FORMAT as `glyph_sig_quantize` outputs so `direct_lsh --sigs_from_file` can load them. Format is already standardized.

## Dependencies

- `m4t_route_*` forward primitives: already exist.
- `m4t_mtfp4_sdot_matmul_bt`: forward SDOT already exists.
- `m4t_pack_trits_*`: already exist.
- New: backward kernels in `train/` directory.
- New: `direct_lsh --sigs_from_file` loader (was synthesized but not built).
- Need to decide: float vs fixed-point for latents.

## Open questions

- **Q1: scope decision.** MVP (ONE routed ternary encoder, ~1-2 weeks, answers S7) or general autograd engine (1-3 months, thesis-grade substrate extension)? MVP first, escalate later.
- **Q2: STE variant.** Naive identity, or clipped to |W_latent|≤1? Clipped is standard; use clipped.
- **Q3: latent precision.** Float32 or INT32 fixed-point? Float32 first pass; INT32 as follow-up.
- **Q4: base architecture.** Linear-only (one W tile, no routing) or actually-routed (T tiles with routing)? Actually-routed. The WHOLE POINT is routing.
- **Q5: training loss.** Auxiliary classifier head vs contrastive vs prototype. Auxiliary classifier — simplest.
- **Q6: where to measure.** CIFAR Selective is the gate. Report brute_1nn as control. Fashion/MNIST as sanity.
- **Q7: test strategy.** Numerical gradient check on every new kernel. No shortcuts.
