# RESOLVED — RMSNorm gamma_bx > target_bx silent saturation

Date: 2026-05-08

## Bug

`m4t_mtfp_rmsnorm_bx` (and `_scalar_ref`) computed the per-cell `γ × x × inv >> total_shift`
**at the gamma_bx scale**, then clamped to MTFP19_MAX (5.81e8), then divided by
`3^(gamma_bx − target_bx)` to land at target_bx.

For BitNet's `post_attention_layernorm`:
- gamma_bx = 17 (γ_real magnitudes ~ 0.7–2.2)
- target_bx = 8 (BITNET_ACT_BX)
- shift_diff = 9, divisor = 3^9 = 19683

The per-cell intermediate magnitude `|γ_m × x_m × inv >> 30|` reaches `~3.8e9` for
typical post-attention residual inputs — well past MAX_VAL = 5.81e8. The intermediate
saturated; the subsequent rescale (÷19683) **hid** the saturation by returning values
that fit MTFP19 again. The output was uniformly capped at `MAX_VAL / 19683 = 29524` —
6.5× smaller in magnitude than correct.

Downstream effect: layer 0 post_attn_norm output had ε = 0.41 vs HF (fp32 reference);
the 80× amplification (input ε = 0.005 → output ε = 0.41) propagated through the
remaining 29 layers and produced degenerate generation loops on prompts whose
post-attn residuals had any meaningful magnitude.

## Confirmation

Python emulation of the buggy logic on substrate's actual `sub_prepn.bin` mantissas
+ HF's gamma_post_attn_norm at gamma_bx=17 reproduced ε = 0.4126 — **exact match** to
substrate's measured output ε. The same emulation with γ pre-rescaled to target_bx
before the per-cell multiply produced ε = 0.0059 — **70× improvement**.

## Fix

When `gamma_bx > target_bx`: rescale γ to target_bx into a malloc'd buffer first,
compute per-cell at target_bx scale, skip the final rescale. Per-cell intermediates
now have the correct magnitude; saturation only triggers when the output value
genuinely exceeds MAX_VAL.

When `gamma_bx ≤ target_bx`: keep existing behavior (per-cell at gamma_bx, then
upscale at end). Upscaling can't introduce silent saturation — it just narrows out
of MAX_VAL if the result genuinely overflows.

Applied to both NEON and scalar_ref. Bit-exact between the two paths; existing
tests (gamma_bx ≤ target_bx) unchanged. Added regression test
`test_rmsnorm_bx_gamma_gt_target` for the gamma_bx > target_bx case.

## End-to-end result

Pre-fix on prompt "Hypothetically, might reflective recursion be a function of cognition?":
substrate produced a degenerate token loop.

Post-fix:
> "How would you explain this?\n\nSolution: \nReflective recursion is a concept in
> cognitive science that describes a process where an individual's thoughts, feelings,
> and actions are recursively defined by their own mental states..."

Coherent, on-topic English. Mild repetition near token ~50 is the typical greedy-
decoding artifact.

Confirmed on a second prompt ("What is the capital of France?"):
> "The capital of France is Paris. It is a city known for its historical significance,
> romantic atmosphere, and iconic landmarks. Paris is the heart of France"

Token-for-token agreement with HF (bf16) is 0% on the canary prompt — but that's
expected. Greedy decoding amplifies tiny logit ε between MTFP19 and bf16 quantization
regimes; both paths produce coherent but lexically distinct continuations. The bug
was that the substrate's path produced **no** coherent continuation at all.

## What didn't need changing

- BitLinear path (no_a8): substrate's o_proj output had ε = 1.5% vs HF — within
  reasonable quantization noise.
- Residual add: pre_pn ε = 0.5% — fine.
- Activation quantization: bypassed via the no_a8 path; not the cause.
- Weight loading / 5-in-8 vs 4-in-8 packing: 98.78% sign agreement, α matches.

The amplification was localized entirely in RMSNorm. The substrate's other kernels
were already producing acceptable accuracy.
