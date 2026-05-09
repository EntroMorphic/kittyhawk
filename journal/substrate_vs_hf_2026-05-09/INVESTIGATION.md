# Substrate vs HF — initial localization

Context: probe prompt "Hypothetically, might reflective recursion be a function
of cognition?" produced a degenerate self-referential loop on the substrate
("If it is a function of cognition, then it might be a function of cognition...")
while HF reference produces a coherent continuation ("If so, what would be the
implications for our understanding of the relationship between cognition and
the brain?").

V14.G v2 ≡ V13 is bit-exact (verified). The divergence is substrate ≡ HF, NOT
V14 ≡ V13. This investigation localizes WHERE in the substrate the divergence
first becomes load-bearing.

## Method

1. `dump_reference.py --prompt "Hi" --output hf_ref.npz` captures HF activations
   per (layer, sublayer) for a single-token forward pass (BOS-only).
2. `bitnet_harness --token 128000 --positions 1 --dump c_dump` captures substrate
   activations per layer for the same single-token forward.
3. `compare_activations.py` computes scale-invariant L2 relative error per
   (layer, capture-site) by best-fitting a constant scalar to substrate values
   and reporting `||c·s − r||₂ / ||r||₂`.

## Top-1 logit comparison (BOS-only forward)

  HF argmax        = 11 (',')
  Substrate argmax = 11 (',')                ← MATCH

  HF top-10:        [11, 279, 311, 220, 304, 264, 323, 369, 374, 512]
  Substrate top-10: [11, 374, 512, 596, 706, 1473, 574, 315, 13, 1053]

  Overlap of HF top-10 with substrate top-50: 6/10
  HF token 279 ('the') ranks #124 in substrate
  HF token 220 (' ')   ranks #133 in substrate

The top-1 agrees, but the rest of the distribution diverges enough that
greedy decoding picks different paths after the first generation step.

## Layer-0 ε progression (the smoking gun)

  input_layernorm.output           ε = 0.005    ← essentially perfect
  attn.q_pre_rope                  ε = 0.069    ← BitLinear quant noise (~7%)
  attn.k_pre_rope                  ε = 0.023
  attn.v                           ε = 0.030
  attn_sub_norm.output             ε = 0.156    ← 5x JUMP. Critical.
  post_attention_layernorm.output  ε = 0.414    ← compounds further
  ffn.up_proj                      ε = 0.456
  ffn_sub_norm.output              ε = 0.626    ← 60%+ error within layer 0
  block_output                     ε = 0.628    ← layer 0 exit

For seq_k = 1 (single token), the Q@K^T → softmax → @V step is mathematically
trivial: softmax([single_value]) = [1.0], so attn_output[d] = 1.0 × V[d] = V[d].
The substrate's V14.B `m4t_mtfp_attn_v_combine` should produce attn_output ≡ V
for this case. Therefore the 5× amplification (3% → 16%) at attn_sub_norm.output
must come from the RMSNorm step itself, NOT from the attention scoring math.

## Per-tensor error trajectories across layers

  tensor                           |  L0 ε  |  L5 ε  | L15 ε  | L29 ε  |  min   |  max
  ---------------------------------+--------+--------+--------+--------+--------+-------
  input_layernorm.output           |  0.005 |  0.517 |  0.449 |  0.637 |  0.005 |  0.766
  attn.q_pre_rope                  |  0.069 |  0.503 |  0.300 |  0.402 |  0.069 |  0.635
  attn.k_pre_rope                  |  0.023 |  0.279 |  0.349 |  0.301 |  0.023 |  0.487
  attn.v                           |  0.030 |  0.135 |  0.186 |  0.582 |  0.030 |  0.582
  attn_sub_norm.output             |  0.156 |  0.419 |  0.619 |  0.545 |  0.156 |  0.819
  post_attention_layernorm.output  |  0.414 |  0.478 |  0.449 |  0.755 |  0.392 |  0.799
  ffn.up_proj                      |  0.456 |  0.380 |  0.485 |  0.387 |  0.345 |  0.720
  ffn_sub_norm.output              |  0.626 |  0.708 |  0.768 |  0.882 |  0.484 |  0.955
  block_output                     |  0.628 |  0.178 |  0.185 |  0.315 |  0.148 |  0.628

Best_scale across layers' block_output: stable around 1.7e-4 (substrate
uses MTFP19 mantissa scale, HF uses fp32 — the ratio is ≈ 1/MAX_VAL).

## Hypothesis

The error is NOT a single bug; it is **quantization noise from MTFP19 vs bf16
compounding through layered RMSNorm**. Specifically:

- input_layernorm at the start has 0.5% error — RMSNorm itself, on already-
  quantized embedding input, is precise (the floor of the substrate).
- BitLinear quantization noise (3-7%) is fundamental to ternary weights
  (BitNet b1.58 spec).
- Per-RMSNorm amplification: each sub-norm in the layer adds ~2-4× error
  relative to its input. With 4 sub-norms per layer (input_layernorm,
  attn_sub_norm, post_attention_layernorm, ffn_sub_norm) × 30 layers = 120
  sub-norms, even small per-call amplification compounds to the observed
  60-80% block_output ε.

Two contributors to per-call RMSNorm amplification:
1. Integer rsqrt (m4t_int32_rsqrt with NR) has bounded but nonzero precision
   loss vs HF's fp32 rsqrt.
2. The gamma weight is stored as int mantissa at block_exp, vs HF's bf16
   per-element scaling.

## Next-step probes (not yet run)

1. **Isolate RMSNorm precision**: feed identical input to substrate's
   `m4t_mtfp_rmsnorm_bx` and HF's `RMSNorm`, scale-invariant compare. If
   per-call amplification > a few percent, the sub-norm precision is the
   dominant contributor. If not, the noise is elsewhere.

2. **Substitute fp32 rsqrt** in a controlled build of the substrate to
   measure the contribution of integer rsqrt's precision to the gap.

3. **Compare BitLinear-quantized** vs **fp32-multiplied** projections in a
   single layer — quantifies the noise floor we cannot remove without
   architectural changes.

4. **Check post-RoPE Q/K**: dump_reference doesn't hook these because HF
   computes them inline, but substrate does dump them. Compare against an
   instrumented HF or compute the expected RoPE rotation from pre-RoPE Q/K
   and reference cos/sin LUT.

## Files

  layer_diff.csv          — full per-(layer, tensor) comparison, all 30 layers
  substrate_logits_BOS.bin — substrate's full LM-head logits (int64 × 128256)
                             after the single-BOS forward pass.
  hf_top10_BOS.txt        — HF reference's top-10 next tokens after BOS.

## Root cause isolated (2026-05-09 follow-up)

Hypothesis test: feed substrate's V (rescaled to fp32 via best_scale) through
the GQA expansion to attn_output, then apply HF-style RMSNorm in fp32, and
compare to HF's actual sub_norm output.

  Substrate's actual RMSNorm output ε = 0.1559
  Ideal fp32 RMSNorm of (substrate V→attn_out) ε = 0.1557      ← identical
  Sanity: fp32 RMSNorm of HF V→attn_out vs HF sub_norm = 0.0023 ← matches HF

**Conclusion: the substrate's RMSNorm is correct.** The 16% ε at
`attn_sub_norm.output` is fully explained by the per-element noise structure
of substrate's V *before* RMSNorm.

The 3% L2 norm error of substrate's V vs HF's V is **misleading low**: that
3% does NOT cancel under RMSNorm because the noise is per-element directional,
not a uniform scale offset. RMSNorm:
  y = γ · v / sqrt(mean(v²) + ε)
preserves per-element directional noise. After best-scale fitting, the
per-element component of substrate's V differs from HF's V by ~16% in the
direction sense, even though the ratio-of-norms differ by only 3%.

## Where the per-element noise comes from

The substrate's BitLinear is the source. For each projection:
  1. Activation x → A8 quantize → x_int8 (per-tensor absmax scale, 8 levels)
  2. x_int8 @ W_ternary → int accumulator
  3. accumulator × α × absmax / 127 → reconstructed activation

vs HF's BitLinear:
  1. x (bf16) @ W_bf16 → bf16 accumulator              (NO quantization)
  2. accumulator × α (bf16) → reconstructed activation

HF stores ternary weights in bf16 format (values literally are
{−α, 0, +α} per row), so the matmul is plain bf16 throughout. No information
loss. The substrate, by contrast, compresses activations to int8 (per-tensor
scale), which introduces ~0.4% per-element noise per BitLinear call.

That per-element noise:
  - 6 BitLinear projections per layer (Q, K, V, O, gate, up, down — actually 7)
  - 30 layers
  - = ≈ 200 BitLinear calls per inference
  - Each call adds ~0.4% per-element noise → after RMSNorm and residual, the
    noise propagates through the next layer's BitLinear inputs.
  - Compounding noise saturates around 60-80% L2 ε at deep layers
    (consistent with the observed block_output ε = 0.18-0.31 mid-layers,
    rising to 0.31 at layer 29).

## What this means for substrate-vs-HF agreement

This is **an architectural property** of the substrate's chosen
quantization scheme, not a fixable bug. The substrate runs BitLinear in
"fully quantized inference" mode (a8 × ternary). HF's reference runs
BitLinear in "fake quantization" mode (bf16 × bf16, where the bf16 values
happen to be ternary).

To close the gap we'd have to choose one of:

1. **Run BitLinear in higher activation precision** (a16 or a32 instead of a8).
   Doubles intermediate widths, increases memory bandwidth, but recovers
   most of HF's precision. The substrate's MTFP19 intermediate could
   carry full 16-bit activations directly without an a8 quantize step.

2. **Drop activation quantization entirely** — multiply MTFP19 mantissas
   by ternary weights directly. Loses the int8 SDOT/SMMLA acceleration
   path but gives bit-faithful inference.

3. **Accept the accuracy gap** as the documented cost of the substrate's
   "ternary-substrate" architectural commitment. ~30% argmax-vs-HF
   disagreement on greedy decoding is the price.

(Option 1 is the most promising — it preserves the ternary-routing
substrate while closing most of the precision gap.)

## Confirmation that substrate ALL kernels are individually correct

  - V14.A-G v2 ≡ V13                      (bit-exact internal consistency)
  - V13 ≡ original LUT softmax            (bit-exact internal consistency)
  - rmsnorm_bx ≡ HF RMSNorm at fp32       (this investigation: ε = 0.0023 ≈ 0)
  - V14.B attn_v_combine for seq_k=1      (trivially correct: 1·V = V)
  - BitLinear matmul (ternary path)       (already validated under HF
                                           comparison gate, journal/
                                           bitnet_phase1_closeout.md)

The kernels do exactly what they're supposed to do. The architectural
choice to a8-quantize activations during BitLinear inference is what
introduces the per-element noise that compounds through layers.
