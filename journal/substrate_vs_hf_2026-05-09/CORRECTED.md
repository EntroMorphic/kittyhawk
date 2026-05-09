# Corrected analysis (second revision, 2026-05-09)

The user pushed back on the "ternary inherently can't match bf16" framing.
They were right. Re-investigation found the substrate is **much closer to
correct** than the first revision claimed, AND HF-with-the-same-ternary
weights produces COHERENT output. So the loop is a substrate bug, not a
ternary-precision limit.

## What we found (correctly this time)

1. HF's BitLinear `online_quant=True` runtime-quantizes the bf16 master
   weights to ternary via `WeightQuant`:
     scale = 1.0 / weight.abs().mean().clamp(min=1e-5)   # per-tensor
     weight = (weight * scale).round().clamp(-1, 1) / scale
   Effective forward weight = ternary_sign × mean(|W_master|).

2. The packed-ternary repo (`microsoft/bitnet-b1.58-2B-4T`) stores the
   ternary signs explicitly + a per-tensor `weight_scale = mean(|W_master|)`.
   Substrate loads from this repo via convert_weights.py.

3. Substrate's loaded ternary signs match HF's runtime-ternarized signs
   at **98.78%**. The per-tensor α matches (1.21875 vs 1.21885). So the
   substrate has the right weights.

4. Direct experiment: load the packed-repo ternary into HF (override
   bf16 master with ternary_sign × scale, set `online_quant=False`,
   `weight_scale=1`). Result: HF generates
     "If so, what would be the implications of this for our
      understanding of the relationship between cognition and behavior?
      Answer: Reflective recursion is a concept that has been"
   — **COHERENT.** Same packed weights as the substrate. So the
   substrate's loop is a substrate-implementation bug, NOT a property
   of ternary inference.

## Per-layer comparison vs the CORRECT reference (HF-with-packed-weights)

Re-running the layer-0 comparison with this proper reference (vs the
incorrect HF-bf16-master reference used previously):

  tensor                    | sub vs HF-bf16 | **sub vs HF-packed** | improvement
  --------------------------+----------------+----------------------+-------------
  input_layernorm.output    |     0.005      |        0.005         |   0×
  attn.q_pre_rope           |     0.069      |      **0.0095**      |   7× better
  attn.k_pre_rope           |     0.023      |      **0.0044**      |   5× better
  attn.v                    |     0.030      |      **0.0036**      |   8× better
  attn_sub_norm.output      |     0.156      |      **0.0195**      |   8× better

The substrate's per-cell ε vs the correct reference is 0.4-2% — much
smaller than the 7-16% we incorrectly attributed earlier (that gap was
because HF-bf16 uses runtime-quantized signs that differ from packed
signs by 1.2%, which we were charging to the substrate).

So the substrate's BitLinear math is **almost right**, within 1-2% per
cell of HF-with-packed-weights. The bug is small enough to be subtle
but large enough to compound across 30 layers into a different
generated sequence (loop vs coherent).

## Where the remaining ε comes from (unresolved)

The 1-2% per-cell residual:
- Is NOT a constant-scale issue (best-scale-fit doesn't remove it).
- Is per-element directional, suggesting different rounding in some op.
- HF runs everything in bf16 (7-bit mantissa); substrate runs in MTFP19
  (29-bit mantissa) with a8 ActQuant (8-bit per element). Theoretically
  substrate is MORE precise per cell, but the noise pattern DIFFERS
  from HF's, producing different argmax decisions downstream.

Open candidate hypotheses (NOT yet tested):
1. RMSNorm internal precision: HF uses fp32 throughout; substrate uses
   integer rsqrt. Equivalent precision but different rounding could
   produce per-element drift in the normalized output that differs from
   HF's exact direction.
2. ActQuant rounding mode: HF uses Python `round()` (round-half-to-even);
   substrate's a8 may use different (round-half-away-from-zero, or
   truncation toward zero on negatives). Per-element disagreement at the
   0.5-LSB threshold.
3. RoPE LUT precision: substrate uses cos/sin LUT at int32 with limited
   resolution; HF computes them in fp32 per-call.
4. FFN's gate × up step uses MTFP19 storage at GATE_ACT_BX; HF uses
   bf16. Squared activations may differ in sign of small values.

## Path forward (NOT yet executed)

1. Verify substrate's ActQuant matches HF's exactly: rounding mode and
   per-tensor vs per-token scale (they should match for batch=1, but
   bigger seq could differ).
2. Compare RMSNorm output bit-by-bit against HF's reference (I claimed
   earlier this matched fp32 to 0.002 ε; that was vs WRONG reference).
3. RoPE LUT precision: if substrate's int LUT loses sub-LSB precision
   on Q/K, downstream attention scores drift.

## What this changes in the architecture story

The substrate IS suitable for matching HF-quality ternary inference in
principle (the math checks out within 1-2%). The remaining gap is
implementation precision details that compound. A careful kernel-by-
kernel audit comparing substrate's intermediate outputs to HF-with-
packed-weights would localize the drift sources.

The substrate is NOT fundamentally broken; it is fundamentally close to
right, and the loop on this specific prompt reflects compounding of
small per-cell drift through 30 layers, not a deep architectural issue.
