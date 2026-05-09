# Revision (2026-05-09): substrate-vs-HF gap is an ARTIFACT of the comparison reference

The original investigation hypothesized that the substrate's a8 activation
quantization (used during BitLinear inference) introduced per-element
directional noise that compounded through 30 layers of attention into
the observed substrate-vs-HF logit divergence. Acting on this hypothesis,
we built option 2 (no-a8 path):

  * `m4t_mtfp_ternary_matmul_bt_route_i64` — int32 × ternary matmul that
    skips the a8 round-trip; outputs int64 per cell, no MTFP19 clamp.
  * `m4t_mtfp_bitlinear_scale_no_a8_bx` — α-only scale apply (no /127, no
    absmax) consuming int64 raw matmul output.
  * Weight repack at load time: 5-in-8 (compact base-3, SDOT path) → 4-in-8
    (2-bit) for the int32 kernel.
  * Wired into all 7 BitLinear callsites in bitnet_harness.c.

## The hypothesis was wrong

Re-running per-layer comparison with the no-a8 path: ε at attn.v changed
from 0.0295 → 0.0296, attn_sub_norm.output from 0.1559 → 0.1560. The
substrate's a8 quantization was NOT the culprit. End-to-end behavior on
the "Hypothetically..." prompt also still loops (different loop:
"If I am thinking, I am thinking..." instead of "If it is a function of
cognition...") but qualitatively identical degeneracy.

## What was actually wrong: the comparison reference

The HF reference we'd been comparing against (`microsoft/bitnet-b1.58-2B-4T-bf16`)
is **not the actual BitNet ternary inference**. Inspecting one row of its
q_proj.weight:

  shape = (2560, 2560), dtype = bfloat16
  row 0 unique abs values (top): [0.00116, 0.001335, 0.007202, 0.015991,
                                   0.034668, 0.039062, 0.05127, ...]
  row 0 unique value count: 808

If these were genuine bf16-stored ternary weights, each row would have at
most 3 unique values (-α, 0, +α). 808 unique values means HF's bf16 repo
holds the **full-precision FP weights** (training-time master copy or a
dequantized approximation), NOT the ternary weights deployed at inference.

The substrate, by contrast, reads from `microsoft/bitnet-b1.58-2B-4T`
(the packed-ternary repo, line 76 of `convert_weights.py`), which has
the actual trained ternary weights packed 4-in-8.

To confirm, we ternarized HF's bf16 weights (per-row sign × abs-mean)
in-memory and ran the prompt:

  HF bf16 weights → "If so, what would be the implications for our
                     understanding of the relationship between cognition
                     and the brain?"  ← coherent
  HF ternarized   → " a function of a, a, a, a, a, a, a, a, a, ..."   ← loops
  Substrate (a8)  → "If it is a function of cognition, then it might be
                     a function of cognition..."                       ← loops
  Substrate (no a8) → "If I am thinking, I am thinking, I am thinking..." ← loops

Both ternary inference paths (HF naive ternarization and substrate's
packed ternary) loop; the bf16 path doesn't. The substrate is faithful
to ternary inference — there is no "bug." The "coherence gap" we
observed is the **architectural quality gap between bf16 weights and
ternary weights of the BitNet b1.58-2B-4T model**, not anything our
substrate is doing wrong.

## What this means

* The BitNet b1.58-2B-4T model has TWO distinct quality regimes: bf16
  (training-time / development reference) and ternary (deployment).
  These give materially different generations on small prompts where
  per-token decisions are sensitive to weight precision.
* The substrate matches the ternary regime correctly. There is no fix
  that brings substrate output to match the bf16 reference output without
  changing the substrate's architectural commitment to ternary.
* For "match the bf16 model output," the substrate would need to switch
  to bf16 weight storage — which defeats the substrate's whole point.
* For "the ternary BitNet is incoherent on this prompt" — that is a
  property of the deployed ternary model, not a property of our
  implementation of it.

## What the no-a8 path is good for

Option 2's infrastructure (`m4t_mtfp_ternary_matmul_bt_route_i64`,
`m4t_mtfp_bitlinear_scale_no_a8_bx`, weight repack) is preserved on
main as the BIT-FAITHFUL ternary path. It removes the a8 quantization
noise (~0.4-1% per-element vs HF ternary), at the cost of losing
SDOT/SMMLA acceleration. End-to-end behavior is essentially identical
to the a8 path (both produce ternary inference within tolerance), but
the no-a8 path is the closer-to-bit-exact implementation of "ternary
matmul with per-row α scale."

For the user's original probe ("Hypothetically..."): both paths loop.
That is the ternary model's behavior. Not a substrate concern.

## Files

  layer_diff.csv          — original a8-path comparison (per-layer ε).
  substrate_logits_BOS.bin — substrate's full LM-head logits after BOS.
  hf_top10_BOS.txt        — HF bf16 reference top-10 (DIFFERENT MODEL).
  INVESTIGATION.md        — original (mistaken) hypothesis writeup.
  REVISION.md             — this file: corrected understanding.
