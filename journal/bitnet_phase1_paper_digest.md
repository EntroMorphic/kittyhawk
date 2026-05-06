---
cycle: bitnet_phase1
phase: ADDENDUM (paper digest)
date: 2026-05-06
scope: digest of BitNet b1.58 2B4T technical report (arxiv 2504.12285); identifies what the paper adds beyond the HF model card + config.json; notes implications for SYNTHESIZE and any pre-EXECUTE adjustments.
companions: bitnet_phase1_{raw,nodes,reflect,synthesize}.md
source: https://arxiv.org/abs/2504.12285 (HTML rendering at /html/2504.12285)
---

# Paper digest — bitnet_phase1

The HF model card + config.json gave us 80% of what SYNTHESIZE needed. The paper closes most of the rest. Five new findings, two refinements, three open questions still unresolved.

## New findings

### F1 — BitLinear formulas are exact

**Weight quantization (absmean):**
```
α = (1/n) Σᵢ |wᵢ|              # per-tensor mean of absolute values
w_q = sign(w) · 𝟙{|w| > α}     # ternarize: {-1, 0, +1} via threshold
```

The α threshold is **per-tensor**, not per-row. Each weight tensor has one scalar α computed over all its elements during training. (Inference doesn't recompute α — weights are already ternarized in the released model.)

**Activation quantization (absmax, per-token):**
```
s = max(|a|) / 127             # per-token scale, FP
a_q = round(a · 127 / max(|a|)) # int8 in [-127, 127]
```

These are unambiguous now. The substrate's existing `audit/tristate_l4_strong.c::ternarize_absmean` matches the weight-quantization shape; we'd promote it to libm4t for consumer-side use during weight-conversion. The activation-quantization is the new `m4t_a8_*` family that work-unit 5 introduces.

### F2 — FFN is ungated ReLU² (verified by parameter arithmetic)

The paper's notation `FFN(x) = (ReLU(Wx))² · V` is ambiguous, but the param count settles it. With 2B total parameters, the budget breaks down approximately:

| Component | Params |
|---|---|
| Embedding (128256 × 2560) | 328 M |
| LM head (likely tied to embedding) | 0 (or 328 M if untied) |
| Attention (30 layers × ~16.4 M/layer) | 491 M |
| Norms + bias | negligible |
| **FFN remainder** | **~1.18 B** |

Per-layer FFN: 1.18 B / 30 ≈ 39.4 M.
- Ungated (2 matrices: up 2560×6912, down 6912×2560) = 35.4 M ≈ 39 M ✓
- Gated (3 matrices: gate + up + down) = 53.1 M ✗

**Ungated**, with high confidence. The formula is:
```
FFN(x) = ReLU²(W_up · x) · W_down
       = max(0, W_up · x)² · W_down
```

Two matmuls per FFN call, not three. ReLU² is `(max(0,x))²` — no LUT needed; computed natively in MTFP19.

### F3 — HF stores ternary weights at 4-in-8 (2 bits/cell)

> *"Four ternary weight values packed into one INT8 for HBM storage."*

The released BitNet weights are **2 bits/cell**, not 1.6. The substrate can use either:
- **Match HF's storage** (4-in-8) via `m4t_pack_trits_1d` — minimal conversion at load time.
- **Use the substrate's denser 5-in-8** (`m4t_pack_trits_5in8_1d`) — 1.25× density improvement vs the reference, for free, as a side benefit of running on m4t.

**Recommendation: use 5-in-8.** The strong-claim story already established that 5-in-8 is the substrate's preferred packing. Loading HF's 4-in-8 and re-packing to 5-in-8 is a one-time conversion cost at model-load. Storage savings: ~80 MB (HF: 400 MB at 4-in-8; m4t: 320 MB at 5-in-8). Modest but cleanly demonstrates the density-ceiling argument on a real model.

### F4 — Master weight in BF16 during training (Phase 2 foreshadowing)

> *"Master weight of BitNet b1.58 2B4T, used for training only [stored in BF16]."*

Confirms BitNet uses **latent FP weights + quantize-during-forward**. This is the "BitNet-style mixed-precision training" fork I flagged in REFLECT for Phase 2. The paper validates that this is mainstream practice, not a hack — Microsoft trained the released model this way.

For Phase 2, this means we have two real options:
- **Phase 2 / Path BitNet-style**: latent FP weights (BF16 or MTFP19), quantize forward to ternary, FP gradients. Minimum-novel-research-required.
- **Phase 2 / Path base-3-native**: research cycle. Latent base-3 weights of higher precision (MTFP39?), gradients also in base-3, no FP anywhere in the training pipeline. Genuinely novel, genuinely risky.

Phase 1 doesn't decide between these — but it should not foreclose either. We should NOT design Phase 1 weight-loading in a way that assumes "weights are eternally ternary post-training."

### F5 — Attention scaling is a pre-computed constant

Standard scaled dot-product attention: `softmax(Q·Kᵀ / √d_k) · V`. With `d_k = head_dim = 128`, the scaling factor is `1/√128 ≈ 0.0884` — pre-computable at substrate-init time as an MTFP19 constant. **No runtime sqrt needed for attention.**

This means **the only runtime sqrt is RMSNorm** (subln's `rsqrt(mean(x²) + ε)`). Confirmed work-unit 2 is the right scope for the rsqrt primitive.

## Refinements to SYNTHESIZE

### R1 — FFN simplification in work-unit 1

Thin-B harness no longer needs to handle the gated-FFN ambiguity. ReLU² ungated, two matmuls. Simpler scaffolding.

### R2 — A8 substrate primitive (work-unit 5) shape locked

Quantize: `s = max(|a|) / 127; a_int8 = round(a · 127 / s)` per token.
Dequantize: `a_mtfp = a_int8 · s` per token.

Both are O(n) elementwise with a per-token reduction (the `max`). NEON: `vmaxvq_s32` + scalar division for `s`, then `vmlaq_s32` family for the multiply-and-round. Per-token reductions don't NEON-ize as cleanly as per-vector ops (cross-lane work), but the per-token cost is `O(hidden/16)` NEON ops + 1 scalar — manageable.

## Refinements that are real changes to plan

### R3 — Storage decision: use 5-in-8

Phase 1 work-unit 1 should convert HF's 4-in-8 ternary weights to substrate's 5-in-8 packing at load time. Adds a small one-time conversion cost; gains the density story.

This is a substantive change: SYNTHESIZE-as-written said "convert HF's storage to substrate-native packed" without specifying which packing. Now we specify 5-in-8.

### R4 — A8 scale-storage decision

The activation scale `s` is FP (per the paper's formula). Substrate has no FP. Options:

(a) Store the scale as MTFP19: `s_mtfp19 ≈ max(|a|) / 127`, but this loses precision (MTFP19 is base-3, not base-10/2; rounding differs).
(b) Store the scale as fixed-point int with separate exponent (effectively MTFP19 with explicit exp).
(c) Store the scale as int32 directly (the absmax itself, before dividing by 127), and apply `127` divide at dequantize time.

Option (c) is the cleanest — defers the FP-flavored division to dequantize, where it's just a constant multiply. Sounds counterintuitive but matches the substrate's MTFP design (mantissa + exponent, no FP).

**Decision (locking)**: store the per-token absmax as `m4t_mtfp_t` (int32). Apply the 127 division at dequantize time as `a_int8 · scale_mtfp / 127`. The `/ 127` is a fixed magic-multiply (we'd add `M4T_DIV_127_M` and `M4T_DIV_127_N` to the substrate's existing magic-multiply machinery, or just use shift3-style for it).

Pre-EXECUTE check: confirm by characterizing the precision loss vs HF reference in thin-B harness.

## Open questions the paper didn't resolve

### O1 — subln exact placement

The paper cites Wang et al. 2022 ("Sub-Layer Normalization") but doesn't detail where in the residual stream the norms sit. Three possibilities:
- **Pre-norm** (norm before attention/FFN, residual unchanged): standard modern transformer pattern.
- **Post-norm** (residual then norm): less common, training-stability concern.
- **subln-paper-style** (norm inside the residual branch + scale): the paper's namesake.

Resolution: inspect HF's reference implementation directly during work-unit 1. The model-card-derived assumption was "subln" but I've now seen the paper say "subln" too — neither pinned the placement. The reference code will.

### O2 — Are intermediate projection outputs re-quantized?

Each transformer block has multiple BitLinear layers (Q, K, V, O, gate-up, down). Are activations quantized to int8 between every BitLinear? Or only at the block-input boundary?

The paper's "W1.58A8" notation suggests "all linear layers are W1.58 with A8 inputs," meaning every BitLinear sees int8 inputs. So yes, **per-BitLinear input quantization**. Implies many quantize/dequantize boundaries inside a single block, which work-unit 1 must implement.

### O3 — Does HF's reference implementation actually do all this faithfully?

The HF model card warned: *"Do NOT expect efficiency gains via transformers library. The standard transformers library lacks specialized optimized kernels."* The reference is a slow, unoptimized path. It likely runs computations in BF16 throughout AND applies the quantization rule on top — i.e., it simulates BitLinear via BF16 with quantize-then-dequantize at each layer. Our per-layer comparison vs HF must match this simulation, which means **we're comparing m4t (real ternary inference in MTFP19/A8) vs HF (simulated ternary inference in BF16)**.

This shifts the fidelity gate's interpretation: ε measures "how different is real-ternary from simulated-ternary" not "how different is m4t from optimal-bitnet." The substrate may match HF closely (because both apply the same quantization rule), or may diverge (because the data path between quantizations has different precision).

**Implication**: bitnet.cpp is a closer comparator for "real ternary inference" than HF's transformers library. We should set up comparison against bitnet.cpp's output, not just HF's, in work-unit 6 — or at minimum acknowledge that HF-comparison validates "we apply the quantization rule the same way" and not "we run optimal BitNet."

## Updated pre-EXECUTE red-team checklist

- ~~Have I read the paper?~~ Yes (this digest).
- Is bitnet.cpp's output a more faithful comparator than HF transformers? **Yes for fidelity-of-real-inference; no for fidelity-of-quantization-rule.** Use both: HF for "quantization applied correctly" gate, bitnet.cpp for "ternary path is correct" gate.
- subln placement question (O1): inspect HF reference code (or bitnet.cpp's source) before work-unit 1 begins.
- Per-BitLinear quantization confirmed (O2). Work-unit 1 needs to budget for many quantize/dequantize calls per block.
- 5-in-8 vs 4-in-8 storage: locked at 5-in-8 for the density-ceiling story (R3).

## Status

Cycle artifacts: RAW + NODES + REFLECT + SYNTHESIZE + (this) DIGEST. SYNTHESIZE's plan stands with the R1/R2/R3/R4 refinements integrated. O1/O2/O3 are pre-EXECUTE gates that work-unit 1 will resolve by inspection. The three user decisions (U1/U2/U3 in SYNTHESIZE) remain open.

Ready to execute when user signs off.
