---
cycle: bitnet_phase1
phase: ADDENDUM (O1 inspection findings)
date: 2026-05-06
scope: pre-EXECUTE inspection of HF transformers reference for BitNet (modeling_bitnet.py). Resolves O1 (subln placement) from the paper digest. Surfaces a correction to F2 (FFN gating) and adds two findings the digest missed.
companions: bitnet_phase1_{raw,nodes,reflect,synthesize,paper_digest}.md
source: github.com/huggingface/transformers/blob/main/src/transformers/models/bitnet/modeling_bitnet.py
---

# O1 findings — bitnet_phase1

## Correction: F2 was wrong. FFN is GATED.

The paper digest argued (F2) that FFN is ungated based on parameter arithmetic. **The HF reference shows it's gated.** Direct quote from `BitNetMLP.forward()`:

```python
down_proj = self.down_proj(
    self.ffn_sub_norm(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
)
```

Three matrices per FFN (gate_proj, up_proj, down_proj) plus an inner subln plus a relu² activation. Per-FFN params: `3 × 2560 × 6912 = 53.1 M × 30 layers = 1.59 B`.

**Where my arithmetic went wrong**: I subtracted attention as 491 M and embedding as 328 M from a 2 B target, leaving 1.18 B for FFN — and concluded ungated. The actual model is closer to **2.4 B parameters with tied embedding/LM head** (491 attn + 1593 FFN + 328 embed). The "2B" label is approximate. Gated arithmetic checks out: 1.59 B FFN fits.

**Lesson**: parameter arithmetic from a rounded headline number is a weaker signal than reading the source. I deferred reading the source because it felt expensive; the cost was paid here.

## subln placement (the actual O1 question): 4 norm sites per block

The paper said "subln" without pinning placement. The HF reference shows **4 RMSNorm calls per transformer block**, in two distinct patterns:

```
# BitNetDecoderLayer.forward()
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)        # NORM 1: pre-attn (standard pre-norm)
hidden_states, _ = self.self_attn(...)                       #   ↳ contains NORM 2 inside
hidden_states = residual + hidden_states

residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states) # NORM 3: pre-FFN (standard pre-norm)
hidden_states = self.mlp(hidden_states)                      #   ↳ contains NORM 4 inside
hidden_states = residual + hidden_states
```

The two "inside" norms are the actual subln-pattern:

- **`attn_sub_norm`** sits *between attention output and o_proj*. The HF code comment is explicit: `# diff with Llama`. Llama applies `o_proj(attn_output)`; BitNet applies `o_proj(attn_sub_norm(attn_output))`.
- **`ffn_sub_norm`** sits *between the gated activation and down_proj*: `down_proj(ffn_sub_norm(relu²(gate(x)) * up(x)))`.

So the structure is: **standard pre-norm at the block boundaries** (norms 1 and 3) **+ subln at the sub-block internal boundaries** (norms 2 and 4).

This is more norm work than I'd planned for. **120 RMSNorm calls per forward pass** (4 per block × 30 blocks). The rsqrt primitive's perf matters more than I thought.

## RMSNorm has a learnable scale parameter (γ)

Direct from `BitNetRMSNorm.forward()`:

```python
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)
```

Formula: **`y = γ · x · rsqrt(mean(x²) + ε)`** where `γ` is a learnable per-channel scale (initialized to ones; loaded from checkpoint). The substrate's RMSNorm primitive must take γ as an input, not assume it's identity.

`ε = 1e-5` per config (HF code default `1e-6` is overridden).

## Softmax runs at FP32 precision in HF reference

Direct: `attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)`.

HF upcasts to fp32 for softmax even though the rest runs in bf16. Reason: numerical stability under attention's large dynamic range. **Substrate analog**: ensure our softmax runs at full MTFP19 precision (int32 mantissa), not at any narrower cell width.

## GQA repetition is naive: `repeat_kv`

Direct: `key_states = repeat_kv(key, module.num_key_value_groups)`. The KV heads (5) are repeated 4× to match Q heads (20). HF implements this as a tensor expand; we'd do the same — no compute, just stride/view manipulation.

## RoPE applies before attention matmul

```python
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
```

RoPE applied to (Q, K) AFTER projection but BEFORE the QKᵀ matmul. Standard placement. The `cos`, `sin` come from a precomputed buffer indexed by position. Pure rotation in 2D pairs.

## What this means for the SYNTHESIZE plan

**Refinements:**

- **R5 — FFN gated.** Three matrices per FFN, not two. Thin-B harness needs gate_proj + up_proj + down_proj + element-wise multiply + ffn_sub_norm + relu² activation.
- **R6 — RMSNorm primitive must accept γ.** Signature: `m4t_rmsnorm(dst, src, gamma, eps_in_mtfp19, n)` (or equivalent). The γ is an int32 vector of length `hidden_size`.
- **R7 — 4 norm sites per block** (not 2). The substrate's rsqrt primitive's perf affects 120-call-per-forward latency. Plan for it.
- **R8 — Softmax must use full MTFP19 precision.** No narrower-width tricks for that primitive. Document this in work-unit 4.

**No major surprises beyond F2:**

- Pre-norm at block boundaries is standard.
- subln-internal norms are conceptually new but mechanically the same primitive (just called more often).
- All 4 norm sites use the same BitNetRMSNorm class with potentially different γ tensors.

## What's still unresolved (and will resolve only by EXECUTE)

- **O2 — Per-BitLinear input quantization.** HF's `modeling_bitnet.py` uses standard `nn.Linear` — the BitLinear quantize/dequantize is pulled in via auto-generation from `modular_bitnet.py` and lives in HF's quantization machinery. To see the actual quantization-aware logic, I'd need to inspect `modular_bitnet.py` or HF's `quanto`/`bitnet` quantizer. Deferring this to work-unit 1, where I'll have to confront it anyway when loading a quantized checkpoint.
- **O3 — bitnet.cpp comparison.** Microsoft's C++ reference is a separate inspection. The HF reference is "simulated ternary in BF16." bitnet.cpp is "real ternary inference." Both useful. Will inspect during work-unit 1 if needed for comparison-target choice.

## Updated pre-EXECUTE checklist status

- [x] U1 default locked (Path α)
- [x] U2 deferred (Phase 4 not in Phase 1 scope)
- [x] U3 resolved (work-units, not calendar)
- [x] O1 resolved (this doc)
- [ ] O2 — defer to work-unit 1 (depends on inspecting quantizer)
- [ ] O3 — inspect bitnet.cpp opportunistically during work-unit 1

**Ready to begin work-unit 1.**
