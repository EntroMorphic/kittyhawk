# gesh/bitnet — BitNet b1.58-2B-4T inference on the m4t substrate

Per `journal/bitnet_phase1_*.md`. This is **Phase 1** of the four-phase ternary-ML arc (inference → fine-tune → train-from-scratch → productize). The harness validates that the substrate's kernel surface composes into a real ternary LLM running in its native numeric system.

## What this directory will contain (when work-unit 1 closes)

- **C harness** (`bitnet_harness.c`) — main driver that loads converted BitNet weights, runs forward pass through one or more transformer blocks, dumps per-layer activations to disk for comparison.
- **Config + data structures** (`bitnet_config.h`, `bitnet_block.{h,c}`) — model dimensions, per-block scratch layouts, weight pointers.
- **Weight loader** (`bitnet_weights.{h,c}`) — reads the substrate-native binary blob produced by `scripts/convert_weights.py` (Python intermediary).
- **Stubs for missing primitives** (`bitnet_stubs.{h,c}`) — temporary scalar implementations of rsqrt, RoPE, softmax, A8 quantize/dequantize. Replaced by NEON primitives in work-units 2-5.
- **Python scripts** (`scripts/`) — weight conversion (HF safetensors → substrate format), HF reference forward pass + activation dump.

## Architecture (verified per `bitnet_phase1_o1_findings.md`)

```
30 × BitNetDecoderLayer:
  ┌─────────────────────────────────────────────────────────┐
  │ residual = x                                            │
  │ x = input_layernorm(x)               # NORM 1 (pre-attn)│
  │ x = attention(x)                                        │
  │   ├─ Q = x @ W_q                                        │
  │   ├─ K = x @ W_k        (GQA: 5 heads, 4× repeat)       │
  │   ├─ V = x @ W_v        (GQA: 5 heads, 4× repeat)       │
  │   ├─ Q, K = rope(Q, K)                                  │
  │   ├─ scores = Q @ Kᵀ * (1/√128)                         │
  │   ├─ scores = softmax(scores)        # full-precision   │
  │   ├─ y = scores @ V                                     │
  │   ├─ y = attn_sub_norm(y)            # NORM 2 (sub-LN)  │
  │   └─ y = y @ W_o                                        │
  │ x = residual + y                                        │
  │                                                         │
  │ residual = x                                            │
  │ x = post_attention_layernorm(x)      # NORM 3 (pre-FFN) │
  │ x = ffn(x)                                              │
  │   ├─ gate = x @ W_gate                                  │
  │   ├─ up   = x @ W_up                                    │
  │   ├─ y = relu²(gate) * up            # element-wise     │
  │   ├─ y = ffn_sub_norm(y)             # NORM 4 (sub-LN)  │
  │   └─ y = y @ W_down                                     │
  │ x = residual + y                                        │
  └─────────────────────────────────────────────────────────┘
```

**Key dimensions (from `config.json`):**

| Parameter | Value |
|---|---|
| hidden_size | 2560 |
| intermediate_size (FFN) | 6912 |
| num_hidden_layers | 30 |
| num_attention_heads | 20 |
| num_key_value_heads | 5 (GQA 4:1) |
| head_dim | 128 |
| max_position_embeddings | 4096 |
| vocab_size | 128256 |
| rms_norm_eps | 1e-5 |
| rope_theta | 500000.0 |
| hidden_act | relu² |

**All BitLinear layers**: ternary weights via absmean rule; A8 (per-token absmax) activation inputs. **No bias** anywhere.

## Substrate primitives this harness needs

**Already in libm4t:**
- Ternary @ ternary matmul (`m4t_ternary_5in8_matmul_*` for the four `W_q/W_k/W_v/W_o`, three `W_gate/W_up/W_down` per layer)
- Cross-exp accumulator (for the residual additions across layers)
- Pack/unpack 5-in-8 (for weight conversion at load time; activations are A8 not packed)
- Width conversions (MTFP19 ↔ MTFP4 may be useful for some intermediate buffers; TBD)
- shift3 (for any positional scaling that surfaces; TBD)

**NEW — added in work-units 2-5:**
- `m4t_mtfp_rsqrt` — rsqrt for RMSNorm (work-unit 2)
- `m4t_rope_apply` — RoPE rotation (work-unit 3)
- `m4t_softmax` — LUT-backed softmax with subtract-max for stability (work-unit 4)
- A8 family (`m4t_a8_quantize`, `m4t_a8_dequantize`) — per-token absmax quantization (work-unit 5)

**Stubbed in work-unit 1**: scalar reference implementations of all four "new" primitives, expressed as `static` functions in `bitnet_stubs.c`. The harness calls these via the same signatures the eventual NEON paths will use; substituting the substrate primitives later requires no harness change.

## Work-unit 1 gate

Layer 0 forward pass produces *some* output. Per-layer comparison driver runs (even if the values diverge from HF). The substrate-gap list is concrete: we know exactly which primitives the harness called, with what shapes, and the empirical input ranges (so work-units 2-5 can characterize their LUTs / Newton-Raphson bounds without guessing).

**Out of scope for work-unit 1:**
- Bit-precision agreement with HF (may differ; that's what work-unit 6 measures)
- All 30 layers (work-unit 6)
- KV cache (work-unit 7)
- Generation loop (work-unit 8)
- Performance comparison with bitnet.cpp (Phase 1's nice-to-have, not required for the gate)

## Build

This subdirectory is **not** in ctest. It's a consumer-side harness, comparable to the audit binaries — built by the main CMake but invoked manually.

```bash
cmake --build build --target bitnet_harness
build/gesh/bitnet/bitnet_harness <weights.bin> <prompt-token-ids>
```

## Cross-references

- LMM cycle: `journal/bitnet_phase1_{raw,nodes,reflect,synthesize,paper_digest,o1_findings}.md`
- Source of truth for architecture: `huggingface.co/microsoft/bitnet-b1.58-2B-4T` model card + `config.json` + `transformers/models/bitnet/modeling_bitnet.py`
- Original paper: arxiv 2504.12285
