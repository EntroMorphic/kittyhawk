# gesh/bitnet — BitNet b1.58-2B-4T inference on the m4t substrate

A C harness that runs the official Microsoft BitNet b1.58-2B-4T model end-to-end
in the substrate's native MTFP19 / packed-ternary numeric system. No floating
point in the runtime kernel path — `bf16` only at weight-conversion time
(Python helper, runs once per checkpoint).

This is **Phase 1** of the four-phase ternary-ML arc (inference → fine-tune →
train-from-scratch → productize). The harness validates that the substrate's
kernel surface composes into a real ternary LLM.

## Status (2026-05-10)

- **Forward pass: 30 layers × any prompt length, KV cache, greedy generation.**
- **Quality characterized on a 24-prompt battery** spanning factual /
  definitional / narrative / math / code / structured-output / long-context /
  edge categories. Strict pass rate **~92%** (~22/24) post-tuning, zero hard
  failures. See
  [`journal/inference_battery_v2_2026-05-09.md`](../../journal/inference_battery_v2_2026-05-09.md),
  [`journal/hp_sweep_2026-05-10.md`](../../journal/hp_sweep_2026-05-10.md) (GATE_ACT_BX=1),
  and [`journal/math_div_atomics_2026-05-10.md`](../../journal/math_div_atomics_2026-05-10.md)
  (atomics + score_shift fudge=2 closing TD-20).
- **Part-B evidence on N4 sparse attention (2026-05-10).** Substrate-routed
  top-k attention via `m4t_route_threshold_extract` + `m4t_route_distance_batch`
  passed all three pre-committed EVIDENCE gates on the 24-prompt × 4-arm × 6-k
  battery (456 runs). Per
  [`journal/cycle2_full_battery_findings.md`](../../journal/cycle2_full_battery_findings.md).
  See "Cycle 2 sparse-attention modes" below for how to invoke.
- **Substrate-vs-HF investigation closed.** A latent silent-saturation bug in
  `m4t_mtfp_rmsnorm_bx` at `gamma_bx > target_bx` (BitNet's typical regime)
  collapsed `post_attention_layernorm` outputs by 6.5× and produced
  degenerate loops; fixed in commit `4d4c917`. See
  [`journal/substrate_vs_hf_2026-05-09/RESOLVED.md`](../../journal/substrate_vs_hf_2026-05-09/RESOLVED.md).
- **Two BitLinear paths shipped:**
  - Default (work-units 1-7): A8 per-token activation quantization + 5-in-8
    SDOT matmul.
  - **No-A8 / bit-faithful** (work-unit 8 follow-on): bypasses A8 quantization
    and runs MTFP19 × packed-ternary matmul (4-in-8 repacked at load time;
    +496 MB residency). Used during the substrate-vs-HF investigation to
    isolate quantization noise.

## Files

- `bitnet_harness.c` — main driver: load weights, forward pass over N layers
  × P positions, optional greedy generation. CLI flags: `--prompt-tokens`,
  `--positions`, `--gen`, `--dump`, `--token`, `--layers`. Two debug-dump
  env hooks: `DEBUG_DUMP_OPROJ`, `DEBUG_DUMP_PREPN` (off by default).
- `bitnet_config.h` — model dimensions (hidden=2560, intermediate=6912, 30
  layers, GQA 4:1, etc.).
- `bitnet_weights.{h,c}` — mmaps the substrate-format weights blob produced
  by `scripts/convert_weights.py`. At load time, repacks ternary weights
  from 5-in-8 (compact, used by the SDOT BitLinear) to 4-in-8 (used by the
  MTFP19 × packed-ternary kernel for the no-A8 path).
- `bitnet_kv_cache.{h,c}` — per-layer K/V cache for incremental decode.
- `bitnet_stubs.{h,c}` — historical (work-unit 1) scalar stubs for primitives
  that were later promoted to libm4t (rsqrt, RoPE, softmax, A8 quantize). Now
  contains only what the harness still calls directly; production paths use
  libm4t kernels.
- `scripts/` — Python helpers (run once per checkpoint):
  - `convert_weights.py` — HF `safetensors` → substrate blob (MTFP19 norms,
    embeddings, α scales; 5-in-8 ternary weights). Per-tensor `block_exp`
    chosen to maximize precision without overflowing MTFP19_MAX.
  - `dump_reference.py` — runs HF's BitNet via `transformers`, captures
    per-layer activations for the substrate-vs-HF comparison gate.
  - `inspect_blob.py`, `compare_activations.py`, etc. — diagnostic helpers.

## Architecture (per `bitnet_phase1_o1_findings.md`)

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

**All BitLinear layers**: ternary weights via absmean rule; A8 (per-token
absmax) activation inputs in the default path. **No bias** anywhere.

## Substrate primitives in use

All NEON-only in production (`feedback_function_over_speed_no_scalar`).

- **Matmul:**
  - `m4t_ternary_5in8_matmul_xpacked_bt` — A8 BitLinear path (default).
  - `m4t_mtfp_ternary_matmul_bt_route_i64` — no-A8 bit-faithful path
    (MTFP19 × packed-ternary, int64 output before α scale).
- **RMSNorm:** `m4t_mtfp_rmsnorm_bx` (bx-aware variant; pre-rescales γ when
  `gamma_bx > target_bx`, see header docstring).
- **BitLinear scale:** `m4t_mtfp_bitlinear_scale_bx` (default), or
  `m4t_mtfp_bitlinear_scale_no_a8_bx` (no-A8 path; combined-divisor magic
  to avoid 7%-CPU bottleneck).
- **Activation:** `m4t_mtfp_relu2_inplace_bx`, `m4t_mtfp_elementwise_mul_bx`
  (both `_bx` variants — divide-before-clamp at int64; non-`_bx` legacy
  forms are not on the harness path).
- **Cross-exp / vec primitives:** `m4t_mtfp_vec_add_inplace`,
  `m4t_mtfp_rescale_bx`.
- **Attention:** `m4t_mtfp_rope_apply` (RoPE rotation; LUT-backed cos/sin
  at Q29 fixed-point), `m4t_mtfp_softmax` (V14.G v2 NEON-gather LUT),
  `m4t_mtfp_vec_dot_i64` (Q·K dot accumulator), `m4t_mtfp_attn_v_combine`
  (weighted V output).

## Running the harness

```bash
cmake --build build --target bitnet_harness

# Greedy decode 30 tokens after a 15-token prompt:
./build/gesh/bitnet_harness data/bitnet_b158_2b4t.bin \
    --prompt-tokens 128000,3923,374,279,6864,315,9822,30 \
    --gen 30

# BOS-only single forward pass with per-layer activation dump:
./build/gesh/bitnet_harness data/bitnet_b158_2b4t.bin \
    --prompt-tokens 128000 --dump /tmp/sub_dump
```

The `--dump` output uses the `ACTV2` format documented in
`scripts/compare_activations.py`. The harness writes to **stderr**, not stdout.

### Cycle 2 sparse-attention modes (research / Part-B experiment)

The harness supports runtime-selectable attention modes for the N4
Part-B experiment (per `journal/cycle2_design.md` and
`journal/cycle2_full_battery_findings.md`):

```bash
# Default (env unset): bit-exact dense (production path).
./build/gesh/bitnet_harness ...

# Substrate-routed top-k: pick K positions per Q via packed-trit
# signature distance (m4t_route_threshold_extract + distance_batch).
BITNET_ATTN_MODE=routed BITNET_ATTN_K=4 ./build/gesh/bitnet_harness ...

# Random top-k baseline (xorshift32 + Fisher-Yates):
BITNET_ATTN_MODE=random BITNET_ATTN_K=4 ./build/gesh/bitnet_harness ...

# Oracle top-k (compute dense scores first, pick top-k by |score|):
BITNET_ATTN_MODE=oracle BITNET_ATTN_K=4 ./build/gesh/bitnet_harness ...
```

When `BITNET_ATTN_MODE` is unset or = `dense`, the production path runs
unchanged (bit-exact). The sparse arms use the NEON
`m4t_mtfp_attn_v_combine` and gather V via memcpy; production-eligible
per the "no scalar in production" foundational rule (2026-05-12).
Off by default; opt-in via env var.

This subdirectory is **not** in ctest. It's a consumer-side harness, comparable
to the audit binaries — built by the main CMake but invoked manually.

## Cross-references

- LMM cycle: `journal/bitnet_phase1_*` (raw → nodes → reflect → synthesize → closeout).
- RMSNorm fix: `journal/substrate_vs_hf_2026-05-09/RESOLVED.md`.
- End-to-end battery: `journal/post_rmsnorm_fix_battery_2026-05-09/SUMMARY.md`.
- Source of truth for architecture: `huggingface.co/microsoft/bitnet-b1.58-2B-4T`
  model card + `config.json` + `transformers/models/bitnet/modeling_bitnet.py`.
- Original paper: arxiv 2504.12285.
