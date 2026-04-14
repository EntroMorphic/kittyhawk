# trix-z Architecture

## Overview

trix-z is a transformer architecture where every matrix multiplication uses ternary weights {-1, 0, +1} and activations flow through Multi-Trit Fixed Point (MTFP) integer arithmetic. The computational core contains zero floating-point multiplications.

## Numerical System: MTFP

All values inside the model are stored as `mtfp_t` (int32) in balanced ternary fixed-point:

```
real_value = mtfp_value / 59049    (59049 = 3^10)
```

Resolution: ~1.69e-5 per step. Range: ±18,183 (practical int32 limit).

**Key property:** multiplying an MTFP value by a ternary weight {-1, 0, +1} requires only add, subtract, or skip — zero float multiplies.

## Data Flow

```
Input pixels (float) → mtfp_from_float → MTFP values
  → Ternary projection W1 (add/sub only) → fan_in_normalize → MTFP bias → GELU table
  → Ternary projection W2 (add/sub only) → MTFP bias
  → MTFP residual stream
    → [N transformer blocks, all MTFP internally]
  → mtfp_to_float → float classifier → cross-entropy loss
```

## Transformer Block

Each block contains:

### 1. Routed QKV Projection (optional — `proj_tiles > 0`)
- LayerNorm (MTFP: integer mean/var, integer sqrt for rstd)
- Weight-derived ternary signatures: `sig_t = sign(colsum(W_Q_t) - mean)`
- Routing: top-K tiles by |dot(x, sig_t)|, sign preserved → {-1, 0, +1}
- Per-tile: `tile_out = x @ W_t^T` via ternary add/sub + fan_in_normalize + bias
- Signed accumulation: `qkv = output_scale × sum(route_t × tile_out_t)`

### 2. Multi-Head Attention
- Split QKV into per-head Q, K, V
- Scores: `Q @ K^T` via `mtfp_matmul_bt` (int64 accumulate + rescale)
- Scale: `scores × inv_sqrt(d_head)` via `mtfp_vec_scale`
- Softmax: pre-computed exp lookup table (range ±6.0), integer normalization
- Output: `scores @ V` via `mtfp_matmul`
- Merge heads

### 3. Routed W_O Projection (optional)
- Same structure as QKV but no LayerNorm, smaller output (D → D)

### 4. Attention Residual
- `mid_res = x + attn_output` in MTFP (integer add)

### 5. Ternary Routed FFN
- LayerNorm (MTFP integer)
- Weight-derived signatures from W1 column sums
- Routing: k-of-T ternary {-1, 0, +1}
- Per-tile: `z = x @ W1_tern^T` (add/sub) → fan_in_normalize → bias → GELU (table lookup)
  → `out = h @ W2_tern^T` (add/sub) → bias
- Signed accumulation: `ffn_out = output_scale × sum(route_t × tile_out_t)`

### 6. FFN Residual
- `output = mid_res + ffn_out` in MTFP (integer add)

## Fan-In Normalization

Ternary matmul accumulates K add/subtract operations. The output magnitude scales as O(sqrt(K)). Without normalization, deep or wide layers produce values that overflow MTFP range or saturate nonlinearities.

`mtfp_fan_in_normalize(x, n, fan_in)` divides each element by `isqrt(fan_in)`.

**Apply before nonlinearities** (GELU, softmax input) to keep values in the lookup table range (±6.0).

**Do NOT apply to:**
- Routing scores (relative ordering preserved regardless of magnitude)
- W2 tile outputs (output_scale handles magnitude)
- Routed projection outputs that feed into LayerNorm (LN normalizes internally)

**Exception:** Routed projection tiles that feed into attention DO need fan-in normalization to prevent Q@K^T overflow.

## Ternary Weight Management

### Shadow Weights
Float32 shadow weights are maintained for gradient updates. Each training step:
1. Forward: quantize float → ternary {-1,0,+1} via threshold (mean_abs × 0.5)
2. Forward: use ternary weights in MTFP matmul (add/sub)
3. Backward: STE (straight-through estimator) — gradients flow through as if ternary wasn't there
4. Update: AdamW on float shadow weights
5. Re-quantize: ternary weights and signatures updated from new shadow weights

### Weight-Derived Signatures
Routing signatures are derived from tile weights, not learned separately:
```
raw_t[d] = sum_h(W1_t[h, d])        # what this tile responds to
mean[d]  = mean_over_t(raw_t[d])     # average across tiles
sig_t[d] = sign(raw_t[d] - mean[d])  # differential: what makes this tile different
```

Updated after every AdamW step. Zero extra parameters.

## Lookup Tables

- **GELU:** 708,589 entries covering [-6.0, +6.0] in float (2.8 MB). Pre-computed once at startup. Zero arithmetic per element.
- **Softmax exp:** Same size and range. Used for the exp() operation in softmax.
- **Integer sqrt:** Newton-Raphson iteration, 8 steps from bit-count initial guess. Used in LayerNorm for rstd.

## Configuration

### `TrixNativeBlockConfig`
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| d_model | int | required | Embedding dimension |
| n_heads | int | required | Attention heads |
| d_head | int | d_model/n_heads | Per-head dimension |
| num_tiles | int | required | FFN expert tiles |
| tile_hidden | int | required | FFN hidden dim per tile |
| active_k | int | num_tiles | FFN active tiles (k-of-T) |
| proj_tiles | int | 0 | Routed QKV/W_O tiles (0 = dense) |
| proj_k | int | proj_tiles | Active routed projection tiles |
| qkv_scale_init | float | 0.5 | QKV output_scale initial value |
| wo_scale_init | float | 0.5 | W_O output_scale initial value |
| ffn_scale_init | float | 0.1 | FFN output_scale initial value |
| ln_eps | float | 1e-5 | LayerNorm epsilon |
| use_ternary_route | bool | true | Use ternary FFN (false = multitrit) |

### `TrixGPT2Config`
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| vocab_size | int | 32000 | Vocabulary size |
| d_model | int | 64 | Embedding dimension |
| n_heads | int | 4 | Attention heads |
| n_layers | int | 20 | Transformer blocks |
| seq_length | int | 256 | Max sequence length |
| num_tiles | int | 8 | FFN tiles |
| tile_hidden | int | 32 | FFN hidden dim |
| active_k | int | 8 | FFN active tiles |
| proj_tiles | int | 0 | Routed projection tiles |
| proj_k | int | 0 | Active routed projection tiles |

## File Map

| File | LOC | Purpose |
|------|-----|---------|
| `trix_mtfp.c/.h` | ~460 | MTFP arithmetic: conversions, ternary matmul, GELU/softmax tables, LayerNorm, integer sqrt |
| `trix_ternary_matmul.c/.h` | ~120 | Float × ternary matmul (used in FFN backward for STE) |
| `trix_ternary_route.c/.h` | ~580 | Ternary routed FFN: forward (MTFP + float), backward, AdamW, signature update |
| `trix_routed_proj.c/.h` | ~440 | Routed linear projection: QKV and W_O with ternary tiles |
| `trix_transformer.c/.h` | ~450 | Transformer block: MHA + FFN, float and MTFP forward paths |
| `trix_gpt2.c/.h` | ~430 | GPT-2 model: embeddings, blocks, LM head, data loading, LR schedule |
| `trix_atoms.c/.h` | ~310 | 29 atomic operations: matmul, LayerNorm, GELU, AdamW, NEON vectorized |

## Test Binaries

```bash
# Vision (MNIST / Fashion-MNIST / CIFAR-10)
./test_zero_dense_vision <idx|bin> <train_imgs> <train_lbls> <test_imgs> <test_lbls> \
    [layers] [D] [K] [lr] [epochs] [n_train] [n_test] [input_dim] \
    [T_proj] [T_ffn] [n_heads] [batch] [n_classes]

# GPT-2 Language Modeling
./test_gpt2_ternary <train.bin> <val.bin> [total_tokens] [d_model] [n_layers] [lr] [proj_tiles]
```
