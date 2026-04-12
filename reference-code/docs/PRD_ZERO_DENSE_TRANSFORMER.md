# PRD: Zero-Dense Glass-Box Transformer

**Date:** 8 April 2026
**Priority:** P0
**Owner:** Tripp Josserand-Austin
**Status:** Building

## 0. Principles (non-negotiable)

1. **Everything is ternary.** +1, -1, 0. Three states. Every routing decision, everywhere. No binary gates. No softmax. No fallback to dense.
2. **Routing is weight-derived.** Column sums, mean subtraction, sign. No learned routers. No auxiliary parameters. The routing lives in the expert weights.
3. **No balance loss.** Mean subtraction prevents collapse structurally. No gradient tricks.
4. **Anti-expert (-1) is half the architecture.** A tile or head that subtracts its output provides signal that skipping (0) cannot. Removing -1 would be removing subtraction from arithmetic.
5. **Dense is what we're eliminating.** Every dense matmul is an implicit routing decision nobody can read. We make it explicit.
6. **Embrace discovery.** Per-token head routing, anti-attention, routed LM heads — none published. That's the point.

## Red-Team Resolutions

| Attack | Resolution |
|--------|-----------|
| 1 (linear degeneracy) | Build linear first. 81 effective matrices is structured selection. Add nonlinearity only if empirically needed. |
| 2 (anti-attention) | **Full ternary {+1, -1, 0}.** No devolving to binary. Anti-attention is part of the architecture. |
| 3 (per-token head routing) | Per-sequence routing with pooled input for batched GEMM. Per-token is the goal — revisit once per-sequence works. |
| 4 (overfitting) | Incremental rollout. Measure each addition against FFN-only baseline. |
| 5 (factored LM head) | Keep Option C. Indirect interpretability is acceptable. |
| 6 (T=2 too small) | T=4 minimum for projections. |
| 7 (QKV signatures) | Derive from Q portion only (columns 0..D-1 of W_QKV_t). |
| 8 (timeline) | 10 days. |

---

## 1. Vision

Every matrix multiplication in the transformer goes through ternary routing. No dense matmuls. Every computation path is observable, editable, and hardware-accelerable via XOR+POPCNT. The model's reasoning is readable from its routing patterns.

This is what "glass-box AI" means: not interpretability bolted on after the fact, but an architecture where the computational structure IS the explanation.

---

## 2. What We Have Today

A GPT-2 model in pure C that routes the FFN through k-of-T ternary tiles with weight-derived signatures. One routed component, four dense ones:

```
Per block:
  LN → W_QKV [DENSE] → Attention heads [DENSE] → W_O [DENSE] → residual
  LN → Ternary FFN [ROUTED] → residual
Output:
  LM head [DENSE]
```

Result: **val PPL 632** on 9.5M FineWeb-Edu tokens, beating both the dense baseline (857) and learned-parameter routing (692).

---

## 3. Target Architecture

```
Per block:
  LN → Routed QKV → Routed Attention → Routed W_O → residual
  LN → Routed FFN → residual
Output:
  Routed LM head
```

Six routing points. All use the same mechanism: weight-derived ternary signatures, k-of-T signed composition, same backward pass pattern.

---

## 4. Design: Routed Projection (shared pattern)

QKV, W_O, and LM head are all projections — `y = x @ W + b`. They share a common routed form. Define it once, reuse three times.

### 4.1 `TrixRoutedProjection`

A generic routed linear projection: `[batch, in_dim] → [batch, out_dim]` via T tiles.

```c
typedef struct {
    int in_dim;
    int out_dim;
    int num_tiles;
    int active_k;
    float ln_eps;
} TrixRoutedProjectionConfig;

struct TrixRoutedProjection {
    TrixRoutedProjectionConfig cfg;

    /* LayerNorm (optional — QKV has it, W_O may not) */
    float* ln_weight;       /* [in_dim] or NULL */
    float* ln_bias;         /* [in_dim] or NULL */

    /* Per-tile weights */
    float* W;               /* [T, out_dim, in_dim] */
    float* b;               /* [T, out_dim] */

    /* Signatures (derived from W column sums) */
    float* signatures;      /* [T, in_dim] */

    /* output_scale */
    float output_scale;

    /* Routing scratch */
    float* scores;          /* [batch, T] */
    int* route;             /* [batch, T] */

    /* Saved activations for backward */
    float* x_norm;          /* [batch, in_dim] (after LN) */
    float** saved_out;      /* [T] → [batch, out_dim] per-tile output */

    /* Gradients + AdamW moments (same pattern as FFN) */
    /* ... */
};
```

**Forward:**
```
1. x_norm = LayerNorm(x) [if LN enabled]
2. scores[t] = dot(x_norm, signatures[t])
3. route = ternary_threshold_topk(scores, k)
4. for each tile t:
     tile_out = x_norm @ W_t^T + b_t
     combined += route_t * tile_out
5. output = combined * output_scale
   (no residual — the caller adds the residual)
```

**Signature update:** `sig_t[d] = sign(colsum(W_t[:, d]) - mean)`
Same as FFN, but over the projection weight columns instead of W1 columns.

**Key decision: tile width.**

Option A — **Full-width tiles:** Each tile has W_t `[out_dim, in_dim]`. Same total output dimension. Tiles specialize on which tokens they handle, not which output dimensions. Output = signed sum of full-width tile outputs. Parameter count = T × original. Only k tiles compute per token.

Option B — **Split tiles:** Each tile has W_t `[out_dim/T, in_dim]`. Each tile produces a slice of the output. Route decides which tiles contribute slices. Parameter-neutral but output structure changes.

**Recommendation: Option A (full-width).** Reason: it's the same pattern as the FFN, where each tile produces a full-rank output and the signed sum combines them. Split tiles would require a fundamentally different routing and accumulation strategy. Full-width tiles with k < T achieves parameter efficiency through sparsity — k/T of the tiles compute per token.

For parameter parity, set `tile_hidden = out_dim / T` so total params across tiles equals the original dense projection. But each active tile still produces the full output dimension.

Wait — that doesn't work for a linear projection (no hidden dimension). The projection IS the weights. For a routed projection, each tile needs the full `[out_dim, in_dim]` matrix. With T tiles and k active, the total params are T × out_dim × in_dim, but the compute per token is k × out_dim × in_dim.

To match the dense parameter count, we'd need T=1 (no routing). For T>1, we're spending more parameters. The tradeoff: more parameters, but only k/T of them compute per token (sparse), and the routing is interpretable.

**Practical approach:** Start with T=4, K=4 (all active, signed composition). This is 4× the parameters of the dense projection, but the signed composition gives 3^4 = 81 distinct effective projections per token. Later optimize with K<T for sparsity.

### 4.2 Residual Pattern

The FFN handles its own residual internally: `out = x + scale * ffn(LN(x))`.

For the routed projections, the caller handles the residual:
- Routed QKV: no residual (QKV is a new representation, not a correction)
- Routed attention output: residual added by the block
- Routed W_O: no residual (output projection, not a correction)

So the routed projection does NOT add a residual — it's just `output = scale * sum(route_t * tile_t(x))`.

The FFN stays as-is with its internal residual.

---

## 5. Design: Routed Attention Heads

### 5.1 Current attention

```
for each head h:
    Q_h, K_h, V_h = split(QKV, head=h)
    scores_h = Q_h @ K_h^T / sqrt(d_head)
    attn_h = causal_softmax(scores_h)
    head_out_h = attn_h @ V_h
result = concat(head_out_0, ..., head_out_{H-1})
```

All H heads compute for all tokens. The concat gives a [batch, seq, D] output.

### 5.2 Routed attention

```
# Derive head routing from Q statistics
q_pooled[batch, D] = mean_over_seq(Q) or Q per-token
head_scores[h] = dot(q_pooled, sig_head_h) for each head
head_route = ternary_threshold_topk(head_scores, k_heads)

for each head h:
    if head_route[h] == 0: skip
    Q_h, K_h, V_h = split(QKV, head=h)
    scores_h = Q_h @ K_h^T / sqrt(d_head)
    attn_h = causal_softmax(scores_h)
    head_out_h = attn_h @ V_h
    result += head_route[h] * expand_to_full_dim(head_out_h)
```

**Signature derivation for heads:** Each head's signature comes from its slice of the Q weight matrix. Head h uses columns `[h*d_head : (h+1)*d_head]` of W_Q. The column-sum + mean-subtraction + sign gives a ternary signature per head.

**Anti-attention (-1 heads):** A head with route -1 has its output subtracted. This means: "what this head attends to should be suppressed." This is a form of negative attention that doesn't exist in standard transformers.

**Key question: per-token or per-batch routing?** 

Per-token: each token gets its own head routing. Maximum expressiveness but complex implementation — different tokens within a sequence use different head subsets.

Per-batch: all tokens in a batch share the same head routing. Simpler but less expressive.

**Recommendation: per-token routing**, consistent with how the FFN routes. Each token independently decides which heads are relevant, which are anti-relevant, and which are irrelevant. This is the full glass-box: for any token, you can read which attention heads were consulted and with what polarity.

### 5.3 Routed attention backward

The backward pass through routed heads follows the same pattern as routed tiles in the FFN:

```
d_head_out[h] = head_route[h] * d_result   # sign modulates gradient
# then standard attention backward within active heads
# heads with route=0 get no gradient
```

---

## 6. Design: Routed LM Head

### 6.1 Current

```
logits = final_normed @ tok_emb^T    # [batch*seq, D] @ [D, vocab] → [batch*seq, vocab]
```

One dense matmul over the full vocabulary.

### 6.2 Routed LM head (Option A — signed sum, full vocab per tile)

Same pattern as the routed projection. Each tile produces logits over the full vocabulary:

```
route = ternary_threshold(dot(final_normed, sig_lm_t))
for each tile t:
    logits_t = final_normed @ W_lm_t^T      # [batch*seq, vocab]
    logits += route_t * logits_t
```

Problem: this requires T copies of the full [D, vocab] matrix. At D=64, vocab=32K, that's 8MB per tile. With T=8, that's 64MB just for the LM head — more than the rest of the model.

### 6.3 Routed LM head (Option B — vocabulary specialists)

Each tile covers a disjoint vocabulary slice. Route decides which slices are relevant:

```
route = ternary_threshold(dot(final_normed, sig_lm_t))
for each tile t:
    start = t * (vocab / T)
    end = (t + 1) * (vocab / T)
    logits[start:end] = final_normed @ W_lm_t^T    # [batch*seq, vocab/T]
```

But this doesn't use signed composition — slices are disjoint. The routing decides which vocabulary regions to compute (sparsity for efficiency) but doesn't provide the +1/-1 semantics.

### 6.4 Routed LM head (Option C — tied weight factored routing)

Keep weight tying with tok_emb, but route through tile-specific projection:

```
route = ternary_threshold(dot(final_normed, sig_lm_t))
for each tile t:
    projected_t = final_normed @ W_proj_t    # [batch*seq, D] — small projection
    logits += route_t * (projected_t @ tok_emb^T)  # [batch*seq, vocab]
```

Each tile has a small `[D, D]` projection, not a full `[D, vocab]` matrix. Total new params: T × D × D = 8 × 64 × 64 = 32K per tile. The final matmul against tok_emb^T is shared.

**Recommendation: Option C.** Preserves weight tying, adds minimal parameters (32K per tile vs 2M for the dense head), and maintains signed composition. Each tile learns a different "lens" through which to view the vocabulary.

---

## 7. Implementation Plan

### Phase 1: Routed Projection Primitive

**File:** `native/src/trix_routed_proj.c` / `.h`

Build `TrixRoutedProjection` with:
- create / destroy / ensure_scratch
- forward (with optional LayerNorm)
- backward
- zero_grad / adamw_step
- update_signatures

**Test:** Unit test comparing routed projection (T=1, K=1) against dense matmul. Should match exactly.

**LOC:** ~300

### Phase 2: Routed QKV + Routed W_O

**File:** Modify `trix_transformer.c` (or new `trix_transformer_routed.c`)

Replace the QKV matmul with a `TrixRoutedProjection(in=D, out=3D, T=4, K=4)`.
Replace the W_O matmul with a `TrixRoutedProjection(in=D, out=D, T=4, K=4)`.

Keep the attention computation unchanged for now.

**Test:** Training run on 1M tokens. Compare val PPL to the current dense-projection + routed-FFN model. The routed projections should match or slightly beat dense.

**LOC:** ~100 (wiring, not new algorithms)

### Phase 3: Routed Attention Heads

**File:** Modify MHA in `trix_transformer.c`

Add per-token head routing:
- Derive head signatures from Q weight slices
- Compute head scores from Q representations
- Ternary threshold to select active heads
- Signed accumulation of head outputs
- Backward with sign-modulated gradients per head

**Test:** Same training run. This is the riskiest change — attention is the most sensitive component. Start with K=H (all heads active) to verify no degradation, then reduce K.

**LOC:** ~150

### Phase 4: Routed LM Head

**File:** Modify `trix_gpt2.c`

Replace the dense LM head matmul with Option C (factored routing):
- Per-tile D×D projections
- Shared tok_emb^T for the final vocabulary projection
- Signed accumulation of logits
- Backward through the factored structure

**Test:** Same training run. LM head routing should provide the most interpretable signal — which tiles predict which vocabulary regions.

**LOC:** ~120

### Phase 5: Full Integration + Benchmarking

Wire all four routed components together. Run the full zero-dense model on:
- 1M tokens (smoke test)
- 10M tokens (comparison to current 632 PPL)
- 50M tokens (data saturation test)

Compare routing telemetry across all six routing points: QKV, attention heads, W_O, FFN, LM head. This produces the complete glass-box story.

**LOC:** ~50 (wiring + telemetry)

---

## 8. Parameter Budget

### Current model (d=64, L=20, T=8 FFN tiles)

| Component | Params | Type |
|-----------|--------|------|
| tok_emb | 2,048,000 | Dense |
| pos_emb | 16,384 | Dense |
| Per block × 20: | | |
|   attn_ln | 128 | — |
|   W_QKV + b_QKV | 12,480 | Dense |
|   W_O + b_O | 4,160 | Dense |
|   ffn_ln | 128 | — |
|   FFN tiles (8×) | 33,025 | Routed |
| final_ln | 128 | — |
| **Total** | **3,073,172** | 1 routed, 4 dense per block |

### Zero-dense model (d=64, L=20, T=4 projections, T=8 FFN, K_heads=4 of 4)

| Component | Params | Type | Change |
|-----------|--------|------|--------|
| tok_emb | 2,048,000 | Shared (lookup + LM) | same |
| pos_emb | 16,384 | Lookup | same |
| Per block × 20: | | | |
|   attn_ln | 128 | — | same |
|   Routed QKV (4 tiles × [64, 192]) | 49,536 | Routed | 4× QKV params |
|   Routed heads (sigs only) | 256 | Routed | new: head signatures |
|   Routed W_O (4 tiles × [64, 64]) | 16,640 | Routed | 4× W_O params |
|   ffn_ln | 128 | — | same |
|   Routed FFN (8 tiles) | 33,025 | Routed | same |
| Routed LM head (8 tiles × [64, 64]) | 32,776 | Routed | new: projections |
| final_ln | 128 | — | same |
| **Total** | **~5,100,000** | **All routed** | +66% params |

The parameter increase comes from tiling QKV and W_O (4× each). With K < T, compute per token stays comparable — only k tiles activate.

To keep parameter parity, reduce T for projections to 2 (each tile = half the original; two tiles signed-sum to full rank) or reduce FFN tiles.

### Parameter-neutral variant (T=2 for projections)

| Component | Params | Delta vs current |
|-----------|--------|-----------------|
| Routed QKV (2 tiles) | 24,768 | +2× |
| Routed W_O (2 tiles) | 8,320 | +2× |
| Everything else | same | same |
| **Total** | **~3,700,000** | +20% |

With T=2 and K=2 (both active, signed): each token gets the sum or difference of two projections. 2^2 = 4 distinct effective projections. Modest parameter increase, meaningful routing.

---

## 9. Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Routed QKV degrades attention quality | HIGH | Start with K=T (all active), compare to dense. If it degrades, the signed sum may interfere with attention's need for precise Q/K alignment. |
| Anti-attention (-1 heads) causes instability | HIGH | Start with K=H (all heads +1), then test K < H. The -1 case is novel and may need gradient clipping per head. |
| 4× parameter QKV tiles are wasteful | MEDIUM | Use T=2 parameter-neutral variant first. Scale T if the signal is positive. |
| LM head factored routing adds latency | LOW | The D×D per-tile projection is tiny. The shared tok_emb^T matmul dominates. |
| Signature derivation for attention heads is unstable | MEDIUM | The Q weight matrix may not have clean column-sum structure. Fall back to learned signatures if weight-derived fails for heads. |
| Too many routing decisions overfit on small data | MEDIUM | Test on 10M tokens. If 6 routing points overfit vs 1, reduce K or T. |

---

## 10. Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Val PPL on 10M tokens | ≤ 632 (match or beat current routed-FFN-only model) |
| All routing points show healthy tile utilization | > 80% tiles active across all 6 routing points |
| Routing is interpretable | Different QKV tiles, attention heads, and LM tiles show meaningful specialization |
| No training instability | Loss decreases monotonically after warmup; no NaN; gnorm < 20 |
| Throughput regression | < 3× slower than current model (routing overhead) |
| Signatures update stably | Column-sum + mean subtraction produces diverse signatures for all projection types |

---

## 11. Deliverables

| Phase | Deliverable | Est LOC | Est Time |
|-------|-------------|---------|----------|
| 1 | `TrixRoutedProjection` primitive + tests | 300 | 1 day |
| 2 | Routed QKV + W_O in transformer block + training run | 100 | 1 day |
| 3 | Routed attention heads + training run | 150 | 1-2 days |
| 4 | Routed LM head + training run | 120 | 1 day |
| 5 | Full integration + 10M benchmark + telemetry | 50 | 1 day |
| — | **Total** | **~720** | **5-6 days** |

---

## 12. The Story This Tells

If the zero-dense transformer works — if every matmul is routed, every routing decision is weight-derived, and the model is competitive — then we have:

1. **A transformer where every computation is observable.** For any token at any layer, you can read: which QKV specialist projected it, which attention heads attended to it (and which subtracted), which output specialist transformed it, which FFN expert processed it, and which vocabulary specialist predicted it.

2. **A complete routing audit trail.** The entire model's behavior is a sequence of ternary decisions (+1/-1/0) derived from the model's own weights. No opaque gating networks. No learned routers. Just weight column sums, mean subtraction, and sign.

3. **Hardware-accelerable routing everywhere.** Every routing decision reduces to XOR+POPCNT. On ARM with VCNT, routing the entire model — all six routing points — costs a handful of instructions per token.

4. **Signature surgery on any computation.** Want to change how a token is projected? Edit the QKV tile signature. Want to change which attention heads process it? Edit the head signature. Want to change which vocabulary region it predicts? Edit the LM head signature. All with the same API, all with audit trails.

This isn't an incremental improvement over existing MoE. It's a different kind of model: one where the structure IS the explanation.

That's the paper. That's what Hinton asked for.
