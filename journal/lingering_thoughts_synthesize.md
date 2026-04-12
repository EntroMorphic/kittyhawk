# Synthesis: Lingering Thoughts → Next Phase

Output of an LMM cycle on the post-pipeline unease. The substrate is complete. This document defines what comes next and why.

---

## One-line answer

**The substrate is a hypothesis. MNIST is the experiment. Run it.**

---

## The experiment

Reproduce trix-z's 98.22% MNIST accuracy using M4T primitives and trix-z's trained weights. This is the minimum viable validation that the ternary fixed-point substrate produces correct results under real model composition.

**Pass threshold:** ≥ 97.5% accuracy. MTFP rounding will cause small deviations from the float reference; anything above 97.5% confirms the substrate works. Below 95% means something fundamental is wrong.

---

## What it requires, in dependency order

### Step 1: Commit the LUT tables

Run `tools/m4t_lut_gen.c`, capture `src/m4t_mtfp_tables.c` (5.4 MB, 708K GELU + 708K exp entries). Add runtime lookup functions to `m4t_mtfp.{h,c}`: `m4t_mtfp_gelu(dst, src, n)` and `m4t_mtfp_softmax(dst, src, rows, cols, causal)`. These are pure-integer table lookups — no float at runtime.

**Unblocks:** nonlinear computation. GELU for FFN activation, softmax for classification.

**Estimated effort:** small. Generator exists. Runtime lookup is ~20 lines per function.

### Step 2: Build the IO tool

`tools/m4t_io.c`: read trix-z's float32 checkpoint (a flat binary of float arrays), convert each parameter to MTFP19 cells, write a glyph-format binary blob. Reverse mode for debugging.

For the MNIST model, the parameters are: projection weight [input_dim, d_model], T tile weights [T, tile_hidden, d_model], T tile biases [T, tile_hidden], classification head weight [d_model, 10], classification head bias [10], LayerNorm weight/bias [d_model] per block. Total: ~398K params × 4 bytes float = ~1.6 MB input.

Ternary weights (the tile W1/W2 matrices) don't need float conversion — they're already {-1, 0, +1}. Only biases, LN params, projection weights, and the classification head need MTFP conversion.

**Unblocks:** loading real weights.

**Estimated effort:** medium. ~200 LOC for a basic reader/writer.

### Step 3: Write the MNIST forward pass

A small C program that loads MTFP weights, reads MNIST test images (28×28 pixels, flattened to 784), and runs the forward pass:

```
input [784] → mtfp19_to_mtfp4 (or keep in mtfp19)
  → projection: input @ W_proj^T + bias_proj       (ternary matmul + bias)
  → GELU activation                                  (LUT lookup)
  → for each routed FFN block:
      → LayerNorm                                    (integer isqrt)
      → signature_update (once, cached)
      → distance_batch (per token)
      → topk_abs (per token)
      → for each selected tile:
          → tile FFN: x @ W1_t^T + b1, GELU, @ W2_t^T + b2   (ternary matmul)
      → apply_signed (accumulate)
      → residual add
  → classification head: hidden @ W_head^T + b_head  (ternary matmul + bias)
  → softmax (LUT lookup)
  → argmax → predicted digit
```

Compare predicted digit to label for all 10K test images. Report accuracy.

**Unblocks:** the experiment.

**Estimated effort:** large. ~500 LOC for the forward pass, plus weight-layout plumbing.

### Step 4: Run and compare

Run the forward pass on MNIST test set. Compare accuracy to trix-z's 98.22% float reference.

If ≥ 97.5%: **hypothesis confirmed.** M4T is validated. Write up the result.

If < 97.5%: investigate. Most likely causes:
- MTFP rounding accumulating through deep layers (fix: check per-layer activations against float reference)
- Saturation at MTFP19 boundary (fix: widen to MTFP39 for the offending layer)
- Off-by-one in a primitive composition (fix: the first real integration bug)

---

## What to do about the other lingering items

| Item | Action | When |
|---|---|---|
| LayerNorm p99 jitter | Fix the isqrt (cap iterations or use small LUT). Document contract status. | During step 3 — LayerNorm is on the forward-pass critical path. |
| SDOT-routing bridge | Not needed for MNIST (MTFP19 path is sufficient). | After MNIST, before GPT-2. |
| MTFP19 range validation | MNIST IS the validation. If it passes, range is sufficient for small models. | Step 4 answers this. |
| Training in MTFP | Research question. Design the update rule on paper while inference ships. | Parallel track, not blocking. |
| External users | After MNIST accuracy is confirmed and documented. | After step 4. |

---

## What this synthesis does NOT plan

- GPT-2 forward pass (too complex for the first experiment; save for after MNIST).
- Attention mechanism (not needed for MNIST).
- MTFP4 SDOT integration into the routing path (performance optimization, not correctness).
- CI pipeline (nice to have, not blocking).
- Contract publication (blocked on M3 measurement, which needs MNIST or equivalent).

---

## Honest assessment

The substrate phase was the safe work. It was important and we did it well. But we stayed in it one pipeline item longer than we needed to because it was comfortable. The items that remain — LUT tables, IO tool, forward pass — are less clean, more messy, and closer to the real problem. That's where the value is.

The single most surprising thing this LMM cycle surfaced: **we're not blocked by anything technical. The LUT generator exists. The routing primitives exist. The matmul exists. The LayerNorm exists. The only thing between us and the MNIST experiment is: sit down and wire them together.** The unease isn't about missing pieces. It's about the transition from building to composing. From safe to unsafe. From known to unknown.

The unknown is where the good stuff is.

---

## Next action

Commit the LUT tables. That's the 5-minute action that changes the state from "can't do nonlinear computation" to "can do nonlinear computation." Everything else flows from there.
