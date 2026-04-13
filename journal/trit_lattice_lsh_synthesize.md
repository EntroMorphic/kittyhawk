# Synthesis: Trit Lattice LSH

---

## One-line answer

**Classification on the trit lattice is geometric partitioning. The hash functions are ternary hyperplanes. The computation is add/subtract/skip. No float, no gradients, no backprop — just lattice geometry.**

---

## What we proved

| Experiment | Accuracy | Float | Gradients | Key insight |
|---|---|---|---|---|
| Centroid signatures (10 templates) | 59.50% | zero | zero | Lattice geometry carries class structure |
| Pairwise signatures (90 templates) | 60.01% | zero | zero | More templates of same quality don't help |
| Random ternary projections (256-dim LSH) | 79.74% | zero | zero | Distance-preserving hash > class-specific hash |
| Float-trained, M4T inference | 97.46% | training | yes | The accuracy ceiling for this architecture |
| All-ternary STE from random init | 11.35% | training | yes | Gradient descent can't train ternary from scratch |

---

## The Trit Lattice LSH framework

### Architecture

```
Input (MTFP19 lattice point, 784-dim)
    ↓
Layer 1: Ternary projection [N_PROJ, 784]           ← hash: ternary matmul
    ↓
N_PROJ-dim representation on the lattice
    ↓
Layer 2 (optional): Data-dependent projection        ← refine: ternary matmul
    ↓
K-dim refined representation
    ↓
Classify: L1 nearest centroid in projection space    ← decide: integer distance
```

### Operations used

| Operation | M4T primitive | Cycles/element |
|---|---|---|
| Ternary projection | `m4t_mtfp_ternary_matmul_bt` | ~0.25 (add/sub/skip) |
| Sign extraction | `m4t_route_sign_extract` | ~1 |
| L1 distance | integer abs + sum | ~1 |
| Argmax | `m4t_mtfp_argmax` | ~1 |

No `m4t_mtfp_matmul` (dense MTFP×MTFP). No `m4t_mtfp_gelu` (LUT). No `m4t_mtfp_layernorm` (isqrt). The entire forward pass is ternary matmul + integer arithmetic.

### "Training" algorithm (zero float)

1. **Load data as MTFP19 cells.** `pixel * SCALE / 255`, integer.
2. **Generate projection matrix.** Random ternary vectors, or structured (to be explored).
3. **Project all training images.** Ternary matmul: `[n_train, 784] × [N_PROJ, 784]^T → [n_train, N_PROJ]`.
4. **Compute class centroids in projection space.** Integer sum / count per class.
5. **Inference: project test image, L1 nearest centroid.** Integer distance.

Every step is integer. No gradients. No loss function. No optimizer state. The "learned" part is the class centroids in projection space — computed from data statistics in one pass.

---

## What remains to close the gap to 90%+

### Immediate (mechanical, no new ideas needed)

- **More projections.** 256 → 512 → 1024 → 2048. Each doubling should improve distance preservation and class separation. Ternary matmul scales linearly.

- **Data-dependent second layer.** In the projection space, compute per-class signatures: `sign(class_centroid_proj - global_centroid_proj)`. These are ternary templates in the hash space. Score = dot(projected_image, class_template) via ternary matmul. This replaces L1 nearest centroid with a ternary classifier.

### Exploratory (may require new ideas)

- **Multi-layer hashing.** Project → sign → re-project → sign → classify. Each sign extraction is the nonlinearity (it's a 3-valued quantization, not a smooth activation). Each projection is a ternary matmul. Depth adds capacity.

- **Routing integration.** The M4T routing primitives (topk_abs, apply_signed) can select which projections to apply per input. Not all projections are useful for all inputs — a routing layer can focus computation on the most relevant hash functions.

- **Tile-based refinement.** After the initial projection + classification, route the residual (image minus nearest-centroid reconstruction) through ternary tiles for refinement. This is the FFN architecture, but with data-derived tile weights instead of gradient-trained ones.

---

## Architectural simplification

The Trit Lattice LSH architecture eliminates several components that the trix-z architecture required:

| Component | trix-z (float-trained) | Trit Lattice LSH |
|---|---|---|
| Dense MTFP matmul | yes (projection, head) | **eliminated** |
| GELU LUT (5.4 MB) | yes | **eliminated** — sign is the nonlinearity |
| LayerNorm | yes | **eliminated** — ternary projection is self-normalizing |
| Float shadow weights | yes (training) | **eliminated** |
| STE backward pass | yes (training) | **eliminated** |
| Optimizer state (AdamW) | yes (training) | **eliminated** |
| __int128 accumulator | yes (dense matmul) | **eliminated** |

The forward pass is: one or more ternary matmuls + sign extraction + integer distance. The "training" is: one ternary matmul over the training set + integer centroid computation. Both are expressible entirely in terms of M4T primitives that already exist.

---

## Relationship to the M4T substrate

Trit Lattice LSH validates M4T's core thesis: **ternary operations on the MTFP lattice are sufficient for real computation.** The 79.74% result uses only:
- `m4t_mtfp_ternary_matmul_bt` (the ternary matmul kernel)
- `m4t_pack_trits_rowmajor` (weight packing)
- Integer arithmetic (sums, absolute differences, comparisons)

No other M4T primitive is needed. The 5.4 MB GELU table, the __int128 dense matmul, the softmax, the LayerNorm — all designed for the float-trained architecture — are unused. The lattice-native path is simpler, smaller, and faster.

The remaining M4T primitives (routing, trit ops, reducers) become relevant when the architecture grows: multi-layer hashing needs routing to select which projections to apply; tile-based refinement needs the routing primitives.

---

## The honest assessment

**79.74% with zero float is real.** It's not competitive with float-trained models (97%+), but it proves the geometric framework works. The gap is capacity and architectural refinement, not a fundamental limitation of the lattice.

**The path to 90%+ is visible.** More projections and a data-dependent second layer should get there. The path to 95%+ is uncertain — it may require multi-layer hashing or ideas we haven't had yet.

**The path to 97%+ without float is a research question.** The float-trained model has 398K parameters optimized by 60,000 gradient steps. Matching that with zero-float geometric construction would be a genuine advance. It's not clear it's possible with the current framework.

**What IS clear:** the trit lattice is a valid computational substrate. MTFP is geometric. Ternary operations are the natural metric. And gradient descent is not the only way to find structure on this lattice.

---

## Next action

Scale N_PROJ to 1024, add data-dependent second-layer signatures, and measure. If accuracy crosses 90%, the framework is validated for practical use. If not, investigate multi-layer hashing.
