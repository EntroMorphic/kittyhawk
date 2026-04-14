# Trit Lattice LSH

Classification as geometric partitioning on the MTFP lattice.

## The idea

MTFP values live on a lattice with spacing `1/3^10`. Ternary operations (add/subtract/skip) are the natural movements on this lattice. A ternary signature `{-1, 0, +1}^D` defines a hyperplane that partitions the lattice into two half-spaces. The dot product `dot(x, sig)` measures which side the input falls on. Multiple signatures = multiple hyperplanes = a hash table over the lattice.

This is locality-sensitive hashing (LSH). Similar inputs hash to the same bucket. The hash functions are ternary hyperplanes. The computation is ternary matmul (add/subtract/skip). No multiplication. No float.

## Connection to M4T routing

The M4T routing layer IS Trit Lattice LSH:

| M4T primitive | LSH interpretation |
|---|---|
| `m4t_mtfp_ternary_matmul_bt` | Hash: project input onto ternary hyperplanes |
| `m4t_route_sign_extract` | Hash bit extraction: which side of each hyperplane |
| `m4t_popcount_dist` | Hash agreement: count matching hash bits |
| `m4t_route_topk_abs` | Bucket selection: pick K most-aligned buckets |
| `m4t_route_apply_signed` | Combine: merge per-bucket results with signs |

## Results on MNIST

All zero-float, zero-gradient, zero-training-iterations:

| Method | Accuracy | Note |
|---|---|---|
| Class centroid signatures (10 templates) | 59.50% | sign(centroid − global_mean), direction-based |
| Pairwise signatures (90 templates) | 60.01% | More templates of same quality don't help |
| Single-trit routes in projection space | 59.44% | Direction-based, one template per class |
| Multi-trit routes (MTFP4) in projection space | 58.87% | Magnitude on one direction doesn't help |
| Pixel-space L1 nearest centroid | 66.85% | Raw pixels, no projection |
| Random ternary proj (256) + L1 centroid | 80.12% | **Distance beats direction** |
| Random ternary proj (1024) + L1 centroid | 81.14% | Scaling helps slightly |
| **Random ternary proj (2048) + L1 centroid** | **81.40%** | **Best zero-float result** |

Two-stage experiments (all degraded accuracy):
| Method | Accuracy | Note |
|---|---|---|
| L1 proj → pairwise ternary refine (top-3) | 65.63% | Direction overrides distance — hurts |
| L1 proj → pixel-space L1 refine (top-3) | 73.71% | Weaker space overrides stronger — hurts |

Comparison (float in pipeline):
| Method | Accuracy | Float |
|---|---|---|
| trix-z reference (float + STE, peak) | 97.41% | everywhere |
| M4T inference on float-trained weights | 97.46% | training only |
| All-ternary STE from random init | 11.35% | training (dead) |

## Key findings

1. **The projection is the intelligence.** Random ternary projection to 256+ dims creates a better representation than raw pixels. L1 in projection space (80-81%) beats L1 in pixel space (67%) by 14 points.

2. **Distance beats direction everywhere.** Every dot-product classifier (centroid signatures, pairwise templates, single/multi-trit routes) scored 58-60%. Every L1 distance classifier scored 67-81%.

3. **Refinement hurts when the refining classifier is weaker.** Both pairwise ternary refinement and pixel-space L1 refinement degraded accuracy when overriding projection-space L1.

4. **Scaling projections saturates around 81%.** The ceiling is the classifier (10 centroids), not the representation.

## Why it works

Random ternary projections preserve distances on the lattice (a ternary analog of Johnson-Lindenstrauss). Two images that are close in pixel space produce similar dot products with a random ternary vector, so they hash to nearby points in the projection space. Class centroids in projection space are better separated than in pixel space because the random hashing decorrelates the dimensions.

## Why centroid signatures are weaker

`sign(centroid − global_mean)` captures only the direction of deviation, not its magnitude. A pixel slightly above average and a pixel far above average both get +1. The signature loses the structure that makes the centroid useful. Random projections don't have this problem — they preserve distances, not just directions.

## Key realization: no dense MTFP matmul needed

The Trit Lattice LSH forward pass uses only:
- Ternary matmul (projection)
- Integer arithmetic (L1 distance, argmax)

No dense MTFP×MTFP matmul (no `__int128`). No GELU LUT. No LayerNorm. No softmax. The entire classification is ternary matmul + integer comparison.

## Path forward

1. **More projections** (512 → 1024 → 2048) for finer distance preservation
2. **Data-dependent second layer** (class signatures in projection space)
3. **Multi-layer hashing** (project → sign → re-project → sign → classify)
4. **Routing integration** (select which projections to apply per input)
5. **Tile-based refinement** (ternary FFN tiles on residuals)
