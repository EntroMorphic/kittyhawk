# Trit Lattice LSH — Complete Experimental Record

Every experiment run during the Trit Lattice LSH development, in chronological order. All results are on the MNIST test set (10,000 images). All methods use zero float arithmetic — integer only.

---

## Context

The Trit Lattice LSH framework emerged from three discoveries during the M4T MNIST validation:

1. **Dense MTFP matmul is 95% of inference compute** — the projection and head layers (which use MTFP×MTFP multiplication with `__int128` rescale) dominate cost, while the ternary routed FFN (which uses add/subtract/skip) is 4%.

2. **MTFP multiplication is the wrong operation on the lattice** — MTFP values live on a lattice with spacing `1/3^10`. Trit operations (add/subtract/skip) stay on the lattice. MTFP×MTFP multiplication leaves the lattice (scale becomes `1/3^20`) and requires an expensive rescale back. The natural operations on the lattice are ternary.

3. **Ternary routing IS locality-sensitive hashing** — a ternary signature `{-1,0,+1}^D` defines a hyperplane on the lattice. The dot product `dot(x, sig)` measures which side the input falls on. Multiple signatures = multiple hyperplanes = a hash-based partition of the lattice. The XOR+POPCNT routing distance is hash agreement.

These led to the question: can we classify MNIST using ONLY lattice operations (ternary matmul + integer arithmetic), with no float anywhere — not in training, not in inference, not in data loading?

---

## Experiment 1: Class centroid signatures

**Date:** 2026-04-12
**Commit:** `a6fbc77`

**Method:**
1. Load MNIST pixels as MTFP19 cells: `cell = pixel * 59049 / 255` (integer)
2. Compute per-class centroid on the lattice: `centroid_c[d] = sum(train_images_of_class_c[d]) / count_c` (int64 sum, int32 divide)
3. Compute global centroid: `global[d] = mean(centroid_c[d] over c)` (integer)
4. Ternary signature per class: `sig_c[d] = sign(centroid_c[d] - global[d])` ∈ {-1, 0, +1}
5. Classify: `score_c = dot(test_image, sig_c)` via `m4t_mtfp_ternary_matmul_bt`, argmax over c

**Result:** 5,950 / 10,000 = **59.50%**

**Signature statistics (trits per class):**
```
class 0: +1=292, -1=387, 0=105
class 1: +1=68,  -1=614, 0=102
class 2: +1=344, -1=334, 0=106
class 3: +1=275, -1=407, 0=102
class 4: +1=203, -1=471, 0=110
class 5: +1=234, -1=445, 0=105
class 6: +1=269, -1=413, 0=102
class 7: +1=239, -1=441, 0=104
class 8: +1=246, -1=430, 0=108
class 9: +1=208, -1=473, 0=103
```

**Analysis:** 6× above random chance (10%). The lattice geometry carries real class structure. But 59.50% is well below usable. The signatures are dominated by -1 trits (pixels below global mean) — they encode "not the average" more than "is this class." Class 1 (digit "1") has only 68 positive trits, meaning its signature is almost entirely "most pixels are below average" — true but not distinctive.

**Key insight:** `sign(centroid - global_mean)` discards magnitude. A pixel slightly above average and a pixel far above average both get +1. The centroid-difference direction is captured; the centroid-difference magnitude is lost.

---

## Experiment 2: Pairwise class signatures

**Date:** 2026-04-12
**Commit:** `a6fbc77` (same run)

**Method:** Same as Experiment 1, but instead of 10 signatures (one per class vs global mean), compute 90 directed pairwise signatures: `sig_ij[d] = sign(centroid_i[d] - centroid_j[d])` for all i ≠ j. Classify by accumulating pairwise votes: for pair (i,j), if `dot(image, sig_ij) > 0`, add the score to class i's total.

**Result:** 6,001 / 10,000 = **60.01%**

**Analysis:** Only 0.5 points above the 10-template baseline. More templates of the same quality don't help. The pairwise centroid differences are highly correlated with the centroid-vs-global differences — they carry nearly the same information. The bottleneck is template QUALITY (sign discards magnitude), not template QUANTITY.

---

## Experiment 3: Random ternary projections + L1 centroid

**Date:** 2026-04-12
**Commit:** `bcf8bcd`

**Method:**
1. Generate 256 random ternary vectors: `proj[p][d] ∈ {-1, 0, +1}`, i.i.d. uniform
2. Project all training images: `projected[i][p] = dot(train_image_i, proj_p)` via `m4t_mtfp_ternary_matmul_bt`. This maps 784-dim pixel space to 256-dim projection space.
3. Compute class centroids in projection space: `centroid_c[p] = mean(projected_images_of_class_c[p])` (integer)
4. Classify: for each test image, project to 256-dim, compute L1 distance to each class centroid, predict the nearest class

**Result:** 7,974 / 10,000 = **79.74%**

**Analysis:** 20-point jump from centroid signatures. The random projections are distance-preserving on the lattice (ternary analog of Johnson-Lindenstrauss). Two images close in pixel space produce similar projections, so they land near the same centroid in projection space. Unlike centroid signatures, the random projections are DIVERSE — each captures a different slice of the input geometry. And L1 distance uses ALL dimensions, not just the alignment with one direction.

**Key insight:** Random projections + L1 distance > data-dependent templates + dot product. Distance beats direction.

---

## Experiment 4: Multi-trit routes (MTFP4 weighted)

**Date:** 2026-04-12
**Commit:** `24b33d8`

**Hypothesis:** If single-trit routes (sign only, 3 levels) are too coarse, multi-trit routes (MTFP4, 40 levels of magnitude) should be better — they preserve HOW MUCH the centroid differs from the global mean, not just the direction.

**Method:** Same as Experiment 3 (random ternary projection to 256 dims), but classify using a weighted dot product instead of L1 distance. The route weight per class per projection dimension is `MTFP4(centroid_c[p] - global[p])` — the centroid difference quantized to a 4-trit value (range ±40). The class score is `sum_p(route_c[p] * projected_image[p])`.

**Results (N_PROJ=256):**
```
Single-trit routes (sign only):   5,944 / 10,000 = 59.44%
Multi-trit routes (MTFP4):        5,887 / 10,000 = 58.87%
L1 nearest centroid:              8,012 / 10,000 = 80.12%
```

**Analysis:** Multi-trit routes are WORSE than single-trit routes. Both are much worse than L1. The multi-trit route is still a dot-product classifier — one direction per class, weighted by magnitude. Adding magnitude to a single direction doesn't help when the problem requires DISTANCE across all dimensions. L1 centroid uses all 256 dimensions independently; the dot product collapses them to a single score along one direction.

**Key insight:** The multi-trit route's extra expressiveness is wasted in a single-template-per-class classifier. Multi-trit routes would help for tile MODULATION (controlling how much each tile contributes), not for direct classification.

---

## Experiment 5: Two-stage with pairwise ternary refinement

**Date:** 2026-04-12
**Commit:** `36c35b4`

**Hypothesis:** Stage 1 (L1 coarse routing) narrows to K candidates. Stage 2 (pairwise ternary voting among candidates) refines the decision using class-pair-specific ternary templates in projection space.

**Method:**
1. Random ternary projection (256/512/1024 dims)
2. L1 nearest centroid → rank all 10 classes
3. Among top-K (K=3 or K=5): for each pair (i,j), compute `dot(projected, sign(centroid_i - centroid_j))`. Positive → vote for i; negative → vote for j. Accumulate votes. Class with most votes wins among the finalists.

**Results (N_PROJ=1024):**
```
L1 centroid only:            8,114 / 10,000 = 81.14%
L1 top-3 → pairwise refine: 6,563 / 10,000 = 65.63%
L1 top-5 → pairwise refine: 6,058 / 10,000 = 60.58%
```

**Analysis:** Pairwise refinement HURTS — by 15-20 points. The direction-based pairwise templates (sign of centroid difference in projection space) override good L1 distance decisions with bad direction-based votes. The pairwise signatures suffer the same weakness as all direction-based approaches: they discard magnitude and project onto a single axis per pair.

**Key insight:** Direction-based refinement cannot improve on distance-based classification. The L1 distance in projection space is a stronger signal than any dot-product-based pairwise vote.

---

## Experiment 6: Pixel-space L1 refinement + scaling

**Date:** 2026-04-12
**Commit:** `f445119`

**Hypothesis:** Maybe the refinement should use a DIFFERENT space. Stage 1 uses projection-space L1 for coarse routing. Stage 2 uses pixel-space L1 (784-dim) for fine refinement among the top-K candidates. The pixel space has more information (784 dims vs 256-2048 projected dims).

**Method:**
1. Compute pixel-space class centroids (integer mean per class, 784-dim)
2. Random ternary projection → projection-space L1 → rank classes
3. Among top-K: compute pixel-space L1 distance to each candidate's pixel-space centroid. Nearest wins.

Also tested: scaling the number of random projections from 256 to 2048.

**Results:**

Baseline:
```
Pixel-space L1 (no projection):  6,685 / 10,000 = 66.85%
```

Projection-space L1 only:
```
N_PROJ=256:   8,012 / 10,000 = 80.12%
N_PROJ=512:   8,046 / 10,000 = 80.46%
N_PROJ=1024:  8,114 / 10,000 = 81.14%
N_PROJ=2048:  8,140 / 10,000 = 81.40%
```

Two-stage with pixel-space refinement (N_PROJ=2048):
```
L1 proj top-3 → pixel refine:  7,371 / 10,000 = 73.71%
L1 proj top-5 → pixel refine:  7,009 / 10,000 = 70.09%
```

**Analysis:** Three critical findings.

**Finding 1: Pixel-space L1 (66.85%) is WORSE than projection-space L1 (80-81%).** The random ternary projection creates a BETTER representation for L1 classification than raw pixels. The projection decorrelates the 784 pixel dimensions, making L1 distance more meaningful. In raw pixel space, L1 distance is dominated by background pixels (which vary across writing styles but don't distinguish digits). The projection hashes away this noise and preserves the discriminative structure.

**Finding 2: Pixel-space refinement hurts.** Using a weaker classifier (pixel-space L1) to override a stronger one (projection-space L1) degrades accuracy. The projection IS the intelligence.

**Finding 3: Projection-space L1 saturates at ~81%.** Scaling from 256 to 2048 projections gains only 1.3 points. The centroid-based classifier in projection space has a capacity ceiling — 10 centroids cannot partition the 10-class space finely enough.

---

## Summary of all results

| # | Method | Accuracy | Key insight |
|---|---|---|---|
| 1 | Centroid signatures (10 ternary templates, pixel space) | 59.50% | Lattice geometry is real |
| 2 | Pairwise signatures (90 ternary templates) | 60.01% | More same-quality templates don't help |
| 3 | Random ternary projections (256) + L1 centroid | 79.74% | Distance beats direction |
| 4a | Single-trit routes in projection space | 59.44% | Direction loses to distance |
| 4b | Multi-trit routes (MTFP4) in projection space | 58.87% | Magnitude on one direction doesn't help |
| 4c | L1 centroid in projection space (same run as 4a/b) | 80.12% | L1 is consistently best |
| 5a | L1 top-3 → pairwise ternary refine (N=1024) | 65.63% | Direction-based refinement hurts |
| 5b | L1 top-5 → pairwise ternary refine (N=1024) | 60.58% | More candidates = more bad overrides |
| 6a | Pixel-space L1 (no projection) | 66.85% | Raw pixels are worse than projected |
| 6b | **Projection-space L1 (N=2048)** | **81.40%** | **Best result — projection IS intelligence** |
| 6c | L1 proj → pixel refine (N=2048, top-3) | 73.71% | Weaker space can't refine stronger |
| 6d | L1 proj → pixel refine (N=2048, top-5) | 70.09% | More candidates = more degradation |

**For comparison (float-trained paths):**
| Method | Accuracy | Float in pipeline |
|---|---|---|
| trix-z reference (float + STE, peak) | 97.41% | everywhere |
| M4T inference on float-trained weights | 97.46% | training only |
| All-ternary STE from random init | 11.35% | training (dead) |

---

## What we learned

### The projection is the intelligence

The single most important finding: a random ternary projection (256+ dimensions) from pixel space creates a representation where L1 nearest centroid achieves 81% — 14 points better than L1 in raw pixel space (67%). The projection doesn't just reduce dimensionality; it creates a BETTER space for distance-based classification.

This is the ternary analog of random projections in float (Johnson-Lindenstrauss): distance is approximately preserved, but the decorrelation of dimensions makes L1 distance more discriminative. In pixel space, L1 is dominated by background pixel variation. In projection space, the background noise is hashed into random offsets that cancel out, while the digit structure (which is consistent within a class) accumulates coherently.

### Distance beats direction everywhere

Every experiment that used dot product (direction-based) classification — centroid signatures, pairwise signatures, single-trit routes, multi-trit routes — scored 58-60%. Every experiment that used L1 distance — nearest centroid in pixel space, nearest centroid in projection space — scored 67-81%.

The reason: dot product with a ternary template is a COUNTING operation (how many dimensions agree in sign). It captures one bit of information per dimension (agree/disagree) along one direction. L1 distance captures MAGNITUDE of deviation per dimension, across ALL dimensions simultaneously. L1 has strictly more information.

### Refinement hurts when the refining classifier is weaker

Both pairwise ternary refinement (Experiment 5) and pixel-space L1 refinement (Experiment 6) degraded accuracy when applied to the top-K candidates from projection-space L1. The reason: the refinement classifier (direction-based or pixel-space L1) is WEAKER than the initial classifier (projection-space L1). A weaker classifier overriding a stronger one can only hurt.

For refinement to help, it must use STRONGER or ORTHOGONAL information. Same-space re-ranking is a no-op. Weaker-space refinement is destructive.

### Scaling projections has diminishing returns for centroid classification

256 → 2048 projections gained 1.3 points (80.12% → 81.40%). The ceiling is the classifier, not the representation: 10 centroids cannot partition the 10-class projection space finely enough. To go higher: more centroids per class (sub-centroids from k-means), k-NN (compare to all training images), or a tree-based classifier.

### Multi-trit routes have the right idea, wrong application

MTFP4 route weights (4-trit, 40 levels of magnitude) don't help for single-template classification because one direction per class is too coarse regardless of resolution. Multi-trit routes would help for tile MODULATION — controlling how much each tile contributes to the output in a multi-tile routed architecture. The application matters more than the resolution.

---

## The 81% → 97% gap

The best zero-float result (81.40%) is 16 points below the float-trained M4T result (97.46%). The gap is:

**Capacity:** The zero-float model uses 10 centroids (one per class) as the classifier. The float-trained model has 398K parameters across 4 layers. The zero-float model has ~10 × 2048 = 20K parameters (10 centroids × 2048 projection dimensions), plus the 2048 × 784 ternary projection matrix (~400K trits, but these are random and not learned).

**Nonlinearity:** The float-trained model has GELU activations and LayerNorm. The zero-float model has no nonlinearity — L1 distance is a linear operation on absolute values. The ternary projection followed by L1 is effectively a linear classifier in a hashed space.

**Routing:** The float-trained model has k-of-T ternary routing with 4 specialized tiles. The zero-float model has no routing — it applies the same classifier to all inputs. Routing would allow different subsets of projections to be used for different input regions.

**To close the gap within zero-float:**
1. Multiple centroids per class (sub-clusters) — adds capacity (~85-90%)
2. k-NN in projection space — uses all training data (~95%+, but expensive)
3. Routing + tiles — apply different projections to different input regions
4. Multi-layer projection — project → sign → re-project → classify

All of these are integer operations on the MTFP lattice. The substrate supports them.

---

## What comes next

The experimental record shows that the zero-float Trit Lattice LSH framework is sound. The 81% result proves the geometry works. The gap to 97% is CAPACITY (not enough centroids, no nonlinearity, no routing), not a fundamental limitation of the lattice.

The most promising next experiments:
1. k-means sub-centroids per class (k=5-10 per class) → should reach 85-90%
2. k-NN in projection space (k=5-10) → should reach 95%+ but expensive
3. Two-layer projection with sign nonlinearity → project → sign → re-project → L1
