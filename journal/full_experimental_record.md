# Full Experimental Record — Glyph / M4T / Trit Lattice LSH

Everything we tried on MNIST, in chronological order, with what worked, what didn't, and what we learned from each.

---

## Phase 1: Float-trained M4T inference

### Experiment 1.1: M4T inference with 1-epoch weights (Hamming routing bug)
- **What:** Loaded float-trained weights (1 epoch), ran M4T forward pass
- **Result:** 65.53%
- **What went wrong:** Routing used Hamming distance on sign-extracted signatures instead of MTFP dot product. Also missing fan-in normalization before GELU.
- **Lesson:** The routing score must be a DOT PRODUCT (ternary matmul), not a Hamming distance. The two are fundamentally different operations.

### Experiment 1.2: M4T inference with 1-epoch weights (routing fixed)
- **What:** Same weights, fixed routing to use ternary matmul dot product, added fan-in normalize
- **Result:** 94.92% (vs 95.42% float reference)
- **Delta:** -0.50%
- **What worked:** The ternary matmul dot product for routing scores. Fan-in normalization preventing GELU saturation.
- **Lesson:** MTFP rounding costs ~0.50% vs float32. The substrate works.

### Experiment 1.3: M4T inference with 20-epoch peak weights
- **What:** Trained 20 epochs, saved weights at peak accuracy (epoch 3, 97.41%), ran M4T
- **Result:** 97.46%
- **Delta:** +0.05% vs float reference
- **What worked:** Peak-epoch weights + proper M4T forward pass.
- **Lesson:** The MTFP19 substrate reproduces — and slightly exceeds — the float reference.

---

## Phase 2: All-ternary training attempts

### Experiment 2.1: All-ternary STE from random init (SGD, lr=0.0174)
- **What:** Quantized ALL weights to ternary in forward pass, STE backward, SGD
- **Result:** 11.35% (random)
- **What went wrong:** SGD updates too small to flip trits. The model is frozen.
- **Lesson:** SGD + STE can't train ternary weights from random initialization.

### Experiment 2.2: All-ternary STE (SGD, lr=0.1)
- **Result:** 11.35% — still frozen
- **Lesson:** Learning rate doesn't fix the gradient signal problem.

### Experiment 2.3: All-ternary STE (SGD, lr=1.0)
- **Result:** 10.10-11.35% — oscillating, random
- **Lesson:** High LR causes random trit flips, not directed learning.

### Experiment 2.4: All-ternary STE (AdamW, lr=0.0174)
- **Result:** 11.35%
- **What went wrong:** AdamW accumulates momentum but the gradient signal through random ternary layers is pure noise. No amount of momentum helps with a noisy gradient.
- **Lesson:** The problem is not the optimizer. It's that random ternary layers shatter gradients. Float layers are needed to carry gradient signal during training.

### Experiment 2.5: Post-hoc ternary quantization of float-trained weights
- **What:** Trained with float projection/head (as in trix-z), then quantized ALL weights to ternary at save time without STE
- **Result:** 67.40%
- **What went wrong:** Projection and head weren't trained to compensate for quantization. Post-hoc quantization destroys the learned structure.
- **Lesson:** Ternary quantization must be in the training loop (STE) to work. Post-hoc quantization degrades dramatically.

---

## Phase 3: Trit Lattice LSH — zero float

### Experiment 3.1: Class centroid signatures
- **What:** 10 ternary templates = sign(class_centroid - global_mean), classify by dot product argmax
- **Result:** 59.50%
- **What worked:** Proved the lattice geometry carries class structure.
- **What didn't:** Sign discards magnitude. One template per class is too coarse.
- **Lesson:** Direction-based classification is weak (~60%).

### Experiment 3.2: Pairwise signatures
- **What:** 90 ternary templates = sign(centroid_i - centroid_j) for all pairs
- **Result:** 60.01%
- **What didn't:** Pairwise centroid differences are redundant with centroid-vs-global differences. More templates of the same quality don't help.
- **Lesson:** Template QUALITY matters more than quantity.

### Experiment 3.3: Random ternary projections + L1 centroid
- **What:** 256 random ternary vectors, project all images, classify by L1 nearest centroid in projection space
- **Result:** 79.74%
- **What worked:** Random projections + L1 distance. The projection decorrelates dimensions; L1 uses all dimensions independently.
- **Lesson:** **Distance beats direction.** This was the first major breakthrough. L1 in projection space (80%) crushed dot-product classification (60%).

### Experiment 3.4: Single-trit routes in projection space
- **What:** sign(centroid_diff) as scoring template in 256-dim projection space
- **Result:** 59.44%
- **Lesson:** Direction-based classification fails even in projection space.

### Experiment 3.5: Multi-trit routes (MTFP4 weighted) in projection space
- **What:** MTFP4(centroid_diff) × projected_image, weighted dot product
- **Result:** 58.87%
- **What didn't:** Adding magnitude to one direction doesn't help. The problem is one-direction-per-class, not resolution.
- **Lesson:** Multi-trit routes don't help for direct classification.

### Experiment 3.6: Scaling projections (256 → 2048) + L1 centroid
- **Result:** 80.12% → 80.46% → 81.14% → 81.40%
- **Lesson:** Diminishing returns. The classifier (10 centroids) is the ceiling, not the projection.

### Experiment 3.7: Two-stage: L1 → pairwise ternary refinement
- **What:** L1 coarse routing to top-K, then pairwise ternary dot-product voting
- **Result:** 65.63% (top-3), 60.58% (top-5) — both WORSE than L1-only
- **What went wrong:** Direction-based pairwise templates override good L1 distance decisions.
- **Lesson:** **Refinement hurts when the refining classifier is weaker.**

### Experiment 3.8: Two-stage: L1 proj → pixel-space L1 refinement
- **What:** L1 in projection space → top-K, then L1 in pixel space among finalists
- **Result:** 73.71% (top-3) — WORSE than projection-space L1
- **What went wrong:** Pixel-space L1 (66.85%) is weaker than projection-space L1 (81.40%). Using a weaker space to refine a stronger one degrades accuracy.
- **Lesson:** **The projection IS the intelligence.** It creates a better representation than raw pixels.

---

## Phase 4: k-NN on the trit lattice

### Experiment 4.1: k-NN in random-projection space (256-dim, L1)
- **Result:** k=1: 96.09%, k=3: 96.41%, k=5: **96.50%**, k=7: 96.34%
- **What worked:** k-NN uses all 60,000 training images as classifiers, not 10 centroids. Massive capacity increase.
- **Lesson:** **The representation was always good enough.** The 81% ceiling was the centroid classifier, not the projection.

### Experiment 4.2: k-NN tuning (512 proj, L2, varying k)
- **Best:** k=5, L2, 512 proj: **96.79%**
- **What worked:** More projections (+0.29%), L2 distance (+0.15%).
- **Lesson:** L2 penalizes large deviations, helping confusable pairs (4/9, 7/1).

### Experiment 4.3: Pixel-space k-NN (no projection)
- **Result:** 96.88% (L2, k=5)
- **What worked:** Pixel space is slightly BETTER than projection space for k-NN (unlike for centroids).
- **Lesson:** For k-NN, the projection's decorrelation advantage is small because k-NN compares to specific images, not blurry averages.

### Experiment 4.4: Integer deskewing (simple whole-pixel shear)
- **Result:** 96.88% → **97.59%** (deskewed pixel k=5 L2), **97.61%** (k=3)
- **What worked:** Integer image moments + horizontal shear. Corrects digit slant. +0.71% gain.
- **Lesson:** **Fix the geometry first.** Skewed digits look like different digits. Deskewing removes the geometric variation, letting L2 distance measure the actual digit shape.

### Experiment 4.5: "Improved" deskewing (subpixel interpolation + centering)
- **Result:** 95.66% — **1.95% WORSE** than simple shear
- **What went wrong:** Centering shifts already-centered images incorrectly. Subpixel interpolation creates non-standard intensity values that change L2 distances.
- **Lesson:** **Simpler is better for preprocessing.** The simple whole-pixel shear preserves the pixel distribution that k-NN relies on. "Smoother" processing can destroy the structure k-NN needs.

### Experiment 4.6: Ensemble (raw + deskewed k-NN)
- **Result:** 97.61% — identical to deskewed alone
- **What went wrong:** The "prefer deskewed on disagreement" rule means the ensemble IS the deskewed classifier.
- **Lesson:** A proper ensemble needs independent voting, not a hierarchy.

---

## Summary table

| # | Method | Accuracy | Float | Key insight |
|---|---|---|---|---|
| 1.1 | M4T inference (Hamming bug) | 65.53% | training | Wrong routing algorithm |
| 1.2 | M4T inference (fixed) | 94.92% | training | MTFP rounding costs 0.50% |
| 1.3 | M4T inference (peak weights) | 97.46% | training | Substrate matches float |
| 2.1-2.4 | All-ternary STE (any optimizer) | 11.35% | training | Random ternary shatters gradients |
| 2.5 | Post-hoc ternary quantization | 67.40% | training | Quantization must be in the loop |
| 3.1 | Centroid signatures | 59.50% | **zero** | Lattice geometry is real |
| 3.2 | Pairwise signatures | 60.01% | **zero** | Quality > quantity |
| 3.3 | Random proj + L1 centroid | 79.74% | **zero** | Distance beats direction |
| 3.4 | Single-trit routes (proj space) | 59.44% | **zero** | Direction fails everywhere |
| 3.5 | Multi-trit routes (MTFP4) | 58.87% | **zero** | Magnitude on one direction doesn't help |
| 3.6 | Scaling projections (2048) | 81.40% | **zero** | Classifier is the ceiling |
| 3.7 | L1 → pairwise refine | 65.63% | **zero** | Weaker refiner hurts |
| 3.8 | L1 proj → pixel refine | 73.71% | **zero** | Projection IS the intelligence |
| 4.1 | k-NN proj space (256, L1) | 96.50% | **zero** | Representation was always good |
| 4.2 | k-NN proj space (512, L2) | 96.79% | **zero** | L2 + more proj helps |
| 4.3 | k-NN pixel space (raw) | 96.88% | **zero** | Pixel > proj for k-NN |
| 4.4 | k-NN pixel deskewed (k=3 L2) | **97.61%** | **zero** | **Fix geometry first** |
| 4.5 | k-NN subpixel deskew | 95.66% | **zero** | Simpler preprocessing wins |
| 4.6 | Ensemble raw + deskewed | 97.61% | **zero** | Needs proper voting |
| 4.7 | Multi-channel naive (pixel+grad+topo, equal L2) | 96.86% | **zero** | Incommensurate channels poison L2 |
| 4.8 | Per-channel weighted L2 (6 weight combos) | 97.55% best | **zero** | Topo doubles 4↔9 confusion; grad adds nothing |

---

## Phase 5: Multi-channel features (SSTT-inspired)

### Experiment 5.1: Gradient channels + flood-fill topology (naive L2)
- **What:** Features = [pixel(784), h_grad(784), v_grad(784), topo(1)] = 2353 dims, unit-weighted L2
- **Result:** 96.86% (k=3), 96.69% (k=5) — **both worse than 97.61% baseline**
- **What went wrong:** Grad has 2x the dimensions of pixel, so unweighted L2 is dominated by gradient matching rather than intensity. Topo adds a cliff in feature space.
- **Lesson:** Concatenating channels into a single L2 is only sound when channels have commensurate magnitudes AND comparable discriminability per dim. Neither held.

### Experiment 5.2: Per-channel weighted L2 sweep
- **What:** Weighted squared-L2 per channel: `total = w_pix*d_pix + w_grad*d_grad + w_topo*d_topo`. Swept 6 configs including pixel-only (baseline), pixel+topo, pixel+grad/4, pixel+grad/8, and combined.
- **Result:** Best = 97.55% (w_pix=8, w_grad=1, grad at 1/8 weight). All configs worse than 97.61% pixel-only baseline.
- **Key diagnostic:** Topo feature doubles 4↔9 confusion (baseline 16 → topo configs 31). Flood-fill enclosed-region count flips between 0 and 1 on thin-stroke 4s and 9s, creating a cliff in feature space that L2 reads as "definitely different digit."
- **What worked:** Grad at w=1/8 is nearly neutral (−0.06). Doesn't help, doesn't hurt much.
- **Lesson:** Gradients are already implicit in pixel L2 on deskewed images. Topology via flood-fill is too brittle on noisy handwriting. The 97.61% ceiling is not broken by naive channel addition even with careful weighting.

---

## The five laws of the trit lattice (empirical)

Established across 20 experiments:

1. **Distance beats direction.** L1/L2 classifiers (67-97%) consistently outperform dot-product classifiers (58-60%) by 10-40 points. True at every scale tested.

2. **The projection IS the intelligence (for centroids).** Random ternary projection improves centroid classification by 14 points (67% → 81%). It destroys spatial correlations that obscure class structure.

3. **The projection doesn't help for k-NN.** Pixel-space k-NN (96.88%) matches projection-space k-NN (96.79%). For k-NN, the decorrelation advantage is marginal because k-NN compares to specific images.

4. **The classifier is the ceiling, not the representation.** 10 centroids saturate at 81%. 60,000 neighbors (k-NN) reach 96-97%. Same representation, 16-point gap.

5. **Fix the geometry first.** Deskewing (integer shear) adds 0.71% to k-NN. Subpixel "improvement" HURTS by 1.95%. Simple geometric preprocessing > complex feature engineering.

---

## What worked (rank ordered by impact)

1. **k-NN instead of centroids:** +15 points (81% → 96.5%). The single biggest gain. More reference points = more capacity.
2. **Integer deskewing:** +0.71 points (96.88% → 97.59%). Cheap, effective, fixes the right problem (geometric misalignment).
3. **Random ternary projection (for centroids):** +14 points (67% → 81%). Decorrelation makes L1 meaningful.
4. **L2 instead of L1:** +0.15-0.45 points. Penalizes large deviations in confusable pairs.
5. **More projections:** +1.3 points (80% → 81.4%) for centroids. Diminishing returns.
6. **Proper routing (dot product, not Hamming):** +29 points (65% → 95%). Not a "gain" — fixing a bug.

## What didn't work

1. **Direction-based classification (dot product/templates):** 58-60% everywhere. One direction per class is too coarse.
2. **Multi-trit routes for classification:** 58.87%. Magnitude on one direction doesn't help.
3. **Pairwise refinement:** ALWAYS degraded accuracy. Weaker classifiers override stronger ones.
4. **All-ternary training from random init:** 11.35%. Gradient signal dies in random ternary layers.
5. **Subpixel deskewing + centering:** 95.66% — 2 points worse than simple shear. Centering hurts; subpixel changes distributions.
6. **Ensemble (naive):** No gain. A hierarchy isn't an ensemble.

## What didn't work (additions from Phase 5)

7. **Raw gradient channels concatenated into L2:** −0.75 points at equal weight, −0.06 at 1/8 weight. Gradients are already implicit in pixel L2 after deskewing.
8. **Flood-fill enclosed-region topology:** 4↔9 confusion doubles (16 → 31). Hole count flips on thin-stroke digits, creating cliffs in feature space.

## What we still don't know

1. Whether **tangent distance** (invariant to rotation/scale/thickness) can close the 0.60% gap to 98.21%. Classical route to 98.5%+ on k-NN MNIST.
2. Whether **per-pair binary classifiers** — small ternary projections trained to separate only the confused pairs (4↔9, 2↔7, 8↔5) and applied only when baseline k-NN returns that pair — can shave the remaining 239 errors.
3. Whether **integer PCA** projections would help k-NN (they help centroids by ~4% in the literature).
4. Whether **multi-layer ternary hashing** (project → sign → re-project) adds useful capacity.
5. Whether 98.21% is achievable at all with k-NN on raw pixels + simple preprocessing, or if it requires a fundamentally different approach.
6. Whether the 97.61% zero-float result is the ceiling for this class of methods. Phase 5 strongly suggests it's a ceiling for *naive* channel addition — but not necessarily for invariant distances.
