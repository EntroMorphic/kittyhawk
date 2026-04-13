# Synthesis: Next Phase of Trit Lattice LSH

---

## One-line answer

**Learning on the trit lattice is finding what to destroy. The projection destroys noise. Sign destroys magnitude. L1 measures what survives. More capacity = finer measurement, not more destruction.**

---

## What the experiments proved

Three laws of the trit lattice, established empirically:

**Law 1: Distance beats direction.** L1 distance classifiers (67-81%) consistently outperform dot-product classifiers (58-60%) by 10-20 points, regardless of template quality or quantity. Dot product collapses all dimensions to one score; L1 uses all dimensions independently.

**Law 2: The projection IS the intelligence.** Random ternary projection improves L1 classification by 14 points over raw pixel space (67% → 81%). The projection decorrelates dimensions, making L1 distance meaningful. The improvement comes from DESTROYING spatial correlations, not from adding information.

**Law 3: The classifier is the ceiling.** Scaling projections (256 → 2048) saturates at ~81%. The 10-centroid classifier can't partition 10 overlapping classes finely enough. The representation is already good; the classifier needs more capacity.

---

## The next four experiments, in priority order

### Experiment A: k-NN ceiling test

**Purpose:** Determine the ceiling of the random-projection representation. If k-NN gets 95%, the representation is excellent and we only need a better classifier. If k-NN gets 85%, the representation needs improvement.

**Method:**
1. Project all 60,000 training images to 256-dim (random ternary matmul)
2. For each test image: compute L1 distance to all 60,000 training projections
3. k=5 nearest neighbors, majority vote

**Expected:** 95%+ (MNIST k-NN in Euclidean space gets ~97%; L1 in random-projection space should be close)

**Cost:** O(60,000 × 256) L1 comparisons per test image × 10,000 test images. ~154 billion integer ops. At 4 GHz, ~38 seconds. Tractable.

**Effort:** ~30 LOC. Zero new M4T primitives needed.

### Experiment B: k-means sub-centroids

**Purpose:** Break the 10-centroid ceiling with per-class sub-clusters.

**Method:**
1. Project training images to 256-dim
2. For each class: run integer k-means (k=10) on the projected training images of that class
3. Result: 100 sub-centroids (10 per class)
4. Classify: L1 nearest sub-centroid → inherit the sub-centroid's class label

**Integer k-means:** assign each point to nearest centroid (L1, integer), recompute centroids (integer mean), repeat for T iterations. All integer.

**Expected:** 85-90%. Each class is partitioned into sub-clusters (different writing styles → different centroids). Confusable pairs (3 vs 5, 4 vs 9) get separate sub-centroids for their distinguishing variants.

**Effort:** ~50 LOC.

### Experiment C: Multi-layer ternary hashing

**Purpose:** Test sign extraction as a nonlinearity for depth.

**Method:**
```
Input [784] MTFP19
  → Layer 1: ternary matmul [256, 784] → 256-dim MTFP19
  → Sign extraction → 256-dim trits
  → Layer 2: ternary matmul [256, 256] → 256-dim MTFP19
  → L1 nearest centroid (10 centroids in layer-2 space)
```

Both ternary matmuls use random ternary weights. The sign extraction between them is the nonlinearity. The centroids are computed from training images passed through both layers.

**Expected:** Unknown. If sign-as-nonlinearity enables useful depth, accuracy should exceed the single-layer 81%. If sign destroys too much information, it might hurt.

**Effort:** ~40 LOC.

### Experiment D: Integer PCA projections

**Purpose:** Replace random projections with data-dependent ones for optimal decorrelation.

**Method:**
1. Compute the covariance matrix of training images in MTFP: `C[i,j] = mean(x[i] * x[j]) - mean(x[i]) * mean(x[j])`. Uses int64 accumulators.
2. Power iteration: start with a random vector, repeatedly multiply by C, normalize (isqrt). Extract top-K eigenvectors by deflation.
3. Quantize each eigenvector to ternary: `sign(v[d])`
4. Use as projection matrix instead of random

**Expected:** ~85% (matches PCA + L1 centroid results in the literature)

**Effort:** ~150 LOC. The covariance matrix is 784×784 = 614K int64 values (~4.7 MB). Power iteration converges in ~20 iterations per eigenvector.

---

## Architecture roadmap

```
Phase 1 (now):     random projection + L1 centroid           → 81%
Phase 2 (next):    + k-means sub-centroids (100 total)       → 85-90%
Phase 3 (after):   + integer PCA or multi-layer hashing      → 88-92%
Phase 4 (future):  + routing (different projections per       → 92-95%?
                     input region) + tile refinement
```

All zero float. All integer. All on the trit lattice.

---

## What this framework eliminates

Compared to the float-trained architecture:

| Component | Float path | Trit Lattice LSH |
|---|---|---|
| Forward matmul | Dense MTFP×MTFP (__int128) | Ternary matmul (add/sub/skip) |
| Nonlinearity | GELU LUT (5.4 MB .rodata) | Sign extraction (free) |
| Normalization | LayerNorm (isqrt, 3 passes) | Not needed (projection decorrelates) |
| Training | Float shadow weights + STE + AdamW | Integer statistics (mean, k-means) |
| Backward pass | Full backprop through all layers | Not needed |
| Optimizer state | Momentum + variance per parameter | Not needed |
| Weight storage | MTFP19 cells (4 bytes/param) | Trits (2 bits/param, packed) |

The Trit Lattice LSH path is simpler, smaller, faster, and fully integer. The tradeoff is accuracy (81% vs 97%). The roadmap closes the gap through capacity, not complexity.

---

## Honest assessment

The framework works. The 81% result is real, zero-float, and the experimental record is thorough. The path to 85-90% (sub-centroids, integer PCA) is clear and mechanical. The path to 95% (k-NN or very deep multi-layer) is visible but expensive. The path to 97% without float is a genuine open question.

The most surprising thing this cycle surfaced: **the intelligence of the system is destruction, not construction.** The random projection works because it destroys spatial correlations. The sign nonlinearity works (we hypothesize) because it destroys magnitude noise. The L1 distance works because it treats each surviving dimension equally. Every component strips away a different kind of irrelevant variation. What's left is the signal.

This is the opposite of the deep learning paradigm, where intelligence is in the ACCUMULATION of learned transformations. On the trit lattice, intelligence is in the SUBTRACTION of noise. The lattice is too coarse to accumulate subtle learned features. But it's perfectly suited to hash away the irrelevant and measure what remains.

---

## Next action

Run Experiment A (k-NN ceiling test). It's 30 lines and tells us the most important thing: is the representation already good enough, or does it need improvement? Every other decision depends on this answer.
