# Nodes of Interest: What the Experiments Told Us

---

## Node 1: The projection destroys structure, and that's why it works

Random ternary projection doesn't ADD information. It DESTROYS the spatial correlations between pixels. In raw pixel space, adjacent pixels are correlated — background regions vary together across writing styles, dominating L1 distance. The random projection mixes all 784 pixels into each of 256+ output dimensions, breaking the 2D spatial structure into a flat, decorrelated representation. What survives the destruction is class-relevant structure, which is more robust than spatial structure.

**Why it matters:** The "intelligence" of the projection is subtraction, not addition. The best operation in our system is information destruction. This reframes what "learning on the lattice" means: it's about finding the RIGHT information to destroy.

---

## Node 2: PCA is the optimal version of what our random projection does

PCA decorrelates optimally (by definition — it rotates to the eigenbasis of the covariance matrix). Our random projection decorrelates approximately. PCA + L1 centroid gets ~85% on MNIST in the literature. Our random + L1 gets 81%. The 4-point gap is the cost of approximate vs optimal decorrelation.

**Why it matters:** If we could do PCA in integer (power iteration on the integer covariance matrix, then quantize eigenvectors to ternary), we'd get data-dependent projections that should close the gap — ~85% with zero float.

---

## Node 3: Dot product is a feature extractor, not a classifier

Dot product as a CLASSIFIER (argmax over template scores): 58-60%. Dot product as a FEATURE EXTRACTOR (256 projections → 256 scores) + L1 as a CLASSIFIER (nearest centroid in score space): 80-81%.

The dot product works when it produces a REPRESENTATION (many projections used simultaneously) and fails when it produces a DECISION (one template per class, argmax).

**Why it matters:** This separates the architecture into two phases: (1) ternary matmul produces features, (2) L1 distance classifies. They have different roles and shouldn't be confused.

---

## Node 4: The classifier is the ceiling, not the representation

Scaling from 256 to 2048 projections gained only 1.3 points (80.12% → 81.40%). The representation is already good at 256 dims. The ceiling is 10 centroids — too few to partition 10 overlapping class distributions.

**Why it matters:** Adding more projections won't break 82%. To go higher: more centroids per class, k-NN, or a fundamentally different classifier.

---

## Node 5: Sign extraction is the natural ternary nonlinearity

In the lattice framework, there's no GELU (too expensive, requires LUT). The natural nonlinearity is sign extraction: `x → sign(x) ∈ {-1, 0, +1}`. It's free (comparison + conditional), it's ternary (output is a trit), and it's a genuine nonlinearity (discontinuous, saturating).

A multi-layer ternary network would be: `project (ternary matmul) → sign → project → sign → ... → classify (L1)`. Each sign extraction creates depth. Each ternary matmul after sign operates on TRITS, not MTFP — it's pure trit-space computation.

**Why it matters:** This is the simplest deep ternary network. No GELU, no LayerNorm, no residuals. If sign-as-nonlinearity enables depth, the capacity ceiling lifts.

---

## Node 6: k-means sub-centroids are the cheap capacity boost

k-means in projection space: assign each training image to the nearest centroid, recompute centroids, repeat. All integer. With k=5 sub-centroids per class (50 total), the classifier can partition each class into sub-clusters (different writing styles for the same digit). Expected: ~85-90%.

**Why it matters:** This is the simplest path to higher accuracy within the current framework. Same projection, same L1 distance, just more centroids.

---

## Node 7: k-NN is the ceiling test

k-nearest-neighbors (k=5) in projection space: for each test image, find the 5 nearest training images (by L1 distance in projection space), majority vote. Expected: 95%+. Cost: O(n_train) per test image — 60,000 × 256 L1 comparisons. Expensive but tells us the ceiling of the representation.

**Why it matters:** If k-NN in random-projection space gets 95%+, the representation is ALREADY good enough for high accuracy. The gap to 81% is entirely the classifier. If k-NN only gets 85%, the representation itself needs improvement.

---

## Node 8: Integer PCA might close the random→optimal gap

Power iteration on the integer covariance matrix:
1. Compute covariance: `C[i,j] = sum_n(x_n[i] * x_n[j]) / N` — integer (MTFP cells, int64 accumulator)
2. Power iteration: `v_{t+1} = C · v_t`, normalize by isqrt — integer
3. Deflation: `C' = C - lambda * v * v^T` — integer
4. Quantize each eigenvector to ternary: `sign(v)`

The result: ternary projection directions that are data-dependent and optimally decorrelating. Expected improvement: 81% → 85%.

**Why it matters:** This is "learning" on the lattice — the projection matrix is derived from the data's structure, not from a random seed. And it's all integer.

---

## Tensions

### Tension A: Depth vs width (Node 5 vs Node 6)

Multi-layer projection (depth via sign nonlinearity) adds capacity through composition. Sub-centroids (width via k-means clustering) add capacity through finer partitioning. Both are integer. Which helps more for the same compute budget?

### Tension B: Random vs data-dependent projections (Node 1 vs Node 8)

Random projections are simple and work (81%). Data-dependent projections (integer PCA) should work better (85%) but require more compute during "training" (power iteration). The random projection's advantage is simplicity and no dependency on the data distribution. The data-dependent projection's advantage is optimality.

### Tension C: Capacity vs generalization (Node 4 vs Node 7)

k-NN uses ALL training data as the classifier — maximum capacity. But it memorizes. Sub-centroids (k-means) average over clusters — less capacity but more robust to noise. The right capacity level depends on whether the projection-space representation is smooth (centroids suffice) or rough (need more resolution).

---

## Dependencies

- **Node 1 → Node 2**: Understanding WHY random projection works (decorrelation) motivates integer PCA (optimal decorrelation).
- **Node 3 → Node 5**: Dot product as feature extractor motivates multi-layer projection (stack feature extractors with sign nonlinearity).
- **Node 4 → Node 6**: Classifier ceiling motivates sub-centroids.
- **Node 4 → Node 7**: Classifier ceiling motivates k-NN ceiling test.
