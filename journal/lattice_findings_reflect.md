# Reflections: What the Experiments Told Us

---

## The "why" ladder

1. **Why does the random projection help?** Because it breaks spatial correlations that dominate L1 in pixel space.
2. **Why do spatial correlations hurt?** Because background pixels co-vary across writing styles — their joint variation is large but class-irrelevant.
3. **Why doesn't the projection help for dot-product classifiers?** Because dot product collapses all projected dimensions to one score per class — it can't exploit the decorrelation.

The root: **L1 distance after random projection works because each projected dimension carries roughly equal, decorrelated information, and L1 weighs them equally.** It's the combination of decorrelation (from the projection) and uniform weighting (from L1) that creates the improvement. Neither alone is sufficient.

---

## Core insight

> **Learning on the trit lattice is finding what to destroy. The random projection destroys spatial correlations. PCA destroys them optimally. The sign nonlinearity destroys magnitude. Each act of destruction removes noise that obscures class structure. What survives is the signal.**

This is the inverse of float-trained networks, where learning is adding capacity (more parameters, more layers, more nonlinearities). On the trit lattice, capacity is inherently limited (each weight is 3 states). The path to accuracy is not more capacity — it's better destruction. Remove the right information and what's left is the answer.

---

## Resolved tensions

### Tension A — depth vs width

**Resolution: try both, they're complementary.**

Width (sub-centroids): adds classifier capacity at the current representation level. Expected: 81% → 85-90%. No new architecture needed.

Depth (multi-layer with sign): adds representation capacity. The sign nonlinearity creates a new feature space after each projection. Expected: unknown, but the theory (deeper random projections → better distance preservation in the long run) supports it.

They compose: multi-layer projection to create a rich representation, then sub-centroids to classify finely in that space. Neither alone reaches the ceiling; together they might.

**Priority:** Width first (sub-centroids) because it's simpler and the expected payoff is known. Depth second (multi-layer) because it's exploratory.

### Tension B — random vs data-dependent

**Resolution: random first, data-dependent as a refinement.**

Random projections at 81% are a strong baseline. Integer PCA should reach ~85%. But PCA has a higher implementation cost (covariance matrix + power iteration + deflation). The payoff is 4 points. Worth doing after sub-centroids (which might get 5-10 points for less implementation effort).

The deeper insight: data-dependent projections are "learning on the lattice" — the projection matrix is derived from the data's covariance structure using integer arithmetic. This is the FIRST actual learning in the zero-float framework (everything before was random + statistics). If it works, it proves that lattice-native learning produces better projections than random.

### Tension C — capacity vs generalization

**Resolution: k-NN is the ceiling test, not the architecture.**

k-NN tells us how good the representation IS. If k-NN in projection space gets 95%, the representation is already excellent and we just need a better classifier (sub-centroids, tree, etc.). If k-NN only gets 85%, the representation itself needs improvement (better projections, more layers).

Run k-NN once as a diagnostic. Don't build an architecture around it — it's O(n) per query and doesn't deploy.

---

## Hidden assumptions surfaced

### Assumption 1: The classifier ceiling at 81% is the centroid, not the projection

We assumed this because scaling projections (256 → 2048) gained only 1.3 points. But what if 256 projections are ALREADY sufficient and the saturation is evidence that the CLASSIFIER needs improvement, not more projections? k-NN will test this.

### Assumption 2: Sign extraction is a good nonlinearity

Sign maps MTFP → trit. It's extremely coarse — 3 levels of output from millions of input levels. In float networks, ReLU is 2 levels (positive/zero), and it works. Sign is 3 levels (positive/zero/negative). The extra level (negative) might help (it distinguishes "anti-features" from "absent features"). But the magnitude destruction is severe.

Alternative: multi-trit quantization. Map the MTFP range to 4-trit (81 levels) or 5-trit (243 levels) instead of 1-trit (3 levels). This preserves more magnitude while staying on the trit lattice. But then the re-projection after quantization is MTFP×multi-trit, not ternary×ternary — it requires rescaling.

Or: use sign but increase the width. If each sign dimension is low-information (3 levels), compensate with many more dimensions. Project 784 → 2048, sign-extract → 2048 trits, re-project 2048 → 2048, sign-extract → 2048 trits, classify by L1 in 2048-dim trit space.

### Assumption 3: L1 is the right distance metric

L1 (Manhattan distance) treats each dimension independently and linearly. L2 (Euclidean) would give more weight to large deviations. But L2 requires squaring — an MTFP×MTFP operation we're avoiding.

L∞ (Chebyshev, max absolute deviation) is also integer and might work for nearest-centroid: it classifies based on the WORST dimension, not the sum. This would penalize outlier dimensions more heavily.

Hamming distance on the sign-extracted hash is yet another option: count how many ternary hash bits disagree. This is our popcount_dist primitive — designed for exactly this. The LSH framework says Hamming distance on sign hashes approximates cosine similarity in the original space.

Worth testing: L1 vs Hamming in the projection space as the classification metric.

---

## What I now understand

The experimental program has answered the first question definitively: **yes, the trit lattice supports classification with zero float.** The answer is 81%, with a clear path to 85%+ (sub-centroids, integer PCA).

The second question — can it reach 95%+ to match float-trained models? — is open. The answer depends on whether the representation (random ternary projections) is rich enough, or whether multi-layer composition with sign nonlinearity can close the gap.

The third question — can it reach 97%+ to match the float-trained ROUTED model? — is likely no with the current framework. The float model has 398K optimized parameters and 4 layers with GELU. Matching it with zero float would require either much more capacity (very wide and deep ternary networks) or a fundamentally different approach (like k-NN, which is expensive).

But 81% with zero float is already meaningful. And the gap is capacity, not capability.

---

## The four paths, ranked by expected payoff / effort

1. **k-means sub-centroids** (k=10 per class, 100 total). Expected: 85-90%. Effort: ~50 LOC of integer k-means. Ratio: HIGH.

2. **k-NN ceiling test** (k=5 in projection space). Expected: 95%+ (tells us the representation's ceiling). Effort: ~30 LOC. Ratio: HIGH (diagnostic value).

3. **Multi-layer projection** (ternary matmul → sign → ternary matmul → L1). Expected: unknown, maybe 83-86%. Effort: ~40 LOC. Ratio: MEDIUM.

4. **Integer PCA** (power iteration on covariance → ternary eigenvectors). Expected: 85%. Effort: ~150 LOC. Ratio: MEDIUM.
