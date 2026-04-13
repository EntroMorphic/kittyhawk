# Raw Thoughts: Trit Lattice LSH

## Stream of Consciousness

We just realized what we've been building. The M4T routing layer IS locality-sensitive hashing on the MTFP lattice. The ternary signatures are hyperplanes. The popcount distance is hash agreement. The routing decision is bucket assignment. The tile computation is per-bucket transform. Zero multiplies. Pure geometry.

The question now: can we build a complete MNIST classifier using ONLY lattice operations? No float anywhere — not in training, not in inference, not in weight initialization. Everything is integer arithmetic on the trit lattice.

Here's what I think the architecture looks like:

**Phase 1: Compute class geometry on the lattice.**

Take all 60,000 training images. Convert pixels to MTFP19 cells (integer: cell = pixel * SCALE / 255, with integer rounding). For each class (0-9), compute the centroid: sum all images of that class per pixel, divide by count. This gives 10 centroid vectors on the lattice, each 784-dimensional.

Compute the global centroid (mean of all centroids, or mean of all images).

For each class: diff = class_centroid - global_centroid. Sign-extract → ternary signature. This is a hyperplane that separates this class from the average.

These 10 signatures ARE routing templates. They ARE the "learned" part. And they were computed with zero float — just integer sums, integer divides, and sign.

**Phase 2: Build projection templates.**

The input is 784-dim (raw pixels). We need to project to a lower dimension for efficiency. The projection weight matrix can be the set of class signatures themselves — a [10, 784] ternary matrix. Or: we can add more diversity by computing intra-class variation.

For each class, compute the variance along each pixel dimension. Pixels with high variance within a class are "ambiguous" for that class → set to 0 in the signature. Pixels with low variance and high mean → +1. Pixels with low variance and low mean → -1.

This sharpens the signatures: instead of just sign(mean - global_mean), it's:
- +1 if the pixel is reliably bright for this class
- -1 if the pixel is reliably dark for this class
- 0 if the pixel is unreliable (high variance) for this class

**Phase 3: Route and classify.**

For a new test image:
1. Convert to MTFP19 cells (integer)
2. Compute dot(image, sig_c) for each class c — ternary matmul, add/sub/skip
3. The class with the highest score is the prediction

This is a nearest-centroid classifier with ternary templates. On MNIST it should get 80-90% with no training beyond computing means.

To go higher: use the ternary routing architecture. Multiple tiles per class, or tiles that capture sub-patterns (e.g., different ways to write "4").

**Phase 4: Ternary FFN as refinement.**

After the initial ternary projection (class templates), the routed FFN refines the representation. The tile weights can be:
- Random ternary matrices (diverse basis functions)
- Data-derived: for each routing bucket, compute the residual (input - centroid) and find ternary templates for the residual structure

The routing signatures are the class templates (or sub-class templates from clustering). The tile computation is a ternary transform of the residual. The head maps the transformed residual to a refined class prediction.

All of this is integer. All of this is on the lattice.

## What scares me

- Nearest-centroid classifiers typically get 80-85% on MNIST. The gap to 97%+ requires more sophisticated structure.
- The "clustering into sub-patterns" step is where complexity enters. K-means on the MTFP lattice is doable in integer, but convergence and the right number of clusters are open questions.
- I might be oversimplifying. The float-trained model that gets 97% has ~398K parameters with a complex multi-layer architecture. Replacing it with "compute means and take signs" is ambitious.

## What excites me

- The simplest possible version (10 ternary templates from class centroids) can be implemented in 50 lines of C with M4T primitives.
- It requires ZERO training in the traditional sense. Just integer statistics.
- The forward pass is a single ternary matmul: [1, 784] @ [10, 784]^T → [1, 10] → argmax. About 100 cycles.
- If it gets 80%+, it validates the geometric framework. Then we refine.

## First instinct

Start with the simplest thing: class-centroid ternary templates. Measure accuracy. That's the proof of concept for Trit Lattice LSH. If it works at all, the framework is valid and we can build up from there.
