# Raw Thoughts: What the Experiments Actually Told Us

## Stream of Consciousness

We ran 12 sub-experiments on MNIST with zero float. The results are in. Let me dump what I actually think — not what I hope, not what sounds good, what I think.

The projection is the intelligence. That's the headline. But what does it mean? A random ternary matrix — {-1, 0, +1} values chosen by a PRNG with no knowledge of MNIST, no knowledge of digits, no knowledge of anything — creates a representation where L1 distance classifies digits better than the raw pixels. How?

I keep saying "Johnson-Lindenstrauss" and "decorrelation" but I'm not sure I believe my own explanation. J-L says distances are preserved. But L1 in projection space doesn't just preserve distances — it IMPROVES classification by 14 points over L1 in pixel space. Preservation doesn't explain improvement.

Here's what I think is actually happening: MNIST pixels are highly correlated. Adjacent pixels tend to have similar values. The 784 dimensions are NOT independent — they're a 2D image with spatial structure. L1 distance in pixel space is dominated by the CORRELATED dimensions (background pixels that vary together across writing styles). The random projection BREAKS these correlations. Each projected dimension is a random linear combination of all 784 pixels — it mixes the spatial structure into a flat, decorrelated representation. In this decorrelated space, L1 distance gives equal weight to each dimension, and each dimension carries roughly equal information.

So the projection isn't learning anything. It's DESTROYING structure — specifically, it's destroying the spatial correlations that make pixel-space L1 bad. What's left after the correlations are destroyed is the CLASS-RELEVANT structure, which is more robust than the spatial structure.

This is unsettling. The best thing our system does is destroy information. The "intelligence" is subtraction, not addition.

But wait — if random projection is just decorrelation, then PCA would do the same thing but better (it decorrelates optimally, by definition). PCA on MNIST + L1 nearest centroid gets ~85% in the literature. Our random ternary projection gets 81%. The gap (4%) is the cost of random vs optimal decorrelation. That makes sense.

Can we close that gap without float? PCA requires eigendecomposition of the covariance matrix — that's float arithmetic (square roots, divisions, iterative algorithms). But there are ternary approximations to PCA:
- Compute the covariance matrix in integer (sum of outer products — O(n * d^2) integer ops)
- Find the top eigenvectors approximately using power iteration (repeated matrix-vector multiply — integer)
- Quantize each eigenvector to ternary

Power iteration on the covariance matrix: start with a random vector, multiply by the covariance matrix, normalize, repeat. The dominant eigenvector emerges. All of these operations can be done in MTFP arithmetic. The normalization step is the tricky one — it requires dividing by the vector norm, which is an isqrt operation (which we already have in m4t_isqrt64).

So: integer PCA → ternary eigenvectors → ternary projection matrix → L1 centroid. This should get ~85% with zero float. The projection matrix would be DATA-DEPENDENT instead of random — the first data-dependent projection we've tried.

But I'm getting ahead of myself. Let me think about what else the experiments told us.

Direction vs distance. Every dot-product classifier scored 58-60%. Every L1 classifier scored 67-81%. This is a 20-point gap. Why?

Dot product: `score = sum(template[d] * image[d])`. This is a projection onto ONE direction (the template). All the information orthogonal to that direction is discarded. For 784-dim data, that's throwing away 783 dimensions of information.

L1 distance: `dist = sum(|image[d] - centroid[d]|)`. This uses ALL dimensions independently. Each dimension contributes its own deviation. No information is discarded.

The dot product is a 1-dimensional measurement in a 784-dimensional space. L1 distance is a 784-dimensional measurement. The information gap is enormous.

But here's the thing: the dot product CAN be used well — you just need MANY dot products (many templates), and you need to combine them with a classifier that uses ALL the dot products, not just the one that's largest. The 256 random projections ARE 256 dot products. The L1 centroid in projection space IS using all 256 simultaneously. That's why it works — it's not 1 direction, it's 256 directions combined via L1 distance.

So: dot product as a FEATURE EXTRACTOR (many projections → many scores) + L1 distance as a CLASSIFIER (nearest centroid in the score space) = good. Dot product as a CLASSIFIER (argmax over template scores) = bad.

The architecture insight: ternary matmul is for FEATURE EXTRACTION. L1 distance is for CLASSIFICATION. Don't use the matmul output directly as a class score. Use it as a REPRESENTATION and classify in that space with distance.

This changes how the routing architecture should work:
- OLD: route[t] = sign(dot(x, sig_t)) → binary decision based on one direction
- NEW: route by L1 proximity to tile-specific regions of projection space → distance-based decision using all dimensions

But L1 nearest-centroid routing is just... nearest centroid. It's not "routing" in the trix-z sense. It's classification.

Unless: the routing determines which TILES to apply, and the tiles produce different REPRESENTATIONS, and the final classification is L1 in the tile-specific representation space. That's a genuine routing architecture where different tiles specialize on different input subspaces.

Hmm. I'm going in circles. Let me step back.

What do we actually know?
1. Random ternary projection + L1 centroid = 81%. This is our ceiling with 10 centroids.
2. The ceiling is the classifier (10 centroids), not the representation.
3. To go higher: more centroids, k-NN, or a non-centroid classifier.
4. All of these are integer operations.

What's the simplest path to 85%+?
- k-means in projection space (k=5 per class = 50 sub-centroids) + L1 nearest sub-centroid
- k-means is iterative but integer: assign each training image to nearest centroid, recompute centroids, repeat

What's the simplest path to 90%+?
- k-NN (k=5) in projection space: for each test image, find the 5 nearest training images, majority vote
- This is O(n_train) per test image — slow but correct

What's the simplest path to 95%+?
- I honestly don't know if it's possible with zero float. The float-trained model has 4 layers with nonlinearities. Our zero-float model is effectively a 1-layer linear classifier (random projection + L1). Getting to 95%+ might require nonlinearity, which in the lattice framework means... sign extraction? Multi-layer projection with sign as the nonlinearity?

Let me think about sign as a nonlinearity. After projecting to 256 dims, take the sign of each dimension: +1 if positive, -1 if negative, 0 if zero. This is a ternary quantization — it maps the MTFP representation to a ternary hash code. Then re-project the hash code to a new space and classify there.

```
Layer 1: ternary matmul → MTFP representation
Layer 2: sign extraction → ternary hash code
Layer 3: ternary matmul → new representation
Layer 4: L1 nearest centroid → classify
```

Each sign extraction is a nonlinearity (it's a discontinuous function). The ternary matmul after sign extraction is a ternary×ternary operation — both the input and the weights are trits. This is a pure trit-space computation.

Would this help? The sign extraction discards magnitude (same issue as the centroid signatures). But in the SECOND layer, the input is already a ternary hash code. The second projection operates on trits, not on MTFP values. The second projection's output is a ternary-weighted sum of trits, which is just a COUNT (how many trits agree with each projection direction). Then L1 in this count space might be discriminative.

This is worth trying. It's a multi-layer ternary network with sign as the nonlinearity. Zero float.

Actually wait — this is basically what I described in the REFLECT phase as "multi-layer hashing": project → sign → re-project → sign → classify. Each layer is a ternary matmul + sign extraction. The sign extraction IS the nonlinearity that creates depth.

And it's the simplest possible deep ternary network. No GELU, no LayerNorm, no residual connections. Just: project (ternary matmul), quantize (sign), repeat, classify (L1).

I think this is the next experiment.

## Questions Arising

- Can integer PCA (power iteration on integer covariance) produce better ternary projections than random?
- Does multi-layer projection (ternary matmul + sign extraction × N layers) improve over single-layer?
- What's the accuracy ceiling of k-NN in random-projection space?
- Is sign extraction the right nonlinearity, or should we use something else (threshold at ±median, or multi-trit quantization)?
- What's the minimum depth / width to reach 90%+ with zero float?

## First Instincts

- Try multi-layer hashing first: it tests the "sign as nonlinearity" idea and adds depth.
- k-means sub-centroids second: it adds classifier capacity within the current framework.
- Integer PCA third: it replaces random projections with data-dependent ones.
- k-NN last: it's the brute-force ceiling test.
