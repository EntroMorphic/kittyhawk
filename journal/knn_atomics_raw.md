# Raw Thoughts: Breaking 98.21% on the Trit Lattice

## Stream of Consciousness

96.79%. Zero float. 0.62 points from the float reference. The user says we can reach >98.21%. That's 1.42 points above where we are and 0.80 points above the float-trained reference (97.41%). Can we beat a float-trained ternary routing model using zero float?

Let me look at the errors. 321 misclassified images out of 10,000. The confusion matrix tells me exactly where they are:

The big error clusters:
- 8 is the worst digit: 63 errors. It gets confused with 3 (15), 5 (16), and several others. The 8's top and bottom loops look like pieces of other digits.
- 9 is next: 50 errors. Confused with 4 (10), 7 (12). The 9's tail looks like a 7 or 4.
- 2 gets 42 errors: confused with 7 (14). The curved stroke vs angled stroke.
- 7 gets 39 errors: confused with 1 (21!). The horizontal crossbar distinguishes them but random projections may not capture it.
- 4 gets 38 errors: confused with 9 (22). Classic — open top vs closed top.
- 3 gets 37 errors: confused with 5 (17). Curve direction.

The "easy" digits: 0 (7 errors), 1 (2 errors), 6 (14 errors). These have distinctive shapes that random projections capture well.

So the errors concentrate on STRUCTURAL confusions between digit pairs. These are cases where the pixel patterns overlap significantly and random projections can't distinguish them. The projections are random — they don't know that the top-right corner distinguishes 4 from 9, or that a horizontal stroke distinguishes 7 from 1.

What if the projections WEREN'T random? What if some of them were specifically designed to separate confusable pairs?

For the 4/9 pair: digit 4 has an open top and a vertical stroke on the right. Digit 9 has a closed top loop. A projection that emphasizes the top-center pixels (where the loop is) and de-emphasizes the bottom (where both digits are similar) would separate them.

How to find such a projection in zero float:
1. Take all training 4s and all training 9s
2. Compute the mean 4 and mean 9 on the lattice
3. `sig_4vs9 = sign(mean_4 - mean_9)` → ternary hyperplane optimized for this pair
4. Add this signature to the projection matrix

We tried this in the pairwise experiments and it didn't help as a CLASSIFIER. But as an additional FEATURE DIMENSION for k-NN, it might help. The difference: before, we used the pairwise signature as a scoring function (dot product → argmax). Now, we add it as an extra projection dimension that k-NN can exploit for distance computation.

With 512 random projections + 45 pairwise projections = 557 total dimensions. The pairwise projections would make the k-NN distance more sensitive to pair-specific differences, without replacing the random projections that capture general structure.

This is DATA-DEPENDENT PROJECTION for k-NN features, not for classification. Different use case, different expected outcome.

But wait — the pairwise signatures are binary (sign of centroid difference). They lose magnitude. What if instead of sign, I used a THRESHOLDED ternary: only set +1/-1 for pixels where the centroid difference is LARGE (above a threshold), and 0 for pixels where it's small or ambiguous? This would focus each pairwise projection on the MOST DISCRIMINATIVE pixels for that pair, ignoring noisy pixels.

Or even better: what if I computed the pairwise projections in the RANDOM PROJECTION SPACE instead of pixel space? The random projection has already decorrelated the dimensions. Pairwise signatures in the projected space might be more discriminative than in pixel space.

Let me think about the k-NN algorithm itself. We're doing brute-force k-NN: compare every test image to all 60,000 training images. This is O(n_train × n_test × n_proj). With n_proj=512, that's 307 billion ops. It ran in reasonable time because the ops are integer add/subtract/compare on int32 values — fast on M4.

Can we make k-NN faster? Yes: use the LSH framework for APPROXIMATE nearest neighbors. Hash each image with ternary signatures → group into buckets → only compare within the same bucket. This reduces the search space from 60,000 to maybe 100-1000 per query. But accuracy might drop with approximate search.

For now, brute-force k-NN at 96.79% is our best. To get to 98.21%, I need ~144 more correct predictions (out of 321 errors). That's fixing 45% of the remaining errors.

The 4/9 pair alone accounts for 32 errors. Fixing half of those = 16 more correct.
The 7/1 pair accounts for 21 errors. Fixing half = 10 more.
The 3/5/8 cluster accounts for 48 errors. Fixing a third = 16 more.
That's 42 more correct → 96.79% + 0.42% = 97.21%. Not enough.

To get to 98.21%: need 163 of 321 errors fixed. That's more than half. Every major confusion pair needs to be substantially reduced.

Approaches:
1. More projections (1024): ~0.1-0.2% gain based on scaling curve
2. Data-dependent projections (pairwise features for k-NN): ~0.3-0.5%?
3. Better k-NN: larger k with fixed L2 weighted voting (fix the bug): ~0.1%
4. Multi-scale projections: project at DIFFERENT random seeds and CONCATENATE the projection spaces. Each seed captures different random structure; concatenation gives the k-NN more dimensions to work with.
5. Local projections: for each test image, identify the top-K candidates from the first pass, then re-rank using PAIR-SPECIFIC projections that distinguish only between those candidates. This is the two-stage approach, but using L1/L2 in a pair-specific projection space instead of a general projection space.

Wait — idea 5 is interesting. The two-stage approach failed before because we used pairwise ternary templates as a DOT-PRODUCT classifier. But what if we used pairwise projections as ADDITIONAL FEATURES for a SECOND k-NN pass?

Stage 1: random projection → k-NN → top-20 candidates
Stage 2: for each candidate pair (ci, cj) among the top-20, project the test image using the ci-vs-cj pairwise projection. The candidate with the most pairwise k-NN wins.

But this is getting complex. Let me think about what's simplest.

Actually, the simplest thing that might get a big jump: RUN K-NN IN PIXEL SPACE. Not projection space. Raw 784-dim L1 or L2 nearest neighbors on the MTFP pixel values. k-NN in pixel space on MNIST gets 96.9% (L2) to 97.2% (L1) in the literature. Our random-projection k-NN gets 96.79%. The pixel-space k-NN might match or exceed it.

Wait — we showed that pixel-space L1 CENTROID (66.85%) is much worse than projection-space L1 centroid (81%). But that was for CENTROIDS. For k-NN, the picture is different: k-NN uses the actual training images (all 60,000), not 10 centroids. The correlation structure that hurts centroids (background variation dominates) might matter less for k-NN (each neighbor is a specific image, not an average).

Actually, k-NN in pixel space on MNIST is well-studied: L2 k=3 gets 97.17% (LeCun 1998). Our projection-space k-NN gets 96.79%. The pixel-space k-NN is BETTER because the projection loses some fine-grained information. But pixel-space k-NN costs O(60000 × 784) per test image vs O(60000 × 512) for projection space. Similar cost.

So: pixel-space k-NN might push us from 96.79% to 97.17%. Still short of 98.21%.

To go further:
- k-NN with DESKEWING (align digits before comparison): gets ~98.4% in the literature
- k-NN with ELASTIC DEFORMATION distance: gets ~99.0%
- These require image processing that might or might not be zero-float

Deskewing: compute the skew angle of each digit image, rotate to upright. Skew angle can be estimated from the image moments (center of mass, second moments). All of these are integer operations on the pixel values. The rotation requires trigonometry... or does it? For small angles, rotation by integer shear is possible (shift rows by a pixel-dependent amount). This is a geometric operation on the lattice.

But this is getting speculative. Let me focus on what's concrete.

Most concrete path to improvement:
1. Run pixel-space k-NN as a ceiling test (expected ~97.2%)
2. Combine pixel-space and projection-space features (concatenate 784 + 512 = 1296 dims, run k-NN)
3. Add pairwise projections as additional features for k-NN
4. Try integer deskewing

## Questions Arising

- Can pixel-space k-NN beat projection-space k-NN?
- Can combining pixel + projected features beat either alone?
- Does integer deskewing (shear-based) help k-NN accuracy?
- What's the theoretical ceiling for k-NN on MNIST without preprocessing? ~97.2%
- What's the ceiling WITH deskewing? ~98.4%
- Can we implement deskewing with zero float?

## First Instincts

- Try pixel-space k-NN first (30 lines, answers whether the projection helps or hurts for k-NN)
- If pixel > projection: combine both
- If pixel < projection: the projection is doing something useful beyond decorrelation
- Then try adding pairwise confusable-pair features
- Then try deskewing if needed to reach 98.21%
