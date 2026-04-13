# Reflections: Breaking 98.21%

---

## The "why" ladder

1. **Why are we stuck at 96.79%?** Because 321 test images are misclassified, concentrated on structural confusions (4/9, 7/1, 3/5/8).
2. **Why can't random projections separate these pairs?** Because random projections are pair-agnostic — they don't know which pixels distinguish 4 from 9.
3. **Why not add pair-specific projections?** We can. As k-NN features, not as classifiers. Each pairwise projection adds one discriminative dimension.
4. **Why not just use pixel space?** We can. Pixel-space k-NN gets ~97.2% in the literature. But it doesn't reach 98.21% either.
5. **What reaches 98%+?** Deskewing. Aligning digits before comparison. Known since 2002.

---

## Core insight

> **The remaining errors are GEOMETRIC, not STATISTICAL. Digits 4 and 9 overlap in pixel space because one is a skewed version of the other. No amount of projection or classification can fix a geometric misalignment — you have to fix the geometry first. Deskewing is a geometric operation on the lattice that removes writing-style variation. After deskewing, the same k-NN that gets 96.79% should get 98%+.**

The insight is that we've been optimizing the CLASSIFIER (centroids → k-NN) and the FEATURES (random projections → more projections → L2 distance). But the bottleneck is the DATA REPRESENTATION — skewed digits look like different digits. The fix is at the input, not the classifier.

---

## Resolved tensions

### Tension A — pixel vs projection for k-NN

**Resolution: try both, expect pixel-space to win slightly for k-NN.** The projection's advantage (decorrelation) matters for centroids but less for k-NN (which compares to actual images, not averages). Combined pixel+projection might win over either alone. Run the experiment.

### Tension B — more features vs better features vs better data

**Resolution: better data first (deskewing), then better features (pair-specific projections), then more features (wider projections).** The payoff ordering is: deskewing (~1.5%) > pair-specific projections (~0.5%) > more random projections (~0.2%). Fix the input geometry first, then refine the feature space.

### Tension C — complexity vs purity

**Resolution: deskewing is integer and operates on the lattice.** The image moments are integer sums (pixel × coordinate). The shear is an integer shift per row. The output is still MTFP cells. The spirit of zero-float is preserved — no `float`, no `double`, no `sqrtf`. The only "impurity" is that the operation is spatial (image geometry) rather than value-space (MTFP arithmetic). But spatial operations on integer pixel grids are as native to M4's integer ALU as MTFP arithmetic.

---

## The plan to 98.21%

### Step 1: Pixel-space k-NN baseline

Run k-NN in raw MTFP pixel space (784-dim, L2, k=5). Expected: ~97.2%. This establishes whether projection helps or hurts for k-NN, and gives us the pixel-space ceiling.

**Effort:** 10 LOC (change the feature vector from projected to raw). **Expected gain:** +0.4%.

### Step 2: Combined pixel + projection features

Concatenate raw pixels (784-dim) with random projections (512-dim) → 1296-dim feature vector. k-NN in this combined space.

**Effort:** 20 LOC. **Expected gain:** +0.1-0.3% over the better of pixel-only or projection-only.

### Step 3: Integer deskewing

For each image (train and test):
1. Compute center of mass: `cx = sum(x * pixel) / sum(pixel)`, `cy = ...` (integer)
2. Compute second moment: `Mxy = sum((x-cx)(y-cy) * pixel) / sum(pixel)` (integer)
3. Shear: for each row y, shift by `round((y-cy) * Mxy / Myy)` pixels (integer)

Apply deskewing to ALL images before projection and k-NN.

**Effort:** 60 LOC. **Expected gain:** +1.0-1.5% (from literature: deskewing adds ~1.5% to k-NN on MNIST).

### Step 4: Pair-specific projection features

For the top confusable pairs (4/9, 7/1, 3/5, 8/3, 8/5), compute pair-specific ternary projections from the DESKEWED class centroids. Add these as extra feature dimensions for k-NN.

**Effort:** 30 LOC. **Expected gain:** +0.2-0.5%.

### Expected trajectory

```
Current:                                96.79%
+ pixel-space k-NN:                     ~97.2%
+ combined features:                    ~97.4%
+ deskewing:                            ~98.5%
+ pair-specific projections:            ~98.7%
```

If deskewing alone pushes us past 98.21%, the pair-specific projections are bonus. If it falls short, the pair-specific projections should close the gap.

---

## What could go wrong

1. **Integer deskewing might be less accurate than float deskewing.** The shear computation uses integer division, which introduces rounding. If the rounding causes jitter in the deskewed images, the improvement might be less than the 1.5% from the literature (which uses float). Mitigation: use int64 for the moment computations to maintain precision.

2. **Pixel-space k-NN might NOT beat projection-space k-NN.** If the random projection is genuinely helpful for k-NN (not just for centroids), pixel-space k-NN might be worse. The combined approach hedges this risk.

3. **The 98.21% target might require more than deskewing + features.** If the remaining errors after deskewing are truly ambiguous (illegible handwriting), no feature engineering can fix them. But the literature shows 98.4% with k-NN + deskewing alone, so 98.21% should be achievable.

---

## What I'm confident about

- Integer deskewing is implementable in zero float (~60 LOC)
- k-NN in combined pixel+projection space should match or beat either alone
- Deskewing is the single biggest lever — it's been validated in the literature for 20+ years
- 98.21% is achievable with deskewing + k-NN + the existing projection framework
- Every operation stays on the MTFP lattice with integer arithmetic
