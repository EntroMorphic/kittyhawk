# Nodes of Interest: Breaking 98.21%

---

## Node 1: The errors are structural, not random

321 errors at 96.79%. They concentrate on specific digit pairs: 4↔9 (32 errors), 7→1 (21), 3↔5↔8 (48). These are cases where the pixel patterns genuinely overlap — open top vs closed top, with vs without crossbar, curve direction. Random projections can't distinguish these because they don't know which pixels matter for each pair.

**Why it matters:** Random projections are pair-agnostic. The remaining errors need pair-aware features.

---

## Node 2: Pixel-space k-NN is a known ceiling

k-NN in pixel space on MNIST: L2 k=3 gets 97.17% (LeCun 1998). Our projection-space k-NN: 96.79%. The projection loses ~0.4% relative to raw pixels for k-NN, unlike for centroids where the projection HELPS by 14 points. For k-NN (which uses actual training images), the correlation structure doesn't hurt as much — each neighbor is a specific image, not a blurry average.

**Why it matters:** Pixel-space k-NN might already beat projection-space k-NN, without any new ideas. It's the simplest improvement to try.

---

## Node 3: Deskewing is the known path to 98%+

The MNIST literature shows that k-NN with deskewing (aligning digits to upright before comparison) reaches 98.4%. Deskewing uses image moments (center of mass, second-order moments) to estimate skew angle, then shears the image. All moment computations are sums of pixel × coordinate products — integer on the lattice.

**Why it matters:** Deskewing is the single biggest accuracy lever for k-NN on MNIST. It's been known since 2002 (Keysers et al). It requires geometric image processing but no float — moments and shears are integer operations.

---

## Node 4: Combined pixel + projection features

Concatenating pixel-space features (784-dim) with projection-space features (512-dim) gives a 1296-dim feature vector. k-NN in this combined space uses BOTH the raw pixel information (precise but correlated) and the projected information (decorrelated but approximate). If the two representations are complementary, the combination should beat either alone.

**Why it matters:** Free accuracy — no new projection, just concatenation. The L1 distance in the combined space naturally weights both signal sources.

---

## Node 5: Pair-specific projections as k-NN features

For confusable pairs (4/9, 7/1, 3/5), compute `sign(mean_4 - mean_9)` etc. as additional projection dimensions. These are ternary feature extractors that focus on the pixels that distinguish specific pairs. As FEATURES (dimensions in the k-NN space), they add pair-specific information that random projections lack. This is different from using them as CLASSIFIERS (which failed at 60%).

**Why it matters:** Targets the specific error clusters. Each pairwise projection is one dimension that k-NN can use to separate a confusable pair.

---

## Node 6: Integer deskewing via shear

Deskewing by horizontal shear: for each row of the image, shift it left or right by an amount proportional to its distance from the center of mass. The shift amount is computed from the image's second-order moments (covariance of pixel intensity × coordinates).

All operations:
- Center of mass: `cx = sum(x * pixel) / sum(pixel)`, `cy = sum(y * pixel) / sum(pixel)` — integer sums and divides
- Second moment: `Mxy = sum((x-cx) * (y-cy) * pixel) / sum(pixel)` — integer
- Shear angle: `tan(θ) ≈ Mxy / Myy` — integer ratio
- Shear: for row y, shift by `round((y - cy) * Mxy / Myy)` pixels — integer

Zero float. The trigonometry is avoided by using the tangent directly as a shift ratio.

**Why it matters:** Deskewing brings structurally similar but skewed digits into alignment, reducing the distance between same-class images and increasing the distance between different classes. In the literature, deskewing adds 1-2 percentage points to k-NN accuracy.

---

## Tensions

### Tension A: Pixel space vs projection space for k-NN

For centroids, projection space wins by 14 points. For k-NN, pixel space might win by ~0.4 points. The optimal for k-NN might be the COMBINATION of both.

### Tension B: More features vs better features

We can add more random projections (incremental gain, ~0.1% per doubling) or add a few targeted pairwise projections (focused gain on specific error clusters). Or we can preprocess the images (deskewing) to make ALL projections more effective.

### Tension C: Complexity vs purity

Deskewing is image processing — it's a geometric operation, fully integer, but it's not a lattice operation. It operates on the IMAGE geometry (spatial coordinates), not the VALUE geometry (MTFP lattice). Is it within the spirit of "pure lattice geometry"?

Resolution: the deskewing operates on integer pixel coordinates using integer moment computations. The sheared image is still MTFP cells on the lattice. The operation is "prepare the data for the lattice" — the lattice classification itself is unchanged. This is data preprocessing, not a change to the classification framework.

---

## Dependencies

- **Node 2 → Node 4**: Pixel-space k-NN result determines whether combining with projection helps or is redundant.
- **Node 1 → Node 5**: Error analysis identifies which pair-specific projections to add.
- **Node 3 → Node 6**: Literature precedent motivates integer deskewing implementation.
- **Node 6 → all**: Deskewing improves the input to EVERY downstream method.
