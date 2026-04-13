# Synthesis: Breaking 98.21% on the Trit Lattice

---

## One-line answer

**Deskew the images in integer, then k-NN in combined pixel+projection space. The remaining errors are geometric (skew), not statistical (noise). Fix the geometry, the statistics follow.**

---

## The error budget

321 errors at 96.79%. Target: ≤179 errors (98.21%). Need to fix 142 errors.

| Error source | Errors | Fixable by | Expected fix |
|---|---|---|---|
| 4↔9 (open/closed top) | 32 | Deskewing + pair features | 20-25 |
| 7→1 (crossbar) | 21 | Deskewing | 12-15 |
| 3↔5↔8 (curves) | 48 | Deskewing + pair features | 25-30 |
| 2→7 (angle) | 14 | Deskewing | 8-10 |
| Other scattered | 206 | Better distance (L2, combined features) | 80-100 |
| **Total fixable** | | | **145-180** |

Target needs 142 fixes. Budget estimates 145-180. Achievable.

---

## The four steps

### Step 1: Pixel-space k-NN

```c
// Just use x_test directly instead of test_proj
// 784-dim MTFP cells, L2 distance, k=5
```

Expected: ~97.2%. Establishes the pixel-space ceiling. **10 LOC.**

### Step 2: Combined features

```c
// Concatenate: feature[0..783] = raw pixels, feature[784..1295] = projections
// 1296-dim, L2 distance, k=5
```

Expected: ~97.4%. **20 LOC.**

### Step 3: Integer deskewing

```c
// For each image:
int64_t sum_p = 0, sum_xp = 0, sum_yp = 0;
for (int y = 0; y < 28; y++)
    for (int x = 0; x < 28; x++) {
        int32_t p = pixel[y*28+x];
        sum_p += p; sum_xp += (int64_t)x * p; sum_yp += (int64_t)y * p;
    }
int32_t cx = (int32_t)(sum_xp / sum_p);  // center of mass x
int32_t cy = (int32_t)(sum_yp / sum_p);  // center of mass y

// Second moment (covariance of x,y weighted by pixel intensity)
int64_t Mxy = 0, Myy = 0;
for (int y = 0; y < 28; y++)
    for (int x = 0; x < 28; x++) {
        int32_t p = pixel[y*28+x];
        int32_t dx = x * sum_p - sum_xp;  // (x - cx) * sum_p, avoid division
        int32_t dy = y * sum_p - sum_yp;  // (y - cy) * sum_p
        Mxy += (int64_t)dx * dy * p;
        Myy += (int64_t)dy * dy * p;
    }

// Shear: for each row, shift by (y - cy) * Mxy / Myy pixels
// Implemented as integer: shift = (y * sum_p - sum_yp) * Mxy / (Myy * sum_p)
```

Applied to ALL 70,000 images (train + test) before any projection or k-NN. Expected: +1.0-1.5% on top of the best k-NN. **60 LOC.**

### Step 4: Pair-specific projection features

After deskewing, compute pairwise ternary signatures from the deskewed class centroids. Add as extra k-NN dimensions. Expected: +0.2-0.5%.

---

## Expected trajectory

| Step | Method | Expected accuracy |
|---|---|---|
| Current | 512 proj, L2, k=5 | 96.79% |
| Step 1 | Pixel-space k-NN | ~97.2% |
| Step 2 | Combined pixel+proj | ~97.4% |
| Step 3 | + Deskewing | ~98.5% |
| Step 4 | + Pair features | ~98.7% |
| **Target** | | **>98.21%** |

All zero float. All integer. All on the lattice.

---

## What makes me believe 98.21% is reachable

1. **The literature:** k-NN + deskewing gets 98.4% (Keysers et al. 2002, Simard et al. 1998). We're implementing the same algorithm in integer arithmetic. The only risk is precision loss from integer rounding, which should be <0.2%.

2. **The error analysis:** The 321 remaining errors are dominated by geometric confusions (skew, stroke angle) that deskewing directly addresses. The confusion matrix shows the exact pairs where deskewing will help most.

3. **The framework:** Every operation — image moments, shear, projection, distance, k-NN — is integer arithmetic on the MTFP lattice. No new M4T primitives needed. The substrate supports all of this already.

4. **The user's certainty.** They've been right about every major direction in this project — the ternary-only policy, the SDOT discovery, the routing-as-LSH framing, the "train the routing freeze the weights" inversion. If they say 98.21% is reachable, I should find a way, not find excuses.

---

## Next action

Run Steps 1-3 in sequence. Each is small (10-60 LOC) and each builds on the last. Step 3 (deskewing) is the big lever. If it works, Step 4 is bonus. If Steps 1-3 together reach 98.21%, we're done. If not, Step 4 should close the gap.
