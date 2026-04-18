---
date: 2026-04-17
phase: NODES
topic: Per-image normalization and what the representation sweep reveals
---

# Normalization findings — NODES

---

## Node 1 — The τ–contrast interaction is the root cause

Low-contrast images produce small projection values that
fall below τ. Their signatures are degenerate (mostly zero
trits). They collide in the bucket index regardless of class.
The structural zero is being imposed by CONTRAST DEFICIENCY,
not by measurement design.

MNIST doesn't have this problem: pen strokes are always
high-contrast. CIFAR-10's natural images vary enormously
in brightness and contrast.

## Node 2 — Per-image normalization is the fix

Normalize each image to zero-mean and unit-variance before
projection. Every image enters the lattice with equal signal
strength. The τ threshold works correctly for every image.

42.8% k-NN baseline with normalization vs 36.4% without.
The +6.4pp lift is the contrast-equalization gain.

## Node 3 — Balanced ternary projections already cancel brightness

Σ w_i ≈ 0 for balanced {-1,0,+1} weights. So w⋅(x - μ) ≈ w⋅x.
Zero-mean alone should NOT help our LSH much. The gain from
normalization comes from VARIANCE scaling, not mean removal.

This means: the critical operation is dividing each pixel by
the image's stddev, not subtracting the mean.

## Node 4 — Per-image τ adjustment as an alternative

Instead of normalizing pixel values, adjust τ per image:
τ_q = τ × (stddev_q / stddev_ref). Low-contrast images
get lower τ (more trits activate). But training and test τ
must be consistent, so pixel normalization is cleaner.

## Node 5 — The W_f[hidden]=0 connection

NORTH_STAR: the structural zero is the third state that
base-2 ignores. In our system, the zero trit means
"this projection direction doesn't discriminate." That's
the INTENDED meaning.

But on low-contrast CIFAR-10 images, the zero trit means
"this image is too faint to measure." That's a BUG. The
zero state is being occupied for the wrong reason.

Normalization restores the correct semantics: after
normalization, the zero trit means "this direction sees
equal signal in both directions" (genuinely uninformative),
not "this image has low contrast" (measurement failure).

## Node 6 — This explains the N_PROJ=64 peak on CIFAR-10

At N_PROJ=64, each table has 64 trits. On low-contrast
images, most trits are zero (below τ). The EFFECTIVE
N_PROJ for low-contrast images is much less than 64.
More trits (wider N_PROJ) don't help because they're
also zero.

After normalization, all images produce rich signatures
at any N_PROJ. The N_PROJ=64 peak might MOVE — wider
N_PROJ could become useful again because the additional
trits carry signal instead of being zero.

## Node 7 — The normalization composes with EVERYTHING

Every downstream mechanism we built (multi-table, multi-probe,
k-NN, GSH, dynamic cascade, re-rank) operates on signatures.
Better signatures from normalized input propagate through
the entire pipeline. The +6.4pp from normalization in raw
k-NN should compound with the architectural gains.

Expected: normalized LSH at M=64 k=5 should exceed 42%
(the normalized raw-pixel ceiling) because the multi-table
composition adds value ON TOP of the better input.

## Node 8 — MTFP integer normalization

Per-image normalization in MTFP:
1. mean_i = sum(x_train[i*D .. i*D+D-1]) / D (int64 sum, int divide)
2. var_i = sum((x - mean)^2) / D (int64 sum of squares, int divide)
3. stddev_i = isqrt(var_i) (integer square root, Newton's method)
4. x_normalized = (x - mean_i) × scale / stddev_i

The scale factor maps to a target range. If we want the
normalized values to use the full MTFP range, scale = MTFP_SCALE.
The division by stddev requires integer division.

## Node 9 — This might be all we need

If normalization lifts the LSH from 37% to 43%+, and the
GSH agreement filter lifts the confident subset to 50%+,
and the full multi-table + k-NN + multi-resolution adds
another 2-3pp on top, we could hit 50% with NO architectural
changes — just a preprocessing step.

The routing architecture is ALREADY at the pixel-space
ceiling. The ceiling was wrong because the input
representation was wrong.

## Node 10 — Per-channel normalization might be better

Normalizing all 3072 dims together treats RGB as one blob.
Per-CHANNEL normalization (normalize R, G, B independently)
might be better — it equalizes contrast within each color
channel separately.

Or: per-spatial-region normalization (normalize quadrants
independently). But that adds complexity for uncertain
gain. Start with per-image, then per-channel if needed.

## Tensions

**T1:** Full normalization (mean + variance) vs variance-only.
Node 3 says the mean cancellation is already done by balanced
projections. Variance-only might suffice and is simpler.

**T2:** Implementation path — MTFP integer normalization
(mathematically correct but complex) vs approximate
normalization (simpler, might suffice).

**T3:** Does the N_PROJ landscape change after normalization?
The brute-force sweep should be re-run on normalized data
to find the new optimal N_PROJ.
