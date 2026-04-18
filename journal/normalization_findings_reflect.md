---
date: 2026-04-17
phase: REFLECT
topic: Per-image normalization and what the representation sweep reveals
---

# Normalization findings — REFLECT

---

## Core insight

The RAW phase found the mechanism: low-contrast images produce
degenerate signatures because projection values fall below τ.
NODES mapped the implications. REFLECT finds the deeper
structure.

**The structural zero was being STOLEN by contrast deficiency.**

In the trit lattice, the zero state has a specific meaning:
"this projection direction does not discriminate this image."
It's the base-3 advantage — the ability to say "I don't know"
on a per-dimension basis, which base-2 cannot.

But on low-contrast CIFAR-10 images, the zero state means "my
signal was too weak to cross the threshold." The projection
MIGHT discriminate — but we can't tell because the signal is
buried in the noise floor. The zero state is occupied by
MEASUREMENT FAILURE, not by genuine uninformativeness.

This is exactly the §18 emission-coverage criterion from the
substrate spec: the three output states must be populated by
the input distribution they're designed for. When contrast
variation causes the zero state to absorb measurement failures,
the emission coverage is broken — the zero state is over-
represented for the wrong reason.

Normalization restores emission coverage. After normalization:
- +1 means "this direction sees positive signal" (genuine)
- -1 means "this direction sees negative signal" (genuine)
- 0 means "this direction doesn't discriminate" (genuine)

All three states are populated by the INPUT STRUCTURE, not
by contrast artifacts. The trit lattice can now function as
designed.

## T1 resolved: full normalization (mean + variance)

Although balanced ternary projections cancel the mean bias
at the PROJECTION level, the τ threshold is calibrated on the
TRAINING distribution's projection magnitudes. If the test
query has systematically different magnitude (lower contrast),
the same τ produces different trit distributions. Full
normalization ensures that train and test projection
magnitudes follow the same distribution.

Zero-mean matters for τ calibration even though balanced
weights cancel it in the dot product. The dot product value
|w⋅x| has different magnitude when x is centered vs offset.
The τ is set at the density-th percentile of |w⋅x| from the
training sample. If test x has a different offset, the
percentile is wrong.

## T2 resolved: implement in the dataset loader

Add `glyph_dataset_normalize(ds)` alongside `glyph_dataset_deskew(ds)`.
Called at load time, before any projection. Applied to both
train and test.

Implementation in MTFP integer arithmetic:
1. Per-image mean: int64 sum / N_pixels → m4t_mtfp_t
2. Per-image centered values: x[j] - mean (m4t_mtfp subtraction)
3. Per-image variance: int64 sum of (x[j] - mean)^2 / N_pixels
4. Integer square root of variance → stddev
5. Scale: x_centered[j] × target_scale / stddev

For step 4, use a simple Newton's method isqrt. For step 5,
the division can be done as multiply-by-reciprocal using
integer arithmetic: x × (target_scale / stddev) where the
ratio is precomputed per image.

The target_scale should preserve MTFP range: use MTFP_SCALE
so that ±1σ maps to ±MTFP_SCALE. Values beyond ±3σ clip to
±3×MTFP_SCALE (within int32 range).

## T3 resolved: re-run the brute-force sweep after normalization

The N_PROJ=64 peak was measured on unnormalized data. After
normalization, the landscape might change — wider N_PROJ could
become useful because the additional trits now carry signal
instead of being zero.

But this is a SECOND experiment. First: verify that normalization
lifts the LSH from 37% to 43%+ at N_PROJ=16 M=64. Then: re-run
the brute-force sweep on normalized data to find the new optimal
configuration.

## What I now understand

1. **The 36% ceiling was a CONTRAST ceiling, not a projection
   ceiling.** The projections are fine. The input was preventing
   them from working. Low-contrast images produce degenerate
   signatures that collide regardless of class.

2. **Normalization restores the trit lattice's emission coverage.**
   The zero state was occupied by measurement failure; after
   normalization, it's occupied by genuine uninformativeness.
   This is the structural zero working AS DESIGNED.

3. **The fix is a ONE-LINE change** in the consumer: add
   `glyph_dataset_normalize(&ds)` after loading. The entire
   downstream architecture (projections, signatures, bucket
   index, multi-probe, resolvers, GSH, cascade) benefits
   automatically because the signatures are now richer.

4. **This compounds with everything.** Multi-table composition
   adds ~3pp on top of the single-table ceiling. k-NN adds
   ~1.7pp. Multi-resolution re-rank adds ~2.5pp. GSH agreement
   lifts the confident subset by ~9pp. All of these operate
   on the SIGNATURE quality, which normalization directly
   improves.

5. **42.8% is the new floor, not the ceiling.** That's the raw
   k-NN L1 baseline on normalized pixels. Our routing
   architecture consistently MATCHES or slightly exceeds the
   k-NN baseline. With normalization lifting the baseline to
   42.8%, the routing should reach 43-45%. With GSH agreement
   filtering the confident subset, 50%+ is plausible.
