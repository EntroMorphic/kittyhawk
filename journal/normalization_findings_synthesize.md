---
date: 2026-04-17
phase: SYNTHESIZE
topic: Per-image normalization and what the representation sweep reveals
---

# Normalization findings — SYNTHESIZE

Executable specification.

---

## What to build

`glyph_dataset_normalize(ds)` — per-image zero-mean and
unit-variance normalization applied at load time, in MTFP
integer arithmetic. Called after load, before any projection.
Analogous to `glyph_dataset_deskew(ds)`.

## Why this is the missing step

The representation sweep showed:
- Raw pixel k-NN: 36.4%
- Normalized k-NN: 42.8% (+6.4pp)
- Our LSH matches raw-pixel ceiling perfectly (37%)

The ceiling is set by the INPUT REPRESENTATION, not by the
architecture. Normalization lifts the ceiling by 6.4pp by
equalizing image contrast so that the τ threshold produces
meaningful trit distributions for every image, not just
high-contrast images.

## Why this is routing-native

Every operation is MTFP integer arithmetic:
- Mean: int64 sum, integer divide
- Subtraction: m4t_mtfp_t element-wise subtract
- Variance: int64 sum of squares, integer divide
- Square root: integer Newton's method
- Scaling: integer multiply and divide

No float. No external computation. Substrate primitives.

## Implementation

In `src/glyph_dataset.{h,c}`:

```c
void glyph_dataset_normalize(glyph_dataset_t* ds);
```

For each image in x_train and x_test:
1. mean = sum(pixels) / n_pixels (int64 sum → int32 divide)
2. pixels[j] -= mean (element-wise subtract)
3. var = sum(pixels[j]^2) / n_pixels (int64 sum of squares)
4. stddev = isqrt(var) (integer Newton's method)
5. If stddev > 0: pixels[j] = pixels[j] * MTFP_SCALE / stddev
   (rescale to target range)

The target scale is MTFP_SCALE so that ±1σ maps to ±MTFP_SCALE.
Values beyond ±3σ are clipped to keep within int32 range.

Integer square root (Newton's method):
```c
static int64_t isqrt64(int64_t n) {
    if (n <= 0) return 0;
    int64_t x = n;
    int64_t y = (x + 1) / 2;
    while (y < x) { x = y; y = (x + n / x) / 2; }
    return x;
}
```

## Testing plan

1. Add glyph_dataset_normalize to the library.
2. Run the multi-table consumer on CIFAR-10 with normalization.
   Expected: ≥42% at M=64 k=5.
3. Run the brute-force N_PROJ sweep on normalized CIFAR-10
   to find the new optimal N_PROJ.
4. Run on MNIST and Fashion-MNIST to verify no regression
   (normalization should be neutral-to-positive on already-
   high-contrast datasets).
5. If ≥42%: add GSH on normalized data, measure agreement
   filter accuracy.

## Go / no-go

**Go:** CIFAR-10 LSH k=5 ≥ 42% after normalization.
**Strong go:** ≥ 45% (normalization + architecture compounds).
**No-go:** < 39% (normalization doesn't help the LSH even
though it helps raw k-NN — would mean the projection is
the remaining bottleneck).

## Estimated effort

- isqrt64 + normalize function: ~30 lines in glyph_dataset.c
- Header declaration: 3 lines
- Consumer flag (--normalize or auto-detect): ~5 lines
- Total: ~40 lines of library code
- Measurement runs: ~20 minutes across 3 datasets

## What the LMM found

The structural zero was being STOLEN by contrast deficiency.
Low-contrast CIFAR-10 images produced degenerate all-zero
signatures not because the projections were uninformative but
because the signal was too weak to cross the τ threshold.
Normalization restores the structural zero's intended meaning:
"this direction doesn't discriminate" instead of "this image
was too faint to measure."

This is a §18 emission-coverage fix: the three trit states
must be populated by input structure, not by measurement
artifacts. Normalization ensures the trit lattice operates
as designed on natural images with variable contrast.
