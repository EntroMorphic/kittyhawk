# Normalization first light — restoring §18 emission coverage

Date: 2026-04-18
Tool: `tools/mnist_routed_bucket_multi.c` with `--normalize`

## The diagnosis

The representation sweep revealed that raw-pixel L1 k=5-NN on
CIFAR-10 is 36.4% — and our LSH was matching it exactly at
37.06%. We weren't underperforming; we were AT the pixel-space
ceiling.

Per-image normalization (zero-mean, unit-variance) lifts the
raw-pixel k-NN to 42.8%. The +6.4pp comes from equalizing image
contrast: a dark cat and a bright cat become the same pattern
after normalization.

Other transforms tested:

```
Raw pixels L1 k=5:      36.4%
Grayscale:               31.4%  (loses color, no help)
Zero-mean:               38.6%  (+2.2, removes brightness bias)
Normalized:              42.8%  (+6.4, removes brightness AND contrast)
Simple HOG:              17.4%  (spatial gradients HURT)
Gradient magnitude:      27.8%  (also worse than raw)
```

Spatial features (gradients, HOG) are LESS discriminative than
raw pixels on CIFAR-10. The class-discriminative signal is in the
GLOBAL color-shape pattern, not in local edge structure. SSTT's
gradients work because they're COMBINED with intensity, not used
alone.

## The root cause: contrast-stolen structural zeros

On low-contrast CIFAR-10 images, projection values fall below
the τ threshold. Their signatures are degenerate (mostly zero
trits). The structural zero — which should mean "this projection
direction doesn't discriminate" — was being occupied by
"this image was too faint to measure."

This is a §18 emission-coverage violation: the three trit states
must be populated by input structure, not by measurement
artifacts. On MNIST (high-contrast pen strokes), emission
coverage was naturally satisfied. On CIFAR-10 (variable-contrast
natural images), it was broken.

Normalization restores the contract:
- +1: this direction sees positive signal (genuine)
- -1: this direction sees negative signal (genuine)
- 0: this direction doesn't discriminate (genuine)

## Implementation

`glyph_dataset_normalize(ds)` added to libglyph. Per-image
transform in MTFP integer arithmetic:
1. Mean: int64 sum / n_pixels
2. Subtract mean (element-wise)
3. Variance: int64 sum of squares / n_pixels
4. Stddev: Newton's method integer square root
5. Scale: pixel × MTFP_SCALE / stddev, clipped at ±3σ

Called via `--normalize` flag. Applied to both train and test
at load time, before any projection.

## Results — normalized CIFAR-10, M=64

```
density   oracle    SUM_16_k5   SUM_32_RR   PTM       avg_union
0.20     99.96%     40.09%      40.50%    39.60%       455
0.25     99.87%     39.82%      39.98%    39.67%       367
0.33     99.84%     39.36%      39.54%    40.51%       325
0.40     99.99%     40.25%      40.32%    40.88%       381
0.50    100.00%     41.36%      41.86%    39.32%       741
```

Best configuration: density=0.50 normalized.
- SUM_16 k=5: **41.36%** (was 37.06% unnormalized, +4.30pp)
- SUM_32_RR: **41.86%** (was 37.10% unnormalized, +4.76pp)
- Oracle: **100.00%** (restored from 99.84% at d=0.33)

## Key findings

### 1. Normalization lifts the architecture by +4-5pp

The entire pipeline benefits because the signatures are richer.
Every downstream mechanism (multi-table composition, k-NN,
re-rank) operates on better input.

### 2. Union size dropped 40×

Average union at M=64: 741 (d=0.50 normalized) vs 12,851
(d=0.33 unnormalized). The normalized signatures are dramatically
more discriminative. The routing is tighter — smaller
neighborhoods, more relevant candidates.

### 3. Density=0.50 is optimal on normalized data

On unnormalized data, density=0.33 (balanced base-3) was optimal.
On normalized data, density=0.50 wins. The normalized input is
centered at zero with symmetric distribution, so a wider zero
band (50% zeros) correctly marks the near-zero projections as
uninformative. This is the structural zero working as designed:
half the trits say "I don't know" and half carry signal.

### 4. PTM gained more than SUM

PTM jumped from 34.94% to 40.88% at d=0.40 (+5.94pp). Per-table
1-NN accuracy improves dramatically when contrast is equalized
because each table's projection produces meaningful distance
measurements for every image, not just high-contrast ones.

### 5. The N_PROJ landscape may have shifted

With richer per-table signatures, the N_PROJ=64 peak found on
unnormalized data may no longer apply. The brute-force sweep
should be re-run on normalized data to find the new optimum.

## Inter-class nearest-neighbor analysis

The representation sweep also revealed the pixel-space class
structure:

```
Nearest class pairs (pixel-space L1 distance of class means):
  bird  ↔ horse  (0.0237)  ← nearly identical mean pixels
  cat   ↔ dog    (0.0247)
  deer  ↔ cat    (0.0272)
  frog  ↔ deer   (0.0290)
  plane ↔ ship   (0.0471)
  auto  ↔ truck  (0.0456)
```

The most-discriminative pixels are ALL in the Blue channel,
rows 0-3 (top of image = sky). Airplane/Ship have blue sky;
animals don't. This is the GLOBAL color pattern that
normalization preserves and gradients destroy.

## Relationship to SSTT

SSTT reaches 53% on CIFAR-10 with structured ternary features.
The representation sweep showed that gradients alone HURT
(17.4% vs 36.4% raw). SSTT works because it combines intensity
+ gradients + ternary quantization, where the quantization acts
as a per-feature normalization. Our per-image normalization is
the analog: equalize the input before projection.

The remaining gap from 41.86% to 53% is in the FEATURE
REPRESENTATION: SSTT uses multi-channel spatial features
(intensity + gradients + block encoding) while we use random
projections of normalized pixels. The normalization closes the
CONTRAST gap; the feature gap remains as the next target.

## Architecture evolution

```
                           CIFAR-10 SUM k=5 M=64
Raw pixels, d=0.33:           37.06%   (baseline)
+ k-NN k=5:                   37.06%   (already k=5)
+ multi-resolution combined:  37.90%   (+0.84)
+ normalization d=0.50:       41.36%   (+3.46 from combined)
+ SUM_32 re-rank:             41.86%   (+0.50)
```

Normalization is the single largest gain in the project's
CIFAR-10 history. It's also the simplest: ~30 lines of integer
arithmetic applied at load time.
