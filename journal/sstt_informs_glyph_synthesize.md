---
date: 2026-04-18
phase: SYNTHESIZE
topic: How SSTT best informs the Glyph LSH/GSH infrastructure
---

# SSTT informs Glyph — SYNTHESIZE

Executable specification.

---

## What SSTT teaches Glyph

One principle: **quantize first, compute features second.**

SSTT's pipeline is trit-native end-to-end: pixels → quantize →
features on trits. Glyph's current pipeline breaks the trit-native
chain: pixels → normalize → gradients on floats → quantize.

The fix reorders the quantization boundary so that gradients
are trit transitions, not quantized float differences. This
eliminates the gradient tau parameter and makes the structural
zero in the gradient channel construction-guaranteed.

## Step 1: trit-native transitions (build now)

### Current pipeline

```
normalize(pixel) → intensity_mtfp
gradient = intensity_mtfp[x+1] - intensity_mtfp[x]    (MTFP float)
quantize(intensity_mtfp, tau_intensity) → intensity_trit
quantize(gradient, tau_gradient) → gradient_trit       (tau=0 on MNIST!)
```

### Fixed pipeline

```
normalize(pixel) → intensity_mtfp
quantize(intensity_mtfp, tau_intensity) → intensity_trit
transition = clamp_trit(intensity_trit[x+1] - intensity_trit[x])  (trit-native)
```

The transition function:
```c
static inline int8_t trit_transition(int8_t a, int8_t b) {
    int d = b - a;  /* values: -2, -1, 0, +1, +2 */
    return (d > 0) ? +1 : (d < 0) ? -1 : 0;
}
```

### What changes

- Gradient tau is ELIMINATED. No calibration. No per-dataset
  gradient density tuning.
- Gradient zeros arise from same-state pairs (a == b → transition
  = 0). Semantically correct: "no edge at this position."
- MNIST gradient zeros come from background-background pairs
  (both zero). Currently MNIST gradients have tau=0 which means
  NO zeros — every tiny gradient maps to ±1. After fix, the
  ~85% of MNIST pairs that are background-background produce
  transition=0 naturally.
- CIFAR-10: smooth regions (adjacent pixels in same ternary
  state) produce transition=0. Edges (state change) produce ±1.
  Same information as the current gradient but with cleaner
  zero semantics.

### Implementation

In `direct_lsh.c`, replace the gradient computation:

```c
/* OLD: gradients on MTFP values, separate tau */
hgrad[idx] = feat[x+1] - feat[x];  /* float-like */
...
quantize(hgrad, tau_gradient) → trit

/* NEW: trit transitions on quantized intensity */
int8_t trit_left  = glyph_read_trit(intensity_sig, x);
int8_t trit_right = glyph_read_trit(intensity_sig, x + 1);
glyph_write_trit(full_sig, grad_offset + idx,
                 trit_transition(trit_left, trit_right));
```

This replaces the two-step (compute MTFP gradient → quantize
with tau_gradient) with one step (compute trit transition from
quantized intensity). No tau_gradient. No gradient calibration
sample. No gradient density parameter.

### Expected outcome

- MNIST: gradient zeros from background pairs instead of all-±1
  noise. Should HELP (fewer noisy trit positions).
- Fashion-MNIST: edges produce same ±1 as before, flat regions
  produce 0 instead of quantized-small-gradient. Should be
  NEUTRAL to slightly positive.
- CIFAR-10: same edge detection, cleaner zero semantics. Should
  be NEUTRAL to slightly positive.

### Measurement

Run all three datasets with trit-native transitions. Compare to
the current float-gradient results:

```
Current:   MNIST 97.23% (no grad)    Fashion 87.78%    CIFAR 44.68%
Expected:  MNIST ≥97% (with grad!)   Fashion ≥87.5%    CIFAR ≥44.5%
```

The key test: MNIST WITH gradients. Currently gradients HURT
MNIST (97.23% without → 96.54% with). After fix, trit transitions
should be NEUTRAL or positive because the gradient zeros are
meaningful on MNIST.

## Step 2: IG-weighted bucket keys (build if Step 1 validates)

Replace spatial-pooling summary with IG-ranked trit positions
for bucket keys. Use the top-16 most class-discriminative trit
positions (by information gain on the training set) as each
table's key.

## Step 3: block-match scoring (build if Step 2 validates)

Add a second scoring dimension: count exact 3-trit block matches
between query and candidate. Complement to Hamming distance.

## Estimated effort

Step 1: ~30 lines changed in direct_lsh.c. Remove gradient MTFP
computation, gradient tau calibration, gradient density parameter.
Add trit-transition loop after intensity quantization.

Steps 2-3: ~100 lines each. Deferred.
