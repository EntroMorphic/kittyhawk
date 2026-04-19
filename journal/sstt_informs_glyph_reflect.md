---
date: 2026-04-18
phase: REFLECT
topic: How SSTT best informs the Glyph LSH/GSH infrastructure
---

# SSTT informs Glyph — REFLECT

---

## Core insight

RAW found the principle. NODES mapped the mechanism. REFLECT
finds what matters and what doesn't.

**The principle is ORDERING, not features.**

SSTT's specific features (block encoding, IG weighting, RGB
interleave) are coupled to SSTT's inverted-index scoring.
Transplanting them into Glyph's Hamming-distance scoring
doesn't work (proven by the interleave experiment: −1.32pp).

What DOES transfer is the ORDER OF OPERATIONS:

```
SSTT:  quantize → features on trits
Glyph: features on floats → quantize
```

Moving quantization BEFORE gradient computation makes the
gradients trit-native. This eliminates the gradient tau
parameter, produces structural zeros that are meaningful by
construction, and makes the pipeline trit-end-to-end.

## T1 resolved: trit-native transitions first, measure

The ordering change is:
- Remove: compute gradients on normalized MTFP values, then
  quantize with a separate tau_gradient.
- Add: quantize intensity to trits first, then compute
  clamp_trit(trit_a - trit_b) as the gradient.

This produces different signatures because the gradient
values are {-1,0,+1} directly instead of quantized continuous
differences. The question is whether this produces better
accuracy. It should help MNIST (where tau_gradient=0 was
meaningless) and be neutral-to-positive on CIFAR-10 (where
the transition captures the same edge information but with
cleaner zero semantics).

The implementation is ~10 lines changed in direct_lsh.c.
Measure on all three datasets.

## T2 deferred: IG-weighted keys are Step 2+

Computing per-position information gain requires:
1. For each trit position, count (value, class) pairs across
   the training set.
2. Compute the entropy reduction per position.
3. Rank positions by IG.
4. Use top-K IG positions as bucket key instead of spatial
   pooling.

This is computable from trit signatures on the training set
(no float, no pixel space — pure trit statistics). But it
changes the key construction and requires a calibration pass.
Defer until trit-native transitions are validated.

## T3 deferred: block-match scoring is Step 3

Block-match adds a second scoring dimension. Worth exploring
after the foundation (trit-native transitions, Step 1) and
IG weighting (Step 2) are validated.

## What I now understand

1. **The deepest lesson from SSTT is operation ordering.**
   Quantize first, compute features second. Everything after
   quantization is trit-native — no float gradients, no
   separate calibration, no gradient tau.

2. **The gradient structural zero becomes construction-
   guaranteed.** Transition between same-state trits = 0.
   No threshold needed. On MNIST, background-to-background
   transitions naturally produce zeros. On CIFAR-10, smooth
   regions naturally produce zeros. The zero semantics are
   correct BY DESIGN, not by calibration.

3. **The staged plan is: (1) trit-native transitions,
   (2) IG-weighted keys, (3) block-match scoring.** Each
   stage is independently testable and builds on the previous.

4. **What NOT to take from SSTT:** specific features,
   interleaving, fixed thresholds, inverted index. These are
   SSTT's implementation, not its principles.
