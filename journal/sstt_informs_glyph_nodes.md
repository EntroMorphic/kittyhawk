---
date: 2026-04-18
phase: NODES
topic: How SSTT best informs the Glyph LSH/GSH infrastructure
---

# SSTT informs Glyph — NODES

---

## Node 1 — Quantize first, compute features second

SSTT: raw pixels → ternary quantize → compute transitions on trits.
Glyph: raw pixels → normalize → compute gradients on floats → quantize.

SSTT's ordering makes the gradient a TRIT OPERATION (transition
between ternary states). Glyph's ordering makes the gradient a
FLOAT OPERATION that gets quantized afterwards.

The fix: normalize → quantize intensity → compute trit-transitions.
The gradient becomes clamp_trit(trit_a - trit_b), which is always
in {-1, 0, +1}. No gradient tau. No calibration. The gradient IS
ternary by construction.

## Node 2 — Trit transitions vs magnitude gradients

clamp_trit(trit_a - trit_b) where trit values are {-1,0,+1}:

| trit_a | trit_b | diff | clamped |
|--------|--------|------|---------|
| +1     | +1     | 0    | 0       |
| +1     | 0      | +1   | +1      |
| +1     | -1     | +2   | +1      |
| 0      | +1     | -1   | -1      |
| 0      | 0      | 0    | 0       |
| 0      | -1     | +1   | +1      |
| -1     | +1     | -2   | -1      |
| -1     | 0      | -1   | -1      |
| -1     | -1     | 0    | 0       |

The transition trit means:
- +1: state increases (darker→brighter, neutral→bright, dark→neutral)
- -1: state decreases
- 0: no change (same ternary state)

The structural zero in the transition means "no edge" — the
ternary state is constant across this pixel pair. Genuinely
uninformative, by construction, without calibration.

## Node 3 — This eliminates the gradient tau problem

Current: gradient tau=0 on MNIST (every gradient maps to ±1,
no structural zeros, adds noise). With trit transitions:
gradient zeros arise naturally from same-state pairs (trit_a ==
trit_b). On MNIST's sparse images (many background-zero pixels),
adjacent background pairs produce transition=0 naturally. The
gradient zeros ARE the background, automatically.

No tau calibration needed. No per-dataset gradient density tuning.
The operation defines the zeros, not a threshold.

## Node 4 — Information-gain weighting maps to the FFN bridge

SSTT's IG weighting: each (position, value) pair gets a weight
proportional to how much it reduces class uncertainty. High-IG
positions dominate the score.

In Glyph: Hamming distance treats all positions equally. The
FFN bridge (from the earlier LMM) would project the M-dimensional
routing vector through a ternary weight matrix, creating hidden
features that weight table contributions by cross-table
correlations.

The analog of IG weighting in Glyph is not per-trit weighting
in the Hamming distance (can't modify the popcount primitive).
It's the FFN: learn which trit positions (via their effect on
routing patterns) are discriminative, and weight accordingly in
the GSH's input.

Or: compute IG per trit position on the training set, and use
only the top-K most informative positions for the bucket key.
The hierarchical summary currently uses SPATIAL blocks. An
IG-ranked summary would use the most DISCRIMINATIVE positions,
regardless of spatial proximity.

## Node 5 — Block-match scoring as a complement to Hamming

SSTT's inverted index does exact-match per block. Glyph's
Hamming distance does per-trit comparison. Block-match captures
pattern identity that Hamming misses.

A hybrid: the LSH scores by Hamming (per-trit). A SECOND
scoring pass counts exact block matches (per-3-trit-group).
Candidates with high block-match count are promoted.

This doesn't require new primitives — compare 3 trits at a
time, check if all match. It's a coarser distance that's
sensitive to patterns.

## Node 6 — The production pipeline with trit-native gradients

```
pixels → normalize → quantize intensity (per-pixel tau)
                         │
                    ternary image (each pixel is -1/0/+1)
                         │
                    trit transitions: h-grad, v-grad
                    (clamp_trit(right - left), clamp_trit(below - above))
                         │
                    concatenate: intensity + h-trans + v-trans
                         │
                    hierarchical summary → bucket keys
                    full signature → Hamming k-NN scoring
                    routing pattern → GSH
```

Every operation after the initial quantization is ternary-native.
No float gradients. No gradient tau. The pipeline is trit-end-to-end.

## Node 7 — What NOT to take from SSTT

- RGB interleaving (doesn't help our Hamming scoring — tested)
- Fixed thresholds 85/170 (too sparse for our balanced emission)
- Inverted index structure (our bucket index serves the same role)
- Base-27 block encoding (our per-trit Hamming is different metric)
- AVX2 implementation (we target NEON on Apple Silicon)

## Tensions

**T1:** Trit-native transitions (Node 1) vs current float
gradients. Does the ordering change actually improve accuracy?

**T2:** IG-weighted keys (Node 4) vs spatial-pooled keys (current).
Which produces more discriminative bucket keys?

**T3:** Block-match scoring (Node 5) vs pure Hamming. Is the
implementation complexity worth the potential gain?
