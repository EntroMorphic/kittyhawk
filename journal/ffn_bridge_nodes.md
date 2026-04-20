---
date: 2026-04-18
phase: NODES
topic: Ternary FFN bridge between LSH and GSH
---

# Ternary FFN bridge — NODES

---

## Node 1 — The FFN serves a different purpose than the LSH or GSH

LSH: spatial attention (find the neighborhood in pixel space).
GSH: topological attention (match routing patterns).
FFN: feature mixing (create cross-table combination features
from single-table routing measurements).

Each serves a distinct role. Without the FFN, the GSH reads
a per-table-independent encoding. With the FFN, the GSH reads
features that encode cross-table INTERACTIONS.

## Node 2 — The FFN input should be the FULL routing vector

The current GSH encodes only per-table labels (4 trits each).
The FFN should read ALL available routing signals:
- Per-table 1-NN label (M integers in 0..9)
- Per-table 1-NN distance (M integers)
- Per-candidate vote count (sparse, variable-length)
- SUM prediction (1 integer)

The first two are fixed-length (M dims each). Concatenated:
a 2M-dimensional routing vector (128 dims at M=64). The FFN
projects this into a hidden space.

## Node 3 — Ternary matmul on routing signals is appropriate

The routing vector is:
- Low-dimensional (128, not 3072)
- Non-spatial (no pixel structure to preserve)
- Cross-dimension patterns are the signal

A ternary matmul EXPANDS 128 → H hidden dims, creating
explicit features for cross-table combinations. This is
expansion, not compression — no information loss. The ternary
quantization after expansion is a meaningful nonlinearity.

This is NOT the same as random projection of pixels. Random
projection of ROUTING SIGNALS is appropriate because the
signals have no spatial locality.

## Node 4 — The nonlinearity IS ternary quantization

In a neural network FFN: σ(W·x) where σ is ReLU or GELU.
In ternary: quantize(W·x) where quantize thresholds to
{-1, 0, +1}. The zero state means "this cross-table pattern
is not present" (structural zero as feature absence). The
±1 means "this pattern is present with this sign."

The quantization reduces continuous projections to three
discrete states. This IS a nonlinearity — it creates features
that are NOT linear combinations of the input.

## Node 5 — Two-layer FFN for mix-and-compress

Layer 1: M_in → H (expand). Creates cross-table features.
Layer 2: H → G (compress). Selects the most informative
cross-table features for the GSH signature.

At M_in=128 (64 labels + 64 distances):
- H = 256 (2× expansion, creates ~256 cross-table features)
- G = 256 (GSH signature dimension, ~64 bytes)

The GSH then hashes the G-dim FFN output using bucket index +
multi-probe + k-NN, same as before.

## Node 6 — The distance signal needs encoding

Per-table distances are integers in [0, 2×N_PROJ] = [0, 32]
at N_PROJ=16. They need to be encoded as MTFP values for the
ternary matmul. Simple mapping: distance → MTFP value scaled
so that the median distance maps to zero. Distances below
median → positive (close = good). Above median → negative.

    encoded_dist = (median - distance) × MTFP_SCALE / median

This centers the distance distribution at zero so the ternary
quantization produces balanced trits.

## Node 7 — FFN weights: routing-learned, not random

For image PIXELS, we deprecated random projections because
they destroy spatial structure. For ROUTING SIGNALS, random
projections are less destructive (no spatial structure), but
routing-learned weights should still be better.

Learning: generate N_cand random ternary weight matrices.
For each, compute the FFN output for all training images.
Score by class separability of the FFN output (same
criterion as the projection selection LMM). Keep the best.

But this is expensive: N_cand × N_train × matmul per layer.
Start with random weights (appropriate for non-spatial data),
measure, then optimize if the mechanism works.

## Node 8 — The GSH on FFN output is a standard LSH

The FFN produces a G-dim ternary vector per query. This IS
a ternary signature — pack it, bucket-index the first 16
trits, multi-probe, k-NN resolve. The downstream is
completely standard.

The FFN transforms the MEANING of the signature from "raw
per-table votes" to "cross-table routing patterns." The
infrastructure doesn't change.

## Node 9 — The full architecture with FFN

```
pixels → normalize → direct quantize → LSH (bucket, probe)
              │
              ├── LSH prediction (SUM k-NN)
              │
              ├── routing vector (labels + distances)
              │
              └── FFN (ternary matmul × 2, quantize)
                    │
                    └── GSH (bucket, probe, k-NN)
                          │
                          └── GSH prediction
                                │
              ┌─────────────────┘
              │
         LSH + GSH combination → final prediction
```

Five ternary-native stages: quantization, LSH, FFN, GSH,
combination. All use the same substrate primitives.

## Node 10 — This must wait until direct quantization is in the LSH

The FFN bridge is speculative. Direct ternary quantization in
the LSH pipeline is proven (50.2% CIFAR brute-force). The
correct order:

1. Integrate direct quantization into the LSH consumer
2. Measure the new LSH baseline on all three datasets
3. Add the FFN bridge
4. Measure the FFN + GSH gain

Building the FFN before step 1 is premature — we'd be
bridging FROM a suboptimal LSH (random projections) TO the
GSH, which limits what the FFN can learn.

## Tensions

**T1:** Random vs routing-learned FFN weights. Random is
cheaper and appropriate for non-spatial data. Routing-learned
is better but expensive.

**T2:** One-layer vs two-layer FFN. One layer (M→G) is
simpler. Two layers (M→H→G) adds the expand-compress
structure that neural network FFNs use.

**T3:** What to include in the FFN input — labels only,
labels + distances, or labels + distances + vote counts?

**T4:** Build order — FFN now, or direct quantization LSH
first?
