---
date: 2026-04-18
phase: REFLECT
topic: Ternary FFN bridge between LSH and GSH
---

# Ternary FFN bridge — REFLECT

---

## Core insight

RAW circled around whether a ternary matmul is appropriate for
routing signals. NODES mapped the full architecture. REFLECT
finds the structural answer.

**The FFN is where the structural zero does its SECOND job.**

In the LSH, the structural zero (W_f[hidden]=0) in the direct
quantization means "this pixel is uninformative." That's its
FIRST job: spatial attention on the input.

In the FFN, the structural zero in the weight matrix means
"this table's signal is irrelevant to this hidden feature."
That's its SECOND job: routing attention on the measurements.

The same mechanism (ternary weight with a zero state) serves
two different attention functions at two different levels:
pixel-level in the quantization, table-level in the FFN. The
architecture is self-similar — the same ternary primitive
appears at every layer, doing the same job (selective attention)
on different inputs.

This is the NORTH_STAR argument realized: base-3 enables
selective attention at every stage because the zero state is
available at every stage. Base-2 ({-1, +1}) cannot say "ignore
this" at any stage.

## T1 resolved: random weights first, learn if it works

Random ternary weights on a 128-dim routing vector are
appropriate because:
1. The input is low-dimensional (no subsampling/information loss)
2. There's no spatial structure to preserve
3. Expansion (128→256) means every input contributes to every
   output — no input is lost

The random projection failure was specific to HIGH-DIMENSIONAL
SPATIAL inputs (3072 pixels → 16 trits = 192:1 compression).
LOW-DIMENSIONAL NON-SPATIAL inputs (128 routing measurements →
256 hidden = 1:2 expansion) don't have this problem.

Start with random. The mechanism either works (cross-table
patterns are detectable via random ternary mixing) or doesn't
(the routing vector is too low-information for any mixing to
help). Random weights answer the MECHANISM question cheaply.

## T2 resolved: one-layer first

Two layers add complexity without changing the mechanism. One
ternary matmul (128→G) with quantization tests whether cross-
table mixing helps at all. If it does, the second layer
(expand-then-compress) is the optimization.

## T3 resolved: labels + distances

Labels alone lose confidence information. Distances alone lose
class identity. Both together encode "table m thinks this is
class c with confidence d." The FFN can mix "high-confidence
Cat from table 3" with "low-confidence Dog from table 7" to
produce a feature that the GSH can't get from labels alone.

Vote counts are sparse and variable-length — harder to encode
as a fixed-dimension input. Defer to a later version.

## T4 resolved: direct quantization LSH FIRST

The FFN bridges FROM the LSH's routing output. If the LSH is
still using random projections (the deprecated path), the
routing output is noisy and the FFN inherits that noise.
Building the FFN on top of direct-quantization LSH gives the
FFN the best possible input.

Order:
1. Integrate direct ternary quantization into the LSH consumer
2. Measure the new baseline
3. Build the FFN bridge + GSH
4. Measure the combined system

## What I now understand

1. **The FFN is appropriate for routing signals** because routing
   signals are low-dimensional and non-spatial. Ternary matmul
   EXPANDS them (no information loss), unlike pixel projection
   which COMPRESSES them (massive information loss).

2. **The structural zero serves two jobs:** pixel attention in
   the quantization, table attention in the FFN. Same mechanism,
   different level. Self-similar architecture.

3. **One layer, random weights, labels + distances** is the
   minimum viable FFN. It tests whether cross-table mixing adds
   discriminative power to the GSH.

4. **Direct quantization LSH must come first.** The FFN needs
   good routing measurements to mix. Building it on top of the
   deprecated random-projection LSH would test the wrong thing.

## What remains uncertain

- Whether the routing vector (128 dims of labels + distances)
  contains enough cross-table structure for the FFN to extract
  useful hidden features.
- Whether the FFN's hidden features produce a GSH that improves
  on the direct multi-trit vote encoding.
- Whether the combination (LSH + FFN-GSH) exceeds either alone.
- The right hidden dimension G (how many cross-table features).
