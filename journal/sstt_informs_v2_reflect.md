---
date: 2026-04-19
phase: REFLECT
topic: How SSTT best informs Glyph — second pass after empirical data
---

# SSTT informs Glyph v2 — REFLECT

---

## Core insight

I've been trying to map SSTT's features onto Glyph's distance
function. But SSTT doesn't USE a distance function. SSTT uses
an INVERTED INDEX. The inverted index is not a distance — it's
a ROUTING MECHANISM where each block value routes to the
training images that share that value.

The breakthrough is recognizing that the inverted index IS
multi-table routing:
- Each block position is a "table"
- Each block value is a "bucket key"
- The inverted list is the "bucket"
- IG weighting is per-table contribution weighting

We already have this architecture. The inverted index IS a
Trit Lattice LSH where:
- N_tables = N_block_positions (3008 for CIFAR-10)
- N_PROJ = 1 block (3 trits → 27 possible keys per table)
- The bucket key IS the block value
- Scoring accumulates IG-weighted hits across all tables

This reframes the entire question. We don't need to "transfer
SSTT's scoring into Glyph's architecture." SSTT's scoring IS
Glyph's architecture at a different scale: many small tables
(one per block position) instead of few large tables (each
covering the whole image).

## T1 resolved: inverted index as primary

The inverted index IS the routing. It doesn't need an LSH
pre-filter because it IS a multi-table hash lookup — the same
mechanism our LSH uses, just at a finer granularity (3008
tables of 1 block each vs 64 tables of 16 trits each).

The LSH's hierarchical key samples 16 trit positions and
looks up candidates in a 43M-key space. The inverted index
samples 3008 block positions and looks up candidates in a
27-key space per position. The inverted index has MORE
tables and SMALLER keys — finer routing.

Use the inverted index as the primary scorer. The LSH
becomes redundant — or rather, the inverted index IS the
LSH at block-level granularity.

## T2 resolved: both, compared

Implement SSTT-style direct class scoring AND k-NN on
IG-retrieved candidates. Compare. SSTT uses direct
scoring; our architecture uses k-NN. The empirical test
will show which is better on our signatures.

## T3 resolved: the GSH is a different instrument

The inverted index operates on SPATIAL block structure.
The GSH operates on ROUTING PATTERNS (which classes the
spatial blocks vote for). These are different levels:
the inverted index asks "what spatial pattern does this
image have?" while the GSH asks "how does the routing
architecture SEE this image?"

The inverted index could replace the LSH (it's a better
version of the same mechanism). The GSH adds a
topological layer on top. The combination: inverted index
as the primary, GSH as the confidence signal.

## What I now understand

1. **SSTT's inverted index is a multi-table LSH at
   block-level granularity.** 3008 tables × 27 keys per
   table. IG weighting is per-table importance. We've
   been reimplementing SSTT's architecture without
   recognizing it.

2. **The block value IS the bucket key.** No hashing
   needed. The 3-trit block maps directly to one of 27
   values. Exact-match lookup. The inverted list IS the
   bucket.

3. **IG weighting IS per-table contribution weighting.**
   High-IG positions (tables) contribute more to the
   candidate score. This is the same principle as our
   multi-table composition where each table's vote
   contributes equally — but with IG, the votes are
   weighted.

4. **The implementation is straightforward:** build
   inverted lists from block-encoded training signatures,
   precompute IG per position, at query time accumulate
   IG-weighted hits per candidate, take top-K.
