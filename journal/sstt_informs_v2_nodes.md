---
date: 2026-04-19
phase: NODES
topic: How SSTT best informs Glyph — second pass after empirical data
---

# SSTT informs Glyph v2 — NODES

---

## Node 1 — The inverted index IS the missing mechanism

SSTT doesn't use Hamming distance. It uses inverted index lookup:
for each (position, value) pair in the query, find training images
with the same value at that position. Accumulate IG-weighted hits.

Our Hamming distance compares all positions equally in one pass.
SSTT's inverted index weights positions by IG and values by
frequency. The inverted index IS IG-weighted scoring, implemented
as a data structure instead of a distance function.

## Node 2 — Block encoding lifts per-position information

3-value trits: 1.58 bits per position. Bayesian can't discriminate.
27-value blocks (3 adjacent trits): 4.75 bits per position. Bayesian
can discriminate.

We have 3-trit blocks already — they're consecutive in the signature.
Encoding them as base-27 byte values is a grouping operation, not
a new feature. The information is already there; we're just reading
it in larger chunks.

## Node 3 — The inverted index on block values is SSTT's architecture

For each of 3008 block positions (9024/3), each of 27 block values:
store the list of training image indices with that value at that
position.

Query: for each block position, look up the query's block value
in the inverted list. For each hit, add ig_w[position] to the
candidate's score. Candidate with the highest total score wins.

This IS routing: the block value routes to a specific list of
training images. The IG weight determines how much that routing
decision contributes. High-IG positions route to small,
discriminative lists. Low-IG positions route to large,
uninformative lists.

## Node 4 — Background blocks self-filter

Block value 13 (base-27 for 0,0,0) = all-zero trits. Many training
images have this at many positions. The inverted list for
(position, 13) is large → low per-entry information. IG weight
for positions that are usually all-zero is low. The structural
zero filters itself through the IG mechanism — no explicit
zero-skipping needed.

## Node 5 — The inverted index composes with LSH and GSH

Option A: inverted index as the PRIMARY scorer (replace Hamming).
No candidate set needed — the inverted index scores ALL training
images implicitly. The score is accumulated from hits across
all block positions.

Option B: LSH filters → inverted index re-ranks. The LSH
produces a candidate set via hierarchical keys. The inverted
index re-scores only those candidates with IG-weighted block
matching.

Option A is SSTT's approach. Option B uses our LSH for speed
and SSTT's scoring for accuracy. Option B is faster (score
~1600 candidates vs accumulate ~5.6M hits) but might miss
candidates that the inverted index would find but the LSH
filter doesn't.

## Node 6 — The inverted index is not brute-force

The inverted index doesn't compare the query to every training
image. It looks up each query block value and finds the SUBSET
of training images with that value at that position. Only images
that SHARE block values with the query accumulate score. Images
that share NO block values get zero score.

This is hash-based retrieval — the block value IS a hash key.
The inverted index IS a multi-table hash lookup where each
"table" is a block position and the "key" is the block value.

In LSH terminology: the inverted index is M tables (one per block
position) with exact-match lookup (block value = bucket key).
Multi-probe is not needed because the block value is already
at the right resolution (27 values, not 3^16 = 43M).

## Node 7 — This is the architecture the GSH was trying to be

The GSH hashed routing patterns to find training images with
similar patterns. The inverted index hashes block values to
find training images with similar spatial structure. Both are
routing-native hash lookups. But the inverted index operates
on SPATIAL block structure (directly from pixel quantization),
while the GSH operates on ROUTING patterns (derived from LSH
output).

The inverted index is a more direct path: pixels → blocks →
hash lookup → classification. No intermediate LSH step needed.

## Node 8 — k-NN on IG-scored candidates vs direct IG classification

Two approaches:
A) Inverted index produces per-training-image scores. Take top-K
   by score. k-NN vote. This uses the inverted index as a
   RETRIEVAL mechanism and k-NN as the CLASSIFIER.

B) Direct classification: for each class, sum the IG-weighted
   block matches. Class with the highest sum wins. No k-NN.
   This is SSTT's approach.

Approach B is simpler and is what SSTT's `classify_bayesian` does.
Approach A adds the k-NN vote which our architecture already
provides.

## Tensions

**T1:** Inverted index as primary (replace LSH) vs as re-ranker
(after LSH filter). Primary is simpler but slower on large
training sets. Re-ranker uses existing infrastructure.

**T2:** Direct class scoring vs k-NN on retrieved candidates.
SSTT uses direct; our architecture uses k-NN.

**T3:** Does the inverted index make the GSH redundant? Both
are hash-based routing. If the inverted index subsumes the
GSH's function, the architecture simplifies.
