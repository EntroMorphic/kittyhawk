---
date: 2026-04-19
phase: RAW
topic: How SSTT best informs Glyph — second pass after empirical data
---

# SSTT informs Glyph v2 — RAW

---

The first LMM on this topic (v1) produced one action: trit-native
transitions. It failed. The red-team forced a second round that
produced IG-weighted scoring. That succeeded (+2.34pp on CIFAR-10)
but only on the hard dataset. Now I need to think about what we
ACTUALLY learned across both rounds and what's left.

What we learned:

1. SSTT's SPECIFIC features (interleave, fixed thresholds, block
   encoding) don't transfer because they're coupled to SSTT's
   scoring (IG-weighted inverted index with 27-value symbols).

2. SSTT's SCORING PRINCIPLE (weight positions by information gain)
   DOES transfer: +2.34pp on CIFAR-10 via pair-IG re-ranking.

3. The gain is REDISTRIBUTIVE, not additive. IG moves accuracy
   from easy classes (Airplane −4.4pp) to hard classes (Deer
   +13.3pp). On datasets where the easy classes dominate (MNIST,
   Fashion), the redistribution hurts overall.

4. Our Hamming distance with 3-value trits is fundamentally
   limited for position-dependent scoring. SSTT's 27-value
   block codes give the Bayesian classifier more signal per
   position.

What I haven't tried:

5. SSTT's BLOCK ENCODING on our trit signatures. Group 3
   adjacent trits into a 27-value block symbol. Use the block
   symbols as the basis for IG scoring instead of individual
   trits. This could give the Bayesian enough resolution per
   position to work.

6. SSTT's INVERTED INDEX structure. Instead of brute-force
   Hamming, build an inverted index keyed on (position, block-
   value). For each non-background block in the query, look up
   which training images have the same block value at that
   position. Accumulate with IG weights. This is SSTT's actual
   scoring mechanism.

7. The combination of IG scoring with our LSH filter. The
   brute-force IG at 45.81% is higher than our LSH at 44.68%.
   If we use the LSH to produce a candidate set and then
   IG-score within that set (instead of Hamming scoring), we
   get the filter's speed with IG's accuracy.

8. Using IG weights to improve the BUCKET KEY. Currently the
   hierarchical key uses spatial majority voting. What if the
   key used the top-16 IG positions instead? This would make
   the filter itself IG-aware.

Let me think about which of these is most promising.

Item 7 is the most direct: replace the Hamming scorer in
direct_lsh with the pair-IG scorer. The LSH finds ~1592
candidates via hierarchical keys. Instead of scoring them
by Hamming distance, score them by pair-IG weighted distance.
The candidates are already found; only the ranking changes.

But pair-IG scoring is O(total_dim × ncand) per-trit operations,
not O(sig_bytes × ncand) popcount operations. At total_dim=9024
and ncand=1592, that's 14.4M per-trit reads per query vs
1592 × 2256 = 3.6M byte operations for popcount. About 4× slower.
Acceptable.

Item 8 is interesting: the hierarchical summary selects
key trits by SPATIAL proximity. An IG-aware key would select
by DISCRIMINABILITY. The 16 most-IG positions across the full
signature would produce keys that discriminate classes better
than spatial summaries, potentially producing smaller, more
relevant unions.

But this is the "random vs spatial vs IG" key selection question
we've been circling. The spatial summary already works reasonably
(union ~1592). An IG key would produce a different union — maybe
smaller (better discrimination) or maybe it would miss some
candidates (worse oracle).

Item 5 (block encoding) addresses a deeper issue: our 3-value
trits limit the per-position information to log2(3) = 1.58 bits.
SSTT's 27-value blocks provide log2(27) = 4.75 bits per position.
With 3× more bits per position, the Bayesian and IG scoring
have much more to work with.

We HAVE blocks already — three adjacent trits are a block.
The issue is that our Hamming distance doesn't treat them as
blocks; it treats them as independent trits. If we scored by
BLOCK MATCH (do all 3 trits in this block match?) instead of
per-trit Hamming, we'd get pattern sensitivity. But block-match
is coarser than Hamming (no partial credit).

Actually, what if we encoded each 3-trit block as a single
byte value (base-27) and built an inverted index on those
byte values? This IS SSTT's architecture, but applied to our
trit signatures instead of SSTT's interleaved pixels.

The inverted index: for each (block_position, block_value),
store the list of training image indices. At query time, for
each block position in the query, look up the training images
with the same block value. Accumulate IG-weighted votes per
training image. The training images with the highest weighted
vote count are the candidates.

This is not brute-force (O(N_train) per query). It's inverted
index lookup — O(hits per block) per block position. Total
cost: Σ_blocks |inverted_list(block_pos, query_block_val)|.
On CIFAR-10 with 3008 blocks (9024/3) and 50K training images,
each block value has ~50000/27 ≈ 1852 entries on average.
Total hits: 3008 × 1852 = 5.6M lookups. Each is a single array
index increment. With IG weighting, each hit adds ig_w[pos] to
the candidate's score.

This IS SSTT's scoring. On our signatures. With our IG weights.
No Hamming distance. No brute-force. The inverted index IS
the routing mechanism.

And it composes with the GSH: the inverted index produces a
prediction. The GSH produces a prediction from routing patterns.
Agreement = high confidence.

I think this is it. The inverted index on 27-value block-encoded
trit signatures, IG-weighted, is SSTT's architecture translated
to Glyph's representation. Not transplanting SSTT's features —
using SSTT's SCORING MECHANISM on Glyph's features.

And the structural zero plays a specific role: block values
where one or more trits are zero (e.g., (0, +1, -1)) are
LESS distinctive than fully-populated blocks (e.g., (+1, -1, +1)).
The inverted index naturally handles this — background blocks
(all-zero = block value 13 in base-27) have a large inverted
list (many training images have all-zero at that position),
so they contribute little discriminative information. High-IG
positions with distinctive block values have small inverted
lists — the lookup is fast AND discriminative.

The zero state filters itself in the inverted index: common
block values (which include blocks with zeros) have large
lists and low per-entry information. Distinctive block values
have small lists and high per-entry information. No explicit
zero-filtering needed.
