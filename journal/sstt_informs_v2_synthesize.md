---
date: 2026-04-19
phase: SYNTHESIZE
topic: How SSTT best informs Glyph — second pass after empirical data
---

# SSTT informs Glyph v2 — SYNTHESIZE

Executable specification.

---

## What to build

An **inverted index classifier** on block-encoded trit signatures
with IG-weighted scoring. This is SSTT's scoring mechanism applied
to Glyph's direct-quantized (normalized + gradient) trit signatures.

## Architecture

```
pixels → normalize → quantize (intensity tau) → gradients (gradient tau)
  → concatenate (intensity + hgrad + vgrad) → 9024 trits
  → group into 3-trit blocks → 3008 blocks × 27 possible values
  → inverted index: (block_pos, block_val) → [training indices]
  → query: accumulate IG-weighted hits per candidate
  → top-K → k-NN vote OR direct class scoring
```

## Block encoding

Group every 3 consecutive trits into a block:
```c
int block_val = (trit[0] + 1) * 9 + (trit[1] + 1) * 3 + (trit[2] + 1);
// Produces values 0..26 (base-27)
```

Total blocks: total_dim / 3 = 9024 / 3 = 3008 for CIFAR-10.
(If total_dim is not divisible by 3, the last 1-2 trits form a
smaller block.)

Background block: value 13 (all-zero: (0+1)*9 + (0+1)*3 + (0+1) = 13).

## Inverted index

For each block position p ∈ [0, 3008):
  For each block value v ∈ [0, 27):
    Store sorted array of training image indices with block
    value v at position p.

Memory: 3008 × 27 posting lists. Total entries across all lists
= 3008 × 50000 = 150.4M. Each entry is a uint32 (4 bytes) =
~600 MB. This is large but fits in Apple Silicon memory.

Optimization: skip background blocks (value 13). Their posting
lists are the largest and least discriminative. This reduces
memory and lookup time.

## IG weights

Per-block-position IG (not per-trit). For block position p,
compute IG from the joint distribution of (block_value, class)
across the training set. This gives 4.75 bits of resolution
(27 values) vs the per-trit 1.58 bits (3 values).

```c
for each position p:
  for each block_value v:
    for each class c:
      count[p][v][c] = number of training images of class c
                       with block value v at position p
  ig[p] = H(class) - H(class | block_value at position p)
```

## Query scoring

For each test query:
1. Encode all 3008 blocks.
2. For each block position p where query_block[p] != background:
   a. Look up the posting list for (p, query_block[p]).
   b. For each training image i in the list:
      candidate_score[i] += ig_weight[p]
3. Take the top-K candidates by score.
4. k-NN vote on their labels.

Also implement SSTT-style direct class scoring:
For each class c:
  class_score[c] = Σ_p ig[p] × P(class=c | block_val=query[p], pos=p)

## Expected outcomes

The inverted index with 27-value blocks should significantly
outperform the 3-value per-trit IG scoring (45.81%) because:
- 3× more information per position (4.75 vs 1.58 bits)
- IG weights are more discriminative on block patterns
- Background blocks (value 13) are naturally filtered
- No brute-force Hamming — the inverted index IS the routing

Expected: 48-52% on CIFAR-10. The combination of Glyph's
normalization + gradients (which lifted the Hamming ceiling
from 36% to 44%) with SSTT's inverted-index scoring (which
adds IG-weighted position sensitivity) should compound.

## Implementation

New tool: `tools/inverted_ig.c`

1. Quantize signatures (same as direct_lsh)
2. Encode blocks: 3 trits → base-27
3. Build inverted index (posting lists per (position, value))
4. Compute per-block-position IG
5. Score queries via inverted index lookup
6. Report k-NN and direct class scoring accuracy

## Estimated effort

~200 lines. The quantization and gradient code is reused from
direct_lsh. The inverted index build is ~40 lines. The query
scoring is ~30 lines. The IG computation is reused from
ig_scored.

## Go / no-go

**Go:** CIFAR-10 ≥ 48% (compound of normalization + IG + block encoding)
**Strong go:** ≥ 52% (approaching SSTT's 53%)
**No-go:** ≤ 46% (block encoding doesn't help over per-trit IG)
