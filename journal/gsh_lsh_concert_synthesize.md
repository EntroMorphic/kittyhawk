---
date: 2026-04-17
phase: SYNTHESIZE
topic: GSH + LSH in concert — functions and relationship
---

# GSH + LSH in concert — SYNTHESIZE

Executable specification.

---

## Architecture

```
                       query image
                           │
              ┌────────────┴────────────┐
              │                         │
         LSH (Layer 1)             LSH routing
         pixels → sigs             pattern extraction
         bucket → probe            (per-table 1-NN from union)
         union → k-NN SUM              │
              │                    vote pattern
              │                    [v₀, v₁, ..., v_{M-1}]
              │                         │
              │                    multi-trit encode
              │                    4 trits per vote
              │                         │
              │                    GSH (Layer 2)
              │                    vote-sigs → buckets
              │                    probe → union → k-NN
              │                         │
         lsh_pred                  gsh_pred
              │                         │
              └────────────┬────────────┘
                           │
                      combination
                           │
                      final_pred
```

**LSH:** pixel-geometric distance. Finds prototypes the query
LOOKS LIKE.

**GSH:** routing-pattern distance. Finds images the query
ROUTES LIKE.

**Combination:** agreement-based. Report both predictions and
their agreement rate. Measure accuracy when they agree vs
disagree to determine whether the GSH earned override
authority.

## Multi-trit vote encoding

Each table's vote label (0-9) → 4 trits.

Encoding table (10 classes → 4 trits, one-to-one):

```c
static const int8_t vote_to_trits[10][4] = {
    {-1,-1,-1,-1},  /* class 0 */
    {-1,-1,-1, 0},  /* class 1 */
    {-1,-1,-1,+1},  /* class 2 */
    {-1,-1, 0,-1},  /* class 3 */
    {-1,-1, 0, 0},  /* class 4 */
    {-1,-1, 0,+1},  /* class 5 */
    {-1,-1,+1,-1},  /* class 6 */
    {-1,-1,+1, 0},  /* class 7 */
    {-1,-1,+1,+1},  /* class 8 */
    {-1, 0,-1,-1},  /* class 9 */
};
```

Each class maps to a unique 4-trit codeword. The Hamming
distance between two codewords encodes class similarity
(adjacent classes differ by 1 trit, distant classes differ
by 2-4 trits). The mapping is arbitrary but deterministic.

M=64 tables × 4 trits/table = 256 trits = 64 bytes.

## GSH bucket key

Use the first 16 trits (first 4 tables' encoded votes) as
the uint32 bucket key. The full 256-trit signature is used
for Hamming distance scoring in the resolver.

Multi-probe at radius 1 flips one trit in the first 16,
which partially changes one of the first 4 tables' encoded
votes. At radius 2, two trits flip — either two different
tables' votes or two trits within one table's encoding.

## GSH construction (no random projection)

The GSH does NOT call glyph_sig_builder_init. It packs
the vote-to-trit encoding directly into packed-trit
signatures using m4t_trit_pack. The signature IS the
encoded routing pattern — no projection, no τ calibration.

For each training image:
1. Probe the LSH, build union, extract per-table 1-NN
   labels (excluding self).
2. Encode M labels → M×4 trits → pack into 64 bytes.
3. Store the packed signature.

Build bucket index on the packed signatures (first 4 bytes).

For each test query:
1. Probe the LSH, build union, extract per-table 1-NN
   labels.
2. Encode → pack → probe the GSH bucket index.
3. Score GSH union by Hamming distance on full 256-trit
   signatures.

## GSH resolver

The GSH's Hamming distance counts trit disagreements
between two images' encoded vote patterns. Two images with
distance 0 have identical routing patterns (every table
voted the same class). Distance 4 means one table's vote
differs completely (all 4 encoding trits flipped).

The k-NN resolver (k=5 rank-weighted) over the GSH union
votes the labels of the 5 training images with the most
similar routing patterns.

## Combination

V1: report LSH prediction, GSH prediction, and agreement.
Compute accuracy conditional on agreement:

```
if (lsh_pred == gsh_pred): agree_count++
  accuracy = P(correct | agree)
else: disagree_count++
  accuracy_lsh = P(lsh_correct | disagree)
  accuracy_gsh = P(gsh_correct | disagree)
```

If P(gsh_correct | disagree) > P(lsh_correct | disagree),
the GSH adds value on disputed queries and deserves override
authority. Otherwise, always defer to LSH.

## Implementation

Rewrite layered_lsh.c:

1. Remove glyph_sig_builder for Layer 2.
2. Add vote-to-trit encoding function.
3. Pack encoded votes into 64-byte signatures directly.
4. Build GSH bucket index on packed signatures.
5. Probe GSH with multi-probe using the existing
   glyph_multiprobe_enumerate (the encoded vote signature
   is a valid packed-trit signature).
6. Score GSH union with m4t_popcount_dist on 64-byte sigs.
7. Report LSH, GSH, agreement, and conditional accuracy.

Key subtlety: glyph_multiprobe_enumerate works on N_PROJ
trits. For the GSH, N_PROJ = 256 (M×4 trits). sig_bytes = 64.
The multi-probe radius 1 at N_PROJ=256 generates ~170 probes
(each non-zero trit produces 1 probe; each zero trit produces
2 probes). This is manageable.

BUT: the bucket index uses uint32 keys (4 bytes = 16 trits).
At 256 trits, the key covers only the first 16 trits (first
4 tables). The remaining 240 trits are used only for scoring.
This means the bucket filter sees only 4 tables' worth of
routing pattern — a very coarse filter. Multi-probe at the
16-trit level would be needed to expand the neighborhood.

## Estimated effort

- Vote-to-trit encoding: ~20 lines
- Signature packing: ~15 lines
- GSH build (remove sig_builder, add direct pack): ~30 lines
- GSH probe + resolve: ~20 lines (reuse existing)
- Combination + agreement reporting: ~30 lines
- Total: ~120 lines changed from current layered_lsh.c

## Go / no-go

**Go:** GSH accuracy > LSH accuracy on ANY of the three
datasets, or combined accuracy > LSH accuracy on CIFAR-10.

**Marginal:** GSH accuracy close to LSH but combination
doesn't help. The two instruments agree too often for the
combination to matter.

**No-go:** GSH accuracy << LSH accuracy again. The routing
pattern doesn't carry enough class signal for even direct
encoding to work. Would mean the per-table 1-NN labels are
fundamentally too noisy for a second routing layer.
