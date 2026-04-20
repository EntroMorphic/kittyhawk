---
date: 2026-04-18
phase: RAW
topic: Ternary FFN bridge between LSH and GSH
---

# Ternary FFN bridge — RAW

---

The user asked whether an FFN could flow between LSH and GSH.
My immediate answer was "ternary matmul on routing measurements."
But I need to think about whether that's right or whether I'm
reflexively reaching for the projection hammer again.

What does an FFN actually DO in a transformer? It takes the
output of attention (which identified relevant context) and
transforms it through two linear layers with a nonlinearity
between them. The FFN's job is to PROCESS the attended
information — to extract higher-order features from the
attention output before the next attention layer reads it.

In our architecture:
- LSH = first attention (finds the relevant neighborhood)
- GSH = second attention (matches routing patterns)
- FFN = the transform between them

What information does the LSH produce that the GSH needs?

The LSH produces, per query:
1. A union of candidate prototypes (indices)
2. Per-table 1-NN labels (M votes)
3. Per-table 1-NN distances (M integers)
4. Per-candidate vote counts (how many tables found each)
5. The SUM prediction (1-NN by summed distance)

The current GSH reads only item 2 (per-table labels), encoded
as multi-trit signatures. It ignores items 1, 3, 4, and 5.
That's a LOT of discarded information.

An FFN between LSH and GSH could COMBINE multiple items into
a richer representation. For example:
- Labels + distances: "table 3 voted Cat with high confidence
  (distance 2)" vs "table 7 voted Cat with low confidence
  (distance 14)." The label alone says "Cat" for both. The
  label + distance says "certain Cat" vs "uncertain Cat."
- Labels + vote counts: "prototype X was found by 12 tables
  and is labeled Cat" vs "prototype Y was found by 1 table
  and is labeled Dog." The vote count encodes retrieval
  strength that the per-table labels miss.

So the FFN isn't just a ternary matmul — it's a FUSION of
multiple routing signals into a single representation for the
GSH to hash.

But wait. I need to be careful here. The user said "route it"
repeatedly. An FFN with learned weights sounds like it's
importing neural network thinking. Is a ternary FFN genuinely
routing-native?

Let me think about what "routing-native" means for an FFN.

In a neural network, the FFN weights are learned via gradient
descent on a loss function. In our architecture, there are no
gradients and no loss function. The weights must come from
somewhere else.

Options:
(a) Random weights (cheapest, but we just learned random
    projections are wrong for spatial data — are they also
    wrong for routing data?)
(b) Routing-learned weights (generate candidates, measure
    which produce discriminative GSH inputs, select)
(c) Structured weights (hand-designed to compute specific
    cross-table features)
(d) Identity / no FFN (just pass the routing signals through
    to the GSH directly, but combine more of them)

Option (d) is interesting. What if the "FFN" isn't a matrix
multiply at all, but just a richer ENCODING of the LSH output?
The current GSH encodes only per-table labels. What if it
encoded labels + distances + vote counts + SUM prediction into
a single signature?

The encoding would be: for each table m, produce K trits that
encode BOTH the label AND the confidence:
- Trit 1-4: class label (4-trit codeword, as in the current GSH)
- Trit 5: distance quantized to ternary
  (+1 if distance < median, -1 if > 2×median, 0 otherwise)
- Trit 6: vote count of the 1-NN prototype quantized to ternary
  (+1 if high votes, -1 if low, 0 if medium)

This produces 6 trits per table instead of 4. At M=64:
384 trits = 96 bytes. The GSH hashes this richer representation.

This isn't an FFN in the neural network sense. It's a
MULTI-SIGNAL ENCODING of the routing output. But it serves
the same PURPOSE as an FFN: combining multiple attention
outputs into a richer representation for the next layer.

Actually, I think the user might be pointing at something
deeper. An FFN in a transformer has a specific structure:

    FFN(x) = W2 · σ(W1 · x + b1) + b2

The key is the NONLINEARITY σ between two linear transforms.
The first linear transform projects the input into a higher-
dimensional HIDDEN space. The nonlinearity creates new features
by thresholding. The second linear transform projects back down.

In ternary:
    FFN(x) = quantize(W2 · quantize(W1 · x))

Where quantize is ternary quantization (threshold → {-1,0,+1}).
W1 is an M×H ternary matrix (M routing dims → H hidden dims).
W2 is an H×G ternary matrix (H hidden dims → G output dims).
The ternary quantization after W1 IS the nonlinearity — it
collapses continuous values to three states.

The hidden dimension H can be LARGER than M (expansion) or
smaller (compression). In transformers, H is typically 4×
the model dimension. Here, M=64 → H=256 would be an
expansion into a space where cross-table correlations become
individual dimensions.

But this is a ternary matmul — the thing we just deprecated
for images. Is it appropriate for ROUTING signals?

The routing vector (M distances or labels) is fundamentally
different from a pixel vector:
- It's low-dimensional (64 dims, not 3072)
- It has no spatial structure to preserve
- Each dimension is already a measurement, not a raw value
- The meaningful information is in CROSS-DIMENSION patterns
  (which tables agree/disagree)

Random ternary projection of a 64-dim routing vector at
H=256 is 1:4 EXPANSION, not 1:192 compression. Every input
dimension contributes to every output dimension — there's
no information loss from subsampling. This is fundamentally
different from projecting 3072 pixels down to 16 trits.

So: ternary matmul on routing signals IS appropriate because:
1. The input is low-dimensional (no subsampling needed)
2. There's no spatial structure to preserve
3. The expansion into hidden dims creates cross-table features
4. The ternary quantization after expansion is a meaningful
   nonlinearity (thresholding cross-table agreement patterns)

OK, I think the FFN makes sense for routing signals. But I
want to be clear about what it ADDS that the current direct
encoding doesn't have.

Current GSH: each table's vote → 4 trits. Independent per
table. No cross-table interaction. The GSH's Hamming distance
counts per-table disagreements, which is a SUM over independent
table comparisons.

FFN GSH: the FFN mixes table signals BEFORE hashing. The
hidden features represent COMBINATIONS of table votes — "tables
3 and 7 agree on Cat," "the distance at table 12 is unusually
high." These combination features can be MORE discriminative
than individual table features.

This is the same argument for multi-layer representations in
neural networks: each layer creates features that are
COMBINATIONS of lower-layer features. The FFN creates cross-
table combination features from single-table measurements.

But here's my doubt: the current GSH already computes
HAMMING DISTANCE on the per-table encoding, which implicitly
counts cross-table disagreements. Two queries with the same
agreement pattern have low Hamming distance. The FFN would
create EXPLICIT cross-table features, but the Hamming
distance already captures cross-table DISAGREEMENT.

What the FFN adds: SPECIFIC cross-table combinations. The
Hamming distance treats all table disagreements equally. The
FFN (via its weight matrix) weights specific table combinations.
"Tables 3+7 disagreeing is important for Cat/Dog classification;
tables 12+15 disagreeing doesn't matter." The FFN encodes
which combinations matter.

This is analogous to attention vs FFN in transformers:
attention finds relevant context (which tables matter), FFN
processes it (which combinations of table signals are
informative). The current GSH has only the attention part
(find routing-similar images). The FFN adds the processing
part (detect specific routing patterns).

I think there's genuine value here. But the implementation
needs to be clear about:
1. What the FFN input is (per-table distances? labels? both?)
2. What the hidden dimension is (expansion ratio)
3. How the FFN weights are determined (random? routing-learned?)
4. What the FFN output feeds into (GSH bucket key? GSH score?)

And critically: should we build this BEFORE or AFTER
integrating direct ternary quantization into the LSH pipeline?
The direct quantization (50.2% CIFAR brute-force) is the
proven gain waiting to be deployed. The FFN is speculative.

I think the user sees the FFN as part of the FULL architecture
— the production system would have:
1. Direct ternary quantization (pixels + gradients → trits)
2. LSH on direct signatures (bucket index, multi-probe)
3. FFN bridge (mix LSH routing signals into cross-table features)
4. GSH on FFN output (match routing patterns)
5. Combination (LSH + GSH agreement)

All five layers are ternary-native. The FFN is the glue
between 2 and 4. Without it, the GSH reads a lossy encoding
of the LSH output. With it, the GSH reads a rich, mixed
representation that captures cross-table routing geometry.
