---
date: 2026-04-18
phase: RAW
topic: How SSTT best informs the Glyph LSH/GSH infrastructure
---

# SSTT informs Glyph — RAW

---

I need to stop trying to transplant SSTT into Glyph. The
interleave experiment proved that SSTT's specific design
choices (fixed thresholds, RGB interleave, block encoding,
IG-weighted inverted index) are tuned for SSTT's architecture,
not ours. Lifting individual features doesn't work because
they're coupled to SSTT's scoring mechanism.

So what SHOULD Glyph learn from SSTT? Not the specific
features. The PRINCIPLES.

What principles does SSTT embody that we haven't internalized?

**Principle 1: the trit is a measurement, not a compression.**

In SSTT, each trit represents a specific physical quantity —
"this pixel is bright," "this gradient points right," "this
block has more positive than negative." Each trit has MEANING.
The classifier reads those meanings.

In Glyph's random-projection era, each trit was a random
mixture — meaningless individually. We fixed this with direct
quantization. But even with direct quantization, our gradients
are computed AFTER normalization, so they measure "normalized
pixel difference" which is a statistical quantity, not a physical
one. SSTT's gradients are computed on the RAW ternary image
(after fixed-threshold quantization), so they measure "ternary
state change between adjacent positions" — a topological
quantity.

The difference: our gradients measure MAGNITUDE differences.
SSTT's gradients measure STATE TRANSITIONS. A gradient of +1
in SSTT means "the ternary state changed from X to Y." It's
categorical, not continuous. The clamp_trit(a-b) operation
on ternary inputs produces a TRANSITION CODE, not a magnitude.

This is deeper than I realized. SSTT doesn't compute gradients
on pixels and then quantize. It quantizes pixels FIRST, then
computes transitions on the quantized image. The transition is
ternary-native: it operates in trit space, not pixel space.

We do it backwards: normalize → compute gradient → quantize.
The gradient is a float-like quantity that gets quantized.
SSTT does: quantize → compute transition. The transition IS
ternary.

**Principle 2: the inverted index IS information-gain weighting.**

SSTT's inverted index stores, for each (position, value) pair,
which training images have that value at that position. The
information gain (IG) measures how much knowing the value at
this position reduces class uncertainty. High-IG positions
contribute more to the retrieval score.

This is analogous to our multi-table composition where each
table contributes equally. SSTT weights positions by
discriminability. We weight positions equally (Hamming distance
treats every trit the same).

The IG weighting is what makes SSTT's sparse signatures (67%
zeros) work — the 33% non-zero positions are weighted by how
informative they are. Our Hamming distance can't do this
because popcount treats every bit equally.

Could we introduce position-dependent weighting into the Glyph
pipeline? Not in the Hamming distance (that's a hardware
primitive). But in the RESOLVER — a weighted sum of per-position
distances instead of uniform Hamming. This would be the FFN
bridge: learn which trit positions matter.

But the user said no random projections for image data. A
position-weight vector computed from IG statistics on the
training set isn't a random projection — it's a routing-
learned measurement of discriminability.

**Principle 3: the block encoding captures SPATIAL PATTERNS.**

SSTT encodes 3-pixel strips as base-27 values. Each block
value represents a specific spatial pattern (e.g., "+1 0 -1"
= "bright, neutral, dark" from left to right). The inverted
index groups training images by which spatial patterns appear
at which positions.

Our direct quantization preserves individual pixel values but
doesn't capture PATTERNS. Two images with the same individual
trits but different arrangements would have the same Hamming
distance to a query, even though their spatial patterns differ.

The block encoding captures RELATIVE structure (the pattern
within a block) rather than ABSOLUTE values (individual trits).
This is why SSTT's signatures are more discriminative per byte
than our per-trit signatures.

Could we add block encoding to Glyph? The block values (27
possible per 3-trit block) could be encoded as 3 trits (which
they already are) and used as the signature. But our Hamming
distance already compares the individual trits within the block.
The block encoding adds value in SSTT because the inverted
index uses it as a LOOKUP KEY (exact match per block), not as
a distance metric.

In Glyph, the hierarchical summary ALREADY does something
similar — majority-voting within spatial blocks. But majority
voting collapses the within-block pattern to a single trit.
SSTT's block encoding PRESERVES the pattern as a multi-trit
symbol.

Wait. Our hierarchical summary is for BUCKET KEYS. The scoring
uses the FULL signature (per-trit Hamming). SSTT's block
encoding is for SCORING (inverted index lookup per block).
These are different roles.

The insight: SSTT uses block-level operations for SCORING.
We use block-level operations only for FILTERING (bucket keys).
If we brought block-level operations into SCORING — scoring by
"how many blocks match exactly" instead of "how many trits
differ" — we'd capture pattern structure.

Block-match scoring: for each block position, check whether
the query's block matches the candidate's block exactly (all
3 trits identical). Count the number of matching blocks.
Candidate with the most matching blocks wins.

This is a COARSER distance than Hamming (a block either matches
or doesn't — no partial credit for 2 of 3 trits matching). But
it's PATTERN-SENSITIVE (a block match means the spatial pattern
is identical, not just the individual values).

Actually, we can mix both: score = α × block_matches + β ×
(total_trits - hamming_distance). The block match score adds
pattern sensitivity; the Hamming score adds per-trit resolution.
The α/β weights could be learned from training data.

But this is getting complex. What's the SIMPLEST thing SSTT
teaches us?

**The simplest lesson: quantize FIRST, compute features SECOND.**

We currently: normalize pixels → compute gradients on floats →
quantize everything. SSTT: quantize pixels to ternary → compute
transitions on trits.

If we quantize the normalized pixels FIRST, then compute
gradients on the ternary image, the gradients are TRANSITIONS
between ternary states, not magnitude differences. The transition
between +1 and -1 is always 2 (or -2). The transition between
+1 and 0 is 1. Between 0 and 0 is 0. These are clean, discrete
values — not continuous magnitudes that need another quantization
step.

The gradient on the ternary image is ALREADY ternary (transitions
between 3 states produce values in {-2,-1,0,1,2}, which we can
clamp to {-1,0,+1} via clamp_trit). No separate gradient tau
needed. No tau=0 problem on MNIST. The gradient IS a trit
operation on trit data.

This is the fundamental lesson: SSTT's pipeline is
TRIT-NATIVE END TO END. Ours is pixel-native with a ternary
quantization boundary. Moving the quantization boundary EARLIER
(before gradients) makes the gradient computation trit-native.

Let me think about what this changes.

Currently: intensity trits have tau_intensity. Gradient trits
have tau_gradient (which is zero on MNIST — meaningless).

After fix: intensity trits have tau. Gradient trits are computed
FROM intensity trits — no separate tau. The gradient IS
clamp_trit(trit_a - trit_b). Always ternary. No calibration.

This eliminates the gradient tau parameter entirely. The
gradient is a deterministic function of the quantized image,
not a separately calibrated measurement. One fewer
hyperparameter. Cleaner pipeline.

And the gradient values are in {-2,-1,0,1,2} → clamped to
{-1,0,+1}. The structural zero in the gradient means "no
state transition between adjacent pixels" — genuinely
uninformative. Not "the magnitude difference was below an
arbitrary threshold."

This is the fix. Not SSTT's specific features. SSTT's
ORDERING of operations. Quantize first. Everything downstream
is ternary.
