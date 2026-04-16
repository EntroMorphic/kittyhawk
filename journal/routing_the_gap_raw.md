---
date: 2026-04-16
phase: RAW
topic: Routing the CIFAR-10 gap — from 37.90% to 53%+ without leaving the lattice
---

# Routing the gap — RAW

Unfiltered thinking. Honest. Doubts included.

---

I keep reaching for pixel-space solutions and the user keeps
pulling me back. PCA — no. Centroids — no. Structured features
— "we should be able to route the solution." That's the thesis
constraint and I need to honor it.

What does "routing the solution" mean concretely? The routing
architecture has proven mechanisms:

1. Ternary projection → packed-trit signature
2. Signature-as-address → bucket lookup
3. Multi-probe → neighborhood expansion
4. Multi-table composition → union construction
5. Multi-resolution re-rank → combined scoring

Every one of these is routing-native. No float, no pixel math.
And together they get us to 37.90% on CIFAR-10. SSTT gets 53%
with structured features. The gap is 15pp.

Let me think about what SSTT ACTUALLY does, reframed as routing.

SSTT's retrieval stage uses an inverted index over block-encoded
ternary signatures. Each block is a 3-pixel horizontal strip
encoded as a ternary symbol. The block encoding is a SPATIAL
ROUTING DECISION — it says "these three adjacent pixels form
this ternary pattern." The inverted index groups training images
by which block patterns they contain. Retrieval finds training
images with similar block patterns.

Our random projections don't make spatial routing decisions.
Each projection direction is a random linear combination of ALL
3072 pixels. The spatial structure — edges, textures, gradients
— gets averaged away.

But here's the thing: the user said "route the solution." What
if the routing architecture itself can discover spatial structure
WITHOUT importing hand-crafted features?

How would the routing discover spatial structure?

Idea 1: ROUTE THROUGH PIXEL SUBSPACES.

Instead of projecting all 3072 dimensions simultaneously, project
SUBSETS of dimensions. Table 0 projects pixels 0..255 (top-left
quadrant). Table 1 projects pixels 256..511. Table 2 projects
pixels 512..767. Etc. Each table sees a SPATIAL REGION, not the
full image.

The multi-table composition already handles combining M
independent projections. If each projection sees a different
spatial region, the union contains candidates that match the
query in DIFFERENT parts of the image. The resolver scores
by summed distance across ALL tables — effectively asking
"does this candidate match in the top-left AND bottom-right
AND center?"

This is SPATIAL ROUTING. Each table routes through a different
region of the image. The composition combines the spatial routes.
No hand-crafted features. No pixel-space computation. Just a
spatial mask on which input dimensions each projection reads.

Wait, I proposed this earlier as "block-structured spatial
projections" in Phase B.2 and the user seemed interested but
we pivoted to the dynamic cascade. Now the cascade exists and
the projection is the remaining bottleneck. This is the right
time.

The implementation is trivial: in glyph_sig_builder_init, instead
of generating random weights over ALL input_dim dimensions,
generate random weights over a SUBSET of dimensions. The
subset is determined by the table index: table m uses
dimensions [start_m, end_m). The existing ternary matmul works
unchanged — we just zero out the weight rows outside the subset,
or equivalently, pass a shorter input vector and a shorter
projection matrix.

But wait — if each table only sees 256 of 3072 dimensions (a
1/12 spatial region), the per-table projection has much less
input to work with. At N_PROJ=16 with 256 input dims, the
projection is much denser (16 trits from 256 dims vs 16 trits
from 3072 dims). That might be BETTER — the projection is more
concentrated on a smaller spatial region and less likely to wash
out local structure.

Idea 2: ROUTE THROUGH MULTI-SCALE REGIONS.

Don't just partition the image into fixed regions. Use overlapping
regions at multiple scales:
- Table 0-3: quadrants (4 × 768 dims each)
- Table 4-7: halves (2 horizontal, 2 vertical, 1536 dims each)
- Table 8-11: thirds (3 × 1024 dims)
- Table 12-15: full image (3072 dims)
- Table 16+: random subsets

This gives each table a different spatial view. The full-image
tables provide global context. The quadrant tables provide local
detail. The resolver combines them.

This is STILL routing — each table routes through a different
spatial subspace of the lattice. The multi-table composition
is the mechanism that fuses the views.

Idea 3: ROUTE THROUGH COLOR CHANNELS.

CIFAR-10 images are 32×32×3 (RGB). The current projection
treats all 3072 values uniformly. But color channels carry
different information:

- R channel (dims 0..1023): red intensity
- G channel (dims 1024..2047): green intensity  
- B channel (dims 2048..3071): blue intensity

Routing through individual channels means some tables project
only the R channel, others only G, others only B. A Ship (blue
sky/water) would produce distinctive signatures on the B channel
that R and G don't capture. A Frog (green) would light up the
G channel. The channel-specific routing exploits color structure
that the mixed-channel projection averages away.

Idea 4: ROUTE, MEASURE, RE-ROUTE.

The dynamic cascade already does this for N_PROJ escalation.
What if we applied the same pattern to SPATIAL REGIONS?

Pass 1: route through the full image (current behavior). Build
union. Identify which class pairs are confused (from the vote
pattern or the top-2 candidates).

Pass 2: for the confused pair (class_i, class_j), route through
the spatial region that most distinguishes them. Which region?
The one where class_i and class_j prototypes have the most
different signatures.

But finding "which region distinguishes them" requires measuring
across the training set — that's the cross-query geometry
accumulation the user pointed at earlier.

Actually, this could be PRECOMPUTED. For each class pair (i, j),
precompute which spatial region has the highest inter-class
distance. Store this as a lookup table: pair(i,j) → best_region.
At test time, after Pass 1 identifies the ambiguous pair, look
up the best region and re-route through it.

This is routing the solution: routing measurements on the training
set inform routing decisions at test time. The lookup table is
LEARNED FROM ROUTING, not from pixel-space computation.

But the lookup table is indexed by class pair, which means we
need the class prediction from Pass 1 to select the region.
Circular? Not quite — Pass 1 identifies the top-2 candidates
(not the true class), and the region lookup helps disambiguate
between them.

Idea 5: the simplest possible spatial routing.

Just partition the image into non-overlapping blocks and assign
each block to a table. CIFAR-10 is 32×32×3 = 3072 pixels. With
M=64 tables, each table gets 3072/64 = 48 pixels. That's too
few for a meaningful 16-trit projection.

Better: with M=16 tables, each gets 192 pixels. Or M=12 tables
for 256 pixels each (a meaningful chunk). Then add more tables
at wider regions or full-image for global context.

But the current filter needs M tables for multi-probe to work.
At M=16 (our filter M), each table gets 192 pixels. N_PROJ=16
over 192 dims is a 1:12 compression. That's similar to the
MNIST configuration (N_PROJ=16 over 784 dims, 1:49 compression)
where we got 97%.

So the DENSITY of the projection matters. At 1:12 compression,
the projection sees more of the local structure. At 1:192
(full CIFAR-10 image), it averages over too much.

This is the key insight: the per-table compression ratio determines
how much local structure the projection preserves. Random projections
over 3072 dims at N_PROJ=16 are 1:192 — too sparse. Random
projections over 192 dims at N_PROJ=16 are 1:12 — dense enough
to capture local structure.

THE FIX IS NOT WIDER PROJECTIONS (we proved N_PROJ=1024 saturates).
THE FIX IS NARROWER INPUT per table — spatial routing that gives
each table a compact, locally-coherent input space.

OK, I think Idea 5 is the right one. Let me stress-test it.

Doubt 1: spatial partitioning for CIFAR-10 means partitioning
RGB pixels. A 4×4 block of a 32×32 image is 48 pixels (16 per
channel). That's very few. A 8×8 block is 192 pixels (64 per
channel). With M_filter=16 tables, 16 tables × 192 pixels = 3072.
Perfect partition.

Doubt 2: the spatial partition breaks the bucket index invariant.
Currently all tables project the SAME query vector (3072 dims)
through different projection matrices. With spatial partitioning,
each table projects a DIFFERENT subset of the query vector. The
existing glyph_sig_builder already handles this — it takes
input_dim as a parameter. We'd just build each table's builder
with a different input_dim and feed it the corresponding pixel
subset.

But the glyph_sig_encode function takes a contiguous m4t_mtfp_t*
input of length input_dim. We'd need to extract the spatial
subset into a contiguous buffer before encoding. Small overhead.

Doubt 3: will the multi-probe work correctly? Multi-probe
enumerates neighbors in trit space, not pixel space. If each
table's signatures encode a different spatial region, the
multi-probe neighborhoods are spatial-region-specific. A radius-1
probe on table 0 (top-left region) expands in the top-left
lattice, while the same radius on table 1 (top-right) expands
in the top-right lattice. The union composition merges candidates
found in different spatial neighborhoods. This is exactly right
— a candidate appears in the union if it matches the query in
ANY spatial region.

Doubt 4: the resolver needs to sum distances across tables, but
now each table scores on a different spatial region. The sum
is "how well does this candidate match across all spatial regions"
— which is what we want. The normalization should be per-table
(divide by the region's N_PROJ) so that smaller regions don't
dominate.

Actually, if all tables have the same N_PROJ=16, the distances
are already on the same scale (0..32 per table). No normalization
needed. The SUM naturally combines spatial matches across regions.

I think this works. The implementation is:
1. Partition 3072 dims into 16 groups of 192 dims.
2. Build each table's sig_builder with input_dim=192 and a
   192-dim subset of the training data.
3. At query time, extract each table's 192-dim subset from the
   query vector.
4. Everything else (bucket, multi-probe, resolve) is unchanged.

The only new code is the spatial partitioning and per-table
subset extraction. ~30 lines.

And it composes with the multi-resolution re-rank: filter at
N_PROJ=16 per spatial region, re-rank with wider N_PROJ at
selected regions. The dynamic cascade becomes a spatial-
resolution cascade.

Let me think about whether this is really "routing the solution"
in the user's sense. The spatial partition is a design choice —
I'm DECIDING which pixels go to which table. That's not the
routing discovering the structure; it's me imposing structure.

The routing-pure version would be: let the tables partition
themselves based on which pixels produce the most discriminative
signatures. But that requires an optimization loop — try random
partitions, measure accuracy, keep the best ones. That's
routing-native (no pixel math, only routing measurements) but
expensive.

The compromise: start with a FIXED spatial partition (non-
overlapping 8×8×3 blocks). Measure accuracy. If it helps, then
optimize the partition using routing measurements. The fixed
partition is the hypothesis; routing-based optimization is the
follow-up.

Actually, the SIMPLEST spatial routing that requires NO design
choices: RANDOM SUBSETS of dimensions instead of spatial blocks.
Each table projects a random subset of 192 dimensions (out of
3072). This is like random projections but with a sparsity
constraint — each weight row has exactly 192 non-zero entries
and 2880 zeros. The subset is random, not spatial.

Why might random subsets help? Because each table sees DIFFERENT
pixels, not the same pixels through different weights. Two tables
with random-subset selection have ~(192/3072)² ≈ 0.4% pixel
overlap. They're looking at almost completely different parts of
the image. Two tables with full-dimension random projection have
100% pixel overlap — they just weight them differently.

Random-subset selection is trivially routing-native: the subset
mask is generated by the same RNG that generates the projection
weights. No spatial knowledge imported.

Hmm, but random subsets aren't spatial. They'd pick a random
scattering of pixels from across the image. That's not
"spatial routing" — it's just sparser projection.

Unless... the random subsets happen to correlate with spatial
structure because nearby pixels in CIFAR-10 are correlated.
If a random subset includes 3 adjacent pixels from the sky
region and 2 adjacent pixels from the ground, the projection
might capture "sky vs ground" structure that a full-image
projection washes out.

I'm going in circles. Let me crystallize:

The simplest routing-native approach that requires no design
choices is random dimension subsets. The approach that maximally
leverages spatial structure is fixed spatial blocks. The
routing-pure approach is learned spatial routing via routing
measurements. Start with random subsets (simplest, no design
choices), measure, then try spatial blocks, then optimize.
