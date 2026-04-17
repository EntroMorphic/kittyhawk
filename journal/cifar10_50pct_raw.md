---
date: 2026-04-17
phase: RAW
topic: Closing the CIFAR-10 gap to >50%
---

# CIFAR-10 >50% — RAW

---

I need to be honest about where we are and what we know.

We're at 38.14% on CIFAR-10 with brute-force M=64 N_PROJ=64 k=5.
SSTT gets 53% on the same dataset. The gap is 15pp. Every resolver
improvement, cascade design, and subsetting experiment we've tried
has moved the needle by at most 2-3pp. We need 15pp. That's a
different order of magnitude.

What we've proven can't close the gap:
- Wider N_PROJ (saturates, actually hurts past 64)
- More tables (M=64 vs M=8 adds 3pp at N_PROJ=64, but we're
  already at M=64)
- Better resolvers (k-NN adds ~1.7pp over 1-NN)
- Multi-resolution combined scoring (+2.6pp but dilutes past N_PROJ=64)
- Dynamic cascade (matches brute-force ceiling, not the bottleneck)
- Density variation (0.5pp between best and worst density)
- Dimension subsetting (random subsets hurt, spatial blocks hurt more)
- Margin-weighted resolver (anti-correlates with correctness)

What we haven't tried:
1. N_PROJ=64 with the FULL LSH architecture (bucket index on first
   16 trits, resolve on all 64). This could be better than
   brute-force if the filtered union is smaller and less noisy.
2. Structured projections (SSTT's approach — but the user says
   "route the solution")
3. Cross-query routing patterns (the user's geometry vision)
4. Second-layer routing (route through vote patterns)
5. Per-class or per-confusion-pair specialized tables

Let me think about what SSTT actually does that gets 53%.

SSTT uses:
- Ternary quantization of pixel intensity + gradients (3 channels)
- Block encoding (3-pixel strips → ternary symbols)
- Inverted index with information-gain weighting
- Structural ranker with topological features

The first two items are FEATURE EXTRACTION — they transform the
raw RGB into a more discriminative ternary representation BEFORE
indexing. Our random projection is both the feature extraction AND
the indexing in one step. SSTT separates them.

What if we separated them too? Not by importing SSTT's specific
features, but by using the routing architecture to build better
input representations?

Here's what I mean: right now we project raw 3072-dim RGB → 16/64
trit signature. What if we first ROUTED the 3072-dim input through
a coarse projection to produce an intermediate representation,
then projected THAT through a second stage?

Layer 1: 3072 RGB → M₁ tables × N₁ trits → M₁ per-table 1-NN
labels → M₁-dimensional "vote vector"
Layer 2: M₁-dim vote vector → M₂ tables × N₂ trits → classification

The vote vector from Layer 1 is a ROUTED representation of the
input — each component says "table m thinks this image is class c."
It's lower-dimensional (M₁=64 vs 3072), discrete (10 classes per
component), and routing-derived. Projecting THAT through a second
routing layer compresses routing measurements, not raw pixels.

But I proposed this before (in the lattice geometry nodes) and
didn't build it because the encoding (integer class labels → MTFP
for ternary projection) wasn't natural. Let me think harder.

Actually, one-hot encode the vote vector: M₁=64 tables × 10
classes = 640 binary dimensions. Each dimension is 0 or 1: "did
table m vote for class c?" This is a natural input for ternary
projection — it's already binary, density is well-defined, and
the projection would capture correlations between table-class
votes.

At M₁=64, the 640-dim one-hot vote vector is MUCH smaller than
3072-dim RGB. Ternary projection of 640 dims at N_PROJ=16 is 1:40
compression — same ballpark as MNIST (784→16 = 1:49) where we get
97%. And the input is ROUTING-DERIVED, not raw pixels.

BUT: this requires computing Layer 1's vote vector for every
training prototype too, so Layer 2 can build its index. That means
routing all 50K training images through Layer 1 — which is exactly
the brute-force pass we just ran. Expensive but one-time.

Wait — there's a subtlety. When computing the Layer 1 vote vector
for training image i, the Layer 1 index contains image i itself.
Image i will be its own nearest neighbor at distance 0 in every
table. We need to exclude the self-match, or the vote vector will
always be "correct class in every table" which carries no
information.

Leave-one-out: for each training image i, find the 2nd-nearest
(excluding self). This is easy if we're brute-forcing but hard
with the bucket index (would need to track whether the 1-NN is
self).

Alternative: split the training set into two halves. Build Layer 1
on half A, compute vote vectors for half B. Build Layer 2 on half
B's vote vectors. At test time, Layer 1 uses the full training set
(no self-match issue for test queries). This halves the effective
training set, which might hurt.

Or: just accept the self-match bias. The vote vector for training
image i will have the correct class over-represented (because
self-match always votes correct), but the PATTERN of which OTHER
tables agree/disagree still carries information. Layer 2 would
learn to discount the inflated correct-class signal and read the
agreement pattern. Noisy but maybe workable.

Actually, the simplest version: don't one-hot encode. Just use
the per-table DISTANCE to the 1-NN (not the label) as the feature.
For each training image i at each table m, the distance to the
nearest non-self neighbor is an integer in [0, 2*N_PROJ]. M=64
tables → 64-dimensional integer vector. Project this through
Layer 2.

This is pure routing geometry: the feature vector IS the routing
pass's distance measurements. Layer 2 routes through the routing
geometry to classify.

Hmm, 64 dimensions is small for ternary projection. N_PROJ=16
over 64 dims is 1:4 compression — very dense. Maybe too dense
(projection can't compress much further).

Let me step back. The two-layer approach is interesting but
complex. What's the SIMPLEST thing that could get us to 50%?

The brute-force data shows N_PROJ=64 M=64 k=5 at 38.14%. That's
with density=0.33. We haven't swept density at N_PROJ=64 M=64.
On Fashion-MNIST, density 0.25 beat 0.33 by 0.39pp. On CIFAR-10
at N_PROJ=16 M=64, density variation gave ~1pp. At N_PROJ=64 the
effect might be different.

But 1pp from density won't get us to 50%. We need 12pp.

What about k? We used k=5 (rank-weighted). The scaffolding tools
found k=5 was the sweet spot on MNIST. On CIFAR-10 with N_PROJ=64
M=64, the brute-force gives 36.45% 1-NN and 38.14% k=5. That's
+1.69pp from k=5. What about k=10, k=20, k=50?

With 50K training prototypes and M=64 tables, the summed-distance
distribution has many candidates at similar distances. A larger k
might help if the correct-class concentration increases in the
top-20 or top-50. On MNIST, k=7 was slightly worse than k=5. But
CIFAR-10's distance distribution is different (denser, more ties).

Let me try k=20 and k=50 in the brute-force tool. Quick
measurement, no code changes needed (just change KNN_K).

Wait, KNN_K is hardcoded to 5 and the top-K buffer is sized to
64. I can change it to sweep k values. But this is a small mod.

Actually, there's a more fundamental question. The brute-force
at M=64 N_PROJ=64 scores each candidate by summing 64 Hamming
distances (range 0..128 per table, total range 0..8192). The
k-NN votes over the top-k by this sum. But the sum treats all
64 tables equally — some tables might be noise at N_PROJ=64.

We proved that margin-weighting hurts at N_PROJ=16 (decisive
tables are less accurate). But we didn't test at N_PROJ=64.
At wider N_PROJ, the tie rate is lower and the decisive tables
might actually be more accurate. The decisive-subset diagnostic
should be re-run at N_PROJ=64.

But even if margin-weighting helps at N_PROJ=64, it's unlikely
to add 12pp. The interventions that can add 12pp are structural,
not parametric.

Let me think about what 12pp means. Going from 38% to 50% means
correctly classifying 1200 additional images out of 10000. That's
1200 images that are currently wrong AND have a path to being
correct within the routing architecture.

The cross-seed overlap showed 61.6% of queries are ALWAYS wrong
(seed-invariant at N_PROJ=16 M=64). That's 6161 images. We need
to rescue 1200 of them. That's 19.5% of the always-wrong set.

Are those 1200 images rescuable within the routing architecture?
The oracle is 100% — the correct training neighbor is always in
the union (or within brute-force distance). The problem is
ranking. So yes, they're rescuable if we can rank correctly.

What does "rank correctly" require? The correct-class nearest
neighbor must have a lower summed distance than the winning
confuser. At N_PROJ=64 M=64, the sum ranges from 0 to 8192.
The gap between the correct-class best and the overall winner
(from the atomics) averages ~15 Hamming units. If we could find
projections where the correct-class neighbor is closer by even
1 unit per table, 64 tables would produce a 64-unit swing — more
than enough to reverse the ranking.

The question is: do such projections EXIST in the random space
we're sampling from? The seed-overlap says errors are seed-
invariant — three different random projections see the same
images as unclassifiable. This suggests the projections that
would correctly classify those images are NOT in the random
space we're drawing from. Random ternary projections over
3072-dim RGB structurally cannot separate Cat from Dog for
a large fraction of CIFAR-10 images.

So the answer is: random projections have a ceiling around
38% on CIFAR-10, and no amount of routing architecture
sophistication can exceed it. To reach 50%, we need
non-random projections.

But the user says "route the solution." How do we get
non-random projections through routing?

Answer: the routing architecture discovers which projections
work by measuring itself. Build many random projections, route
training data, measure per-table per-class accuracy, keep the
projections that work, discard the ones that don't.

This is ROUTING-NATIVE FEATURE SELECTION:
1. Generate 1000 random projection directions.
2. For each direction, compute the per-class accuracy on the
   training set (using the other directions for scoring).
3. Keep the 64 most discriminative directions.
4. Build M=64 tables using only those 64 directions.

The selection is based on routing measurements (Hamming distance
accuracy), not pixel-space computation. The directions are still
ternary. The architecture is still routing-native. But the
directions are CURATED by routing, not random.

This is analogous to feature selection in classical ML — but the
selection criterion is routing accuracy, not mutual information or
correlation. The routing measures itself and keeps what works.

How much could this help? If the top 64 out of 1000 random
directions are meaningfully more discriminative than a random
set of 64, the accuracy should improve. The question is: what's
the variance of per-direction discriminability? If all random
directions are equally bad (uniformly ~10% accuracy on CIFAR-10),
selection doesn't help. If some random directions happen to align
with discriminative axes (edge detectors, color channels, texture
gradients), selection would find them.

Random ternary projection over 3072 dims: each direction is
~1024 random {-1,0,+1} weights. The dot product is sum(w_i × x_i)
for 1024 non-zero w_i. This is a random linear combination of
~1024 pixels. SOME such combinations will correlate with
class-discriminative features (e.g., a direction that weights
sky pixels positive and ground pixels negative would correlate
with "has sky" → Airplane/Ship vs "no sky" → Cat/Dog). Most
won't. Selection would find the ones that do.

I think this is the path. But measuring per-direction accuracy
on the training set requires a brute-force pass per direction.
1000 directions × 50K training images × 50K scoring = expensive.

Cheaper version: measure per-direction CLASS SEPARABILITY, not
accuracy. For each direction, compute the mean projection value
for each class. If the class means are well-separated along this
direction, it's discriminative. This is computable from the
training set in O(N_train × N_directions) — linear, not quadratic.

Class separability of a direction w:
    sep(w) = Σ_{i≠j} |μ_i(w) - μ_j(w)| / σ(w)

where μ_i(w) is the mean of w⋅x for class i, σ(w) is the
overall stddev. Directions with high sep(w) separate classes.

This is a routing-compatible computation: the projections are
ternary matmuls (m4t_ternary_matmul), the means are integer
sums. No float needed.

But wait — this is Fisher's Linear Discriminant computed on the
ternary projection outputs. It's a well-studied method. The issue
is it uses class labels, which means it's supervised feature
selection. The user might consider this "not routing."

Actually, the user's SSTT repo uses class labels too (inverted
index with information-gain weighting). And our existing system
uses class labels in the resolver (y_train). Supervised is fine;
the question is whether the computation is routing-native.

Ternary matmul → integer sum per class → class separability
score → sort → keep top 64. Every step uses routing primitives.
No float, no gradient, no pixel-space distance.

I think this is the move. Let me crystallize it in NODES.
