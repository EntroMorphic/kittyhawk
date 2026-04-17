---
date: 2026-04-17
phase: RAW
topic: GSH + LSH in concert — functions and relationship
---

# GSH + LSH in concert — RAW

---

The results are in and the GSH is failing. Not at retrieval — the
oracle is 99.98% on all three datasets. At resolution. The GSH
finds the right neighborhood in routing-pattern space but can't
pick the right answer from it. And the combination formula is
making things worse, not better.

I need to think about WHY the GSH resolver fails before trying to
fix it. The user said "Rhythm & Blues" — two instruments in harmony.
Right now it's one instrument (LSH) and a broken speaker (GSH)
producing static that drowns out the music.

What does the LSH do well? It takes a query, projects it through M
random ternary directions, hashes the signature into buckets, finds
nearby prototypes, and scores them by summed Hamming distance across
M tables. The SUM resolver reads M independent geometric measurements
and aggregates them. Each measurement is a direct comparison between
the query and the candidate in a specific projection. The SUM is
COHERENT — every table is comparing the same query-candidate pair,
just through a different lens.

What does the GSH do? It takes the routing PATTERN (which classes
each LSH table voted for) and hashes THAT into a second set of
buckets. It finds training images whose routing patterns match.
Then it scores them by summed Hamming distance in the routing-
pattern signature space.

The problem: the GSH's scoring has TWO levels of indirection. Level
1: the LSH routing pattern is a lossy summary of the LSH's geometric
measurements (labels only, no distances). Level 2: the GSH projects
the one-hot encoding through ANOTHER set of random ternary weights,
compressing 640 binary dims to 16 trits. Each trit in the GSH
signature is a random combination of one-hot positions — a random
combination of "did table m vote class c?" flags.

That second random projection is the noise source. The one-hot
vector is ALREADY ternary-structured (binary, low-dimensional,
class-meaningful). Projecting it through random weights ADDS noise
to an already-clean signal. It's like taking a clear photograph,
converting it to text, then converting the text back through a
random cipher. Each conversion loses information.

What if the GSH didn't project at all? What if the routing pattern
IS the signature — no second-layer random projection? The one-hot
vector has 640 binary dimensions (at L1_M=64). That's too large
for a uint32 bucket key (4 bytes = 16 trits). But it COULD be
hashed directly — a hash of the 640-bit vector into a 32-bit
bucket key. The hash is not a random projection; it's a
deterministic mapping that preserves exact matches.

But exact-match hashing in 640 dims would produce one unique
bucket per distinct routing pattern. With 10^64 possible patterns
and 60K training images, every training image has a unique pattern.
No collisions → no bucket mates → no candidates. Multi-probe
would need to flip individual bits in the 640-dim vector to find
neighbors, which is exponential.

So the GSH NEEDS some form of dimensionality reduction to make
bucket lookup feasible. Random projection is one form. Others:
- Hash only a SUBSET of the one-hot positions (like the dimension
  subsetting idea)
- Use the class DISTRIBUTION (10-dim histogram of votes) as the
  hash input, not the full one-hot
- Fold the one-hot by XOR (hash it without projecting)

Wait. The class vote histogram is a 10-dimensional integer vector.
Table m voted class c → histogram[c]++. The histogram summarizes
the routing pattern as "how many tables voted for each class."
This is MUCH lower-dimensional (10 vs 640) and directly class-
structured.

A 10-dim integer vector can be ternary-projected at N_PROJ=16
with 1:0.6 compression — extremely dense. The projection would
capture "class A got more votes than class B" relationships,
which is exactly what we want for classification.

But wait — we ALREADY HAVE a resolver that reads the vote
histogram. It's the VOTE resolver: argmax of vote-weighted class
counts. VOTE at M=64 gets 89.77% on MNIST. That's the same
information the histogram-based GSH would route through.

So the histogram GSH would be routing through information that
the VOTE resolver already reads. The VOTE resolver reads it
DIRECTLY (argmax of counts); the GSH would route through it
(project → hash → find similar histograms → k-NN). The GSH
adds a layer of indirection to information the VOTE resolver
already processes optimally.

This means: the GSH can only add value if it reads DIFFERENT
information than what the LSH's resolvers already have. The
LSH's SUM resolver reads per-table distances. The VOTE
resolver reads per-table labels. The PTM resolver reads
per-table 1-NN labels. The GSH needs to read something NONE
of these read.

What does the routing pass produce that no resolver currently
reads?

1. **Union membership pattern.** WHICH prototypes are in the
   union. Not their labels, not their distances — their IDENTITY.
   Two queries that pull the same set of prototypes into their
   unions are likely the same class, even if the SUM ranking
   differs. The union membership is a binary vector over N_train
   (very high-dimensional but very sparse).

2. **Cross-table agreement structure.** Not just "how many tables
   voted class c" but "WHICH tables voted class c." Table
   agreement patterns encode which subspaces of the lattice see
   the query-class relationship. Two queries with the same
   agreement pattern (tables 0,7,23 vote Cat; tables 3,12 vote
   Dog) are geometrically similar in a way that vote counts
   can't capture.

3. **Distance profile shape.** The DISTRIBUTION of distances at
   each table — not just the 1-NN distance, but the full histogram.
   A table where the 1-NN is at distance 5 and the 2-NN is at
   distance 20 has a sharply peaked profile (high confidence).
   A table where 1-NN and 50-NN are both at distance 8 has a
   flat profile (low confidence). The profile shape encodes local
   lattice geometry that the SUM resolver (which reads only the
   1-NN distance) discards.

Of these three, #1 (union membership) is the most fundamental.
The union is the RESULT of the LSH's routing — it's the lattice's
answer to "who is near this query?" Two queries with similar
unions are similar in the lattice's view, regardless of how the
resolver ranks the candidates.

But union membership is N_train-dimensional and unique per query
(each union has ~5K-13K members out of 50-60K). Hashing it
directly isn't feasible. We'd need to summarize it.

Summarization options for union membership:
- Class distribution in the union (N_CLASSES-dim histogram)
- Top-K most-voted prototypes (their indices, hashed)
- Per-class nearest-neighbor index (N_CLASSES prototype indices)
- Binary vector hashed via MinHash (constant-size signature of
  set membership)

MinHash is interesting. It produces a fixed-size signature of a
set such that the Jaccard similarity of two sets is approximated
by the fraction of matching signature positions. The signature
is computed by hashing each set element and taking the minimum
hash per position. It's routing-compatible: the hash can be
computed with integer arithmetic, the comparison is exact match
per position.

A MinHash signature of the union membership would capture "these
two queries have similar candidate sets" in a fixed-size key
suitable for bucket lookup. The GSH would find training images
whose union membership overlaps with the query's union membership.
Those are images the lattice routes to the same neighborhood.

But MinHash requires multiple hash functions (one per signature
position) and the hash is over PROTOTYPE INDICES, not over trit
signatures. It's routing-adjacent (uses the LSH union) but not
trit-lattice-native (the hash is on indices, not on trit
distances).

I'm overcomplicating this. Let me go back to the user's framing.

"Like Rhythm & Blues." Two instruments, each with its own voice,
complementing each other. The LSH's voice is GEOMETRIC — it
measures distances in trit-lattice space. The GSH's voice should
be TOPOLOGICAL — it measures which neighborhoods the query falls
in, not how far away things are.

GEOMETRIC (LSH): "this candidate is at summed distance 47."
TOPOLOGICAL (GSH): "this query routes to the same neighborhoods
as training image 12345, which is class Dog."

The LSH measures distance. The GSH measures neighborhood identity.
They're different measurements. The combination says: "the nearest
candidate geometrically is Cat (LSH) but the query routes like a
Dog (GSH) — which do we trust?"

For this to work, the GSH needs to identify neighborhood
IDENTITY, not re-measure neighborhood DISTANCE. The one-hot
encoding was trying to capture identity (which class each table
voted for) but then re-projected through random weights, which
converted the identity signal back into a (noisy) distance
signal. The GSH ended up being a bad version of the LSH instead
of a different instrument.

THE GSH SHOULD NOT USE RANDOM PROJECTION. It should use
EXACT or NEAR-EXACT matching on the routing pattern.

Concretely: the GSH should hash the routing pattern into a
bucket key such that training images with IDENTICAL routing
patterns land in the same bucket. Multi-probe expands to
SIMILAR routing patterns (one table vote different).

The routing pattern is M labels, each in {0..9}. At M=64,
that's 64 digits in base 10 — astronomically many possibilities.
But in practice, many patterns are common (e.g., "most tables
vote 8" for Ship images). A hash of the pattern into a 32-bit
key, with multi-probe flipping one table-vote at a time, would
find training images with nearly-identical routing patterns.

This is EXACTLY what LSH does — but the "signatures" are the
routing labels, not pixel projections, and the "Hamming
distance" is the number of tables that disagree. The GSH IS
an LSH, but operating on routing patterns instead of pixel
signatures.

The existing multi-probe infrastructure works: "radius 1" in
routing-pattern space means "change one table's vote." That's
exactly glyph_multiprobe_enumerate applied to a signature where
each trit represents a table's vote (quantized to ternary from
10-class labels).

But 10 classes don't map to 3 trit values. We'd need to
quantize the 10 labels to {-1, 0, +1} somehow. Or use a
different encoding.

Actually, the simplest approach: use the VOTE PATTERN ITSELF
as a packed-trit signature. Quantize each table's vote to a
trit:
- If the vote matches the plurality class: +1
- If the vote is "other" (neither plurality nor runner-up): 0
- If the vote matches the runner-up class: -1

This produces a meaningful M-trit signature: +1 = agrees with
consensus, -1 = disagrees specifically, 0 = noise. Two queries
with the same agree/disagree pattern are routing-similar.

At M=64, this is a 64-trit signature = 16 bytes. The existing
bucket index handles 4-byte keys → use the first 16 trits
(tables 0-15) as the bucket key and the full 64 trits for
scoring. EXACTLY the same filter-ranker decomposition the
LSH uses.

And multi-probe at radius 1 flips one trit from +1 to 0 or
-1, which means "what if one table's vote changed?" This
explores the local neighborhood in routing-pattern space.

THIS is the GSH voice: not re-projecting the routing pattern
through random weights, but treating the routing pattern
ITSELF as a trit signature and hashing it directly.

The GSH's Hamming distance is "how many tables disagree
between two images' routing patterns." Two images with
Hamming distance 0 route identically. With distance 5,
five tables disagree. The SUM resolver over GSH tables
scores by "how similar are the routing patterns overall."

This is beautiful. The LSH measures geometric distance in
pixel-projection space. The GSH measures topological distance
in routing-pattern space. Same infrastructure. Different input.
Rhythm and Blues.
