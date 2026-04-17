---
date: 2026-04-17
phase: NODES
topic: GSH + LSH in concert — functions and relationship
---

# GSH + LSH in concert — NODES

---

## Node 1 — The GSH failed because it re-projected an already-clean signal

The one-hot routing pattern (640 binary dims) was projected
through random ternary weights to produce a 16-trit GSH
signature. This added noise to a signal that was already
class-structured. The GSH became a noisy copy of the LSH
instead of a different instrument.

## Node 2 — The GSH's voice should be TOPOLOGICAL, not geometric

LSH measures: "how far is this candidate in trit-lattice
pixel-projection space?" (geometric distance)

GSH should measure: "does this training image route the same
way as this query?" (topological identity)

These are fundamentally different questions. The combination
adds information only if the two voices are independent.

## Node 3 — The routing pattern IS a trit signature

Each table's vote can be quantized to a trit relative to the
consensus:
- +1: agrees with the plurality class
- -1: agrees with the runner-up class
- 0: votes for neither (noise)

This produces an M-trit signature (64 trits at M=64 = 16
bytes). The existing bucket index, multi-probe, and SUM
resolver work directly on this signature.

Multi-probe radius 1 in this space means: "what if one
table's vote changed?" — exploring the local neighborhood
in routing-pattern space.

## Node 4 — The GSH's Hamming distance IS routing-pattern similarity

The Hamming distance between two routing-pattern signatures is
the number of tables where the two images' votes disagree
(relative to their respective consensuses). Low distance =
similar routing patterns. High distance = different routing
patterns.

The SUM resolver over GSH signatures finds training images
with the most similar routing patterns to the query. k-NN
majority-votes their labels.

## Node 5 — The combination is non-redundant

LSH finds the geometrically nearest PROTOTYPES and reads their
labels. GSH finds the routing-pattern-nearest IMAGES and reads
their labels. A query that is geometrically near a Cat prototype
(LSH says Cat) but routes like Dog images (GSH says Dog) gets
BOTH signals. The combination resolves the tension.

Key: the GSH doesn't re-score the LSH's candidates. It finds
DIFFERENT candidates — training images with similar routing
patterns, which may not be in the LSH union at all.

## Node 6 — The plurality/runner-up quantization requires a pre-pass

To quantize table votes to trits, we need to know the
plurality and runner-up classes for this query. This requires
counting votes across all M tables BEFORE quantizing. The
pre-pass is O(M) per query — negligible.

For training images: the pre-pass uses each image's own
routing signature (computed from the LSH union, excluding
self). The plurality and runner-up are computed from the
M per-table labels.

## Node 7 — But the plurality IS the LSH's VOTE prediction

The plurality of per-table votes is exactly what the VOTE
resolver computes. And the runner-up is the second-most-voted
class. So the trit quantization is:
- +1: table agrees with VOTE prediction
- -1: table votes for the VOTE runner-up
- 0: table votes for something else

The GSH signature encodes "the pattern of agreement and
disagreement with the VOTE prediction." Two queries with the
same agreement pattern route similarly.

## Node 8 — Actually, the quantization should be query-independent

The problem with plurality-relative quantization: the plurality
depends on the query. Two queries with different pluralities
produce trit signatures in DIFFERENT reference frames. Table 3
voting "Cat" is +1 for a Cat-plurality query and -1 for a Dog-
plurality query. The signatures aren't comparable.

The GSH needs a FIXED reference frame so signatures are
comparable across queries. Options:

(a) **Fixed class assignment per trit position.** Position 0
means "agrees with class 0." M=64 tables × 10 classes = 640
trit positions. Each position is: +1 if that table voted that
class, 0 otherwise. But this is back to one-hot, which we
already tried.

(b) **Per-table fixed reference.** For each table, precompute
the most common class in the training set's per-table votes.
The trit is +1 if the vote matches the table's most common
class, -1 if it matches the runner-up, 0 otherwise. The
reference frame is fixed (computed from training data) and
consistent across queries.

(c) **Direct label encoding.** Encode each table's vote as a
small integer (0-9) and pack several votes into a trit
signature. E.g., quantize to ternary by (label mod 3) → trit
value. This preserves label identity but with collisions.

Option (b) is interesting: it creates a per-table "expected"
class and the trit measures deviation from expectation. But
the "most common class" per table is just the prior distribution
of the training set (10% per class for balanced CIFAR-10).
Every table's most common class would be the same. Not useful.

Option (c) is lossy — mod 3 maps classes 0,3,6,9 to the same
trit. Too much collision.

I think the answer is simpler than I'm making it. The routing
pattern is M integers, each in {0..9}. The GSH should find
training images with SIMILAR patterns — where "similar" means
"most tables agree on the label."

**The right encoding: treat each table's vote as a categorical
variable and define distance as the number of tables that
DISAGREE.** This is exactly Hamming distance if we encode each
vote as a symbol and compare symbol-by-symbol.

With 10 possible symbols per position and M=64 positions, the
natural distance is:

    d(q, t) = number of positions m where vote_q[m] != vote_t[m]

This is NOT trit Hamming distance (which has 3 symbols). It's
a 10-symbol categorical distance. But we can APPROXIMATE it
with trit Hamming by encoding each vote as multiple trits.

Actually, the simplest thing: encode each vote as a 4-trit
value (3^4 = 81 > 10, so 4 trits can represent 10 classes
without collision). M=64 tables × 4 trits/table = 256 trits =
64 bytes. The Hamming distance between two 256-trit signatures
counts per-trit disagreements, which approximates the per-table
vote disagreements.

256 trits = 64 bytes. The bucket index needs the first 16 trits
(4 bytes) as a key. Those 16 trits encode the votes of the
first 4 tables (4 trits each). Multi-probe flips individual
trits, which partially changes one table's encoded vote.

This is better than one-hot (which was 640 dims and needed
random projection to reduce). 256 trits are directly hashable
with the existing infrastructure.

## Node 9 — The functions of LSH and GSH, precisely stated

**LSH function:** map each query to a set of geometrically
nearby PROTOTYPES by hashing pixel signatures into buckets
and probing the neighborhood. Score by summed Hamming distance
across M pixel-projection tables. Output: a label based on
the nearest prototype(s) in pixel-signature space.

**GSH function:** map each query to a set of routing-pattern-
similar IMAGES by hashing vote-encoded signatures into buckets
and probing the neighborhood. Score by summed Hamming distance
across vote-encoded tables. Output: a label based on the
nearest image(s) in routing-pattern space.

**Relationship:** LSH finds prototypes the query LOOKS LIKE
(pixel geometry). GSH finds images the query ROUTES LIKE
(lattice topology). An image can look like a Cat (pixel-space
nearest neighbor is Cat) but route like a Dog (its routing
pattern matches Dog images). The combination resolves by
asking: does the geometric evidence or the topological evidence
better predict the true class?

## Node 10 — The combination should be a third routing decision

Not a hardcoded formula (2× LSH + 1× GSH + PTM votes). The
combination should ITSELF be routed: a third signal that
reads BOTH the LSH and GSH predictions and routes to the
final answer.

Simplest version: if LSH and GSH agree, accept. If they
disagree, defer to whichever has higher k-NN margin. The
margin is already computed in the k-NN resolver.

Or: take the class that appears in BOTH the LSH's top-K
and the GSH's top-K (intersection of the two ranked lists).

## Tensions

**T1:** How to encode the routing pattern — one-hot (Node 1,
failed), trit-quantized (Node 3), or multi-trit categorical
(Node 8)?

**T2:** The self-match problem for training routing sigs —
does excluding self from the LSH union produce a meaningfully
different routing pattern than including self?

**T3:** The combination formula — fixed weights, margin-based
arbitration, or intersection of ranked lists?
