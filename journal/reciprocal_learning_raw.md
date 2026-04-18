---
date: 2026-04-17
phase: RAW
topic: Reciprocal LSH+GSH learning and the role of W_f[hidden]=0
---

# Reciprocal learning + structural zero — RAW

---

There are two threads here and I think they're the same thread.
Let me pull on both and see where they converge.

Thread 1: the GSH and LSH learning from each other. The GSH
already learns from the LSH — it indexes the LSH's routing
patterns. But the LSH doesn't learn from the GSH. It's a
one-way relationship: LSH teaches, GSH learns. The user is
asking about making it RECIPROCAL. The GSH's disagreement
signal tells the LSH where it's failing. Can the LSH use
that signal to improve?

Thread 2: W_f[hidden] = 0 as informed attention. The structural
zero in the ternary weight matrix means "don't look at this
pixel." Currently those zeros are random (density=0.33 means
~33% zeros). Making them informed — placing zeros deliberately
to expose discriminative structure — turns the zero from noise
into a routing decision.

Where do these threads converge?

The GSH says: "query Q is uncertain — LSH and GSH disagree."
The LSH says: "my projection for table m has zeros on pixels
{p1, p2, ..., pk}." If the discriminative information for
query Q lives on those zeroed-out pixels, the LSH is blind
BY CONSTRUCTION. The fix: build a new projection for Q's
confusion pair where the zeros are moved AWAY from the
discriminative pixels.

So: the GSH identifies WHERE the LSH fails (which queries).
The structural zero identifies WHY the LSH fails (which pixels
are hidden). Combining them: build new projections where the
zeros are placed to expose the pixels that the failing queries
need.

This is attention through routing. The structural zero is the
attention mask. The GSH is the attention controller — it tells
the system where to redirect attention.

But wait. The current system uses the SAME projection matrix
for all queries. Every training image and every test query sees
the same zeros. We can't have per-query zeros without per-query
projections, which would defeat the bucket index (the index
assumes all images use the same projection).

So the attention must be at the TABLE level, not the query level.
Build SPECIALIZED tables — tables whose projections have zeros
placed to expose specific class-pair discriminative structures.
Table A has zeros placed to distinguish Cat from Dog. Table B
has zeros placed to distinguish Ship from Truck. Each table is
a specialist in a particular confusion.

At query time, the GSH identifies which confusion the query is
involved in (from the disagreement pattern), and the resolver
weights the specialist table's score more heavily.

This is a three-instrument arrangement:
- LSH: general-purpose routing (random projections, broad coverage)
- GSH: confidence and confusion identification (routing topology)
- Specialist tables: targeted routing for specific confusions
  (informed zeros, narrow but deep coverage)

The GSH tells the system which specialist to consult. The
specialist's projection has zeros placed to expose exactly
the discriminative axis the general LSH misses. The structural
zero is the mechanism that makes specialization possible — it
says "for THIS confusion, look at THESE pixels and IGNORE the
rest."

OK, this is getting architectural. Let me think about what's
actually measurable and buildable.

The simplest version: for each of the top-K confusion pairs
(Cat/Dog, Ship/Truck, etc.), compute which pixels differ most
between the two classes (class-mean difference). Build a
projection direction where those pixels have ±1 weights (high
difference → +1 if class A is higher, -1 if class B is higher)
and everything else has 0 weight. This is a SINGLE ternary
projection direction that's optimized for one confusion pair.

Build M_specialist such directions (one per confusion pair,
K=45 for 10 classes). Add these as additional tables. At
resolve time, score the union using BOTH the general tables
AND the specialist tables.

But this is importing pixel-space computation (class means,
pixel differences) into the projection layer. The user has
been clear: route the solution. Can we discover the specialist
projections through routing?

Yes. Here's how:

For each confusion pair (i, j): look at the training images
that the LSH+GSH system gets wrong (LSH predicts class i,
true class is j). In the LSH's union for those failing queries,
find training prototypes of both class i and class j. Compute
the per-table Hamming distance between the query and a class-i
prototype versus a class-j prototype. Tables where the class-i
prototype is closer contribute to the wrong answer; tables
where the class-j prototype is closer would contribute to the
right answer.

The tables that get it right for this confusion pair have
projections whose non-zero weights happen to align with
the discriminative axis between classes i and j. The tables
that get it wrong have non-zero weights that miss this axis.

Now: which TRIT POSITIONS in the right-answer tables are
responsible? The trits where the class-j prototype and the
query have the same value (contribute 0 to Hamming distance)
while the class-i prototype and the query differ (contribute
>0). Those trit positions encode the discriminative axis.

The projection rows corresponding to those trit positions are
the directions that distinguish the two classes. Their non-zero
weights identify which pixels matter for this confusion. Their
zero weights identify which pixels don't.

So: the ROUTING MEASUREMENTS (per-table, per-trit distances
between query and correct vs incorrect prototypes) DISCOVER
which trit positions — and therefore which projection
directions — are discriminative for each confusion pair. No
pixel-space computation. Pure routing analysis.

The specialist projection is built by:
1. Identifying which confusion pair the query is in (GSH
   disagreement pattern).
2. Finding the LSH tables that distinguish this pair (per-table
   analysis of correct vs incorrect distances).
3. Building new projections that amplify those tables' weight
   patterns.

Step 3 is the only part that touches the projection matrix,
and it does so by COPYING the weight patterns of routing-
discovered successful tables. It's routing learning from
routing — the system identifies which of its own projections
work for each problem and builds more like them.

This is natural selection in projection space. Random
projections are the initial population. Routing measurements
are the fitness function. Successful projections reproduce
(their patterns inform new specialist tables). Unsuccessful
projections are not reused. The structural zero is the
mechanism of variation — where zeros are placed determines
what each projection sees.

But I'm getting ahead of myself. Let me think about what's
actually DIFFERENT about the structural zero from normal
feature selection.

In classical feature selection, you measure feature importance
and select the most important features. In ternary projection,
the "features" are not individual pixels — they're SIGNED
LINEAR COMBINATIONS of pixels. A projection direction with
weights [+1, -1, 0, +1, 0, ...] is a feature that says "pixel 0
minus pixel 1 plus pixel 3." The zero at pixel 2 means "pixel 2
is irrelevant to this feature."

The structural zero doesn't just select features — it shapes
the GEOMETRY of the measurement. A projection with zeros at
the right positions creates a measurement that's aligned with
the discriminative axis. A projection with zeros at the wrong
positions creates a measurement that's orthogonal to it.

This is deeper than feature selection. It's MEASUREMENT DESIGN.
The structural zero designs the measurement to see what matters
and be blind to what doesn't. In base-2, there's no structural
zero — every weight is ±1, every pixel contributes, and you
can't design blindness. The third state (zero) is what makes
selective attention possible.

NORTH_STAR says base-2 ignores 1/3 of the signal — the
structural zero. But the zero isn't just "missing signal" — it's
the mechanism that ENABLES signal. Without the ability to say
"don't look here," every measurement is forced to see everything,
and on dense inputs like CIFAR-10, "see everything" means "see
mostly noise."

The zero is the lens. It focuses the measurement. And the routing
architecture can learn WHERE to focus by measuring which zeros
produce the best routing outcomes.

I think there's a clean four-part architecture here:

1. **LSH (broad):** random ternary projections with random zeros.
   General-purpose routing. Finds the neighborhood.

2. **GSH (deep):** hashes the LSH routing pattern. Identifies
   confident vs uncertain queries. Identifies which confusion
   the uncertain query is in.

3. **Specialist projections (focused):** ternary projections with
   INFORMED zeros, built from routing measurements on the training
   set. Each specialist is optimized for a specific confusion pair.
   The zeros expose the discriminative axis; the ±1 weights measure
   along it.

4. **Combination (orchestrated):** the GSH decides which specialist
   to consult. The LSH provides the neighborhood. The specialist
   provides the discriminative re-ranking. Agreement across all
   three → high confidence.

This is a routing-native attention mechanism. The zero is the
attention mask. The GSH is the attention controller. The specialist
projections are the attention heads. The whole thing operates on
packed trits through Hamming distance.

One more thought: the specialist projections don't need their own
bucket index. They can RE-SCORE the LSH's union. The LSH found
the candidates (oracle=100%); the specialist provides a better
scoring function for the confused subset. This is the re-rank
pattern we already proved works — but with INFORMED projections
instead of random wider projections.

The re-rank at N_PROJ=64 random added ~1pp. Re-rank with
INFORMED projections (zeros placed to expose the confusion
pair's discriminative axis) could add much more, because each
trit in the re-rank signature is measuring something RELEVANT,
not random.

This is the path to 50%+ on CIFAR-10. Not wider random
projections (saturate at 38%). Not more tables (saturate at
+3pp). Informed projections where the structural zero is the
routing architecture's mechanism for focused measurement.
