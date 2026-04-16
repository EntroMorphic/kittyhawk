---
date: 2026-04-16
phase: RAW
topic: Lattice Geometry Resolver — reading the routing pass's own measurements
---

# Lattice Geometry Resolver — RAW

Unfiltered thinking. Honest. Doubts included.

---

There's something exciting here and something I'm worried about.
Let me separate them.

The exciting part: the user's reframe is correct. I was about to
import pixel-space geometry (PCA, centroids) to fix a routing
problem. The user pointed out that the routing pass ALREADY
produces geometry — per-table distances, margins, agreement
patterns — and we're throwing most of it away. The SUM resolver
reads M×N_candidates measurements and collapses them to one
number per candidate. That's a massive information loss. On
CIFAR-10 where 75% of measurements are ties, the loss is
catastrophic — we're averaging mostly-zeros with a thin signal
buried inside.

Margin-weighting is the simplest way to stop the loss. Instead
of treating all tables equally, weight each table by how
decisively it separates candidates. The lattice tells you which
projections are informative for THIS query — not on average, not
across all queries, but right now for this specific image. That's
the kind of per-query adaptation that no fixed projection change
(PCA, centroids, Hadamard) can provide.

Now the worries.

First worry: what exactly IS the margin measuring? I defined it
as d_2nd - d_1st at each table — the gap between the nearest and
second-nearest candidate in the union under that table's
projection. But "nearest candidate" is not the same as "nearest
candidate of the CORRECT class." The margin measures how
decisively the table prefers its 1-NN over the field, not how
correctly it classifies.

A table with margin=10 might be very decisive about the WRONG
candidate. Weighting it heavily amplifies a wrong answer. The
margin says "I'm confident" not "I'm right."

This is the same distinction between confidence and correctness
that plagues calibration in neural networks. High margin doesn't
mean high accuracy. It means high commitment.

Second worry: interaction with the union size. On CIFAR-10 at
M=16, the union has ~4800 candidates. At each table, the 1-NN
and 2-NN among 4800 candidates in a 16-trit Hamming space are
very likely to be at the SAME distance (that's the 75% tie rate).
So margin = 0 for most tables. The non-zero margins come from
tables where one candidate happens to be slightly closer — often
by just 1 Hamming bit.

A margin of 1 out of a possible range of 0..32 (at N_PROJ=16)
is a very thin signal. Weighting by 1 vs 0 is a binary gate
(include vs exclude), not a nuanced weighting. The effective
resolver becomes "SUM restricted to the ~16 decisive tables"
rather than "SUM with continuous weights." That might still be
better than uniform SUM, but it's a much simpler mechanism than
the design document implies.

Third worry: the margin is computed from the union, which is
built by the N_PROJ=16 filter. The union contains ~4800
candidates out of 50,000 training prototypes. The per-table
margin depends on which candidates are IN the union. If the
union is biased (e.g., oversampling a particular class because
its signatures cluster near the query), the margins reflect
that bias. We're measuring the geometry of a FILTERED
neighborhood, not the full lattice.

This isn't necessarily bad — the filter is supposed to select
relevant candidates — but it means the margin's meaning depends
on the filter quality. On CIFAR-10 where the filter's oracle is
100%, the union always contains the correct answer, so the bias
is benign. But the geometry is still conditioned on what the
filter selected.

Fourth worry: the relationship to vote-weighted SUM (Phase A,
falsified). Vote-weighted scored by `sum_dist / (1 + votes)`.
This was ALSO trying to read filter-stage information (how many
tables voted for each candidate). It was falsified because
votes at N_PROJ=16 are near-chance and dividing by noise adds
noise.

Is margin-weighting different enough? The design document claims
yes — margins read RESOLVER-stage information (how confident is
each table's measurement) rather than FILTER-stage information
(how many tables found the candidate). But the margin is
computed from the same Hamming distances the resolver already
sums. If margin = 0 because the top two candidates are tied,
that's because the Hamming distance can't distinguish them —
and the unweighted sum ALSO can't distinguish them. The margin
just makes explicit what the sum already knows implicitly:
"this table contributed nothing to the ranking."

So the margin doesn't ADD information — it just REWEIGHTS the
existing information. The question is whether reweighting helps
when the information itself is thin.

Fifth worry (and this is the one I'm most uncertain about): the
user said "the routing manifests temporary geometries that can be
used for measurement across distances." I interpreted this as
per-table margins. But I think the user might mean something
deeper — that the ACT of routing (traversing the lattice,
finding neighborhoods, building unions) creates a geometric
structure that persists across queries and can be reused. Like
the routing pass is building a map of the lattice, not just
scoring one query.

If that's the intent, the margin-weighted resolver is a pale
shadow of what the user is imagining. The real idea might be:
use the geometric structure manifested by routing MANY queries
to learn which regions of the lattice are discriminative and
which are degenerate. Build a per-region confidence map. Use
that map to weight future queries.

That's a much bigger idea. It's a routing-native equivalent of
"learning" — but instead of gradient-based parameter updates,
it's lattice-traversal-based confidence mapping. The lattice
self-organizes by measuring which of its own geometries are
useful.

I don't know how to implement that in a single pass. But the
margin-weighted resolver might be the first step — it reads
the geometry for a single query. Extending it to read geometry
ACROSS queries (accumulated confidence map) would be the next
step.

Let me set aside the big vision and focus on what's testable
now. The margin-weighted resolver is testable. It might work.
If it does, the vision of lattice self-measurement is validated
at the per-query level. If it doesn't, the diagnosis tells us
whether the problem is the mechanism (margins aren't predictive
of correctness) or the implementation (V1 is too crude, need
V2/V3).

One more thought: the margin at wider N_PROJ stages should be
more informative. At N_PROJ=16, 75% of tables are tied and
margins are mostly 0 or 1. At N_PROJ=512, the Hamming space
is 1024-dimensional and ties should be much rarer. The margin-
weighted resolver combined with the dynamic cascade — where
each escalation stage uses wider projections with better
margins — is the natural deployment. Test margin-weighting at
N_PROJ=16 first (baseline), but expect the real gains to come
at N_PROJ=128-512 where the margins are richer.
