---
date: 2026-04-16
phase: NODES
topic: Lattice Geometry Resolver — reading the routing pass's own measurements
---

# Lattice Geometry Resolver — NODES

Discrete ideas extracted from RAW. Tensions and dependencies mapped.

---

## Node 1 — Margin measures confidence, not correctness

The per-table margin (d_2nd - d_1st) tells you how decisively
that table prefers its winner. It does NOT tell you whether the
winner is the correct class. A table with margin=10 on the wrong
candidate amplifies the error.

This is a calibration problem, not a mechanism problem. The
question is: does high margin CORRELATE with correctness? If
decisive tables are more often right than wrong, margin-weighting
helps. If they're decisive but randomly right/wrong, it doesn't.

**Measurable.** For each table on each failing query, compute
margin AND whether the table's 1-NN is correct. If margin > 0
implies correctness > 50%, weighting works.

## Node 2 — Margin at N_PROJ=16 is mostly binary (0 vs 1)

At N_PROJ=16, Hamming range is 0..32 per table. With ~4800
candidates, most tables have multiple candidates at the minimum
distance (75% tie rate → margin = 0). The 25% of tables with
margin > 0 likely have margin = 1 (one Hamming bit difference).

So margin-weighting at N_PROJ=16 degenerates to a BINARY GATE:
include the table (margin > 0) or exclude it (margin = 0). This
is simpler than "continuous weighting" but might still be
effective — it's equivalent to "SUM restricted to decisive
tables only."

**Tension with V1 design:** the design proposes continuous
weighting (multiply by margin value). At N_PROJ=16, continuous
reduces to binary. At N_PROJ=512, continuous is genuinely
continuous (margins span a wider range). The mechanism works
differently at different N_PROJ.

## Node 3 — The margin is computed from the same distances SUM already reads

SUM = Σ_m dist(m). Margin-weighted SUM = Σ_m margin(m) × dist(m).
Both read the SAME Hamming distances. The margin doesn't add new
information — it reweights existing information.

This is both a limitation and an advantage:
- Limitation: no new signal enters the resolver. If the Hamming
  distances are genuinely uninformative, no reweighting helps.
- Advantage: no new computation required. The margin is a
  summary statistic of the distances the resolver already sees.

**The question is whether reweighting is sufficient** or whether
new information (e.g., candidate-to-candidate distances, cross-
table correlation) is needed.

## Node 4 — Relationship to the Phase A falsification

Vote-weighted SUM (Phase A) tried to fold votes into the
resolver: score = sum_dist / (1 + votes). It was falsified.

Margin-weighted SUM weights by per-table confidence:
score = Σ_m margin(m) × dist(m).

Key differences:
- Vote-weighted uses a CANDIDATE-LEVEL signal (how many tables
  found this candidate) as a DENOMINATOR. This discounts
  candidates that many tables agree on, which is backwards.
- Margin-weighted uses a TABLE-LEVEL signal (how decisive this
  table is) as a MULTIPLIER. This amplifies tables that see
  clear separation.

The signals are orthogonal: votes measure filter-stage consensus
on the candidate; margins measure resolver-stage confidence on
the table's measurement. Phase A failed because filter-stage
consensus at N_PROJ=16 is near-chance. Resolver-stage confidence
might have a cleaner signal.

But Node 1's concern applies: confident-but-wrong tables would
be amplified.

## Node 5 — The user's deeper vision: lattice self-measurement

RAW captured a distinction between two readings of the user's
insight:

(a) Per-query margin weighting (what the design doc proposes).
    The lattice measures its own geometry for this query and
    weights accordingly.

(b) Cross-query geometry accumulation. The routing pass over
    MANY queries maps which regions of the lattice are
    discriminative and which are degenerate. This accumulated
    map could weight future queries.

(a) is implementable now. (b) is a research direction. Start
with (a) and see if the per-query geometry reading has signal.

## Node 6 — The right N_PROJ for margin-weighting is NOT 16

At N_PROJ=16: 75% ties → margins are mostly 0 → binary gate.
At N_PROJ=512: far fewer ties → margins span a wider range →
continuous weighting has meaning.

The margin-weighted resolver should be tested at MULTIPLE
N_PROJ to find where it adds value. The dynamic cascade already
computes distances at every stage. Adding margin-weighting per
stage is cheap.

**Prediction:** margin-weighting helps most at intermediate
N_PROJ (64-256) where ties are fewer but the projection is
still narrow enough that equal-weight SUM wastes signal.

## Node 7 — An even simpler variant: table selection

If margin-weighting at N_PROJ=16 degenerates to binary (include
tables with margin > 0, exclude tied tables), then the simplest
implementation is not weighting at all — it's table SELECTION.

    score(c) = Σ_{m: margin(m) > 0}  dist(q_m, c_m)

This is "SUM over decisive tables only." No multiplication.
No per-candidate weights. Just skip the tied tables.

If ~16 of 64 tables are decisive (25%), this is equivalent to
SUM at M=16 but with the BEST 16 tables (self-selected by the
lattice) instead of a fixed M=16.

**Tension:** table selection is simpler but throws away the
margin VALUE. At wider N_PROJ where margins are continuous,
selection loses information that weighting preserves.

## Node 8 — Implementation must handle the margin=0 case

If ALL tables are tied (margin=0 everywhere), the weighted sum
is zero for every candidate. Need a fallback: if total margin
is 0, fall back to unweighted SUM. This is the degenerate case
where the lattice admits "I can't see anything" — and it should
happen rarely at M=64 unless the query is genuinely ambiguous
in every projection.

## Node 9 — The k-NN resolver question is separate but related

The user asked about 52% on CIFAR-10. I identified two
bottlenecks: resolver (1-NN vs k-NN) and projection (random
vs structured). The lattice geometry resolver addresses the
resolver bottleneck from a different angle: instead of changing
from 1-NN to k-NN, change from equal-weight to confidence-
weight. Both improve the resolver; they're not mutually
exclusive.

A k-NN margin-weighted resolver would:
- Find the top-k candidates (not just argmin)
- Weight each table by margin
- Majority-vote (or rank-weight vote) the top-k labels

This combines the two resolver improvements. But testing them
separately isolates which mechanism matters.

## Tensions to resolve in REFLECT

**T1:** Margin measures confidence, not correctness. Does high
margin correlate with the 1-NN being correct? (Node 1 vs Node 4)

**T2:** At N_PROJ=16, margin-weighting degenerates to binary
table selection. Is that enough, or do we need to test at wider
N_PROJ to see the real mechanism? (Node 2 vs Node 6)

**T3:** Table selection (Node 7) vs continuous weighting (V1).
Which to implement first?

**T4:** Margin-weighted vs k-NN. Both improve the resolver.
Test margin-weighted first (routing-native, reads lattice
geometry) or k-NN first (proven on MNIST, established technique)?
(Node 9)
