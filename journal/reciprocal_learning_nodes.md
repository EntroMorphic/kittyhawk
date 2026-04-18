---
date: 2026-04-17
phase: NODES
topic: Reciprocal LSH+GSH learning and the role of W_f[hidden]=0
---

# Reciprocal learning + structural zero — NODES

---

## Node 1 — The two threads are one thread

Thread 1 (reciprocal learning): GSH tells LSH where it fails,
LSH adapts. Thread 2 (structural zero): the zero weight in a
ternary projection controls what the measurement sees.

Convergence: the GSH identifies WHICH queries fail. The structural
zero determines WHAT each projection sees. Reciprocal learning
means: use the GSH's failure signal to PLACE the zeros so the
projection sees what was previously hidden.

## Node 2 — The zero is the lens, not the absence

In base-2 ({-1, +1}), every pixel contributes to every projection.
The measurement can't be selective. In base-3 ({-1, 0, +1}), the
zero says "this pixel is structurally absent from this measurement."
The zero is the mechanism for ATTENTION — the ability to focus on a
subset of the input.

Without the zero, a random projection of 3072 CIFAR-10 dims sums
~2048 pixels (at density=0.50 binary weights). Every direction
mixes signal with noise. With the zero at density=0.33, each
direction sums ~1024 pixels and ignores ~2048. The zero already
filters 2/3 of the noise. But it filters RANDOMLY.

Making the zero INFORMED — placing it deliberately to filter noise
and expose signal — turns random filtering into measurement design.

## Node 3 — Natural selection in projection space

Random projections are the initial population. Each projection's
zero pattern is a random attention mask. Routing measurements
(per-table accuracy on the training set, per-confusion-pair
discrimination) are the fitness function.

Selection: identify which projections distinguish which class
pairs. Reproduction: build new projections whose zero patterns
resemble successful projections' zero patterns, extended or
refined. The structural zero VARIES between projections, and
routing SELECTS for the zero patterns that produce discriminative
measurements.

This is evolution without gradients. The variation is ternary
(which pixels are ±1 vs 0). The fitness is routing accuracy. The
selection is deterministic (keep the best, discard the rest).

## Node 4 — The routing measurements that identify useful zeros

For a confusion pair (class i, class j), the LSH produces per-table
distances to the nearest class-i prototype and nearest class-j
prototype. The tables where d(class-j) < d(class-i) would have
produced the correct answer (assuming class j is true).

Those "correct" tables have projection weights where the non-zero
positions (±1 weights) happen to align with the pixels that
distinguish class j from class i. Their zero positions hide pixels
that are COMMON to both classes (non-discriminative).

The zero pattern of a correct table IS the attention mask for that
confusion pair. Copying it produces a specialist projection.

This is entirely routing-native: the analysis uses Hamming
distances from the probe pass. No pixel-space computation.

## Node 5 — Specialist projections re-rank the LSH union

The specialist projections don't need their own bucket index.
They RE-SCORE the LSH's existing union. The LSH found the
candidates (oracle=100%); the specialist provides a better
scoring function for the specific confusion the query is in.

This is the re-rank pattern, but with INFORMED projections
instead of wider random projections. Each trit in the
specialist's signature measures something RELEVANT to the
confusion pair, not something random.

Expected gain: the re-rank at N_PROJ=64 random added ~1pp.
Specialist re-rank with informed zeros should add more because
the signal-to-noise ratio per trit is higher.

## Node 6 — The four-instrument architecture

1. **LSH (broad):** random projections, random zeros. General
   routing. Finds the neighborhood.
2. **GSH (deep):** hashes routing patterns. Identifies
   confident vs uncertain, and which confusion the uncertain
   query is in.
3. **Specialist projections (focused):** informed zeros,
   built from routing measurements. Re-ranks the LSH union
   for specific confusion pairs.
4. **Combination:** GSH decides whether to trust LSH alone
   (agreement) or consult the specialist (disagreement).

## Node 7 — Building specialist projections from routing data

For each of the K most common confusion pairs:

1. Identify failing training queries where LSH predicts class i
   but true class is j.
2. For each such query, find which LSH tables produced the
   correct class-j 1-NN vs incorrect class-i 1-NN.
3. The "correct" tables' projection weight patterns encode
   the discriminative axis. Specifically: the non-zero weight
   positions that are +1 or -1 in those tables point at the
   pixels that distinguish class j from class i.
4. Build a specialist projection by: taking the weight pattern
   of the most frequently "correct" table for this confusion
   pair.

Simpler version: for each confusion pair (i, j), generate
N_cand random projections and measure which ones produce the
largest per-table distance gap (d(class-i) - d(class-j) for
class-j queries). Keep the best few. Their zero patterns are
routing-discovered attention masks for that confusion pair.

## Node 8 — The reciprocal learning loop

Iteration 0: random LSH → random GSH → baseline accuracy.
Iteration 1: GSH identifies failures. Build specialists from
  routing measurements. Re-rank with specialists. Improved
  accuracy on the confused subset.
Iteration 2: re-compute GSH on the specialist-augmented system.
  Smaller disagreement set. Build new specialists for the
  REMAINING confusions. Re-rank again.
...
Iteration N: disagreement set stops shrinking. Remaining
  failures are genuinely ambiguous in the projection space.

Each iteration:
- LSH → GSH → failure identification → specialist construction
  → re-rank → improved accuracy → updated GSH → smaller
  failure set → ...

The loop converges when the routing measurements can't find
any more discriminative zero patterns.

## Node 9 — How many specialists are needed?

CIFAR-10 has 10 classes → 45 confusion pairs. The top-8
confusion pairs from the atomics account for the majority
of failures:

```
  9→8 (231), 4→2 (221), 0→8 (219), 6→2 (213),
  2→4 (204), 1→8 (199), 6→4 (197), 7→4 (187)
```

8 specialist projections, one per pair, would cover ~1671
failures out of ~6468 total (25.8%). If each specialist
corrects even half its target failures, that's ~835 queries
→ +8.4pp on CIFAR-10 (37% → 45%).

## Node 10 — The density parameter controls the zero budget

At density=0.33, ~33% of weights are ±1 and ~67% are zero.
The specialist has a "zero budget" of 67% — it can hide
67% of the input. For a 3072-dim CIFAR-10 image, that means
each specialist sees ~1024 pixels and hides ~2048.

A specialist for the Cat/Dog confusion would place its 1024
non-zero weights on the pixels that most distinguish cats
from dogs. The 2048 zeros are on background, common textures,
and irrelevant regions.

The density itself could be a per-specialist parameter.
A specialist at density=0.10 sees only ~307 pixels — very
focused. At density=0.50, it sees ~1536 — broader but less
selective. The right density depends on how concentrated
the discriminative signal is.

## Tensions

**T1:** How to discover specialist projection patterns —
copy successful random tables (Node 4/7a) or generate and
select new candidates (Node 7b)?

**T2:** When to consult the specialist — only on GSH
disagreement, or always as a secondary score?

**T3:** How many iterations of the reciprocal loop before
diminishing returns? Is one iteration (build specialists,
stop) sufficient?

**T4:** Can the specialist projections be discovered
entirely from routing measurements, or do they need
class-conditional pixel statistics (which would break the
routing-native constraint)?
