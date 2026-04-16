---
date: 2026-04-16
phase: REFLECT
topic: Lattice Geometry Resolver — reading the routing pass's own measurements
---

# Lattice Geometry Resolver — REFLECT

Finding the structure beneath the content. Resolving tensions.

---

## Core insight

The RAW and NODES phases circled around a tension: the margin
says "I'm confident" but not "I'm right." Resolving this tension
reveals the core insight.

The margin doesn't need to predict correctness for INDIVIDUAL
tables. It needs to predict correctness ON AVERAGE across the
decisive tables. Here's why:

SUM over all M tables gives equal weight to noise (tied tables)
and signal (decisive tables). If the decisive tables are correct
51% of the time (barely better than chance), margin-weighting
still helps — because it's concentrating the SUM on a subset
with 51% correctness instead of diluting with a larger set at
50% correctness. The bar is not "decisive tables must be right."
The bar is "decisive tables must be more often right than the
average table."

And we already have evidence they are. Atom 3 on CIFAR-10
measured the per-table distance gap (d_winner - d_true):

    All pairs:       +0.020 mean, 13.4% true closer, 11.5% winner closer
    Excluding ties:  the non-tied pairs are where that +0.020 comes from

The +0.020 is positive (true closer on average) but washed out
by the 75% tied pairs. The non-tied pairs — exactly the ones
margin-weighting amplifies — carry the directional signal that
SUM currently dilutes. Margin-weighting doesn't need decisive
tables to always be right. It just needs the non-tied fraction
to have a better true/confuser ratio than the tied fraction,
which is trivially true since tied pairs have no signal at all.

**T1 resolved:** margin weighting works whenever the decisive
subset's accuracy exceeds the full set's accuracy. Since the
full set includes 75% pure noise (tied pairs), any non-random
directional signal in the decisive subset is an improvement.

## T2 resolved: test at N_PROJ=16 first, expect gains at wider

NODES correctly identified that margin-weighting at N_PROJ=16
degenerates to binary table selection (margin is mostly 0 or 1).
But binary selection is STILL worth testing because:

1. It isolates whether the decisive subset has signal at all.
2. If binary selection helps at N_PROJ=16, continuous weighting
   at wider N_PROJ should help MORE (richer margins, less
   degenerate).
3. If binary selection doesn't help at N_PROJ=16, we know the
   decisive subset doesn't have signal — and wider N_PROJ
   won't fix a mechanism that's fundamentally broken.

So N_PROJ=16 is the right first test: it's the cheapest and
it's a strict lower bound on the mechanism's potential. The
dynamic cascade already has wider stages available — adding
margin-weighting to each stage is trivial once the resolver
exists.

## T3 resolved: implement continuous weighting, let N_PROJ=16 naturally degenerate

Table selection (Node 7) is a special case of continuous
weighting where all margins are 0 or 1. Implementing continuous
weighting is no harder than selection — one extra multiply per
(table, candidate) pair. At N_PROJ=16 it naturally degenerates
to selection. At wider N_PROJ it's genuinely continuous. One
implementation covers both.

## T4 resolved: margin-weighted first, k-NN later

Margin-weighting is routing-native: it reads the lattice's own
geometry. k-NN is a classical technique applied to routing
outputs. Both improve the resolver, but margin-weighting tests
the thesis — that the lattice's geometric measurements contain
usable information the current resolver discards. k-NN doesn't
test that thesis; it just uses a better aggregation rule on the
same information.

If margin-weighting works, it validates the lattice self-
measurement principle. If it doesn't, k-NN is the fallback.

**Decision:** implement margin-weighted SUM as
`glyph_resolver_sum_marginweighted`. Test on CIFAR-10 at
N_PROJ=16 (binary gate lower bound), then at N_PROJ=512 via
the dynamic cascade (continuous weighting upper bound).

## What I now understand

1. **The margin doesn't need to predict correctness per-table.**
   It needs the decisive subset's accuracy to exceed the full
   set's accuracy. Since the full set includes 75% pure noise,
   ANY directional signal in the decisive subset is an
   improvement.

2. **The mechanism is table selection at N_PROJ=16 and
   continuous weighting at wider N_PROJ.** Same code, different
   operating regimes. Test the lower bound first.

3. **The user's deeper vision (cross-query geometry accumulation)
   is deferred.** The per-query margin is the first step. If it
   works, the principle — "the lattice can read its own
   geometry" — is validated, and cross-query accumulation is
   the natural extension.

4. **Margin-weighted + dynamic cascade is the production
   architecture.** The cascade provides the right resolution
   per query. The margin-weighted resolver extracts the most
   information at that resolution. They compose naturally.

## What remains uncertain

- Whether the decisive-subset accuracy on CIFAR-10 is
  meaningfully above the full-set accuracy. (Measurable.)
- Whether continuous margin values at N_PROJ=512 carry more
  information than binary selection at N_PROJ=16. (Measurable.)
- Whether the margin-correctness correlation is stable across
  seeds. (Should be, given the 94% seed-invariance of query
  fates, but worth confirming.)
- The right fallback when all tables are tied (margin=0
  everywhere). Unweighted SUM is the natural default.
