# Fashion-MNIST atomics — where the upper-body cluster fails

Date: 2026-04-15
Tool: `tools/fashion_atomics.c`
Config: N_PROJ=16, density=0.33, M=64, max_radius=2, min_cands=50, no_deskew

## Phase B.1 — radius-aware SUM resolver (falsified)

Hypothesis: filter-stage multi-probe radius carries a resolver-usable
signal. Candidates found only via deeper multi-probe shells should be
penalized relative to r=0 hits.

Implementation: `glyph_resolver_sum_radiusaware` scores each candidate
as `sum_dist + λ · min_radius[c]` where `min_radius[c]` is the
smallest probe radius at which any table placed `c` in the union.
The tool's `probe_cb` tracks `current_radius` and lazy-zeros
alongside `votes`.

Result (M=64):

| dataset | λ=0 (scalar) | λ=2 | λ=8 | λ=16 | λ=32 |
|---|---|---|---|---|---|
| MNIST         | 97.31 | —    | 97.27 | —    | —    |
| Fashion-MNIST | 85.15 | 85.15 | 85.02 | 84.82 | 84.49 |

Monotone degradation on Fashion-MNIST as λ grows. λ=0 reproduces
scalar. Any positive penalty is neutral-to-harmful. Hypothesis dead.

Reason in hindsight: multi-probe radius is a coarsening of information
already present in `sum_dist`. A candidate found at r=1 on one table
but r=0 on three others has `sum_dist` already reflecting the r=1
table's larger Hamming distance. There is no residual signal.

Code stays in the tree as a falsified artifact — the wiring is
necessary infrastructure for any future filter-geometry-aware
resolver and was exercised end-to-end.

## Atomics instrumentation — the three measurements

Oracle is 100% at M≥16, so the correct neighbor is *always* in the
union. The failure is downstream of filtering. `fashion_atomics`
runs a single M=64 pass, identifies misclassified queries, and for
each one emits three measurements.

### Atom 1 — rank & gap of best true-class prototype

For each of 1485 failures (all with y_true present in the union):

```
rank 0              35   2.4%
rank 1             546  36.8%
rank 2-3           367  24.7%
rank 4-7           231  15.6%
rank 8-15          142   9.6%
rank 16-63         105   7.1%
rank 64-255         35   2.4%
rank >=256          24   1.6%

gap 0               35   2.4%
gap 1               63   4.2%
gap 2-3            134   9.0%
gap 4-7            248  16.7%
gap 8-15           386  26.0%
gap 16-31          391  26.3%
gap >=32           228  15.4%
mean gap: 17.64 Hamming units
```

Only 6.6% of failures are within 1 Hamming unit of the winner. 41% are
gap ≥ 16. Most failures are not tiebreak losses — they are cases where
the wrong class is meaningfully closer in summed lattice distance.

### Atom 2 — per-table 1-NN vote on failing queries

Mean per-query votes across M=64 tables:

```
true class    : 19.52  (30.5%)
winner (wrong): 22.47  (35.1%)
other         : 22.01  (34.4%)
```

The plurality of individual tables *already votes for the wrong
class* on failing queries. Fusion is not breaking good signals —
it is faithfully summing bad ones. This is a projection
representation failure, not a fusion failure.

### Atom 3 — per-table sig-distance gap (d_winner − d_true)

95040 (failing query × table) pairs where both labels appear:

```
mean per-table gap : -0.036 Hamming bits
  true closer  (>0):  16.0%
  tied         (=0):  65.0%
  winner closer(<0): 19.1%
```

65% of pairs are tied at the per-table min-Hamming level. The mean
per-table gap is effectively zero (−0.036 bits out of a max 32 bits
per table). The 17.64-unit sum_dist gap accumulates over M=64 tables
from this vanishingly small directional bias — the lattice is
effectively agnostic between upper-body classes in per-table
signature space.

### Confusion-pair breakdown

```
 true->pred  count  mean_sum_gap  vote_true  vote_winner  mean_tbl_gap
    6->0      149        15.59    15.70      30.67       -0.057
    0->6      144        14.45    27.57      16.38       +0.018
    6->2      121        17.36    15.26      22.98       -0.044
    4->2      108        15.84    22.31      23.83       -0.050
    2->4      107        12.21    21.52      23.59       +0.012
    2->6       96        14.17    21.23      17.18       -0.020
    4->6       84        12.01    22.79      17.89       -0.039
    6->4       77        15.43    14.10      24.30       -0.025
```

Class 6 (Shirt) is a magnet: on 6→X failures per-table votes lean
strongly to the winner (24–31 / 64). But on 0→6, 4→6, 2→6 the
per-table 1-NN actually votes for the true class, yet SUM picks
Shirt anyway — meaning some Shirt prototype sits uniformly close
to many queries across tables, dragging the summed distance down
even though it's rarely the per-table winner.

## Magnet audit — no pathological prototypes

1361 distinct training prototypes won at least one of the 1485
failures. Top-20 share: 4.1%. The worst offender (proto 30872,
Shirt) won only 9 failures (0.61%). The top-20 is dominated by
upper-body labels (6/2/0/4) but spread across 20 different
prototypes with no concentration.

Verdict: **no magnet prototypes**. The upper-body gap is a
structural cluster-center problem, not a pathological-prototype
problem. Pruning individual training examples would buy nothing.

## Implication for B.2

The three atomics converge on a single diagnosis: the
**density-0.33 ternary projection cannot encode a discriminative
axis between T-shirt / Pullover / Coat / Shirt**. Per-table
signatures place them at the same Hamming distance 65% of the
time; the residual 0.036-bit directional bias is effectively noise.

Any fix has to live *at or before* the projection stage. No
resolver reweighting, no filter-geometry trick, no prototype
pruning, and no magnet mitigation can recover a discriminative
axis that was never encoded into the signatures.

This hard-validates **Phase B.2 — density-varied multi-table
sweep**. Different projection densities sample different input-
pixel subsets and carve different lattice faces. Mixing density
families is the next experiment, targeted directly at collapsing
the 65% tied-gap rate on the upper-body cluster.

Calibration must be per-density: each density family's τ has to
be computed on its own projection distribution. Half-measures
without per-density calibration would give a misleading negative
signal.
