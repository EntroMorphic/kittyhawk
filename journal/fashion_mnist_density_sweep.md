# Phase B.2 — density-varied multi-table sweep

Date: 2026-04-15
Tool: `tools/mnist_routed_bucket_multi.c` with `--density_schedule {fixed,mixed}`
Config: N_PROJ=16, M=64, max_radius=2, min_cands=50
Baseline: Fashion-MNIST no-deskew, density=0.33, SUM=85.15% at M=64
Atomics context: `journal/fashion_mnist_atomics.md`

## Hypothesis

The atomics showed the per-table min-Hamming gap between upper-body
classes is −0.036 bits (65% tied) at density 0.33. Mixing multiple
densities across the M tables should diversify the projection
geometries — each density family calibrates τ on its own distribution,
and summing across families should surface an axis no single family
sees.

## Implementation

`glyph_config_t.density_schedule` added as a new CLI knob:

- `fixed` (default): every table uses `--density`. Byte-exact Phase 3
  reproduction preserved.
- `mixed`: table `m` uses `densities[m % 3]`, round-robin. The density
  triple is compiled into the tool (not CLI-exposed yet) so the
  experiment can iterate by recompile without churning the CLI surface.

## Attempt 1 — wide spread {0.20, 0.33, 0.50}

Fashion-MNIST mixed: 84.57% at M=64 (−0.58pp vs fixed baseline).
MNIST mixed: 97.16% at M=64 (−0.15pp vs fixed baseline).

Both datasets regressed. Upper-body cluster on Fashion-MNIST:

```
class  fixed   mixed   Δ
  0   80.40  80.70   +0.30
  2   76.30  76.40   +0.10
  4   77.00  75.40   −1.60
  6   61.50  59.50   −2.00
```

Shirt (class 6), the magnet class, got meaningfully worse. Confusion
pair 6→0 grew 149 → 164.

## Single-density baseline scan

To understand whether the regression was a family composition problem
or a fundamental density-diversification problem, I ran standalone
M=64 baselines across five densities:

```
density   Fashion-MNIST SUM
  0.20    84.97%
  0.25    85.54%   ← best on Fashion-MNIST
  0.33    85.15%   (baseline)
  0.40    84.36%
  0.50    84.05%   ← worst; dominated the wide-mix regression
```

The curve is peaked near 0.25 with ~0.2pp sensitivity inside
(0.20, 0.33) and a ~1pp falloff toward 0.50.

Cross-check on MNIST: density 0.25 gives 97.18% at M=64 vs 0.33's
97.31% (−0.13pp). MNIST prefers 0.33, Fashion-MNIST prefers 0.25.
**Dataset-specific density optima** — MNIST's sparse pen-stroke
foreground rewards denser projections (more pixels per weight);
Fashion-MNIST's dense fabric foreground rewards sparser projections
(already-dense inputs don't need extra aggregation).

## Attempt 2 — narrow spread {0.25, 0.33, 0.40}

Built as a second attempt hypothesizing that the wide spread regressed
because 0.50 was genuinely weaker. Narrow spread picks two strong
families (0.25, 0.33) and one mildly weak one (0.40) to preserve
diversification without catastrophic dilution.

Fashion-MNIST mixed narrow: **85.13%** at M=64.

Compared to the components:

```
  density 0.25 alone : 85.54%
  density 0.33 alone : 85.15%
  density 0.40 alone : 84.36%
  mixed narrow       : 85.13%
```

The mix is essentially the arithmetic mean of its components. **No
synergistic gain from combining families.** Density mixing is
strictly dominated by picking the single best density.

## The deeper finding — density is a weak diversification knob

Density mixing was supposed to carve different lattice faces by
selecting different pixel subsets per family. But the per-class
improvements from density 0.25 alone vs density 0.33 alone:

```
class   0.33    0.25     Δ        class
  0   80.40  82.10   +1.70    T-shirt
  2   76.30  78.40   +2.10    Pullover
  4   77.00  78.50   +1.50    Coat
  6   61.50  61.30   −0.20    Shirt
  (others mostly flat / minor noise)
```

Density 0.25 moves the needle exactly inside the upper-body cluster
Atoms 1-3 flagged as signal-starved — +1.5 to +2.1pp on T-shirt,
Pullover, and Coat. But Shirt, the magnet class, doesn't budge.
And density mixing produces no additive gain beyond picking the
single best density.

The reason density variation is a weak diversifier: every density
operates on **the same input vector**. At d=0.20, ~157 of 784 pixels
are sampled; at d=0.33, ~259. Both draws are uniform-random over the
same pixel population, so the projections overlap heavily in
*which pixels matter*. The lattice face changes by a small re-
weighting, not by selecting a different feature set.

**What would actually diversify:** projections constrained to
disjoint spatial regions. A "top-half" table sees only pixels in
rows 0–13; a "bottom-half" table sees only rows 14–27; a
"left-half" table sees columns 0–13, etc. Those are genuinely
different feature sets (not re-weightings of the same set), and
each carves a different lattice face by construction. Still
purely routing-native — random ternary projections with a spatial
selection mask. This is the Phase B.2-next hypothesis.

## Phase B.2 — verdict

Density mixing: **falsified**. Infrastructure (density_schedule CLI)
stays as a carrier for future block-structured schedules — the
round-robin builder-density dispatch is exactly what spatial mask
variation needs.

Concrete win discovered: **per-dataset density tuning**. Fashion-
MNIST should default to density 0.25 for a free +0.39pp. Paper-
worthy finding on its own: density is a dataset-distribution-
dependent hyperparameter, not a universal 0.33 default.

Shirt (class 6) remains the magnet class. It did not improve
under any density and remains the dominant failure mode after
this phase.

## Next phase

**B.2-next: block-structured spatial projections.** Random ternary
projections constrained to a spatial region mask. Tables rotate
through a small set of masks (quadrants, halves, center block)
so each table encodes a different spatial feature set. Still
uses the same signature/bucket/multi-probe infrastructure —
only `glyph_sig_builder_init` needs a mask parameter.

Expected effect: tables that sample the lower quadrant of images
should be able to distinguish Trouser/Shirt from Dress/Coat by
hemline geometry alone, an axis the current uniform-density
projection cannot isolate.
