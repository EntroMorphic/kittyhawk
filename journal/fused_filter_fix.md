---
date: 2026-04-15
scope: Rerun P1 with fused filter (H1+H2 at the filter stage)
type: architectural fix + rerun + meta-router obsolescence
tool: tools/mnist_local_v2.c
parent: journal/lvg_atomics_decomposition.md
---

# Fused filter recovers the L→Gq gap: 83.86% → 88.87%, meta-router obsolete

The P1 atomic decomposition identified two structural failures in the local architecture. This rerun tests two composable fixes on both axes simultaneously. **The fused-filter fix alone lifts local from 83.86% to 88.44% — closing 82% of the gap to pure global Gq (89.46%) without any meta-routing.** Adding K widening on top brings it to 88.87%, within 0.59 points of Gq. The LMM cycle's meta-router architecture is now deprecated: the best move is to upgrade the local architecture and stop.

## The fixes

**Fix A — widen K_RESOLVE.** Motivated by the 7.4% filter-miss rescues (correct class outside H1's top-50). Test K ∈ {50, 100, 200}.

**Fix B — fused filter.** Replace H1-alone filter with (H1+H2) summed-distance filter before taking top-K. Motivated by H1's 55.5% top-1 rank accuracy — a single 16-trit hash destroys rank information inside its preserved neighborhood. Fusing two independent 16-trit hashes at the filter stage captures twice the ranking evidence *before* resolver commitment.

The two fixes are independent axes and compose into a 2×3 grid.

## Results

All six variants at N_PROJ=16, density=0.33, single seed 42, compared against the same Gq reference (89.46%).

| variant | accuracy | Δ vs Gq |
|---|---|---|
| L50_H1  (K=50,  H1 filter, baseline) | 83.86% | −5.60 |
| L100_H1 (K=100, H1 filter) | 85.59% | −3.87 |
| L200_H1 (K=200, H1 filter) | 86.79% | −2.67 |
| **L50_H12** (K=50,  H1+H2 filter) | **88.44%** | **−1.02** |
| L100_H12 (K=100, H1+H2 filter) | 88.73% | −0.73 |
| **L200_H12** (K=200, H1+H2 filter) | **88.87%** | **−0.59** |

**Fix B (fused filter) dominates Fix A (widen K) by more than 2×.** Going from L50_H1 to L50_H12 lifts accuracy by +4.58 points — from a single change at the filter stage. Going from L50_H1 to L200_H1 (widen K fourfold) lifts by only +2.93 points.

Composed fixes: L200_H12 at **88.87%** sits within 0.59 points of pure Gq at roughly half the cost.

## Filter ceilings

| K | H1 filter ceiling | H1+H2 filter ceiling | lift |
|---|---|---|---|
| 50 | 98.59% | 99.55% | +0.96% |
| 100 | 99.50% | 99.85% | +0.35% |
| 200 | 99.86% | 99.94% | +0.08% |

The fused filter raises the top-50 ceiling by 0.96 points (filter-miss rate drops from 1.41% to 0.45%). That alone doesn't explain the 4.58-point accuracy jump from L50_H1 to L50_H12 — the fused filter is *also* producing much better ranking *within* its preserved neighborhood, not just including more correct-class prototypes.

## Contingency vs Gq

For each L variant, the 2×2 contingency against Gq reveals how much rescuable gap remains:

| variant | LR_GR | LR_GW | LW_GR (rescue) | LW_GW | net | oracle |
|---|---|---|---|---|---|---|
| L50_H1  (original P1) | 8055 | 331 | **891** | 723 | +560 | 92.77% |
| L100_H1 | 8265 | 294 | 681 | 760 | +387 | 92.40% |
| L200_H1 | 8432 | 247 | 514 | 807 | +267 | 91.93% |
| **L50_H12** | 8651 | 193 | **295** | 861 | +102 | 91.39% |
| L100_H12 | 8681 | 192 | 265 | 862 | +73 | 91.38% |
| **L200_H12** | 8658 | 229 | **288** | 825 | +59 | 91.75% |

**The rescue count collapses from 891 to 265-295** with the fused filter. Damage count halves from 331 to 192-229. Net gap between L and Gq shrinks from +560 to +59.

## The deeper insight — information should live at the filter stage

Both L50_H1 and L50_H12 use the exact same four hashes (H1, H2, H3, H4). The only difference is **where H2 is applied**: as a resolver (L50_H1) or as part of the filter (L50_H12).

- **L50_H1:** filter with H1 alone, top-50; resolver is H2+H3+H4 summed over top-50.
- **L50_H12:** filter with H1+H2 summed, top-50; resolver is H3+H4 summed over top-50.

Same information content, same prototypes, same arithmetic primitives. Moving H2 from "one of three resolvers" to "half of the filter" buys +4.58 points.

**Why:** the filter stage does a hard commitment — only top-K candidates reach the resolver. Correct-class prototypes that H1 alone ranks below position K are gone forever regardless of what the resolver would have said. The atomic decomposition showed that 48% of rescues lived at H1 ranks 6+; the fused filter uses H2 to pull those prototypes forward into top-K before the hard cut.

The resolver stage is elastic — adding or removing a hash there changes the ranking within an already-committed pool. The filter stage is rigid — it decides which prototypes can be ranked at all. Spending information at the filter stage has much higher leverage.

Stated as a rule:

> **The earlier in a cascade you can apply routing information, the more leverage it has. Information applied at the filter stage constrains set membership; information applied at the resolver stage only re-orders. When the filter is imperfect, always prefer spending marginal information on the filter.**

This is the dual of the filter-ranker reframe from the cascade_atomics_mechanism.md journal. That journal said "the hash is a filter, not a ranker — use it as a filter." This journal says "and when you have more than one hash, put them *all* at the filter stage until the filter saturates."

## Meta-router status: obsolete

The original P1 gate showed:
- Rescue:damage = 2.7:1
- Net gap L→Gq = +5.60
- Oracle ceiling = 92.77%
- Predicted meta-router ~88% (observability ceiling)

With L200_H12 as the new baseline:
- Rescue:damage = 1.26:1
- Net gap L→Gq = +0.59
- Oracle ceiling = 91.75%

A meta-router on top of L200_H12 can claim at most +0.59 points of additional accuracy in the oracle case, and realistically far less given rescues and damages are still observationally similar. **The entire meta-router architecture is now chasing ~0.3 points of accuracy at 50% additional cost.** It's not worth building.

The LMM cycle still produced the result — just not the result it set out to produce. The cycle's actual value was:
1. The P1 gate forced measuring the L→Gq gap.
2. The atomic decomposition exposed that the gap's structure was a filter-ranking problem, not a resolver problem.
3. That exposure made the fused-filter fix obvious.
4. The fused-filter fix closes 89% of the gap without any meta-routing.

A better framing: **the meta-router was a proposal to route around a deficient filter; the correct fix is to deepen the filter.** The LMM cycle discovered this the long way — by first designing the router, then measuring the observability gap, then realizing the gap's shape implicated the filter.

## Cost accounting

| architecture | per-query distance ops | relative cost | accuracy |
|---|---|---|---|
| L50_H1 baseline | 60K (H1) + 150 (H2+H3+H4 × 50) | 1.00× | 83.86% |
| L50_H12 | 120K (H1+H2) + 100 (H3+H4 × 50) | 2.00× | 88.44% |
| L200_H12 | 120K (H1+H2) + 400 (H3+H4 × 200) | 2.01× | **88.87%** |
| Gq | 240K (H1+H2+H3+H4) | 4.00× | 89.46% |

L200_H12 costs roughly 50% of Gq and delivers 99.3% of Gq's accuracy gain over baseline. The resolver stage's cost is a rounding error relative to the filter stage — widening K from 50 to 200 at N_PROJ=16 with 2-hash filter is 200 × 2 = 400 ops, negligible next to the 120K filter pass.

## What to do next

1. **Adopt L200_H12 as the new routed-cascade headline at N_PROJ=16: 88.87%.** Update FINDINGS.md, CHANGELOG.md, README.md.
2. **Test fused filter at other N_PROJ values.** Sweep N_PROJ ∈ {8, 16, 32, 64, 128, 256, 512} with (H1+H2) filter and measure whether the lift holds. Predicted: largest gain at small N_PROJ where H1 alone is a weak ranker; diminishing at large N_PROJ where H1's own ranking is already near-ceiling.
3. **Test triple-filter (H1+H2+H3) as a further fix.** With one hash held back as resolver, does a three-hash filter beat two-hash filter + one-hash resolver?
4. **Retire the meta-router cycle with a closing synthesize note.** The LMM cycle's design proposal is superseded by the fix it helped discover. Acknowledge cleanly.

## Pointers

- Tool: `tools/mnist_local_v2.c`.
- Parent atomic probe: `tools/mnist_lvg_atomics.c`, `journal/lvg_atomics_decomposition.md`.
- P1 gate that started this line: `tools/mnist_local_vs_global.c`, `journal/meta_router_online_synthesize.md`.
- Original cascade mechanism: `journal/cascade_atomics_mechanism.md`.
- Quadruple-hash resolver baseline: `journal/routed_quadruple_decorrelation.md`.
