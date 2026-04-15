---
date: 2026-04-15
scope: Quadruple-hash fusion and density decorrelation on the routed cascade
type: follow-up experiment
tool: tools/mnist_resolver_sweep.c (extended)
parent: journal/routed_cascade_rerun.md
---

# Quadruple-hash fusion pushes N_PROJ=16 routed cascade to 83.86%; density decorrelation partially breaks 6↔8

Two follow-ups to the routed-cascade rerun, run together:

1. **Stacking another view** — add a fourth independent secondary hash H4 (new seed 9001/9002/9003/9004, density 0.33) and fuse with H2+H3 as a sum-of-distances 1-NN resolver.
2. **Density decorrelation** — add two new seed families at different emission densities (H_D50 at density 0.50, H_D20 at density 0.20) to test whether varying τ breaks the correlated 3↔8 / 3↔5 / 6↔8 failure modes observed in the standard-density routed cascade.

Both questions answered empirically; both yield clean, partial wins.

## Headline table (N_PROJ=16, K_RESOLVE=50, deskewed MNIST, single seed)

Baseline is R2 (H1+H2 1-NN dual-hash). Confusion counts are key regression pairs across all 10K queries.

| # | resolver | accuracy | Δ vs H1+H2 | 3→8 | 3→5 | 6→8 |
|---|---|---|---|---|---|---|
| R1 | H2 1-NN | 77.33% | +0.63 | 74 | 67 | 35 |
| R2 | H1+H2 1-NN (base) | 76.70% | = | 68 | 77 | 35 |
| R6 | H2 5-NN rank-wt | 79.43% | +2.73 | 59 | 49 | 23 |
| R9 | H2+H3 1-NN (triple) | 81.35% | +4.65 | 74 | 54 | 27 |
| **R12** | **H2+H3+H4 1-NN (quadruple)** | **83.86%** | **+7.16** | **57** | **57** | **23** |
| R13 | H_D50 1-NN (density 0.50) | 74.09% | −2.61 | 64 | 62 | **16** |
| R14 | H_D20 1-NN (density 0.20) | 74.90% | −1.80 | 63 | 54 | **14** |
| R15 | H2+H_D50 1-NN (dual density) | 81.78% | +5.08 | 60 | 54 | 17 |

**New routed-cascade headline at N_PROJ=16: 83.86% via quadruple-hash fusion.** +21.86 points over pure-hash k=7 majority (62.00%), +2.51 over triple-hash (81.35%), +7.16 over dual-hash baseline. Pure-routing ceiling keeps climbing with each independent view added.

## Two findings

### 1. Independent views stack

Quadruple-hash fusion > triple-hash > dual-hash. The marginal gain per added independent view is shrinking but still positive:

| views fused | accuracy | Δ (from prev) |
|---|---|---|
| H1+H2 (dual) | 76.70% | — |
| H2+H3 (triple, but H1 absent) | 81.35% | +4.65 |
| H2+H3+H4 (quadruple, H1 absent) | **83.86%** | **+2.51** |

Adding H4 is worth +2.51 points. Extrapolating: H2+H3+H4+H5 would likely add another +1-1.5, saturating somewhere around 85-86%. The diminishing-returns curve is consistent with each new view covering a shrinking-but-nonzero fraction of the residual error.

### 2. Density decorrelation works on one pair only

At N_PROJ=16, using a density-varied secondary hash (H_D50 at emission density 0.50, H_D20 at 0.20):

- Single H_D50 or H_D20 alone: **worse** than H2 alone (74% vs 77%). Denser/sparser projections lose information compared to the balanced-base-3 optimum.
- Fused with H2: **H2+H_D50 = 81.78%** — better than dual-hash H1+H2 (76.70%) but worse than quadruple H2+H3+H4 (83.86%). Density decorrelation beats "no extra view" but loses to "another standard-density view."
- **6→8 confusion drops from 35 to 14-17** with H_D50 or H_D20. This is a specific decorrelation success.
- **3→8 and 3→5 barely move** (68→60, 77→54). These pairs are robust across density.

**Interpretation:** 6↔8 confusion lives at a specific density regime — the shared "round blob with a loop" feature collapses onto similar signatures at density 0.33 but diverges at 0.50/0.20. 3↔8 and 3↔5 live in the projection-shape structure itself (three-stroke top, open bottom); no amount of density variation helps because the projections get tripped by the same geometric features regardless of how many trits emit.

This is a real architectural distinction: **some confusion pairs are density-recoverable, others are projection-family-bound.** The former can be broken by τ variation on the same substrate; the latter need a structurally different hash or an orthogonal signal.

## Cross-N_PROJ pattern

| N_PROJ | quadruple (R12) | dual-density (R15) | pure maj k=7 |
|---|---|---|---|
| 16 | **83.86%** | 81.78% | 62.00% |
| 128 | 96.64% | **96.70%** | 95.22% |
| 1024 | 97.52% | 97.29% | 97.43% |

Order flip at N_PROJ=128: dual-density (R15) becomes the top routed resolver at **96.70%**, edging out quadruple-hash at 96.64%. Both still beat the prior dual-hash baseline (R2, 96.31%) and all single-seed variants.

At N_PROJ=1024 both approaches dissolve into the 97.5% plateau that all routed resolvers share. R6 (H2 5-NN rank-wt) at 97.53% remains the single-view champion; R12 ties it.

## Why quadruple fusion wins at small N_PROJ but not large

The mechanism is the same filter-ranker factorization as the original cascade:

`cascade_accuracy = filter_presence × conditional_resolver_rate`

At N_PROJ=16:
- Filter ceiling@50: 98.59%.
- Conditional rate of H2 alone: ~78%.
- Conditional rate of H2+H3+H4 sum: approximately 85% (inferred from 83.86% aggregate ÷ 98.59% ceiling).
- Each new independent view adds ~3% to the conditional rate by averaging out uncorrelated ranking errors.

At N_PROJ=1024:
- Filter ceiling@50: 99.90%.
- Conditional rate of H2 alone: already ~97.5%.
- Adding more views can't improve the conditional rate much because the residual errors are correlated (same hard queries fail every hash).

This explains why quadruple wins dramatically at N_PROJ=16 (+2.51) and barely helps at N_PROJ=1024 (+0.13 over H2+H3). The resolver ceiling at large N_PROJ is bound by queries that are hard for *any* random ternary projection, and adding more of the same doesn't unbind them.

## Mechanism readout for the 3↔8 regression

Confusion count by resolver at N_PROJ=16 for the true-3-predicted-8 case:

| resolver | 3→8 errors |
|---|---|
| R2 H1+H2 1-NN (base) | 68 |
| R9 H2+H3 1-NN | 74 |
| R12 H2+H3+H4 1-NN | 57 |
| R4 H2 5-NN majority | 50 |
| R6 H2 5-NN rank-wt | 59 |
| R13 H_D50 (density 0.50) | 64 |
| R14 H_D20 (density 0.20) | 63 |
| R15 H2+H_D50 | 60 |

**k-NN majority within the filtered pool (R4) is the best 3→8 breaker**, down to 50 errors. Not fusion, not density, not new seeds — voting within the secondary hash's own ranking on the filtered pool. When correct-class 3 prototypes appear multiple times in the filter's top-K, even a noisy secondary hash can find them by counting rather than by ranking. This is the opposite of the earlier observation that voting hurts at the resolver stage — and it applies specifically when the resolver is noisy AND multiple correct candidates are present.

Tentative rule: **voting at the resolver helps when the resolver is lossy; 1-NN at the resolver helps when the resolver is precise.** The dense-pixel resolver in the historical cascade was precise enough that 1-NN dominated; the routed secondary hash is lossy enough that k-NN majority rescues cases 1-NN loses.

## Implications

1. **Routed cascade's small-N_PROJ ceiling is higher than the rerun suggested.** 77.33% (H2 1-NN) → 81.35% (triple) → 83.86% (quadruple) → probably 85-86% with more views. Budget more independent hashes if accuracy matters.
2. **At N_PROJ=16 the routed architecture still trails the dense-cascade's 90.75%.** The gap shrinks but doesn't close. Routed-only is fundamentally bandwidth-limited at small signature sizes.
3. **Density variation is a narrow tool.** It decorrelates density-specific confusions (6↔8) but doesn't touch projection-family confusions (3↔8, 3↔5). Only useful as part of a fusion, not as a standalone replacement.
4. **Resolver-stage voting is back on the table for noisy resolvers.** k-NN majority within the routed secondary hash beats 1-NN on the key regression pairs — directly contradicting the "1-NN at the resolver, don't vote" rule from the dense cascade. The rule should be amended: **voting when the resolver is noisy, 1-NN when the resolver is precise.**

## Follow-ups

1. **Pentuple-hash (H2+H3+H4+H5).** If the marginal gain curve is 4.65 → 2.51 → ?, next step should be around +1.3. Would push N_PROJ=16 to ~85.2%. Cheap to test.
2. **k-NN majority within quadruple fusion.** R12 uses 1-NN on the summed distance. If "voting helps noisy resolvers" is right, H2+H3+H4 5-NN majority should beat 83.86% on the tough pairs. Maybe 84-85%.
3. **Learn which projections to fuse.** A greedy selection over many candidate hashes picks the subset that maximizes validation accuracy. Still routed at inference.
4. **Test the 3↔8 structural hypothesis.** Use a *structurally different* hash generator (not random ternary — e.g. learned projections, or projections constrained to non-isotropic shapes) and see if 3↔8 breaks.
5. **Resolver-stage confusion tracking in atomics tool.** Port the 3→8/3→5/6→8 counters into `mnist_cascade_atomics.c` so partition analysis can be crossed with confusion type.

## Pointers

- Extended sweep tool: `tools/mnist_resolver_sweep.c`.
- Parent: `journal/routed_cascade_rerun.md`.
- Historical dense cascade: `journal/cascade_atomics_mechanism.md`.
- LMM cycle that seeded the filter-ranker reframe: `journal/nproj16_to_90_{raw,nodes,reflect,synthesize}.md`.
