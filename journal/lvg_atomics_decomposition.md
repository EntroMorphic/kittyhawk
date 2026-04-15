---
date: 2026-04-15
scope: Atomic decomposition of the local-vs-global contingency at N_PROJ=16
type: mechanism probe + P2 scope revision
tool: tools/mnist_lvg_atomics.c
parent: journal/meta_router_online_synthesize.md
---

# L-vs-Gq atomics: rescues and damages share nearly all observable features

The P1 gate passed: global quadruple rescues 891 queries at the cost of 331 damages, net +5.60%. But the gate only tells us that rescuable headroom exists — it does not tell us whether a meta-router can *find* the rescuable queries without triggering the damages. This probe answers that question and it reshapes the P2 design.

**Headline finding:** on every inference-available signal we measured, rescues (LW_GR) and damages (LR_GW) have nearly indistinguishable distributions. The meta-router can only cleanly detect *hard* queries (~60% disagreement threshold separates easy from hard) but cannot tell rescue from damage from double-fail within the hard set. **Expected P2 accuracy: ~88%, not the 92.77% oracle ceiling.** This is a real but bounded win.

## Contingency refresher (N_PROJ=16)

| cell | label | count | fraction |
|---|---|---|---|
| LR_GR | both right | 8055 | 80.55% |
| LR_GW | **damage** | 331 | 3.31% |
| LW_GR | **rescue** | 891 | 8.91% |
| LW_GW | both wrong | 723 | 7.23% |

Oracle ceiling (L right ∪ Gq right): 9277 = 92.77%.

## A. H1 min-distance — **weak separator**

| cell | d≤1 | d≤2 | d≤4 |
|---|---|---|---|
| both right | 90.1% | 8.6% | 1.3% |
| damage | 83.4% | 12.1% | 4.5% |
| rescue | 84.3% | 13.1% | 2.5% |
| both wrong | 81.3% | 14.8% | 3.9% |

All four cells cluster at d≤2. The filter is so confident globally that its own min-distance carries almost no information about cascade outcome. **Unusable as a rescue detector.**

## B. H1 tied-min count — **weak separator**

| cell | tied=1 | tied=2-4 | tied=5-15 | tied=16+ |
|---|---|---|---|---|
| both right | 23.6% | 31.2% | 31.3% | 13.9% |
| damage | 24.2% | 40.8% | 27.2% | 7.9% |
| rescue | 24.2% | 34.1% | 29.5% | 12.1% |
| both wrong | 27.5% | 34.2% | 30.3% | 8.0% |

Damages skew slightly toward tied=2-4 (40.8% vs 31.2% for easy). Rescues look structurally like easy queries. A threshold on tied-count can't separate rescue from damage because damages are *more* tied-2-4 than rescues are.

## C. Correct-class rank in H1 top-50 — **strong observable, unavailable at inference**

| cell | rank 1 | 2-5 | 6-10 | 11-20 | 21-50 | >50 |
|---|---|---|---|---|---|---|
| both right | 63.2% | 27.2% | 5.8% | 2.5% | 1.3% | 0.0% |
| damage | 41.7% | 34.7% | 10.6% | 10.0% | 3.0% | 0.0% |
| rescue | **21.1%** | 30.5% | 15.0% | 14.5% | 11.5% | **7.4%** |
| both wrong | 18.4% | 27.8% | 16.3% | 14.7% | 12.5% | **10.4%** |

This is the clearest mechanism signal in the probe — but it requires knowing the true label and is therefore useless at inference time. Two diagnostic readings:

1. **Filter-miss failures dominate the hard cells.** 7.4% of rescues and 10.4% of double-fails have correct class *outside* H1's top-50 pool. These queries cannot be fixed by any resolver that only reads the filtered pool — global fusion rescues them precisely because it sees all 60K prototypes. This is the specific architectural gap H1-top-50 imposes.

2. **Rescues and double-fails look identical on this axis.** Same distribution across all rank buckets. The meta-router cannot distinguish them because the feature that separates them (rank of correct class) is hidden behind the label.

## D. Ensemble disagreement — **strong hard-vs-easy separator, blind to cell**

| cell | H2-H3 disagree | H3-H4 disagree | H2-H4 disagree |
|---|---|---|---|
| both right | 21.7% | 23.1% | 20.1% |
| damage | 59.5% | 58.6% | 52.6% |
| rescue | 67.9% | 68.2% | 67.7% |
| both wrong | 58.9% | 61.6% | 60.3% |

This is the **clearest observable signal** for hard-vs-easy. Disagreement triples from ~21% (easy) to ~60% (hard). But look at the three hard rows: 60%, 68%, 59%. They're essentially flat across the hard cells. **Disagreement says "this query is hard" without saying "this query is rescuable."**

## E. Fusion pick rank in H1 ordering — **mild damage signal**

| cell | rank=1 | 2-5 | 6-10 | 11-20 | 21-50 |
|---|---|---|---|---|---|
| both right | 4.4% | 13.7% | 13.5% | 21.9% | 46.5% |
| damage | 2.7% | **6.7%** | 10.9% | 26.9% | **52.9%** |
| rescue | 2.6% | 13.0% | 13.0% | 21.8% | 49.6% |
| both wrong | 2.9% | 13.0% | 12.7% | 21.7% | 49.7% |

One modest signal worth noting: damages have fusion picking *deeper* in the H1 ordering (52.9% in ranks 21-50 vs 46.5% for easy) and *less often* in the 2-5 bucket (6.7% vs 13.0% for rescues). The interpretation: when damage happens, local fusion has gone far down the H1 pool to find its (correct) answer, probably because the right prototype is rare in the neighborhood. Global fusion then trips on a superficially similar wrong-class prototype from the wider 60K pool.

This is a 6-8 point gap, marginally usable. Small sample for damages (331 queries) makes it noisy. Worth packing into the routing-context signature but not a silver bullet.

## F. Fusion margin — **very weak**

| cell | m=0 | m≤2 | m≤5 | m≤10 | m>10 |
|---|---|---|---|---|---|
| both right | 23.6% | 53.0% | 20.2% | 3.1% | 0.1% |
| damage | 24.8% | 57.1% | 16.3% | 1.8% | 0.0% |
| rescue | 29.3% | 55.2% | 15.2% | 0.3% | 0.0% |
| both wrong | 21.0% | 55.6% | 19.9% | 3.5% | 0.0% |

Rescues skew slightly toward m=0 (29.3% vs 23.6% easy). Damages are close to easy. Marginal separator.

## G. Per-class and per-confusion mass

Rescue and damage cell counts by true class:

| class | rescues | damages | rescue:damage ratio |
|---|---|---|---|
| 0 | 50 | 15 | 3.3 |
| 1 | 4 | 3 | 1.3 |
| 2 | 104 | 39 | 2.7 |
| 3 | 94 | 65 | **1.4** |
| 4 | 117 | 34 | 3.4 |
| 5 | 130 | 47 | 2.8 |
| 6 | 77 | 19 | 4.1 |
| 7 | 67 | 17 | 3.9 |
| 8 | 145 | 52 | 2.8 |
| 9 | 103 | 40 | 2.6 |

**Class 3 has the worst rescue:damage ratio (1.4).** On class 3 queries, global damages almost as often as it rescues. Classes 4, 6, 7 are the best escalation targets (ratios 3.4-4.1). Class 1 is essentially solved (4 rescues, 3 damages — negligible).

Top L-confusion pairs and their global-fix status:

| true→pred | L err | Gq err | Δ fixed |
|---|---|---|---|
| 4→9 | 123 | 83 | **+40** |
| 8→3 | 62 | 39 | +23 |
| 8→5 | 51 | 28 | +23 |
| 5→8 | 50 | 28 | +22 |
| 2→8 | 49 | 27 | +22 |
| 5→0 | 39 | 21 | +18 |
| 3→8 | 57 | 42 | +15 |
| 9→4 | 56 | 44 | +12 |
| 5→3 | 59 | 50 | +9 |
| 3→5 | 57 | 51 | +6 |

**Global fusion fixes 4-9 the hardest (net +40 on `4→9`).** 3-5 and 5-3 are the most stubborn pairs: global fixes them by only 6 and 9 respectively, and creates new damages in the same pair (24 `3→5` damages, 15 `5→3`).

Damage-specific pairs (what Gq says when it's wrong and L is right):

| true→Gq_pred | damage count |
|---|---|
| 3→5 | 24 |
| 4→9 | 23 |
| 9→4 | 23 |
| 5→3 | 15 |
| 3→8 | 14 |
| 8→9 | 14 |
| 2→3 | 11 |
| 3→2 | 11 |

The 3↔5 and 4↔9 axes are *the same pairs in both directions*: global helps and hurts on the same confusion regions. This is a direct statement that **these are intrinsically ambiguous queries** where neither architecture has a decisive advantage — only a small statistical edge.

## The mechanism, in one sentence

**At N_PROJ=16, "hard" is visible as a single axis (ensemble disagreement) but "rescuable vs damaging" is not visible on any inference-available signal we could compute; the features that actually separate those two cells (rank of the true label in H1's pool) are hidden behind the label itself.**

This explains why the oracle ceiling (92.77%) is ~3 points above pure global (89.46%): the oracle can see the true label and route perfectly; the meta-router cannot see it and must route based on features that are ~80% informative about hardness but ~20% informative about rescuability.

## P2 scope revision

The original SYNTHESIZE set a success criterion of "≥2% over pure local AND beat random-escalation-at-same-rate by ≥1%." Given the atomic data, I now expect:

### Expected aggregate accuracy bounds

Using ensemble disagreement as the primary escalation signal:

- Rescues captured when disagreement is high: `891 × 0.68 ≈ 606`
- Damages triggered when disagreement is high: `331 × 0.59 ≈ 195`
- Easy queries escalated (no-op because Gq is also right): `8055 × 0.21 ≈ 1692`
- Double-wrong queries escalated (no-op because both are wrong): `723 × 0.60 ≈ 434`

Net accuracy change: `+606/10000 - 195/10000 = +4.11%`
Expected P2 aggregate: **~87.97%**

Compare:
- Pure local: 83.86%
- Disagreement meta-router (predicted): **~88.0%**
- Pure global: 89.46%
- Oracle ceiling: 92.77%

The meta-router is predicted to land ~1.5 points below pure global, at an escalation rate of ~30% (which is roughly `(hard_escalations + easy_false_positives) / total = (1235 + 1692) / 10000 ≈ 29.3%`). Cost ~50% of pure global.

### Revised P2 success criteria

1. **Primary:** aggregate accuracy ≥ 86% (≥ +2 over local) — still feasible.
2. **Stretch:** ≥ 88% — feasible within atomic predictions.
3. **Cost:** meta-router total cost < 60% of pure global. Feasible.
4. **Must beat random-escalation-at-same-rate.** Important because if the meta-router is only as good as random, the routing-context signature isn't adding anything.

### Caveats about P2 meaningfulness

With the atomic picture in hand, P2 is now primarily a **cost-efficiency test**, not an accuracy-ceiling test. The routing-context signature cannot separate rescue from damage more than what disagreement already does; P2's value is in confirming that the signature-based router gets the *same* net gain as a disagreement threshold but with lower false-positive rate on easy queries (saving cost).

If P2 accuracy lands between 86-88% with escalation rate under 35%, the experiment confirms the filter-ranker-at-global scale and validates the meta-router architecture as a cost-reducing optimization. If P2 lands below 86%, the routing-context signature is inferior to a raw disagreement threshold and the LMM cycle produces a negative result on its P2 scope.

## What would actually push above 88%

Reading the atomics, the gap between meta-router (~88%) and pure global (89.46%) is structural: ~1.5 points of accuracy live in queries whose rescue/damage labels are indistinguishable on observable features. To close that gap requires one of:

1. **Broadening H1's pool.** Many rescues come from correct-class-at-deep-rank queries. Widening to K=100 or K=200 increases the ceiling-at-K but costs linearly.
2. **Per-class routing priors.** Class 3 has 1.4 rescue:damage ratio — escalating class-3 queries rarely helps. Skipping class-3 from escalation saves damage. Requires the local prediction as a context feature.
3. **Two-stage escalation.** Cheap stage (disagreement → maybe escalate) followed by an evaluation stage (check global's prediction margin; only commit if confident).
4. **Broader hash family.** Projections that decorrelate class-3-specific failures. We saw small gains from H_D50/H_D20 in the resolver sweep; extending to more density families might help.

None of these are free and none will hit the 92.77% oracle without solving the fundamental observability gap.

## What I think the right P2 now is

Two options, listed by ambition:

**Option A (minimal):** build a simple disagreement-threshold meta-router (no learning, no bank, no k-NN lookup). Trigger escalation when any of {H1-H2, H2-H3, H3-H4} disagreement is high. Report accuracy and cost. This is the honest baseline for what observable-signal meta-routing can achieve on this contingency.

**Option B (LMM-faithful):** build the full routing-context k-NN bank from the SYNTHESIZE plan, pack {disagreement, fpick_rank, fmargin, predicted_class, tied_count, mind_bucket} into a 16-trit signature, cold-start from offline training, run live. Compare against Option A as the informative baseline. If Option B only ties Option A, the atomics prediction was right and the signature-based architecture is unnecessary; if Option B beats Option A, the signature carries combinations the threshold misses.

**My lean:** build both. A is ~150 lines, B is ~500 lines. Running them side-by-side is the only way to know if the bank architecture adds value over a threshold, and it's the cleanest experimental comparison for the LMM cycle's closing synthesis phase.

## Pointers

- Tool: `tools/mnist_lvg_atomics.c`
- Parent P1 gate: `tools/mnist_local_vs_global.c`, `journal/meta_router_online_synthesize.md`
- Cascade atomics context: `journal/cascade_atomics_mechanism.md`
- Quadruple-hash baseline: `journal/routed_quadruple_decorrelation.md`
