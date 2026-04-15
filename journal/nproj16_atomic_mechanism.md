---
date: 2026-04-15
scope: Atomic investigation of the vote-rule inversion at N_PROJ=16
type: atomic probe
tool: tools/mnist_probe_nproj16.c
---

# N_PROJ=16 atomic mechanism: partition asymmetry explains the inversion

The complete scaling curve (`journal/full_scaling_curve.md`) showed majority voting beating rank-weighted at N_PROJ ≤ 16, reversing the pattern at N_PROJ ≥ 32. Focused probe reveals exactly why: the aggregate accuracy comparison hides that each strategy is optimal on its own partition of the input space.

## Setup

Single seed (42), deskewed MNIST, N_PROJ=16, density=0.33. Signature size: 4 bytes. Max Hamming distance: 32 bits. For each of the 10K test queries, compute all 60K (query, prototype) distances and record:

- Global Hamming-distance distribution
- Min distance per query
- Size of the tied-at-minimum set per query
- Class distribution within the tied set
- Whether correct class is in tied-min / at ranks 2-10 / not in top-10
- Majority k=7 and rank-weighted k=7 recovery rates, split by the location partition

## The tied regime is dense

97% of queries have min_d ≤ 2 bits. **52% have an exact signature match** somewhere in training (min_d = 0).

Tied-at-top-1 set size histogram:

| Size | Fraction of queries |
|---|---|
| Exactly 1 | 24% |
| 2–5 | 38% |
| 6–10 | 18% |
| 11–25 | 14% |
| 26+ | ~6% |

**Average distinct classes in tied-min set: 2.10 / 10.** A typical query is making a binary-ish discrimination among classes that all have prototypes at exactly the minimum distance.

## The correct-class location partition

Categorizing queries by where the correct class sits in the distance ranking:

| Location | Count | Fraction |
|---|---|---|
| In tied-min set | 7585 | **75.85%** |
| Elsewhere in top-10 | 1562 | 15.62% |
| Not in top-10 | 853 | 8.53% |

Most of the classification action happens in the tied-min regime. Everything else is minority.

## The conditional accuracy pattern

The crux:

| Vote rule | Tied-min partition (75.85%) | Elsewhere partition (15.62%) |
|---|---|---|
| Majority k=7 | **76.66%** recovered | 24.65% recovered |
| Rank-weighted k=7 | **77.65%** recovered | 18.76% recovered |
| Δ (rank − maj) | **+0.99%** | **−5.89%** |

Rank-weighted WINS in the dominant regime and LOSES in the minority regime. The weighted sum:

- Tied-min gain:  +0.99% × 75.85% = +0.75%
- Elsewhere loss: −5.89% × 15.62% = −0.92%
- **Net: −0.17%**

This matches the observed aggregate rank-wt-vs-majority gap almost exactly. The inversion is fully explained by partition asymmetry.

## Why each partition behaves as it does

### Rank-wt wins tied-min

When correct class has ≥ 2 prototypes in the tied set, rank-weighting concentrates vote mass on the top ranks. Multiple correct-class prototypes at high ranks (weights 7, 6, 5) dominate wrong-class prototypes at low ranks (weights 2, 1) even when majority count is equal.

Specifically: if tied-min has 3 correct-class and 3 wrong-class prototypes, majority is a 3-3 tie broken by class index. Rank-wt gives the top-3 positions weight 7+6+5=18; if all three happen to be correct (because insertion order in tied sets is arbitrary), correct gets 18 vs 3+2+1=6 for wrong. Random insertion order determines which three slots each class occupies; on average rank-wt gets a +50% weight swing when the correct class happens to be inserted first.

Over 75% of queries, this arbitrary-but-favorable rank assignment is enough to give rank-wt its +1% edge.

### Rank-wt loses elsewhere

When correct class is at rank 2 or beyond and top-1 is a wrong-class prototype, rank-wt's steepness amplifies the wrong top-1. Concretely:

- Correct at rank 2: majority gives correct 1/7 = 14.3% of the vote; wrong top-1 gives wrong (say) 3/7 = 42.9%. Majority gap: 28.6 points.
- Rank-wt: correct gets 6/28 = 21.4%; wrong top-1 gets 7/28 = 25% plus anything else. Rank-wt gap smaller in ratio but the wrong-top-1 amplification means any tie-breaking between wrong classes goes to the one that won rank-1.

More dramatic:
- Correct at rank 7 (majority 1/7 = 14.3%; rank-wt 1/28 = 3.6%). Rank-wt penalizes correct 4× more heavily than majority does when it's far from the top.

Averaged across the elsewhere partition (where correct can be at any rank 2-10), rank-wt is −6% worse.

## The architectural insight

**The aggregate "which vote rule is better" question is malformed.** At N_PROJ=16, neither rule is globally better. Each is optimal on its own partition of the input space:

- **Tied-min partition** (large tied set, correct among the tied): rank-wt wins.
- **Elsewhere partition** (singleton or small tied-min, correct at later ranks): majority wins.

An adaptive classifier that routes per partition can beat both:

Predicted accuracy under perfect partition-routing:
- Rank-wt applied to tied-min queries: 5890 / 7585 correct
- Majority applied to elsewhere queries: 385 / 1562 correct
- Nowhere queries remain lost: 0 / 853
- **Total: 6275 / 10000 = 62.75%**

Vs majority-only: 62.00%. Vs rank-wt-only: 61.83%. **Adaptive routing predicted +0.75% over the better of the two pure strategies.**

## How to detect the partition at inference time

No oracle needed — the partition signal is just the **tied-at-top-1 count**. This is directly computable during the k-NN:

- While scanning 60K prototypes, find min_d.
- Count how many prototypes have distance == min_d.
- If `tied_count >= threshold` (say 2), use rank-weighted.
- Else, use majority.

Implementation: ~10 lines added to the k-NN loop.

## Generalizing beyond N_PROJ=16

This pattern likely applies at any N_PROJ where tie counts are non-trivial — which per the probe data is anywhere below N_PROJ ≈ 64 (where distance space becomes wide enough that ties at top-1 are rare).

At larger N_PROJ:
- Tied-min set is usually a singleton (one nearest prototype).
- Rank-wt's tied-min advantage doesn't apply (no tied set to distribute weights across).
- Rank-wt's elsewhere-disadvantage also shrinks (correct is usually at top-1 or top-2, not rank 7).
- Both effects shrink; rank-wt's small-but-consistent elsewhere advantage wins.

At smaller N_PROJ (≤ 8):
- Tied-min set is typically huge (100+ prototypes).
- Rank-wt's rank-ordering within the tied set is arbitrary; weights go to random prototypes.
- Majority at least counts votes correctly regardless of order.

So the adaptive strategy is only needed in the narrow regime (N_PROJ ≈ 16-32). At small N_PROJ, majority dominates; at large N_PROJ, rank-wt dominates.

## Deeper insight: the amplification fallback pattern, refined

The prior amplification experiment (`journal/amplification_negative_result.md`) tried to route hard cases to a different classifier (pixel k-NN). It failed because pixel k-NN isn't actually better on hard cases — they're hard for every classifier.

Adaptive voting at N_PROJ=16 works on a different axis: it doesn't try to help hard cases with a stronger classifier. It picks the right strategy for each *shape* of easy case. Each strategy is already strong in its own partition; routing between them captures both strengths.

**The pattern: find an audit signal that separates input-space partitions where different strategies win, then route each partition to its best strategy.**

- Amplification: audit signal = K-projection agreement; strategies = routed vs pixel k-NN. Failed because pixel ≤ routed on hard cases.
- Adaptive voting (proposed): audit signal = tied-min count; strategies = majority vs rank-weighted. Predicted to succeed because neither is strictly dominant.

## Follow-up experiments

1. **Implement adaptive voting and verify the +0.75% prediction.** Straightforward — extend `mnist_probe_nproj16.c` or add a new tool. ~15 lines of code for the adaptive classifier; the distance-sorting is already done.

2. **Check generalization across N_PROJ.** Run the same probe at N_PROJ=32, 64, 128. Expected: partition asymmetry fades as N_PROJ grows. Adaptive voting should add less at larger N_PROJ.

3. **Sweep the tied-count threshold.** What's the optimal cutoff? 2? 5? Some seed-dependent value? Easy to sweep since the probe already records tied counts.

4. **Test adaptive voting on the best-accuracy config (N_PROJ=4096).** Expected: near-zero gain because tied-min is singleton for essentially all queries. Would confirm that adaptive voting is a small-N_PROJ-specific trick.

## Pointers

- Probe tool: `tools/mnist_probe_nproj16.c`.
- Parent scaling-curve journal: `journal/full_scaling_curve.md`.
- Mechanism-cycle prediction this confirms: `journal/mechanism_that_worked_synthesize.md` ("rank-weighted fails in the highly-tied regime").
- Prior-experiment pattern this refines: `journal/amplification_negative_result.md`.
