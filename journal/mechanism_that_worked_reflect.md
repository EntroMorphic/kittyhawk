---
date: 2026-04-14
phase: REFLECT
topic: What mechanism makes rank-weighted k=5 voting recover ~9 cases per seed?
---

# Reflect

## Core insight

**The winning mechanism is not "weighted voting helps" or "larger k helps." It is a specific structural match between the weighting profile's steepness and the k that provides hedge.** Rank-k=5 occupies the only spot in the (rule, k) search space where both properties are satisfied; every other spot fails on one or the other.

The mechanism in one compressed sentence: **rank-weighted k=5 preserves the signal-dominance of a correct top-1 while also allowing ranks 2-5 to outvote a wrong top-1, and no other (rule, k) combination we tested does both.**

This is a property of the *profile*, not of the classifier. Majority at any k has uniform weight (no signal-dominance preservation). Distance-weighted is uniform-ish in practice (weights don't vary enough). Rank-k=3 is signal-dominant but has no hedge. Only rank-k=5 sits in the "steep profile AND enough neighbors to overrule top-1 when they agree" region.

## Resolved tensions

### T1 (hedge vs noise) → RESOLVED: the profile decouples them
Majority k=5 fails because hedge comes with equal-weight noise. Rank-k=5 succeeds because the weighting profile suppresses ranks 4-5 (weights 2, 1) that would otherwise dilute the top-3's information. You get the hedge (k=5) without paying the noise cost.

### T2 (steepness vs top-1 dependence) → RESOLVED: k governs the tradeoff
Rank-k=3 is steep but has no counterweight to top-1 errors. Rank-k=5 is the same steepness profile but with enough neighbors that ranks 2-5 combined can outvote rank 1. Steepness is good; but at small k, steepness without counterweight is fatal.

### T3 (trace categorization) → RESOLVED: trace is directional, not prescriptive
The NARROW MISS bucket (74 cases) is inclusive: "some correct-class prototype was close." The actual recoverable subset depends on the specific shape A pattern (correct at ranks 1, 3 with wrong cluster at 2, 4, 5). Only running the (rule, k) sweep reveals which cases each configuration actually flips.

### T4 (task specificity) → ACKNOWLEDGED, NOT RESOLVED
Rank-k=5 wins here. Will it win on CIFAR-10, long-tailed classification, etc.? Untested. The mechanism ("steep profile + adequate hedge") might generalize, but the specific (rule, k) = (rank, 5) almost certainly won't translate uniformly across tasks.

## Hidden assumptions challenged

1. **"Weighted voting is better than majority."** False. Rank-k=3 hurts; distance-k=5 barely helps. The rule AND k must match.
2. **"Larger k is more robust."** False. Majority k=5 slightly underperforms k=3 on this task.
3. **"Distance-weighting is the natural choice."** False. On Hamming-LSH with narrow distance distributions, distance-weighting is effectively uniform and doesn't beat majority.
4. **"The trace predicts recoverable cases accurately."** False. The trace identified directionally correct cases but overestimated magnitude 3× because it conflated multiple recovery patterns.
5. **"The best configuration is independent of k."** False. The best rule CHANGES with k — majority best at k=3, rank-weighting best at k=5. Adaptation must sweep both.

## What I now understand

The failure-guided adaptation loop worked not because weighted voting is universally good, but because a specific (profile, k) combination matches the specific failure distribution of this problem:

- **Failure distribution** has many shape-A cases (correct top-1 outvoted by wrong-class cluster) and fewer shape-C cases (correct class only at later ranks in wrong-class-dominant top-5).
- **Rank-k=5** uniquely favors shape-A recovery because of its steep-profile-with-hedge structure.
- **The 9-case net gain** is shape A wins minus shape C losses; on MNIST, A > C by 9.

The mechanism is compositional. You can't reach it by optimizing one axis (k OR rule) at a time. The search space is (rule × k), and the gradient isn't smooth — k=3 with rank-weighting is worse than k=3 majority, while k=5 with rank-weighting is better than k=5 majority. Only a 2D search reveals the winning point.

## What this predicts about failure-guided adaptation more generally

1. **Adaptation is pairs or tuples, not single-axis.** Any future vote-rule / scoring-rule adaptation should sweep the hyperparameter combined with any adjacent parameter (k, threshold, k-of-T, etc.).

2. **Trace-predicted recoverable counts are upper bounds.** The trace's coarse failure categories are inclusive; they count cases where SOME intervention could help. Actual gain from any specific intervention is a subset. Plan for 1/3 to 1/2 of the trace's predicted recovery.

3. **Profile steepness is the key lever.** Distance-weighting fails because Hamming distances are narrow. Rank-weighting succeeds because its profile is steep by construction. If we want to push further (exponential, 2^(k-i)?), we need to test both steeper profiles AND whether the k has enough hedge.

4. **The mechanism generalizes to other routing-surface adaptations.** Per-class τ, confusion-pair masking, prototype pruning — each has analogous "profile × scope" structure. E.g., per-class τ is "which dims to threshold tighter" (profile) × "how many dims total" (scope). The same "optimize the pair, not the axis" pattern applies.

## What remains uncertain

- **Whether exponential weighting (2^(k-i)) beats rank.** At k=5, weights {16, 8, 4, 2, 1} ratio 16:1. Might be too steep — could reintroduce top-1 dominance problems. Testable.
- **Whether this generalizes to tasks where distance variance is larger.** On tasks where top-k Hamming distances span a wider range, distance-weighting might have real discrimination and outperform rank-weighting. Untested.
- **Whether rank-k=7 or rank-k=9 would win further.** More hedge with the same profile shape. Untested.
- **Whether this mechanism is specific to LSH classifiers.** In a routing transformer (which we haven't built), the "vote" is the final layer's decision; does the same (profile, scope) pairing principle apply? Untested.

## What to do next

The immediate mechanism is understood. Three follow-ups would deepen it:

1. **Exponential weighting at k=5 and k=7.** Verify the "too steep" hypothesis. ~30 minute experiment.
2. **Rank-weighted k=7.** Does more hedge help further, or does it start to flatten the profile effectively? ~30 minute experiment.
3. **Sub-categorize the trace's NARROW MISS bucket** to predict which cases each (rule, k) pair recovers. Requires reading the top-5 label sequence for each failure. ~45 minute tool addition.

None need gradients. All are integer statistics. This is the texture of what base-3 adaptation looks like: exploring a small discrete configuration space, using trace data to identify which configurations are worth testing, measuring honestly, and committing to the result that survives.
