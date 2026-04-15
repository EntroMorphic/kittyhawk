---
date: 2026-04-14
phase: NODES
topic: What mechanism makes rank-weighted k=5 voting recover ~9 cases per seed?
---

# Nodes

## N1. The structure is (rule, k) coupled, not "weighted voting helps"
The six configurations tested don't sort by "weighted vs unweighted" or by "k value" separately. Rank-weighting *hurts* at k=3 and *helps most* at k=5. Distance-weighting helps slightly at k=3 and barely at k=5. The useful axis is the *pairing* of rule and k, not each independently.

## N2. Majority k=5 slightly underperforms majority k=3
Counter-intuitive for k-NN: more neighbors should hedge noise. Instead it slightly hurts (97.77 vs 97.79). The mechanism: ranks 4-5 carry more noise than signal for MNIST + ternary LSH; adding them to an equal-weight vote dilutes the top-3's information.

## N3. Rank-k=3 hurts because it triples top-1's weight
Weights {3, 2, 1}. Top-1 gets 50% of the total vote (3 of 6). If top-1 is wrong, its wrongness is structurally amplified. k=3 doesn't have enough ranks to hedge against top-1 errors.

## N4. Rank-k=5 works because it balances signal-preservation and hedge
Weights {5, 4, 3, 2, 1}. Top-1 gets 33% of total (5 of 15). Still heavy, but ranks 2-5 can outvote it when they agree. The hedge exists but isn't forced — decisive top-1s still dominate, noisy top-1s can be overruled.

## N5. Distance-weighting is effectively flat for Hamming-LSH on MNIST
Typical top-5 Hamming distances: 600-800. `max_dist - d` at max_dist = 4096 gives weights 3296-3496. A 6% ratio. That's not meaningful discrimination — distance-weighted k=3 behaves nearly like majority k=3.

## N6. Rank-weighting has structural discrimination
Weights 5:1 at k=5. 5× ratio between top-1 and top-5. Decisive discrimination regardless of distance distribution. This is why rank wins over distance on this task: rank's profile is steep by design, distance's profile is only steep when distances have large spread (not here).

## N7. Recovered case pattern (shape A)
Top-5 = {correct, wrong, correct, wrong, wrong}.
Rank-k=5: correct=5+3=8, wrong=4+2+1=7 → correct wins.
Majority k=5: correct 2, wrong 3 → wrong wins.
Majority k=3: top-3 = {correct, wrong, correct} → correct 2, wrong 1 → correct wins.
Rank-k=5 preserves majority-k=3's win in a case where simply extending to majority-k=5 would lose it.

## N8. Lost case pattern (shape C)
Top-5 = {wrong, correct, wrong, correct, correct}.
Rank-k=5: wrong=5+3=8, correct=4+2+1=7 → wrong wins.
Majority k=5: correct 3, wrong 2 → correct wins.
Majority k=3: wrong 2, correct 1 → wrong wins.
Rank-k=5 loses a case that majority-k=5 recovers.

## N9. Shape A dominates over shape C in MNIST near-misses
The net +0.09% improvement over majority-k=3 is the difference: shape-A gains minus shape-C losses. That shape A dominates implies top-1 is more often correct-but-outnumbered than correct-only-in-a-later-cluster.

## N10. Why shape A dominates (structural argument)
N_PROJ=2048 random ternary projections give enough information that the true nearest signature is usually the correct class. When top-1 IS correct, the failure pattern is: wrong class has cluster at ranks 2-4 that outvotes a lone correct top-1. Shape A. When top-1 is wrong (less common), correct class is typically at low ranks but not always with a cluster (because failure is often "correct class is one distinctive outlier among many wrongs"). Shape A is simply more common because top-1 is usually correct.

## N11. The trace overestimated because it aggregated shape A + shape C + un-recoverable
The trace's NARROW MISS bucket (74 cases) includes all three. Distance-weighting recovers ~5 cases (shape A only, partially). Rank-weighting at k=5 recovers ~9 (shape A, loses some shape C). Cases where top-1 is wrong AND correct class is at rank >3 are not recoverable by any vote rule; those need different interventions (projection changes, per-dim τ, signature refinement).

## N12. The general principle for future vote-rule tuning
Effective weighting = steep profile × sufficient k. A weighting scheme needs:
- **Steepness**: weight(rank 1) / weight(rank k) should be ≥ 2× at minimum.
- **Sufficiency of k**: k must be large enough that ranks 2..k can outvote rank 1 when they agree (else top-1 errors amplify).

Rank k=5 satisfies both. Distance-k=3 fails steepness. Rank-k=3 fails sufficiency.

## N13. Prediction: exponential weighting might outperform rank
Weights 2^(k-i): at k=5, {16, 8, 4, 2, 1}. Ratio 16:1. Steeper than rank (5:1). Top-1 weight = 52% of total. Possibly too steep — might reintroduce the rank-k=3 problem (top-1 errors amplify). Testable.

## N14. Prediction: this mechanism is Hamming-LSH-specific
The "distance-weighting is flat" observation depends on Hamming distance distribution having narrow spread. On a task with wider distance variation (deeper features, different metric), distance-weighting might have more discrimination and beat rank-weighting. Generality is untested.

---

## Tensions

### T1. Hedge vs signal amplification
More k = more hedge but also more noise. Majority k=5 slightly underperforms because noise ≥ hedge gain on this task. Weighted schemes recover by suppressing the noise while preserving the hedge.

### T2. Steepness vs top-1 dependence
Steeper weighting profiles make correct top-1 more decisive but also make wrong top-1 more damaging. Rank-k=3 sits in the bad zone (steep AND no hedge). Rank-k=5 sits in the good zone (steep AND adequate hedge). Exponential-k=5 might sit in the bad zone (too steep even with hedge).

### T3. Trace coarse-categorization vs mechanism specificity
The trace tool identified 74 NARROW MISS cases but couldn't distinguish which vote rule would help which case. The specific mechanism (shape A vs shape C) only became visible by running the experiment. The trace is useful for direction, not for precise recoverable-count prediction.

### T4. Task-specific vs general tuning
Rank-k=5 is the best on MNIST + ternary-LSH + N_PROJ=2048. It may not be optimal elsewhere. Failure-guided adaptation is per-task; tuning one setting doesn't generalize.
