---
date: 2026-04-14
phase: RAW
topic: What mechanism makes rank-weighted k=5 voting recover ~9 cases per seed?
---

# Raw thoughts

What actually happened when rank-weighted k=5 produced +0.09% over majority k=3 at 2.6σ. I want to decompose this beyond "weighted voting helps sometimes" into "here is exactly the interaction between the weighting scheme, k, and the failure distribution that produced this result."

## What I know from the numbers

The six configurations tested, all at deskewed N=2048 across 3 seeds:

| Rule | k=3 | k=5 |
|---|---|---|
| Majority | **97.79 ± 0.05** | 97.77 ± 0.02 |
| Distance-weighted | 97.84 ± 0.04 | 97.78 ± 0.03 |
| Rank-weighted | 97.72 ± 0.06 | **97.86 ± 0.01** |

Structure that jumps out:

1. Going from majority k=3 to majority k=5 *loses* 0.02% (97.79 → 97.77). Increasing k adds hedge but also noise.
2. Distance-weighting at k=3 helps (+0.05); at k=5 it's essentially flat (+0.01 vs its k=3 baseline).
3. Rank-weighting at k=3 *hurts* (-0.07).
4. Rank-weighting at k=5 *helps most* (+0.09).

So the pattern isn't "weighted voting helps." It's specifically "rank-weighting helps BUT ONLY AT LARGER k."

## What that constraint tells me

Rank weights at k=3 are {3, 2, 1}. Top-1's weight is 3× a single bottom vote. If top-1 is wrong, its wrongness is tripled relative to the other two votes combined. That's why rank-k=3 hurts: it amplifies top-1 errors.

Rank weights at k=5 are {5, 4, 3, 2, 1}. Top-1's weight is still large (5), but it's now only ~35% of the total (5/15). The other ranks can outvote it when they agree.

So: rank-weighting needs enough ranks to hedge against top-1 being wrong. k=3 is too small; the hedge isn't there.

## What specific case pattern does rank-k=5 recover?

Let me think through exemplars:

**Case A.** Top-5 = {correct, wrong_X, correct, wrong_X, wrong_X}.
  Labels at ranks {1, 2, 3, 4, 5}. Correct class: ranks 1, 3 → rank-weights 5+3=8. Wrong class X: ranks 2, 4, 5 → weights 4+2+1=7. **Rank-k=5: correct wins 8-7.**
  Majority k=5: correct 2 votes, wrong 3. **Majority k=5: wrong wins.**
  Majority k=3: top-3 is {correct, wrong_X, correct}. Correct 2, wrong 1. **Majority k=3: correct wins.**

So this case: majority-k=3 gets it right, majority-k=5 gets it wrong, rank-k=5 gets it right.

**Case B.** Top-5 = {wrong, correct, correct, wrong, wrong}.
  Labels ranks {1, 2, 3, 4, 5}. Correct at ranks 2, 3 → weights 4+3=7. Wrong at ranks 1, 4, 5 → weights 5+2+1=8. **Rank-k=5: wrong wins 8-7.**
  Majority k=5: correct 2, wrong 3 → wrong wins.
  Majority k=3: top-3 = {wrong, correct, correct}. Correct 2, wrong 1 → correct wins.

So case B: majority-k=3 right, majority-k=5 wrong, rank-k=5 wrong. Rank-k=5 does NOT recover this.

**Case C.** Top-5 = {wrong, correct, wrong, correct, correct}.
  Correct at ranks 2, 4, 5 → weights 4+2+1=7. Wrong at ranks 1, 3 → weights 5+3=8. **Rank-k=5: wrong wins 8-7.**
  Majority k=5: correct 3, wrong 2 → correct wins.
  Majority k=3: {wrong, correct, wrong}. Wrong 2, correct 1 → wrong wins.

Case C: majority-k=3 wrong, majority-k=5 right, rank-k=5 wrong. Rank-k=5 *loses* a case that majority-k=5 gets.

So rank-k=5 vs majority-k=5 is a trade. Rank-k=5 wins cases of shape A (top-1 correct, wrong cluster at close ranks). Majority-k=5 wins cases of shape C (correct cluster at later ranks, top-1 wrong). Net effect is +0.09 because shape A is more common than shape C in MNIST near-misses.

## Why shape A dominates in MNIST

If the projection is discriminative, the TRUE top-1 tends to be correct for most queries — random ternary projection with N=2048 gives enough information that the nearest signature is usually the right class. The failure mode is: when top-1 IS wrong, close-class prototypes cluster in the top-5, but so do multiple correct-class prototypes. The correct class being at top-1 + some-other-rank (not-bottom) is a more common near-miss pattern than correct class being only at bottom ranks.

Also: if top-1 is wrong, there's usually a STRONG wrong-class cluster (because digits that look similar have many look-alike prototypes). So the interesting question is whether the correct class manages a top-1 despite a wrong-class cluster; when it does, shape A arises.

## Why distance-weighting didn't help much

Distance-weighting uses weight = max_dist − d. For MNIST at N=2048, Hamming distances in the top-5 are typically 600-800, against max_dist = 4096. So weights range ~3296 to ~3496. A 6% variation at best. The discrimination between ranks is tiny.

Rank-weighting has a 5× discrimination (weight 5 to 1 at k=5). Much more decisive.

So: the EFFECTIVE discrimination of the weighting scheme matters. Flat profiles (distance-weighted when distances are close) don't add much over majority. Steep profiles (rank-weighted) DO add.

## What I'm uncertain about

- Is the +0.09% gain genuinely the mechanism I described, or did some other quirk contribute? I should probably sample some actual rank-k=5 wins from the data to verify shape A dominance.
- Is there a better weighting profile? Exponential (2^(k-i)) would be even steeper than rank. Might work better. Or might overweight top-1 too much and reintroduce the rank-k=3 problem.
- Does this mechanism generalize? On a different task (CIFAR-10, char n-grams) the failure distribution might look different. Shape A might not dominate there.

## First instincts to watch

- "Weighted voting is just better." No — rank-k=3 shows it's worse in some configs. The (rule, k) pairing matters.
- "Higher k is always a hedge." No — majority-k=5 slightly underperforms majority-k=3. More neighbors can add noise.
- "The trace predicted this well." Partially. The trace identified a set that includes the cases this recovers; it didn't identify the specific (rule, k) pair that helps most. The overestimate was in magnitude, and the sub-mechanism only became visible through running the experiment.

## Questions arising

1. Is shape A dominance testable by sampling actual top-5 labels on misclassified queries?
2. Does exponential weighting outperform rank-weighting?
3. Is there a weighting scheme that recovers BOTH shape A and shape C?
4. Does the optimal k depend on N_PROJ and the projection's discriminative power? At smaller N_PROJ (512), top-1 is less reliable; maybe k=7 becomes optimal.
5. What does this mechanism say about GENERAL failure-guided adaptation? Is "weight by top-rank heavily but hedge with more neighbors" a universal principle, or specific to Hamming-LSH?
