---
date: 2026-04-15
scope: Atomic decomposition of why the N_PROJ=16 cascade worked
type: mechanism
tool: tools/mnist_cascade_atomics.c
parent: journal/nproj16_cascade_result.md
---

# Cascade mechanism: the hash is a great filter and a terrible ranker

The cascade at N_PROJ=16 achieved 90.75% (K=50) / 92.72% (K=100). The pure-hash baseline was 62%. This probe decomposes the +30-point gap into its atomics. Result: the gap is not about adding pixel signal in some diffuse way — it is a specific asymmetry. **The 16-bit hash is excellent at not losing the correct class but terrible at ranking within the shortlist. Pixel L2 has no ranking problem.** The cascade works because these two signals fail on opposite sides of the filter-ranker distinction.

## The two numbers that explain everything

**Ceiling at top-50 (correct class present anywhere in top-50): 98.59%.**
**Hash-rank-1 correct: 55.48%.**

The hash is 43 points more accurate at "contains" than at "first." That gap is the filter-ranker asymmetry, and it's the entire reason cascade works.

## A. Ceiling curve — the hash preserves signal

| K | correct in top-K |
|---|---|
| 1 | 55.48% |
| 2 | 68.78% |
| 5 | 83.25% |
| 10 | 90.81% |
| 20 | 95.52% |
| 50 | 98.59% |
| 100 | 99.50% |
| 200 | 99.86% |

The 16-bit hash loses very little information about the correct class in absolute terms. By K=50 only 1.41% of queries have the correct class excluded from the candidate pool. But **the same hash gets top-1 right only 55.48% of the time.** The hash's ranking collapses to near-random inside the candidate pool, but the pool itself remains highly informative.

Why? 16 bits over 60K prototypes means ~40-50% of queries have multi-prototype ties at the minimum distance (the probe showed this). When ties dominate, rank-1 is assigned by insertion order — arbitrary. Meanwhile the *set* of prototypes at distance ≤ some small threshold covers most of the relevant neighborhood.

## B. Conditional resolver: pixel L2 is a near-perfect ranker

| K | P(cascade correct \| correct in top-K) |
|---|---|
| 1 | 100.00% |
| 10 | 90.30% |
| 20 | 90.45% |
| 50 | 92.05% |
| 100 | 93.19% |
| 200 | 94.45% |

Given the correct class is anywhere in the top-K, pixel L2 1-NN picks it at a ~91-94% rate — and the rate *increases* as K grows. This is the opposite of the feared "wider K confuses the resolver." Pixel L2 gets BETTER at picking correct as more candidates arrive, because more candidates → higher chance the *closest pixel neighbor* of the correct class is a tight match.

Cascade's hard ceiling at K=200 is: 99.86% × 94.45% ≈ 94.3%. The empirical result matched this exactly (94.32% global).

## C. Rescue/damage matrix at K=50

|  | cascade right | cascade wrong |
|---|---|---|
| pure-hash top-1 right | 5407 | 141 |
| pure-hash top-1 wrong | **3668** | 784 |

- **Rescued:** 3668 queries (36.68% of all queries).
- **Damaged:** 141 queries (1.41%).
- **Rescue:damage ratio = 26:1.**

Pure-hash top-1 was wrong on 44.52% of queries; cascade rescues 82% of those. The damage rate (pure-right → cascade-wrong) is negligible — about 2.5% of queries where pure-top-1 was correct. The cascade is almost all upside.

## D. Where do cascade's correct picks come from? — the hash doesn't rank

| hash-rank of cascade's correct pick | fraction |
|---|---|
| rank 1 (top-1) | 4.26% |
| rank 2 | 3.75% |
| ranks 3–5 | 8.89% |
| ranks 6–10 | 12.44% |
| ranks 11–20 | 19.93% |
| **ranks 21–50** | **50.72%** |

**Half of cascade's correct picks live in hash-ranks 21 through 50.** Only 4.26% come from hash-rank 1. This is the most important table in the probe.

Interpretation: the hash places the correct prototype *somewhere* in the top-50, but almost never at the front. Pixel L2 reliably finds it wherever it sits. The hash and the resolver work on orthogonal axes of the problem: the hash picks the neighborhood; the pixel L2 picks the right neighbor.

This reframes what "N_PROJ=16" means:
- Does N_PROJ=16 give good class separation? **Barely.** Top-1 is 55%.
- Does N_PROJ=16 give good neighborhood coverage? **Yes.** Top-50 ceiling is 98.6%.
- Is N_PROJ=16 useful? **Absolutely, as a filter.**

## E. Per-partition cascade accuracy

| Partition (from top-10) | count | cascade accuracy |
|---|---|---|
| correct in tied-min set | 7519 | **95.96%** |
| correct elsewhere in top-10 | 1562 | 86.94% |
| correct nowhere in top-10 | 919 | 54.62% |

- Tied-min partition: 96% — cascade essentially solves it.
- Elsewhere-in-top-10: 87% — cascade finds correct even when it's at rank 7.
- Nowhere-in-top-10: 55% — cascade still rescues over half, because K=50 is wider than the probe's top-10 window. The "nowhere" partition at K=50 is only 1.41% (from the ceiling table).

Comparison against the original probe at N_PROJ=16 with vote rules:
- Rank-wt on tied-min: 77.65%. **Cascade: 95.96%.** +18.3 points.
- Majority on elsewhere-top-10: 24.65%. **Cascade: 86.94%.** +62.3 points.
- The elsewhere partition was the biggest voting failure; cascade is its biggest win.

## F. Class-pair confusion delta

Top cascade-improved confusions (pure-hash k=7 → cascade K=50):

| true | pred | hash errors | cascade errors | Δ |
|---|---|---|---|---|
| 8 | 0 | 149 | 12 | +137 |
| 5 | 0 | 120 | 12 | +108 |
| 9 | 4 | 130 | 35 | +95 |
| 4 | 9 | 170 | 85 | +85 |
| 6 | 0 | 100 | 16 | +84 |
| 9 | 7 | 101 | 17 | +84 |
| 8 | 3 | 119 | 38 | +81 |
| 0 | 6 | 98 | 18 | +80 |

**Digit 0 was a sink under pure hash.** Five of the top eight improvements involve class 0 (either true or predicted). Interpretation: ternary projections with density 0.33 and popcount Hamming tend to produce similar sub-32-bit patterns for "digits with large dark blob" — 0, 8, 5, 6 all register high in the same projection cells. Pixel L2 separates them trivially because the *shapes* differ.

Worst regressions:
| true | pred | hash errors | cascade errors | Δ |
|---|---|---|---|---|
| 4 | 1 | 10 | 15 | -5 |
| 3 | 9 | 6 | 11 | -5 |
| 1 | 7 | 1 | 2 | -1 |

**Total damage: 11 queries across all regressions.** vs 137 gain on just the 8→0 pair alone. The asymmetry is overwhelming.

## G. Pixel-distance margin

Average relative margin (wrong_min − correct_min) / (wrong_min + correct_min + 1) over 9301 queries with both correct and wrong prototypes in top-50: **+0.3255.**

The correct-class nearest pixel prototype is on average ~33% closer to the query than the nearest-wrong prototype within the top-50 candidate pool. That's why pixel L2 1-NN is so reliable: the margin is large and consistent.

## The mechanism, in one sentence

**Ternary 16-bit LSH is a lossy locality hash: it preserves neighborhood membership (98.6% at K=50) but destroys rank information (55% at K=1) — and pixel L2 on a filtered neighborhood of ~50 is a high-margin ranker that doesn't care about the destroyed rank.**

Every other observation is a consequence of this:
- Why voting plateaus at 62%: voting reads the destroyed rank information.
- Why rank-weighted barely beat majority: both read destroyed signal.
- Why cascade hit 92.72% at K=100: the pixel ranker finds correct wherever it sits.
- Why damage rate is 1.4%: pixel L2 only misses correct when pure-hash-top-1 *already* happens to be correct AND a visually-similar wrong prototype is in the top-50 AND pixel distance happens to prefer the wrong one.
- Why digit 0 was a sink: hash's destroyed ranking funneled blob-like digits toward a few shared signatures.

## Why the earlier amplification failed, now fully understood

The prior amplification experiment routed hard queries to pixel k-NN over **all 60K** prototypes and gained nothing. The reason is clear from this probe:

- Pixel L2 over 60K prototypes has no filter and must rank globally. Global pixel-L2 is not 91% accurate — MNIST pixel k-NN alone is ~97% at full tuning but the amplification implementation wasn't tuned equivalently.
- **The filter isn't optional.** The cascade's ~91% conditional resolver accuracy is a property of the *filtered* pool, not of pixel L2 in the abstract.

Amplification tried to use pixel as a classifier; cascade uses pixel as a ranker over a filter. Same primitive, different role, 30-point accuracy difference.

## Scaling prediction for larger N_PROJ

As N_PROJ grows:
- Ceiling at top-50 stays near 100% (was 98.6% at 16, won't drop).
- Hash-rank-1 accuracy rises sharply (was 55% at 16, approaches 98% at 4096).
- Conditional resolver stays ~91-94% (pool composition unchanged).

Therefore cascade headroom shrinks as N_PROJ grows. At N_PROJ=4096 where pure-hash is 97.99%, cascade should gain ≤ 1 point. Predicted crossover where cascade no longer helps: **N_PROJ ≈ 256.** A sweep would confirm.

## Architectural takeaways

1. **Treat LSH signatures as neighborhood filters, not classifiers.** The whole Glyph substrate is routing-first; this finding is the strongest empirical validation of that philosophy. The hash routes; the resolver decides.

2. **Voting is the wrong primitive for top-k extraction from a coarse hash.** Voting collapses rank and distance, which is exactly what the hash has already destroyed. Use a resolver that can look at the candidates directly.

3. **The cost of pixel access on a filtered pool is trivial.** K=50 × 784 = 39K ops. Negligible next to the 960K-op primary hash pass.

4. **Pure-signature accuracy is NOT the right benchmark for a coarse hash.** Ceiling-in-top-K is the right benchmark, because that's what the hash is actually doing well.

## Follow-ups

1. Sweep cascade across N_PROJ values (8, 16, 32, 64, 128, 256, 512, 1024, 4096) to verify the shrinking-gap prediction.
2. Try cheaper resolvers: per-class centroid distance, per-feature subset projection, Hamming on secondary seed ensemble (beyond single secondary).
3. Test if K=200 or K=500 pushes past 94% (ceiling says 99.86% × 94.45% ≈ 94.3%; may saturate).
4. Characterize what the 5.7% unrecoverable queries look like (damaged + truly-nowhere). Candidates: out-of-distribution handwriting, ambiguous digits (4/9, 3/5).

## Pointers

- Tool: `tools/mnist_cascade_atomics.c`.
- Cascade result: `journal/nproj16_cascade_result.md`.
- LMM cycle: `journal/nproj16_to_90_{raw,nodes,reflect,synthesize}.md`.
- Original vote-rule probe: `journal/nproj16_atomic_mechanism.md`.
- Failed amplification (now explained): `journal/amplification_negative_result.md`.
