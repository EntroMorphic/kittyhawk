# Routing Footprint as Primary Metric — REFLECT

**Date:** 2026-04-20

---

## What the nodes are telling us

The architecture has been building instruments that extract routing-pattern information — GSH, vote counts, probe radii, selective switching — and treating them as auxiliary to Hamming distance. The nodes suggest these instruments are measuring a signal that Hamming cannot: the topology of the route through the multi-table structure.

But several things give me pause.

## Concern 1: Radiusaware already tested routing-as-signal and failed

Phase B.1 added `lambda × min_radius` to Hamming distance. It degraded monotonically. The post-mortem said: "multi-probe radius is a coarsening of information already present in sum_dist."

If radius is redundant with Hamming, why would a richer routing signal (co-bucket membership, probe-depth vector) be any different?

**Counter:** Radiusaware tested one bit of routing signal (single-table radius) as a penalty on an existing metric. The hypothesis here is about the *full routing footprint across all tables* as an independent metric. The single-table radius is noisy and correlated with Hamming. The cross-table pattern is a different kind of signal — it captures which *spatial regions* of the image matched, not how closely they matched.

**Verdict:** Not falsified by radiusaware, but the burden of proof is on the routing footprint to show independence from Hamming. The first experiment must measure this directly.

## Concern 2: GSH agreement is only 56-59% accurate on CIFAR-10

If routing patterns were strongly discriminative, GSH agreement should predict correctness at a higher rate. 56-59% conditional accuracy (vs 46.6% unconditional) is a real lift but not overwhelming.

**Counter:** GSH is a heavily compressed routing footprint — it keeps only the class label of each table's nearest candidate, discarding identity and distance. The raw footprint (which candidates, at which radii, in which tables) is orders of magnitude richer. GSH may be extracting 10% of the available routing signal.

**Verdict:** GSH's modest accuracy doesn't falsify the routing-footprint hypothesis. But it doesn't strongly support it either. Need to measure the raw footprint, not the compressed version.

## Concern 3: Is this just ensemble voting by another name?

Multi-table LSH with vote counting is structurally similar to ensemble classification: each table casts a vote, and the winner is the majority. The "routing footprint" may be a rebranding of ensemble agreement.

**Counter:** Ensemble voting treats each table's vote as a class prediction. The routing footprint treats each table's decision as a *feature dimension* in a routing-similarity space. The difference: ensemble voting asks "what does each table predict?" Routing footprint asks "are two images *routed the same way* across tables?" The former is a classification decision. The latter is a representation.

**Verdict:** Genuinely different from ensemble voting. But the distinction needs to produce measurably different behavior — if routing-footprint k-NN gives the same ranking as vote-weighted Hamming, it's a distinction without a difference.

## Concern 4: Co-candidate overlap (Tier 3) may be circular

If prototype B is in query A's candidate set, and prototype C is also in query A's candidate set, then B and C are co-candidates of A. But this just means B and C are both "near" A by the routing metric — which we're trying to define. The circularity: using the candidate set to define similarity, when the candidate set is produced by the similarity we're trying to improve.

**Counter:** The candidate set is produced by the BUCKET KEY (spatial summary), not by the trit distance. Co-candidates share a spatial summary but may have very different full trit signatures. The co-candidate signal is orthogonal to Hamming — it lives in the spatial-summary space, not the full-signature space.

**Verdict:** Not circular. The bucket key is a different representation (hierarchical spatial summary, 16 trits) from the full signature (784-3072 trits). Co-candidate overlap in key space IS independent of distance in full-signature space.

## What's surprising

The strongest version of this idea — that the routing footprint is the PRIMARY signal and Hamming distance is SECONDARY — would mean the current architecture has the ranking backwards. The filter (LSH routing) is carrying more classification information than the scorer (Hamming distance). That would explain the CIFAR-10 gap: the routing infrastructure is doing the hard work, but the final scoring throws away the routing signal and falls back to per-trit independence.

If true, the fix isn't a better distance metric. It's promoting the routing signal from filter to scorer.

## What's missing

1. **No measurement of routing-footprint independence from Hamming.** We don't know whether two prototypes with high co-bucket overlap but high Hamming distance tend to share a class.

2. **No measurement of sparsity-pattern similarity.** We don't know whether shared-zero patterns correlate with class identity on CIFAR-10.

3. **No measurement of cross-table routing stability.** We don't know whether the routing footprint at M=16 predicts the footprint at M=64. If it does, the routing signal is robust. If it doesn't, it's noise.

## The falsification test

**Prediction:** On CIFAR-10, for queries where Hamming k-NN is wrong but the correct class IS in the candidate union (oracle hit, resolver miss), the correct prototype will have higher routing-footprint similarity to the query than the Hamming-nearest incorrect prototype.

If this is false — if the Hamming-nearest wrong answer also has the highest routing similarity — then the routing footprint carries no independent signal and the hypothesis is falsified.

This is measurable with the existing architecture. No new code required for the measurement — only for the analysis.
