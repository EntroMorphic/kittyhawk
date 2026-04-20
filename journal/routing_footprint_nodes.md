# Routing Footprint as Primary Metric — NODES

**Date:** 2026-04-20

---

## Node 1: The route IS the representation

The multi-table LSH architecture produces a routing footprint for every image: which tables place it in which buckets, at which probe radii, alongside which co-candidates. This footprint is M-dimensional (one decision per table) × radius-deep × co-candidate-rich. It's a representation of the image in routing space, not pixel space.

**Connection to thesis:** NORTH_STAR says "routing asks the shape of the computation itself." The routing footprint IS the shape of the computation for each image. Using it as the metric is the thesis in its most literal form.

## Node 2: Three tiers of routing signal

The routing footprint decomposes into three tiers of decreasing granularity:

**Tier 1 — Co-bucket membership.** Two images share a bucket key in table m. This is a 1-bit signal per table: same key or not. At M=64 tables, this gives a 64-bit co-routing vector. Jaccard similarity on this vector = fraction of tables that route both images to the same bucket.

**Tier 2 — Probe-radius overlap.** Image A reaches prototype B at radius r_m in table m. The vector of radii (r_0, r_1, ..., r_{M-1}) is the probe-depth profile. Two prototypes with similar probe-depth profiles relative to a query are routing-similar at a finer grain than co-bucket membership.

**Tier 3 — Co-candidate overlap.** The union of candidates produced by probing all M tables for query Q is a SET of prototype indices. Two queries that produce heavily overlapping candidate sets are routing-similar. This is the richest signal — it captures not just "did they land in the same bucket" but "did the multi-probe expansion reach the same neighborhood."

## Node 3: The structural zero is a routing signal

In the current architecture, a zero trit means "value within the neutral band." Hamming distance assigns cost 0 to matching zeros — treating shared uninformativeness as neither positive nor negative evidence.

But shared sparsity IS positive evidence. Two images that are both uninformative in the same spatial region share a structural property: the neutral band captured the same part of the image for both. This is exactly the "zero is a location on the lattice" claim from NORTH_STAR.

A routing-aware metric would score shared zeros differently from shared non-zeros:
- Shared +1/+1 or -1/-1: both signals agree (strong evidence)
- Shared 0/0: both routed through the neutral band at this location (structural evidence)
- Mismatch +1/-1: signals disagree (strong counter-evidence, cost 2 in Hamming)
- Mismatch ±1/0: one is informative, one is neutral (weak counter-evidence, cost 1 in Hamming)

The question is whether scoring shared-zero as positive evidence (not just zero cost) improves discrimination.

## Node 4: Radiusaware failed because it was a penalty, not a metric

The `radiusaware` resolver (Phase B.1, falsified) added `lambda × min_radius[c]` to each candidate's Hamming distance — penalizing candidates found at deeper probe radii. It degraded monotonically with λ.

Why: min_radius is a property of the (query, prototype) pair in a single table's routing. It's noisy and redundant with Hamming distance (high Hamming → likely found at high radius). The penalty adds noise to a signal that already carries the same information.

The routing footprint hypothesis says: don't penalize individual radii. Compare the *vector of radii across tables* between prototype and query. A prototype found at radius 0 in 14 of 16 tables is routing-close even if its Hamming distance is moderate. A prototype found at radius 2 in all 16 tables is routing-far even if an accidental bit-match gives low Hamming.

## Node 5: GSH is already a routing-footprint metric — but lossy

GSH encodes the routing footprint as: for each table, what class label does the nearest candidate carry? This is a compressed routing footprint — it throws away which candidate was nearest and keeps only its label. The compression from (candidate_index, distance) to (class_label) is extreme and lossy.

GSH agreement (two instruments predict the same class) is a confidence filter, not a similarity metric. But the raw signal — which candidates each table selects — is richer than what GSH extracts from it.

## Node 6: The computational cost of routing-footprint distance

**Co-bucket Jaccard (Tier 1):** Compare M-bit vectors. At M=64 this is 8 bytes; popcount of XOR gives Hamming distance on the co-routing vector. Cost: one popcount_dist call at 8 bytes ≈ 3 ns. Negligible.

**Probe-depth cosine (Tier 2):** Compare M-element radius vectors. At M=64, integer dot product of two 64-byte vectors. Cost: ~50 ns with NEON. Negligible.

**Candidate-set Jaccard (Tier 3):** Compare two sets of prototype indices. Set sizes are typically 50-500. Sorted merge: O(|A| + |B|). Cost: ~1-10 μs depending on set sizes. Non-trivial but feasible.

The first two tiers are essentially free. Tier 3 is the expensive one but also the richest.

## Node 7: The emergence question

Does the routing footprint carry classification signal that is NOT already captured by the trit-level Hamming distance?

If the footprint is just a noisy proxy for Hamming distance (high footprint similarity ↔ low Hamming distance), then using it adds noise, not signal. Radiusaware already tested one version of this and found it adds noise.

If the footprint carries INDEPENDENT signal — images that are routing-similar but trit-distant, or routing-distant but trit-close — then combining the two metrics should improve over either alone.

The selective scorer already exploits one case of this: when LSH and GSH disagree, the routing signal (GSH) and the trit signal (Hamming) are giving different answers, and pair-IG re-ranking helps. The question is whether this disagreement regime is the exception or the rule.

**Testable prediction:** If routing-footprint distance is informative beyond Hamming, then the CIFAR-10 images where routing and Hamming disagree should cluster by class in routing space but NOT in Hamming space. Measurable by comparing class purity of the k-nearest-by-routing vs k-nearest-by-Hamming for the disagreement subset.
