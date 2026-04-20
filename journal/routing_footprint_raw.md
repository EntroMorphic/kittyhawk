# Routing Footprint as Primary Metric — RAW

**Date:** 2026-04-20
**Subject:** The hypothesis that the routing path through the multi-table LSH architecture carries classification signal that trit-level distance cannot recover.
**Trigger:** Birds-eye review after code quality refactor. Three instruments (LSH, GSH, pair-IG) are already extracting routing-pattern information. The question is whether this information is auxiliary or primary.

---

## Raw observations

### What the current architecture already measures

1. **LSH bucket co-occurrence.** Two prototypes that land in the same bucket in the same table share a spatial summary key — the first 16 trits of a permuted hierarchical summary. This is a routing decision: the spatial pooling chose which block-level patterns to hash on.

2. **Multi-table vote count.** `st.votes[idx]` counts how many of M tables placed prototype `idx` in the query's neighborhood. Higher vote count = more tables agreed on the route. The SUM resolver ignores this (sums Hamming distance). The VOTE resolver uses only this. Neither treats it as a similarity signal between prototypes.

3. **GSH agreement.** The Global Signature Hash encodes *which class label* each table's nearest candidate carries — the routing pattern across tables. When LSH and GSH agree on a prediction, accuracy jumps (P(correct|agree) = 56-59% on CIFAR-10 vs 46.6% overall). This is a routing-topology signal.

4. **Probe radius at discovery.** `min_radius[idx]` records the ternary Hamming cost at which each prototype entered the union. Radius 0 = exact key match (strong routing agreement). Radius 2 = two bit-flips away (weak). The `radiusaware` resolver tried to use this — it was falsified (monotone degradation with λ). But it was tested as a *penalty*, not as a *similarity signal between prototypes*.

5. **Pair-IG selective switching.** The selective scorer changes distance function based on a routing signal (LSH-GSH agreement). The *choice* of distance function is routing-driven. The distance itself is still trit-level.

6. **Cross-seed overlap on CIFAR-10.** 94% of queries have fixed fate across 3 random seeds. The routing path they take — which prototypes they land near — is structurally determined, not seed-dependent. This means the routing footprint is a property of the image, not of the table construction.

### What trit-level Hamming distance captures

- Per-trit independent match: each trit contributes 0 (match) or 1-2 (mismatch) to the distance.
- No correlation between adjacent trits.
- No awareness of spatial structure (pixel (3,4) and pixel (3,5) are independent dimensions).
- No awareness of which trits are informative for which class pair.
- Treats matching zeros as "both inactive, cost 0" — structurally identical to "both +1, cost 0."

### What trit-level Hamming distance misses

- **Block-level patterns.** A 2×2 block of (+1,+1,-1,0) carries a spatial texture signal. Hamming sees 4 independent trits. SSTT's 3-trit block encoder sees one of 27 symbols.
- **Sparsity structure.** Two images with zeros in the same spatial locations share a structural similarity (same regions are uninformative). Hamming assigns cost 0 to matching zeros — correct locally, but it loses the global information that the *pattern of sparsity* matches.
- **Routing co-path.** Two prototypes that a query reaches through the same bucket in 15 of 16 tables are routing-similar in a way Hamming can't express. The route overlap is a 16-dimensional binary signal (hit/miss per table) that's orthogonal to the trit-level distance.

### What the profiler tells us about computational headroom

- `popcount_dist` at N_PROJ=16: 4.4 ns, 225 Mops/s.
- The routing decision (probe, bucket lookup, vote increment) is ~4 ns per query-table pair.
- A routing-footprint distance that operates on M-dimensional vote vectors or co-occurrence counts would add O(M) integer ops per candidate — at M=64 this is ~64 integer compares, well under 100 ns. The headroom exists.

### Measurements that exist but aren't combined

| Signal | Where measured | How used now | Routing content |
|---|---|---|---|
| Bucket co-occurrence | `probe_cb` → `votes[idx]` | VOTE resolver (argmax class by sum) | Per-prototype route agreement |
| Probe radius | `min_radius[idx]` | `radiusaware` resolver (falsified as penalty) | Route distance to query |
| Table-level 1-NN label | `vote_labels[m]` | GSH vote encoding | Per-table routing decision |
| LSH-GSH agreement | selective scorer | Strategy switch (Hamming vs pair-IG) | Confidence in routing path |
| Cross-table Hamming distance | `glyph_resolver_sum` | Primary scorer | Trit-level, NOT routing-level |

### What I don't know

- Whether routing co-path between prototypes is more discriminative than their trit distance.
- Whether the routing footprint is stable across different M values (does M=16 footprint predict M=64 footprint?).
- Whether the structural-zero pattern (where images have zeros) correlates with class identity more than the non-zero pattern (what sign the non-zeros carry).
- Whether Fashion-MNIST's +1.4pp advantage over SSTT is already explained by routing-pattern discrimination (GSH) or by something in the trit distance itself.
