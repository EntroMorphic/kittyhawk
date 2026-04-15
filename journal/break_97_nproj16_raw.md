---
date: 2026-04-15
scope: LMM cycle — can local + global routing with Trit Lattice LSH reach 97% at N_PROJ=16?
phase: RAW
---

# RAW: the 97% target at N_PROJ=16

User hypothesis: with local and global routing composed over the Trit Lattice LSH primitive at N_PROJ=16, we can break 97% accuracy on deskewed MNIST. Let me grapple honestly before reaching for a plan, especially with the discipline reset still fresh.

## Where we are right now

At N_PROJ=16 our measurements are:

- Pure H1 k-NN (dense scan, k=7 majority): 62.00%
- Dense L50_H1 (H1 filter + H2+H3+H4 resolver, O(N) scan): 83.86%
- Dense L50_H12 (H1+H2 fused filter + H3+H4 resolver): 88.44%
- Dense L200_H12 (fused filter + K=200): 88.87%
- Dense Gq (H1+H2+H3+H4 summed over all 60K): 89.46%
- Oracle ceiling over (L50_H1 ∪ Gq): 92.77%
- **Routed bucket consumer (signature-as-address, r≤2, MIN_C=100): 82.58% at 9.9 μs/query**

For reference, the pure-signature scaling curve at larger N_PROJ:

- N_PROJ=32 pure k-NN: 80.75%
- N_PROJ=64: 91.55%
- N_PROJ=128: 95.22%
- N_PROJ=256: 96.56%
- N_PROJ=512: 97.06%
- N_PROJ=2048: 97.79% (three-seed mean)

97% first appears in the scaling curve around N_PROJ=512. So the information density required to hit 97% is approximately what 512 independent random ternary projections carry — about 512 × log₂(3) ≈ 812 bits of per-query signal. At N_PROJ=16 each projection carries ~25.4 bits of signature, so reaching 97% via independent tables requires something like **32 independent 16-trit tables**. That's the first-order estimate before any of the "local routing helps because of multi-probe" effects kick in.

## What "local and global routing" probably means

The user's phrasing is ambiguous and I should name the ambiguity. Three readings:

**A. Multi-table LSH (classical).** Build M independent bucket indexes at N_PROJ=16, each with a different RNG seed. At query time, probe all M tables, union the candidate sets, run the resolver over the union. "Local" = per-table multi-probe within a single bucket index. "Global" = union of candidates across all M tables. This is the standard LSH literature technique for boosting recall without widening per-table radius, and it composes naturally with the routed bucket consumer I just built.

**B. Hierarchical indexing.** A coarse index narrows to a region; a fine index refines within that region. "Global" = the coarse routing to pick which fine-grained index to query. Classical metric index trees (M-tree, VP-tree, HNSW) have this shape but with a dense distance function at the coarse level. In our substrate the coarse level would itself be a hash, so it collapses into a two-level multi-table case.

**C. Cross-query coordination.** A "global observer" that sees query-to-query patterns and informs per-query routing. This is what the closed meta-router LMM cycle was about, and we deprecated it because observable signals couldn't separate rescues from damages.

Reading C is off the table — the meta-router cycle already settled that. Reading B collapses into Reading A in any routing-only substrate. **Reading A is the productive interpretation.** I'll work with it but flag explicitly in NODES so the user can correct if the ambiguity is real.

## Is 97% actually reachable at N_PROJ=16?

First-order information-theoretic bound: the signature codebook is 3^16 ≈ 43 million codes. 60K prototypes with random projections fill ~37 906 distinct buckets (from the routed bucket measurement). The codebook is heavily under-saturated. A single 16-trit hash can achieve 62% classification accuracy; the scaling curve says more trits help. The question is whether M independent 16-trit tables approximate the behavior of a single M×16-trit hash.

In classical LSH theory, multi-table does better than single-table at the same total information budget because:
1. Each table partitions signature space independently, so the UNION of their collision neighborhoods covers more input-space neighborhoods than a single hash at the same bit count.
2. Random projections concentrate on different axes; pooling tables averages out per-table projection bias.
3. Multi-probe within each table adds cost-efficient recall on top.

In the worst case multi-table is equivalent to a single bigger hash (when the tables are too correlated). In the best case it's better.

Concretely: if M=32 tables each with 16-trit hashes reach 97%, the accuracy per table would be... hmm, this is tricky. Each individual table's accuracy as a k-NN classifier is 62%. Their union's *recall* (correct prototype is in the union) approaches 100% as M grows. The key question is whether the resolver stage (summed distance or fused distance across all M hashes) can distinguish the correct-class prototype in the union.

My rough model: at M tables, the candidate union is of size ~M × avg_bucket_size ≈ M × 136 for our current bucket-consumer baseline. At M=32 that's ~4000 candidates — well below the 60K prototype set but large enough to include the correct class for virtually every query. The resolver then has to rank 4000 candidates using... what? The summed distance across all M hashes? That's 32 × 16 = 512 trits of summed distance per candidate. That's equivalent to running `popcount_dist` with a 512-trit signature, which is the N_PROJ=512 dense scan's cost shape per candidate.

Wait — this is a problem. If the resolver uses all M hashes summed, we're back to an O(N × M × sig_bytes) shape per candidate. Total work: 4000 candidates × 32 hash lookups × 4 bytes = ~500K operations per query. Better than a dense 60K × 512-trit scan but not dramatically so.

Let me think about this more carefully. The resolver doesn't need all M hashes — it only needs enough to rank the candidates reliably. And the candidates are already pre-filtered to be likely matches. So the resolver could use only a subset of the hashes, or a weighted fusion where distant hashes contribute less.

Alternatively: **the resolver could itself be bucket-based.** Instead of computing distance against the candidate set, look up each candidate's own bucket position in the M tables and score by agreement count. "How many of the M tables have this candidate in the query's bucket neighborhood?" That's a pure bucket operation, no distance computation needed at the resolver stage.

Hmm, that's interesting. Let me think. For each candidate c:
- score(c) = |{m in 1..M : c is in query's neighborhood in table m}|

This is just "vote count" — how many tables agree that c is a near neighbor. Candidates with high agreement across tables are more likely to be correct. No distance arithmetic at the resolver stage; only set membership.

This has an elegant shape and it matches classical multi-probe LSH scoring.

**Honest concerns I have:**

1. Will M=32 actually be enough? I'm extrapolating from the scaling curve, but multi-table at independent seeds may have diminishing returns if projections correlate (e.g., all ternary projections with density 0.33 have similar statistical structure).

2. Does the local multi-probe radius interact badly with M tables? If r=2 per table already covers ~99% of queries in one table, do M=32 tables just all cover the same queries and waste M-1 of them?

3. The bucket consumer's 82.58% floor was dominated by 175 empty queries at r=2. Multi-table might rescue those (a query empty in table m may not be empty in table m'), which would push accuracy up even at small M.

4. Cost. At M=32 tables, per-query wall time is 32 × 9.9 μs ≈ 317 μs. That's still 6× faster than the dense L50_H12 cascade (1950 μs) for a potentially much higher accuracy. Memory: 32 × 937 KB ≈ 30 MB of bucket indexes. Not free but very manageable.

5. Is there a smarter way than uniform M tables? Learned hash selection, supervised table construction, adaptive per-query table selection? All of these sit at the edge of "research direction" vs "what we can build in ~500 lines of C this session."

## What I lean toward

**Build multi-table routed bucket consumer with M as a knob.** Start at M=4, M=8, M=16, M=32. For each M, measure:
- Accuracy at multi-probe r=2
- Empty-query rate (should shrink as M grows)
- Wall time per query
- Candidate union size

Then plot the accuracy curve and see where it crosses 97%. If it does, we've broken the target. If it doesn't, either (a) the curve saturates below 97% meaning tables are too correlated, or (b) the resolver stage is the bottleneck and needs a smarter scoring rule than 1-NN over the union.

## Doubts

- **I'm unsure whether multi-table is monotonically good.** LSH literature says yes for recall; I don't know about accuracy-at-k=1. Worth measuring.
- **I'm unsure the user means multi-table by "local + global routing."** I should name Reading A explicitly and ask if it matches.
- **I'm unsure about the resolver.** Vote count across tables (set-membership scoring) vs summed distance (signature-space scoring) vs 1-NN in candidate union (single-hash scoring) — these may give different accuracies at matched M. The synthesize phase should sweep at least a couple.
- **I'm unsure about 97% specifically.** The scaling curve predicts 97.06% at pure N_PROJ=512 single-hash; I'm betting multi-table at N_PROJ=16×32 reaches similar. That's an extrapolation that could miss by a few points in either direction.
- **I'm unsure about the "global" framing.** If "global" in the user's sentence means "the union covers all N_train via bucket-routing with no dense scan," then rule 7 (signature-as-address) already satisfies it at M=1. Multi-table just strengthens the reach. But "global" might mean something architecturally distinct that I'm missing.

## What RAW surfaces for NODES

- Multi-table LSH at M ∈ {4, 8, 16, 32} is the mechanical experiment
- Resolver choice (vote-count vs summed-distance vs 1-NN-over-union) is a secondary sweep
- Multi-probe radius per table interacts with M (smaller r may suffice as M grows)
- Ambiguity in "local + global routing" must be surfaced, not silently assumed
- Cost-vs-accuracy curve matters — 97% at 1 ms/query is worse than 95% at 50 μs/query for many use cases
- The 175-empty-query floor at M=1 is a testbed: multi-table should crush it
- Oracle upper bound is reachable: take ceiling over all M tables' candidate unions, compute label coverage, that's the max any resolver can extract

Ready to extract nodes.
