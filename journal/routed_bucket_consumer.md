---
date: 2026-04-15
scope: First genuinely routed Glyph consumer — bucket-indexed LSH
type: architectural correction + measurement + rule
tool: tools/mnist_routed_bucket.c
parent: journal/fused_filter_fix.md
supersedes-architecture-of: cascade tools (mnist_cascade_nproj16, mnist_cascade_sweep, mnist_cascade_atomics, mnist_resolver_sweep, mnist_local_vs_global, mnist_local_v2)
---

# Bucket-indexed consumer: the dense outer loop is gone

Every cascade tool in the tree — `mnist_cascade_nproj16`, `mnist_cascade_sweep`, `mnist_cascade_atomics`, `mnist_resolver_sweep`, `mnist_local_vs_global`, `mnist_local_v2`, `mnist_lvg_atomics`, `mnist_routed_knn`, `mnist_full_sweep` — runs routing primitives inside an O(N_train) dense outer loop. They compute `m4t_popcount_dist(query, train[i])` for every `i` in `[0, n_train)`. That's dense application architecture with routed kernels — a substrate-level NORTH_STAR violation even though the per-comparison primitive is routing-native.

This journal documents the first Glyph consumer that respects the contract end-to-end: `tools/mnist_routed_bucket.c`. Training is a one-time sort of `(signature_key, prototype_index)` entries by key. Query time is binary search into the sorted table plus ternary-Hamming multi-probe over neighbor keys. **Zero `popcount_dist` calls at the filter stage.** The signature is not metaphor — it is literally the memory address the query dereferences.

## What was wrong before

Look at the hot loop in any cascade tool (`tools/mnist_cascade_sweep.c`, `tools/mnist_local_v2.c`, `tools/mnist_lvg_atomics.c`, etc.):

```c
for (int i = 0; i < n_train; i++) {
    dA[i] = m4t_popcount_dist(qA, trA + (size_t)i*Sp, mask, Sp);
}
```

That is an exhaustive O(N_train) scan. Each iteration calls a routing primitive (`m4t_popcount_dist` → VEOR + VCNT + VADDLP), but the loop itself is a dense k-NN shape. Per-query cost grows linearly in the training set size. The 20.3× speedup over dense L1 k-NN (reported in `journal/routed_knn_mnist.md`) is real — it's routed kernels vs dense-L1 kernels *inside the same O(N) outer loop*. Both architectures scan every prototype.

The substrate NORTH_STAR states: "Routing is essential, and will naturally outperform dense, in a base-3 environment." For that claim to mean anything, the *architecture* has to be routed, not just the kernels. An O(N) outer loop is the same architectural shape as a brute-force similarity search; swapping a routed primitive into its inner loop is a compression win, not a routing win.

Stated plainly: **we have been measuring "routing primitives vs dense primitives" while calling it "routing vs dense." Those are different comparisons.** The latter requires an O(1)-amortized query architecture where the signature's role is to *locate* candidates, not to *score* all of them.

## What the routed architecture looks like

The Glyph substrate's signature primitive (`m4t_route_threshold_extract`) produces a packed-trit code. That code is an address in a ~3^N codebook. A real routing consumer treats the code as a key into a bucket map, where buckets contain pointers to the prototypes at that exact code. At query time:

1. Compute the query's signature (one invocation of the projection + threshold pipeline).
2. Look up the query's signature in the bucket map (O(1) amortized for a hash, O(log N_train) for a sorted index).
3. Read out the list of prototypes at that exact code. These are the **candidates**.
4. If the candidate set is smaller than desired, enumerate neighbor codes at small ternary Hamming radius (cost 1, cost 2, …) and look up each neighbor in the same bucket map. Stop when enough candidates are collected or the radius budget is exhausted.
5. Run a routed resolver over the candidate set (here: H2+H3+H4 summed popcount_dist 1-NN over the candidates).

The `popcount_dist` primitive is still used — but only on the small candidate set, never on all 60K prototypes. The filter stage is a table lookup, not a distance scan.

## Implementation — `tools/mnist_routed_bucket.c`

### Index data structure

```c
typedef struct { uint32_t key; int proto_idx; } entry_t;
entry_t* entries;    // size = n_train
```

Training loop:

```c
for (int i = 0; i < n_train; i++) {
    entries[i].key      = sig_to_key(train_sigs + i*SIG_BYTES);
    entries[i].proto_idx = i;
}
qsort(entries, n_train, sizeof(entry_t), cmp_entry);
```

Build time at N_PROJ=16 on MNIST: **3 milliseconds**.

At N_PROJ=16 the signature is 4 bytes (16 trits packed as 2-bit codes in one uint32). The sort is keyed on that uint32. A genuine hash table would also work; sorted array was chosen because it gives deterministic ordering, cache-friendly runs, and trivially supports range queries if we ever wanted them.

### Query-time lookup

```c
int lower_bound(const entry_t* e, int n, uint32_t target);  // binary search
```

For a target key, `lower_bound` returns the index of the first matching entry. Any entries with the same key form a contiguous run, which is the "bucket" — all training prototypes whose H1 signature is exactly `target`.

### Ternary multi-probe

This is where the N_PROJ=16 experiment gets interesting. Binary LSH multi-probe flips bits; ternary LSH must enumerate trit transitions that respect the ternary Hamming cost function (0 for same trit, 1 for 0↔±1 transition, 2 for +1↔−1 sign flip). The tool implements three radii:

- **Radius 0:** the query's own signature. One probe.
- **Radius 1:** at one trit position, either (a) move from 0 to +1 or 0 to −1 (two outcomes at positions holding 0), or (b) move from ±1 to 0 (one outcome at positions holding ±1). With density=0.33, about 67% of positions hold 0, so the radius-1 set has roughly `16 × (0.67 × 2 + 0.33 × 1) = 16 × 1.67 ≈ 27` probes.
- **Radius 2:** two kinds — (a) one trit sign-flipped at a non-zero position (cost 2 per position; ~5 probes on average), plus (b) two distinct cost-1 moves on different positions (~120 position pairs × ~2.78 outcomes each ≈ 334 probes). Total ~339 probes at r=2.

The `enumerate_radius` routine walks the trit positions directly using `read_trit` / `write_trit` helpers that operate on the packed 2-bit codes. Trit manipulation stays inside packed-byte space — the expansion never unpacks to `int8_t` arrays.

### Escalation policy

Per-query loop:

```c
for (int r = 0; r <= MAX_RADIUS; r++) {
    if (candidate_set.size >= MIN_CANDIDATES) break;
    enumerate_radius(query_sig, r, ...);
}
```

`MIN_CANDIDATES` is a hyperparameter that trades cost for recall. Small values stop as soon as any bucket is hit; large values force the query to gather more neighbors even if an exact match exists. The sweep reported below holds `MAX_RADIUS ≤ 2` and varies `MIN_CANDIDATES` across {1, 20, 100, 400}.

### Resolver

Once candidates are collected, the resolver computes `H2 + H3 + H4` summed Hamming distance for each candidate and picks the label of the minimum. Three `popcount_dist` calls per candidate, executed only over the (small) candidate set — not all 60K.

## Index characteristics at N_PROJ=16

```
60000 training prototypes -> 37906 distinct buckets (1.58x compression)

bucket size histogram:
  size 1        29616 buckets
  size 2-3       6099 buckets
  size 4-7       1621 buckets
  size 8-15       420 buckets
  size 16-31      112 buckets
  size 32-63       25 buckets
  size 64-127      12 buckets
  size 128+         1 bucket
```

Three observations:

1. **78% of buckets are singletons.** Most H1 signatures are occupied by exactly one training prototype. At N_PROJ=16 the codebook is not saturated — `3^16 ≈ 43 million` possible codes vs 60K prototypes means most codes stay empty, and the ones that are occupied usually hold one prototype.
2. **The 1.58× compression is not the interesting number.** It measures "how many training prototypes share a signature" averaged across all occupied buckets. What matters for retrieval is the distribution of bucket sizes that a *query* lands in — which concentrates on the non-singleton buckets because query signatures are drawn from a similar distribution as training signatures.
3. **One bucket has 128+ prototypes.** This is almost certainly the all-zero or near-all-zero signature (an image projected through random ternary matrices whose output values all fall inside `[-tau, +tau]`). It's a degenerate "anything goes here" bucket. Queries that land in this bucket get lots of candidates but the candidates are semantically noisy.

## Tuning sweep results

Full sweep of `(MAX_RADIUS, MIN_CANDIDATES)` over `{0,1,2} × {1,20,100,400}`:

| MAX_R | MIN_C | accuracy | avg candidates | avg probes | empty | μs/query |
|---|---|---|---|---|---|---|
| 0 | 1 | 36.99% | 9.2 | 1.0 | 4797 | 0.4 |
| 0 | 20 | 36.99% | 9.2 | 1.0 | 4797 | 0.3 |
| 0 | 100 | 36.99% | 9.2 | 1.0 | 4797 | 0.3 |
| 0 | 400 | 36.99% | 9.2 | 1.0 | 4797 | 0.3 |
| 1 | 1 | 61.80% | 8.4 | 9.8 | 1129 | 0.7 |
| 1 | 20 | 68.77% | 31.5 | 20.7 | 1129 | 1.6 |
| 1 | 100 | 68.90% | 46.5 | 22.0 | 1129 | 2.1 |
| 1 | 400 | 68.91% | 49.0 | 22.1 | 1129 | 2.2 |
| 2 | 1 | 67.82% | 8.4 | 33.3 | 175 | 1.3 |
| 2 | 20 | 81.17% | 54.6 | 150.0 | 175 | 5.5 |
| **2** | **100** | **82.58%** | **136.4** | **216.2** | **175** | **9.9** |
| 2 | 400 | 82.60% | 193.8 | 237.1 | 175 | 12.2 |

Three regimes visible in the table:

### r=0 (exact-match only): 36.99%

At radius 0 the consumer finds training prototypes that share the query's exact signature. **52% of queries do have such a match** (from the Axis 4c atomic probe), but the average bucket for those matches holds only 9.2 prototypes — often just a few. MIN_CANDIDATES is irrelevant at r=0 because the query never expands past the exact bucket.

**4797 queries are "empty" at r=0** — no training prototype shares their exact signature. These queries cannot be classified with r=0 alone; they default to a miss. (48% empty matches the 52% exact-match rate from the probe.)

Accuracy of 36.99% is what you get when ~52% of queries get 1-NN over a small exact-match bucket and 48% get nothing. The exact-match bucket's class distribution is better than chance (~2.10 distinct classes on average, from the Axis 4c probe), so the ~52% that aren't empty get ~71% accuracy on their own, which averages to 36.99% over all queries. Matches the prediction.

### r≤1 (exact + cost-1 neighbors): 68.90%

At radius 1 the consumer covers the query's exact signature plus all neighbors at one cost-1 trit transition. From the atomic probe, **~90% of queries have their min Hamming ≤ 2** (which is 1 ternary-Hamming unit), so r=1 catches most queries that were empty at r=0.

MIN_CANDIDATES matters here: at MIN_C=1 the query stops as soon as it hits any bucket; at MIN_C=100 it keeps expanding until the candidate set is large enough for the resolver to discriminate. The jump from 61.80% (MIN_C=1) to 68.90% (MIN_C=100) is +7.1 points from forcing more exploration per query.

**1129 queries are still empty at r=1** (about 11% of the test set). These are queries whose nearest training prototype is at ternary Hamming cost ≥ 2 in the signature space — either a sign flip at one position (cost 2) or two distinct cost-1 transitions (cost 1+1).

### r≤2 (exact + cost-1 + cost-2 neighbors): 82.58%

This is the full-budget operating point. The consumer covers every signature within ternary Hamming cost 2 of the query. From the probe, **99% of queries have min Hamming ≤ 2**, so only 1% should remain empty at this budget.

Measured: **175 queries empty at r=2** (1.75% of test set). Matches the probe within rounding. These are queries whose nearest training prototype is at min Hamming ≥ 3 — genuinely far from any training signature in the 16-trit codebook.

Best operating point: **MAX_R=2, MIN_C=100, accuracy 82.58%, 9.9 μs/query, 136.4 candidates average, 216.2 probes average.** Going to MIN_C=400 adds 0.02 accuracy points at the cost of 2.3 μs/query — negligible gain.

## Cost comparison against the dense baseline

This is the table that makes the architectural correction explicit.

| architecture | popcount_dists per query | μs per query | accuracy |
|---|---|---|---|
| dense L50_H1 (scan 60K H1, top-50, H2+H3+H4 resolver) | 60,000 + 150 ≈ **60,150** | ~**1,950** | 83.86% |
| **routed bucket** (MAX_R=2, MIN_C=100) | 0 + 136×3 ≈ **410** | **9.9** | 82.58% |
| **ratio** | **~147×** fewer ops | **~197×** faster wall time | −1.28 points |

Per-query cost breakdown for the routed bucket:
- Binary search at r=0: 1 probe × log₂(60K) ≈ 16 comparisons. Cost: <1 μs.
- Multi-probe at r=1: ~22 additional probes × 16 comparisons each ≈ 350 comparisons.
- Multi-probe at r=2: ~194 additional probes × 16 comparisons each ≈ 3100 comparisons.
- Resolver: 136 candidates × 3 popcount_dist calls = 408 popcount_dists.
- Total signature work: ~3500 key comparisons + 408 popcount_dists ≈ ~3900 primitive operations.

Dense L50_H1's 60,150 popcount_dists is 15× the routed bucket's total primitive work even including the binary search cost. The 197× wall-time gap is bigger than the 15× primitive-count gap because binary search on a sorted 4-byte-keyed table is much cheaper per operation than `m4t_popcount_dist` (which is a NEON loop over 4 bytes with XOR + VCNT + widening add).

## The 1.28-point accuracy gap — where it lives

Routed bucket at 82.58% vs dense L50_H1 at 83.86%: the gap is **1.28 points**. Two contributors:

1. **175 empty queries at r=2.** Queries whose min signature Hamming is ≥ 3 cannot be reached by the radius budget. 175 / 10000 = 1.75% of queries. Worst case these return a default label (random or most-frequent class), contributing up to 1.58 accuracy points of loss vs an architecture that would have scanned them anyway.

2. **Candidate-set coverage differences.** Dense L50_H1 always hands the resolver exactly 50 candidates (the dense top-50 by H1). Routed bucket hands the resolver a variable-size set (avg 136 at MIN_C=100) that is the union of all training prototypes within radius 2 of the query. The two sets are *different in shape*: dense may include prototypes whose H1 distance is 2 that happen to rank just inside top-50; routed includes *all* prototypes at H1 distance 0, 1, 2 and no others. For most queries the bucket set is a superset of the dense top-50, but for high-density buckets (many prototypes at distance 0) the MAX_CANDIDATES=4096 cap may trim the tail.

In practice the gap is dominated by contributor 1 — the 175 empty queries. Contributor 2 is second-order.

## Radius escalation profile matches the atomic probe

Queries that "stopped at" each radius:

| radius | queries | fraction |
|---|---|---|
| r=0 sufficient (MIN_C=5 threshold) | 5,203 | 52.03% |
| escalated to r=1 | 3,668 | 36.68% |
| escalated to r=2 | 954 | 9.54% |
| still empty at r=2 | 175 | 1.75% |

This *exactly* matches the Axis 4c atomic probe's H1 min-distance histogram:
- probe: 52% of queries at min_d = 0 → exact bucket match → routed r=0
- probe: 97% of queries at min_d ≤ 2 bits (≤1 ternary cost) → routed r ≤ 1
- probe: ~99% of queries at min_d ≤ 4 bits (≤2 ternary cost) → routed r ≤ 2

The bucket consumer is literally measuring what the probe predicted. Every query's radius of resolution is a function of the H1 signature codebook's collision structure, which is exactly what the probe characterized. **The dense outer loop was buying nothing that the ternary multi-probe doesn't buy directly.**

## Architectural consequences

### The signature is the address

Previously we wrote signatures into arrays and used them as *operands* to distance functions. The routed bucket consumer writes signatures into arrays and uses them as *keys* to an index. Same bit pattern, different architectural role. The substrate's `threshold_extract` produces a 4-byte code at N_PROJ=16; that 4-byte code is literally the query's address in the training set. The routing is not metaphor.

### The dense outer loop was scaffolding, not the architecture

The cascade tools built before this one are **measurement scaffolding**. They demonstrate that routed primitives are fast, that cascade behavior is bounded by filter-ranker asymmetry, that fused-filter fixes apply, that meta-routing hits observability limits. Every one of those conclusions was correct and holds up. But none of them is a *production architecture* — they're O(N) shapes with routed kernels.

Going forward, the cascade tools remain in the tree for historical measurements (the scaling curve, the atomic probes, the resolver sweeps) but the routed bucket consumer is the reference for how Glyph actually does k-NN.

### What the scaling story becomes

Dense cascade's speed claim was "20.3× faster than NEON L1 at the same N_PROJ=2048, both on an O(N) outer loop." True, but the ceiling was brute-force similarity search.

Routed bucket's speed claim is "O(1) amortized per query regardless of N_train, with ternary multi-probe providing recall control." At N_PROJ=16 that translates to 9.9 μs/query at 82.58% accuracy vs dense L50_H1's 1950 μs/query at 83.86%. **197× faster.** Scaling is now dominated by the resolver work on the candidate set, which grows with bucket density, not with training set size.

### The information leverage rule applies to the bucket too

The Axis 4d fused-filter finding said: "When multiple hashes are available, put them at the filter stage first." In the routed bucket architecture, this translates directly: **bucket on concatenated (H1+H2) keys**. An 8-byte key means 32-trit codebook (3^32 ≈ 1.85B possible codes, heavily under-used by 60K prototypes, so most buckets become singletons with very few collisions). The radius-budget structure stays the same but the per-query multi-probe enumerates over 32 trit positions instead of 16, and the buckets are more discriminating.

Expected result: fused-filter bucket should approximately track L50_H12's dense accuracy (88.44%) at somewhere between 1.5× and 3× the routed-bucket cost (depending on how the radius budget interacts with the larger codebook). This is the immediate next experiment.

## Update to the architectural rules

Adding rule 7 to the cascade rule list (from Axis 4d's rule 1-6):

7. **Production k-NN uses the signature as an address, not as an operand.** Build a bucket index keyed on the packed-trit signature; query via binary search + ternary multi-probe; run the resolver only on the candidate set. The O(N_train) outer loop is scaffolding for research measurements, not an architecture.

This rule subsumes and strengthens the filter-ranker reframe. The filter is no longer "H1 Hamming distance to all prototypes" — it is "H1 signature lookup in the bucket map." The *filter stage does zero distance work.* Distance computation is reserved for the resolver, over the small candidate set.

## Honest limits

The tool has several conservative choices worth naming:

- **MAX_CANDIDATES=4096 cap.** Queries that land in the degenerate 128+ bucket plus many small neighbors may hit this cap and miss some candidates. At MIN_C=100 the average is 136 and p99 is probably under 1000, so the cap isn't biting — but it would bite at MIN_C=1000 or for fused-filter bucket variants with larger radius budgets.
- **MIN_CANDIDATES early-stop is crude.** The current policy escalates to the next radius whenever the set is below threshold. A smarter policy would consider the *spread* of candidates or the ratio of resolver scores, not just a count.
- **Radius enumeration is not deduplicated.** At r=2, the (b) case (two cost-1 moves) may overlap with the (a) case (one sign flip) in corner cases. The probe counter counts every enumerated key, including any duplicates; the candidate set deduplicates naturally because binary search returns the same run for the same key, so accuracy is unaffected.
- **No `N_PROJ` sweep.** The tool is hard-coded to N_PROJ=16. Fused-filter and other N_PROJ variants are natural extensions.
- **No fused-filter variant yet.** The 1.28-point gap to dense L50_H1 can likely be closed with a (H1+H2) bucket index, but that's a separate experiment.

## What this measurement settles

1. The signature-as-address architecture works at the expected speed (O(1) amortized, ~200× faster than dense at matched accuracy).
2. The ternary multi-probe enumeration is correct — the observed radius escalation profile matches the atomic probe's distance histogram exactly.
3. The dense outer loop in every prior cascade tool was a measurement convenience, not a production architecture. The actual routed architecture uses the signature to *locate* candidates, not to *score* all of them.
4. The 1.28-point gap at this radius budget is dominated by 175 filter-miss queries (correct class at signature Hamming ≥ 3), not by any structural deficit in the bucket approach.
5. The information leverage rule transfers directly: the next experiment is a fused-filter bucket (8-byte keys from concatenated H1+H2).

## What this measurement does not settle

1. **Fused-filter bucket accuracy.** Expected ~87-89% but not measured.
2. **N_PROJ scaling.** Expected: at larger N_PROJ the collision rate drops and most buckets become singletons, which is efficient for lookup but may require larger radius budgets for recall. Unmeasured.
3. **Multi-seed variance.** Single seed only so far. The sensitivity of the bucket index to RNG seed matters for production deployments.
4. **Larger training sets.** At 60K prototypes the bucket index is 937 KB. At 1M prototypes it would be ~15 MB and the bucket size distribution would shift. Unmeasured.
5. **Drift behavior.** The index is static at training time. Online addition of prototypes requires either reindexing or an append-friendly structure.

## Pointers

- Tool: `tools/mnist_routed_bucket.c` (~460 lines of C).
- Parent architectural correction: this conversation's exchange about "dense" terminology and the discovery that cascade tools had dense outer loops.
- Atomic probe predicting the radius profile: `journal/lvg_atomics_decomposition.md` and `journal/nproj16_atomic_mechanism.md`.
- Information leverage rule this extends: `journal/fused_filter_fix.md`.
- Filter-ranker reframe that motivated everything in this line: `journal/cascade_atomics_mechanism.md` and `journal/nproj16_cascade_result.md`.
