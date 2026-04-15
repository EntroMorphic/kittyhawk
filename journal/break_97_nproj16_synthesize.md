---
date: 2026-04-15
scope: LMM cycle — can local + global routing with Trit Lattice LSH reach 97% at N_PROJ=16?
phase: SYNTHESIZE
---

# SYNTHESIZE: the multi-table routed bucket experiment

## Thesis

**Multi-table routed bucket LSH at N_PROJ=16 — where "global routing" is the union-merge of M independent per-table neighborhoods — can break 97% accuracy on deskewed MNIST.** The minimum M needed is predicted to be ~32, based on an extrapolation from the pure-signature scaling curve (N_PROJ=512 single-hash reaches 97.06%). Per-query wall time at M=32 is predicted ~400 μs, which is ~10× faster than the dense N_PROJ=512 scan at matched accuracy. Zero dense scans anywhere in the architecture.

## Named commitment

I am committing to **Reading A** from RAW: "local + global routing" = classical multi-table LSH with per-table ternary multi-probe (local) and cross-table candidate union (global). If the user meant something structurally different — hierarchical indexing, learned hash selection, per-query adaptive routing over the M tables — the commitment should be rejected here and the cycle revisited. **Flag this as a go/no-go checkpoint before any code is written.**

## Architecture

```
TRAINING (one-time)
  for table m in 1..M:
      W_m = random_ternary_projection(seed_m, N_PROJ=16)
      tau_m = percentile(|W_m @ x_train_subset|, density=0.33)
      for each prototype i in 0..N_train:
          sig_m[i] = threshold_extract(W_m @ x_train[i], tau_m)
          append (sig_m[i], i) to entries_m[]
      qsort(entries_m, by sig_key)
  // M sorted bucket indexes, each ~937 KB at N_train=60000

QUERY (per test sample)
  candidate_votes = hash_map<proto_idx, int>()

  for table m in 1..M:
      q_sig_m = threshold_extract(W_m @ query, tau_m)
      for radius r = 0..2:
          for each probe_key in ternary_neighbors(q_sig_m, r):
              start = lower_bound(entries_m, probe_key)
              while entries_m[start].key == probe_key:
                  candidate_votes[entries_m[start].proto_idx] += 1
                  start += 1
          if |candidate_votes| >= MIN_UNION_SIZE: break

  // Resolver — three variants measured independently:
  //   VOTE: label = argmax over class of sum(votes for prototypes in class)
  //   SUM:  label = argmin over candidates of sum_m popcount_dist(q_sig_m, candidate_sig_m)
  //   1NN:  per-table top-1 candidate label; M labels; majority vote
  return label
```

**Key properties:**

- Every stage is a routing primitive. No dense scan at any point.
- Per-query cost is O(M × probe_budget) for the filter stage + O(|candidate_union|) for the resolver. Both are independent of N_train.
- Memory is M × ~937 KB for the bucket indexes plus a hash-map sized by candidate union.
- M is the primary experimental knob. Everything else is held constant across the M sweep.

## Experimental protocol

### Phase 1 — Build the tool

New consumer: `tools/mnist_routed_bucket_multi.c`, cleanly built on top of the patterns from `tools/mnist_routed_bucket.c`.

Components:
1. `build_signature_set(N_proj, seeds, density)` — already exists, reuse.
2. `build_bucket_index(entries, n_train)` — sort by key, already exists.
3. `enumerate_radius(query_sig, radius, cb)` — ternary multi-probe, already exists.
4. **New:** `union_candidates(tables, query_sigs, M, r_max)` — probe each table at radii 0..r_max, insert into a candidate vote map.
5. **New:** `resolver_vote(votes, labels)` — argmax class by vote total.
6. **New:** `resolver_sum(candidates, query_sigs, table_sigs, labels)` — argmin of summed popcount_dist across M hashes.
7. **New:** `resolver_per_table_majority(tables, query_sigs, labels)` — per-table 1-NN, majority vote of M labels.
8. Reporting harness — for each (M, resolver) pair, report accuracy, avg candidate union size, avg probes/query, wall time.

Estimated size: ~600 lines of C.

### Phase 2 — Oracle upper bound pass (cheap pre-check)

Before running any resolver, measure the oracle ceiling:

```
for M in {1, 2, 4, 8, 16, 32, 64}:
    count = 0
    for each test query:
        union = candidates from M tables at r<=2
        if y_test[query] in labels_of(union):
            count += 1
    oracle_ceiling[M] = count / n_test
```

**If `oracle_ceiling[32] < 0.97`, the experiment falsifies the hypothesis immediately.** The union set at M=32 simply doesn't contain the correct class often enough, and no resolver can rescue it. Report the negative result and close the cycle.

**If `oracle_ceiling[32] >= 0.97`, proceed to Phase 3.** The upper bound is in reach; the question becomes which resolver approaches it.

Oracle pass cost: cheap. One pass over 10K queries × 32 tables × ~216 probes/query = ~70M table lookups ≈ 5-10 seconds wall clock.

### Phase 3 — Resolver sweep at fixed M

For each M in {1, 2, 4, 8, 16, 32, 64} and each resolver in {vote, sum, per_table_majority}:
- Run on full 10K test set
- Report accuracy, escalation profile, wall clock

The resulting table is 7 × 3 = 21 cells. The goal is to find the (M, resolver) pair that first breaks 97%.

### Phase 4 — Diagnostic passes

If the main sweep succeeds, add:
- Cross-table agreement rate (how often tables agree on top-1 per query) — measures effective independence
- Union-size distribution (how large is the candidate set on average at each M)
- Per-class accuracy at the best (M, resolver) pair — does the ceiling hit specific digits harder
- Multi-probe radius ablation — does r=1 suffice at large M?

### Success criteria

1. **Primary:** some `(M, resolver)` pair reaches ≥97.00% accuracy on deskewed MNIST, single seed.
2. **Stretch:** the same pair runs in ≤500 μs/query.
3. **Cost ceiling:** any pair that reaches 97% but runs >1500 μs/query is considered a partial success — the target was broken but the pitch shifts.
4. **Honest negative:** if no pair reaches 97% at M≤64, report the oracle ceiling as the ground truth on what multi-table LSH with random projections can achieve at N_PROJ=16 and close the cycle.

### Failure modes to watch

- **Saturation below 97%.** Oracle ceiling climbs but stalls below target. Indicates projection family has structural coverage gaps. Fix: density variation, seed-dependent tau, or different hash construction.
- **Resolver gap.** Oracle ceiling ≥97% but every resolver underperforms. Indicates the candidate union contains the correct class but no observable signal separates it from the wrong candidates. Closely related to the Axis 4c observability ceiling.
- **Cost blowup.** M=32 but per-query cost exceeds 1 ms. Probably union-merge overhead is dominant. Fix: smaller M + larger per-table radius, or better union data structure.

### Cost model (realistic estimates)

At M=32, per-query work:

- 32 table lookups × ~216 probes/table × binary search (16 comparisons) = ~110K key comparisons
- Union merge across 32 tables: ~4000 unique candidates into a hash map with ~8K bucket insertions = ~20 μs
- Resolver vote-count: ~4000 candidate lookups in label table = ~10 μs
- **Total wall time per query estimate: 300-500 μs**

At M=64:

- ~220K key comparisons + larger union (~6K unique candidates)
- Total wall time: 600-1000 μs

Both are well under the dense N_PROJ=512 estimate of ~4000 μs.

## Predicted M-accuracy curve

| M | predicted acc | rationale |
|---|---|---|
| 1 | 82.58% | measured (Axis 5 bucket consumer) |
| 2 | 88-90% | empty-query rescue + union rescues radius-2 misses |
| 4 | 92-93% | approaches N_PROJ=64 pure single-hash territory |
| 8 | 94-95% | approaches N_PROJ=128 pure single-hash |
| 16 | 95.5-96.5% | approaches N_PROJ=256 pure single-hash |
| **32** | **96.8-97.3%** | approaches N_PROJ=512 pure single-hash — **target crossing** |
| 64 | 97.3-97.7% | approaches N_PROJ=1024 pure single-hash |

Crossing 97% is predicted between M=16 and M=32. The actual crossing depends on independence between random ternary projections — which is what Phase 2's oracle pass measures.

## What this cycle *is*

A direct extension of Axis 5 (signature-as-address) to a multi-hash routed architecture. The local half is the per-table bucket index already built. The global half is the union-merge across M tables. Nothing else changes. This is the thinnest possible elaboration of the Axis 5 architecture that can plausibly reach 97% at N_PROJ=16.

## What this cycle *is not*

- Not a proposal to introduce dense paths. Every operation is routed.
- Not a proposal for learned hashes, supervised hash selection, or trained projections. Pure random ternary at density 0.33, only the seed varies per table.
- Not a meta-router. The global routing is the union-merge — a set operator, not a decision router.
- Not dependent on N_train scaling. The architecture is O(1) amortized per query regardless of N_train.

## Next action

**Go/no-go checkpoint:** confirm Reading A (multi-table LSH with union-merge) matches the user's "local + global routing" framing. If yes, build `tools/mnist_routed_bucket_multi.c` with the oracle pass (Phase 2) gating the resolver sweep (Phase 3). If no, request the user name the intended architecture and revise the cycle.

Budget: ~600 lines of C, one build, one oracle pass (~10 sec), one resolver sweep (~30 sec × 21 cells ≈ 10 min). The full experiment runs in under an hour of wall clock.
