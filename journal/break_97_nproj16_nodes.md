---
date: 2026-04-15
scope: LMM cycle — can local + global routing with Trit Lattice LSH reach 97% at N_PROJ=16?
phase: NODES
---

# NODES

## Discrete ideas

1. **Multi-table routed bucket LSH.** Build M independent bucket indexes, each keyed on a different 16-trit hash (different RNG seed). Every table is its own sorted `(sig_key, proto_idx)` structure, identical in layout to the Axis 5 bucket consumer. Query time: binary-search each table with multi-probe, union the candidate lists. Memory is M × 937 KB ≈ M × 1 MB.

2. **Local = per-table multi-probe, Global = cross-table candidate union.** This is the concrete mapping of the user's "local + global routing" phrase to a routing-only architecture. Every query lookup is still O(1) amortized per table; the global aspect is the union operation, which is O(M × avg_bucket_size) set-merge — cheap.

3. **Resolver vote-count scoring.** For each unique candidate in the union, count how many tables voted for it (i.e., how many tables placed it within the query's multi-probe neighborhood). Candidates with high counts are more likely to be correct. No distance arithmetic at the resolver stage — pure set-membership.

4. **Resolver summed-distance scoring.** Alternative: compute the summed popcount_dist across all M hashes for each unique candidate. This is equivalent to the dense-resolver family (Gq, L50_H1, etc.) but operating only on the pre-filtered candidate union — no full 60K scan.

5. **Resolver 1-NN-per-table then vote.** For each table independently, take the candidate with the smallest popcount_dist in that table's probe neighborhood. That's M candidate labels (one per table). Majority vote across the M labels, with ties broken by some rule.

6. **Adaptive per-query table subset.** Instead of probing all M tables every query, probe just enough tables to satisfy a confidence threshold (e.g., keep adding tables until the top-1 margin crosses a threshold). Reduces average-case cost; may help when many queries are easy.

7. **Aggressive multi-probe per table.** Instead of growing M, grow r inside each table. At r=3 the probe count per table is ~16 × 1.67² × 16 + ... (gets large). Trades table count for per-table depth. Less likely to help than multi-table because correlated neighbors don't add new information.

8. **Empty-query rescue via table union.** At M=1 our routed bucket has 175 empty queries (1.75%) at r≤2. At M tables with independent seeds, a query empty in table m may not be empty in table m' — empty rate should shrink multiplicatively as M grows. Predicted: empty rate at M=2 is roughly `0.0175² ≈ 0.0003`, effectively zero.

9. **Oracle upper bound from the union set itself.** For each query, compute the UNION of all M tables' multi-probe candidates. If the correct class is present in the union, an oracle resolver could pick it. `P(correct in union)` as a function of M is the tightest upper bound on what multi-table can achieve. Measure it cheaply before committing to resolver engineering.

10. **Information-theoretic model.** Each table is 16 trits = ~25 bits of signature. M tables = ~25M bits of independent signal (modulo correlation between random projections). Pure scaling curve suggests ~800 bits of signature reach 97%. So 32 tables may be a floor; 16 may be enough if the tables are nearly independent; 64 may be needed if they're correlated.

11. **Cost model.** At M tables, per-query wall time ≈ M × 10 μs (bucket consumer cost per table) + union-merge overhead. At M=32 that's ~320 μs/query — ~6× faster than the dense L50_H12 cascade (1950 μs) at a potentially much higher accuracy.

12. **Memory footprint.** M × 1 MB for the bucket indexes. At M=64 that's 64 MB — still fits comfortably in M-series unified memory but not negligible. At M=32 it's 32 MB, fine.

13. **Correlation control.** Random projections at the same density with different RNG seeds are approximately independent in expectation but have non-zero correlation on specific input distributions. Measuring per-table disagreement on a validation set would quantify effective independence. Worth a diagnostic pass.

14. **Hash-dimension ablation.** What if each table has a DIFFERENT N_PROJ? Table 1 at N_PROJ=16 (coarse, high recall), table M at N_PROJ=32 (finer, higher precision). This is the "adaptive recall" move. Out of scope for the 97%-at-N_PROJ=16 target but interesting for future work.

15. **Routing-context guard: no dense fallback.** Every component in this architecture must stay routed. No pixel access. No dense scans anywhere. The resolver is either set-membership vote-count or routed popcount_dist on the candidate union — never a dense operation.

## Tensions

**T1. Is the user's "local + global routing" precisely multi-table LSH?**
I am committing to Reading A (multi-table LSH) based on the RAW analysis. But the user might mean something different — hierarchical indexing, or a two-level routing structure, or something genuinely novel I haven't considered. The synthesize phase must name the commitment explicitly and request user confirmation before building.

**T2. Independence vs correlation of random tables.**
The theoretical best case is fully independent tables where recall scales as `1 − (1 − p)^M` for per-table recall p. The theoretical worst case is fully correlated tables where M tables = 1 table. Actual random ternary projections at the same density fall somewhere between. We don't know empirically where — and whether it's closer to independent or correlated decides whether M=16 or M=64 is needed for 97%.

**T3. Resolver choice is a separate experimental axis.**
Vote-count, summed-distance, per-table-1-NN-then-vote, and learned-weighting are all candidates. Each has different cost and different information-use profiles. If the resolver is the bottleneck, we may hit a ceiling at 93-94% regardless of M.

**T4. The 175 empty-queries floor is a double-edged measurement.**
Good: multi-table should crush it — queries empty in one table are probably not empty in all M. Bad: if our multi-table architecture still has empty queries at M=8 or M=16, it tells us the random-projection family has structural coverage gaps, which is hard to fix without changing hash construction.

**T5. Cost budget for 97%.**
The dense N_PROJ=512 scan reaches 97.06% at ~4000 μs/query (extrapolated from the scaling curve's throughput data). Our target is to match that accuracy at dramatically lower per-query cost. If multi-table at M=32 lands at 97% at 300-400 μs/query, that's a 10× cost reduction at matched accuracy — a real win. If it needs M=128 at 1200 μs/query, the win shrinks to 3×. Still good, but changes the architectural pitch.

**T6. Diminishing returns.**
Per the scaling curve, going from N_PROJ=128 (95.22%) to N_PROJ=512 (97.06%) requires 4× more signature budget for +1.84 accuracy points. The accuracy curve flattens. So crossing 97% at M=32 vs M=64 may be the difference between comfortable and strained. Need to measure the actual M-accuracy curve, not extrapolate from the dense scaling curve.

## Dependencies

- **Multi-table construction** (node 1) is cheap — reuse `build_signature_set` and the bucket index code. Each table adds ~10 lines of C.
- **Candidate union** (node 2) is a hash set of (proto_idx) with union-merge across tables. ~20 lines of C.
- **Resolver alternatives** (nodes 3, 4, 5) are each ~30 lines. Worth running all three as a resolver sweep within the same tool.
- **Oracle upper bound** (node 9) is a diagnostic pass — compute per-query union coverage before running any resolver. This gives the ceiling that bounds P2 and tells us if 97% is even reachable by the union set.
- **Correlation diagnostic** (node 13) is a secondary pass — compute cross-table disagreement on a validation subset. Not strictly necessary for the main experiment but informs the "why did M=X land where it did" question.
- **Independence from existing cascade tools.** The new tool should be a clean consumer built on top of `mnist_routed_bucket.c`'s patterns (sorted entries, lower_bound, ternary multi-probe, trit manipulation on packed codes). No dense cascade code should appear in the new tool.
