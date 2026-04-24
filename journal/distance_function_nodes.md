---
date: 2026-04-21
scope: LMM cycle — can a better distance function on direct ternary signatures recover some of the 55pp CIFAR-10 oracle gap that uniform Hamming cannot?
phase: NODES
---

# NODES

## Discrete ideas

1. **Distance-function design is the right layer.** The `lr_scaffold_closeout.md` diagnosis established that uniform Hamming on direct-quantized signatures is the ceiling regardless of classifier architecture. Pair-IG is the only scorer that broke the Hamming ceiling (+1.95pp CIFAR-10). Further gains come from better distance, not better classifier.

2. **Hamming and SDOT inner-product are structurally different ternary distances.** Hamming costs {0, 1, 2} per dim based on trit pair state, penalizing `(q=+1, t=0)` with cost 1. SDOT inner-product scores `q × t ∈ {-1, 0, +1}`, assigning *zero* contribution to `(q=+1, t=0)`. Hamming penalizes "one-has-signal, other-doesn't"; SDOT doesn't. These rank candidates differently.

3. **`m4t_mtfp4_sdot_matmul_bt` already exists and is substrate-native.** 55–60 Gops/s on M3 (fastest primitive on the substrate). Operating on int8 ternary values is its design case. No invention required to use it as a similarity measure — just a consumer that computes `sdot(q, t)` instead of `popcount_dist(q, t, mask, bytes)`.

4. **Global per-dim weights (Family A).** One weight vector `w` ∈ ℝ^D or ℤ^D, applied as `Σ_d w[d] × cost(q[d], t[d])`. Derivation options: per-dim entropy over classes (single IG vector, not per-pair), per-dim variance across training set, inverse per-dim frequency. Integer-quantizable. Implementation: either bake weights into the trit encoding (tricky), or compute a weighted Hamming sum at scoring time (straightforward, per-dim scan, no popcount kernel shortcut).

5. **Block-level Hamming (Family B).** Split D dims into blocks of B trits. Per-block: count mismatches. Combine via an aggregator. Block size 4 aligns with the 2-bit-per-trit byte packing — one packed byte = 4 trits = one natural block. TBL dispatches over 4-trit patterns directly.

6. **Aggregator choice in Family B is load-bearing.**
    - **Sum over blocks** = equivalent to full Hamming, no gain.
    - **Max over blocks** = single worst block drives score — very sensitive to outlier blocks, may be too aggressive.
    - **Threshold-count** = each block either "matches" (cost ≤ T) or "doesn't"; score is count of matching blocks. Loses distance magnitude but captures locality.
    - **Weighted sum** (per-block weight) = generalizes global per-dim weights to per-block weights. Interpolates between A and B.
    - **Top-K block sum** = sum of the K smallest-cost blocks; ignores the K worst. Robust to block-level outliers.

7. **Pattern distance (Family C) is what SSTT does.** Learn a set of prototype trit patterns (per-class or global), score each block against the nearest prototype. Gets ~53% CIFAR-10 in the published baseline — that's the 7pp gap we'd like to close from current Glyph 46%. Copying this shape is sanctioned scaffolding (NORTH_STAR §4) but is not base-3-native discovery.

8. **Pair/triple correlation distance (Family D).** Score over multi-dim interactions rather than per-dim independence. Captures signal that A/B/C cannot. But: O(D²) pair features on D=9024 CIFAR-10 dims = 81M. Expensive; out of scope for this cycle, record as future.

9. **Hamming's symmetry is a design decision, not a law.** `d(q, t) = d(t, q)` because Hamming is XOR-based. A signed/directional distance could distinguish query-has-signal-target-doesn't from target-has-signal-query-doesn't. Might matter if the query's structural zeros are specifically where the target class has signal.

10. **SDOT similarity is NOT a distance metric.** It's a similarity (higher = better match). Inverting gives a non-negative similarity score. No triangle inequality. Shouldn't break direct_lsh's k-NN, which ranks candidates by score; doesn't need metric properties.

11. **CSA scored via SDOT instead of popcount_dist is an unmeasured variant.** My current csa_classifier uses popcount_dist (Hamming); SDOT scoring would be a drop-in swap. Should measure — another potential 1–2pp lurking.

12. **Pair-IG is the existing per-dim-per-classpair weighted Hamming.** It's the benchmark to beat. Global per-dim weights (Family A) should at least match pair-IG on the per-dim axis; they lose expressivity per-pair but gain efficiency.

13. **Substrate kernels available.** Current substrate surface:
    - `m4t_popcount_dist` (Hamming distance, masked, int32 return). Uniform, any sig_bytes.
    - `m4t_mtfp4_sdot_matmul_bt` (SDOT matmul). Inner-product similarity, batched.
    - `m4t_trit_mul` (elementwise trit product via TBL). Base3-native pairwise operation.
    - `m4t_trit_eq` (trit equality via TBL). Produces {0, +1} agreement pattern.
    - `m4t_trit_counts`/`m4t_trit_signed_sum`/`m4t_trit_sparsity` (masked-VCNT reducers). Sum signed/unsigned trits after any trit op.
    - The composite `trit_mul + signed_sum` = SDOT, but TBL-path instead of SDOT-path. Might be better on short vectors or when interleaved with masking.

14. **direct_lsh's resolver stage is where a distance function lives.** The tool calls `m4t_popcount_dist(q_sig, train_sig, mask, sig_bytes)` to score each candidate in the union. Swapping the kernel here is a minimal edit — same loop, different scoring call.

15. **Pair-IG's re-rank lives in the same slot.** The `pair_ig_correct` scoring loop at `tools/direct_lsh.c:736-748` computes a weighted Hamming: `dig = Σ_d pw[d] × (q[d] ≠ t[d])`. Per-dim weighted, per-class-pair lookup table. Swapping the weight source from pair-IG to global per-dim entropy is a one-function change to test Family A.

16. **The 55pp CIFAR-10 gap ceiling matters for scope.** If the gap is intrinsic (direct_lsh signature doesn't linearly separate classes at ≥50%), then A/B/C/D are all bounded below the oracle. Pre-experiment sanity check: brute-force run of direct_lsh Hamming k-NN with larger k or wider multi-probe to confirm we're at the Hamming ceiling, not under-resourced at filter-stage.

17. **Block-level TBL dispatch aligns with base-3-native thinking.** TBL is a three-way lookup already — it's how the substrate expresses "dispatch on trit pattern." Scoring a block against a pattern is literally `vqtbl1q_u8(pattern_table, block_bytes)` followed by summation. No invention required, just a consumer.

18. **SSTT's 53% is a target, not an aspiration.** Glyph's direct_lsh Selective is at 46.63%. The close-out's implicit target is closing the 7pp gap. Oracle at 99.99% says much more is theoretically there; practically, beating SSTT is a meaningful first falsification target for the thesis claim that base-3-native primitives outperform dense-shape baselines at the same task.

## Tensions

- **T1 (Node 2 vs Node 10).** SDOT inner-product is structurally different from Hamming. Is it *better* for classification? Not obvious — ignoring `(q=+1, t=0)` mismatches might help (noise tolerance) or hurt (actual signal that happens to align with zero trits). Only measurement resolves.

- **T2 (Node 4 vs Node 15 vs pair-IG precedent).** Global per-dim weights (Family A) have strictly less expressive power than per-class-pair weights (pair-IG). Can A beat pair-IG on CIFAR-10? Probably not. But A could be *comparable* at much lower build cost (one vector vs 45 matrices) — a production simplification rather than an accuracy improvement. Question: is the cycle about accuracy or about finding a simpler equally-good scorer?

- **T3 (Node 7 vs Node 17 vs NORTH_STAR §4).** Pattern distance (Family C) copies SSTT. The §4 scaffolding sanction applies — but the §3 rule "rage against the trodden" warns against importing comfortable base-2 shapes. TBL-based pattern dispatch is substrate-native; SSTT's exact implementation may not be. Is "TBL dispatch over 4-trit blocks" already the substrate's version of pattern distance, or a distinct thing?

- **T4 (Node 6 aggregator choice).** Within Family B, the aggregator (sum/max/threshold/top-K) determines the behavior. No prior strong reason to pick one. Risks: running all variants is a combinatorial sweep; picking one prematurely biases results. Need a principled reason to prefer one.

- **T5 (Node 16 vs ceiling assumption).** We assume the 55pp oracle gap has SOME recovery-able signal at the distance-function layer. If it doesn't — if the signature is genuinely too lossy at ~50% — then no Family A/B/C helps. Need a cheap pre-experiment to confirm scoring headroom is real before investing.

- **T6 (Node 3 vs Node 11 vs Node 14).** SDOT is already in the substrate. Why hasn't any consumer used it as a distance? Because `direct_lsh`'s architecture was built around Hamming from day one, and we never audited whether popcount_dist was the right kernel for the job. Node 11's proposed measurement (CSA with SDOT) is the cheapest test of this — one kernel swap away.

## Dependencies

- Any experiment depends on having: a distance-swap point in `direct_lsh.c` (exists at the resolver stage, line 619-622 and 739-748), or a new consumer tool (csa_classifier already provides one).
- Family A depends on: a weight-derivation routine (entropy, variance, or frequency over training). Integer-only implementation needed; fixed-point log if entropy.
- Family B depends on: block-packing layout (existing: one byte = 4 trits = one natural block), per-block score kernel (TBL lookup or popcount_dist with byte-length mask), aggregator function.
- SDOT swap for CSA/direct_lsh depends on: converting packed-trit signatures to int8 ternary cells at query time (cheap via `m4t_trit_unpack` or a packed-to-int8 helper not currently on the substrate surface). OR: precompute int8 copies of test signatures at load time.

## Open questions

- **Q1 (measurement-first):** does CSA with SDOT scoring differ meaningfully from CSA with popcount_dist scoring? Test on MNIST/Fashion/CIFAR to measure.
- **Q2 (weight derivation):** if Family A, what's the cheapest per-dim weight that beats uniform? Variance of trit value across training, or entropy of class-conditional trit distribution, or something simpler?
- **Q3 (block size):** if Family B, does 4-trit blocks win over 8-trit or 16-trit blocks on CIFAR-10? Pre-commit measurement on block_distance.c could surface the answer before committing to an aggregator.
- **Q4 (ceiling):** what is the Hamming k-NN accuracy on CIFAR-10 at k=full-training (no bucket filter, brute-force) with larger top-K vote (e.g., k=25, k=100)? This establishes the Hamming-distance ceiling vs classifier-architecture ceiling within the Hamming family.
- **Q5 (scope):** is the cycle's success criterion "beat pair-IG on CIFAR-10" (+1.95pp baseline), "match SSTT" (~53%), or "find a substrate-native primitive that reveals itself as the right shape regardless of number"?
