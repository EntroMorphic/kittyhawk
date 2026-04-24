---
date: 2026-04-21
scope: LMM cycle — distance-function design for direct ternary signatures
phase: SYNTHESIZE
---

# SYNTHESIZE: SDOT-as-distance and threshold-count block distance

## The reframe

The seed question "how do we improve the distance function?" assumed the thing to fix was *uniformity* (like pair-IG does) or the lack of *locality* (like SSTT-style block distance). The reflection surfaces a third axis that had been invisible: **how the distance handles the structural zero**. Ternary Hamming assigns cost 1 to the `(zero, non-zero)` trit pair — treating an absent signal as a near-miss. SDOT inner-product assigns 0 to the same pair — treating the zero as "this dim isn't active, don't count it." That semantic distinction is the NORTH_STAR geometric-fullness claim applied to distances, and the substrate already has the SDOT kernel running at 55–60 Gops/s with no consumer using it for distance.

The cycle's primary falsifiable bet: **SDOT inner-product as a scoring kernel produces better class rankings on direct ternary signatures than Hamming popcount**, because it doesn't over-penalize structural zeros. If true, we gain a base-3-native distance primitive with zero invention; if false, we've learned that Hamming's symmetry + uniform zero-handling is load-bearing on these signatures and NORTH_STAR-style reasoning doesn't predict the empirical answer here.

## Decision

**Run three staged experiments, E1 first (cheapest), E2 in parallel, E3 conditional:**

**E1: SDOT-as-distance swap in direct_lsh.** Replace `m4t_popcount_dist(q, t, mask, bytes)` with an SDOT-based score at the resolver stage. Score → argmin by `-sdot(q, t)` (higher sdot = closer → smaller negative distance). Three datasets, direct comparison to existing Hamming baseline. Expected runtime: implement + measure ≈ 2–3 hours end to end. Decides T1 and T6.

**E2: global per-dim-weighted Hamming.** Derive a single weight vector from per-dim entropy or variance across training classes. Apply weighted Hamming as a new scorer in `direct_lsh` (same slot as pair-IG re-rank). Three datasets. Decides T2: is per-dim weighting alone enough, or is per-class-pair (pair-IG) irreducibly important?

**E3 (conditional on E1 or E2 signal, skip otherwise): 4-trit block distance with threshold-count aggregator.** Block size 4 (one byte), aggregator = count of blocks whose per-block Hamming is below a per-dataset-calibrated threshold T. Implementation: composition of existing `m4t_trit_eq` + block-wise VCNT, or a new kernel if measurement shows the decomposition is slow. One new scorer path in `direct_lsh`. Decides T3 and T4.

**Out of scope this cycle:**
- Pattern distance (Family C) with learned prototype codebooks. That's a whole separate LMM cycle.
- Pair/triple correlation distance (Family D). O(D²) feature cost, deferred.

## Success criteria

**Cycle-level:**
- [ ] E1 lands a measured SDOT-vs-Hamming delta on three datasets. Decides the structural-zero-handling question empirically. Success = the delta is named and measured; direction of the delta is not a pass/fail condition.
- [ ] E2 lands a per-dim-weighted Hamming number comparable to pair-IG's +1.95pp on CIFAR-10 (or meaningfully below it, which is also a useful finding). Success = per-dim-only contribution is characterized.
- [ ] If E1 or E2 shows ≥0.5pp improvement over Hamming on any dataset, run E3. If neither does, write the cycle close-out with the "distance function has smaller headroom than oracle suggested" finding.

**Primitive-level (what counts as thesis-relevant):**
- SDOT-as-distance wins on at least one dataset → base-3-native distance primitive validated, direct_lsh gets a new scorer option.
- Weighted-Hamming wins on at least one dataset → per-dim-weighting architecture is established as a cheaper pair-IG substitute.
- Block distance beats both → pattern-level locality exists in direct signatures and is unlocked via TBL-native aggregation.

## Implementation specification

### E1: SDOT-as-distance

**Where:** `tools/direct_lsh.c` resolver. Current popcount_dist loop at lines 619-622 (Stage 1 scoring) and 739-748 (pair-IG re-rank). Add a resolver variant selected via `--resolver_distance sdot`.

**How:**
1. At startup, after signatures are built, additionally allocate and populate int8 ternary buffers parallel to the packed signatures. `glyph_sig_quantize` already produces the trit value at sign time; keep an int8 copy alongside the 2-bit packed copy. Memory cost: `n_train * total_dim` int8 = ~600MB for CIFAR-10 with gradients. If that's too much, unpack on the fly inside the scoring loop via `m4t_unpack_trits_1d`.
2. In the scoring loop, call the new kernel: `sdot_score(q_int8, t_int8, total_dim)`. Implementation:
    ```c
    int32_t sdot_score(const int8_t* a, const int8_t* b, int n) {
        // Equivalent to: Σ_d a[d] * b[d], using vdotq_s32.
        // a, b ∈ {-1, 0, +1}^n
    }
    ```
    Use `m4t_mtfp4_sdot_matmul_bt(Y=&score, X=a, W=b, M=1, K=n, N=1)` for a single-output call, or just call the underlying vdotq_s32 pattern directly.
3. Argmin selection becomes argmax (higher sdot = better).
4. Report alongside existing `direct_lsh` metrics.

**Test:** at N=1 and trivial inputs, verify sdot_score matches hand-computed values. Bit-identity isn't required against Hamming (they're different metrics); only against the mathematical definition of sdot.

### E2: global per-dim-weighted Hamming

**Where:** same resolver slot. New function `weighted_hamming_score(q, t, pw, total_dim)` that computes `Σ_d pw[d] × (q[d] ≠ t[d])`, bytes-wise via trit unpacking.

**Weight derivation:** at startup, compute per-dim entropy over training trit distributions (marginalized over class). Or per-dim variance of trit value × class agreement. Or simpler: per-dim frequency of non-zero trits (common dims get low weight; rare-active dims get high weight). Pick the cheapest to implement; all three are integer-derivable from existing training-set counts. Preferred default: **per-dim marginal entropy × 16 / max_entropy**, integer-quantized to [1, 16] using the same LUT as pair-IG.

**Integration:** add CLI flag `--resolver_distance weighted_hamming` that selects this scorer. Reuse the integer-log table from pair-IG (already built).

### E3 (conditional): 4-trit block distance, threshold-count

**Where:** same resolver slot. New function `block_threshold_score(q, t, sig_bytes, T)` that for each packed byte computes per-byte Hamming, compares to T, accumulates count of matches.

**Kernel:** single byte-wise loop using `__builtin_popcount` on `(q_byte ^ t_byte) & mask_byte`, compared to T. NEON kernel possible but not required for first pass.

**Threshold calibration:** per-dataset T, swept in {0, 1, 2, 3, 4} (per-4-trit-block max cost is 2×4=8, but most dims' ternary-Hamming cost is <2 in practice). Pick T that maximizes accuracy on a held-out sample of training.

**Aggregator alternatives (if threshold-count disappoints):** weighted-block-sum (per-block weight learned from per-block entropy); top-K blocks (keep K smallest costs, sum). Don't implement these until threshold-count is measured.

## Handling the major tensions

- **T1 (SDOT vs Hamming):** E1 directly measures. Three datasets, same harness.
- **T2 (global vs per-pair weights):** E2 measures per-dim-only against pair-IG baseline.
- **T3 (substrate-native vs SSTT-copy):** E3 uses TBL-native block dispatch + VCNT aggregator, which is the substrate's shape. Declines SSTT's learned codebook path.
- **T4 (aggregator choice):** default to threshold-count because it's the one VCNT directly expresses. Swap only if measured.
- **T5 (ceiling assumption):** E1 and E2 are both cheap sanity checks. If both fail to improve, the reframe is wrong and we learn it cheaply.
- **T6 (SDOT path-dependence):** E1 directly tests; if it wins, the path dependence is exposed and resolved by adding a new scorer.

## Quality check

- **Could someone else execute this?** Yes. Each E has a named code-location, a named kernel, a named CLI flag, a named measurement. Metrics tie back to existing `direct_lsh` outputs.
- **Does it address all major tensions?** Six tensions mapped to experiments; three deferred with explicit reasons.
- **Is it simpler than starting point?** RAW had four distance families and anxieties about reaching for SSTT; synthesis reduces to one primary bet (SDOT), one cheap calibrator (weighted Hamming), one conditional substrate-native extension (block threshold).
- **Surprised?** Yes — I entered the cycle expecting "pick a better aggregator over blocks" and left with "SDOT's structural-zero semantics is the actual base-3-native distance and the substrate has been waiting for a consumer to use it." Same pattern as lr_scaffold — the right move was latent in the substrate, not pattern-matched from base-2 literature.

## Immediate next actions

1. **E1 first:** implement SDOT-as-distance in `direct_lsh` behind `--resolver_distance sdot`. Measure three datasets against Hamming baseline. Expected cost: ~2 hours.
2. **E2 in parallel:** implement global per-dim-weighted Hamming behind `--resolver_distance weighted_hamming`. Measure three datasets. Expected cost: ~1 hour (reuses pair-IG LUT infrastructure).
3. **Report:** e1_results.md + e2_results.md with metrics and trajectory observations.
4. **Gate:** if either E1 or E2 shows ≥0.5pp improvement on any dataset, implement E3. If neither, write cycle close-out with the "distance-function headroom smaller than oracle implied" finding.

## What this cycle produces regardless of outcome

The LMM pass itself is the deliverable: a named primitive (SDOT-as-distance) with a clear falsifiable test, a realistic calibration experiment (weighted Hamming vs pair-IG), and a substrate-native extension path (TBL block threshold) that's pre-specified. Whether the measurements land pro or con, the reframe on how ternary distance should handle the structural zero survives the cycle as substrate-level design learning.
