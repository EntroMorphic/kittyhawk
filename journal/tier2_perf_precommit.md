# Pre-Commit: Tier 2 NEON Underuse Remediation

Per the conversation analysis (2026-05-04). Three places where existing NEON hardware is underused; all three have straightforward vectorization paths.

## Per-item disposition

| ID | Item | Approach |
|----|------|----------|
| **T2-A** | `m4t_route_select` is scalar | Add NEON path using `vbslq_s32` with masks derived from packed-trit control. Per 4 cells (one MTFP19 block), one trit byte → 4 sign codes → mask vectors → cascaded bit-select. |
| **T2-B** | `m4t_route_confidence_weighted_dist` per-position scan with branches | Replace branchy per-position loop with **branchless per-byte bitwise math** + popcount. The sum-of-conf-bits-at-tracked-opposite-positions is `popcount(indicator & q_conf) + popcount(indicator & t_conf)` per byte. Hardware popcount makes this fast. (NEON-vector across multiple bytes is a follow-on; branchless scalar is the clear first win.) |
| **T2-C** | `m4t_mtfp_vec_accum_aligning` is fully scalar | Same-exponent case (no rescaling) reduces to `m4t_mtfp_vec_add_inplace`, which is already NEON. Refactor the same-exp branch to call it. The rescale branches still need scalar `m4t_pow3_round_div` per cell (NEON lacks integer division); vectorizing those needs magic-number-division and is **out of scope** for this cycle — documented as Tier 2.5 / Tier 3 follow-on. |

## Pre-committed gates

A Tier 2 PASS requires all of:

1. **G1 (select correctness):** `m4t_elemental_floor` test continues to PASS bit-equivalently.
2. **G2 (select speedup):** new NEON `select` is **≥2× faster** than scalar over 100K iterations on a 64-cell vector.
3. **G3 (confidence-weighted distance correctness):** `m4t_route` test continues to PASS bit-equivalently.
4. **G4 (conf-dist speedup):** new branchless scalar version is **≥2× faster** than current branchy scalar over 100K iterations at sig_dim=16.
5. **G5 (accum_aligning correctness):** `m4t_mtfp_accum_aligning` test continues to PASS bit-equivalently.
6. **G6 (accum_aligning same-exp path):** same-exp branch demonstrably uses NEON (call goes through `m4t_mtfp_vec_add_inplace`) — verifiable by code review. No timing gate (the speedup is structural, not measurable on the existing benchmark).
7. **G7 (no regression):** all 15 prior ctest binaries continue to PASS.

A WEAK is meeting all correctness gates (G1, G3, G5, G7) but missing one or both speedup gates (G2, G4) — code is correct but performance hasn't moved enough.

A FAIL is any correctness regression (G1, G3, G5 fail) — the new code is wrong.

## Out of scope (deliberately)

- **NEON vectorization of `m4t_pow3_round_div`** (the rescale paths in accum_aligning). NEON has no integer division; vectorizing requires magic-number-multiply-high techniques with a precomputed table per power of 3. Real engineering project. Documented as Tier 2.5 follow-on.
- **NEON vectorization of `m4t_route_confidence_weighted_dist`** beyond branchless scalar. The cross-byte parallelism is awkward because trit-pack and conf-pack alignments differ. Branchless scalar is the clear first win; further vectorization is a Tier 2.5 question if profile demands it.
- **Refactoring `m4t_mtfp_vec_accum_aligning` rescale branches.** The bulk-of-cycles in the rescale paths is the per-cell scalar division. Without addressing that (Tier 2.5), refactoring around it adds code complexity for no measurable benefit.

## Timing harness

A new bench binary `m4t_perf_tier2.c` measures G2, G4 with simple wall-clock comparisons (clock() + N iterations). No statistical confidence intervals at this scope — direction-of-effect is what matters. If results are within 2x of the gate but not over, document as WEAK.

## Order of execution

1. Write this doc (in progress).
2. Implement T2-A (NEON select), update tests.
3. Implement T2-B (branchless conf-dist), update tests.
4. Implement T2-C (accum_aligning same-exp refactor).
5. Build perf harness, measure speedups.
6. Verify all gates. Closeout.
