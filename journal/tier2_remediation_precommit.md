# Pre-Commit: Tier 2 Red-Team Remediation

Per `journal/tier2_perf_redteam.md`. 13 findings, 1 critical, 3 high, 6 medium, 3 low.

## Per-finding disposition

| ID | Severity | Disposition |
|----|----------|-------------|
| **C1** | Critical | FIX — fair-comparison re-measurement of T2-A. Put scalar reference IN THE LIB (`m4t_route_select_scalar_ref`); bench times both via identical lib boundary. |
| **H1** | High | FIX — property test exercising the `flags == NULL` same-exp branch of `m4t_mtfp_vec_accum_aligning`. |
| **H2** | High | SUBSUMED by H1. |
| **H3** | High | FIX — amend Tier 2 closeout's verdict for T2-A based on C1's fair re-measurement. |
| **M1** | Medium | FIX — switch perf timing from `clock()` to `clock_gettime(CLOCK_MONOTONIC)` for ns resolution. |
| **M2** | Medium | FIX — perf harness uses 3 data distributions (seeded random; structured pattern; sparse-zero) per measurement. |
| **M3** | Medium | FIX — use a pool of N=8 different data arrays per measurement, cycled per iteration to defeat steady-state cache + branch prediction. |
| **M4** | Medium | FIX — add `m4t_route_confidence_weighted_dist_branchless` to lib for fair comparison; re-measure paired with C1. |
| **M5** | Medium | FIX — this pre-commit doc explicitly red-teams the gate design before code (you're reading it). |
| **M6** | Medium-low | FIX — trim the 13-line inline comment in `m4t_route.c` to 2-3 lines pointing at the journal. |
| **L1** | Low | FIX — CMakeLists comment pointing at the perf harness. |
| **L2** | Low | FIX — rename `test_m4t_tier2_perf.c` → `bench_m4t_tier2_perf.c`. |
| **L3** | Low | FIX — closeout's "what stays open" gets per-item cost estimates. |

**13/13 disposed.**

## Pre-committed gates (with harness design red-teamed)

A Tier 2 remediation **PASS** requires all of:

1. **R-G1 (fair-measurement T2-A):** scalar reference and NEON, both called through lib boundary via the new `m4t_route_select_scalar_ref` function. Speedup ratio computed on identical machinery. PASS if NEON ≥1.5× faster than scalar via fair comparison (lower bar than the prior unfair 2.0× because real algorithmic gain net of equal call overhead).

2. **R-G2 (fair-measurement T2-B):** branchless and branchy, both via lib boundary. Diagnostic only — no PASS/FAIL gate. Whatever the fair comparison shows is the data; substrate keeps branchy as the production version per the prior revert.

3. **R-G3 (T2-C path exercised):** small property test that calls `m4t_mtfp_vec_accum_aligning` with `flags == NULL` and same-exp inputs, verifies bit-equivalence to int64 reference. PASS if test PASSes.

4. **R-G4 (data-distribution coverage):** perf measurements run under 3 data patterns. PASS if direction-of-effect is consistent across all 3.

5. **R-G5 (cache-defeated measurement):** perf harness cycles through pool of 8 arrays per iteration. PASS if measured timings change ≤30% between unprimed (first 100 iter) and primed (steady-state) runs — i.e., we're not measuring pure steady-state.

6. **R-G6 (no regression):** all 16 ctest binaries still PASS (15 prior + 1 new for H1).

7. **R-G7 (gate-design red-team applied retroactively):** this pre-commit doc must list specific risks of the gate design itself BEFORE measurement (below).

## Gate-design red-team (R-G7 satisfied here, per M5)

Risks of THIS remediation's gate design, surfaced before code:

- **Risk A: M3's "cycle through pool of 8" might still be steady-state in disguise.** 8 arrays = small enough to fit in L1 cache. If the pool is loaded once and accessed repeatedly, branch predictor still learns the cycle. **Mitigation:** make the cycle index pseudo-random rather than sequential; use a larger pool (32+) if measurements are still suspiciously fast.

- **Risk B: clock_gettime(CLOCK_MONOTONIC) measures wall time, not CPU time.** Other processes can preempt; measurements include OS scheduling jitter. **Mitigation:** run multiple trials per measurement, report median + min. Don't fixate on a single number.

- **Risk C: the "fair-via-lib-boundary" approach assumes lib-call overhead is identical for both functions.** If one function has more arguments or different stack-frame size, the call overhead differs. `m4t_route_select_scalar_ref` will have the EXACT same signature as `m4t_route_select` to mitigate. **Same applies to T2-B.**

- **Risk D: R-G1's threshold of ≥1.5× is itself somewhat arbitrary.** Lower than the original 2.0× to allow for the call-overhead handicap on both sides. Set in advance to avoid post-hoc adjustment. If NEON is genuinely better than scalar, 1.5× over a fair-comparison baseline should be readily achievable (NEON does 4 cells per cycle vs scalar's 1).

- **Risk E: R-G3's test might not actually exercise the NEW path if `vec_add_inplace` was already exercised by `test_m4t_mtfp`.** The new path adds NO new arithmetic — just reroutes. The test must verify the WIRING (accum_aligning with flags=NULL produces same result as direct vec_add_inplace), not the arithmetic itself.

## Out of scope (explicitly)

- **Magic-number-multiply vectorization of `m4t_pow3_round_div`.** Tier 2.5; would unlock NEON for accum_aligning rescale branches. Real engineering project; not addressed here.
- **NEON-vector across multiple bytes for confidence_weighted_dist.** Even if R-G2's fair measurement shows branchless is faster than branchy, the further step (multi-byte SIMD) is its own cycle.
- **Adversarial probe data for the speedup gates.** R-G4 covers 3 distributions; truly adversarial (worst-case branch-prediction patterns) is a follow-on.

## Order of execution

1. Write this doc (in progress).
2. Add `m4t_route_select_scalar_ref` and `m4t_route_confidence_weighted_dist_branchless` to lib (m4t_route.{h,c}).
3. Trim T2-B's revert comment in m4t_route.c (M6).
4. Rewrite the perf harness with: clock_gettime, 3 data distributions, pool of 8 arrays, fair lib-boundary calls (M1, M2, M3, C1, M4).
5. Add R-G3 property test for T2-C's flags=NULL path (H1).
6. Rename test_m4t_tier2_perf.c → bench_m4t_tier2_perf.c (L2); update CMakeLists with comment (L1).
7. Build, run perf, run all tests.
8. Closeout doc with verdicts and L3-style prioritized open items.
