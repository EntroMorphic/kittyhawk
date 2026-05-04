---
status: P0 — owner directive 2026-05-04
authority: owner directive — close the three Tier 2 residuals
predecessor: journal/tier2_remediation_closeout.md
---

# Pre-Commit: Tier 2 Residual Concerns Closure

Three residuals from `tier2_remediation_closeout.md`, all P0. Pre-committed gates per project H4 discipline.

## Gates

**RES-1 (cache-defeat done right):** Replace the broken consecutive-runs verification with explicit cache-trashing between trials. PASS iff post-cache-trash measurement of select shows ≥30% slower per-iter than the steady-state measurement, demonstrating that cache effects ARE real and that the steady-state numbers were optimistic. (If post-trash and steady-state are within 30%, either the cache wasn't really being defeated by trash, or the working set is small enough to fit in L1 even after eviction — both informative.)

**RES-2 (adversarial distributions):** Subagent designs ≥2 adversarial input distributions per kernel, blind to the existing 3 distributions. Implement and run. PASS iff the adversarial distributions are tested and their results reported honestly. No threshold gate on what the results show — adversarial findings are data, not failures.

**RES-3 (LTO for fair-AND-accurate timing):** Add `-flto` to compiler flags. PASS iff (a) all 16 ctest binaries still PASS with LTO enabled; (b) all probe binaries still build clean and produce expected output; (c) perf re-measurement under LTO produces meaningful absolute timings (sub-µs per call, indicating inlining happened across lib boundary).

**RES-4 (no regression):** All 16 ctest binaries PASS through every step of the work.

## What this work does NOT do

- Does not change the production T2-A select kernel (NEON path is in production from prior cycle).
- Does not flip T2-B's production (branchy → branchless) — that's a separate owner decision based on R-G2 finding.
- Does not introduce a new substrate primitive.

## Order of execution

1. Spawn subagent for RES-2 adversarial distributions (background; it runs while I work).
2. Implement RES-1 cache-trashing in the bench harness.
3. Implement RES-3 LTO build flag, verify ctest + probes.
4. Integrate subagent's adversarial distributions when ready.
5. Run full perf suite under LTO with cache-trashing and adversarial distributions.
6. Closeout.
