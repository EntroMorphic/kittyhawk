---
status: P0 — owner directive 2026-05-04 (close the V3 honest residuals; no scope-hiding)
authority: owner directive — "Identify and remediate the threats inherent in the -UNDEBUG residual"
predecessor: journal/tier2_residuals_v3_closeout.md
---

# Pre-Commit: -UNDEBUG Residual Threats

## The threats inherent in the V3 -UNDEBUG residual

V3's `gesh_test_undebug()` applies `-UNDEBUG` to test executables only. **Substrate libraries (libm4t.a, libgesh.a) stay compiled with `-DNDEBUG` because they're shipped artifacts.** This creates a real gap:

**Threat T1 (substrate-internal asserts silenced in tests):** when a test calls a substrate function with arguments that violate the function's preconditions (e.g., `m4t_route_topk_abs(decisions, scores, T=200, k=10)` — T exceeds `M4T_ROUTE_MAX_T = 64`), the substrate's `assert(T <= M4T_ROUTE_MAX_T)` is a no-op in the shipped libm4t.a. The function continues with the bad input, possibly producing garbage output or memory corruption. The test sees the garbage, may not detect it, reports PASS.

**Concrete failure modes:**
- `m4t_mtfp4_sdot_matmul_bt`: `assert(K <= M4T_SDOT_K_MAX_EXACT)` — huge K silently produces wrong output
- `m4t_route_threshold_extract`: `assert(tau >= 0)` — negative tau silently mis-classifies
- `m4t_route_topk_abs`: `assert(T <= M4T_ROUTE_MAX_T)` — overrun in fixed bitmask
- `m4t_mtfp_vec_accum_aligning`: `assert(running != addend)` — aliasing UB silently
- ~50 more across m4t/src and gesh/src

**Threat T2 (verification-by-grep is incomplete):** wider grep used standard regex; could miss multi-line asserts, asserts inside macros, or asserts using non-standard names (`debug_assert`, `dbg_check`, etc.).

**Threat T3 (principled bound looseness):** the `dim * SCALE/10` mean-drift bound catches order-of-magnitude bugs but not 2-3× regressions. A real precision bug in normalize_one could pass the test silently.

**Threat T4 (LTO inlining opacity):** verified function is called externally (`bl _image_canon_load_mnist` × 4); didn't verify what LTO actually does cross-TU. Could be doing nothing useful, could be doing wrong things.

This pre-commit addresses T1 (the deepest threat) directly. T2-T4 get addressed where straightforward.

## Disposition

| ID | Threat | Disposition |
|----|--------|-------------|
| **T1** | Substrate-internal asserts NDEBUG-disabled in tests | FIX — build libm4t and libgesh in TWO variants: production (NDEBUG, `m4t`/`gesh`) and test (no NDEBUG, `m4t_test`/`gesh_test`). All test executables link against the test variants. Substrate asserts then fire when tests trigger them. |
| **T2** | Verification-by-grep incomplete | FIX — verify by SYMBOL inspection of test binaries: `nm` for `__assert_rtn` (macOS) presence indicates substrate asserts are real in the test binary. |
| **T3** | Principled bound looseness | FIX — tighten where defensible. Add a SECOND check (sum/dim is small absolute, e.g., `|sum/dim| < dim`) that catches subtle drift. |
| **T4** | LTO inlining opacity | VERIFY — write a benchmark-style measurement that compares LTO-inlined vs non-inlined timings to characterize what LTO actually does. |

## Pre-committed gates

A V4 PASS requires all of:

1. **V4-G1 (T1 closed by build-config split):** libm4t and libgesh have test variants compiled without NDEBUG. All 15 ctest binaries link against the test variants. All 15 still PASS. Symbol inspection confirms test binaries contain `__assert_rtn` calls.

2. **V4-G2 (T1 closed by deliberate-trigger test):** add a unit test that calls a substrate function with arguments designed to trip an assert. Test FORKs a child process, calls the bad function, expects the child to abort with SIGABRT. PASSes iff the abort happens (asserts are real in test build). If asserts are still no-ops, the test FAILS.

3. **V4-G3 (T2 closed by symbol verification):** `nm` on at least 3 test binaries confirms `__assert_rtn` symbols present.

4. **V4-G4 (T3 closed):** mean-drift test gets a second tighter check on the absolute mean (`|sum/dim| < dim` for typical normalize). Existing loose bound stays as catastrophic-bug catcher.

5. **V4-G5 (T4 measured):** bench harness reports LTO-vs-no-LTO per-call timing for the existing select kernel. If LTO has no measurable benefit, document. If it does, document the magnitude.

6. **V4-G6 (no regression):** all 15 ctest binaries PASS through every step.

## Risks of the fix itself

- **Risk A: doubling library compilation increases build time.** Acceptable; substrate is small.
- **Risk B: linking against the wrong variant accidentally.** Mitigation: name the test variants distinctly (`m4t_test`, `gesh_test`), make production targets unmistakable.
- **Risk C: a substrate assert that fires during normal test execution.** If existing tests trigger assertions that were always silently disabled, those are real bugs we surface. That's the point — but fixes might be needed.
- **Risk D: the deliberate-abort test (V4-G2) might be flaky on different OSes.** macOS posix fork/wait is reliable; CI on other platforms could differ. Document the macOS-only assumption.

## Order of execution

1. Write this doc (in progress).
2. Add `m4t_test`, `gesh_test`, `gesh_bench_test`, `gesh_image_canon_test` library targets.
3. Update test executables to link against the test variants.
4. Build; observe any test failures from newly-firing substrate asserts. Fix any real bugs surfaced.
5. Add deliberate-abort meta-test for V4-G2.
6. Tighten mean-drift bound (V4-G4).
7. LTO measurement (V4-G5).
8. Closeout.
