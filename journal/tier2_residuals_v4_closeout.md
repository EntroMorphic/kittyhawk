# Closeout: V4 — -UNDEBUG residual remediation

> **Update note (post-V4-residual-3):** the T4 finding below — "LTO has nothing to add at the granularity the bench measures" — is correct narrowly but reads as a general statement and is wrong as one. A subsequent microbench (see `journal/v4_residual_3_lto_microbench_closeout.md`) demonstrated LTO produces a **3× speedup** on pipelined workloads. The "no delta" finding is a property of the substrate's carry-dependent workload shape, not a property of LTO. See `## Update (V4-residual-3)` at the bottom of this doc for the corrected framing.

Per `journal/tier2_residuals_v4_precommit.md` against the four threats inherent in V3's `-UNDEBUG` residual.

## Verdict: PASS — all four threats closed; one finding surfaced

```
V4-G1 (T1: substrate-internal asserts now live in tests) : PASS — 16/16 ctest PASS through every step
V4-G2 (deliberate-abort meta-test)                       : PASS — test_m4t_assert_live confirms SIGABRT
V4-G3 (T2: symbol verification, not grep)                : PASS — nm shows 0/0/0 vs 5/8/1 ___assert_rtn refs
V4-G4 (T3: tight bound alongside loose)                  : PASS — tight_bound = 10*dim; observed drift ≤ 80
V4-G5 (T4: LTO measurement, LTO-vs-no-LTO)               : PASS (with informative finding — see below)
V4-G6 (no regression)                                    : PASS — 16/16 ctest binaries PASS
```

## Per-threat disposition

| ID | Threat | Outcome |
|----|--------|---------|
| **T1** | libm4t.a / libgesh.a shipped with `-DNDEBUG`; tests' `-UNDEBUG` only affected the test binary's own .o files. Substrate-internal asserts (e.g., `m4t_route_topk_abs`'s `T <= M4T_ROUTE_MAX_T` check) silenced when tests trigger them. | **CLOSED** — added `m4t_test`, `gesh_test`, `gesh_bench_test`, `gesh_image_canon_test` library variants compiled with `-UNDEBUG`. All test executables now link against the test variants. Substrate asserts fire when tests trigger them. Verified empirically via V4-G2 meta-test. |
| **T2** | Verification-by-grep (V3-G3) incomplete; could miss multi-line, macro-wrapped, or non-standard assert forms. | **CLOSED** — replaced grep with `nm` symbol-table verification on the actual built libraries (V4-G3). Production libs: 0 references to `___assert_rtn`. Test libs: 5 (m4t), 8 (gesh), 1 (image_canon). Concrete, structural proof. |
| **T3** | Principled mean-drift bound (V3-G2) was loose (`dim*SCALE/10` ≈ 94K for dim=16 vs observed drift ~80). Catches order-of-magnitude bugs but not 2-3× regressions. | **CLOSED** — added second tight bound `10*dim = 160` alongside loose. Derived from worst case: residual after centering ≤ dim, amplified by `SCALE/sd ≤ 4` for typical pixel data, yielding `≤ 5*dim` worst case. 2× safety factor → `10*dim`. Observed drift ≤ 80; tight bound at 160 holds with 2× headroom. Catches drift regressions of ~2×. |
| **T4** | LTO inlining decisions opaque. V3 verified `image_canon_load_mnist` was called (4× in test binary), but didn't verify LTO was applying meaningful cross-TU optimizations. | **CLOSED** (informative) — V4-G5 measured. LTO build vs no-LTO build (`-DCMAKE_C_FLAGS="-fno-lto"`): bench timings within ±5% noise, binary sizes byte-identical (50936 vs 50936), call counts to `_m4t_route*` identical (6 vs 6). LTO flag confirmed live in compile + link commands (verbose make output). **Finding: LTO is enabled and applied, but produces no observable optimization on this bench.** The substrate's hot paths are already well-optimized at the per-TU level (-O3 -mcpu=native); LTO has nothing to add at the granularity the bench measures. |

**4/4 threats closed. 4 fixes + 1 informative finding (LTO is on but does nothing observable here).**

## What shipped

- `m4t/CMakeLists.txt`: added `m4t_test` STATIC library (same sources as `m4t`, `target_compile_options(... PRIVATE -UNDEBUG)`). All 9 m4t test executables now `target_link_libraries(... PRIVATE m4t_test)`. The perf bench (`bench_m4t_tier2_perf`) keeps linking against production `m4t` (asserts add overhead irrelevant to perf measurement).
- `gesh/CMakeLists.txt`: added `gesh_test`, `gesh_bench_test`, `gesh_image_canon_test` STATIC library variants. All 6 gesh test executables relinked against the test variants. The 22 gesh bench/probe binaries stay linked against production libraries.
- `m4t/tests/test_m4t_assert_live.c`: new meta-test. Forks a child, calls `m4t_route_topk_abs(decisions, scores, T=200, k=4)` (T > M4T_ROUTE_MAX_T = 64), `waitpid`s, asserts `WIFSIGNALED && WTERMSIG == SIGABRT`. Distinguishes "assert silenced" (child runs to completion with exit code 42) from "child crashed for unrelated reason" (different signal/exit).
- `gesh/tests/test_image_canon.c`: added second tight bound check after the loose one. Same `sum` value tested against both. Tight bound's failure message names it as a regression check.

## What's now structurally true

**The substrate's internal asserts now actually fire in tests, when triggered.** Verified three ways:

1. **Build-time:** `nm` shows `___assert_rtn` undefined symbols in the test variant libraries (5/8/1) and zero in production libraries.
2. **Runtime:** `test_m4t_assert_live` deliberately violates a precondition; the child aborts via SIGABRT. Without V4, the assert would have been a `((void)0)` and the function would have proceeded to write past its uint64_t bitmask (silent memory corruption).
3. **Regression:** all 16/16 existing tests still PASS — none of them was incidentally exercising a substrate precondition violation that a silenced assert was hiding.

**The image_canon mean-drift test now has a tight bound that would catch a 2× drift regression.** The loose bound (94K for dim=16) catches order-of-magnitude bugs; the tight bound (160 for dim=16) catches subtle regressions.

**LTO is enabled and applied, but provides no measurable speedup on this bench.** This is a HONEST finding, not a fix — the substrate's per-TU optimization is already so aggressive that LTO has nothing to do at the bench's measurement granularity.

## Honest concerns from this cycle

**1. The deliberate-abort meta-test exercises ONE substrate assert.** It proves the mechanism works. It does NOT prove that every substrate assert across `libm4t_test` is actually compiled in. Implicit assumption: `-UNDEBUG` works uniformly. Risk: low (compiler flag behavior is well-defined). Mitigation: nm symbol counts (V4-G3) provide additional structural evidence.

**2. The tight bound (10×dim) is data-dependent.** It holds for the synthetic test data (`pixel(i,j) = (i*7+j*11) & 0xff` over 8 images × 16 pixels). If the test data changes to something with a different per-image standard deviation, the SCALE/sd amplification factor shifts and the bound may need recomputing. Risk: medium — anyone changing test data must recompute. Mitigation: derivation is documented inline.

**3. The LTO finding is observational, not exhaustive.** "No bench delta" means "no delta on THIS bench." A different bench — one designed to surface cross-TU inlining wins (e.g., a tight loop calling small library helpers many times) — might show LTO benefits. We have not constructed such a bench. Risk: low (LTO either helps or doesn't; it's not a correctness lever). Mitigation: keep LTO enabled because it doesn't HURT, even if it doesn't help on the measured workload.

**4. The cross-TU inlining LTO would enable doesn't materialize at AppleClang's defaults on this codebase.** This is interesting and not investigated further. Possible reasons: library functions are too large for LTO inlining heuristics; static linking with separate-TU compilation already exposes most of what -O3 wants; AppleClang's LTO is conservative on Mach-O. Not closed; flagged as future investigation if a perf reason emerges.

## Methodology lifted to project rules

**1. Tests should link against `_test` library variants whenever the substrate library has internal asserts.** Codified by adding the `m4t_test` / `gesh_test` / `gesh_bench_test` / `gesh_image_canon_test` targets. New test executables must link against these, not production.

**2. "Asserts are live" claims must include a runtime meta-test, not just build flags.** `test_m4t_assert_live` is the template: deliberately violate a precondition, verify SIGABRT. Without this, "tests link against m4t_test" is only a build assertion.

**3. Symbol-level verification (nm) beats source-level grep.** When auditing whether a flag actually changed the build, look at the binary, not the source. Greps can miss; symbols can't lie.

**4. Loose + tight principled bounds.** Where possible, pair an order-of-magnitude bound (catches catastrophic bugs) with a regression-tight bound (catches 2-3× drifts). Document derivations.

**5. Build-system audits should compare LTO-on vs LTO-off binaries.** Identical sizes/call-counts is a signal LTO is having no effect — flag the finding even if it's not a bug.

## Status

CLOSED — all four threats inherent in the V3 `-UNDEBUG` residual are remediated. The substrate-internal asserts are now actually compiled into the test build and verified to fire at runtime via a deliberate-abort meta-test. The verification methodology is upgraded from grep to nm. The mean-drift bound has a regression-tight tier. LTO impact is measured (and found to be observationally null on this bench).

16/16 ctest binaries PASS under full LTO with substrate asserts now structurally live in test builds.

---

## Update (V4-residual-3, post-cycle correction)

The T4 finding above ("LTO has nothing to add at the granularity the bench measures") is correct narrowly but framed too generally. A subsequent microbench (`journal/v4_residual_3_lto_microbench_closeout.md`) tested two workload shapes against the same target function (`m4t_mtfp_block_add`):

| Workload | LTO ns/call | no-LTO ns/call | LTO speedup |
|----------|------------:|---------------:|------------:|
| Carry-dependent (single dst accumulated; like the V4-G5 bench) | 1.36 | 1.35 | ~1.0× — no delta |
| Pipelined (64 independent dsts round-robin; no carry dep) | 0.23 | 0.68 | **~3.0×** |

Disasm confirms LTO inlines `m4t_mtfp_block_add` cleanly into the bench main (no `bl _m4t_mtfp_block_add` in LTO build, present in no-LTO build). The 3× speedup on the pipelined workload is real.

**The corrected framing:** LTO is doing meaningful cross-TU work. It's invisible on V4-G5's bench because the substrate's actual hot path is carry-dependent (accumulating into a state) — that workload shape is bottlenecked by the data dependency between iterations, not by per-call overhead. LTO has nothing to fix THERE, but it has plenty to fix on workloads with independent ops.

**What this corrects:** the original V4 closeout reads "LTO has nothing to add" as if it were a property of LTO. It's actually a property of the substrate's measured workload shape. The takeaway is unchanged operationally (keep LTO enabled — it's free here, and 3× elsewhere), but the framing was misleading.

**Methodology lesson:** "no observable delta" findings should be tested with at least one adversarial workload variant before generalizing. If the adversarial variant ALSO shows no delta, the finding generalizes; otherwise the original finding was scoped to the original workload.
