# Closeout: V4 Residual #1 — Parameterized assert-live meta-test

Per the V4 closeout's honest concern #1: the original V4-G2 meta-test exercises ONE substrate assert (in `m4t_route.c`). It proves the mechanism works, but does NOT prove every substrate assert across `libm4t_test` is actually compiled in. nm symbol counts (5/8/4/1) provide structural evidence multiple call sites exist; this cycle adds runtime evidence at every source file.

## Verdict: CLOSED

```
Cycle: remediate → red-team → fix → non-tautology check → doc/commit
Result: 5/5 cases PASS against m4t_test; 0/5 PASS against production m4t.
        Substrate asserts are uniformly live across every source file with asserts.
```

## Source-file coverage

Surveyed via `grep -c "assert(" m4t/src/*.c`:

| Source | Assert count | Covered? |
|--------|-------------:|----------|
| `m4t_route.c`           | 37 | ✓ |
| `m4t_mtfp4.c`           | 13 | ✓ |
| `m4t_mtfp.c`            | 13 | ✓ |
| `m4t_ternary_matmul.c`  |  5 | ✓ |
| `m4t_trit_pack.c`       |  3 | ✓ |
| `m4t_trit_reducers.c`   |  0 | (no asserts by design — pure compute, no preconditions) |
| `m4t_trit_ops.c`        |  0 | (no asserts by design — pure compute, no preconditions) |

5 of 7 substrate sources have asserts; the parameterized meta-test covers all 5.

## Per-case design

Each case violates a DIFFERENT precondition pattern (variety, not just "negative size everywhere"):

| Source | Function | Violation | Assert site |
|--------|----------|-----------|-------------|
| `m4t_route.c` | `m4t_route_topk_abs` | `T = 200 > M4T_ROUTE_MAX_T = 64` | `m4t_route.c:320` |
| `m4t_mtfp.c` | `m4t_mtfp_vec_zero` | `n = -1` (must be ≥ 0) | `m4t_mtfp.c:68` |
| `m4t_mtfp4.c` | `m4t_mtfp4_sdot_matmul_bt` | `M = -1` (must be ≥ 0) | `m4t_mtfp4.c:36` |
| `m4t_ternary_matmul.c` | `m4t_mtfp_ternary_matmul_bt` | `Y == X` (aliasing forbidden) | `m4t_ternary_matmul.c:213` |
| `m4t_trit_pack.c` | `m4t_pack_trits_1d` | `src[0] = 5` (trits must be in {-1, 0, +1}) | `m4t_trit_pack.c:43` |

## What shipped

`m4t/tests/test_m4t_assert_live.c` rewritten:

- Five `violate_*` functions, one per source file, each minimal (~5-10 lines).
- `assert_case_t` struct + `cases[]` array binding source file → label → violate function pointer.
- `run_case` harness: forks, runs violate in child, waits for SIGABRT in parent. Distinguishes "assert silenced" (child exits cleanly with sentinel `EXIT_ASSERT_SILENCED = 42`) from "child crashed for other reason" (different signal/exit) — so a regression points to the actual cause.
- `main` runs all 5 cases serially, prints per-case PASS/FAIL, exits 0 only if all 5 PASS.

## Red-team and remediation

Five substantive findings examined. One required code change; the others were verified or accepted as documented honest concerns.

| ID | Finding | Disposition |
|----|---------|-------------|
| RT-1 | Stdio buffer duplication: child inherits parent's buffered stdout, re-emits buffered lines on exit. Output had "test_m4t_assert_live: 5 cases..." duplicated 5×. | **FIXED** — `fflush(stdout)` and `fflush(stderr)` before each `fork()`. |
| RT-2 | Are violations triggering the INTENDED assert, not some other one in the same function? | **VERIFIED** — output's `assert()` messages name file:line; each case's message matches the targeted assert. |
| RT-3 | Earlier preconditions could short-circuit before reaching the targeted assert. | **VERIFIED** — for every case, the targeted assert is reached (confirmed in output). |
| RT-4 | -UNDEBUG silently undone by something in the build? | **VERIFIED** — all 5 cases SIGABRT under m4t_test linkage; 0/5 under production m4t linkage. Inversion is empirical proof. |
| RT-5 | Sentinel exit code 42 collision. | **ACCEPTED** — only path to exit 42 is the explicit `exit(EXIT_ASSERT_SILENCED)`; violate functions only call substrate APIs. Risk negligible. |
| RT-6 | New substrate sources with asserts won't auto-update the test. | **DOCUMENTED** — concern is real but documentation-level. The test's source comment lists which files it covers; future asserts in new files should add a case here. No automated check. |
| RT-9 | Two substrate sources have NO asserts (`m4t_trit_ops.c`, `m4t_trit_reducers.c`). | **CLARIFIED** — by design (pure compute, no preconditions). Documented in the test source. |

## Validation

**Test variant linkage (m4t_test):** 5/5 cases PASS. Each child SIGABRTs with an `Assertion failed:` message naming a different file. Console output is now clean (no duplicates after RT-1 fix).

**Non-tautology check (production m4t linkage):** built the same test source against `build/m4t/libm4t.a` (production, NDEBUG). Result: **0/5 PASS** — every case correctly reports `assert SILENCED (child returned cleanly)`. The inversion proves:

1. Every targeted assert is actually compiled away under NDEBUG (production behavior preserved).
2. The test would catch a regression in any one of the 5 source files individually.
3. The test is not vacuous — it actually depends on `-UNDEBUG` being applied uniformly across all m4t_test sources.

**Full regression:** 16/16 ctest binaries PASS.

## What's now structurally true

**Substrate asserts are runtime-verified live in every source file that has them.** Pre-V4-residual-1, "asserts are live" was supported by:
- nm symbol counts: structural evidence (multiple `___assert_rtn` refs exist in test variants).
- One runtime case in `m4t_route.c`: proof of mechanism.

Now also supported by:
- Five runtime cases, one per source file with asserts: proof per source.
- Non-tautology check: production linkage produces 0/5 PASS, test linkage produces 5/5 — uniform behavior.

## Honest concerns from this cycle

**1. Coverage is named, not counted.** The test enumerates 5 specific cases. If `m4t_route.c` had been compiled with asserts disabled while the other 4 sources were enabled, the test wouldn't catch this — it would simply skip the route case via PASS in some other path. (Actually it WOULD catch it: m4t_route case would produce "assert SILENCED" since the code wouldn't abort.) On reflection: this concern is wrong. The test does catch per-source regressions because each case targets a specific source.

**2. Asserts in HEADER FILES are not exercised here.** If someone adds an `assert()` inside an inline header function, this test won't cover it (header asserts compile into the calling translation unit, governed by THAT TU's NDEBUG state). Not a current issue — substrate headers don't have asserts at the moment. Worth flagging if that changes.

**3. The parameterized harness is fork-per-case, which is process-creation-heavy.** 5 forks = 5 child processes. Cheap (each runs <1ms), but if this expanded to dozens of cases, fork overhead would dominate. Not a current concern.

**4. RT-6 (future-proofing) remains a documentation-level gap.** Could be auto-checked by adding a build-script step that counts substrate sources with asserts and verifies it matches `N_CASES`. Not done; defer until a new substrate source actually adds asserts.

## Methodology lifted

**1. Parameterized meta-tests cover surface area; per-case assertions verify each cell of that surface.** A single case proves a mechanism; an enumerated set of cases proves uniform application. Use the latter when the claim is "X holds across N entities."

**2. Non-tautology checks belong in red-team, not just unit tests.** Building the same test source against the production lib should produce the OPPOSITE result. If both paths PASS, the test is vacuous.

**3. Stdio buffering and fork interact badly.** Always `fflush()` before `fork()` if either parent or child writes to stdio after the fork.

## Status

CLOSED — V4 residual #1 (one-assert proof) is structurally remediated. The parameterized meta-test covers every substrate source file with asserts (5 cases). Non-tautology check confirms the test is not vacuous. Substrate-internal asserts are now empirically verified to fire from every source file that has them, not just one.
