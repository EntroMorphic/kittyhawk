# Closeout: Concern #3 — Auto-coverage for parameterized assert-live test

Per the post-cycle conversation about outstanding concerns: the V4-residual-1 parameterized meta-test enumerates 5 substrate sources by hand. If a future source file gains asserts, the test won't auto-update. Today's coverage is correct; tomorrow's might silently lose a file.

## Verdict: CLOSED

```
Cycle: design → implement → run (caught own regex bug) → fix → negative test → red-team → doc/commit
Result: ctest gate verifies cases[] matches the set of substrate sources with asserts.
        Both directions of drift caught: missing case (silent loss) AND stale case (no asserts).
```

## What shipped

`m4t/tests/check_assert_coverage.sh` — bash script, two args (substrate src dir, test file). Two list comparisons:

- **List A (truth):** `m4t/src/*.c` files containing at least one runtime `assert(` call. Matched via `grep -E '(^|[^A-Za-z_])assert\('` to exclude `_Static_assert`.
- **List B (covered):** the `source_file` string literals from the test's `cases[]` array. Matched via `grep -oE '"m4t_[a-z0-9_]+\.c"'` (digits included for `m4t_mtfp4.c`).

If A == B: PASS. If they differ: print both lists plus diff (`comm -23` for missing-from-cases, `comm -13` for stale-cases-with-no-asserts), exit 1.

`m4t/CMakeLists.txt` — wired as ctest entry `m4t_assert_coverage`. Runs on every ctest invocation.

## Validation

**Happy path:** `ctest -R m4t_assert_coverage` reports `PASS: 5 substrate source(s) with asserts; all covered by cases[]`.

**Negative test:** removed the `m4t_trit_pack.c` case from `cases[]` via sed; ran the check; it FAILED with:
```
Sources missing from cases[]:
  - m4t_trit_pack.c
```
Restored file; check PASSed again. The gate fires in the right direction.

**False-positive audit:** verified no current substrate file has `assert(` in a comment or string literal. The current grep pattern is robust for the current state. Future false positives would surface as confusing "missing case" errors rather than silent passes — visible failure mode.

## Red-team and remediation

| ID | Finding | Disposition |
|----|---------|-------------|
| RT-A | First version of the regex `[a-z_]+` excluded digits, so `m4t_mtfp4.c` was incorrectly flagged as missing from cases[]. | **FIXED** — caught by the script itself on first run. Pattern updated to `[a-z0-9_]+`. Lesson: even check scripts need their own first-run validation. |
| RT-B | A future `assert(` in a comment (e.g., `/* don't call assert() here */`) would be falsely matched. | **DOCUMENTED** — current substrate clean (verified). Failure mode would be visible (confusing "missing case" error), not silent. Defer hardening until a real false positive surfaces. |
| RT-C | New substrate file with no asserts → no case needed. Check correctly handles. | **VERIFIED** — by symmetry of `comm` operations. |
| RT-D | New asserts added to currently-empty file (`m4t_trit_ops.c`, `m4t_trit_reducers.c`) → check correctly demands a case. | **VERIFIED** — equivalent to negative test outcome. |
| RT-E | bash + grep + sort + comm portability — all POSIX, cross-platform. | **VERIFIED** for macos-14 (CI runner) and macOS local. |
| RT-F | Hardcoded path to test file in CMakeLists. | **ACCEPTED** — renaming would produce a build error, not silent failure. |
| RT-G | `assert(` inside a string literal (e.g., `printf("assert(...)")`) would be falsely matched. | **DOCUMENTED** — current substrate clean (`grep -nE '"[^"]*assert\('` returns empty). Same disposition as RT-B. |

## What's now structurally true

**Coverage drift is caught at ctest time, not at code-review time.** Pre-#3, the parameterized meta-test enumerated 5 sources by hand; a future regression (a new substrate file gains asserts but no case is added) would have produced a passing test (5/5 cases PASS, but 5/6 sources covered) and gone unnoticed until someone audited by hand. Now the gate fires automatically.

**The check is symmetric.** Catches both directions:
- Missing case: a substrate source gains asserts but no case enumerates it.
- Stale case: a case enumerates a source that no longer has asserts (silently passing the meta-test trivially because no precondition exists to violate).

## Honest concerns from this cycle

**1. Comment/string false positives are possible but not currently triggered.** If a substrate source ever contains `assert(` in a comment or string, the check produces a confusing "missing case" failure. Documented; not preemptively hardened.

**2. The check assumes single-file matching.** If a single substrate source were ever split across multiple .c files (e.g., `m4t_route_part_a.c` + `m4t_route_part_b.c`), the check would treat each as independent. Currently all substrate files are single-file. Documented.

**3. The check is not exercised in CI as a STANDALONE regression.** It runs as part of ctest, which IS in CI (now matrix-tested per concern #2). So a coverage drift WOULD be caught by CI. Not a residual.

**4. The script is bash-specific.** Pure-POSIX `sh` would work for most of it, but `set -euo pipefail` is bash. Not a current concern; the explicit `bash` in the ctest command is intentional.

## Methodology lifted

**1. Hand-enumerated coverage lists need automated drift detection.** When a test exhaustively lists "every X with property Y" by hand, add a check that compares the list against the actual set. The list will drift; the check catches it.

**2. Negative tests for gates.** A gate that PASSes in normal use should also be exercised in the failing direction at least once during development — to confirm it actually blocks the regressions it's meant to block. Done here via temporarily removing a case and verifying the failure message.

**3. Even check scripts need their first-run validation.** My initial regex bug (`[a-z_]+` missing digits) almost shipped. The script's own output during first run caught it. Don't wire a check into CI without running it locally first.

## Status

CLOSED — Concern #3 (hand-enumerated coverage) remediated. The `m4t_assert_coverage` ctest gate fires on every run, both locally and in CI. 17/17 ctest binaries PASS (was 16; +1 for the new coverage check).
