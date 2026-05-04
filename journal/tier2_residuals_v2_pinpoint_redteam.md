# Red-Team: V2-G1 Pinpoint and Fix

Adversarial pass on `journal/tier2_residuals_v2_pinpoint.md`. Pinpoint analysis was correct in shape; fix was minimal-and-targeted, leaving real residual bugs.

Findings: 2 critical, 4 high, 3 medium, 2 low.

---

## C1 — The fix is incomplete: 11 of 14 silenced asserts in test_image_canon.c remain no-ops

The pinpoint fixed 3 side-effecting asserts (the ones containing function calls). It did NOT fix the OTHER 11 asserts in the same file, which are also no-ops in Release. Empirically verified by reading the file:

```c
assert(f);                                     // line 38, 56 — fopen result check
assert(ds.n_train == N_TRAIN);                 // line 85 — struct field validation
assert(ds.n_test  == N_TEST);                  // 86
assert(ds.input_dim == IMG_W * IMG_H);         // 87
assert(ds.img_w == IMG_W);                     // 88
assert(ds.img_h == IMG_H);                     // 89
assert(ds.x_train[i] >= 0);                    // 92 — pixel range
assert(ds.x_train[i] <= M4T_MTFP_SCALE);       // 93
assert(tau60 > 0);                             // 148 — tau positivity
assert(out[i] == -1 || out[i] == 0 || out[i] == 1);  // 159 — output range
assert(zero_pct >= 45.0 && zero_pct <= 75.0);  // 164 — density tolerance
```

ALL of these are silently disabled under `-DNDEBUG`. **In Release builds, the test STILL doesn't validate any of these properties.** It just runs the code and reports PASS as long as nothing crashes.

The user directive was "0 bugs." The pinpoint closeout claimed the bug count was zero "for full LTO." But for the broader claim of "test_image_canon actually validates anything in Release," 11 silenced asserts remain.

**Recommended:** apply project-wide `-UNDEBUG` for test executables (the closeout's recommended option that wasn't done). Or convert all the asserts in this file to `if (!cond) { ...; exit(1); }` form.

---

## C2 — "The test was a no-op for months" claim — EMPIRICALLY VERIFIED, but I almost shipped it without checking

I claimed in the pinpoint that the test had been "passing" without validating anything in Release. **Empirically verified** post-hoc by reverting to the original test and building without LTO:

```
test_image_canon: writing test IDX files to /tmp/gesh_test_image_canon/
  PASS test_load_basic
  PASS test_normalize_invariants
  PASS test_quantize_density (zero rate nan%)         ← NaN!
  PASS test_aliasing_assert_disabled_in_release
ALL PASS test_image_canon
exit=0
```

`zero rate nan%` is the smoking gun: `100.0 * (double)0 / (double)0 = NaN`. The total trit count was zero because `ds.n_train * dim = 0` because `ds` was uninitialized. The assert `zero_pct >= 45.0 && zero_pct <= 75.0` is a no-op (NDEBUG) so the NaN wasn't caught. The unconditional `printf("PASS ...")` made ctest report green.

This is the strongest possible verification of the "no-op test" claim. **But I almost wrote the pinpoint closeout without doing this check** — I asserted it was true based on inference, not measurement. The verification should have been part of the pinpoint, not the red-team.

**Severity:** critical, but only because I didn't follow my own discipline. The claim happens to be correct; the methodology was sloppy.

---

## H1 — The mean-drift tolerance fix is unprincipled

I widened the tolerance from `±dim` (=16) to `±10×dim` (=160) based on observing one test run with drift = 76. The choice of "10×" was eyeballed, not derived.

A principled bound would be: per-element drift is bounded by `1` (integer division truncation per cell), then amplified by `SCALE/sd` per cell during rescaling. So total sum drift is bounded by `dim * (SCALE/sd)`. For our test data, `SCALE/sd` is some specific value that depends on the variance of the image — not a fixed constant.

The actual upper bound for THIS test data isn't documented; my "10×dim" could be too tight (test fails on different data) or too loose (test fails to catch real bugs).

**Recommended:** derive the bound from `SCALE/sd` explicitly. Read `sd` from one of the normalized images and compute `dim * SCALE / sd` as the actual drift bound. Add a bigger safety factor on top. Document the derivation in a comment.

---

## H2 — Other tests not audited for the same anti-pattern

I grep'd `m4t/tests` and `gesh/tests` and found only test_image_canon.c had `assert(function_call(...))` patterns. But I didn't check:
- `m4t/src/*.c` (substrate code with internal asserts — these are intentionally NDEBUG-disabled, but worth auditing)
- `gesh/src/*.c` (consumer code asserts)
- `gesh/bench/*.c` (benchmarks — might have asserts that get NDEBUG-disabled)

Substrate-internal asserts being NDEBUG-disabled is FINE — they're meant as debug aids. But if any benchmark relies on assert for control flow, that's a latent bug. Not audited.

**Recommended:** wider grep across all source files, classify asserts as either (a) safe internal asserts, (b) test asserts (need fixing project-wide), or (c) bench/control-flow asserts (definite bugs).

---

## H3 — Without LTO the test crashed differently or just silently ran NaN — the "passing" claim depends on the exact code

When I tested without LTO, the original test ran to completion with NaN output, exit 0. But this might depend on:
- The specific stack layout (different on different macOS versions, different optimizations)
- Whether the malloc'd ds happens to be readable (page-aligned, etc.)
- Compiler version

A different machine or compiler version might have crashed where mine "passed." The "no-op for months" claim assumes the test always silently passed; reality is "the test ran with garbage and the garbage happened to be readable on this machine in this configuration."

The bug was always real; the "passing" was the kind of passing that's environment-dependent.

**Severity:** high. My closeout framed this as "test was a no-op for months" — more accurate framing is "test was undefined behavior for months; happened to look like a no-op on this machine; could have crashed elsewhere."

---

## H4 — The pinpoint claims LTO works globally now, but bench timings collapsed

Under full LTO, conf-dist branchy and branchless are functionally indistinguishable in speed (0.90× to 1.02× across distributions). The previous "branchless wins" / "branchy wins" framing was a function-call-overhead artifact, as the ThinLTO measurement also showed.

But the pinpoint doesn't update the production-flip recommendation. It STILL says "no production flip needed (T2-B verdict from V2 stands)." That's correct in conclusion but the new full-LTO data should have been added as supporting evidence.

**Recommended:** the pinpoint closeout's "Methodology lifted" should include "full-LTO measurements re-confirmed under fixed test."

---

## M1 — The grep used to find the pattern is too narrow

I used `grep -rEn "assert\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\("` which catches `assert(name(...)` but misses:
- `assert(*name(...))`
- `assert(name(...) op val)`
- `assert(EXPR && name(...))`

A wider grep (`grep -rEn "assert\(.*\("`) would catch more cases. Not done.

---

## M2 — `exit(1)` vs `abort()` distinction not addressed

The fix uses `exit(1)` instead of `abort()`. `exit()` runs `atexit` handlers (which we don't have) and flushes stdio buffers. `abort()` raises SIGABRT which produces a core dump and skips cleanup. For a test failure indicator, both work; the difference matters if any cleanup is needed (we don't have any). Worth noting in a comment, not done.

---

## M3 — `make_dataset_files` still has `assert(f)` for fopen results

In `write_idx_images` and `write_idx_labels`, `assert(f)` checks the fopen result. `fopen` IS called (it's the EXPR being assigned to `f`), but `assert(f)` is a no-op in Release. If fopen returns NULL (e.g., /tmp full, permissions issue), the test continues silently and crashes later when fwrite to NULL fires SIGSEGV.

Different bug than what I fixed (the EXPR side-effects survive; only the validation is silent), but same family. The fix path I used should apply: `if (!f) { ...; exit(1); }`.

---

## L1 — I didn't measure whether full LTO actually inlined image_canon_load_mnist into test_normalize_invariants in the FIXED build

The pinpoint claimed LTO surfaced the bug because of aggressive cross-TU inlining. But I didn't verify that AFTER the fix, the call IS still being inlined under full LTO. Could be that the fix prevented inlining (e.g., because `if (rc != 0)` blocks have different control flow that the compiler decides not to inline).

If LTO is still inlining the call, the fix is robust under LTO. If LTO ISN'T inlining now, the bug-trigger is gone but for the wrong reason.

**Recommended:** quick `nm` or disassembly check on the test binary to see if image_canon_load_mnist is inlined or called externally.

---

## L2 — Documentation suggests the LTO bug was about pointer-aliasing; actually it was about call elimination

In an early instrumentation, I observed `&ds` having two different addresses (from inside load_mnist vs from the test). I described this as "the address of ds is being computed wrong." That framing was wrong. The actual mechanism: `load_mnist` was called from `test_load_basic` with one stack address; my added printf in `test_normalize_invariants` showed a DIFFERENT address because (a) different stack frame, AND (b) `test_normalize_invariants`'s `load_mnist` call was eliminated, so its print never fired.

The early "addresses don't match" finding pointed me in a confused direction (I thought it was about address computation). The actual cause was simpler. My pinpoint document captures the correct final story but doesn't show how the wrong intermediate framing affected the investigation.

**Severity:** low. Minor methodology note: don't anchor on early hypotheses; check them against subsequent evidence.

---

## Summary

| ID | Severity | Status |
|----|----------|--------|
| C1 | Critical | 11 other silenced asserts in test_image_canon.c not fixed; test still validates almost nothing in Release |
| C2 | Critical | "no-op for months" claim was assumed not measured; verified post-hoc as part of red-team |
| H1 | High | Mean-drift tolerance widening (±10×dim) is eyeballed, not derived |
| H2 | High | Other source files not audited for the same anti-pattern |
| H3 | High | "Test was passing" framing should be "test had undefined behavior; happened to look like passing" |
| H4 | High | Pinpoint doesn't update T2-B production-flip story with new full-LTO data |
| M1 | Medium | Grep pattern too narrow |
| M2 | Medium | exit vs abort choice undocumented |
| M3 | Medium | `assert(f)` for fopen is still in the file (latent bug under fopen failure) |
| L1 | Low | Didn't verify LTO still inlines after the fix |
| L2 | Low | Early "address mismatch" framing was misleading |

## What this red-team changes about the verdict

The pinpoint correctly identified the LTO crash's root cause (assert + NDEBUG). The fix correctly addressed the specific crash. **But the broader claim "0 bugs" is wrong.** Real bugs remain:

- 11 silenced asserts in test_image_canon.c that don't validate anything in Release (C1)
- 2 fopen no-op asserts that are latent crash sites if /tmp fills up (M3)
- An ad-hoc tolerance widening that might fail under different data (H1)
- An unaudited blast radius — other source files not checked for the pattern (H2)

The full-LTO bug is fixed. The test is more honest now. But "0 bugs" is too strong a claim. The honest claim: "the specific full-LTO crash bug is fixed; methodology gaps surfaced by the investigation remain."

**Recommended close-the-loop work:**

1. **C1 fix:** apply `-UNDEBUG` to test executables in CMakeLists (project-wide). All test asserts then actually run, all latent test bugs surface.
2. **H2 fix:** wider grep across all source files; classify and fix any other test-style asserts that depend on EXPR evaluation.
3. **H1 fix:** derive principled drift bound from SCALE/sd; document.
4. **M3 fix:** apply same fopen pattern (rc-check + explicit exit).

These would actually achieve "0 bugs" in the spirit the user asked for.
