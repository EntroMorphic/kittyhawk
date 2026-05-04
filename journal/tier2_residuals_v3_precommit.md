---
status: P0 — owner directive 2026-05-04 (close the V2-pinpoint red-team's 11 findings)
authority: owner directive — "Remediate 100/100, methodically"
predecessor: journal/tier2_residuals_v2_pinpoint_redteam.md
---

# Pre-Commit: V2-Pinpoint Remediation (V3)

## Per-finding disposition

| ID | Severity | Disposition |
|----|----------|-------------|
| **C1** | Critical | FIX — apply `-UNDEBUG` to all 15 test executables; force assertions on regardless of substrate's NDEBUG. Verify all 15 ctest binaries still PASS. |
| **C2** | Critical | DONE — verified during red-team that original test ran with NaN under no-LTO (proving the no-op claim). Document the verification approach. |
| **H1** | High | FIX — derive principled mean-drift tolerance from SCALE/sd. Use relative bound: `|sum/dim| < SCALE/10` (mean within 10% of unit scale). |
| **H2** | High | FIX — wider grep across ALL source files (substrate src, gesh src, bench, tests). Classify and fix any side-effecting asserts in non-test code. |
| **H3** | High | FIX — update pinpoint doc framing: "test had undefined behavior; happened to look like passing" instead of "no-op for months". |
| **H4** | High | FIX — pinpoint doc gets a brief addendum noting full-LTO measurements re-confirmed branchy ≈ branchless. |
| **M1** | Medium | SUBSUMED by H2 — wide grep is the fix. |
| **M2** | Medium | DOCUMENT — exit vs abort: exit(1) is acceptable; document the choice in the test code. |
| **M3** | Medium | FIX — replace `assert(f)` for fopen results with explicit if-error-exit (covered by H2 audit). |
| **L1** | Low | VERIFY — use `nm`/`objdump` to confirm image_canon_load_mnist is still inlined under LTO with the fix. |
| **L2** | Low | DOCUMENT — methodology note about not anchoring on early hypotheses. |

**11/11 disposed: 7 fixes + 2 verifies + 2 documents.**

## Pre-committed gates

A V3 PASS requires all of:

1. **V3-G1 (C1 + M3 closed):** all 15 ctest binaries compile with `-UNDEBUG` on the test executable, and all 15 ctest binaries PASS. If any test fails when its asserts are enabled, that's a real bug to fix or a tolerance to widen.

2. **V3-G2 (H1 closed):** mean-drift assertion uses a principled relative bound (`|sum/dim| < SCALE/10`) with documented derivation. Test PASSes.

3. **V3-G3 (H2 + M1 + M3 closed):** wide grep across ALL source files (m4t/src, m4t/tests, gesh/src, gesh/bench, gesh/tests) for `assert(...)` patterns. Classify into: (a) substrate-internal (intentionally NDEBUG-disabled, fine), (b) test code (covered by V3-G1), (c) bench/control-flow (any found are fixed). Audit results documented.

4. **V3-G4 (L1 verified):** `nm` or symbol inspection of `test_image_canon` confirms `image_canon_load_mnist` is still inlined under LTO with the fix (or, if not inlined, the reason is benign).

5. **V3-G5 (H3 + H4 + L2 documented):** pinpoint doc updated with corrected framing and full-LTO data; methodology lesson on early-hypothesis anchoring captured.

6. **V3-G6 (no regression):** all 15 ctest binaries PASS through every step.

## Risk register

- **Risk A: -UNDEBUG might surface other latent bugs in tests.** That's the point — but might mean fixing more tests than just image_canon. Mitigation: when failures appear, fix them honestly (don't widen tolerances to hide bugs).
- **Risk B: m4t kernel-internal asserts could fire under test if libm4t is recompiled with -UNDEBUG.** Mitigation: apply -UNDEBUG only to test EXECUTABLES, not to libm4t (which stays compiled with whatever the substrate inherits).
- **Risk C: Any side-effecting assert found in bench code might require more invasive fixes.** Acceptable; bench code isn't shipping production but is a discipline issue.

## Order of execution

1. Write this doc (in progress).
2. Apply -UNDEBUG to all 15 test executables in CMakeLists.
3. Build and run; observe any failures; fix or widen tolerances as appropriate.
4. Wide grep audit; classify and fix.
5. Update H1 mean-drift tolerance to principled bound.
6. Verify LTO inlining via nm.
7. Documentation updates (H3, H4, L2, M2).
8. Closeout doc.
