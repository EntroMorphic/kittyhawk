# PRE-COMMIT: shift3 NEON cycle remediation (100/100)

Per `journal/shift3_neon_redteam.md`. Locks in 12 gates BEFORE execution. Verification follows the V4-residuals discipline pattern (per `CONTRIBUTING.md` post-commit doc-currency checklist).

## Verdict commitment

I am committing now to PASS verdicts on every gate below. If any FAIL during execution, the cycle stops there and the failure is recorded honestly (no rationalizing the gate to fit the result).

## Gates

| ID | Closes | What | PASS bar |
|----|--------|------|----------|
| **R-G1** | foundation | Add `m4t_mtfp_shift3_scalar_ref` to public substrate API: scalar-only divide, lifted from current fallback. Production never calls it; tests use it as oracle. | Compiles; new symbol present in `nm libm4t.a` |
| **R-G2** | M2 | Extract NEON path inside `m4t_mtfp_shift3` into a `static` helper (e.g., `shift3_div_neon_path`). Clean separation. | 18/18 ctest still PASS |
| **R-G3** | C1 | Update test: `m4t_mtfp_shift3` (NEON) vs `m4t_mtfp_shift3_scalar_ref` (scalar oracle). | Sample test still 19/19 PASS — and now is comparing NEON-vs-scalar (not NEON-vs-NEON) |
| **R-G4** | C2 | Update perf bench: scalar measurement uses `m4t_mtfp_shift3_scalar_ref`. | Bench produces speedup > 1.0× across both shapes |
| **R-G5** | C1, C3 | Re-run G1 exhaustive against the actual scalar reference. 22.08 × 10⁹ test points. | 19/19 EXHAUSTIVE PASS |
| **R-G6** | H1 | Remove `m4t_shift3_div_neon` (prototype kernel copy) from the test file. Test calls production `m4t_mtfp_shift3` directly. | Test still PASSes; only ONE NEON kernel exists in the repo |
| **R-G7** | H2 | Re-measure speedup with corrected comparison; record numbers in CLOSEOUT and CHANGELOG. | Numbers reported with workload-shape declared (per CONTRIBUTING scope-match rule) |
| **R-G8** | M1 | Rename `test_m4t_shift3_neon_proto.c` → `test_m4t_shift3_neon.c`. Update CMakeLists. | Build clean; ctest entry name unchanged (`m4t_shift3_neon`) |
| **R-G9** | M3 | Add CI-skipped exhaustive ctest target (`m4t_shift3_neon_exhaustive`). Default ctest skips it; can be invoked via ctest label or env. | New target exists; default ctest unchanged at 18/18; explicit invocation runs the 25-second G1 verify |
| **R-G10** | M4 | Soften CLOSEOUT framing: divide is NEON, multiply is partly auto-vectorized with further headroom. | CLOSEOUT statement no longer overstates |
| **R-G11** | L1 | Update `m4t/docs/M4T_SUBSTRATE.md` tree listing for new files. | New files appear in the tree |
| **R-G12** | L2 | Add code comment in production NEON path explaining vqrdmulh-pivot reasoning + journal pointer. | Comment present in `m4t_mtfp.c` |

## Order of execution

R-G1 first (foundation; everything else depends on it). Then R-G2 (refactor; cosmetic but reduces churn during R-G3-R-G6). Then R-G3-R-G6 in sequence (each depends on the previous). Then R-G7 (measurement). Then R-G8-R-G12 (cleanup, parallel-ish).

After every code change: `cmake --build build && ctest`. Don't proceed if regression.

## Risk register

- **RR1 (R-G1):** the scalar reference function might not be link-preserved if no test actually calls it; LTO could DCE it. Mitigation: tests reference it, so it stays in the binary.
- **RR2 (R-G2):** extracting the NEON path could subtly change behavior (e.g., if the helper takes references vs values). Mitigation: bit-exact verify post-extraction (R-G5).
- **RR3 (R-G3):** the test's "ref" call needs to handle the same input semantic (k ∈ [-19, -1]) as before. Mitigation: don't change function signatures; just swap the function being called.
- **RR4 (R-G7):** the re-measured speedup may differ from the previous "9.5×" claim. The current red-team already flagged this; the new number is the honest one. Mitigation: document the change explicitly in the CLOSEOUT update.
- **RR5 (R-G8):** renaming files in git can break linkage if not careful. Mitigation: `git mv` to preserve history; update CMakeLists in same commit.

## Out of scope

- Original vqrdmulhq-with-per-k-specialization (~1.5–2× further speedup) — deferred per closeout.
- Multiply-direction NEON optimization — deferred per closeout.
- Cross-exp accumulator per-cell-varying-k variant — deferred per closeout.
- Magic-table CI drift check — deferred (R-G9 covers manual-mode; full CI integration is later).

## Done when

All 12 R-G gates PASS, CLOSEOUT written + committed + pushed, CI matrix green on both LTO=ON/OFF.
