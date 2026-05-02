---
cycle: m4t_matmul_redteam
phase: CLOSEOUT (adversarial review of tier 3b/3c, all findings remediated)
date: 2026-05-01
scope: post-build red-team of m4t_mtfp4_sdot_matmul_bt + cell-width
       conversions + m4t_mtfp_ternary_matmul_bt; 11 findings; all remediated
companions: m4t/src/m4t_mtfp4.{h,c} · m4t/src/m4t_ternary_matmul.{h,c} ·
            m4t/tests/test_m4t_mtfp4.c · m4t/tests/test_m4t_ternary_matmul.c ·
            journal/xexpo_kernel_redteam.md (prior cycle's red-team)
status: COMPLETE — all findings remediated 2026-05-01
---

# Tier 3b/3c kernel red-team — SDOT MTFP4 + MTFP19 ternary matmul

Adversarial pass over the SDOT MTFP4 matmul, cell-width conversions, and MTFP19 ternary matmul built earlier in the same session. 11 findings (2 high, 5 medium, 4 low). All remediated in the same commit cycle.

The pattern from `journal/xexpo_kernel_redteam.md` repeats: same-author red-team finds discipline issues that paid attention to the prior cycle's lessons but missed new failure modes specific to the matmul shape.

## High-severity findings

### H1 — SDOT silent invariant violation for K > 14,528,268 (FIXED)
The kernel computed `acc` in int32 and wrote directly to `m4t_mtfp_t` storage. For `K > MAX_VAL_MTFP19 / MAX_VAL_MTFP4 = 14528268`, the worst-case output `|acc| = K · 40` exceeded MTFP19's documented mantissa range, but the kernel wrote the out-of-range value silently — no flag, no clamp, no assert. Subsequent kernels reading those cells would see out-of-range mantissas (substrate invariant violation).

The CHANGELOG and m4t/README claimed "exact by construction" without flagging this bound as a precondition. Promise mismatch.

**Fix:** Added `M4T_SDOT_K_MAX_EXACT` macro (compile-time-derived from cell-type max values) to `m4t_mtfp4.h`. Added `assert(K <= M4T_SDOT_K_MAX_EXACT)` in the kernel. Updated header docstring to declare the K bound as a hard precondition. Beyond-bound usage is documented as caller's responsibility (partition into chunks, use cross-exp accumulator across chunks, or wait for a wider-output variant when a consumer asks).

### H2 — `test_sdot_matmul_max_bound` didn't actually test the max bound (FIXED)
The test used K=4096 — far below the 14.5M boundary. The name implied coverage of the kernel's worst-case input space; in fact it tested moderate-K behavior at high magnitudes.

**Fix:** Renamed to `test_sdot_matmul_high_mag` (truthful). Added a new `test_sdot_matmul_long_k` running K=1M with adversarial mixed-sign random inputs against an int64 reference. K=1M is partway to K_MAX_EXACT (~7%); covers the algorithm under realistic large-K workloads without requiring multi-MB allocations.

## Medium-severity findings

### M1 — No property test for `m4t_mtfp19_to_mtfp4` narrow (FIXED)
The cross-exp accumulator earned 14 properties via random sampling against an int64 reference. The narrow conversion has equivalent rounding-rule complexity (round-to-nearest-even with odd divisor 6561) but only 4 hand-derived test cases.

**Fix:** Added `test_narrow_property` — 10,000 random src vectors of 64 cells each (~640k random samples total). Mixed-distribution: 7/8 uniform random over MTFP19 range, 1/8 boundary-targeted (rescale halfway points, saturation edges). Bit-exact comparison against an `int8 narrow_reference()` helper that mirrors the kernel algorithm structurally.

### M2 — No long-K stress test for either matmul (FIXED)
Both kernels claimed safety up to large K (14.5M for SDOT, 1.59e10 for ternary). Tests capped at K=1024 (SDOT) and K=100 (ternary).

**Fix:** Added `test_sdot_matmul_long_k` (K=1M) for SDOT and `test_long_k` (K=1M) for ternary matmul. Both use bit-exact int64 reference comparisons. K=1M exercises the NEON inner loop tens of thousands of times per kernel call.

### M3 — No partial-block-flag-bits test for ternary matmul (FIXED)
The cross-exp accumulator's `prop_accum_partial_block` verifies trailing-block bits past `n` stay zero. The ternary matmul writes flags over `M·N` output cells using the same per-block layout, but `M·N` may not be a multiple of 4.

**Fix:** Added `test_partial_block` with `M·N = 5` (forces `flags[1]` to hold cell 4 in bits 0-1 only; bits 2-7 must stay zero). Saturating inputs ensure the kernel writes flag bits for cell 4; bits 2-7 of `flags[1]` are checked for unintended modification.

### M4 — Reserved trit code (0b11) untested (FIXED)
Per `m4t_trit_pack.h`, code 0b11 is "reserved (treated as 0)." The decode LUT maps it to 0 in both NEON and scalar paths. A future refactor could diverge the two paths; no test caught this.

**Fix:** Added `test_invalid_trit_code` — packs two weight buffers, identical except one uses code 0b00 and the other uses 0b11 in the zero positions. Kernel must produce identical output. K=20 ensures both NEON loop body (16 trits) and scalar tail (4 trits) execute.

### M5 — Sample-based W validity check covers 2 of N·K weights (DOCUMENTED)
SDOT matmul's debug assert checks `W[0]` and the last cell of W. For N=10, K=64, that's 2/640 = 0.3% coverage. A buggy caller with valid first/last weights but invalid middle weights passes.

**Fix:** Made the trade-off explicit in the header docstring: "The substrate trusts the caller at the boundary; debug builds spot-check W[0] and the last cell of W[N-1] only — exhaustive validation would scan O(N·K) per call, too expensive for the hot path. Consumers that need exhaustive validation should run it once at W setup time, outside the matmul loop." The discipline is preserved (substrate trusts caller); the documentation now accurately describes what the assert does.

## Low-severity findings

### L1 — Unused locals in tests (FIXED)
`test_saturation_clamp` and `test_saturation_flags` in `test_m4t_ternary_matmul.c` declared `int Kp = M4T_TRIT_PACKED_BYTES(K);` then immediately `(void)Kp;`. Dead code.

**Fix:** Removed. Replaced the awkward `[1 * 1]` array sizing with explicit `[1]` and `[2]`, with comments explaining the derivation.

### L2 — Unused helper in `test_m4t_mtfp4.c` (FIXED)
`rand_mtfp19()` was declared and silenced via `(void)rand_mtfp19;` at the end of `main`. Dead code.

**Fix:** `rand_mtfp19` is now used by `test_narrow_property` (M1's remediation). The `(void)rand_mtfp19;` line was removed.

### L3 — Header docstring didn't pin "valid inputs" (FIXED via H1)
"No flags parameter: by §8.4 contract this kernel does not saturate or round under valid inputs." Without the K bound made explicit, "valid inputs" was under-specified.

**Fix:** H1's remediation made the K bound a hard precondition with a public macro. The header now reads: "the kernel does not saturate or round under valid inputs (where 'valid' includes K ≤ M4T_SDOT_K_MAX_EXACT)."

### L4 — Cross-reference gap in DESIGN_X-EXPO (FIXED)
The cross-exp design doc's flag-layout section described per-block layout for the cross-exp kernel only. Consumers reading the design wouldn't realize the layout is shared substrate-wide.

**Fix:** Updated the section header to "Flag layout (§14.4 status array — per-block, substrate-wide)" and added a paragraph listing every Case-S/Case-R kernel that uses the same layout, plus the location of the shared setter (`m4t_internal.h`) and reader (`m4t_mtfp.h`).

## Outcome

Build passes 8/8 ctest binaries. Tier 3b/3c test counts grew:
- `test_m4t_mtfp4`: 10 → **12 tests** (added `sdot_matmul_long_k`, `narrow_property`; renamed `max_bound` → `high_mag`).
- `test_m4t_ternary_matmul`: 6 → **9 tests** (added `long_k`, `partial_block`, `invalid_trit_code`).

Total m4t test surface: 8 ctest binaries, ~50 distinct test functions, 14 cross-exp properties + 12 mtfp4 + 9 ternary_matmul + the rest of the substrate.

The kernels ship with:
- Hard precondition on K for SDOT, with public `#define` macro and runtime assert.
- Property-tested narrow conversion (10k random samples + boundary-targeted distribution).
- Long-K stress tests on both matmuls.
- Trailing-block-bits-zero verification for ternary matmul.
- Reserved trit code 0b11 path coverage.
- Documented sampling limitation on weight-validity check.

## Methodology note

This is the second same-author red-team in this session. The first (`xexpo_kernel_redteam.md`) found 14 issues in a single kernel; this one found 11 issues across three kernels + their tests. Lessons from the first cycle visibly transferred (per-block flag layout used from the start, aliasing assertions in place, n=0 contracts honored), but new failure modes specific to matmul shape (K-bound precondition, decode-table coverage, partial-block flag indexing) emerged. The pattern is consistent with the LMM doc's claim that adversarial review finds issues in distribution-shifted regions of the design space.

Independent external review would catch a different distribution. These findings are what same-author adversarial review surfaces.
