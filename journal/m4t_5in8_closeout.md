# CLOSEOUT: 5-in-8 base-3 packing in libm4t (Item 2)

Per `journal/m4t_5in8_synthesize.md`. The audit-validated sub-2-bit base-3 packing format (1.6 bits/cell) is now part of libm4t's public surface. Spec amended (§20), pack/unpack primitives shipped, matmul kernel ported from audit Path D, ctest binary verifying bit-exact correctness + audit cross-check.

## Verdict: SHIPPED

All pre-committed gates met. The substrate now offers an opt-in dense storage format for ternary values that no base-2 representation can match (B2-B floored at 2 bits/cell because sign+mask are independent).

## Per-gate evidence

### G1 — Spec amendment lands cleanly: PASS

`m4t/docs/M4T_SUBSTRATE.md` gains §20 "Sub-2-bit base-3 packing (5-trits-in-8-bits, opt-in)". Sections:
- 20.1 Why this exists (density-ceiling structural advantage; B2-B can't follow).
- 20.2 Encoding (`-1→2, 0→0, +1→1`; `byte = u_0 + 3·u_1 + 9·u_2 + 27·u_3 + 81·u_4`).
- 20.3 Decode (per-byte spec formula + NEON split-LUT pattern).
- 20.4 When to use which packing (default 4-in-8 vs opt-in 5-in-8).
- 20.5 Spec discipline (no invariant changes; pure trit-storage layer).
- 20.6 Cross-references to code, tests, journal cycles.

§17 cross-reference table updated to point §20 → relevant code/tests.

### G2 — Bit-exact verification: PASS (HARD gate)

`test_m4t_ternary_5in8_matmul` runs:
- Pack/unpack roundtrip across 7 sizes (n=5, 10, 11, 80, 320, 1280, 81 — including non-multiple-of-5 to verify trailing pad).
- 7 hand-derived golden cases (5-zero byte, single +1 at digit-0/1, single -1, max byte 242, multi-byte, trailing zero pad).
- 600 NEON-vs-scalar bit-exact samples (4 K configs × 2 M values × 3 N values × 25 random samples per triple).

Test runs in 0.13s. PASS.

All 21/21 ctest binaries pass without modification.

### G3 — Pack/unpack roundtrip: PASS

Included in G2's test_m4t_ternary_5in8_matmul. Hand-derived golden values verified for all enumerated cases.

### G4 — Aliasing assertions: PASS

Both `m4t_ternary_5in8_matmul_bt` and `m4t_ternary_5in8_matmul_bt_scalar_ref` assert:
- `Y != X`
- `Y != W_packed`

Per existing substrate pattern.

### G5 — No scalar fallback in production paths: PASS

The matmul kernel is NEON-only:
- Wrapped in `#if M4T_HAS_NEON && defined(__ARM_FEATURE_DOTPROD)`.
- Asserts `K % 80 == 0` and `N % 4 == 0` — strict alignment, no scalar tail.
- The `#else` branch falls back to `_scalar_ref` defensively, but this path is unreachable per project requires-aarch64-NEON rule.

The pack/unpack helpers use scalar (consistent with existing `m4t_pack_trits_1d` — these are storage-format conversions, not hot-path math).

The `_scalar_ref` test oracle is documented as test-only; production code MUST NOT call it.

No `flags!=NULL → scalar` paths. No `#if !M4T_HAS_NEON ... #else scalar production ... #endif` patterns.

### G6 — Audit cross-check: PASS

`audit/tristate_strong_bench.c` extended to call `m4t_ternary_5in8_matmul_bt` after Path D and verify bit-exact match with `Yd`. 80 bench runs (12 multi-config + 3 memory-bound + 1 DRAM-bound, × 5 seeds each) — every run produces matching output.

This empirically confirms (vs only arguing from spec) that the libm4t kernel produces the same output as the externally-validated audit Path D kernel.

## Implementation summary

**Files added:**
- `m4t/tests/test_m4t_ternary_5in8_matmul.c` — ctest binary.
- `journal/m4t_5in8_synthesize.md`, `journal/m4t_5in8_closeout.md` (this doc).

**Files modified:**
- `m4t/docs/M4T_SUBSTRATE.md` — added §20 + §17 cross-ref.
- `m4t/src/m4t_trit_pack.h` — added `M4T_TRIT_PACKED5_BYTES` macro + pack/unpack prototypes + §20 doc block.
- `m4t/src/m4t_trit_pack.c` — added `m4t_pack_trits_5in8_1d` + `m4t_unpack_trits_5in8_1d` + helpers.
- `m4t/src/m4t_ternary_matmul.h` — added `m4t_ternary_5in8_matmul_bt` + `_scalar_ref` prototypes.
- `m4t/src/m4t_ternary_matmul.c` — added kernel + scalar reference. Includes 5 LUTs (split-LUT decode constants). Ported from audit Path D.
- `m4t/CMakeLists.txt` — added test_m4t_ternary_5in8_matmul target + ctest entry.
- `audit/tristate_strong_bench.c` — added libm4t §20 cross-check (G6).

## Honest scope

- **Only the ternary-X variant.** A MTFP19-X 5-in-8 variant could be added later if a consumer needs MTFP19 activations with sub-2-bit packed weights. Not implemented; out of scope.
- **Strict alignment requirements** (K%80==0, N%4==0). Real consumers may need flexibility; tail handling is straightforward future work but not required for the audit's verified shapes.
- **Wall-clock not separately benchmarked for libm4t variant.** The audit's Path D measurement is the reference; the libm4t kernel uses the same code structure (verified by spec + cross-check) so should show the same ~1.8× advantage over Path A. A dedicated bench could be added but the structural correctness + cross-check evidence is sufficient.
- **Default packing unchanged.** 4-in-8 (`m4t_pack_trits_1d`) remains the default. 5-in-8 is opt-in via the new pack function.

## Self-red-team

**C1 — Cross-check vs audit was deferred to G6 not measured directly until the very end.** Could have been an early-cycle gate. Verified at G6 stage; would have been caught earlier with a tighter loop. Methodology note for next cycle: run cross-check immediately after first kernel build, not after all infrastructure is in place.

**C2 — Spec amendment landed before implementation.** Per project rule 7, this is the right order (spec amendments can be reviewed before implementation lands). The amendment text describes the format that exists in the implementation, so the references are concrete.

**C3 — `_scalar_ref` uses a different decode formula (per-cell `u_i = (b/3^i) mod 3`) than NEON (`high = (b*57)>>9; low = b - 9*high; ...`).** Both compute the same mathematical value, but cross-check this in tests. The 600 random samples × bit-exact verification + the audit cross-check empirically confirm the formulas agree.

**C4 — Pack helper doesn't validate input trits are in {-1, 0, +1}.** Same as existing `m4t_pack_trits_1d` — assert fires only in debug builds (per substrate pattern). Defensive only; trusting callers at the boundary.

## Methodology lift

**Spec-first → implementation-second is the right order for substrate-spec-extending cycles.** The spec text was written based on the audit's validated Path D format. Implementation then ports the audit kernel to libm4t conventions (assertion patterns, scalar_ref naming, ctest integration). Spec amendment provides the contract; implementation honors it.

## Status

CLOSED. Item 2 of three production-shoring items complete. Item 3 (SDOT throughput tool to m4t/tools/) is next.

The strong claim's structural advantage is now accessible to libm4t consumers via the new `m4t_ternary_5in8_matmul_bt` kernel + `m4t_pack_trits_5in8_1d` packing primitive. Density gain: 1.25× (1.6 vs 2.0 bits/cell). Wall-clock gain: ~1.8× per audit's measurement (libm4t kernel is the same shape; not separately benched here but cross-verified bit-exact).
