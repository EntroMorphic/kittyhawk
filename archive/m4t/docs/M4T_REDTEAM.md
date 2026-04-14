# Red-Team: M4T v0 Substrate

Audit of all files under `m4t/src/`, `m4t/tests/`, `m4t/CMakeLists.txt` after the glyph → m4t extraction. Each finding is tagged by severity.

---

## Critical — wrong answers or crashes

- [x] **R1. `m4t_mtfp.h:7` doc comment says "no int8 numeric operand" — stale after policy revision.** MTFP4 (int8) and MTFP9 (int16) are now approved cell types. The banner must be updated to reflect the revised policy. Currently misleads readers into thinking int8 is banned in m4t.

- [x] **R2. `m4t_ternary_matmul.c:20` bound analysis comment says "|X[k]| ≤ M4T_MTFP_MAX_VAL ≈ 1.07e9" — stale.** MAX_VAL is now 581,130,733, not 1.07e9. The bound is still safe (smaller MAX_VAL = smaller max sum), but the comment is numerically wrong and misleads anyone checking the invariant.

- [x] **R3. `m4t_mtfp.c:22` comment says "MAX_VAL = INT32_MAX/2" — stale.** MAX_VAL is now `(3^19-1)/2 = 581,130,733`, which is less than `INT32_MAX/2 = 1,073,741,823`. The halving invariant still holds (sum of two in-range cells ≤ 1,162,261,466 < INT32_MAX), but the comment references the old value. Misleading.

- [x] **R4. `m4t_mtfp.h:53` doc comment says "e.g. 2.0 * 2.0 = 4.0, well below ±18183" — stale.** Real range is now ±9842, not ±18183. Both are above 4.0 so the statement is still *true*, but the cited number is wrong and will confuse readers.

## High — correctness/coverage gaps

- [x] **R5. No test exercises a value near the NEW MAX_VAL boundary (581,130,733).** All existing tests use small values (≤ 17·S ≈ 1M). We don't know if the tightened clamp produces correct results at the new boundary. Need a test that pushes values to ±MAX_VAL and verifies saturation.

- [ ] **R6. `m4t_types.h` defines four cell types but only MTFP19 has any opcode or test.** MTFP4, MTFP9, and MTFP39 are declared as typedefs and constants but have zero functions and zero tests. The types compile but are not exercised. A reader could construct an MTFP4 cell and call `m4t_mtfp_add` (which takes `m4t_mtfp_t = int32`), getting implicit promotion — silent wrong behavior since the MTFP4 cell has a different scale (9, not 59049).

- [x] **R7. `m4t_mtfp_mul_trit` does not saturate.** `return a * (int32_t)t;` where `a = MAX_VAL` and `t = -1` gives `-MAX_VAL`, which is fine. But if `a` somehow exceeds MAX_VAL (e.g., caller passes a corrupted cell), `a * -1` could overflow int32 for `a = INT32_MIN`. Low probability since inputs should be in-range, but every other arithmetic op saturates and this one doesn't. Asymmetric.

- [x] **R8. `m4t_mtfp.h:8` says "no LUT-backed nonlinearity in this version" — still true but should be updated to say "planned for pipeline item 1 (TBL-based trit ops)" to point readers toward the roadmap.** Doc gap, not a bug.

## Medium — quality, consistency, documentation

- [x] **R9. `m4t_ternary_matmul.c` NEON kernel uses `vshlq_u8(dup, vnegq_s8(shift_s))` for variable right-shift.** Correct but the idiom is non-obvious. No inline comment explains why the shift count is negated. The REFLECT phase flagged this (S4 from glyph REDTEAM_FIXES) and it was deferred. Still deferred.

- [ ] **R10. `m4t_ternary_matmul.c` uses `vmulq_s32` for ternary sign select.** Correct but wasteful — pays a full int32 multiply for a conditional negate. Previously flagged; still not benchmarked. Blocked on pipeline item 7 (cycle-count harness).

- [ ] **R11. `m4t_mtfp.c` `vec_scale` is scalar-only.** Comment acknowledges it; NEON vectorization deferred. Still true.

- [x] **R12. `m4t_mtfp.c` `fan_in_normalize` uses truncating division without symmetric rounding.** The same pattern that was caught in LayerNorm (NEW1) but not applied here. `x[i] = x[i] / norm` truncates toward zero. For negative cells this biases toward zero by up to one ULP per normalization. Low impact per call but compounds across layers.

- [ ] **R13. No `m4t_` prefix on the decode LUT symbol.** `M4T_TRIT_DECODE_LUT` is correctly prefixed, but the internal static functions in `m4t_trit_pack.c` (`trit_to_code`, `code_to_trit`) are unprefixed. These are `static` so no linkage collision, but for grep-ability and consistency they should be `m4t_trit_to_code` / `m4t_code_to_trit`. Minor.

- [ ] **R14. CMakeLists.txt checks `CMAKE_SYSTEM_PROCESSOR` for aarch64 but does not check for NEON specifically.** NEON is mandatory on all AArch64 cores per the ARMv8 spec, so this is safe in practice, but if a toolchain emits `arm64` (Apple's name) or `aarch64` (Linux) the regex `"arm64|aarch64"` handles both. Verified correct.

- [ ] **R15. `m4t_internal.h` `#error` fires if `__ARM_NEON` is not defined, but the scalar fallback code in the .c files is still present.** The fallback is unreachable dead code (the `#error` prevents compilation). This is intentional per the glyph REDTEAM_FIXES S6 resolution — the scalar code serves as documentation — but it's worth noting that it's never compiled and could rot silently.

## Low — cosmetic, doc-only

- [x] **R16. Test file header still says "test_m4t_mtfp_smoke.c" in the comment but filename is `test_m4t_smoke.c`.** Cosmetic mismatch from the sed rename.

- [ ] **R17. No `docs/M4T_CONTRACT.md` yet.** The seven-clause contract exists in `journal/ternary_opcode_synthesize_v2.md` but not as a standalone caller-facing document. Pipeline item 6 will create the opcode tables; the contract doc should precede or accompany it.

- [x] **R18. `m4t_types.h` MTFP39 MAX_VAL uses `LL` suffix on the literal.** Technically `int64_t` is `long long` on most platforms, so `LL` is correct, but the pedantically portable form is `INT64_C(2026277576509488133)` from `<stdint.h>`. Minor.

---

## Severity summary

| Severity | Count | Action |
|---|---|---|
| Critical | 4 (R1–R4) | Stale comments — all quick doc fixes, no code logic change |
| High | 4 (R5–R8) | R5 needs a test; R6 is a design note (expected, deferred); R7 is a real asymmetry; R8 is doc |
| Medium | 7 (R9–R15) | Mix of deferred smells and doc gaps |
| Low | 3 (R16–R18) | Cosmetic |

## Honest assessment

The sed-based rename was clean — no broken symbol references, no stale include guards, no mismatched function signatures. The code compiles warning-free and all 18 tests pass with the tightened MAX_VAL. The structural changes (dispatch removal, MAX_VAL tightening) are correct.

The main debt is **stale comments** (R1–R4): four doc strings reference the old glyph policy or the old MAX_VAL value. These are easy to fix and should be fixed now before they mislead anyone.

R7 (`mul_trit` not saturating) is a real asymmetry with the rest of the arithmetic surface. Low risk in practice but worth closing for API consistency.

R12 (`fan_in_normalize` truncation) is the same class of bug as NEW1 (the LayerNorm rounding asymmetry) but in a lower-traffic path. Worth fixing while we remember what the pattern looks like.

R5 (boundary test at new MAX_VAL) is the most important test gap — the cell range changed and no test probes the new boundary.
