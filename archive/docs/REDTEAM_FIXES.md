# Red-Team Fixes — v0 Core

Tracking document for the red-team pass on `src/glyph_mtfp.c`, `src/glyph_trit_pack.c`, `src/glyph_ternary_matmul.c`, and `tests/test_glyph_mtfp_smoke.c`. Items are worked in severity order and each fix is red-teamed after landing.

Status markers: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` escalated / rejected

## Critical — wrong-answer bugs

- [x] **NEW1. LayerNorm rounding is asymmetric for negative values.** `(norm * weight + S/2) / S` in `layernorm_row` only adds `S/2` regardless of sign, so for negative intermediate products C's truncation-toward-zero bias the result by one ULP toward zero. Concrete reproducer: `norm = -1`, `weight = S` → expected `-1`, actual `0`. The dense `matmul_row` uses the correct symmetric pattern; LayerNorm regressed from it. Fix: apply the `(num >= 0 ? +S/2 : -S/2)` pattern, same as `glyph_mtfp_mul`.

- [x] **C4. `glyph_mtfp_layernorm` divides by `cols` with no guard.** `sum / cols` crashes if `cols == 0`. Fix: early-return or assert.

- [x] **T5. LayerNorm non-constant-row test.** The existing `test_layernorm_constant_row` passes trivially because `centered == 0` erases every downstream path. The rstd/isqrt/scale/bias pipeline is currently unexercised, which is how NEW1 hid. Fix: add a test with an analytically computed, integer-valued expected output and bracket against NEW1 specifically.

## High — defensive correctness and coverage

- [x] **C3. LayerNorm `centered * rstd` theoretical overflow.** Resolved as documentation: `centered` and `rstd` are anti-correlated (large `centered` implies large variance, which shrinks `rstd`), so the product is physically bounded by `O(sqrt(N) · S²)`. For any reasonable N the int64 accumulator has >99% headroom. Documented the invariant; no code change. In the int64 path, for pathological `var_eps = 1` combined with `centered = ±MAX_VAL`, the product can reach ~7.5e18 (82% of `INT64_MAX`). Physically the two cannot co-occur (large `centered` implies large `var`, which shrinks `rstd`), but the type system doesn't know that. Fix: widen the inner multiply to `__int128` on aarch64, or clamp `rstd` to a documented ceiling. Prefer `__int128` — eliminates the concern without adding heuristics.

- [x] **T2. No test for dense `glyph_mtfp_matmul` / `_bt`.** Added `test_dense_matmul_2x2` and `test_dense_matmul_bt_2x2` with hand-derived integer golden values.

- [x] **T6. Ternary matmul NEON tail untested.** Added `test_ternary_matmul_bt_k17_tail` — K=17 exercises NEON block (16 trits) + scalar tail (1 trit) with three weight patterns.

- [x] **T7. Ternary matmul `M > 1` untested.** Added `test_ternary_matmul_bt_m3` — M=3, distinct rows per batch, exercises dispatch (which now also runs serial since M<threshold).

- [x] **T4. `mtfp_mul` near `±MAX_VAL` untested.** Rounding-at-boundary and saturation behavior is unverified. **Found real bug: mul did not saturate — now saturates at `±GLYPH_MTFP_MAX_VAL`.**

- [x] **NEW2. `glyph_mtfp_add` / `_sub` and `vec_add` / `vec_add_inplace` do not saturate.** Discovered during T4 red-team. `add(MAX, MAX) = 2·MAX_VAL` silently exceeds spec; residual connections and bias adds in a real transformer accumulate drift and eventually wrap int32. Fix: scalar inlines clamp; NEON paths use `vqaddq_s32` (zero perf cost on M4). Severity: Critical (wrong-answer bug in hot path).

## Medium — quality, deduplication, perf

- [x] **S1. `clamp_mtfp` and `GLYPH_HAS_*` macros duplicated across TUs.** `glyph_mtfp_clamp64` is now a single public inline in `glyph_mtfp.h` (added during NEW2 fix). Platform macros live in private `src/glyph_internal.h`. `glyph_mtfp.c` has `clamp_mtfp`, `glyph_ternary_matmul.c` has `clamp_mtfp_i64` — same function, two names. `GLYPH_HAS_NEON` and `GLYPH_HAS_DISPATCH` are triplicated. Fix: private `src/glyph_internal.h`, not exported.

- [x] **A1. `dispatch_apply` at `M=1` launches libdispatch for nothing.** `GLYPH_SERIAL_ROW_THRESHOLD = 4` in `glyph_internal.h`. All four dispatch sites (matmul, matmul_bt, layernorm, ternary_matmul_bt) now fall through to a serial loop when rows < threshold.

- [x] **A5. `GLYPH_MTFP_MAX_VAL = INT32_MAX/2` is unjustified in the source.** Documented in `glyph_types.h`: the ÷2 is load-bearing so that non-saturating `vaddq_s32` followed by `vminq/vmaxq` clamp is safe — the sum of two in-range operands cannot wrap int32 before the clamp runs. Cross-referenced to NEW2. Inherited from trix-z's MTFP21, but glyph's 3¹⁰ scale and int32 container do not motivate halving. Fix: either justify in `glyph_types.h` or change the clamp. Needs thought — the ÷2 leaves headroom for sums without overflow, which matters for accumulator spilling into output cells.

- [ ] **T3. No tests for `vec_scale`, `bias_add`, `fan_in_normalize`.** Coverage gap.

## Low — cosmetic, doc, nice-to-have

- [ ] **C1. Document why `glyph_mtfp_ternary_matmul_bt` does not rescale.** Ternary weights are dimensionless; no divide-by-SCALE needed. Add a comment so the asymmetry with dense matmul is obvious.

- [ ] **C2. Document `S/2 = 29524` rounding-tie boundary.**

- [ ] **T1. `M_HALF = S/2 = 29524` is off by 1 from true 0.5.** Document or use a constant computed from an integer tie-break.

- [ ] **S2. `matmul_row` and `matmul_bt_row` near-duplicates.** Optional template merge.

- [ ] **S3. `ternary_dot` NEON block is ~80 lines of a static function.** Extract to `ternary_dot_block_neon` for readability.

- [ ] **S4. `vshlq_u8(dup, vnegq_s8(shift_s))` idiom needs a longer comment.**

- [x] **S5. Pedantic VLA-folded warnings in test.** Fixed by converting test locals from `const int` to `enum` constants so array sizes are true compile-time constants. `uint8_t W_packed[2 * Kp]` folds to a constant but warns under `-Wpedantic`. Fix with static sizing.

- [x] **S6. NEON fallbacks on an M4-only target.** `glyph_internal.h` now `#error`s if `__ARM_NEON` isn't defined. Scalar fallback code is retained as executable documentation of the NEON kernels but is unreachable on the supported target. Either commit to NEON-only via `#error` on non-aarch64, or justify the scalar fallback. Leaning toward `#error` since the CMake already hard-errors on non-Apple.

- [ ] **S7. `vec_scale` scalar TODO not tracked.**

- [x] **S8. No debug assertions.** Added `assert()` preconditions on pointers and non-negative dimensions at all public entry points (matmul, matmul_bt, layernorm, ternary_matmul_bt). Zero cost under `NDEBUG`. Add `assert()` on K/N/M/pointer preconditions under `NDEBUG`.

- [ ] **P1. Container-type comments for `int8_t` / `int8x16_t`.** Document they are trit containers, not numbers, at first use in each file.

- [ ] **P2. Scratch-type comment for `int64_t` accumulators.**

- [x] **P3. CMake assertion that `reference-code/` is not on the include path.** `CMakeLists.txt` now iterates `glyph`'s `INCLUDE_DIRECTORIES` at configure time and hard-errors if any entry matches `reference-code`.

---

## Work log

(Entries appended in order as fixes land. Each entry: what changed, what was verified, what was red-teamed.)

### T4 + NEW2 — mul saturation, add/sub/vec_add saturation

Changes:
- `glyph_mtfp_mul` now saturates at `±GLYPH_MTFP_MAX_VAL` after the int64 rescale. Rounding is preserved.
- `glyph_mtfp_clamp64` introduced as a public inline helper (`glyph_mtfp.h`), replacing two local duplicates.
- `glyph_mtfp_add` / `_sub` now route through `glyph_mtfp_clamp64` on an int64 intermediate.
- `glyph_mtfp_neg` left alone (safe for in-range inputs: `|-MAX_VAL| = MAX_VAL`).
- NEON `vec_add` / `vec_add_inplace` now clamp: `vaddq_s32` → `vminq_s32` → `vmaxq_s32` against splats of `±MAX_VAL`. Zero perf cost on M4.
- Three new tests: `test_mul_boundary`, `test_add_sub_saturation`, `test_vec_add_saturation`.

Verification:
- All new tests fail on the pre-fix code (confirmed by running the boundary test first and seeing `got 2147483646 expected 1073741823`), pass after the fix.

Red-team of the fix:
- The ÷2 in `MAX_VAL = INT32_MAX/2` is what allows non-saturating `vaddq_s32` to not wrap before the min/max clamp. This dependency is now documented in `glyph_types.h`.
- Scalar path uses int64 intermediate, so there's zero risk of wrap regardless of operand values.
- Composition check: `bias_add` calls `vec_add_inplace` internally, so it inherits saturation for free.
- `vec_scale` calls `glyph_mtfp_mul` scalar-per-element; already saturating through composition.

### T2 + T6 + T7 — matmul coverage

Changes: five new tests covering dense `matmul` / `matmul_bt`, ternary NEON-tail boundary at K=17, and ternary multi-row at M=3.

Verification: all pass on first run.

### A1 + A5 + S1 + S5 + S6 + S8 + P3 — quality pass

Changes:
- `src/glyph_internal.h` introduced as private platform-macro home. `GLYPH_HAS_NEON`, `GLYPH_HAS_DISPATCH`, `GLYPH_SERIAL_ROW_THRESHOLD = 4`.
- Four `dispatch_apply` sites now check `M >= threshold` and fall through to serial for small problems.
- `glyph_internal.h` `#error`s if NEON is unavailable.
- Local `clamp_mtfp` / `clamp_mtfp_i64` deleted; all callers now use `glyph_mtfp_clamp64` from the public header.
- Precondition `assert()`s added to public matmul and layernorm entry points.
- `GLYPH_MTFP_MAX_VAL` rationale is now a six-paragraph comment in `glyph_types.h`.
- Test locals converted to `enum` constants to eliminate the `-Wgnu-folding-constant` warnings.
- `CMakeLists.txt` asserts that `reference-code/` is not on glyph's include path.

Verification: all tests still pass; build is warning-free.

### C3 — LayerNorm overflow (documentation resolution)

Changes: none to code. Added invariant analysis: `|centered| ≤ sqrt((N-1)·var)` and `rstd ≈ S²/sqrt(var)` together give `|centered · rstd| ≤ sqrt(N-1) · S²`. For N = 10000, that's 3.49e11, well under 4% of `INT64_MAX`. The "pathological" overflow identified in the red-team requires physically inconsistent inputs (simultaneously tiny variance and huge deviations) that cannot occur when `centered` is actually computed from the row.

### NEW1 + C4 + T5 — LayerNorm rounding, zero-cols guard, symmetric-row test

Changes:
- `layernorm_row` now rounds `norm * weight` half-away-from-zero (`(num >= 0 ? +S/2 : -S/2) / S`), matching `glyph_mtfp_mul`.
- `glyph_mtfp_layernorm` early-returns on `rows <= 0 || cols <= 0`.
- Three new tests: `test_layernorm_symmetric_row` (integer-derived golden), `test_layernorm_symmetric_row_with_bias`, `test_layernorm_zero_cols` (no-crash).

Derivation of the symmetric-row golden: `x = [-3S, -S, +S, +3S]` → mean 0, var `5 S²`, `isqrt(5 S²) = 132037`, `rstd = S² / 132037 = 26407`, `norm[k] = k · 26407`. With `weight = S`, `bias = 0`, `scaled = norm` only if the scale multiply rounds symmetrically.

Verification:
- New tests pass on the fixed code.
- Temporarily reverted the fix and re-ran: `test_layernorm_symmetric_row` fails with `y[0] = -79220` (expected `-79221`) and `y[1] = -26406` (expected `-26407`) — exact predicted bug signature. Restored the fix.

Red-team of the fix:
- Symmetry probed at tie boundaries (`num = ±29524, ±29525, ±88573, ±88574`). Round-half-away-from-zero is preserved across sign for every case.
- `test_layernorm_constant_row` (zero-variance path) still passes: `centered = 0` makes `num = 0`, which both branches route through the same `0 / S = 0`.
- Guard position (`rows <= 0 || cols <= 0`) returns before touching any pointer, so the caller can pass `NULL` safely in the empty-range degenerate case.

