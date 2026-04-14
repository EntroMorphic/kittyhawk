---
title: Remediation Plan — First-Light Red-Team Findings
opened: 2026-04-14
scope: commit 95f5bab ("First light — minimal MTFP core")
source: red-team conducted 2026-04-14 against the first-light increment
---

# Remediation Plan

Tracks findings from the red-team of the first-light rebuild increment. Each item has a severity, specific remediation, and completion criterion.

Severity: **H** (high — blocks real spec progress), **M** (medium — correctness or claim integrity), **L** (low — hygiene).

---

## H1. The rebuild didn't rebuild the shape — it rebuilt the documentation

**Finding.** The spec's load-bearing novelty is per-block exponent as sidecar metadata, with the 16-byte block as the atomic unit. First-light has no block type, no block-native primitives, no SoA tensor abstraction. `m4t_mtfp_t` is still `int32_t` operated on in cell-native vector ops.

**Remediation.**
- [x] Define block-native atomic primitives `m4t_mtfp_block_add` / `_sub` that operate on exactly `M4T_MTFP_CELLS_PER_BLOCK` cells (one NEON vector).
- [x] Rewrite `vec_add_inplace` / `_sub_inplace` as compositions: loop of block ops + scalar tail.
- [x] Add `_Static_assert(sizeof(m4t_mtfp_t) * M4T_MTFP_CELLS_PER_BLOCK == M4T_BLOCK_BYTES, ...)`.

**Complete when.** Block-native primitives exist as the substrate's atomic unit; vec ops are compound.

---

## H2. The new primitives have zero direct tests

**Finding.** `vec_zero`, `vec_add_inplace`, `vec_sub_inplace`, `clamp64` exercised only transitively through `test_m4t_route.c`. No direct tests for boundary saturation, n=0, scalar-tail, aliasing, NEON/scalar equivalence.

**Remediation.**
- [x] Create `m4t/tests/test_m4t_mtfp.c`.
- [x] Test `clamp64` at ±MAX_VAL, ±(MAX_VAL+1), 0, INT64 extremes.
- [x] Test `vec_zero` at n=0, 1, 4, 7, 16, 1024.
- [x] Test `block_add` / `_sub`: positive saturation, negative saturation, all-zero, mixed-sign, identity, aliasing.
- [x] Test `vec_add_inplace` / `_sub_inplace`: NEON-only (n=4), scalar-only (n=3), mixed (n=5, 7), empty (n=0), large (n=1024), aliasing.
- [x] Register in CMake, confirm pass.

**Complete when.** New binary passes with ≥15 distinct assertions.

---

## H3. §8.5 "widen, don't round" silently doesn't apply to fixed-output vec ops

**Finding.** `vec_add_inplace` outputs MTFP19; widening to MTFP39 would break the output type. Current code saturates. Saturation is neither widening nor rounding — the spec didn't name this case.

**Remediation.**
- [x] Edit `m4t/docs/M4T_SUBSTRATE.md` §8.5 to distinguish three cases: widen (output admits wider cell), saturate (fixed output), round (named cross-block opt-in only).
- [x] Specify saturation is *informative* (sets status flag under §14.4 opt-in), not silent.
- [x] Cross-reference §8.5 from the new block-primitive headers.

**Complete when.** Spec names saturation explicitly; code matches spec without contradiction.

---

## H4. Consumer-inference instead of consumer-choice

**Finding.** Primitives selected by grep of kept-file callers, not by naming a consumer. Same failure mode as the prior dev's drift.

**Remediation.**
- [x] Edit `docs/THESIS.md` §3 to name `tools/mnist_trit_lattice.c` as the provisional primary consumer.
- [x] Add a `consumer-demand` line to each new block-primitive header citing the consumer.
- [x] Update `feedback_working_style.md` memory: "primitive selection by grep" is a failure mode.

**Complete when.** Current consumer named; every new primitive cites consumer demand.

---

## M1. Block-geometry constants are orphaned

**Remediation.**
- [x] Use `M4T_MTFP_CELLS_PER_BLOCK` in block-primitive signatures (array size).
- [x] Use `M4T_BLOCK_BYTES` in a `_Static_assert`.
- [x] `_Static_assert(M4T_MTFP_CELLS_PER_BLOCK == 4)` in `test_m4t_mtfp.c`.

**Complete when.** No added constant unused.

---

## M2. Same-block contract is aspirational, not enforced

**Remediation.**
- [x] Block-native primitives enforce same-block by construction (signature takes exactly one block).
- [x] Vec primitives document their single-tensor contract in the header.
- [x] *Deferred:* block-aware tensor type. Not implemented until a consumer drives it. Note in substrate §14.

---

## M3. Overstated commit claims

**Remediation.**
- [x] Fix unused-variable warning in `tools/mnist_trit_lattice.c:155`.
- [x] Add note to `feedback_working_style.md`: commit claims distinguish "builds" / "tests pass" / "measured faster."
- [x] *Deferred:* NEON benchmark in `m4t/tools/`. Add when a consumer's performance matters.

**Complete when.** LSH tool compiles with zero warnings; future commits observe the builds/tests/measured distinction.

---

## M4. `vec_add_inplace` trust-boundary

**Remediation.**
- [x] Document in-range input precondition in the header.
- [x] `_Static_assert((int64_t)M4T_MTFP_MAX_VAL * 2 < INT32_MAX, ...)`.
- [x] *Deferred:* debug-mode bounds-check. Not in this pass.

**Complete when.** Precondition named; compile-time assert catches config drift.

---

## L1. Aliasing (dst == a) undocumented

**Remediation.**
- [x] Document that `vec_add_inplace(dst, dst, n)` computes `dst[i] *= 2` with saturation; `_sub_inplace` → zeros.
- [x] Aliasing test case in `test_m4t_mtfp.c`.

---

## L2. `clamp64` `static inline` in header

**Resolution.** No action. `static inline` is correct for a 4-instruction function. Logged so it isn't re-raised.

---

## L3. Scalar tail and NEON use different clamp implementations

**Remediation.**
- [x] Test exercising both paths with inputs that hit the saturation boundary (n=5: NEON + scalar tail).

---

## Execution order

Doc edits first (fast, low risk), then code + tests:

1. H3 + H4 doc edits.
2. M3 warning fix.
3. Memory update.
4. H1 + M1 + M2 + M4 + L1 + L3: block primitives, vec rewrite, static asserts, header docs.
5. H2 + L1 test binary.
6. Build + test + commit.

---

## Completion

Check off items as they land. When all unchecked items (non-deferred) are done, the remediation is complete and the increment can be claimed as spec-realizing rather than legacy-compatible.

---

# Second red-team round (2026-04-14, full-surface)

A comprehensive red-team of all live code after first-round remediation. All first-round items remained closed; the second round surfaced additional drift and latent cliffs.

## H-RT1. Documentation drift — fixed-point framing in derivative headers

**Finding.** `m4t_types.h` was reframed to mantissa/per-block-exponent, but `m4t_mtfp4.{h,c}`, `m4t_ternary_matmul.h`, and `tools/mnist_trit_lattice.c` still use the rejected "real = cell / SCALE" language. Every reader of those files absorbs the collapsed model.

**Remediation.**
- [x] Rewrite `m4t_mtfp4.h` top comment in mantissa/block-exponent terms. Drop the "real = cell / 9, range ±4.44" lines.
- [x] Rewrite `m4t_ternary_matmul.h` header — drop "int32 ternary fixed-point," describe MTFP19 mantissas and Case S saturation on store (reference §8.5).
- [x] Update `m4t_mtfp4.c` conversion comments to frame `SCALE_RATIO` as an inter-block-exponent offset under the default convention.
- [x] Add a one-line annotation at `mnist_trit_lattice.c:35` naming the default-block-exponent convention used for the pixel → MTFP mapping.

## H-RT2. `m4t_mtfp4_mul` does silent rounding — §8.5 violation

**Finding.** `m4t_mtfp4.h:60-65` does `((int16)a * (int16)b + SCALE/2) / SCALE` and returns MTFP4. That's Case R (round) without a named opt-in. The scalar mtfp4 primitives (`add`, `sub`, `neg`, `mul`, `mul_trit`) are exercised only by the test binary — no live consumer calls them.

**Remediation.**
- [x] Delete the unmotivated scalar primitives (`m4t_mtfp4_add`, `_sub`, `_neg`, `_mul`, `_mul_trit`). Keep `m4t_mtfp4_clamp` (used internally by `sdot_matmul_bt` and the conversions).
- [x] Remove the corresponding tests from `test_m4t_mtfp4.c`.
- [x] When a real consumer needs MTFP4 scalar arithmetic, it will earn its way back in with spec-compliant semantics (Case W widening to MTFP9 mantissa).

## H-RT3. `m4t_ternary_matmul` header doesn't reference §8.5

**Remediation.**
- [x] Header commentary on `m4t_mtfp_ternary_matmul_bt` names Case S saturation on store and references §8.5.

## H-RT4. `test_proj_buf[4096]` silent cliff in LSH tool

**Finding.** `mnist_trit_lattice.c:155` has a 16 KB stack array that caps `N_PROJ` at 4096 with no assertion. Current sweep reaches 2048; larger N_PROJ would produce silent stack corruption.

**Remediation.**
- [x] Replace with a malloc'd buffer sized by N_PROJ, or add an explicit assertion against 4096. Malloc is cleaner.

## H-RT5. `m4t_route_signature_update` truncation on means

**Finding.** `means[d] /= T` at m4t_route.c:167 is integer division (truncation toward zero). On boundary cases where `col_sum == true_mean`, the sign flips. Spec §11 doesn't specify rounding for the compound op. Behavior is consistent but undocumented.

**Remediation.**
- [x] Document the truncation behavior explicitly in the header and file comment; cross-reference from §11 in the substrate spec as a clarification.

## M-RT1. `m4t_ternary_matmul` uses `vmulq_s32` over {-1, 0, +1} signs

**Finding.** The inner loop at m4t_ternary_matmul.c:100-103 multiplies MTFP19 activations by signs in {-1, 0, +1}. Hardware-native shape is bit-select + conditional negate (`vbslq_s32` / `vnegq_s32`), not multiply. The header claims "zero multiplies" but relies on the compiler to do the reduction.

**Remediation.**
- [x] Replace `vmulq_s32` with explicit `vbslq_s32`-based zero selection plus `vnegq_s32` for the -1 case, so the hardware-native shape is expressed in source. Measured speedup deferred; correctness must be preserved.

## M-RT3. `M4T_ROUTE_MAX_T` hard-coded internally

**Remediation.**
- [x] Promote `MAX_T = 64` to a public constant in `m4t_route.h` alongside `M4T_ROUTE_MAX_DIM`.

## M-RT4. `m4t_route_signature_update` has a 4 KB stack buffer

**Remediation.**
- [x] Replace `m4t_trit_t row_buf[4096]` with `malloc(D)`. Remove the `M4T_ROUTE_MAX_DIM` assertion on this path; the caller owns upper bound.
- [x] `M4T_ROUTE_MAX_DIM` becomes informational (documents the previous stack cap as a historical reference) or gets removed if no other path uses it.

## M-RT7. `SCALE_RATIO` safety proof not enforced

**Remediation.**
- [x] Add `_Static_assert((int64_t)M4T_MTFP4_MAX_VAL * SCALE_RATIO <= (int64_t)M4T_MTFP_MAX_VAL, ...)` in `m4t_mtfp4.c`.

## M-RT8. `trit_to_code` silently maps out-of-range inputs to 0

**Finding.** `m4t_trit_pack.c:28`: values outside {-1, 0, +1} become code 0 (zero trit) without warning. Could hide bugs in trit generators.

**Remediation.**
- [x] Add an `assert(t >= -1 && t <= 1)` at the top of `trit_to_code`. This is a debug-mode check only; `NDEBUG` releases skip it. Fails loudly when a generator is broken.

## L-RT4. Unused `glyph_mtfp_w_t` alias

**Remediation.**
- [x] Remove from `src/glyph_types.h`. If a consumer needs it, it re-emerges with a named demand.

## L-RT5 / L-RT7. Orphan tools not in CMake

**Remediation.**
- [x] Add `m4t_trit_golden` as a dev-only executable in `m4t/CMakeLists.txt` (gated by a `M4T_BUILD_TOOLS` option, default OFF).
- [x] Do the same for `m4t_lut_gen` — it is the sanctioned build-time-float tool; it should build cleanly when requested.

## L-RT6. No `-Werror`

**Remediation.**
- [x] Add `-Werror` to the m4t CMake compile options. Warnings become build failures.

## Deferred (second round)

- [ ] **M-RT10.** LSH regression test. Tracked — lands when a cheap synthetic smoke test can exercise the full pipeline without MNIST data.
- [ ] **T-RT2 / T-RT3.** Broader `signature_update` edge cases and ternary_matmul near-saturation tests. Lands in a dedicated test-expansion pass.
- [ ] **T-RT4.** End-to-end LSH regression (same as M-RT10).
- [ ] **T-RT5.** Explicit §8.5 Case-semantic assertions in tests. Annotate existing tests with Case labels rather than adding new ones.

## Execution order (second round)

1. Documentation sweep: H-RT1, H-RT3, H-RT5, N-RT* (single pass).
2. Delete unmotivated primitives: H-RT2, L-RT4 (and update tests).
3. Cliffs and asserts: H-RT4, M-RT4, M-RT7, M-RT8.
4. Expose constant: M-RT3.
5. Bit-select rewrite: M-RT1.
6. Build hygiene: L-RT5, L-RT6, L-RT7.
7. Build + test + commit.
