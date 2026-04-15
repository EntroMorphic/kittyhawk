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

- [x] **M-RT10.** LSH regression test. Tracked — lands when a cheap synthetic smoke test can exercise the full pipeline without MNIST data.
- [x] **T-RT2 / T-RT3.** Broader `signature_update` edge cases and ternary_matmul near-saturation tests. Lands in a dedicated test-expansion pass.
- [x] **T-RT4.** End-to-end LSH regression (same as M-RT10).
- [x] **T-RT5.** Explicit §8.5 Case-semantic assertions in tests. Annotate existing tests with Case labels rather than adding new ones.

## Execution order (second round)

1. Documentation sweep: H-RT1, H-RT3, H-RT5, N-RT* (single pass).
2. Delete unmotivated primitives: H-RT2, L-RT4 (and update tests).
3. Cliffs and asserts: H-RT4, M-RT4, M-RT7, M-RT8.
4. Expose constant: M-RT3.
5. Bit-select rewrite: M-RT1.
6. Build hygiene: L-RT5, L-RT6, L-RT7.
7. Build + test + commit.

---

# Third red-team round (2026-04-14, post sign_extract → threshold_extract)

Red-team of commit `ea0e519` (the sign_extract replacement). Subsequent LMM cycles and doc consolidation. Six new findings.

Severity key unchanged: **H**, **M**, **L**.

## M-RT3A. §18 applies cleanly only to enumerable-output primitives; doesn't say so

**Finding.** §18's "every output state must be emitted non-trivially" is well-defined for primitives whose output space is finite and enumerable (trit codes, decision signs). For primitives with continuous outputs (`distance_batch` returns int32 distances; `apply_signed` returns MTFP mantissa arrays), "every output state" becomes "every int32 value" — nonsense. As written, §18 doesn't say where it applies, so applying it to continuous-output primitives leads either to confusion or to silently skipping the review gate.

**Remediation.**
- [x] Edit `m4t/docs/M4T_SUBSTRATE.md` §18 to name scope explicitly:
  - **Enumerable-output primitives** (trit-producing extractors, decision primitives): emission coverage applies to the OUTPUT space.
  - **Continuous-output primitives whose behavior depends on trit structure at the INPUT** (distance_batch over packed trits, apply_signed consuming decision-sign trits): emission coverage applies to the INPUT space.
  - **Primitives outside the trichotomy** (pure arithmetic on integer mantissas, conversions, etc.): §18 does not apply.
- [x] Update §18.1 examples to illustrate both input-side and output-side coverage.

**Complete when.** §18 names its own scope; every live primitive has a clear answer for which side of the criterion (output / input / not applicable) applies.

## M-RT3B. topk_abs and apply_signed docstrings not updated

**Finding.** The scrutiny-cycle synthesize committed to updating docstrings on `topk_abs` and `apply_signed` to name their sanctioned input classes under §18. The commit shipped without these updates. `topk_abs` has a three-state output (sign field); `apply_signed` consumes a three-state input (decision sign). Both want explicit contracts.

**Remediation.**
- [x] `m4t_route.h::m4t_route_topk_abs` docstring additions: enumerated output space (tile_idx, sign ∈ {+1, -1, 0-sentinel}); sanctioned input class (score arrays with mixed nonzero signs, k ≤ T); coverage-test pointer.
- [x] `m4t_route.h::m4t_route_apply_signed` docstring additions: enumerated input decision-sign space; sanctioned input class (decisions from `topk_abs` output); coverage-test pointer.

**Complete when.** Both primitives' headers carry the §18 review-gate data: enumerated space, sanctioned input class, coverage-test pointer.

## M-RT3C. Coverage tests are not labeled as such

**Finding.** The tests that serve the §18 review-gate coverage role are not labeled. `test_threshold_extract_tau5` IS the coverage test for threshold_extract's primary sanctioned deployment, but a future auditor has to reverse-engineer that. Same gap for `test_topk_abs_*`, `test_apply_signed_*`, `test_signature_update`. Machine-greppable labels fix this.

**Remediation.**
- [x] Add `/* §18 coverage test: ... */` header comments above each test function that serves the coverage role in `test_m4t_route.c`.
- [x] For compound tests (e.g., topk_abs coverage spans multiple test functions), a single group comment at the top of the related tests block.

**Complete when.** Every §18 coverage test is labeled with a comment a `grep "§18 coverage"` can find.

## L-RT4A. Meta: re-audit pass not documented

**Finding.** The scrutiny-cycle synthesize committed to "Re-audit all routing primitives against emission coverage before declaring the architecture correct." I did the audit implicitly (§17 cross-reference includes the §18 row), but did not produce a visible audit trail for each primitive. A future auditor can't see which primitive was evaluated against which side of the criterion.

**Remediation.**
- [x] Add a subsection to §18 (or a new §18.5) enumerating all live routing primitives and their §18 status: output-side vs input-side vs not-applicable; coverage-test reference for each.

**Complete when.** `m4t/docs/M4T_SUBSTRATE.md` contains an explicit per-primitive §18 audit table.

## L-RT1A. `-tau` UB safety is assertion-dependent

**Finding.** `if (v < -tau)` is undefined behavior if `tau == INT64_MIN`. The `assert(tau >= 0)` catches this in debug; in release, a contract-violating caller gets UB.

**Remediation (accepted risk).**
- [ ] *Defer.* This is the standard C pattern: runtime assertion in debug, precondition documented, contract-violating callers get UB in release. Fixing it would either add a runtime guard (hides bugs) or require restructuring arithmetic (adds complexity for a hypothetical caller error). The paranoid fix does not improve correctness materially; the contract is stated and the assertion is in place.

## L-RT2A. Test coverage gaps

**Finding.** Threshold_extract tests don't cover: `n = 0`; values near INT64_MAX (large positive); packed-byte boundaries (n = 3, 5, 7, 8 — to exercise the pack-bit OR across byte edges).

**Remediation.**
- [x] Add `test_threshold_extract_n_zero` — `n = 0` returns cleanly; no writes to dst.
- [x] Add `test_threshold_extract_pack_boundaries` — n ∈ {3, 5, 7, 8}, verify correct placement across packed-byte edges.
- [x] Add `test_threshold_extract_extremes` — values including INT64_MAX, INT64_MIN+1 (avoid INT64_MIN itself per L-RT1A); verify sign discrimination.

**Complete when.** All three tests land and pass.

## L-RT3A. glyph_route.h untested (pre-existing)

**Finding.** Pre-existing state. The wrapper test was archived with the dense-path aliases. `glyph_route.h` aliases exist but are never compiled under the current build. Not a regression from recent changes.

**Remediation (deferred).**
- [ ] *Defer.* Untested glyph wrapper lands back in scope when a glyph consumer emerges. Until then, aliases remain theoretical.

---

## Execution order (third round)

1. §18 scope qualifier and per-primitive audit (M-RT3A, L-RT4A). Doc-only.
2. Docstring updates on `topk_abs` and `apply_signed` (M-RT3B). Doc-only.
3. Coverage-test labels (M-RT3C). Comments in test file.
4. Test coverage additions (L-RT2A). New tests + register.
5. Build + test + commit.

L-RT1A and L-RT3A explicitly deferred with rationale.

---

# Fourth red-team round (2026-04-14, post routed-knn 97.31% claim)

Red-team of commit `663c355` (the "routed 97.31% beats dense 97.05% by 10.8×" headline). Findings qualify but don't invalidate the core result. Seven items across three severity levels.

## H-RT1D. Speed comparison is apples-to-oranges (scalar L1 vs NEON popcount)

**Finding.** `mnist_routed_knn.c`'s L1 k-NN baseline uses a scalar `int64_t` inner loop with no SIMD; `m4t_popcount_dist` is hand-NEON. The 10.8× speedup conflates algorithmic win (popcount-over-bits vs abs-diff-sum per dim) with SIMD deployment. The raw claim inflates the algorithmic edge.

**Remediation.**
- [ ] Rewrite the L1 k-NN inner loop with NEON (`vabdq_s32` + `vaddw_s32` widening accumulate to int64). Lanes: 4 int32 absolute differences per vector, widen-add into two int64x2_t accumulators, horizontal reduce at the end.
- [ ] Re-run the comparison with the vectorized baseline.
- [ ] Update the reported speedup — expected ~3-5× once L1 is vectorized (popcount processes 128 bits per VCNT; abs-diff-sum processes 4 int32s per vabdq; algorithmic ratio is ~5-10× but real overhead compresses this).

**Complete when.** L1 baseline is NEON-vectorized; routed vs L1 wall time reflects only the algorithmic difference, not SIMD deployment asymmetry.

## H-RT2D. Accuracy gap is ~1.5σ; single-run, not statistically significant

**Finding.** 97.31% vs 97.05% at n=10 000 is a 26-sample difference. Binomial standard error at p≈0.97, n=10K is ≈0.17%. The 0.26% gap is ~1.5σ — suggestive, not decisive. One RNG seed isn't enough to distinguish "routing really wins" from "this run's random projection happened to favor routing."

**Remediation.**
- [ ] Run at least 5 RNG seeds per configuration.
- [ ] Report mean ± stddev for each (N_PROJ, k) cell in the results table.
- [ ] Apply honest interpretation: if routed's mean minus L1's mean is within one stddev, call it a tie. Otherwise claim the gap as measured.

**Complete when.** Every reported accuracy has a mean ± stddev over multiple seeds; no single-run numbers quoted as headlines.

## M-RT1D. Best-of-six selection bias on the 97.31% headline

**Finding.** 6 configurations ran for routed (2 N_PROJs × 3 k values). 97.31% was the best of 6. Best-of-k from noise-distributions with σ≈0.17% biases the maximum upward by ~0.3-0.5%. The honest out-of-sample routed number is probably ~97.0% ± 0.2%.

**Remediation.**
- [ ] Report all configurations tested, not just the winner. Include the full grid in the output table.
- [ ] Caveat the "best" number with explicit selection-bias acknowledgment.

**Complete when.** Every reported accuracy is contextualized with the sweep it was the maximum over.

## M-RT2D + M-RT3D. Dense baseline is too weak; no deskewing tested

**Finding.**
- The "L1 k-NN (MTFP19 projections)" baseline in the current tool isn't the strongest classical dense baseline. It's L1 over *the same ternary projections* as the routed path — isolates the classifier contribution but isn't "dense ML wins on this task."
- The stronger dense baseline is k-NN over deskewed pixels (journal: 97.61%). We haven't re-measured this on the rebuilt substrate.
- Deskewing boosts accuracy by ~1.5-2 points. The routed path hasn't been run on deskewed inputs either.

**Remediation.**
- [ ] Port `deskew_image` / `deskew_all` from `archive/tools/mnist_knn_lattice.c` into the k-NN tool.
- [ ] Add a deskewed-pixel dense k-NN baseline (L1 distance over 784-dim MTFP pixel vectors, k ∈ {1, 3, 5}).
- [ ] Add an optional deskewing pass for the routed path (deskew pixels → project → extract signature → k-NN).
- [ ] Report all four tracks side-by-side: raw-proj L1, raw-proj routed, deskewed-pixel L1, deskewed-proj routed.

**Complete when.** Results include both deskewed and non-deskewed variants on both the dense (pixel + projection) and routed paths.

## M-RT4D. Zero density verified but not full trit distribution

**Finding.** We verified train %zero = 32.87%, test %zero = 32.58% — close to 33%. We did NOT verify that the nonzero trits split ~50/50 between +1 and -1. If the distribution is skewed (e.g., 33% zeros, 40% +1, 27% -1), §18 passes at emission-coverage but the base-3 distribution isn't symmetric.

**Remediation.**
- [ ] Instrument to count +1, 0, -1 trits separately.
- [ ] Report the full three-way distribution on both train and test signatures.
- [ ] Verify ~33/33/33 split (or document the actual split if asymmetric).

**Complete when.** Output shows %+1, %0, %-1 distributions for each signature population; symmetric-trichotomy confirmed or the asymmetry explicitly documented.

## L-RT1D. Memory 600+ MB at N_PROJ=2048 (hygiene)

**Finding.** `train_proj` alone is 480 MB. With `test_proj` and signatures, peak memory is ~600 MB. Fine on modern hardware but limits the tool on tighter machines.

**Remediation (accepted).**
- [ ] *Defer.* Acceptable for a research tool; memory is not the binding resource. A streaming-projection variant is straightforward if a tighter consumer arises.

## L-RT2D. Document revised claims everywhere

**Finding.** The prior `journal/routed_knn_mnist.md`, `CHANGELOG.md`, and `README.md` all carry the overclaimed "routing beats dense by 10.8×" framing. Once the remediation measurements are in, every doc that quotes the headline needs correcting.

**Remediation.**
- [ ] Update `README.md` Results table with multi-seed, vectorized-baseline, deskewed numbers.
- [ ] Add a correction entry to `CHANGELOG.md` retracting the overclaim with revised measurements.
- [ ] Append a "Revised after fourth red-team" section to `journal/routed_knn_mnist.md` with the new numbers; keep the prior text for historical record.

**Complete when.** No repo document quotes the 10.8× speedup or 97.31% as "beats dense by X points" without the seed-variance and baseline-fairness qualifiers.

---

## Execution order (fourth round)

1. Update `mnist_routed_knn.c`: add vectorized L1, deskewing, multi-seed loop, ±1 trit count, full-sweep output.
2. Rebuild.
3. Run the new configuration. Capture full results.
4. Update docs with honest revised claims (README, CHANGELOG, journal entry).
5. Commit everything as one cohesive correction.

Deferred: L-RT1D (memory).
