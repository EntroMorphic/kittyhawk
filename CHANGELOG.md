# Changelog

All notable changes to this project are recorded here. The project is pre-1.0 and in active rebuild; there are no tagged releases yet. Format follows [Keep a Changelog](https://keepachangelog.com).

The first entry below marks the ground-zero rebuild that restructured the substrate. Changes before the rebuild are summarized at the bottom; their detail lives in git history and in `journal/`.

---

## [Unreleased] — Ground-zero rebuild (2026-04-14 →)

Triggered by a full audit that identified a collapse of Multi-Trit Floating Point into a fixed-point reading with a shared global scale, and a substrate drifting toward dense computation over base-3 hardware. The rebuild restores MTFP as base-3 floating point (mantissa cells + per-block exponent) and puts routing primitives first.

### Added

**Documents.**
- `NORTH_STAR.md` — compass document. Why base-3, why routing, what the end-game is not.
- `docs/THESIS.md` — thesis brief: falsification criteria, provisional primary consumer (`tools/mnist_trit_lattice.c`), benchmark bed as an open empirical question, hardware-alignment measurement as a future discharge item.
- `docs/REMEDIATION_PLAN.md` — two rounds of red-team findings tracked against completion status.
- `m4t/docs/M4T_SUBSTRATE.md` — canonical 16-section substrate spec. §17 added later as spec-to-code cross-reference.
- `archive/README.md` — orientation for archived code: what's there, why, what might come back under named consumer demand.
- `LICENSE` — MIT. Previously claimed in README but the file was missing.
- `CHANGELOG.md` — this file.

**Code.**
- `m4t/src/m4t_mtfp.{c,h}` — block-native MTFP19 primitives: `block_add` / `block_sub` (exactly one NEON vector each), composed into `vec_add_inplace` / `vec_sub_inplace` / `vec_zero` with scalar tails. Saturating `clamp64` for accumulator stores. Same-block contract; §8.5 Case S.
- `m4t/tests/test_m4t_mtfp.c` — 24 direct assertions covering clamp64, vec_zero, block_add/sub saturation in both directions, aliasing, and NEON/scalar path equivalence.
- `M4T_BLOCK_BYTES`, `M4T_MTFP{4,9,,W}_CELLS_PER_BLOCK` in `m4t_types.h`, enforced by a `_Static_assert` against the 16-byte invariant.
- `M4T_ROUTE_MAX_T = 64` promoted to public constant in `m4t_route.h`.
- `M4T_BUILD_TOOLS=ON` CMake option — builds `m4t_trit_golden` (truth-table enumerator) and `m4t_lut_gen` (the only sanctioned binary-float code in the ecosystem; runs at build time, not linked into `libm4t.a`).
- `_Static_assert` on `SCALE_RATIO × MTFP4_MAX ≤ MTFP19_MAX` in `m4t_mtfp4.c`.

**Research.**
- `journal/seven_open_decisions_{raw,nodes,reflect,synthesize}.md` — LMM cycle triaging the spec's §14 opens; six of seven dissolved under triage.
- `journal/sdf_and_ternary_lessons.md` — postmortem of the SDF pivot that introduced Law #7 ("ternary projections apply to MTFP data, not ternarized data").

### Changed

- **MTFP vocabulary.** Reframed from fixed-point to base-3 floating point: a value is `mantissa × 3^exponent`, with the exponent as sidecar block metadata. `SCALE` and `RADIX` constants survive as "default block-exponent convention" for legacy consumers, not as type properties.
- **§8.5 invariant.** Three resolution cases explicitly named: **W**iden (output type admits a wider cell), **S**aturate (fixed-output type), **R**ound (named opt-in for cross-block alignment, not the default). Every op's contract cites which case applies.
- **`m4t_ternary_matmul` inner loop.** Replaced `vmulq_s32` over decoded signs in {-1, 0, +1} with `vbslq_s32` + `vnegq_s32`. Multiplication by a sign was a base-2 shortcut through a general opcode; the base-3-native expression is a mask and a conditional negate.
- **Stack-buffer cliffs removed.** `m4t_route_signature_update` row buffer (stack[4096] → `malloc(D)`). `tools/mnist_trit_lattice.c` test projection buffer (stack[4096] → `malloc(N_PROJ)`). Both paths now have no artificial dimension cap.
- **Root `README.md`.** Rewritten to reflect current state. Stale metrics (97.61% k-NN, 81.40% LSH, 17.7 KB .text) removed — they were measured against the pre-rebuild substrate.
- **`m4t/README.md`.** Rewritten. Lists only the live primitive surface.
- **CMake.** `-Werror` added. `m4t_trit_golden.c` cleaned up for strict-warnings compliance.

### Removed

- `m4t_mtfp4_add`, `_sub`, `_neg`, `_mul`, `_mul_trit` — unmotivated scalar inlines (test-only callers; `_mul` silently rounded, violating §8.5). `m4t_mtfp4_clamp` kept (used internally by SDOT matmul and width conversions).
- `glyph_mtfp_w_t` alias — no live consumer.
- `M4T_ROUTE_MAX_DIM` from the substrate contract — caller now owns upper bound.
- `MTFP21` compatibility constants — no live consumer.

**Moved to `archive/`** (retained for historical reference, not on build path):
- `m4t_mtfp.{c,h}` pre-rebuild (dense matmul, LayerNorm, bias, fan_in_normalize bundled with element-wise arithmetic).
- `m4t_mtfp_w.{c,h}` — MTFP39 wide-cell arithmetic (dense path; no routing consumer).
- `m4t_mtfp_nonlinear.c` + `m4t_mtfp_tables.c` — GELU/softmax LUTs (dense-transformer consumers).
- `m4t_ops.{c,h}` — function-pointer dispatch table (mixed dense/routing; needs pruning to return).
- `m4t_bench.c` — benched the dense path.
- `test_m4t_smoke.c`, `test_m4t_mtfp_w.c`, `test_m4t_ops.c`.
- `glyph_mtfp.h`, `test_glyph_wrapper.c`.
- `mnist_knn_lattice.c`, `mnist_m4t_infer.c`, `mnist_train_dump.c`.
- `reference-code/` — original trix-z C kernels (contained float paths).
- Pre-rebuild design docs: `M4T_CONTRACT.md`, `M4T_PIPELINE.md`, `M4T_BEYOND.md`, `M4T_REDTEAM.md`, `TRIT_LATTICE_LSH.md`.
- Pre-rebuild remediation artifacts: `REMEDIATION_PLAN.md` (trix-z era), `REDTEAM_FIXES.md`.

### Fixed

- `m4t_trit_pack.c::trit_to_code` — now asserts input is in {-1, 0, +1}. Previously mapped out-of-range inputs silently to zero, masking bugs in trit generators.
- `tools/mnist_trit_lattice.c` — unused `tproj` variable (silent warning for an unknown duration).
- Documentation drift across derivative files: `m4t_mtfp4.{h,c}`, `m4t_ternary_matmul.h`, MNIST tool, all reframed from "real = cell / SCALE" fixed-point language to mantissa/block-exponent.
- `m4t_route_signature_update` header — now documents the integer-division (truncation) behavior on means.

### Verified

- `cmake --build` green under `-Wall -Wextra -Wpedantic -Werror`.
- `ctest`: 5/5 test binaries pass (`m4t_mtfp`, `m4t_trit_ops`, `m4t_trit_reducers`, `m4t_route`, `m4t_mtfp4`).
- `M4T_BUILD_TOOLS=ON` builds `m4t_trit_golden` and `m4t_lut_gen` cleanly.
- `tools/mnist_trit_lattice.c` builds via the root CMake (`GLYPH_BUILD_TOOLS=ON`, default ON).

### Measured (first light on rebuilt substrate)

- **LSH on MNIST, N_PROJ=2048: 81.40%.** Bit-for-bit reproduction of the pre-rebuild baseline. See `journal/rebuilt_substrate_first_light.md`.
- Full sweep (4 projection sizes × L1 / refine-3 / refine-5) wall clock: 41.6 s, single core.
- The `m4t_ternary_matmul` bit-select rewrite preserved consumer numerics exactly. No silent regression from the base-2-shortcut → base-3-native shape transition.

### Deferred (tracked in `docs/REMEDIATION_PLAN.md`)

- Block-aware tensor type carrying an exponent array (M2). Lands with the first consumer that needs cross-block exponent tracking.
- NEON benchmarking (M3). Lands with a consumer whose performance matters.
- LSH end-to-end regression test (M-RT10 / T-RT4). Lands with a synthetic data path.
- Broader `signature_update` / near-saturation ternary_matmul test expansion (T-RT2 / T-RT3).
- Explicit §8.5-Case-semantic annotations on existing tests (T-RT5).

---

## Pre-rebuild era (through 2026-04-13)

MNIST experimentation under the collapsed fixed-point reading of MTFP.

Notable results from that era, not re-measured against the rebuilt substrate:

- Trit Lattice LSH (L1 centroid, 2048 projections): 81.40% MNIST, zero float.
- Trit Lattice k-NN (L2, 512 projections): 96.79% MNIST, zero float.
- Trit Lattice k-NN (deskewed pixels, L2): 97.61% MNIST, zero float.
- Float-trained, M4T all-ternary inference: 97.46%.

Full experimental record: `journal/full_experimental_record.md` and the companion LMM cycles (`trit_lattice_lsh_*`, `knn_atomics_*`, `lattice_findings_*`, `lingering_thoughts_*`, `ternary_opcode_*`). Git history up to commit `e412b50` ("Archive dense paths; lock substrate spec at 16B blocks") is the authoritative source for pre-rebuild work.
