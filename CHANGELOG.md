# Changelog

Notable changes to Glyph since the 2026-05-01 ground-zero rebuild. Older entries are preserved in `01MAY26_archived/CHANGELOG.md`.

## [Unreleased]

### Added
- `01MAY26_archived/` snapshot of the prior implementation (gitignored, retained on disk as reference).
- Repository scaffolding: `LICENSE`, `README.md`, `CONTRIBUTING.md`, `NORTH_STAR.md`, `.github/` (workflows, PR template, issue templates, CODEOWNERS), top-level CMake, `docs/` skeleton (`THESIS.md`, `FINDINGS.md`).
- `docs/REMEDIATION_PLAN.md` — kernel rebuild plan covering tiers 2 (route primitives + MTFP19 mantissa arithmetic) and 3 (cross-exponent kernel + MTFP4 + ternary matmul, all consumer-gated).
- `docs/REMEDIATION_PLAN_REDTEAM.md` — adversarial review of the plan; 12 findings; substantive ones (T2, T10) reshaped tier 3 to begin with a consumer-discovery cycle rather than a design memo.
- `docs/DESIGN_X-EXPO.md` — design exploration for the cross-exponent kernel. Forward-looking specification, not a build commitment.
- `journal/xexpo_design_{raw,nodes,reflect,synthesize,closeout}.md` — first LMM cycle in the rebuild, applied to the cross-exponent kernel design and its intent. Cycle surfaced two design findings: (a) the saturation-contract error bound was too tight by a factor of 3 (`≤ 3^(e_d − 1)` should be `< 3^e_d`, with truncate-toward-zero rounding rule explicitly stated), and (b) the design's primary API should be accumulator-shaped, not pairwise — the cited consumers naturally accumulate, and `m4t_route_apply_signed` is already an accumulator at the trivial `e_running == e_new` case. The cross-exp kernel is its generalization.
- **Tier 1 kernel lift** (pure base-3 layer): `m4t_types.h`, `m4t_internal.h`, `m4t_trit_pack`, `m4t_trit_ops`, `m4t_trit_reducers` + their tests. 3/3 ctest binaries green.
- **Tier 2 hygiene pass**: lifted `m4t_mtfp.{h,c}` (MTFP19 mantissa arithmetic at one shared block exponent) and `m4t_route.{h,c}` (five route primitives). Added missing input asserts on `distance_batch` (T, sig_dim) and `apply_signed` (k, dim, per-decision tile_idx and sign). Added `m4t_route_decisions_emit_coverage` helper for §18 testability. 5/5 ctest binaries green.

### Changed
- `docs/DESIGN_X-EXPO.md` — revised per `journal/xexpo_design_closeout.md`: primary API became `m4t_mtfp_vec_accum_aligning` (stateful accumulator), pairwise `vec_add_aligning` retained as thin convenience wrapper; saturation-contract bound corrected to strict `< 3^e_result` with truncate-toward-zero rounding rule named; `Δ ≥ 19` softened from hard assertion to documented degenerate behavior; property tests refit to sequence-shaped accumulator semantics.
- `docs/REMEDIATION_PLAN.md` — cycle protocol gained explicit call-pattern measurement (pairwise-vs-accumulator) with both static-analysis and API-shape-sketch evidence sources, plus a §14.2 spec re-read prerequisite.
- `CONTRIBUTING.md` — added principle 7: substrate-level specs are upstream of kernel designs.

### Deferred
- **Tier 3** (`m4t_mtfp_vec_accum_aligning` cross-exponent accumulator, `m4t_mtfp4.*` SDOT path, `m4t_ternary_matmul.*`): pending the consumer-discovery cycle. The substrate is currently *MTFP-capable, fixed-point-in-practice*. See `docs/REMEDIATION_PLAN.md` for the cycle's pre-committed decision endpoints.

## [2026-05-01 — tier 3a: cross-exponent accumulator built]

Owner-authorized direct build, skipping the consumer-discovery cycle (codified principle 5 reading: named consumer demand suffices, not measured cost). Per principle 7, the substrate spec was re-read before implementation — that re-read changed two design choices the LMM cycle had specified.

### Added
- **`m4t_mtfp_vec_accum_aligning`** (canonical) and **`m4t_mtfp_vec_add_aligning`** (pairwise wrapper) in `m4t/src/m4t_mtfp.{h,c}`. Path A alignment, **base-3 round-to-nearest** (§8.2; the original LMM design specified truncate-toward-zero — overridden by spec re-read). Per-cell status flags with `M4T_FLAG_SATURATED` (bit 0) and `M4T_FLAG_ROUNDED` (bit 1), sticky-OR'd across calls (§14.4).
- `M4T_FLAG_SATURATED` and `M4T_FLAG_ROUNDED` macros in `m4t/src/m4t_mtfp.h`.
- `m4t/tests/test_m4t_mtfp_accum_aligning.c` — six property tests at 10,000 samples each, with a bit-exact `int64` reference implementation as the oracle. No floating-point in the test path.
- `m4t/CMakeLists.txt`: register the new test as `m4t_mtfp_accum_aligning`.

### Changed
- `m4t/docs/M4T_SUBSTRATE.md` §14.2 status: DEFERRED → IMPLEMENTED. Document table updated. §8.2 reference status updated.
- `docs/DESIGN_X-EXPO.md` revised to reflect the spec-driven changes: round-to-nearest replaces truncate, flag layout includes ROUNDED bit, parameter rename `new` → `addend` (C++ portability), bit-exact reference replaces fp decode oracle.
- `m4t/README.md` "Live surface" extended with the cross-exponent accumulator section. Tier-3 surfaces split into 3a (built) and 3b (pending). Tests table extended to 6 binaries.

### Substrate status

The substrate is now **floating-point in base 3 at per-tensor exponent granularity**. The cross-exponent kernel that distinguishes MTFP from fixed-point exists, ships under property-test coverage, and honors the spec's "named opt-in for the lossy path" framing. Per-block exponent storage (§7's stated intent) remains a separate kernel deferred until a consumer asks. Tier 3b (MTFP4 SDOT and ternary matmul) remains consumer-gated.

## [2026-05-01 — tier 3a: kernel red-team remediation]

Adversarial pass over the cross-exponent accumulator surfaced 14 findings (5 high, 5 medium, 4 low). All remediated in this commit. Recorded in `journal/xexpo_kernel_redteam.md`.

### Changed (kernel and tests)

- **Flag layout (H1):** migrated from per-cell `uint8_t[]` to per-block `uint8_t[M4T_FLAG_BYTES(n)]`, matching `M4T_SUBSTRATE.md` §14.4 verbatim. Each byte encodes 2 events × 4 cells. Added helpers `M4T_FLAG_BYTES(n)` and `m4t_flag_test(flags, cell, event)`.
- **Round-to-nearest-even invariant (H2):** added compile-time `_Static_assert` on every `M4T_POW3_*` constant verifying odd-LSB; runtime `assert(s & 1)` in `m4t_pow3_round_div`. Documents and enforces that ties cannot occur.
- **Aliasing test (H3):** renamed misleading `prop_accum_aligning_aliasing` → `prop_accum_determinism`. Added genuine aliasing test `prop_add_dst_alias_a` exercising the wrapper's `dst == a` path.
- **Wrapper assertions (M5):** added `assert(dst != b)` to `vec_add_aligning` and `vec_sub_aligning`.
- **n=0 contract (M1, kernel bug discovered):** added `if (n == 0) return;` at the top of the accumulator. Original code updated `*running_exp` even with n=0; the new `prop_accum_n_zero` test caught this.
- **Coverage hardening (M1, M3, M4):** expanded property tests from 6 to 14:
  - `prop_accum_partial_block` — trailing-block flag bits past n stay zero.
  - `prop_accum_long_sequence` — 200 sequences × K=256 calls.
  - `prop_accum_boundary` — curated edge cases (MAX_VAL, 0, Δ ∈ {0,1,19,20}, n=1).
  - `prop_accum_n_zero` — n=0 no-op contract.
  - `prop_add_dst_alias_a` — wrapper aliasing.
  - `prop_add_out_e_nullable` — wrapper accepts NULL out_e.
  - Saturation-targeted RNG (`rand_mantissa_near_max`) in `prop_accum_flags`.

### Added (new kernel)

- **`m4t_mtfp_vec_sub_aligning` (L2):** pairwise subtract wrapper, sibling of `vec_add_aligning`. Negates `b` inline within the four-case structure (no temporary buffer). Property-tested via `prop_sub_via_negation` (matches `add(a, neg(b))`) and `prop_sub_self` (`sub(x, x) == 0`).

### Changed (documentation)

- `docs/DESIGN_X-EXPO.md` — flag layout section rewritten for per-block; property-test table expanded to 14 entries; subtract wrapper documented; per-cell-deviation history noted.
- `m4t/README.md` — header clarified to distinguish NEON vs scalar kernels (L3); cross-exp section updated for per-block flags + sub wrapper; `apply_signed`-as-degenerate-case framing tightened (L4) — generalizes the *arithmetic*, not the routing semantics.
- `m4t/docs/M4T_SUBSTRATE.md` §14.2 amendment expanded with the per-block layout and odd-divisor invariant; §14.4 disambiguated to "1 byte per block."
- `journal/xexpo_spec_amend.md` (H4) — lightweight synthesize cycle documenting the §14.2 + §14.4 amendments per principle 7.
- `journal/xexpo_kernel_redteam.md` — full red-team record, all findings, all remediations.

### Substrate status

Unchanged from the prior tier-3a entry: floating-point in base 3 at per-tensor exponent granularity. The remediation hardened the implementation; the substrate's overall capability is the same. 6/6 ctest binaries green from clean rebuild under `-Werror`; 14 properties pass at full sample counts.

## [2026-05-01 — tier 3b + 3c: MTFP4 SDOT and ternary matmul online]

Owner-authorized direct build of the remaining tier-3 kernels. The consumer-discovery cycle gate was overridden ("the consumer wall was holding back progress"); the substrate's full surface ships under property-test coverage.

Spec re-read (§8.3, §8.4, §8.5) before implementation per principle 7. The re-read surfaced one design correction:

### Changed
- **SDOT MTFP4 matmul (`m4t_mtfp4_sdot_matmul_bt`):** archived implementation was `MTFP4 × ternary → MTFP4` with case-S clamp on store. With K=64 and |X|=40, accumulator can reach 2560 — well over MTFP4's max of 40 — so saturation fired on basically every cell, making the kernel unusable for any real K. **Spec §8.4 specifies Case W (output widens to MTFP19, exact by construction).** The shipped kernel implements §8.4 verbatim: `MTFP4 × ternary → MTFP19`, exact for K ≤ ~14.5M. Consumers needing MTFP4 output chain `m4t_mtfp19_to_mtfp4` after the matmul.

### Added
- **`m4t_mtfp4.{h,c}`** — SDOT ternary matmul (Case W per §8.4) plus widening (`mtfp4_to_mtfp19`, exact, static-asserted bound) and narrowing (`mtfp19_to_mtfp4`, base-3 round-to-nearest-even + saturate, optional flag tracking) cell-width conversions.
- **`m4t_ternary_matmul.{h,c}`** — MTFP19 × packed-ternary matmul (Case S per §8.5). NEON-accelerated 16-trit decode + bit-select + conditional negate + int64 accumulator + saturating clamp on store. Optional per-block SATURATED flag tracking (new vs the archived version, which silently saturated).
- **Aliasing assertions** on both new kernels (`Y != X`, `Y != W`).
- **Sample-based weight-validity assertion** in the SDOT matmul (debug builds only) — catches "caller forgot to use ternary."
- **Shared `m4t_flag_or` helper** in `m4t_internal.h` — used by both the cross-exp accumulator and the ternary matmul to write per-block flag bits. Eliminates duplication between kernels.

### Tests
- **`test_m4t_mtfp4`** (10 tests): clamp boundaries, SDOT golden 2×4×3, SDOT random vs int64 reference (200 trials, K up to 1024 — exercises NEON + tail), SDOT extreme bounds (4096 cells × 40 mantissa × ±1 weight, verify no saturation), zero-dim edges, widen exact, narrow round-to-nearest, narrow saturate, narrow flags (per-block layout), widen-narrow roundtrip.
- **`test_m4t_ternary_matmul`** (6 tests): golden 2×4×3, random vs reference, saturation clamp, saturation flags (per-block layout), zero-dim, determinism.

### Build
8/8 ctest binaries green from clean rebuild under `-Werror`. All tier-3 kernels are now ONLINE; the substrate ships its full routing-first base-3 surface.

### Substrate status (final)

Tier 1 (pure base-3) + Tier 2 (route primitives + MTFP19 mantissa arithmetic) + Tier 3 (cross-exponent accumulator + SDOT MTFP4 matmul + MTFP19 ternary matmul + cell-width conversions) — **complete**. The substrate supports:
- Base-3 floating-point arithmetic at per-tensor exponent granularity.
- Hardware-native ternary matmul via SDOT (Case W exact MTFP4 → MTFP19).
- Wider-precision matmul via the MTFP19 × ternary path (Case S saturating).
- Bidirectional cell-width conversion.
- Full §14.4 status flag tracking on every Case-S/Case-R kernel.

What remains: consumer-side rebuild (libglyph, libtrain, tools) — these are separate plans, scoped outside this commit.

## [2026-05-01 — tier 3b/3c red-team remediation]

Adversarial pass over the SDOT MTFP4 matmul, cell-width conversions, and MTFP19 ternary matmul surfaced 11 findings (2 high, 5 medium, 4 low). All remediated in this commit. Recorded in `journal/m4t_matmul_redteam.md`.

### Changed (kernel)

- **SDOT K-bound precondition (H1):** added `M4T_SDOT_K_MAX_EXACT` macro to `m4t_mtfp4.h` (compile-time-derived: `MAX_VAL_MTFP19 / MAX_VAL_MTFP4 = 14,528,268`). Added `assert(K <= M4T_SDOT_K_MAX_EXACT)` in the kernel. Header now declares the K bound as a hard precondition with documentation of caller responsibility beyond the bound. Closes the silent invariant violation where K > 14.5M produced out-of-range MTFP19 mantissas.
- **Header docstring (M5, L3):** SDOT header now explicitly describes the sample-based weight-validity assertion as "spot-check W[0] and the last cell of W[N-1] only — exhaustive validation would scan O(N·K) per call. Consumers that need exhaustive validation should run it once at W setup time."

### Added (tests)

- **`test_sdot_matmul_long_k` (H1, M2):** K=1M with adversarial mixed-sign random inputs against int64 reference. Partway to K_MAX_EXACT.
- **`test_narrow_property` (M1):** 10,000 random samples (mixed uniform + boundary-targeted distribution) for `m4t_mtfp19_to_mtfp4`. Bit-exact comparison against an int64 narrow-reference helper. Covers mantissa output, ROUNDED bit, and SATURATED bit per cell.
- **`test_long_k` (ternary matmul, M2):** K=1M with MTFP4-magnitude operands against int64 reference.
- **`test_partial_block` (M3):** verifies trailing-block flag bits past `M·N` stay zero in the ternary matmul. Forces `M·N=5` to exercise the partial-trailing-block layout.
- **`test_invalid_trit_code` (M4):** packs the same logical weight pattern with codes 0b00 and 0b11 (reserved); verifies kernel produces identical output. K=20 covers both NEON loop body and scalar tail.
- **`test_sdot_matmul_high_mag`:** renamed from `test_sdot_matmul_max_bound` (H2 — the original name implied coverage of the kernel's worst-case input space, which it didn't actually test).

### Changed (tests)

- **Dead Kp locals removed (L1):** `test_saturation_clamp` and `test_saturation_flags` in `test_m4t_ternary_matmul.c` had unused `int Kp = ...` declarations followed by `(void)Kp;`. Cleaned up.
- **`rand_mtfp19` unused-helper marker removed (L2):** the `rand_mtfp19` function is now used by `test_narrow_property`; the dead `(void)rand_mtfp19;` line at the end of `main` was removed.

### Changed (documentation)

- **`docs/DESIGN_X-EXPO.md` flag layout section (L4):** retitled "Flag layout (§14.4 status array — per-block, substrate-wide)" with a new opening paragraph listing every Case-S/Case-R kernel that uses the layout, plus the location of the shared setter (`m4t_internal.h:m4t_flag_or`) and reader (`m4t_mtfp.h:m4t_flag_test`).

### Test surface growth

- `test_m4t_mtfp4`: 10 → **12 tests**.
- `test_m4t_ternary_matmul`: 6 → **9 tests**.

### Build

8/8 ctest binaries green from clean rebuild under `-Werror`. Substrate's overall capability unchanged from the prior tier-3b/3c entry; the remediation hardened tests and tightened preconditions.

## [2026-05-02 — Gesh Phase A.2 red-team: 13 findings, multi-seed methodology promoted]

End-to-end pressure on Phase A.2's code, measurement methodology, and documentation ensemble after the sig_dim sweep landed. Modeled after the m4t kernel red-teams and Phase A.1's red-team. 13 findings; 12 remediated; 1 lifted to project-level methodology rule. Recorded in `journal/gesh_phase_a2_redteam.md`.

The single-seed sweep that the prior CHANGELOG entry described had three single-seed artifacts that **did not survive multi-seed averaging**:
- "+15pp peak at sig_dim = 16" → multi-seed mean **+8.0pp ± 4.6pp**
- "+13pp at sig_dim = 32" → multi-seed mean **+8.2pp ± 2.4pp**
- "−2pp anomaly at sig_dim = 64" → multi-seed mean **+1.8pp ± 2.3pp** (anomaly evaporates)

The qualitative story (compression regime helps, expansion saturates, random ternary at sig_dim=D beats identity) survived. The headline-number narratives did not. The "implicit denoising" framing was demoted from a finding to a hypothesis with a proposed mechanism test.

### Added
- `journal/gesh_phase_a2_redteam.md` — 13 findings tabled, remediations recorded, methodology lessons promoted.
- **Multi-seed sweep tool**: `gesh/bench/sweep_dims.c` rewritten to run N_SEEDS=5 with independent (init, train) seed pairs per cell, reporting mean ± stddev. Links `m` for sqrt.
- **Hot-loop scratch**: `gesh_train_scratch_t` allocated once in `gesh_train_lattice_update`; eliminates per-flip mallocs (M4 fix). ~10× faster sweep.
- **Intra-epoch refresh**: `bank_refresh_every` and `batch_refresh_every` config knobs in `gesh_train_config_t` (H1, H2 fixes). Bank rebuilt and batch resampled every (n_flips/4) flip-evaluations during sweep runs.
- **Early stopping**: `early_stop_patience` config (M5 fix). Cuts wasted compute on plateaued epochs.
- **Budget warning**: `gesh_train_lattice_update` now emits `[gesh_train] warn:` when flip budget is below R's trit count (M6 fix).
- **Balanced random init**: `gesh_init_random_projection_balanced` (L2 cleanup).
- **`test_multi_seed_stability`** — 3-seed test, requires avg gain ≥ 3pp (M2 fix).
- **`test_no_catastrophic_regression`** — requires `trained ≥ random − 5pp` (M3 fix).
- **CONTRIBUTING.md checklist additions**: "Multi-seed validation for any directional measurement claim" and "Hypothesis vs finding distinction in measurement docs". Both lifted from this red-team's C1 and H3 findings.

### Changed
- `gesh/docs/sweep_dims_results.md` — full rewrite for multi-seed numbers. Reports mean ± stddev table, retracts the single-seed peak/anomaly narratives, adds a "Hypotheses (NOT verified findings)" section.
- `gesh/README.md` — Phase A.2 status block reflects multi-seed results (+8pp plateau, +7pp identity-vs-random, hypothesis flagging).
- `journal/gesh_design_closeout.md` — added a "Post-implementation revision" section distinguishing **STE-shadow refresh** (correctly absent) from **R-derivative refresh** (correctly present, the bank is a derived statistic of R) (L5 fix).
- `gesh/tests/test_gesh_train.c::test_trains_reduces_loss` — gate tightened from `< batch_size` (trivially-pass) to `< batch_size / 2` (M1 fix).
- `gesh_train_default()` now sets `bank_refresh_every`, `batch_refresh_every`, `early_stop_patience`, `init_balanced` defaults; documented `seed = 0` as valid (L1 fix; xorshift state mixed with `0x12345678u` to break the all-zero degenerate case).

### Multi-seed sweep table

| sig_dim | random          | trained         | gain    |
|---------|------------------|------------------|---------|
|       2 |  15.6% ± 3.1pp |  21.0% ± 2.4pp |  +5.4 pp |
|       4 |  21.2% ± 1.6pp |  26.8% ± 2.3pp |  +5.6 pp |
|       8 |  31.8% ± 3.1pp |  36.2% ± 0.8pp |  +4.4 pp |
|      16 |  43.4% ± 3.8pp |  51.4% ± 4.6pp |  +8.0 pp |
|      32 |  59.0% ± 2.5pp |  67.2% ± 2.4pp |  +8.2 pp |
|      64 |  76.4% ± 2.1pp |  78.2% ± 2.3pp |  +1.8 pp |
|     128 |  90.0% ± 1.7pp |  89.2% ± 1.5pp |  −0.8 pp |
|     256 |  95.4% ± 0.9pp |  95.4% ± 0.5pp |  +0.0 pp |

Identity (sig_dim = D = 64, no projection): 69%.

### Build
12/12 ctest binaries green. Sweep tool builds clean under `-Werror`. Total sweep runtime ~22s on Apple Silicon (5 seeds × 8 sig_dims × 2 variants).

## [2026-05-02 — Gesh Phase A.2: sig_dim sweep across 8 dims × 3 variants]

> **Note (2026-05-02 red-team):** the +15pp / +13pp / −2pp narratives below were single-seed artifacts. Multi-seed numbers in the entry above supersede them. The qualitative story (compression helps, expansion saturates, random ternary at sig_dim=D beats identity) survives.


A `gesh/bench/sweep_dims.c` benchmark tool sweeps sig_dim ∈ {2, 4, 8, 16, 32, 64, 128, 256} and runs random R / trained R / identity at each. Deterministic. Results saved to `gesh/docs/sweep_dims_results.md`. Three load-bearing findings:

### 1. Lattice update earns its complexity in the compression regime
Peak gain **+15pp at sig_dim = 16** (which exactly equals the informative-dim count K = 16). At sig_dim = 32, +13pp. At sig_dim ≥ 64, training adds 1–2pp on this benchmark — random R already encodes most of what training would.

### 2. Random ternary projection beats identity at sig_dim = D
Identity at sig_dim = 64 hits 69%; random ternary projection at sig_dim = 64 hits 79% — **+10pp over identity at the same dimensionality.** The mechanism: random ternary projection of the 48 noise dims produces incoherent signal that the class-mean bank averages toward zero, while informative dims still carry through. Implicit denoising via random projection. Worth retesting on richer benchmarks.

### 3. Anomaly at sig_dim = 64: trained −2pp vs random
At sig_dim matching D, random hits 79% but trained drops to 77%. Within seed noise (±10 samples for ±2pp on n_test=500), but suggestive of a "training walks into a worse basin than random ternary's implicit regularization" regime. Multi-seed measurement queued; flagged as the most interesting finding for Phase B investigation.

### Sweep table

| sig_dim |  2  |  4  |  8  | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| random  | 19% | 24% | 35% | 47% | 62% | 79% | 89% | 95% |
| trained | 23% | 30% | 44% | **62%** | 75% | 77% | 91% | 96% |
| gain    | +4 | +6 | +9 | **+15** | +13 | −2 | +2 | +1 |

Identity (sig_dim = D = 64, no projection) = 69%. Total sweep runtime: ~17 seconds.

### Added
- `gesh/bench/sweep_dims.c` — sweep tool (built but not in ctest; benchmark, not regression).
- `gesh/docs/sweep_dims_results.md` — full results with curve sketch and discussion.
- `gesh/CMakeLists.txt` adds `gesh_sweep_dims` executable.

### Build
12/12 ctest binaries still green. Sweep tool builds clean under `-Werror`.

## [2026-05-02 — Gesh Phase A.2: lattice-update training online]

The substrate's first measured consumer learns. No STE, no shadow parameters, no Gumbel-softmax — coordinate descent over R's ternary trits with bit-exact loss deltas. The lattice IS the geometry; the optimization walks it directly.

### Added
- **`gesh/src/gesh_train.{h,c}`** — `gesh_train_lattice_update` (training entry point), `gesh_init_random_projection` (random ±1 ternary init), `gesh_train_default` (config helper). Per epoch: sample fresh batch → compute baseline error → evaluate `n_flip_evals_per_epoch` random trit positions, applying the flip that reduces error → rebuild bank end-of-epoch.
- **`gesh/tests/test_gesh_train.c`** — three properties: `trains_reduces_loss` (training reduces error count meaningfully), `beats_random_baseline` (trained R outperforms random R on the test set), `train_determinism` (same seed → same final R + bank).
- Aliasing assertions on `gesh_train_lattice_update` per the new CONTRIBUTING.md checklist item.

### Measured (synthetic prototype classification, D=64 with K=16 informative + 48 noise, 10% per-trit noise, sig_dim=32)

| Variant | Test accuracy |
|---|---|
| Random R (untrained, sig_dim=32) | **62%** |
| Identity projection (Phase A.1, all 64 dims) | 69% |
| Trained R (50 epochs × 200 flips, batch=128, sig_dim=32) | **73%** |

Gain over random init: **+11 percentage points.** Trained R beats Phase A.1's identity baseline using half the dims. The substrate-claim probe at Phase A scope: lattice-native training works.

### Build
- 12/12 ctest binaries green from clean rebuild under `-Werror`.
- New library file: `libgesh.a` now includes `gesh_train.o`.
- Discipline: zero new substrate primitives. Training composes from existing forward pass + bank rebuild + integer error counting. No floats anywhere in the training loop.

### Notes
- PCA-init / variance-ranked init were design candidates for Phase A.2; random init was tried first per discipline ("simplest thing that could work"). Random init reached +11pp, sufficient for Phase A; PCA-init becomes a Phase B+ optimization gated on whether init quality is the bottleneck.
- Lattice update accepts only loss-reducing flips. No simulated annealing, no escape from local minima yet — if convergence stalls before adequate accuracy on harder tasks, escalate to smarter move-acceptance.

## [2026-05-01 — Gesh Phase A.1 red-team remediation]

13 findings (2 high, 5 medium, 6 low) on the Phase A.1 build; 10 fixed in this commit, 3 deferred with rationale. Recorded in `journal/gesh_phase_a1_redteam.md`.

Aliasing assertions added to `gesh_forward_classify` (out_predictions vs queries / bank tiles / projection R). Label-positivity assert added in the n_classes derivation. `sig_dim > 0` assert. Dead variables (`class_counts`, `n_classes_seen`) removed. Three new tests: determinism, aliasing-safety, n_queries=0. Class-balance test tolerance tightened ±25%→±15%. README/code drift on `m4t_route_threshold_extract` corrected. `gesh_bank.h` future-variants clarified per phase.

CONTRIBUTING.md "post-commit doc-currency checklist" extended with: "Aliasing assertions on every writable output." Discipline transfer across architectural layers — substrate patterns don't auto-propagate to consumer code; checklists are the mitigation.

## [2026-05-01 — Gesh Phase A.1: forward pass + synthetic benchmark]

The substrate's first measured consumer. Phase A.1 ships the forward pipeline end-to-end on a synthetic prototype-classification task, no learned projection update yet.

**Library structure (`gesh/`):**
- `bench/synth_proto.{h,c}` — synthetic benchmark generator. C=10 classes, D=64 dims, K=16 informative + 48 noise, 10% per-trit noise. Closed-form deterministic. Pure integer arithmetic.
- `src/gesh_bank.{h,c}` — class-conditional ternary mean bank. One tile per class.
- `src/gesh_forward.{h,c}` — forward pass: optional ternary projection of query → Hamming distance to all bank tiles → top-k smallest → class-vote prediction. Composes `m4t_popcount_dist`. No new substrate primitives.

**Phase A.1 baseline measurements:**
- Identity projection on clean signal: 82%
- Identity projection on 10% noise (the realistic baseline): 69%
- Random ternary 64→32 projection on 10% noise: 61%

**Discipline outcome:** zero new substrate primitives.

**LMM cycle artifacts (`journal/gesh_design_*`):**
- RAW + NODES + REFLECT + SYNTHESIZE: scoped Gesh against task demand (not attention's surface area).
- CLOSEOUT: owner observation surfaced "the lattice IS the geometry." STE dropped — base-2 fix for a problem that doesn't exist in the lattice. Three Gs collapsed to two (Lattice-Geometric + deferred Global). Phase A.1 became forward-pass-only on a prototype-classification probe.

## [2026-05-01 — end-to-end doc-currency remediation]

End-to-end adversarial review of the rebuilt codebase (not the kernels alone) surfaced 11 documentation-drift findings (3 high, 5 medium, 4 low). All remediated in this commit. The kernels were red-teamed thoroughly; the documentation ensemble was not, and stale claims accumulated across the four landing commits. **No code changes** — pure documentation update plus one new journal entry.

### Changed (documentation)

- **`README.md` Status section (H1):** rewrote from "rebuild starts from kernels" (stale, predicting future work) to a complete tier-by-tier status reflecting that tier 1+2+3 all shipped under property-test coverage. Added a journal-cycle subsection to the Documentation table.
- **`m4t/README.md` test surface (H2):** "Six test binaries" → "Eight ctest binaries"; "10 tests" → "12 tests" for `test_m4t_mtfp4`; "6 tests" → "9 tests" for `test_m4t_ternary_matmul`; "Live surface — Tiers 1 + 2" → "Tiers 1 + 2 + 3"; "Tests — Tiers 1 + 2 + 3a" → "Tiers 1 + 2 + 3". Test-table rows expanded to enumerate the new test cases added during the prior red-team remediations.
- **`m4t/docs/M4T_SUBSTRATE.md` broken paths (H3):** frontmatter `supersedes` line now references `01MAY26_archived/m4t/docs/` rather than the broken `archive/` path; §0 status updated to reflect tier 1+2+3 completion; §12 binary-float section rewritten to describe sanctioned categories rather than path-specific links to archived files; §13 file-organization tree clarifies that `m4t/tools/` is deferred and prior-cycle versions are in `01MAY26_archived/`; §17 cross-reference rows for §12 and §13 updated.
- **`m4t/src/m4t_types.h`, `m4t/src/m4t_trit_ops.c`, `m4t/tests/test_m4t_trit_ops.c`:** stale `m4t/tools/...` path references rewritten to point at `01MAY26_archived/m4t/tools/...` or rephrased to be path-free. No code changes; comment hygiene only.
- **`docs/THESIS.md` (M1):** "Does cross-block-exponent MTFP arithmetic earn its complexity?" moved from "Open questions" to a new "Closed questions (substrate-side)" section noting it shipped 2026-05-01. Two remaining open questions (benchmark arbiter, SDOT-as-load-bearing) reframed for the consumer-layer rebuild. Added LMM-methodology note tracking the four cycles run during the substrate rebuild.
- **`docs/REMEDIATION_PLAN.md` (M2):** status header changed from "REVISED — red-team findings folded in" to "EXECUTED 2026-05-01 — owner authorization overrode the consumer-discovery cycle gate." Added a status note at the top of the body pointing readers to CHANGELOG for the actual narrative. Plan body preserved as historical artifact.
- **`docs/FINDINGS.md` (M3):** added Axis 0 "Substrate kernel correctness (regression guard)" with the 8-test inventory. Status updated from "ground-zero (no measurements yet)" to "substrate complete; no benchmark axes yet." Made explicit that this is a regression guard, not a substrate-claim measurement.
- **`.github/CODEOWNERS` (M5):** added a NOTE explaining the username may need to be verified against the EntroMorphic/kittyhawk repo's actual reviewer set.

### Added

- **`journal/m4t_matmul_spec_amend.md` (M4):** lightweight synthesize-only cycle documenting the §8.4 / §8.5 / §17 cross-reference table amendments that landed alongside tier-3b/3c. Companion to `journal/xexpo_spec_amend.md` (which covered §14.2 + §14.4). All substrate-spec edits in this rebuild are now journal-traced per principle 7.
- **`CONTRIBUTING.md` "Post-commit doc-currency checklist":** an 8-item checklist for sweeping the documentation ensemble after any kernel / spec / status-flipping commit. Captures the methodology lesson from this end-to-end red-team — kernel red-teams catch kernel issues, documentation red-teams catch ensemble drift, both are needed.

### Build

No code changes. 8/8 ctest binaries remain green; rebuilt to verify no documentation edit accidentally touched a kernel.

---

### Changed
- (none — ground zero state; tiers 1 and 2 are first landings.)

### Removed
- (nothing — DELETE=never; see `01MAY26_archived/`).
