# Changelog

All notable changes to this project are recorded here. The project is pre-1.0 and in active rebuild; there are no tagged releases yet. Format follows [Keep a Changelog](https://keepachangelog.com).

The first entry below marks the ground-zero rebuild that restructured the substrate. Changes before the rebuild are summarized at the bottom; their detail lives in git history and in `journal/`.

---

## [Unreleased] — Ground-zero rebuild (2026-04-14 →)

Triggered by a full audit that identified a collapse of Multi-Trit Floating Point into a fixed-point reading with a shared global scale, and a substrate drifting toward dense computation over base-3 hardware. The rebuild restores MTFP as base-3 floating point (mantissa cells + per-block exponent) and puts routing primitives first.

### Fashion-MNIST generalization + resolver-gap diagnosis (2026-04-15)

- Architecture generalizes to Fashion-MNIST at 85.15% (M=64, density=0.33, SUM) — matching classical pixel k-NN baselines. Resolver gap is ~6× wider than MNIST, concentrated in upper-body-garment cluster.
- Fix 1: `m4t_popcount_dist` builtin popcount fast paths (2-3× speedup, bit-exact).
- Fix 2: `glyph_resolver_sum_neon4` — NEON-batched SUM resolver, 4 candidates per 16-byte vector (1.2-1.3× additional speedup, bit-exact).
- Phase A: `glyph_resolver_sum_voteweighted` — vote-weighted SUM resolver. Falsified on both datasets. Per-class instrumentation revealed Fashion-MNIST gap concentrated in classes {0, 2, 4, 6}.
- Phase B.1: `glyph_resolver_sum_radiusaware` — radius-aware SUM resolver. Falsified (monotone degradation with λ).
- `tools/fashion_atomics.c` — diagnostic tool measuring 3 atoms per failing query: rank/gap, per-table vote agreement, per-table sig-distance gap. Found: 65% of per-table pairs are tied, −0.036 bit mean gap. Magnet audit: no pathological prototypes, gap is structural.
- Phase B.2: `--density_schedule {fixed,mixed}` + `--density_triple a,b,c` CLI flags. Density mixing falsified (tied-gap rate increased from 65% to 67.7%). Per-dataset density tuning confirmed (multi-seed, p<0.02): Fashion-MNIST peaks at 0.25, MNIST at 0.33.
- `--no_deskew` flag for datasets without a canonical shear axis.
- `--resolver_sum {scalar,neon4,voteweighted,radiusaware}` and `--radius_lambda` CLI flags.
- `tests/test_multi_smoke.c` — smoke test covering fixed, wide-mixed, and narrow-mixed density schedules. 11/11 ctest green.
- FINDINGS.md Axis 7, LIBGLYPH.md resolver/config sections, HYPERPARAMETERS.md flag table, README.md headline results — all updated.

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
- `ctest`: 8/8 test binaries pass (`m4t_mtfp`, `m4t_trit_ops`, `m4t_trit_reducers`, `m4t_route`, `m4t_mtfp4`, `m4t_ternary_matmul`, `glyph_wrapper`, `routed_tool_smoke`).
- `M4T_BUILD_TOOLS=ON` builds `m4t_trit_golden` and `m4t_lut_gen` cleanly.
- `tools/mnist_trit_lattice.c` builds via the root CMake (`GLYPH_BUILD_TOOLS=ON`, default ON).

### Routed architecture completion

- Converted `tools/mnist_trit_lattice.c` from projection-L1 plus pixel refine to routed class-signature classification.
- Converted `tools/mnist_routed_amplified.c` from audit-triggered dense fallback to routed fallback over flattened per-head signature evidence.
- Converted `tools/mnist_cascade_sweep.c`, `tools/mnist_cascade_nproj16.c`, and `tools/mnist_cascade_atomics.c` from routed-filter-plus-dense-resolver experiments to routed-filter-plus-routed-resolver experiments.
- Replaced the dense `tools/mnist_resolver_sweep.c` with a routed resolver sweep over secondary, tertiary, and fused hash variants.
- Added `tests/test_routed_tool_smoke.c`, which writes a tiny synthetic IDX dataset and runs `mnist_trit_lattice` end to end so a routed consumer path is now covered by `ctest` without external data.
- Fresh MNIST rerun for the routed-only architecture now completed on `/Users/aaronjosserand-austin/Projects/trix-z/data/mnist`. Headline numbers (deskewed MNIST, density=0.33, K_RESOLVE=50, single seed):

| N_PROJ | pure maj k=7 | routed cascade H2 1-NN | Δ |
|---|---|---|---|
| 8 | 38.74% | 54.21% | +15.47 |
| 16 | 62.00% | **77.33%** | +15.33 |
| 32 | 80.75% | 89.25% | +8.50 |
| 64 | 91.55% | 93.87% | +2.32 |
| 128 | 95.22% | 95.67% | +0.45 |
| 256 | 96.56% | 96.44% | **−0.12** |
| 4096 | 97.65% | 97.41% | −0.24 |

  Routed crossover at N_PROJ=256 (one step earlier than the historical dense crossover at 512). Best routed resolver at N_PROJ=16 is H2+H3 triple-hash fusion at **81.35%** (+4.65 over H1+H2 dual-hash). Atomics on the routed cascade: rescue:damage 5:1, conditional resolver 78.44%, relative margin +0.2056 — mechanism identical to the historical dense run, magnitudes ~60-80%. Full writeup in `journal/routed_cascade_rerun.md`.

### Audit follow-up remediation

- Root `CMakeLists.txt` now applies the warning set at the repo entrypoint, so top-level tools and glyph tests honor the same `-Werror` contract as standalone `m4t` builds.
- `tools/mnist_full_sweep.c` no longer relies on folded VLAs, removing the warnings that prevented the repo-root `-Werror` contract from being true in practice.
- Added direct coverage for `m4t_mtfp_ternary_matmul_bt` (`m4t/tests/test_m4t_ternary_matmul.c`) including exact, NEON-width, NEON+tail, and saturation cases.
- Rebuilt glyph wrapper coverage with `tests/test_glyph_wrapper.c`, so the public alias surface is exercised by `ctest` instead of being documented as pending.

### Measured (first light on rebuilt substrate)

- **LSH on MNIST, N_PROJ=2048: 81.40%.** Bit-for-bit reproduction of the pre-rebuild baseline. See `journal/rebuilt_substrate_first_light.md`.
- Full sweep (4 projection sizes × L1 / refine-3 / refine-5) wall clock: 41.6 s, single core.
- The `m4t_ternary_matmul` bit-select rewrite preserved consumer numerics exactly. No silent regression from the base-2-shortcut → base-3-native shape transition.

### §18 scope + per-primitive audit + coverage-test labels

Third-round red-team remediation on commit `ea0e519`. Six findings; four executed, two deferred with rationale (see `docs/REMEDIATION_PLAN.md` third round).

- **§18 now names its own scope** — three cases (output-side, input-side, not applicable) with worked examples. Applies cleanly to `threshold_extract`, `distance_batch`, `topk_abs`, `apply_signed`; explicitly not applicable to `m4t_mtfp_*` arithmetic and conversion primitives.
- **§18.5 per-primitive audit table** added. Every live routing primitive listed with its §18 scope, sanctioned input class, and coverage-test pointer. Replaces the implicit audit trail with a machine-greppable one.
- **`topk_abs` and `apply_signed` docstrings** updated with §18 contract data (enumerated three-state position, sanctioned input class, coverage-test pointer). Completes the scope commitment from the scrutiny cycle's synthesize.
- **Coverage tests labeled** with `/* §18 coverage test: ... */` comments in `test_m4t_route.c`. Group comments at the top of `threshold_extract`, `topk_abs`, `apply_signed`, `signature_update` test blocks. Audit trail is now discoverable by grep.
- **`threshold_extract` test coverage extended** — three new tests: `n_zero` (defensive), `pack_boundaries` (n ∈ {3, 5, 7, 8}), `extremes` (INT64_MAX, INT64_MIN+1, INT64_MIN). Pack-byte edges and extreme input values now covered.
- **Deferred:** `-tau` UB paranoid fix (contract + assert deemed sufficient); glyph_route.h testing (pre-existing, no consumer).

### Architectural correction (sign_extract → threshold_extract)

- **Removed `m4t_route_sign_extract`.** Advertised three output codes in its type system but produced only two on continuous-valued inputs (zero state was measure-zero for MTFP projections). Type-system theater. The fully-routed MNIST experiment's 23-point accuracy loss vs. the L1 baseline was the visible symptom.
- **Added `m4t_route_threshold_extract(dst, values, tau, n)`** as the sole extractor. Rule: `v > tau → +1`, `v < -tau → -1`, `|v| <= tau → 0`. tau=0 degenerates to sign-extraction exactly — preserves prior behavior at every call site.
- **Updated `m4t_route_signature_update`**: internal call now `threshold_extract(..., 0)`. Structurally the same operation; the call site now makes the "this is sign-only because inputs realize zero naturally" contract visible.
- **Updated `tools/mnist_routed_lattice.c`**: internal calls are `threshold_extract(..., 0)` with docstring noting emission-coverage failure on MTFP projection inputs. Behavior preserved bit-for-bit (58.37% at N_PROJ=2048). A tau>0 variant is the candidate follow-up experiment.
- **Added §18 to `m4t/docs/M4T_SUBSTRATE.md`:** "Base-3 native: emission coverage and the review gate." Single-part, behavioral, per-(primitive, input-distribution)-pair criterion. Review gate requires every new primitive to ship with enumerated output space, sanctioned input-class contract, and a coverage test.
- **Spec history:** derived from two LMM cycles in journal/ (`base3_native_criterion_*` and `updated_model_scrutiny_*`). The first cycle produced a two-part criterion (C-sub + C-con); the scrutiny meta-cycle found the two parts collapsed structurally and converged on single-part emission coverage.

### Complete scaling curve mapped — N_PROJ from 2 to 8192

Spot-probed the `mnist_full_sweep.c` tool at extended N_PROJ values (2, 4, 8, 16, 32, 64, 128, 256, 512, 8192) to complete the scaling curve across four orders of magnitude. Canonical tool restored to sweep {1024, 2048, 4096}; other values reproducible by editing `N_PROJ_VALUES[]`.

**The curve is a clean sigmoid in log-space:**

| N_PROJ | Accuracy | Throughput | Notes |
|---|---|---|---|
| 2 | 18.84% | 4800/s | Above chance floor (10%) |
| 16 | 63.47% | 3300/s | Majority beats rank-wt (inversion regime) |
| 64 | 92.01% | **11000/s** | Throughput peak (NEON alignment) |
| 256 | 96.89% | 7700/s | Matches dense pixel baseline |
| 2048 | 97.86% | 1400/s | Sweet-spot deployment |
| 4096 | 97.99% | 500/s | Knee of the curve |
| 8192 | 98.00% | 260/s | Saturation (+0.01% for 2× compute) |

**Five new observations confirmed:**

1. **Information-theoretic consistency.** Steep-climb regime matches Shannon bounds: 3^n possible signatures vs 10 MNIST classes. Steepest gain N_PROJ 8→32 where signature space expands from 6561 to 1.85 billion.
2. **Vote-rule inversion at N_PROJ ≤ 16.** Majority beats rank-weighted in the highly-tied-distance regime. Rank-weighted reclaims at N_PROJ ≥ 32. Predicted by the mechanism cycle — rank-wt amplifies tie-noise when distance quantization is coarse.
3. **Density 0.33 dominates from N_PROJ=4 to 4096.** Balanced base-3 is empirically optimal across 10 orders of magnitude of signature capacity. Only exception: N_PROJ=2 where density=0.50 wins.
4. **k=7 bookends k=5.** k=5 wins the middle range (256-4096); k=7 wins both tails (very small and very large N_PROJ).
5. **Exponential weighting collapses at every scale.** Identical accuracy across k=3/5/7 for every (N_PROJ, density) from 2 to 8192. The "too-steep" failure mode is scale-invariant. Cleanest predicted-failure validation in the project's history.

**N_PROJ=64 is the throughput peak.** 11 000 queries/sec at 92% accuracy. 16-byte signature = one NEON vector; popcount instruction's natural unit. Below 64, scalar tail dominates; above 64, compute grows linearly with signature size.

**Saturation confirmed at N_PROJ=4096.** 8192 adds 2× compute for +0.01% accuracy. Further gains require representational changes (multi-stage, per-class τ, augmentation), not more of the same.

Full writeup: `journal/full_scaling_curve.md`.

### Full matrix sweep — 97.99% new best; mechanism predictions confirmed

New consumer: `tools/mnist_full_sweep.c`. First comprehensive sweep over (N_PROJ × density × k × vote_rule). 81 configurations × 3 seeds = 243 measurements. Runtime 9.1 minutes.

**New headline: 97.99 ± 0.01% at N_PROJ=4096, density=0.33, k=5, rank-weighted.** Three-seed measurement with ±0.01% variance.

Per-N_PROJ scaling (d=0.33, k=5, rank-wt):

| N_PROJ | Accuracy | Δ from prior N_PROJ |
|---|---|---|
| 1024 | 97.75 ± 0.07% | — |
| 2048 | 97.86 ± 0.01% | +0.11% |
| 4096 | **97.99 ± 0.01%** | +0.13% |

Clean ~0.12% gain per doubling. Not saturated; N_PROJ=8192 plausible.

**Mechanism cycle predictions confirmed empirically:**

1. **Exponential weighting (2^(k-rank-1)) collapses to top-1 classification.** At any k, top-1's weight exceeds the sum of all other weights combined (57%, 52%, 50.4% at k=3, 5, 7). The matrix shows exp-wt producing identical accuracy across k=3, 5, 7 for every (N_PROJ, density) — empirical demonstration of the "too-steep" failure mode predicted in `journal/mechanism_that_worked_*.md`.

2. **Rank-weighted k=5 is the dominant (rule, k) sweet spot.** Appears in 7 of top 10 configurations. Rank-k=7 is close but slightly below at the peak (97.95 vs 97.99 at N_PROJ=4096).

**Density 0.33 confirmed empirically optimal:**

| Density | Accuracy (at N_PROJ=4096, k=5, rank-wt) |
|---|---|
| 0.25 | 97.91 ± 0.05% |
| **0.33** | **97.99 ± 0.01%** |
| 0.50 | 97.73 ± 0.06% |

Balanced base-3 isn't just aesthetic; it's empirically the peak AND the most stable across seeds (tightest stddev).

**Amplification ceiling for this representational family: ~98%.** Each extra basis point costs doubling compute. To push further would need either structurally different representations (multi-stage, per-class τ, pair-specific masks) or data-level interventions (augmentation).

**Gap to dense baseline has grown.** Deskewed-pixel dense L1 k-NN: 97.16%. Routed 97.99%. Routing wins by **0.83 points** on accuracy, still ~20× faster on wall time.

Full writeup: `journal/full_matrix_sweep.md`.

### Amplification experiment — predictions failed honestly

New consumer: `tools/mnist_routed_amplified.c`. Tested two amplification paths proposed from the inspectability analysis: (1) K=5 independent ternary projections with majority-vote ensemble, (2) audit-triggered pixel-k-NN fallback for uncertain queries.

**Predictions:** ensemble +0.3 to +0.5%, fallback +0.1 to +0.2%.
**Actual results:** ensemble +0.04%, fallback *negative*.

| Configuration | Accuracy |
|---|---|
| Best solo projection (rank-k=5) | 97.91 ± 0.04% |
| **Ensemble (K=5, no fallback)** | **97.90 ± 0.02%** |
| Ensemble + FB (agree≥5) | 97.75 ± 0.02% (−0.15%) |
| Ensemble + FB (agree≥4) | 97.86 ± 0.05% (flat) |
| Ensemble + FB (agree≥3) | 97.90 ± 0.01% (tied) |

**Why the predictions failed:**

1. **Ensemble:** errors don't decorrelate across random projections on MNIST. Failure modes are input-driven (genuinely ambiguous digits) not projection-driven. Different random seeds see the same ambiguities. The ensemble's actual value is deterministic stability (matches best-of-5 with half the variance), not accuracy ceiling.

2. **Fallback:** on the hardest subset (where ensemble is uncertain), pixel-k-NN is *worse* than routing, not better. 46.7% correct on trigger set vs ensemble's ~50% baseline on same subset. Swapping loses ~15 cases per seed.

**What this tells us:**
- The ~2.1% residual error on MNIST is at or near the floor for this representational family. Aggregating more of the same doesn't recover missing information.
- The routing surface beats dense pixel k-NN *even on hard cases* — stronger than the previous "wins on average" claim.
- Audit-based adaptation is a detection tool; correction requires a classifier with *different information*, not just a different view.

**Prediction failure diagnostic:** both predictions were made from whole-set averages (pixel-k-NN is 97.16%, projection errors "should" decorrelate). The relevant performance distribution on the *hard subset* is dramatically different. Future adaptation predictions should use the audit trail to measure on the targeted subset, not extrapolate from headline averages.

Full writeup: `journal/amplification_negative_result.md`.

### Adapted (failure-guided vote-rule modification — first adaptation loop on substrate)

New consumer: `tools/mnist_routed_weighted.c`. Tests whether distance- or rank-weighted voting recovers the NARROW MISS cases identified by the inspectability trace. Same pipeline as `mnist_routed_knn.c`; only the vote rule differs.

**Result, 3 seeds, deskewed N=2048:**

| Vote rule | k=3 | k=5 |
|---|---|---|
| Majority (baseline) | 97.79 ± 0.05% | 97.77 ± 0.02% |
| Distance-weighted | 97.84 ± 0.04% (+0.05%) | 97.78 ± 0.03% |
| Rank-weighted | 97.72 ± 0.06% (-0.07%) | **97.86 ± 0.01%** (+0.09%) |

**New best configuration: rank-weighted k=5 at 97.86 ± 0.01%.** Consistent across all three seeds. Paired t-test ≈ 2.6σ.

**Prediction vs actual:** predicted +0.25-0.30% from the trace; actually got +0.09%. Overestimated by ~3× because the NARROW MISS coarse category is inclusive (captures "correct class was close") rather than discriminative (captures "which vote rule would flip it"). Direction of the effect was correct.

**Significance for the substrate:** first measured ADAPTATION (not measurement, not primitive, not benchmark) on the rebuilt substrate. Trace observation → classifier modification → measured accuracy improvement. All in integer arithmetic over discrete structures — no gradients, no floats, no STE. The full gradient-free adaptation loop works in principle.

**Asymmetric finding:** rank-weighted k=3 *hurts* by -0.07%. The 3/2/1 weighting amplifies top-1's vote; if top-1 is wrong, the wrongness triples. Weighting scheme and k are coupled.

Full writeup: `journal/weighted_voting_adaptation.md`.

### Demonstrated (routed k-NN inspectability — the third axis)

New consumer: `tools/mnist_routed_trace.c`. For each misclassified MNIST test image, prints the complete audit trail of the routed decision: top-5 nearest training prototypes with distances, vote composition at k=3, per-trit decomposition of the distance to the top-1 (agreements split by trit value; disagreements split by sign-flip cost-2 vs zero-vs-sign cost-1), per-class nearest-prototype distance over all 60 000 prototypes, and a failure classification derived from those integer thresholds.

**Failure distribution over 221 misclassifications at 97.79% deskewed N=2048 k=3:**
- NARROW MISS (correct class within 10 bits of winner): 74 (33.5%)
- VISUAL CONFUSION (both classes have near prototypes): 65 (29.4%)
- SEPARATED (correct class genuinely far): 82 (37.1%)
- OUTLIER (no class has a close prototype): 0 (0.0%)

**Structural observation from per-trit breakdown:** errors cluster at the quantization boundary, not at semantic opposition. Of the ~30% disagreement bits in typical near-misses, 90-95% are zero-vs-sign (cost 1) mismatches; sign-flips (cost 2, full opposition) are a small minority. The router rarely says "+1" where the correct class says "-1"; it more often says "0" where the correct class says ±1.

**What dense k-NN cannot produce:** per-prototype reasoning. Dense L1 is a scalar sum; the contributing dims aren't separable. Routed Hamming IS the per-trit cost sum by construction; inspectability is a substrate-level property, not an add-on.

Full writeup: `journal/routed_inspectability_trace.md`.

### Measured (fair-comparison edition — routing wins stronger than initial claim)

Red-team of commit `663c355` flagged three comparison-fairness holes: scalar-vs-NEON speed comparison, single-seed accuracy claim, weak dense baseline. Remediated all three in `tools/mnist_routed_knn.c`; re-ran the sweep. Outcome: **the win strengthened, not weakened.**

**Headline, fair comparison (3 seeds, NEON-vectorized L1, deskewed-pixel baseline included):**
- **Routed k-NN, deskewed proj, N_PROJ=2048, k=3: 97.79 ± 0.05%.**
- Dense deskewed-pixel k-NN (classical baseline), k=3: 97.16%. Routing wins by 0.63 points.
- NEON-vectorized L1 k-NN over same projections: 97.62 ± 0.07%. Routing wins by 0.17% (2σ) at k=3; 0.25% (4.6σ) at k=5.

**Routed vs L1 over same projections, raw mode, N_PROJ=2048:**
- k=3: 97.30 ± 0.03% vs 97.00 ± 0.05%, Δ +0.30% at 5.2σ.
- k=5: 97.18 ± 0.04% vs 96.85 ± 0.05%, Δ +0.32% at 5.9σ.

**Speedup is larger under fair comparison, not smaller:**
- 20.3× at N_PROJ=2048 (routed 7.0s vs NEON-L1 141.4s; was 10.8× against scalar L1).
- 12.0× at N_PROJ=512.
- The compression of 2048 int32 mantissas into 512-byte packed-trit signatures drives cache-locality advantages; popcount processes trit information at NEON-native VCNT throughput; L1 fights L2 cache pressure from 480 MB of training projections.

**Trit distribution verified:** +33.4% / 0 32.9% / -33.7% across all configurations. Genuine balanced base-3.

### Retracted / revised

- Prior claim "10.8× faster than dense" → actual speedup against fair NEON baseline is 20.3× at N_PROJ=2048. The prior number was against a scalar baseline.
- Prior claim "Routed beats dense by 0.26 points" → marginal as single-run. Fair 3-seed measurement at the same config is +0.30% at 5.2σ; the win is confirmed and quantified.
- Prior claim "first empirical confirmation of NORTH_STAR §Claim" → the original evidence was 1.5σ single-run. It carries now at 5σ multi-seed.

### Qualifications remaining

- N_PROJ=512: routed loses by ~0.12% vs L1. The routed win is N_PROJ-dependent.
- MNIST k-NN is cooperative for both approaches; the thesis-testing bed remains open (see `docs/THESIS.md` §4).

Full writeup in `journal/routed_knn_mnist.md` "Revised after fourth red-team" section; remediation plan in `docs/REMEDIATION_PLAN.md` fourth round.

### Measured (real Trit Lattice LSH with k-NN — routing beats dense) [superseded by fair-comparison entry above]

New consumer: `tools/mnist_routed_knn.c`. Full LSH architecture — 60 000 training signatures as prototypes, k-NN classification via Hamming distance, symmetric balanced-base-3 zero distribution via per-side τ calibration.

- **N_PROJ=2048, k=3, fully routed: 97.31% MNIST accuracy.** Beats MTFP19 L1 k-NN at the same config (97.05%) by 0.26 points; runs **10.8× faster** (7.0 s vs 75.9 s).
- Routed wins at every k value for N_PROJ=2048: k=1 (96.93% vs 96.88%), k=3 (97.31% vs 97.05%), k=5 (97.21% vs 96.91%).
- Symmetric deployment verified: train %zero = 32.87%, test %zero = 32.58%; both within 0.5% of the 33% target.
- **First empirical confirmation of NORTH_STAR §Claim on the rebuilt substrate.** "Routing is essential, and will naturally outperform dense, in a base-3 environment." Demonstrated on MNIST at 10K test images.

**Supersedes** the conclusion in `journal/tau_sweep_routed_mnist.md` ("three-state routing loses to sign-only on MNIST"). That claim was correct within its measurement (centroid-based routing with asymmetric τ) but wrong as a general statement. The full LSH k-NN consumer with symmetric τ contradicts it.

Full writeup: `journal/routed_knn_mnist.md`.

### Measured (τ sweep on the fully-routed MNIST classifier)

First experiment that empirically distinguishes a §18-failing deployment from §18-passing deployments on the same routing primitive. tools/mnist_routed_lattice.c now sweeps τ ∈ {0, 10K, 50K, 200K, 1M} for each N_PROJ.

- **At every N_PROJ, three-state routing (τ > 0, §18-passing) loses to sign-only routing (τ = 0, §18-failing).** The gap grows with τ; at τ=1M the band swallows everything and accuracy collapses to chance (9.80% ≈ 10% random).
- **Best-case routed accuracy is 58.37%** (τ = 0, N_PROJ = 2048) — unchanged from the prior measurement; bit-for-bit reproduction.
- **L1-over-mantissa baseline still at 81.40%** (N_PROJ = 2048).
- **§18 is a utilization criterion, not a quality criterion.** A primitive can be properly base-3-deployed AND produce worse downstream accuracy than a base-2-degenerate deployment, if the task's discriminative signal lives in a representation the three-way primitive throws away. MNIST's signal is in projection magnitudes; widening the band suppresses that signal.
- **NORTH_STAR §4 confirmed empirically** ("Running routing-native on [MNIST] is a test of adapter efficiency, not the thesis"). The most §18-correct routing deployment loses to dense by ~25 points.

Full writeup: `journal/tau_sweep_routed_mnist.md`.

### Measured (fully-routed MNIST classifier)

- **New consumer: `tools/mnist_routed_lattice.c`.** First tool that exercises the full routing surface (`m4t_route_sign_extract` + `m4t_route_distance_batch` + `m4t_route_topk_abs`) end-to-end.
- **Routed accuracy at N_PROJ=2048: 58.37%** (vs. 81.40% L1-over-mantissa on the same projections and training). The routed path loses ~23 points across every N_PROJ; accuracy barely responds to more projections (sign-extract is the binding constraint).
- Reading: the routing primitives produce arithmetically-correct results; the accuracy gap is an adapter-efficiency artifact of sign-extracting 19-trit mantissas into 1-trit signatures on a task (MNIST nearest-centroid) whose signal lives in magnitudes. Consistent with NORTH_STAR §4: "Running routing-native on [MNIST] is a test of adapter efficiency, not the thesis."
- Full writeup: `journal/fully_routed_mnist.md`.

### N_PROJ=16 atomic probe — vote-rule inversion explained

New consumer: `tools/mnist_probe_nproj16.c`. At N_PROJ=16 the complete scaling curve showed majority beating rank-weighted — reversing the pattern at higher N_PROJ. Probe decomposed the mechanism.

- **52% of queries have an exact signature match** (min Hamming = 0). 97% have min_d ≤ 2 bits. Tied-at-top-1 set averages ~4-10 prototypes with 2.10 distinct classes.
- **Correct-class location partition:** 75.85% tied-min, 15.62% elsewhere top-10, 8.53% nowhere.
- **Partition asymmetry:** rank-wt wins tied-min by +0.99% and loses elsewhere by −5.89%. Weighted sum matches the observed aggregate gap exactly (−0.17%). Both vote rules are near-Bayes-optimal on their own partition; neither is globally better.
- Predicted adaptive voting (`tied_count ≥ 2 → rank-wt, else majority`) at +0.75% aggregate gain.
- Full writeup: `journal/nproj16_atomic_mechanism.md`.

### LMM cycle — "can N_PROJ=16 reach 90%?" → filter-ranker reframe

Full four-file LMM cycle on whether >90% is attainable at N_PROJ=16. Core insight: the 16-bit hash has been asked to do two jobs simultaneously — filter candidates AND classify them — and voting is the wrong primitive for the second job. Voting reads rank information the hash has destroyed; the hash preserves set membership, which a *resolver* can read directly.

- Cycle files: `journal/nproj16_to_90_{raw,nodes,reflect,synthesize}.md`.
- Synthesized prediction: cascade architecture (16-bit hash as primary filter, pixel-L2 1-NN resolver over top-K) reaches 85-91%.

### Cascade at N_PROJ=16 — 92.72% on 16-bit hash

New consumer: `tools/mnist_cascade_nproj16.c`. Implements the LMM's predicted cascade (E1-E5) in a single pass.

**Headline (single seed, deskewed MNIST, density=0.33):**

| K | pure-hash majority | cascade L1 1-NN | cascade L2 1-NN |
|---|---|---|---|
| 20 | 64.14% | 85.74% | 86.40% |
| **50** | 63.31% | 90.15% | **90.75%** |
| **100** | 63.00% | 92.27% | **92.72%** |

Ceiling at top-100: 99.50%. LMM prediction of 87-91% validated; stretch goal of 90% exceeded.

**Controlled variants at K=20:**
- Pixel-L1 3-NN majority: 82.38% — voting at the resolver stage hurts.
- Partition-aware (singleton → top-1, else L1 resolve): 78.50% — singletons are unreliable; apply resolver uniformly.
- Secondary-hash (different seed) Hamming 1-NN: 74.84% — quantifies "more bits" vs "pixel access" contribution.

Takeaway: roughly half the cascade gain comes from "more bits" (secondary hash gets to 74.84%), the other half from pixel-signal access (gets to 85.74%). To cross 90% the resolver must include pixel access.

Full writeup: `journal/nproj16_cascade_result.md`.

### Cascade atomic decomposition — hash is filter, not ranker

New consumer: `tools/mnist_cascade_atomics.c`. Dissects *why* the cascade gained +30 percentage points.

**The single explanatory fact:** at N_PROJ=16 the hash preserves neighborhood membership at 98.59% (correct class in top-50) while getting top-1 correct only 55.48% of the time. 43-point gap. The hash is a filter with destroyed ranking.

**Rescue/damage at K=50:** 3668 queries rescued (pure-hash top-1 wrong → cascade right), 141 damaged (reverse). Ratio **26 : 1**.

**Hash-rank distribution of cascade's correct picks:** only 4.26% come from hash-rank 1; **50.72% come from hash-ranks 21–50**. The hash places correct prototypes in the candidate pool; pixel L2 finds them wherever they sit.

**Per-partition cascade accuracy:**
- Tied-min partition: **95.96%** (was 77.65% under rank-wt — +18 points).
- Elsewhere-in-top-10: **86.94%** (was 24.65% under majority — +62 points).
- Nowhere-in-top-10: 54.62% (rescued because K=50 is wider than probe's top-10 window).

**Pixel-distance margin within top-50:** correct class is on average **33% pixel-closer** than nearest-wrong class (relative margin +0.3255). The margin exists *because* the filter has removed most wrong-class mass — explains why earlier amplification over unfiltered 60K failed.

**Class-pair confusion:** digit 0 was the hash's error sink (five of the top eight improved pairs involve class 0). Ternary popcount homogenizes blob-shaped digits; pixel L2 separates shapes. Total damage across all worst regressions: 11 queries. Asymmetric by two orders of magnitude.

Full writeup: `journal/cascade_atomics_mechanism.md`.

### Cascade sweep across N_PROJ — crossover at 512, resolver ceiling at 97.57%

New consumer: `tools/mnist_cascade_sweep.c`. Verified the atomic prediction that cascade headroom shrinks as N_PROJ grows.

| N_PROJ | pure maj | cascade | Δ |
|---|---|---|---|
| 8 | 38.74% | 82.61% | **+43.87%** |
| 16 | 62.00% | 90.75% | +28.75% |
| 32 | 80.75% | 95.04% | +14.29% |
| 64 | 91.55% | 96.65% | +5.10% |
| 128 | 95.22% | 97.28% | +2.06% |
| 256 | 96.56% | 97.51% | +0.95% |
| 512 | 97.06% | 97.57% | +0.51% |
| 1024 | 97.43% | 97.58% | +0.15% |
| **4096** | **97.65%** | **97.57%** | **−0.08%** |

**Two new findings:**

1. **Practical crossover at N_PROJ=512.** Below 512, cascade adds measurable accuracy; above 512 it adds ≤ 0.5 points. First *negative* gain appears at N_PROJ=4096 (pixel-L2 misranks a few queries the filter had placed correctly at top-1).

2. **The resolver has its own ceiling — 97.57%.** Cascade accuracy plateaus at 97.57-97.58% from N_PROJ=512 through 4096, regardless of how accurate the filter becomes. The factorization `cascade = filter_presence × conditional_resolver_rate ≈ 99.87% × 97.7% ≈ 97.6%` is quantitatively exact. To push past 97.57% on deskewed MNIST, the resolver must change (richer metric, learned comparator, convolutional features); more bits in the filter won't help.

**Cost-accuracy consequence:** N_PROJ=8 cascade (82.61%) beats pure N_PROJ=32 (80.75%). Cascade buys approximately one octave of N_PROJ at small scales. N_PROJ=64 cascade (96.65%) matches pure N_PROJ=2048 at 1/32 the hash cost.

Architectural rule added to `docs/FINDINGS.md` Axis 4: use cascade when N_PROJ ≤ 128, wash at N_PROJ ≥ 512, change resolver to exceed 97.57%.

Full writeup: `journal/cascade_sweep_crossover.md`.

### Routed quadruple-hash fusion at N_PROJ=16: 83.86%

Extended `tools/mnist_resolver_sweep.c` with H4 (fourth independent seed) and H_D50 / H_D20 (density-varied secondaries) plus per-resolver confusion tracking for 3→8 / 3→5 / 6→8. New routed-cascade record at N_PROJ=16:

    R12 H2+H3+H4 1-NN (quadruple)   83.86%  +7.16 over dual-hash
    R15 H2+H_D50 1-NN               81.78%  +5.08 (dual-density fusion)
    R9  H2+H3 1-NN (triple)         81.35%  +4.65
    R2  H1+H2 1-NN (baseline)       76.70%

Marginal-gain curve: dual → triple +4.65, triple → quadruple +2.51. Independent views stack with diminishing returns.

Density decorrelation breaks 6→8 confusion (35 → 14-17 errors with H_D50 or H_D20) but leaves 3→8 / 3→5 essentially unchanged. Those pairs are projection-family-bound, not density-sensitive.

Second finding: k-NN majority at the resolver stage beats 1-NN on the hard regression pairs (R4 gets 3→8 down to 50, the lowest of any resolver). Amends the "don't vote at the resolver" rule from the dense cascade era: **vote when the resolver is noisy, 1-NN when it's precise.**

At N_PROJ=1024 all routed resolvers collapse to ~97.5% — the routed cascade ceiling is independent of resolver choice once the filter is near-perfect.

Full writeup: `journal/routed_quadruple_decorrelation.md`.

### Meta-router LMM cycle — online inline observer as k-NN over routing-context signatures

Four-file LMM cycle (`journal/meta_router_online_{raw,nodes,reflect,synthesize}.md`) on the user's proposal for a global observer that runs inline with the local cascade and learns from prior failures online.

Core insight from REFLECT: the observer is not a predictor of failure (which would require labels or self-supervised proxies that invite closed-loop drift). It is a **routing primitive whose keys are routing-context signatures and whose values are routing decisions**, with a bank that grows from cascade execution traces. The update rule is append-on-execution, not append-on-error — there is no "failure label" to drift against because the bank stores behaviors, not truths.

The routing-context signature packs six cascade byproducts into 16 trits (4 bytes, one NEON Hamming unit): H1 min-distance bucket, tied-min count bucket, H1-H2 disagreement flag, H2-H3 disagreement flag, quadruple-fusion margin bucket, and a query-signature checksum. The meta-router lookup is k-NN over a 4096-entry ring buffer at ~1.7% of the H1 primary cost. Bank update is append-on-ring.

Two-phase experiment plan, gated on P1 (prerequisite): does global quadruple fusion actually beat local quadruple fusion on the queries local fails on?

### P1 gate — global quadruple rescues local 2.7:1 at N_PROJ=16 (PASS)

`tools/mnist_local_vs_global.c` compares three variants at N_PROJ=16, K_RESOLVE=50, single seed, all 10K test queries:

| variant | accuracy |
|---|---|
| L — local quadruple (H1 top-50 + H2+H3+H4 fusion) | 83.86% |
| Gt — global H2+H3+H4 summed over 60K | 86.64% |
| **Gq — global H1+H2+H3+H4 summed over 60K** | **89.46%** |

2×2 contingency `(L vs Gq)`: 8055 both-right, 331 damages (L right, Gq wrong), 891 rescues (L wrong, Gq right), 723 both-wrong. Rescue:damage ratio **2.7:1**, net +5.60 accuracy if Gq is applied blindly. Oracle ceiling (L ∪ Gq) = **92.77%**.

Gate: PASS. Meta-router P2 has real architectural headroom — +8.91 points of oracle potential over pure local.

### P1 atomics — rescues and damages share observable features (revised P2 ceiling ≈ 88%)

`tools/mnist_lvg_atomics.c` decomposes the P1 contingency across seven signals: H1 min-distance, H1 tied-count, correct-class rank in H1 top-50, ensemble disagreement (H2-H3, H3-H4, H2-H4), fusion pick H1-rank, fusion margin, per-class, and confusion pairs. Reports distributions per contingency cell (LR_GR easy, LR_GW damage, LW_GR rescue, LW_GW both-wrong).

**Headline: on every inference-available signal, rescues and damages have nearly indistinguishable distributions.** Ensemble disagreement cleanly separates easy (21.7%) from hard (~60-68%) but sees all three hard cells at roughly the same rate. H1 min-distance is flat across cells. Tied-count is weak. Fusion margin is weak. The only meaningful secondary signal is fusion pick H1-rank — damages pick deeper (52.9% at ranks 21-50 vs 46.5% easy, 6.7% at ranks 2-5 vs 13.0% rescue) — a 6-8 point gap, marginally usable.

The feature that *would* separate rescue from double-fail is the rank of the correct class in H1's top-50 pool (rescues: 7.4% outside top-50 and 21.1% at rank 1; double-fails: 10.4% outside and 18.4% at rank 1). But that feature requires the true label and is unavailable at inference.

Per-class: class 3 has the worst rescue:damage ratio (94:65 = 1.4:1). Classes 4, 6, 7 are best escalation targets (3.4-4.1 ratio). Class 1 is solved (4:3, negligible). Global has its own 3↔5 / 3↔8 confusions that damage nearly as often as local's failures get rescued.

**Revised P2 prediction:**

| architecture | predicted / measured |
|---|---|
| pure local (L) | 83.86% |
| disagreement meta-router | **~88.0%** |
| pure global (Gq) | 89.46% |
| oracle ceiling | 92.77% |

Meta-router's natural ceiling is ~88% at ~30% escalation rate and ~50% of pure-Gq cost. P2 is now primarily a **cost-efficiency test**, not an accuracy-ceiling test — the routing-context signature cannot separate rescue from damage beyond what a disagreement threshold already does.

**New verified claim: observability ceiling at N_PROJ=16.** The 4.77-point gap between the disagreement meta-router and the oracle lives in queries whose rescue/damage label is hidden in information the substrate structurally cannot observe at this signature size. Closing the gap requires structural changes (wider K, fused filter, per-class policy, more independent hashes), not a better meta-router.

Full writeup: `journal/lvg_atomics_decomposition.md`.

### Fused filter fix — information leverage rule, meta-router obsolete

`tools/mnist_local_v2.c` reruns the P1 gate after applying the two composable fixes the Axis 4c atomics suggested:

- **Fix A — widen K_RESOLVE** from 50 to 100 / 200. Targets the 7.4% of rescues whose correct class sits outside H1's top-50 pool entirely.
- **Fix B — fused filter.** Replace the H1-alone filter with an `(H1+H2)` summed-distance filter, then resolve locally with H3+H4. Moves H2 from "one of three resolvers" to "half of the filter" — same four hashes, same arithmetic, different cascade position.

The two fixes are independent axes and compose into a 2×3 grid. All variants compared against the same Gq reference at N_PROJ=16, density=0.33, single seed, deskewed MNIST, 10K test queries.

**Result grid:**

    variant       filter          resolver          K    accuracy   Δ baseline   Δ Gq
    L50_H1        H1 alone        H2+H3+H4          50   83.86%     —            -5.60
    L100_H1       H1 alone        H2+H3+H4         100   85.59%     +1.73        -3.87
    L200_H1       H1 alone        H2+H3+H4         200   86.79%     +2.93        -2.67
    L50_H12       (H1+H2)         H3+H4             50   88.44%     +4.58        -1.02
    L100_H12      (H1+H2)         H3+H4            100   88.73%     +4.87        -0.73
    L200_H12      (H1+H2)         H3+H4            200   88.87%     +5.01        -0.59
    Gq reference  H1+H2+H3+H4 over all 60K              89.46%     +5.60        —

**Headline: fused filter alone — without widening K — lifts accuracy by +4.58 points.** Widening K alone tops out at +2.93. Composed fixes land at 88.87%, closing 89% of the original L→Gq gap (5.60 → 0.59).

**Filter ceilings** at K=50: H1 alone = 98.59%, (H1+H2) fused = 99.55% (filter-miss rate 1.41% → 0.45%, a 3× reduction). At K=200: H1 = 99.86%, (H1+H2) = 99.94%. But the ceiling lift is small (0.96 points at K=50) relative to the accuracy lift (4.58 points) — most of the fused filter's benefit comes from *ranking improvement* inside the preserved pool, not from adding new correct-class prototypes.

**Contingency vs Gq:**

    variant       LR_GR   LR_GW   LW_GR (rescue)   LW_GW   net   oracle
    L50_H1 P1     8055     331         891          723   +560   92.77%
    L200_H1       8432     247         514          807   +267   91.93%
    L50_H12       8651     193         295          861   +102   91.39%
    L200_H12      8658     229         288          825    +59   91.75%

Rescues collapse from 891 to 265-295 as the fused filter absorbs most of global's previously-unique recoveries into local's own answers. Damages also drop (331 → 192-229). The oracle ceiling (L right ∪ Gq right) falls from 92.77% to 91.38-91.75%, leaving only ~2.88 points of theoretical headroom for any meta-router to add on top.

**Cost accounting** (popcount distance operations per query):

    L50_H1 baseline   60K (H1)   +  150 (H2+H3+H4)  ≈  60K   1.00×   83.86%
    L200_H12         120K (H1+H2) +  400 (H3+H4)    ≈ 120K   2.01×   88.87%
    Gq               240K (H1+H2+H3+H4)                     4.00×   89.46%

L200_H12 captures 99.3% of Gq's accuracy gain at 50% of Gq's cost. The resolver-stage widening from K=50 to K=200 is a rounding error because the filter dominates; each 60K global pass is ~600× the cost of a 50-candidate resolver pass.

**Mechanism.** Both L50_H1 and L50_H12 use the exact same four hashes with the exact same seeds and the exact same Hamming kernel. Only the cascade position of H2 differs. In L50_H1, H2 contributes ranking *after* H1 has already hard-committed the top-50 pool. In L50_H12, H2 contributes set-membership *before* the hard commitment. Moving H2 across this boundary buys +4.58 points. The atomic decomposition of the Axis 4c probe showed that 48% of the original rescues had correct class at H1-ranks 6 or deeper — prototypes H1 alone was failing to pull into the shallow positions of top-K. The fused filter uses H2's second opinion to rescue those prototypes before the K-cut.

**The information leverage rule.** Stated as a new architectural principle:

> In a cascade, information applied at the filter stage constrains set membership; information applied at the resolver stage only re-orders an already-committed pool. When the filter is imperfect, spend marginal routing information on the filter first. Information has higher leverage earlier in the cascade.

This is the dual of the filter-ranker reframe from the `journal/cascade_atomics_mechanism.md` era. The original reframe said "the hash is a filter, not a ranker — use it as a filter." The information-leverage rule extends it: "and when you have more than one hash available, put them *all* at the filter stage until the filter saturates. Only spend remaining hashes at the resolver." Together the two rules give a concrete allocation policy for multi-hash cascades.

**Meta-router deprecation.** The meta-router LMM cycle (`journal/meta_router_online_{raw,nodes,reflect,synthesize}.md`) proposed an online, inline k-NN-bank architecture to close the L→Gq gap by routing hard queries to global on a per-query basis. The Axis 4c atomic decomposition predicted a ceiling of ~88% for any meta-router built on inference-available signals. The fused-filter fix produces the same ~88% accuracy *without any routing at all*, and it lowers the oracle ceiling to 91.75% — leaving only 2.88 points of theoretical headroom. Even an oracle meta-router on L200_H12 with perfect rescue/damage separation could add at most 2.88 points, and realistic meta-routers with observable-signal features would capture only a fraction of that at substantial extra cost (~50% of pure Gq).

Verdict: **the meta-router was a proposal to route around a deficient filter; the correct fix is to deepen the filter.** The cycle is closed with a negative verdict on its proposed primary artifact but a positive verdict on its research process — the P1 gate forced the atomic decomposition, the atomic decomposition exposed the filter-ranking structure of the gap, and the fused-filter fix became visible as a direct consequence.

Full writeup: `journal/fused_filter_fix.md`.

### Architectural correction — dense outer loop discovered in every cascade tool

Audit finding surfaced in a user exchange about the word "dense" in conversational response text. Every cascade tool built in this session — `mnist_cascade_nproj16`, `mnist_cascade_sweep`, `mnist_cascade_atomics`, `mnist_resolver_sweep`, `mnist_local_vs_global`, `mnist_local_v2`, `mnist_lvg_atomics`, plus the earlier `mnist_routed_knn` and `mnist_full_sweep` — runs routing primitives (`popcount_dist`, `threshold_extract`) inside an `O(N_train)` dense outer loop. Every per-query pass touches all 60K training signatures one by one. That is dense application shape with routed kernels: a substrate-level NORTH_STAR violation, even though the per-comparison primitive is routing-native.

The "20.3× speedup over dense L1 at N_PROJ=2048" headline (Axis 2 in FINDINGS.md) is apples-to-apples on the same `O(N)` outer loop: trit-sig-popcount scanning 60K prototypes vs MTFP-L1 scanning 60K prototypes. The speedup is real but it is a *compression* win, not a *routing* win at the architecture level. The cascade tools are reframed as **measurement scaffolding** — they produced correct research observations about filter-ranker asymmetry, fused-filter leverage, observability ceilings, atomic confusion structure, and adaptive routing — but none of them is a production architecture.

### First genuinely routed consumer — `tools/mnist_routed_bucket.c`

New consumer that uses the signature as an **address**, not as an operand. Training-time build sorts `(signature_key, prototype_index)` pairs by signature_key. Query-time resolution is binary search into the sorted table plus ternary-Hamming multi-probe over neighbor codes at radius 0, 1, 2.

Index structure at N_PROJ=16 (60K prototypes):

```
60 000 prototypes → 37 906 distinct buckets (1.58× compression)
  29 616 singleton buckets (78.1%)
     6 099 buckets of size 2-3
     1 621 buckets of size 4-7
       420 buckets of size 8-15
       112 buckets of size 16-31
        25 buckets of size 32-63
        12 buckets of size 64-127
         1 bucket with 128+ prototypes (all-zero sig region)
build time: 3 ms (one-time)
```

Ternary multi-probe enumeration operates directly on packed 2-bit trit codes — no unpacking to int8 arrays. Neighbor sets respect the ternary Hamming cost function: r=0 one probe, r=1 ~27 probes per query, r=2 ~340 probes per query, all 4-byte key lookups that binary-search the 937 KB sorted table.

**Tuning sweep over `(MAX_RADIUS × MIN_CANDIDATES)`:**

    MAX_R  MIN_C  accuracy   avg_cands  avg_probes  empty   us/qry
     0       1    36.99%         9.2         1.0    4797     0.4
     0     100    36.99%         9.2         1.0    4797     0.3
     1       1    61.80%         8.4         9.8    1129     0.7
     1     100    68.90%        46.5        22.0    1129     2.1
     2       1    67.82%         8.4        33.3     175     1.3
     2     100    82.58%       136.4       216.2     175     9.9
     2     400    82.60%       193.8       237.1     175    12.2

Best operating point: **MAX_R=2, MIN_C=100 → 82.58% accuracy at 9.9 μs/query.**

**Comparison against dense baseline:**

| architecture | popcount_dists/query | μs/query | accuracy |
|---|---|---|---|
| dense L50_H1 (scan 60K H1 + H2+H3+H4 top-50 resolver) | 60 150 | ~1 950 | 83.86% |
| routed bucket (MAX_R=2, MIN_C=100) | ~410 | 9.9 | 82.58% |
| **ratio** | **~147× fewer popcount ops** | **~197× faster wall time** | **−1.28 points** |

The H1 pass is completely eliminated. All filter work is binary search plus multi-probe enumeration. The resolver runs `popcount_dist` only over the small candidate set (avg 136 prototypes per query).

**Radius escalation profile matches the Axis 4c atomic probe exactly:**

| radius | queries | fraction | probe-predicted fraction |
|---|---|---|---|
| r=0 sufficient | 5 203 | 52.03% | 52% (exact-match rate) ✓ |
| escalated to r=1 | 3 668 | 36.68% | ~37% (cumulative ~89% at min_d ≤ 2 bits) ✓ |
| escalated to r=2 | 954 | 9.54% | ~10% (cumulative ~98% at min_d ≤ 4 bits) ✓ |
| empty at r=2 | 175 | 1.75% | ~2% (min_d > 4 bits) ✓ |

The consumer is literally measuring what the probe predicted. Every query's radius of resolution is a direct function of the H1 signature codebook's collision structure.

**The 1.28-point gap to dense L50_H1 is dominated by the 175 empty queries** whose nearest training signature exceeds Hamming radius 2. Closing the gap requires either r=3 (more probes per query, worth ~1 point of accuracy) or applying the Axis 4d information-leverage rule to the bucket — concatenate H1+H2 into 8-byte keys so the codebook is `3^32` instead of `3^16` and most buckets become singletons with precise collision semantics.

**New architectural rule (extending Axis 4d's rules 1-6):**

**Rule 7. Production k-NN uses the signature as an address, not as an operand.** Build a bucket index keyed on the packed-trit signature; query via binary search + ternary multi-probe; run the resolver only on the candidate set. The `O(N_train)` outer loop is scaffolding for research measurements, not an architecture. This rule subsumes the filter-ranker reframe: the filter now does zero distance work — it performs a table lookup. Distance computation is reserved for the resolver, over the small candidate set.

Full writeup with full derivation, code structure, and follow-up list: `journal/routed_bucket_consumer.md`.

### LMM cycle — can local + global routing reach 97% at N_PROJ=16?

Full four-file LMM cycle (`journal/break_97_nproj16_{raw,nodes,reflect,synthesize}.md`) on the user's hypothesis. Core insight from REFLECT:

> In a routing-only substrate, "local routing" and "global routing" are not two different mechanisms. They are two different roles of the same primitive applied at two different aggregation scales. Local = per-query neighborhood lookup in a single bucket index. Global = union-merge of neighborhood lookups across M independent bucket indexes. The global operator is the union itself — it requires no new primitive, no new data structure, no dense scan.

Committed to Reading A (classical multi-table LSH with union-merge) as the concrete mapping of "local + global" to a routing-only substrate. Predicted target crossing around M=32 based on scaling-curve extrapolation. Gated execution on an oracle-pass prerequisite.

### Phase 1 + 2 — multi-table routed bucket tool + oracle gate

`tools/mnist_routed_bucket_multi.c` extends `tools/mnist_routed_bucket.c` to M independent bucket indexes with per-table ternary multi-probe and cross-table union-merge. Three resolver variants wired up (VOTE, SUM, PTM) gated behind `--full` flag. Oracle mode (default) is a fast pre-check.

Red-teamed the build before running: no dense paths at query time, seed 0 matches canonical `(42,123,456,789)`, ternary multi-probe matches Axis 5 tool exactly, per-table-majority resolver semantics are "constrained to union" (named explicitly), runtime envelope ~100 s per resolver at M=64.

Sanity check at runtime: table 0 distinct-bucket count is 37906, matching the Axis 5 single-table consumer exactly.

Oracle ceiling pass (Phase 2) results:

    M     oracle   avg_union   avg_probes
    1    94.30%        94.3      194.4
    2    97.90%       132.5      425.9
    4    99.75%       315.9      801.1
    8    99.99%       543.6     1657.2
    16   100.00%     1072.8     3274.0
    32   100.00%     1985.8     6538.7
    64   100.00%     3521.2    13064.4

Wall time: 5.43 s for 10K queries × full M=64 sweep. ~0.54 ms/query for probing alone.

**Gate: PASS.** M=2 already hits 97.90% oracle, M≥16 is 100%. Two observations that reshaped the cycle:

1. My scaling-curve extrapolation was wrong. I conflated "classification accuracy" (rank-destroyed) with "set-membership in multi-probe neighborhood" (neighborhood-preserved). The Axis 4c atomic probe (52% exact-match, 97% at min Hamming ≤ 2 bits) was the right anchor; the scaling curve was not.

2. Tables are moderately correlated (~6× miss-rate factor vs fully-independent LSH theory at M=2). Random ternary projections at matched density share structural miss modes. The architecture still composes well because correlation is pairwise while miss events require ALL M tables to miss.

### Phase 3 — multi-table routed bucket SUM at M=32 reaches 97.24% — target broken

First routed architecture in Glyph to exceed 97% accuracy on deskewed MNIST at N_PROJ=16. Full resolver sweep:

    M    VOTE      SUM       PTM      oracle
    1    62.96%   54.50%    54.63%    94.30%
    2    71.82%   77.78%    62.20%    97.90%
    4    76.75%   88.91%    75.34%    99.75%
    8    81.83%   93.84%    86.07%    99.99%
    16   85.78%   96.13%    91.48%   100.00%
    32   88.50%  *97.24%*   94.25%   100.00%    <- target crossing
    64   89.77%  *97.31%*   95.36%   100.00%

Wall time: 68.42 s total sweep. VOTE=0.21s, PTM=25.12s, SUM=34.55s (cumulative across all M checkpoints).

**Sanity checks (Phase 3 red-team):**

1. M=1 VOTE = 62.96% ≈ pure N_PROJ=16 k-NN (62.00%). Within 1 point of the scaling curve. Multi-probe widens the neighborhood slightly. Passes.

2. M=1 SUM (54.50%) is lower than M=1 VOTE (62.96%). At M=1 the SUM resolver reads table 0's own rank — the rank-destroyed signal the filter-ranker reframe warned about. The 8.5-point VOTE-SUM gap at M=1 is the filter-ranker asymmetry reappearing as an internal consistency check. Passes.

3. At M=4 this tool matches the Axis 5 single-table consumer's 4-hash budget (table 0 + H2/H3/H4). SUM=88.91% here vs Axis 5's 82.58% — **+6.33 points from the multi-table architectural win on matched information budget.** The union-merge structure adds real value beyond the information content.

**Comparison to the pure-signature scaling curve at matched total bits:**

    architecture              total_bits   accuracy
    Pure N_PROJ=256            256         96.56%
    Pure N_PROJ=512            512         97.06%
    Pure N_PROJ=1024           1024        97.37%
    M=32 SUM (32 x 16)         512         97.24%    (+0.18 vs pure)
    M=64 SUM (64 x 16)         1024        97.31%    (within noise)

Multi-table at matched total bits matches or slightly beats the pure scaling curve. The +0.18 bonus at 512 bits comes from the independence structure — 32 independent 16-trit random projections with multi-probe neighborhoods collectively cover more input-space geometry than a single monolithic 512-trit random projection. The bonus vanishes at higher bits because the curve flattens.

**Resolver behavior — SUM dominates, VOTE plateaus weak, PTM middles.** SUM is the best resolver at every M ≥ 2. At M=32 SUM (97.24%) beats VOTE (88.50%) by 8.74 points and PTM (94.25%) by 2.99 points. VOTE saturates at 89.77% at M=64 — a 7.54-point gap to SUM — because set-membership voting discards the distance gradient. PTM is middle at 95.36% at M=64 — majority-voting noisy per-table 1-NN estimates loses to summing distances.

**Unexpected finding documented:** the synthesize-phase prediction that VOTE might beat SUM at low M because of "cross-table tie-breaking" was wrong. SUM beats VOTE at every M ≥ 2 and the gap widens. Set-membership voting is architecturally weaker than summed-distance ranking in this measurement.

**Resolver gap plateau at M ≥ 32.** The gap between oracle (100%) and best resolver (SUM) shrinks from 31.34 points at M=1 to 2.76 at M=32 and 2.69 at M=64. Adding tables beyond M=32 does not shrink the gap further. This is the **structural ceiling of random-ternary-SUM ranking on this task**: ~2.7% of queries have the correct class in the multi-probe union but summed popcount-Hamming ranks a wrong-class prototype higher. Closing this gap requires density variation, structurally different hash generators, or learned projections — not more tables of the same family.

**Cost-accuracy at operating points:**

    M     accuracy   ms/query    notes
    16    96.13%     ~0.67       cost sweet spot below target
    32   *97.24%*    ~1.92       target crossing
    64    97.31%     ~4.13       diminishing returns

Reference baselines: dense L200_H12 was 88.87% at ~1.95 ms/query; dense Gq was 89.46% at ~3.9 ms/query; pure N_PROJ=512 dense scan was 97.06% at ~4.0 ms/query. **M=32 SUM matches the wall time of dense L200_H12 while being +8.37 accuracy points higher. It matches pure N_PROJ=512's accuracy at ~2× faster wall speed.** Zero dense paths anywhere.

**New architectural rule 8 (extending the Axis 4d + Axis 5 rule list):**

> Multi-table composition reproduces the scaling curve at equivalent total bits, with a small independence bonus. The signature-as-address architecture (Axis 5 rule 7) composes through M independent bucket tables via union-merge at the global step and summed-distance resolver at the scoring step. At matched total bits (`M × N_PROJ` = equivalent single-hash `N_PROJ`), multi-table SUM matches the pure scaling curve within noise and may gain up to ~0.2 points from independence structure. This gives a concrete allocation policy for building routed k-NN consumers at any target accuracy on the scaling curve: pick a base `N_PROJ` small enough for cheap per-table operations, then compose M tables until matched-bits accuracy hits the target.

Full writeup: `journal/break_97_nproj16_phase3_results.md`.

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
