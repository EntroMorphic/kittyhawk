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
