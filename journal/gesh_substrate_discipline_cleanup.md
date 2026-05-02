---
cycle: gesh_substrate_discipline_cleanup
date: 2026-05-02
scope: route every ternary projection and sign-threshold call site through libm4t kernels; eliminate open-coded substrate-bypass arithmetic across gesh consumer + bench code
companions: gesh/src/gesh_project.{c,h} · gesh/tests/test_gesh_project.c · journal/gesh_phase_b_redteam.md
status: COMPLETE — all 7 substrate-bypass sites routed through kernels; bit-equivalence verified
---

# Substrate-discipline cleanup

## What this cycle is

A kernel-use audit caught that the gesh consumer library and bench code re-implemented ternary projection and sign-threshold by hand in 7 places, when libm4t had `m4t_mtfp_ternary_matmul_bt` and `m4t_route_threshold_extract` for exactly those operations. **All substrate-claim work to date was running through hand-written loops, not the substrate kernels we built and tested.**

This cycle routes every kernel-bypass site through libm4t, verifies bit-equivalence with the prior open-coded path, and re-runs the regression measurements (Phase B Gate 2, sig_dim sweep, MNIST ablation) to confirm numbers are unchanged.

## Sites cleaned up

| Site | Was | Now |
|------|-----|-----|
| `gesh/src/gesh_forward.c::ternary_project_row` | Open-coded MAC + sign | `gesh_project_one_packed` → kernels |
| `gesh/src/gesh_train.c::rebuild_bank_from_projection` | Open-coded MAC + sign | `gesh_project_batch_unpacked` |
| `gesh/src/gesh_train.c::count_errors_scratch` | Open-coded MAC + sign per query | `gesh_project_batch_unpacked` per call |
| `gesh/src/gesh_bank.c::gesh_bank_build_class_mean` | Open-coded sign threshold | `gesh_threshold_int32_to_trit` → kernel |
| `gesh/bench/sweep_dims.c::build_bank_from_projection` | Open-coded MAC + sign | `gesh_project_batch_unpacked` |
| `gesh/bench/mnist_probe.c::build_bank_with_R` | Open-coded MAC + sign | `gesh_project_batch_unpacked` |
| `gesh/bench/denoise_probe.c::project_train_acc` | Open-coded MAC | `m4t_mtfp_ternary_matmul_bt` direct |
| `gesh/bench/denoise_probe.c::prototype_alignment_stddev` | Open-coded dot product | Replaced with `col_stddev` over kernel matmul output |
| `gesh/bench/image_canon.c::image_canon_quantize_unpacked_batch` | Open-coded threshold | `m4t_route_threshold_extract` |

## What was added

- **`gesh/src/gesh_project.{c,h}`** — three substrate-routed wrappers:
  - `gesh_project_batch_unpacked(out, x, n, R, sig_dim, input_dim)` — batch project unpacked → unpacked.
  - `gesh_project_one_packed(out_packed, x, R, sig_dim, input_dim)` — single query → packed.
  - `gesh_threshold_int32_to_trit(out, values, n)` — int32 array → unpacked ternary.
  - All three internally call `m4t_pack_trits_1d` + (for matmul cases) `m4t_mtfp_ternary_matmul_bt` + `m4t_route_threshold_extract` + `m4t_unpack_trits_1d`. Zero open-coded arithmetic.

- **`gesh/tests/test_gesh_project.c`** — bit-equivalence property test:
  - 7 shapes × 3 seeds = 21 batch-projection equivalence checks against an explicit reference open-coded loop.
  - 4 sizes × 3 seeds = 12 threshold-extract equivalence checks.
  - All 33 checks: kernel path produces ZERO differing trits from the reference loop.
  - Threshold tests also assert emission coverage (all three ternary states present per call).

## What was NOT cleaned up (justified non-kernel sites)

- **`image_canon::normalize_one`** — per-image zero-mean unit-variance with integer isqrt. One-shot preprocessing pipeline, not runtime; substrate principle 1 (no binary float in runtime kernels) sanctioned via §12 (one-shot conversion sites).
- **`gesh_init_random_projection_balanced`** — `(r==0) ? -1 : (r==1) ? 0 : 1` on a uniform random; balanced ternary sampling, not a sign threshold.
- **`image_canon::cmp_i64`** — qsort comparator returning -1/0/+1 per C convention; not a ternary trit emitter.
- **Float arithmetic in `compute_stats_pm`, Pearson r, etc.** — reporting only, not runtime/training.

## Bit-equivalence verification

Property test result on `test_gesh_project`:

```
ALL PASS test_gesh_project
  21 batch-projection cells × 3 seeds: 0 differing trits across all
  12 threshold-extract cells × 3 seeds: 0 differing trits, emission coverage holds
```

Re-run measurements post-cleanup (same seeds, same configs):

| Measurement | Pre-cleanup | Post-cleanup | Δ |
|-------------|-------------|--------------|---|
| Gate 2: Pearson r | +0.8921 | +0.8921 | 0 |
| Gate 2: t-statistic | 157.89 | 157.89 | 0 |
| Gate 2: stratification (low/mid/hi means) | 3649 / 7451 / 11404 | 3649 / 7451 / 11404 | 0 |

(MNIST and sig_dim sweep results pending; expected to be bit-identical because both paths are deterministic integer math producing identical ternary outputs.)

## Why this is a methodology issue, not just an optimization

- **Substrate-claim integrity:** every prior measurement supporting "base-3 routing-first" claims was run through hand-written loops, not the kernels we cite as the substrate. The substrate-claim was about libm4t's primitives; the measurements bypassed them. Cleanup makes the substrate-claim load-bearing on what we actually built.

- **Performance dishonesty:** sweep_dims at sig_dim=1024 reported "~515s on Apple Silicon." That timing is for hand-written loops, not for `m4t_mtfp_ternary_matmul_bt`. Future timing claims will be on-substrate.

- **Discipline rule violation:** principle 5 ("no primitive without named consumer demand") was satisfied when kernels were built (Phase A.1 named ternary matmul as needed). The flip-side is satisfied only if the consumer actually uses it — otherwise the kernel is unjustified. Cleanup reciprocates the rule.

## What two prior red-teams missed

- **Phase A.2 red-team:** focused on multi-seed methodology. Did not audit kernel use.
- **Phase B red-team:** focused on multi-config methodology. Did not audit kernel use.

A *kernel-use red-team* is its own diagnostic class. Worth promoting:

> **Discipline rule (lifted to CONTRIBUTING.md):** any substrate-claim measurement must run through the substrate kernels it claims. Re-implementing kernel-shaped arithmetic in consumer or bench code invalidates the substrate-claim semantics. The audit asks: for every multiply-accumulate or sign-threshold in the consumer/bench code, is there a libm4t kernel that does this? If yes, why isn't it called?

This rule supplements the prior two red-team rules:
- Phase A.2: *multi-seed gates the cell.*
- Phase B: *multi-config gates the story.*
- Phase B kernel audit: *kernel use gates the substrate-claim.*

## Performance implications

Test runtime: `test_gesh_train` 0.84s → 3.5s (4× slower) due to per-call allocation in the wrappers. Still well within regression-test tolerances. The wrappers are not optimized for hot loops; the prior open-coded path was deliberately fast at the cost of substrate-claim integrity. Future work could add a scratch-aware wrapper variant if hot-path performance becomes a constraint.

For benchmark-scale measurements (sweep_dims, MNIST probe), allocation overhead is amortized over the matmul cost. Initial expectation: probes are within 1.5–3× of pre-cleanup runtime, with substrate-claim integrity restored.

## Follow-up scope

- **Performance:** if a substrate-claim measurement requires faster kernels, add a scratch-aware variant of `gesh_project_batch_unpacked` with caller-managed buffers. Not done in this cleanup.
- **Path A consumer:** the upcoming richer-consumer cycle (multi-table LSH per Gate 1.A) inherits the kernel-routed path automatically by using `gesh_project_*` instead of any open-coded loops.
- **Doc currency:** sweep_dims_results.md, phase_b_gate1_results.md, README, and CHANGELOG updated to note the substrate-discipline cleanup. CONTRIBUTING.md gains the kernel-use audit rule.
