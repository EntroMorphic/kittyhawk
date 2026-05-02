---
title: Findings
status: substrate complete (2026-05-01); no benchmark axes yet
companions: NORTH_STAR.md · docs/THESIS.md · docs/REMEDIATION_PLAN.md · CHANGELOG.md
---

# Findings

The running ledger of measurements and what they mean.

The prior cycle's findings are preserved in `01MAY26_archived/docs/FINDINGS.md` (eleven axes covering accuracy, speed, inspectability, signature-as-address, multi-table composition, Fashion-MNIST and CIFAR-10 generalization, Go-position substrate-distance refinement, image-pipeline gate, underused-features sweep, and substrate-legal LSH cost characterization).

## Axis structure

Each axis records:
1. The question it answers.
2. The measurement that answers it.
3. What that measurement *cannot* be read as also showing.
4. The journal cycle (raw → nodes → reflect → synthesize → closeout) that produced it.

## Axis 0 — Substrate kernel correctness (regression guard)

**Question.** Do the rebuilt substrate kernels behave correctly under their stated contracts?

**Measurement.** Eight ctest binaries pass from a clean build under `-Werror`:

| Binary | Tests / properties | Oracle |
|---|---|---|
| `test_m4t_trit_pack` | Hand-derived golden values | Self-consistent |
| `test_m4t_trit_ops` | All 9 input pairs × 6 ops | Hand-derived truth tables |
| `test_m4t_trit_reducers` | Mixed inputs across 3 reducers | Hand-derived |
| `test_m4t_mtfp` | Block + vec ops, NEON + tail + aliasing | Hand-derived |
| `test_m4t_route` | All 5 route primitives + emission coverage helper + e2e mini-pass | Hand-derived |
| `test_m4t_mtfp_accum_aligning` | 14 properties × 10k random samples per property | Bit-exact int64 reference |
| `test_m4t_mtfp4` | 12 tests including 10k-sample narrow property + K=1M long-K | Bit-exact int64 reference |
| `test_m4t_ternary_matmul` | 9 tests including K=1M long-K + partial-block + reserved-trit-code | Bit-exact int64 reference |

**What this is not.** This is housekeeping, not a substrate-claim measurement. Bit-exact correctness against a reference says the kernel implements its specification; it doesn't say the specification is the right shape, or that any benchmark exercises the kernel in a way that justifies its complexity.

**Journal cycles.** `journal/xexpo_design_*` (cross-exp design), `journal/xexpo_kernel_redteam.md` (tier 3a remediation, 14 findings), `journal/xexpo_spec_amend.md` (§14.2 + §14.4 amendments), `journal/m4t_matmul_redteam.md` (tier 3b/3c remediation, 11 findings).

## Axes (consumer-layer benchmarks) — none yet

The first axis with measured consumer demand will arrive when the consumer-side rebuild begins. Substrate-claim measurements (throughput, energy, accuracy on a benchmark) require a consumer; the substrate alone produces only correctness regression guards.
