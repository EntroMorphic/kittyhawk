---
title: Findings
status: substrate complete + 3 substrate-claim axes landed (2026-05-05)
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

## Axis 1 — R1 dual-threshold signature rule (METHODICALLY FALSIFIED)

**Question.** Does a per-expression-tau dual-threshold signature rule (sign + confidence) discriminate expression-routing equivalence classes better than a sign-only rule?

**Measurement.** 4-axis methodical falsification on the standard expression-routing benchmark (multi-seed, multi-config). Pre-committed numerical gates per axis:

| Axis | Pre-committed gate | Result |
|---|---|---|
| F-G1 — class count + intra-class consistency | dual ≥ 20% more classes AND ≥ 80% intra-class consistency | WEAK SUPPORT (more classes, but non-quality metric) |
| F-G2 — inter-class minimum distance | dual ≥ sign-only AND dual ≥ 4 trits | FAIL (dual=1 vs sign-only=3 at arity-1) |
| F-G3 — partition-change rate | ≥ 30% partition change | FAIL (4.2% mean, 96% rule agreement) |
| F-G4 — substrate-novelty (third-state utilization) | zero-band ∈ [20%, 60%] for both arities | FAIL (arity-1 zero-band 66.5% — third state OVER-DOMINATES) |
| F-G5 — held-out routing accuracy | ≥ 5pp accuracy improvement | DEFERRED (requires external equivalence ground truth) |

**Verdict.** R1 methodically falsified across 4 substantive axes. The dual-threshold rule does NOT outperform sign-only on any quality-of-discrimination metric.

**What this is not.** Not a falsification of vision claim 3 broadly. R1 is one specific operationalization of "third state is load-bearing"; other operationalizations (different test-input strategies, different signature derivation, different consumer patterns) remain testable.

**Journal cycle.** `journal/r1_falsify_*` (RAW → NODES → REFLECT → SYNTHESIZE → CLOSEOUT, each with red-team where applicable).

## Axis 2 — Tri-state utilization audit (intra-substrate, weak claim)

**Question.** Where in the substrate's existing kernels is the third state load-bearing vs sink-like vs under-exploited? (Restricted to intra-substrate utilization; does NOT make comparative claims against base-2 alternatives — that's Axis 3.)

**Measurement.** Two-gate audit on a 2-layer ternary GEMM workload modeling 1.58-bit LLM forward pass. 12 configs (3 sizes × 2 weight zero-fracs × 2 activation zero-fracs) × 5 seeds = 60 runs. Layers L1, L2, L3, L4, L6 measured (L5 deferred — not exercised by GEMM-only workload).

- **Gate I (info-theoretic):** Shannon entropy of third-state distribution. Load-bearing: H ≥ 1.4 bits; sink-like: H < 1.0.
- **Gate II (algorithmic dependence):** cosine similarity native vs forcibly-binary-collapsed Y2. Load-bearing: cos ≤ 0.95.

| Layer | Verdict (post-R-G1 collapse-design fix) |
|---|---|
| L1 (weight third-state) | LOAD-BEARING (cos ≈ 0.49) |
| L2 (activation third-state) | LOAD-BEARING (cos ≈ 0.62) |
| L3 (per-MAC product third-state) | MIXED — sparsity-dominated (entropy sink in sparse regimes) |
| L4 (post-reduction Y1 mantissa) | MIXED — least load-bearing measured layer (cos ≈ 0.94), but not invisible |
| L6 (post-ternarization X2) | LOAD-BEARING (cos ≈ 0.74) |

**Red-team caught critical artifact.** Initial L4 collapse design substituted median-magnitude values, which were reabsorbed by the downstream quantile threshold — producing artifact cos = 1.000. Per `journal/tristate_op_redteam.md` C1, redesigned to override-after-ternarize semantics.

**What this is not.** Intra-substrate utilization, NOT comparative advantage. A layer being load-bearing in our substrate does NOT imply base-3 outperforms a base-2 alternative at the same density — that's Axis 3.

**Journal cycle.** `journal/tristate_op_*` (RAW → SYNTHESIZE → CLOSEOUT + red-team R-G1).

## Axis 3 — Strong-claim L1 weights (comparative, base-3 vs base-2)

**Question.** At the L1 weight layer, does base-3 outperform base-2-with-mask (B2-B = sign bit + sparsity bit) on density, precision, or kernel cost?

**Measurement.** 5-kernel bench (`audit/tristate_strong_bench`) with NEON-only kernels, K-aligned to 80, register-tiled by 4 j-cells. Bit-exact verification across all kernels + external grounding via substrate's `m4t_ternary_dot_matmul_bt`.

| Kernel | Storage | Density |
|---|---|---|
| Path A (base-3 4-in-8 packed) | packed trit | 2 bits/cell |
| Path B (B2-B honest, separate sign+mask decode) | sign + mask packed | 2 bits/cell |
| Path B-skip (B2-B + all-masked-block skip) | as Path B | 2 bits/cell |
| Path C (B2-B optimal, unified TBL decode) | sign + mask packed | 2 bits/cell |
| Path D (base-3 5-in-8 packed) | 5 trits per byte | **1.6 bits/cell** |
| Substrate (`m4t_ternary_dot_matmul_bt`) | unpacked int8 ternary | 8 bits/cell |

**Verdict (post P0-1 + P0-2 + P0-3 with apples-to-apples tiling):**

| Axis | Verdict |
|---|---|
| Density at fixed packing | **PARITY** — both 2 bits/cell at the substrate's current packing. |
| Density CEILING | **base-3 STRUCTURAL ADVANTAGE** — base-3 reaches 1.6 bits/cell (5-in-8); B2-B floored at 2 bits/cell because sign+mask are independent. **B2-B cannot follow base-3 below 2 bits/cell.** |
| Precision | **PARITY** (60/60 bit-exact across all kernels and substrate). |
| Kernel cost at 2 bits/cell | **PARITY** — Path A (base-3) ≡ Path C (B2-B optimal) byte-for-byte at the disassembly level. Encoding labels are aliases at fixed density. |
| Kernel cost at sub-2-bit | **base-3 wins ~1.8×** — Path D vs Path A 0.55-0.58× wall-clock across all tested regimes (L1-resident through DRAM-bound), apples-to-apples (both register-tiled). Mechanism: better SDOT pipeline saturation via amortizing setup overhead over 5 SDOTs per 80-trit block (vs Path A's 1 SDOT per 16-trit block). |

**Red-team rounds (cumulative):**
- R-G1 (P0-2 round): vqtbl4q register pressure → switched to vqtbl2q; eliminated mov.16b padding.
- C1 (strong-claim initial): B2-B-honest was a strawman (separate sign+mask decode is unnecessarily expensive). Added Path C as B2-B-optimal; confirmed Path A ≡ Path C at fixed density.
- C2 (membw addendum): cache-warming bias between consecutive kernel runs of same workload. Added cache_flush() between kernels.
- C3 (membw addendum): tested regime never actually exceeded L2. Added DRAM-bound config (K=12800, N=8192, W=25.6 MB exceeds L2). Showed trajectory PLATEAUS at ~1.16-1.24×, doesn't crossover (with prior P0 baseline).
- P0-3 fairness: only Path D was register-tiled initially → 3× headline was tile asymmetry. Remediation: tiled Path A and Path C too; honest 1.8× win preserved.

**What this is not.** L1 ONLY. L2/L4/L5/L6 strong-claim cycles are deferred. The 1.8× wall-clock advantage holds on Apple Silicon's NEON pipeline characteristics; other architectures may shift the balance. The structural density-ceiling advantage is hardware-independent.

**Journal cycles.** `journal/tristate_strong_*` (RAW → SYNTHESIZE → CLOSEOUT, multi-round red-team) + `journal/tristate_strong_5in8_addendum.md` (sub-2-bit packing) + `journal/tristate_strong_membw_*` (memory regime test) + `journal/p0_kernel_opt_redteam.md` (P0-1/P0-2/P0-3 with per-item red-team).

## Open axes (not yet measured)

- **Strong-claim L2 (activations).** Same shape as L1; likely similar verdict at fixed density. Not yet run.
- **Strong-claim L4 (cross-layer requantization).** Audit's Track A. Currently de-prioritized given the L1 verdict.
- **Strong-claim L5 (cross-exp accumulator).** Requires residual-style workload not produced by GEMM. Not yet run.
- **Strong-claim L6 (post-ternarization activations).** Same shape as L2.
- **Vision claim 3 (broad form).** Each layer's strong-claim cycle is a tile of this broader question. The L1 verdict is the first defensible empirical point.
