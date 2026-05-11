---
title: Findings
status: substrate complete + 3 substrate-claim axes + BitNet inference characterized + first direct Part-B evidence on N4 sparse attention (2026-05-10)
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

**Measurement.** **29 ctest binaries** pass from a clean build under `-Werror` (as of 2026-05-09). Original 8 listed below; the additional 21 cover NEON-vs-scalar_ref bit-exactness for the no-scalar-audit cycle (pack/unpack 4-in-8 + 5-in-8, MTFP4↔MTFP19 conversions, shift3, ternary_matmul_neon, accum_aligning_neon), the §20 5-in-8 W-packed and X-packed matmul kernels, the V14.C–V14.G Phase-2 BitNet primitive paths (rmsnorm_bx with the gamma_bx > target_bx regression case at commit `4d4c917`, relu²_bx, elementwise_mul_bx, bitlinear_scale_bx, softmax NEON-gather LUT, RoPE, A8 quantize), and gesh consumer tests:

| Binary | Tests / properties | Oracle |
|---|---|---|
| `test_m4t_trit_pack` | Hand-derived golden values + 4-in-8 + 5-in-8 NEON-vs-scalar_ref | Self-consistent + NEON-vs-scalar |
| `test_m4t_trit_ops` | All 9 input pairs × 6 ops | Hand-derived truth tables |
| `test_m4t_trit_reducers` | Mixed inputs across 3 reducers | Hand-derived |
| `test_m4t_mtfp` | Block + vec ops + softmax + RoPE + bx-aware kernels (rmsnorm_bx incl. gamma_bx > target_bx regression, relu²_bx, elementwise_mul_bx, bitlinear_scale_bx) + vec_dot_i64 incl. bound-constant pin | Hand-derived + bit-exact NEON-vs-scalar_ref |
| `test_m4t_route` | All 5 route primitives + emission coverage helper + e2e mini-pass | Hand-derived |
| `test_m4t_mtfp_accum_aligning` | 14 properties × 10k random samples per property | Bit-exact int64 reference |
| `test_m4t_mtfp4` | 12 tests including 10k-sample narrow property + K=1M long-K | Bit-exact int64 reference |
| `test_m4t_ternary_matmul` | 9 tests including K=1M long-K + partial-block + reserved-trit-code | Bit-exact int64 reference |
| (+ 21 additional binaries — see `m4t/README.md` "Tests" section for the full inventory) | | |

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

## Axis 4 — BitNet b1.58-2B-4T end-to-end inference quality on the substrate

**Question.** Does the substrate run a real ternary LLM (BitNet b1.58-2B-4T, 2B parameters, 30 layers) end-to-end with quality comparable to the bf16 reference?

**Measurement.** 24-prompt greedy-decoded battery across 8 categories: factual recall (3), definitional (3), narrative (3), math/reasoning (3), code (3), dialog/structured (3), long-context (3), edge cases (3). Each prompt manually classified ✓ / ⚠ / ✗ for correctness + coherence. HF (bf16) cross-checked on the substrate-specific failure subset.

| Configuration | ✓ | ⚠ | ✗ | Strict pass |
|---|---|---|---|---|
| Pre-RMSNorm-fix substrate | (degenerate loops on canary) | — | — | — |
| Post-RMSNorm-fix, baseline (BX=2) | 15 | 4 | 5 | 63% |
| Post-fix + GATE_ACT_BX = 1 | ~19 | ~5 | 0 | ~80% |

The ~80% configuration matches HF's behavior on 4 of 5 previously-failing prompts (`reason_word`, `code_loop`, `code_comment`, `json_format`). The single remaining substrate-specific divergence at this config is `factual_hamlet` (gives a "Hint: It's a famous play..." instead of "Shakespeare"); see TD-22.

**What this is not.** Not a benchmark score. Not a competitive eval against any standard NLP suite (24 prompts is small; there's no public BitNet eval). It's a substrate-quality characterization at the 2B-LLM scale: the substrate's MTFP19 / packed-ternary numeric system runs the model to coherent end-to-end output across diverse task categories, with measurable but well-characterized quality limits.

**What this is.** First empirical confirmation that the substrate can host a real ternary LLM at production scale. Prior to this work, the substrate had been validated at the kernel level (Axis 0) and the synthetic-routing level (Gesh phase A.1/A.2), but never on a real consumer model. The 24-prompt battery across diverse task shapes makes the kernel-correctness story (Axis 0) downstream-meaningful: the kernels not only pass their unit tests, they compose into a model whose outputs are coherent.

**Journal cycles.** `journal/bitnet_phase1_*` (raw → nodes → reflect → synthesize → closeout, work-units 1-8); `journal/substrate_vs_hf_2026-05-09/RESOLVED.md` (RMSNorm bug fix); `journal/post_rmsnorm_fix_battery_2026-05-09/SUMMARY.md` (initial 8-prompt confirmation); `journal/inference_battery_v2_2026-05-09.md` (24-prompt characterization, including HF cross-check on failures); `journal/hp_sweep_2026-05-10.md` (GATE_ACT_BX retuning that recovered 4 of 5 failures).

## Axis 5 — Part-B evidence on N4 sparse attention (substrate-claim)

**Question.** Does the substrate's routing-first base-3 architecture provide a measurable advantage over no-routing alternatives at fixed compute, AND does the advantage widen as task structure (here: attention sparsity, ≈ task-richness in the test-design sense) increases? This is the central thesis Part B claim, untested with pre-committed gates until this cycle.

**Measurement.** N4 (post-hoc sparse attention via the substrate's `m4t_route_threshold_extract` + `m4t_route_distance_batch` primitives) on BitNet b1.58-2B-4T inference. 24 prompts × 4 arms (dense, random top-k, substrate-routed top-k, oracle top-k) × 6 k values (128, 64, 32, 16, 8, 4) = 456 runs. Pre-committed gates per `journal/partB_experiments_synth.md` and `journal/cycle2_probe_findings.md` (refined methodology after the Phase 2.5 probe).

| k | dense | random | routed | oracle |
|---|---|---|---|---|
| 128 / 64 | 19/24 | 19/24 | 19/24 | 19/24 |
| 32 | 19/24 | 19/24 | 18/24 | 20/24 |
| **16** | 19/24 | **14/24** | **18/24** | 16/24 |
| **8** | 19/24 | **13/24** | **18/24** | 12/24 |
| **4** | 19/24 | **16/24** | **22/24** | 15/24 |

**All three EVIDENCE gates passed:**
- At k=64, routed within 10pp of dense (0pp gap)
- At k=16, routed beats random by **+16.7pp** (75.0% vs 58.3%)
- Gap (routed − random) widens monotonically with sparsity: k=16 +4 prompts → k=8 +5 → k=4 +6

**Surprise result: routed at k=4 (22/24) outperforms dense (19/24)** on three loop-prone prompts (`code_comment`, `edge_question`, `edge_repetitive`). Mechanism (per `journal/loop_regularizer_atomics_2026-05-10.md` and `journal/td27_mechanism_2026-05-10.md`):

- **Loop-breaking mechanism: sparsity itself**, not substrate routing specifically. Dense's softmax distributes mass across all positions; the mid-weight positions cumulatively reinforce loop patterns; aggressive sparsification (random/oracle/routed alike) cuts this off. The substrate-distinct contribution to loop-breaking is producing COHERENT sparse output because signature-distance selection is relevance-aware (random sparsity breaks loops but produces garbage).

- **Routed > oracle gap mechanism: direction-awareness in selection** (TD-27 H2 confirmed). The substrate's trit signatures encode direction (sign + zero) NATIVELY, so signature-distance selection automatically excludes opposite-direction positions. Oracle's `|Q·K|` is direction-blind — it picks high-magnitude negatives that contribute ~0 to softmax (since softmax weight ∝ exp(positive)), wasting sparsity budget. Adding a "posracle" arm that picks top-k by SIGNED Q·K (same logic as oracle but signed not abs) recovered 4 of 5 oracle failures (oracle 3/10 → posracle 7/10 ≈ routed 8/10 on focused subset).

- **Routed vs posracle at full-battery scale (#5 closure, 2026-05-10):** posracle at full battery shows broadly equivalent quality to routed at most k values. At k=8/16/32 the loop heuristic showed posracle "winning" by 3-4 prompts but spot-checking the disagreement prompts revealed heuristic systematically misflags routed's coherent noun-repetition prose as loops; manual review shows both arms produce equivalent quality. At k=4 routed retains a +2 prompt edge that may be real or noise. **The substrate-distinct claim narrows: substrate routing is a COMPETITIVE implementation of direction-aware sparse attention, not a uniquely superior one in this workload at most k.** Per `journal/td27_5_posracle_full_2026-05-10.md`.

- **Substrate-distinct claim, final:** direction-aware sparse attention beats direction-blind sparse attention (NOT substrate-specific — signed-score posracle achieves it). The substrate provides one valid implementation via native trit signatures + popcount distance. Per-step compute cost is potentially lower for substrate routing (popcount on packed trits vs full Q·K); compute-parity verification (TD-24) would settle whether cost-distinct survives quality-parity.

**Substrate-distinctiveness.** HIGH. The routing pipeline (packed-trit signatures via `m4t_route_threshold_extract` with 1/3-quantile-of-|Q| tau choice, popcount distance via `m4t_route_distance_batch`) cannot be replicated on a base-2 substrate without trit packing infrastructure.

**What this is.** First direct Part-B evidence on a real workload, surviving pre-committed gates. The mode-shift framing from `journal/step_change_synth.md` (substrate-building → substrate-testing) was vindicated: one strong inference-only candidate produced direct Part-B evidence without requiring training-first sequencing.

**What this is not.** Not a closure on Part B. One workload (BitNet inference), one model scale (2B params), one decoding strategy (greedy). Loop heuristic is preliminary (manual classification would refine numbers but pattern survives). Oracle baseline is NOT a true upper bound (top-k by |score| is suboptimal vs dense softmax) — the substrate-vs-random comparison is the load-bearing signal. Compute-parity not measured (only quality). Single-seed for random arm.

**Journal cycles.** `journal/step_change_*.md` (LMM-derived mode shift); `journal/cycle1_plan*.md` + `journal/partB_experiments_*.md` (Cycle 1 design + Part-B operationalization + 7-axis scoring rubric); `journal/cycle2_design.md` + `journal/cycle2_probe_findings.md` + `journal/cycle2_full_battery_findings.md` (Cycle 2 execution + verdict).

## Open axes (not yet measured)

- **Strong-claim L2 (activations).** Same shape as L1; likely similar verdict at fixed density. Not yet run.
- **Strong-claim L4 (cross-layer requantization).** Audit's Track A. Currently de-prioritized given the L1 verdict.
- **Strong-claim L5 (cross-exp accumulator).** Requires residual-style workload not produced by GEMM. Not yet run.
- **Strong-claim L6 (post-ternarization activations).** Same shape as L2.
- **Vision claim 3 (broad form).** Each layer's strong-claim cycle is a tile of this broader question. The L1 verdict is the first defensible empirical point.
- **BitNet inference under sampling** (top-k, top-p, temperature). Axis 4 is greedy-only. Sampling may soften some greedy-failure modes (loops); also may introduce different failure shapes. Open.
- **HF-vs-substrate full 24-prompt comparison.** Currently only the 5-failure subset has HF cross-check. A full HF run would establish the substrate's quality delta vs reference across every prompt.
- **Manual classification of all 456 Cycle 2 outputs.** Loop heuristic is preliminary; manual review would refine Axis 5's pass-rate numbers. Pattern (routed > random, gap widens) should survive but absolute rates will shift.
- **Compute-parity verification for Cycle 2 (wall-clock per token).** Sparse attention SHOULD save FLOPs at small k; this hasn't been verified empirically.
- **Cycle 3: routing-native attention with training (N1).** The training-required architectural test of Part B. Cycle 2 gave us evidence in the post-hoc form; Cycle 3 would test whether the gain amplifies under joint optimization.
