# Glyph

A routing-first ternary compute stack for Apple Silicon. Built on the thesis that base-2 systems ignore one-third of the natural signal — the structural zero — and that base-3 silicon primitives (TBL, masked-VCNT, SDOT) are already ternary-shaped underneath the base-2 framings that pave them over.

**Start here:** [`NORTH_STAR.md`](NORTH_STAR.md) — the compass.

## Status (as of 2026-05-10)

Ground-zero rebuild **complete through Tier 3** plus substrate-claim measurements landed AND **first real consumer (BitNet b1.58-2B-4T) producing coherent end-to-end inference** on the substrate. Layers:

**Substrate (libm4t)** — routing-first base-3 compute surface:
- **Tier 1 — pure base-3 layer.** Trit types, packing, element-wise ops (TBL), reductions (masked-VCNT). Zero MTFP entanglement.
- **Tier 2 — route primitives + MTFP19 mantissa arithmetic.** Five route primitives and same-block-exponent MTFP19 add/sub.
- **Tier 3a — cross-exponent accumulator** (`m4t_mtfp_vec_accum_aligning`). NEON-routed via vmlal_s32 magic-multiply; same-exp + flags path productionized; bit-exact verified. Per `journal/cross_exp_accum_routing_*`.
- **Tier 3b — SDOT MTFP4 matmul + cell-width conversions.**
- **Tier 3c — MTFP19 × packed-ternary matmul** (`m4t_mtfp_ternary_matmul_bt`). NEON-routed via vmlal_s32. Per `journal/ternary_mac_routing_*`.
- **shift3 elemental floor primitive** (`m4t_mtfp_shift3`) — base-3 positional scaling via NEON magic-multiply table. Per `journal/shift3_*`.

Project-wide invariants enforced: NEON-only production paths; no scalar fallbacks; bit-exact verification via `_scalar_ref` test oracles inside libm4t.

**Gesh consumer (substrate's first measured consumer):**
- **Phase A.1 — forward pass + synthetic benchmark.** Bank construction, ternary projection, top-k tile retrieval, k-NN vote classification.
- **Phase A.2 — lattice-update training.** Coordinate descent over R's ternary trits, no STE. +11pp gain over random init on synthetic prototype-classification.
- **Expression-routing R1 (FALSIFIED).** Per-expression-tau dual-threshold signature rule methodically falsified across 4 substantive axes. Per `journal/r1_falsify_*`.

**BitNet b1.58-2B-4T inference (substrate's first real-LLM consumer):**
- **30 layers × any prompt length × KV-cached greedy generation.** Default A8 BitLinear path plus a bit-faithful no-A8 path used for substrate-vs-HF localization. Per `gesh/bitnet/`.
- **Quality characterized on a 24-prompt battery** across factual / definitional / narrative / math / code / structured-output / long-context / edge categories. Strict pass rate **~92%** (~22/24, zero hard failures) post-tuning. Per `journal/inference_battery_v2_2026-05-09.md`, `journal/hp_sweep_2026-05-10.md`, `journal/math_div_atomics_2026-05-10.md`.
- **Substrate-vs-HF investigation closed** — a latent silent-saturation bug in `m4t_mtfp_rmsnorm_bx` at `gamma_bx > target_bx` (BitNet's typical regime) collapsed `post_attention_layernorm` outputs by 6.5× and produced degenerate generation loops; fixed at commits `4d4c917` / `8bb78d2`. Per `journal/substrate_vs_hf_2026-05-09/RESOLVED.md`.
- **Hyperparameter retuning** — `BITNET_GATE_ACT_BX 2 → 1` (commit `d2c01ae`) recovered 4 of 5 substrate-specific quality failures; subsequent atomics investigation found math_div was noise-accumulation rather than a single-bug, and the untested `gate1 + score_shift fudge=2` combination (commit `b34c8e4`) recovered the holdout. Final tuned defaults bring strict pass rate to ~92%.

**Part-B evidence (substrate-claim, 2026-05-10):**
- **First direct empirical evidence for thesis Part B** (routing is essential, gap widens with task richness) on a real workload. Cycle 2 of the LMM-derived mode-shift plan executed N4 (post-hoc sparse attention via `m4t_route_threshold_extract` + `m4t_route_distance_batch`) on the 24-prompt × 4-arm × 6-k battery (456 runs).
- All three pre-commit EVIDENCE gates passed: at k=64 routed within 10pp of dense (0pp gap); at k=16 routed beats random by **+16.7pp** (75% vs 58.3%); gap (routed − random) widens monotonically with sparsity (k=16 +4 prompts → k=8 +5 → k=4 +6).
- Surprise finding: routed at k=4 (22/24) outperforms dense (19/24) on three loop-prone prompts. Mechanism hypothesis (not confirmed): aggressive substrate-routed attention selects diverse-by-signature K positions, breaking attention-loop dynamics that dense locks into.
- Per `journal/step_change_*.md` (LMM-derived mode shift), `journal/cycle1_plan*.md` + `journal/partB_experiments_*.md` (Cycle 1 design), `journal/cycle2_design.md` + `journal/cycle2_full_battery_findings.md` (Cycle 2 execution + verdict).

**Substrate-claim measurements (this cycle's landing):**
- **Tri-state utilization audit** (`journal/tristate_op_*`). Two-gate audit of third-state utilization across substrate layers L1-L4 + L6 on a 1.58-bit-LLM-shape workload. Verdict: L1, L2, L6 unambiguously load-bearing per both gates; L4 least load-bearing (mean cos ≈ 0.94); L3 sparsity-dominated.
- **Strong-claim L1 cycle** (`journal/tristate_strong_*` + `audit/`). Compares base-3 packing vs B2-B (sign+mask) base-2 alternative. Multi-round red-team established: at fixed 2 bits/cell density, encoding labels are aliases (Path A ≡ Path C); at sub-2-bit density, base-3 wins (Path D 5-in-8 = 1.6 bits/cell, ~1.8× faster than 2-bit packings on Apple Silicon when both are register-tiled). B2-B is structurally floored at 2 bits/cell because sign+mask are independent.
- **Kernel optimizations** (P0-1 pre-permute X, P0-2 split-LUT decode, P0-3 register-tile by 4 j-cells). Each P0 item went through red-team + remediation. Final cumulative: Path D's wall-clock penalty dropped from 1.16-1.95× to 0.55-0.58× of Path A across all tested regimes (L1-resident through DRAM-bound).

**Substrate-distinctive geometric claim — robustness arc (2026-05-12):**

> **2026-05-13 caveat.** The Phase α-ε oracle activations
> (`data/c_dump_v3`) were generated from BOS=1 gibberish-tokenizer
> prompts, NOT natural language (`journal/cdumpv3_provenance_2026-05-13.md`).
> The findings below are oracle properties on out-of-distribution
> inputs. The only natural-language territory measurement for the
> eviction corollary is `journal/td28_phase_zeta_planB_redteam_2026-05-13.md`
> (qsigdist +6pp vs random, CI [-5.6, +18.1] at N=20). A separate
> measurement of vision claim 2 (math-as-signatures-via-routing)
> landed positive 2026-05-13 — see `journal/claim2_first_measurement_2026-05-13.md`.

- Phase α/β/γ tested the vision's strongest form ("base-3 carries information base-2 collapses") on BitNet K-cache. Six commits and three verdict reversals across categorical Hamming → L1 cell-graph metric → multi-normalization robustness matrix.
- **Original "robust findings" (gibberish-prompt regime):** the centrality of 0 in the path-graph metric is load-bearing across all six methodology variations (ROBUST 6/6); the close-regime substrate compression in within-(layer, kv_head, site) groups is large (47pp d̂/D_amb gap to PCA-binary at equal capacity).
- **Methodology caveat:** the Macocco intrinsic-dim estimator fails calibration by ~45% on correlated synthetic. Absolute d̂ values for real K-cache are biased low (true ≈ 2× reported); relative comparisons remain valid.
- **Net for the vision (revised):** the strong claim was tested on gibberish-prompt activations; a qualified form (substrate captures close-range geometric similarity under the L1 cell-graph metric anchored at 0) was robust within that regime. Re-validation on natural language is open work. Downstream applications must be validated by application-level benchmarks on natural-language inputs.
- Per `experiments/README.md` and `journal/td28_phase_gamma_robustness_2026-05-12.md`; current state in `journal/INDEX_2026-05-13.md`.

Each kernel layer + science cycle was red-teamed adversarially after landing; remediation cycles caught silent invariant violations, strawman comparisons, tile asymmetry, and trajectory extrapolation errors. Full trail in `journal/`.

The prior implementation is preserved on disk in `01MAY26_archived/` (gitignored) as reference. The substrate's complete narrative is in [`CHANGELOG.md`](CHANGELOG.md).

What remains: consumer-side rebuild (libglyph, libtrain, tools). Out of scope for the substrate; planned separately when consumer demand drives it. Strong-claim follow-on cycles for L2/L4/L6 layers are also deferred.

## Discipline

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full set of invariants. The headlines:

- **No binary floating point** in runtime kernels of `libm4t` or per-query/per-batch paths above it. MTFP (base-3) is the substrate-legal continuous geometry.
- **No random projections** in image classification — direct ternary quantization of pixels and gradients is the production representation.
- **No random weights** anywhere — every dimension must represent something specific.
- **No primitive without named consumer demand** — *for derived/composite kernels only*. Foundational primitives named in the vision (the six-primitive floor) are exempt; their demand IS the foundation. See `CONTRIBUTING.md` §5 for the full carve-out and the disclaimer history.
- **Substrate-level specs are upstream of kernel designs** — re-read the relevant spec section before any design memo.
- **DELETE = never.** Superseded code moves to an archive directory; it does not get removed.

## Numerical system

MTFP — Multi-Trit Floating Point, base 3. A value is `mantissa × 3^exponent`. Mantissa is an n-trit signed integer in one of four cell widths; exponent is sidecar metadata at the block level.

| Type | Container | Mantissa trits | Mantissa range | Cells per block |
|---|---|---|---|---|
| `m4t_mtfp4_t` | int8 | 4 | ±40 | 16 |
| `m4t_mtfp9_t` | int16 | 9 | ±9 841 | 8 |
| `m4t_mtfp_t` | int32 | 19 | ±581 130 733 | 4 |
| `m4t_mtfp_w_t` | int64 | 39 | ±1.72·10¹⁸ | 2 |

All blocks are 16 bytes — exactly one NEON vector. See [`m4t/docs/M4T_SUBSTRATE.md`](m4t/docs/M4T_SUBSTRATE.md) for the full contract.

## Build

```bash
# Requires aarch64 + NEON (Apple Silicon or compatible ARM).
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

`-Werror` is on by default. Eight ctest binaries, all green from a clean build.

## Documentation

| File | Purpose |
|---|---|
| [`NORTH_STAR.md`](NORTH_STAR.md) | The vision. Why base-3, why routing, what the end-game is not. Re-read when base-2 gravity pulls. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | The discipline — what every contribution must honor. |
| [`CHANGELOG.md`](CHANGELOG.md) | Complete narrative of the rebuild, commit-by-commit. |
| [`docs/THESIS.md`](docs/THESIS.md) | What would falsify the thesis. Open questions. |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | Running ledger of measurements. |
| [`docs/TECHNICAL_DEBT.md`](docs/TECHNICAL_DEBT.md) | Centralized index of deferred work — functional gaps, follow-on research cycles, doc drift, spec deferrals, open questions. |
| [`docs/TRIT_ROUTING_APPLICATIONS.md`](docs/TRIT_ROUTING_APPLICATIONS.md) | Application surface for the validated trit-routing primitive (direction-aware top-k via packed-trit signatures). 13 items: 4 improvements to the validated sparse-attention application, 3 research questions, 6 applications beyond sparse attention. |
| [`docs/REMEDIATION_PLAN.md`](docs/REMEDIATION_PLAN.md) | Original kernel rebuild plan; preserved as historical artifact. |
| [`docs/REMEDIATION_PLAN_REDTEAM.md`](docs/REMEDIATION_PLAN_REDTEAM.md) | Adversarial review of the plan; 12 findings folded in. |
| [`docs/DESIGN_X-EXPO.md`](docs/DESIGN_X-EXPO.md) | Cross-exponent accumulator design (§14.2 named opt-in). |
| [`m4t/docs/M4T_SUBSTRATE.md`](m4t/docs/M4T_SUBSTRATE.md) | The substrate specification. Canonical reference for the numeric system. |
| [`m4t/README.md`](m4t/README.md) | Substrate-layer build, surface, and test inventory. |
| [`audit/README.md`](audit/README.md) | Strong-claim measurement tools (B2-B comparison kernels, two-gate audit, op-count + wall-clock harness). NOT in ctest — science deliverables. |

### Journal — LMM cycles and red-team records

| File | Purpose |
|---|---|
| [`journal/xexpo_design_raw.md`](journal/xexpo_design_raw.md) → [`reflect`](journal/xexpo_design_reflect.md) → [`synthesize`](journal/xexpo_design_synthesize.md) → [`closeout`](journal/xexpo_design_closeout.md) | LMM cycle that scoped the cross-exponent kernel design before build. |
| [`journal/xexpo_kernel_redteam.md`](journal/xexpo_kernel_redteam.md) | Tier-3a kernel red-team (14 findings, all remediated). |
| [`journal/xexpo_spec_amend.md`](journal/xexpo_spec_amend.md) | Lightweight cycle: §14.2 + §14.4 spec amendments. |
| [`journal/m4t_matmul_redteam.md`](journal/m4t_matmul_redteam.md) | Tier-3b/3c kernel red-team (11 findings, all remediated). |
| [`journal/cross_exp_accum_routing_*.md`](journal/cross_exp_accum_routing_synthesize.md) | Cross-exp accum NEON productionization (vmlal_s32 routing) + red-team + 100/100 remediation. |
| [`journal/ternary_mac_routing_*.md`](journal/ternary_mac_routing_synthesize.md) | Ternary MAC NEON productionization (vmlal_s32 pipeline). |
| [`journal/shift3_*.md`](journal/shift3_neon_synthesize.md) | shift3 NEON productionization (magic-multiply table). |
| [`journal/r1_falsify_*.md`](journal/r1_falsify_closeout.md) | R1 dual-threshold signature rule methodically falsified across 5 axes (4 original + F-G5 held-out routing per TD-8). |
| [`journal/tristate_op_*.md`](journal/tristate_op_closeout.md) | Tri-state utilization audit (L1-L6 third-state load-bearing analysis) + red-team R-G1 (L4 collapse design fix). |
| [`journal/tristate_strong_*.md`](journal/tristate_strong_closeout.md) | Strong-claim cycle on L1: base-3 vs B2-B comparison; multi-round red-team. |
| [`journal/tristate_strong_5in8_addendum.md`](journal/tristate_strong_5in8_addendum.md) | Sub-2-bit base-3 packing (5-in-8) addendum: density-ceiling structural advantage. |
| [`journal/tristate_strong_membw_*.md`](journal/tristate_strong_membw_addendum.md) | Memory-bandwidth regime test + red-team (trajectory plateau finding refutes first-draft crossover prediction). |
| [`journal/p0_kernel_opt_redteam.md`](journal/p0_kernel_opt_redteam.md) | P0-1/P0-2/P0-3 kernel optimizations (pre-permute X, split-LUT decode, register-tile) with per-item red-team. Path D ratio: 1.95× → 0.55× across all regimes. |
| [`journal/strong_claim_retrospective.md`](journal/strong_claim_retrospective.md) | Single-doc consolidation of the strong-claim verdict across 8 refinement rounds. Where to land if you don't want to traverse the cycle history. |
| [`journal/tristate_l4_strong.md`](journal/tristate_l4_strong.md) | TD-4 closeout (v2): A.1 absmean negligible (gap +0.002); A.2 (cohort-selection) shows STRUCTURAL per-cell impact 5.06 vs DECAY 1.67 (3× — SUGGESTIVE); A.3 magnitude-bin negligible. Audit's MIXED verdict (cos 0.946 on Y1==0 cohort) holds. |
| [`journal/tristate_l5_strong.md`](journal/tristate_l5_strong.md) | TD-5 closeout (v2): cross-exp alignment ERASES cancel-90% load-bearingness (Δ=0 cos 0.844 LOAD → Δ=1 cos 0.949 MIXED). Skip-conn (real residual) firmly SINK at all Δ. L5 strong-claim is narrower than v1 reported. |
| [`journal/tristate_l6_strong.md`](journal/tristate_l6_strong.md) | TD-6 closeout (v2): aggregate L6 cos 0.7390 LOAD-BEARING; DECAY cohort drives the verdict (cos 0.7568, 1424 cells); STRUCTURAL is small (106 cells) and MIXED (cos 0.946). Q2 kernel-output equivalence verified by 60/60 byte-identical Path A vs Path C. |
| [`journal/tristate_dram_regime.md`](journal/tristate_dram_regime.md) | TD-9 closeout (v2): tightened gate FAILS — deep-DRAM D/A (0.631) > 0.8 × L1 D/A (0.492 threshold). PLATEAU verdict from membw addendum extends; Path D's ~1.7× advantage is SDOT-amortization-driven, workload-independent, no bandwidth crossover on Apple Silicon. |
| [`journal/large_cycles_redteam_2026_05_06.md`](journal/large_cycles_redteam_2026_05_06.md) | Consolidated red-team of TD-4/5/6/9 v1 with 100/100 remediation. RC-1 through RC-15 finding-and-fix record. |
| [`journal/td7_xpacked_bench.md`](journal/td7_xpacked_bench.md) | TD-7 X-packed §20 wall-clock bench: §20-xp 14-26% faster than §20 across all M/K (mechanism: NEON-vectorized permutation, NOT bandwidth savings as predicted). Single-token inference at K ≥ 4480: §20-xp also beats unpacked. |
| [`journal/no_scalar_audit_2026_05_06.md`](journal/no_scalar_audit_2026_05_06.md) | No-scalar audit: 9 fully-scalar production functions found and NEON-ized (pack/unpack 4-in-8 + 5-in-8, mtfp4↔mtfp19 conversions, shift3 k>0 multiply, k≥20 saturation collapse). ~13,500 new NEON-vs-scalar_ref bit-exact assertions. Substrate now fully NEON in production. |

## License

[MIT](LICENSE).
