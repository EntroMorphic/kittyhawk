# Changelog

Notable changes to Glyph since the 2026-05-01 ground-zero rebuild. Older entries are preserved in `01MAY26_archived/CHANGELOG.md`.

## [Unreleased]

### Changed — Large-cycle (TD-4/5/6/9) red-team + 100/100 remediation (2026-05-06)
Per user-requested red-team after the four-cycle batch landed. RC-1 through RC-15 documented in `journal/large_cycles_redteam_2026_05_06.md`; all benches, closeouts, and verdicts updated to v2.

**Critical findings remediated:**
- **RC-1 (TD-5):** v1 bench did NOT invoke `m4t_mtfp_vec_accum_aligning` despite the cycle being named for cross-exp accum (used plain int32 addition). v2 calls the cross-exp primitive with explicit Δexp ∈ {0, 1, 3}. Critical new finding: cross-exp alignment ERASES the cancel-90% load-bearingness (Δ=0 cos 0.844 LOAD → Δ=1 cos 0.949 MIXED). L5 strong-claim is much narrower than v1 reported.
- **RC-2 (TD-6):** v1 verified base-3 ↔ B2-B round-trip preservation and overclaimed it as R-G1 generalization. v2 replaces with real kernel-output equivalence test (Path A vs Path C at L6 inputs); 60/60 byte-identical.
- **RC-3 (TD-4):** v1's "cohort-size confound" framing misrepresented the audit. v2 reframes honestly: the audit's Y1==0 cohort IS the deliberate L4 definition.
- **RC-4 (TD-9):** v1's pre-committed gate ("D/A < 1.0 at any DRAM-bound config") was trivially met because Path D was already winning at L1. v2 tightens to require monotone improvement (deep-DRAM D/A ≤ 0.8 × L1 D/A). New gate FAILS — confirms PLATEAU verdict.
- **RC-5 (TD-9):** K=51200 isn't a real ML workload. v2 marks K=25600 / K=51200 rows as sanity-check shapes; verdict based on realistic-K (K ≤ 12800).

**Important findings remediated:**
- **RC-6:** per-cell impact metric flagged SUGGESTIVE only (non-linear); applied across TD-4/5/6 closeouts.
- **RC-8 (TD-9):** deep-DRAM reps doubled from 2-3 to 5-10.
- **RC-9/RC-10 (TD-4):** A.2 and A.3 implemented as cohort-selection rules (no substrate extension needed — v1 deferred too aggressively). A.2 shows STRUCTURAL per-cell 5.06 vs DECAY 1.67 (3×, SUGGESTIVE); A.3 negligible.
- **RC-11 (TD-6):** Q1 strengthened with per-cohort cos breakdown (ALL/STRUCTURAL/DECAY).
- **RC-12 (TD-5):** new SKIP_CONN regime (independent matmul as residual, no anti-correlation). Result: SINK at all Δ.

**Process lift:**
1. Read cycle name back to implementation at synthesize-time (TD-5 RC-1 caught here).
2. Pre-committed gates must require directional shift, not just threshold crossing.
3. Per-cell impact metrics need explicit "suggestive only" tagging.
4. "Substrate extension required" is a convenient deferral — check if reframing within existing infra works first.
5. Red-team between cycles, not just at the end of a batch.

### Added — TD-9 closure: DRAM-bound regime test (Path D wins consistently across full W range) (2026-05-05)
Extends `tristate_strong_membw_addendum.md`'s sweep from W = 25.6 MB up to W = 200 MB. Compares Path A (4-in-8 packed W) vs Path D (5-in-8 packed W) at 9 configs spanning L1-resident to far-past-DRAM, with cache-flush + warmup discipline mirrored from the existing strong-claim bench.

D/A ratios: 0.625 (L1) → 0.571 (L2) → 0.554 (3.2 MB) → 0.583 (25.6 MB) → 0.611 (51.2 MB) → 0.584 (102 MB) → 0.573 (204 MB). Path D wins by ~1.6-1.8× at every regime. The ratio is roughly stable across the W spectrum (no monotone decrease that would indicate a true bandwidth-driven crossover).

**Verdict:** the membw addendum's "PLATEAU not crossover" finding extends to W = 200 MB. Path D's advantage is workload-independent — driven by SDOT amortization (per `journal/p0_concern1_mechanism.md`), not by Path D's 0.8× density advantage. Apple Silicon's unified memory bandwidth (~70-200 GB/s) is generous enough that decode work saved by SDOT amortization dominates over the bandwidth savings from tighter packing throughout the tested range. True DRAM-bound crossover may exist on hardware with tighter bandwidth/compute ratio (older ARM, embedded, non-Apple Silicon); not a substrate finding. Closes TD-9 per `journal/tristate_dram_regime.md`.

### Added — TD-6 closure: L6 strong-claim cycle (LOAD-BEARING + encoding equivalence verified) (2026-05-05)
Two-question cycle. **Q1 (load-bearingness):** Gate II at L6 cohort (X2==0 cells) gives mean cos = 0.7390 across 12 configs × 5 seeds — LOAD-BEARING (cos < 0.85), matches original audit's reported cos_L6 ≈ 0.74 within RNG variance. **Q2 (encoding-label equivalence at L6):** base-3 ↔ B2-B round-trip preserves all per-cell trit values across 60/60 runs. The L1 R-G1 verdict (encoding-label equivalence at L1, established by disasm comparison) generalizes to L6 by direct round-trip evidence rather than the previous symmetry argument. Closes TD-6 per `journal/tristate_l6_strong.md`.

### Added — TD-5 closure: L5 cross-exp accum strong-claim (consumer-pattern-dependent) (2026-05-05)
Tests L5's third state (exact-zero output of cross-exp accumulation) across four residual regimes. Workload pattern: `Y_post = Y_pre + R` where Y_pre is a ternary GEMM output and R varies per regime.

Results (mean cos across 12 configs × 5 seeds):
- Cancel 90%: cos = 0.844 (LOAD-BEARING)
- Cancel 50%: cos = 0.930 (MIXED)
- Decay (small-exp): cos = 0.954 (SINK aggregate, but highest per-cell impact at 4.085)
- Independent: cos = 0.992 (SINK)

**Verdict:** L5 IS load-bearing in residual-style workloads with structural cancellation. SINK in independent-residual workloads. Confirms why GEMM-only audits gave no L5 verdict — they don't exercise cross-exp accum scenarios. Per-cell, even decay-regime zeros are highly load-bearing (4.085 per-cell impact, the highest of any cohort). Closes TD-5 per `journal/tristate_l5_strong.md`.

### Added — TD-4 closure: L4 strong-claim cycle (per-cell load-bearing; A.1 no improvement; A.2/A.3 substrate-deferred) (2026-05-05)
Two-axis test on L4's third state.

**Part 1 (cohort-definition sensitivity):** The audit's verdict that L4 is "least load-bearing" was driven by cohort SIZE, not per-cell weakness. The audit's Y1==0 cohort is the SMALLEST tested (~106 cells avg) and has by far the HIGHEST per-cell impact (5.06 ×10000), 3× higher than the broader X2==0 cohort (1.75) or near-threshold cohort (2.13). Reframe: L4's third state is small in count but each cell carries disproportionate downstream weight.

**Part 2 (A.1 test on L4 cohort):** Comparing absmean (BitNet b1.58) rule vs quantile rule on the same Y1==0 cohort: cos 0.946 (quantile) vs 0.944 (absmean), gap +0.002 — well below the 0.05 verdict threshold. A.1 does NOT meaningfully change L4's load-bearingness.

**RC-1 (caught pre-execution):** A.2 (zero-flag forwarding) and A.3 (two-channel sign+magnitude) require Layer 2 matmul augmentation (4- or 5-state input instead of ternary). Implementing them is a multi-cycle substrate extension; both are documented as design-only with explanation.

**Methodology lift:** TD-4's RC-1 (cohort-size confound) was applied as a check against TD-5 and TD-6 — both report per-cell impact alongside aggregate cos to avoid the same artifact.

Closes TD-4 per `journal/tristate_l4_strong.md`.

### Added — TD-8 closure: F-G5 (held-out routing accuracy) for R1 (2026-05-05)
Closes the 5th axis of the R1 falsification matrix that the original 4-axis closeout deferred for "external equivalence ground truth requires substantial engineering."

Method: K_TOTAL = 8000 random arity-1 expressions; behavioral fingerprint via int64 evaluation on N_FP = 32 fixed inputs; group by fingerprint into equivalence classes; filter to ≥ 4 members per class; per class use first member as bank anchor and rest as held-out test set; build sign-only and dual signature banks (one tile per class); route held-out exprs and tally accuracy.

Pre-committed gate: |dual − sign-only| ≥ 2 pp triggers verdict.

**Red-team RC-1 caught BEFORE finalizing.** First run used the wide {−30..30} input band and showed dual beating sign-only by +8.23 pp — apparent verdict shift. RC-1 identified that depth-4 random expressions on wide inputs overflow int64 (max |x|^16 ≈ 10^23, vs int64 ceiling ≈ 10^19), fragmenting equivalent expressions into spurious distinct classes and biasing the test set toward "trivial" classes where dual happens to win. Remediation: rerun with tight {−3..3} band (max |x|^16 ≈ 43 M, well within int64). Bench now runs both bands; tight is the canonical verdict, wide is reported as a sanity-check.

Result on canonical (tight, no-overflow) configuration:
- sign-only routing accuracy: 23.68% (1448 / 6116)
- R1 dual routing accuracy:   21.09% (1290 / 6116)
- gap: −2.58 pp (dual underperforms by > 2 pp)
- per-class breakdown: 18 classes dual-better, 20 classes dual-worse out of 195 — essentially symmetric, no systematic dual advantage

**R1 status: methodically falsified across 5 substantive axes (was 4).** Per `journal/r1_falsify_f_g5.md` and updated `journal/r1_falsify_closeout.md`. README updated to reflect 5-axis verdict.

### Added — TD-7 closure: §20 X-packed sibling `m4t_ternary_5in8_matmul_xpacked_bt` (2026-05-05)
Symmetric to §20 (5-in-8 W) but with X also packed at 5-in-8 (1.6 bits/cell). Per i, decodes `X_packed[i, :]` into the same 5 stride-aligned int8 arrays via the split-LUT pattern (1× div-by-9 magic-multiply + 5× vqtbl1q/vqtbl2q lookups per 16-byte chunk; scalar geometric tail for trailing Kp%16 bytes). Tile body identical to §20: 5 SDOTs × 4 j cells per 80-trit chunk. Same arbitrary-(K, N) shape support as §20 (K%80 + N%4 tail handling per TD-1).

Verification (`test_m4t_ternary_5in8_xpacked.c`):
- **G1 — NEON vs scalar_ref:** bit-exact across aligned (K ∈ {80, 160, 320, 640} × N ∈ {4, 16, 64} × M ∈ {4, 8, 16}) and tail (K ∈ {5, 17, 85, 159, 161, 287} × N ∈ {1, 2, 3, 4, 5, 7, 16}) configurations.
- **G2 — cross-equivalence with §20:** xpacked kernel produces the same Y as `m4t_ternary_5in8_matmul_bt(X_unpacked, W_packed, ...)` when `X_unpacked = unpack(X_packed)` and the same W_packed is reused. Strong cross-check: any X-decode bug surfaces as a Y mismatch against the canonical X-unpacked kernel (catches silent decode-side drift that G1 alone could miss if scalar_ref shared the same bug).

22/22 ctest binaries green. Spec §20.6 added documenting the X-packed sibling and the bench-deferral rationale (primitive ships per project rule; wall-clock comparison vs §20 deferred until a consumer demands it). Closes TD-7.

### Changed — TD-1 closure: §20 `m4t_ternary_5in8_matmul_bt` accepts arbitrary K and N (2026-05-05)
Relaxed the strict `K % 80 == 0` and `N % 4 == 0` preconditions on `m4t_ternary_5in8_matmul_bt`. Tile body unchanged: still 5 SDOTs × 4 j cells per 80-trit chunk, register-tile-by-4. Two new tail paths bring non-aligned shapes to bit-exact correctness without breaking the project's no-scalar-in-production rule:
- **K%80 tail** — per-trit scalar accumulation for the trailing K%80 trits (geometric sub-block scalar tail; allowed by project rule). 4 j cells in lockstep.
- **N%4 tail** — single-acc NEON inner loop covering full K (tile body + K-tail) for each of the trailing 1-3 j cells. NEON throughout the bulk; no scalar fallback for the main path.

`m4t_ternary_5in8_matmul_bt_scalar_ref` updated symmetrically (uses `Kp = (K+4)/5` for K%5 != 0 support). Test `test_m4t_ternary_5in8_matmul.c` extended with K-tail-only / N-tail-only / both-tails coverage (K ∈ {5, 17, 85, 159, 161, 287}, N ∈ {1, 2, 3, 5, 6, 7, 17}).

**Red-team caught a test bug:** initial test allocation used `Kp = K/5` (pre-relaxation assumption) but the new kernel uses `Kp = (K+4)/5`. For K%5 != 0 this under-sized W_pkd by N bytes, causing W_pkd to overlap Y_ref in heap layout. The bug surfaced as a single-trit mismatch only when both kernels ran back-to-back (Y_ref's writes overwrote W_pkd's last byte before scalar_ref read it). Fix: test now uses `Kp = (K+4)/5` consistently. Mechanism is a useful artifact: silent buffer-size drift between caller and kernel is exactly the class of bug that grows with parameter relaxations, so the test discipline now matches the kernel contract.

Spec §20.4.1 added to document the alignment recommendation (best throughput when K%80==0 and N%4==0; off-alignment shapes are functionally correct but pay a small per-call overhead). Header docs in `m4t_ternary_matmul.h` updated; the previous "future work" note is removed. Closes TD-1 from `docs/TECHNICAL_DEBT.md`.

### Added — `docs/TECHNICAL_DEBT.md` centralized debt index (2026-05-05)
Per session housekeeping. Project had no central debt-tracking doc; debt was scattered across journal closeouts' "future work" sections, THESIS.md "open questions", the pending task list, and spec-level deferrals in M4T_SUBSTRATE.md. New doc consolidates 18 deferred items organized by category (functional gaps / open follow-on cycles / housekeeping / spec deferrals / open questions / methodology debts). Each entry includes Source (cycle/file pointer), State, Unblocks, Priority hint. README.md Documentation table updated; CONTRIBUTING.md post-commit checklist gained a TECHNICAL_DEBT-currency rule (cycles closing with deferred work add an entry; resolved items get removed). Subsequent commit closed TD-12 (task #87 A-G6 was completed inline in the cross-exp accum routing test; tracker was just stale).

### Added — Production-shoring of audit-validated work into libm4t (3 items + red-team)
Items derived from "what's missing from production code that we have successfully demonstrated" — bringing audit-validated kernel patterns into the substrate where consumers can use them.

- **Item 1 — register-tile both libm4t matmul kernels.** `m4t_mtfp_ternary_matmul_bt` (vmlal route via new `ternary_dot_vmlal_x4` helper) and `m4t_mtfp4_sdot_matmul_bt` (SDOT route) restructured to use 4 parallel accumulator chains per outer iteration, with N%4 tail handled by the original single-cell NEON path (geometric tail rule). Wall-clock measurements (M=8, N=64, K∈{1280, 12800, 51200}, before/after via `git stash`): vmlal route 2.0-2.5×; SDOT route 2.5-3.9×. Bit-exact preserved (21/21 ctest including property tests). New `m4t/tests/bench_m4t_matmul_tile.c` for reproducible measurement.
- **Item 2 — 5-in-8 base-3 packing in libm4t.** New §20 in M4T_SUBSTRATE.md (substrate spec amendment per project rule 7). Pack/unpack primitives (`m4t_pack_trits_5in8_1d`, `m4t_unpack_trits_5in8_1d`, `M4T_TRIT_PACKED5_BYTES` macro). New matmul kernel `m4t_ternary_5in8_matmul_bt` (NEON-only, K%80==0 + N%4==0 strict) ported from audit Path D with split-LUT decode + register-tile-by-4. Test oracle `_scalar_ref`. New ctest binary `test_m4t_ternary_5in8_matmul` (600 NEON-vs-scalar bit-exact samples + 7 hand-derived golden values + pack/unpack roundtrip across 7 sizes). Audit cross-check extended to `tristate_strong_bench` (libm4t §20 vs audit Path D bit-exact across 80 runs).
- **Item 3 — `sdot_pipeline_bench` moved from `audit/` to `m4t/tools/`.** Joins `bench_vmlal_throughput`, `bench_accum_baseline`, `gen_pow3_magic` per project tools convention (manually compiled per file headers; not in cmake build). Spec §17 cross-reference updated: m4t/tools/ is now active.
- **Red-team caught CRITICAL rule violation.** Item 2's `#else m4t_ternary_5in8_matmul_bt_scalar_ref(...)` was a "fall back to scalar when X" pattern — exact prohibited form per project rule + memory. The "defensive unreachable" framing didn't excuse it (rule is about CODE PRESENCE, not runtime behavior). Replaced with `#error` directive. Build hard-fails on any platform without NEON+DOTPROD; converts runtime fallback into compile-time guarantee.
- **Reframing on Item 2 wall-clock.** Direct libm4t measurement showed §20 is 1.14-1.5× SLOWER than `m4t_ternary_dot_matmul_bt` (which uses UNPACKED W = 8 b/c, no decode). Audit's "1.8× advantage" was 5-in-8 vs 4-in-8 (both packed) — different comparison axis than libm4t's 5-in-8 vs unpacked. §20's value prop in libm4t is the storage-vs-decode tradeoff (5× tighter storage at modest compute cost). Spec §20.4 already framed this correctly; reframing aligns the verdict with the spec.

### Methodology lifted from production-shoring
- **Even "defensive unreachable" `#else scalar` paths violate the no-scalar rule.** The rule is about code presence, not runtime reachability. Use `#error` to convert runtime fallbacks into compile-time guarantees.
- **Apples-to-apples comparison must use the same axis.** Audit's Path D vs Path A was both packed; libm4t's §20 vs ternary_dot is packed-vs-unpacked. Don't carry one cycle's wall-clock numbers into a different comparison context without checking the axes line up.
- **Audit cross-check is the right verification, but should run early.** When porting audit-validated code to libm4t, run cross-check immediately after first kernel build, not after all infrastructure (tests, CMakeLists, etc.) is in place.
- **Before/after wall-clock measurement via `git stash` + same bench binary** is the right discipline for "preserve correctness, prove speedup" cycles. Single binary, two source-tree states, identical inputs + sampling — eliminates measurement-shape confounds.

### Added — P0-Concern-2 L2 strong-claim cycle (verdict generalizes partially)
Per session self-review concern: strong-claim verdict was L1-only; generalizing to L2/L4/L5/L6 would be premature. New audit kernels Path E (base-3 4-in-8 packed X + W) and Path F (B2-B 4-in-8 packed X + W). Disassembly comparison: Path E ≡ Path F byte-for-byte at inner-loop level — encoding-label equivalence (R-G1 finding from L1) extends to L2 by structural symmetry, now empirically verified rather than only argued. Wall-clock: Path E ~20% slower than Path A across all regimes (X decode adds cost without bandwidth savings at L1-resident M=8 workloads). **L1 verdict's two components scope differently:** encoding-label equivalence at fixed density EXTENDS to L2 (verified); 1.8× wall-clock advantage does NOT extend at L1-resident workloads (decode cost > bandwidth savings since X stays L1-resident at M=8). Methodology lift: structural symmetry arguments DO transfer across layers; empirical wall-clock claims DO NOT — they require per-layer measurement.

### Added — P0-Concern-1 direct SDOT-amortization mechanism measurement
Per session self-review concern: the 1.8× wall-clock advantage of Path D over Path A was attributed to "SDOT amortization" but never directly measured. New `audit/sdot_pipeline_bench.c` (later moved to m4t/tools/) measures pure SDOT throughput on M-series at three controlled scenarios: T1 (1 acc chain, latency-bound) = 0.33 SDOTs/cycle; T2 (4 acc chains) = 1.52 SDOTs/cycle; T3 (8 acc chains) = 3.08 SDOTs/cycle (peak). Production kernels' measured rates: Path A = 0.46/cycle (30% of T2); Path D = 0.82/cycle (54% of T2). Ratio = 1.78× — matches wall-clock 1.8× exactly. **Mechanism empirically confirmed:** SDOT dispatch density IS the wall-clock determinant. Both kernels far from peak (decode/load-bound); Path D's 5 SDOTs per outer block vs Path A's 1 amortizes setup overhead, allowing more SDOT dispatch slots per cycle.

### Added — Documentation gap shoring (4 surfaces updated)
Identified 4 documentation gaps from the recent audit + strong-claim landings: (1) `audit/` directory had NO README; (2) top-level README Status section dated 2026-05-01, predating audit + strong-claim + P0 work; (3) docs/FINDINGS.md said "no benchmark axes yet" but 3 substrate-claim axes had landed; (4) CONTRIBUTING.md missing methodology lifts from recent cycles. All addressed: new `audit/README.md` orienting the directory; README.md Status refreshed; FINDINGS.md gained 3 new axes (R1 falsification, tri-state utilization audit, strong-claim L1); CONTRIBUTING.md gained 5 methodology lifts (NEON-vs-NEON cross-check + external grounding, tile fairness, disassembly hidden costs, trajectory extrapolation risk, substitution-collapse threshold survival).

### Added — P0-1, P0-2, P0-3 kernel optimizations on Path D (audit cycle)
Three kernel optimization items applied iteratively to audit's Path D (5-in-8 base-3 packing), with red-team between each. Cumulative result: Path D's wall-clock penalty vs Path A dropped from 1.16-1.95× to 0.55-0.58× (Path D ~1.8× FASTER than Path A apples-to-apples).
- **P0-1:** pre-permute X via row-level scalar permutation into 5 stride-aligned arrays (vld5_s8 doesn't exist on NEON; fall-back to explicit pre-permutation). Inner-loop X gather replaced by direct vld1q_s8. ~30% reduction in op count; restored trajectory toward crossover at memory-bound regimes.
- **P0-2:** split-LUT decode replacing magic-mul cascade. New 5 LUTs (2× 16-byte for low digits via vqtbl1q + 3× 32-byte for high digits via vqtbl2q). Critical red-team save: vqtbl4q with 64-byte LUTs caused mov.16b padding (register-allocation overhead); switching to vqtbl2q with 32-byte LUTs eliminated padding. Drop from 1.01× to 0.82× of Path A at L1-resident; near-tie at memory-bound.
- **P0-3:** register-tile by 4 j cells. 4 parallel SDOT chains pipeline better on M-series. Critical red-team save: initial implementation only tiled Path D — apparent 3× advantage was tile asymmetry. Remediation: also tiled Path A and Path C for apples-to-apples comparison. Honest 1.8× win preserved across all regimes.

### Methodology lifted from P0-1/P0-2/P0-3
- **Always disassemble after kernel optimization.** Op count from intrinsics doesn't map 1:1 to ASM. The compiler's register allocator can introduce hidden costs that small LUT-size adjustments can eliminate (vqtbl4q → vqtbl2q saved ~5 ops).
- **Tile fairness in kernel comparisons.** If you optimize one kernel's outer-loop structure (tile, prefetch, etc.), apply the same to comparable kernels before claiming structural wins. Tile asymmetry inflates the apparent advantage.
- **SDOT amortization is structural to packed-W kernels.** Denser packing → more SDOTs per setup overhead → better SDOT pipeline saturation. This is the kernel-cost expression of the density-ceiling structural advantage.

### Added — Strong-claim memory-bandwidth-regime test + red-team (trajectory PLATEAUS)
Tests whether the storage-vs-decode tradeoff inverts when W exceeds L1, L2, then DRAM cache levels. Added 4 configs to `audit/tristate_strong_bench`: K=12800 (W=200KB, exceeds L1), K=25600, K=51200, K=12800/N=8192 (W=25.6MB, exceeds L2 → DRAM-bound). Cache-flush helper between kernels (32MB buffer); per-config standard deviation reporting; per-config REPS scaling.

**Critical finding from red-team R-G1/R-G2/R-G3:** the first-draft "trajectory toward crossover" prediction was wrong. Path D's wall-clock ratio narrows from 2.10× (L1-resident) to 1.16-1.24× (memory-bound) and then PLATEAUS — does not crossover at any tested regime including DRAM-bound. Apple Silicon's unified memory bandwidth (~70-200 GB/s) is generous enough that 5.1 MB savings per call (Path D vs Path A at K=12800/8192) only saves ~0.05 ms. The decode penalty (~71% more NEON ops) costs much more than that bandwidth savings. Substrate (4× more bytes) is STILL slightly faster than Path A at DRAM-bound (0.97× ratio) — confirming memory bandwidth is not the bottleneck on M-series for these workloads.

The strong-claim's defensible foothold is the **density ceiling alone** — base-3 can pack below 2 bits/cell where B2-B cannot. Whether that matters depends on whether the metric is memory-cost (yes) vs throughput on M-series (no).

### Methodology lifted from membw addendum red-team
- **Trajectory extrapolation is risky.** First-draft predicted DRAM crossover from L2-trajectory data; actual DRAM measurement showed plateau, not crossover. Always test the destination, not extrapolate from interior points.
- **Cache-flush between kernel runs is essential for memory-regime measurements.** Without it, kernel n+1 finds W warm from kernel n. Use a 32MB flush buffer (exceeds M-series L2) for cold-cache isolation.
- **SD reporting catches noise floor early.** Per-config standard deviation alongside mean confirms measurement reliability without burying it in raw CSV.

### Added — Strong-claim sub-2-bit base-3 packing addendum (Path D)
Adds Path D (base3_5in8_matmul_neon) to test whether base-3 has a STRUCTURAL density advantage where B2-B cannot follow. Base-3 packs at 1.6 bits/cell (5-in-8 packing); B2-B is structurally floored at 2 bits/cell because sign and mask are independent — there is no analogous sub-2-bit packing. Initial Path D wall-clock: 1.77-1.96× of Path A across L1-resident regimes (decode penalty dominates at cache-resident workloads). Density advantage of 1.25× confirmed at the storage layer; kernel-cost direction at sub-2-bit on Apple Silicon is initially decode-bound (later improved by P0-1/P0-2/P0-3).

### Added — Strong-claim L1 cycle (red-team CONDITIONALLY remediated to "encoding-label equivalence")
LMM cycle on the strong claim. Built `audit/b2b_matmul.{h,c}` with NEON-only kernels: Path A (base-3 4-in-8 via SDOT), Path B (B2-B sign+mask honest decode), Path B-skip (B2-B with all-masked-block skip), Path C (B2-B optimal unified-LUT). Bench `audit/tristate_strong_bench.c` with NEON-vs-NEON cross-check + external grounding via libm4t's `m4t_ternary_dot_matmul_bt`.

**Pre-red-team verdict:** "STRONG CLAIM SUPPORTED" — Path A (7 ops) < Path B-honest (10 ops) < Path B-skip (13 ops) per disassembly. Wall-clock matches.

**Red-team CRITICAL save (R-G1):** B2-B-honest was a strawman. A skilled implementer would use unified TBL decode (Path C), not separate sign+mask extraction. Added Path C; disassembly shows it's byte-for-byte identical to Path A (only LUT contents differ). At 2 b/c density, "base-3" and "optimal B2-B" are aliases — same kernel with different LUT entries. **Verdict shifted to "STRUCTURAL ADVANTAGE" framing → "ENCODING-LABEL EQUIVALENCE at fixed density."** External grounding (R-G2) via substrate cross-check confirmed Path A/Path C correctness (60/60 bit-exact).

### Added — Tri-state operationalization audit (LMM cycle: where is the third state load-bearing?)
Two-gate audit (Gate I info-theoretic + Gate II algorithmic-dependence) of third-state utilization across substrate layers L1-L4 + L6 on a 2-layer ternary GEMM workload modeling 1.58-bit LLM forward pass. 60 runs (12 configs × 5 seeds). Realism gate 60/60 PASS.

**Initial verdict:** L1, L2, L6 LOAD-BEARING per both gates; L3 sparsity-dominated MIXED; **L4 UNDER-EXPLOITED with cos=1.000** (third state at L4 invisible to downstream). Headline: "L4 is the highest-leverage operationalization target."

**Red-team CRITICAL save (R-G1):** L4's cos=1.000 was an artifact. The L4 collapse design substituted median-magnitude values that the downstream quantile threshold reabsorbed back to zero — collapse silently undid itself. Redesigned to override-after-ternarize semantics. Re-run results: L4 cos=0.86-0.99 (MIXED, not invisible). Headline shifted from "UNDER-EXPLOITED" to "MIXED — least load-bearing measured layer." No layer is "broken/invisible"; substrate's third state is broadly load-bearing across measured layers. Methodology lift: substitution-based collapse must validate it survives downstream thresholds.

### Methodology lifted from tri-state audit + strong-claim cycle
- **Substrate utilization vs comparative advantage are different claims.** Intra-substrate utilization (weak claim) doesn't establish comparative advantage over base-2 alternatives (strong claim). Memory `feedback_substrate_claim_scope` saved.
- **NEON-vs-NEON cross-check + external grounding** is sufficient verification when no scalar reference exists. Audit kernels cross-checked against substrate's externally-validated `m4t_ternary_dot_matmul_bt` (60/60 bit-exact).
- **Two-gate design (info-theoretic + algorithmic) catches different failure modes.** Gate I caught L3's entropy collapse; Gate II flagged L4 collapse-design artifact.

### Added — R1 dual-threshold signature rule METHODICALLY FALSIFIED across 4 axes
LMM falsification cycle on the per-expression-tau dual-threshold signature rule (R1) from the prior expression-routing work. Pre-committed gates per axis: F-G1 (class count + intra-class consistency), F-G2 (inter-class minimum distance), F-G3 (partition-change rate), F-G4 (substrate-novelty / third-state utilization), F-G5 (held-out routing accuracy — DEFERRED).

**Verdict: R1 METHODICALLY FALSIFIED across 4 substantive axes.** F-G2 dual=1 vs sign-only=3 (FAIL); F-G3 4.2% partition change vs ≥30% gate (FAIL); F-G4 arity-1 zero-band 66.5% — third state OVER-DOMINATES (FAIL); F-G1 +36-41% more classes (WEAK SUPPORT — non-quality metric). The dual-threshold rule does NOT outperform sign-only on any quality-of-discrimination metric. Scope: R1 is one operationalization of vision claim 3; broader claim 3 remains testable.

### Added — Project-wide no-scalar audit (4 dead production fallbacks removed)
Audit of all `#if M4T_HAS_NEON ... #else scalar ... #endif` patterns in production dispatchers, per the just-saved no-scalar rule. Found 4 dead production fallbacks: `m4t_mtfp_block_add` (lines 30-46), `m4t_mtfp_block_sub` (50-66), `m4t_mtfp_shift3` dispatcher (738-742), `m4t_ternary_matmul.c::ternary_dot` dispatch (104-108). All 4 removed. 20/20 ctest still PASS. The cross-exp accum cycle's H1 finding (inherited scalar fallback in same-exp+flags branch) was the canary; this audit closed the broader pattern.

### Added — cross-exp accum routing red-team remediation (8 R-G gates PASS)
Per `journal/cross_exp_accum_routing_redteam.md` and `..._remediation_*.md`. The red-team caught 10 findings — load-bearing one was a violation of the just-saved no-scalar production rule (same-exp + flags!=NULL fell back to scalar via inherited code path). All closed.

- **R-G1 (H1 fix — no-scalar rule violation):** new `accum_same_exp_with_flags_neon` static helper. Pipeline per 4 cells: vaddq_s32 + min/max clamp + cmeq for SATURATED + per-lane flag OR. Stays in int32 throughout. The dispatcher now: same-exp + flags=NULL → vec_add_inplace (existing); same-exp + flags!=NULL → new helper (NEW). Cross-exp branches unchanged. **No scalar fallback in production for any (delta, flags) combination.**
- **R-G2 (C1 cross-exp saturation):** added 2 cross-exp saturation cases (delta=1, running=±MAX_VAL aligned to ±MAX_VAL/3, addend=±MAX_VAL → sum overflows MAX_VAL → clamp). Both PASS, saturation actually triggered, NEON matches scalar (output + SATURATED flag).
- **R-G3 (C2 _neon API cleanup):** removed `m4t_mtfp_vec_accum_aligning_neon` from public API. Body inlined directly into `m4t_mtfp_vec_accum_aligning`. Test calls updated to production function. Cleaner public surface.
- **R-G4 (M1 dispatcher inlining):** `otool -tv` shows `bl _accum_aligning_neon_block` in the dispatcher (1 call site; compiler merged the two cross-exp branches). Helper not inlined. ~5-10 cycle per-call overhead. Documented; not a fix-needed.
- **R-G5 (L2 closeout correction):** original closeout claimed "all lessons applied at cycle start." Amended with header note documenting that the H1 inherited violation was scope-missed. Original analysis preserved; correction appended.
- **R-G6 (H2 + audit-time rule, methodology lifts):** CONTRIBUTING.md throughput-microbench-discipline checklist extended:
  - "REFLECT NEON-vs-scalar speedup estimates should bound by compiler auto-vectorization of the scalar baseline" — concrete example: this cycle's REFLECT estimated 12-20×; measured 1.6-6× because compiler vectorized scalar.
  - NEW post-commit checklist item "No-scalar audit": apply the no-scalar rule at AUDIT time to inherited code, not just to new code. List every function the new dispatcher delegates to; verify each is NEON-only in production. The cross-exp cycle's H1 was the lesson: rule was applied to NEW code only.
- **20/20 ctest PASS.** All 1030+ bit-exact configurations still match (curated + 1000 random + saturation-edge same-exp + saturation-edge cross-exp NEW). 3 production binaries identical.

### Methodology lifted from cross-exp accum remediation
- **Audit-time application of the no-scalar rule.** Project rule (no scalar in production) must be checked against every function the dispatcher delegates to, not just the new code. Inherited fallback patterns are the dangerous ones; they're easy to miss.
- **REFLECT estimates of NEON-vs-scalar speedup must account for compiler auto-vectorization.** Treat the scalar baseline as "what -O3 actually emits," not "what hand-written naive scalar would do." Estimates that don't account for this overshoot by 2-4×.
- **Saturation-edge tests should cover EVERY branch.** Don't skip the "harder-to-construct" branch — constructed cross-exp saturation (delta=1, MAX_VAL inputs) is straightforward and worth the lines of code.
- **Productionization removes the prototype wrapper.** Any `_neon` / `_vmlal` / `_path` prototype function ships gets folded into the dispatcher at productionization, not left in the public API as a courtesy. The cross-exp cycle initially missed this; remediation cleaned up.

### Added — cross-exp accumulator routing through vmlal_s32 (9 A-G gates PASS)
Per `journal/cross_exp_accum_routing_{raw,nodes,reflect,synthesize,closeout}.md`. The user named per-block-exponent management as "software doing the work of hardware" — the ternary equivalent of an IEEE FPU's internal align+round step. Cycle scope: compose existing shift3-divide pipeline + block_add into a fused accumulator inner loop.

- **STRUCTURAL CHANGE:** `m4t_mtfp_vec_accum_aligning` cross-exp branches (addend>running, running>addend) now route through a NEON helper (`accum_aligning_neon_block`) using the SAME `vmlal_s32` magic-multiply pipeline as `m4t_mtfp_shift3`. `m4t_pow3_magic.h` is now SECOND-consumer validated (was: shift3 only). Same-exp branch unchanged (already NEON-fast via `vec_add_inplace`). Degenerate-delta branches (delta ≥ 20) unchanged (memcpy + flag annotation).
- **NEW: `m4t_mtfp_vec_accum_aligning_scalar_ref`** — public scalar-only test oracle. Exposed BEFORE prototype work (A-G1) per shift3 remediation lesson.
- **NEW: `m4t/tools/bench_accum_baseline.c`** — pre-cycle baseline measurement; informational-only per "function over speed" rule.
- **NEW: `m4t/tests/test_m4t_accum_aligning_neon.c`** — bit-exact regression test (15 n boundary + 13 delta cases + flag-NULL paths + 2 saturation-edge + 1000 random = 1030+ configs) + alias test + 6-shape perf bench.
- **A-G3 design insight:** the inner loop stays in int32 throughout (no int64 widening for the add) because |aligned + other| ≤ MAX_VAL/3 + MAX_VAL ≈ 7.7×10⁸ < INT32_MAX. The current scalar's int64 intermediate was overcautious.
- **A-G4 (bit-exact):** all 1030+ configurations match output AND BOTH flag bits (ROUNDED + SATURATED). Flag reconstruction via NEON `cmeq.4s` (saturated: sum != clamped) and `cmeq.4s` (rounded: aligned * s != val) — full fidelity, no scalar fallback.
- **A-G6 (bench, min-of-5, scope-match-compliant):** speedup range **1.6× to 6.0×** depending on (n, delta, flags). Best: n=64 delta=5 NO-flags = 6.0×. Worst: tiny-n with-flags = 1.6×. With-flags average ~2.3× across the typical (n=64, delta=10-19) regime; without flags ~6× at the same shape.
- **A-G7 productionized:** `m4t_mtfp_vec_accum_aligning` dispatcher now calls the NEON helper directly. **No scalar fallback in production dispatch** per the new project rule (memory: feedback_function_over_speed_no_scalar). `_scalar_ref` remains as test oracle; geometric scalar tail (sub-block n) remains.
- **A-G9 (no-scalar audit, cycle scope):** cleaned the `#if !M4T_HAS_NEON ... fall back to scalar ...` branch this cycle's prototype had introduced. Cross-cutting audit (`block_add`, `block_sub`, `ternary_dot` dispatch, etc. — ~5-6 other locations with dead scalar fallback) flagged as follow-on cycle.
- **20/20 ctest PASS** (was 19, +1 for `m4t_accum_aligning_neon`).
- **3 production binaries identical** before/after.

### Methodology lifted from cross-exp accum cycle
- **Cycle scope shrinks as foundational work pays off.** shift3 invented the vmlal-magic-multiply technique; ternary MAC reapplied it to packed-trit matmul; this cycle reapplies it to per-block-aligned accumulation. Each cycle smaller than the last because the foundation is reusable.
- **Speedup-estimate calibration: REFLECT was optimistic.** Estimated 12–20×; measured 1.6–6.0× depending on shape. Reasons: compiler auto-vectorizes scalar baselines effectively at higher delta; per-lane flag bookkeeping (vget_lane × 4 + scalar OR) is the dominant remaining NEON cost. The estimate's qualitative shape was right; the constant was off by ~3×.

### NEW SAVED MEMORY: feedback_function_over_speed_no_scalar
Two project rules saved 2026-05-05 after the user caught a SYNTHESIZE doc that violated both: (1) "Don't stop based on speed up. We can tune the speed later." Cycles gate on correctness, not speedup magnitude. Pre-committed gates with "stop if speedup < N×" are a self-imposed limit that prevents foundational work from landing. (2) "Function is most important and definitely no scalar." Production dispatchers are NEON-only; CMake configure already requires aarch64+NEON and FATAL_ERRORs on non-NEON, so `#if !M4T_HAS_NEON` fallback branches in production code are non-load-bearing. The `_scalar_ref` test oracle is a SEPARATE concept (test-only verification artifact) and is preserved. Geometric scalar tails (n<16) for NEON kernels are also preserved (implementation detail, not a fallback). Pattern-recognition triggers added so the drift doesn't recur.

### Added — ternary MAC routing red-team remediation (10 R-G gates PASS)
Per `journal/ternary_mac_routing_redteam.md` and `ternary_mac_routing_remediation_closeout.md`. The original 10 T-G gates passed but the red-team surfaced 10 evidence-completeness and framing-accuracy gaps. All closed.

- **R-G1: 1000 random bit-exact configurations** added via `test_random_stress`. Bit-exact coverage now: 23 curated + 1000 random + 3 saturation-edge = **1026+ configurations**, up from 23.
- **R-G2: saturation-edge cases.** Constructed configs where dot products exceed MAX_VAL × K → output clamps to ±MAX_VAL with SATURATED flag. Both production NEON and scalar_ref produce same clamped output AND same flag bits. 3/3 PASS (`+MAX_VAL × +1 → +sat`, `+MAX_VAL × -1 → -sat`, `-MAX_VAL × +1 → -sat`).
- **R-G3: 5-shape BATCHED bench.** **Headline correction: speedup over scalar_ref ranges 4.2× to 17.6×** depending on (M, K, N) — the original claimed "16.7×" was at the high end. Wide aspect (M=N=128, K=1024): 17.6×. Slim aspect (M=N=8, K=4096): 4.2×. Per CONTRIBUTING scope-match rule, the speedup is now reported as a range with shapes named.
- **R-G4: alias test for both forbidden cases.** Y==X AND Y==W_packed both abort via SIGABRT (the second case was untested in the original cycle).
- **R-G5/R-G6/R-G7: closeout update note** added to `journal/ternary_mac_routing_closeout.md`. Documents (a) no current consumer touches the kernel — kernel-microbench numbers don't propagate to consumer perf; (b) custom-silicon ceiling is ~4-17× faster than vmlal — we're operating ~17× off the silicon ceiling, "routed through hardware" accurate but "close to silicon" would overstate; (c) Case W via MTFP4 activations + SDOT-direct is the strategically larger lever (~17× more throughput when activations fit int8).
- **R-G8: closeout per-gate disposition table corrected** for T-G3 (distinguishes the permanent `ternary_dot_vmlal` helper from the transient public wrapper).
- **R-G9: bsl-NEON pointer comment** added to `m4t/src/m4t_ternary_matmul.c` with git SHA recovery instruction (`git show 35e5b58~1:m4t/src/m4t_ternary_matmul.c`) and a note about why the bsl approach is structurally important even though vmlal beat it for ternary.
- **R-G10: test file header rewritten** to reference "production NEON path" instead of stale "vmlal-routed" naming.
- **19/19 ctest PASS** — no regressions; new tests integrated into existing `m4t_ternary_matmul_neon` ctest entry.

### Methodology lifted from ternary MAC remediation
- **Sample-based bit-exact gates need stochastic stress + edge-case construction.** Curated samples explain coverage classes; random fills the breadth; constructed edge cases probe boundaries. The three together are stronger than any alone. Pattern: curated for explainability + random for breadth + edge-construction for boundaries.
- **"Shape-dependent speedup" should be reported as a range.** Single-shape numbers are misleading even within a workload class (BATCHED here showed 4.2× to 17.6× across 5 shapes). For any kernel optimization claim, sweep at least 3-5 shapes within the claimed regime.

### Added — ternary MAC routing through vmlal_s32 (10 T-G gates PASS)
Per `journal/ternary_mac_routing_{raw,nodes,reflect,synthesize,closeout}.md`. The user named ternary MAC as "software doing the work of hardware" earlier in the session and asked what existing M4/NEON features could route it. Answer: `vmlal_s32` (signed multiply-accumulate long, int32×int32→int64 widening) is the closest hardware analog at int32 width. Multiplying by trit ∈ {-1, 0, +1} subsumes both conditional-negate and zero-gate, collapsing the prior bsl + mask-widening pattern.

- **STRUCTURAL CHANGE:** `m4t_mtfp_ternary_matmul_bt` divide path now uses `vmlal_s32` on Apple Silicon NEON. The prior bsl + conditional-negate pipeline (~57 NEON ops per 16-trit block, dominated by ~40 ops of mask widening) is replaced by decode → sign-extend → 8× vmlal_s32 (~18 ops per block).
- **NEW: `m4t_mtfp_ternary_matmul_bt_scalar_ref`** — public scalar-only reference function. Exposed BEFORE prototype work (T-G2) so the bit-exact verification gate survives productionization. Direct application of the shift3 remediation methodology.
- **NEW: `m4t/tools/bench_vmlal_throughput.c`** — vmlal_s32 throughput characterization microbench. Required two iterations to defeat compiler constant-folding (factored `acc += K*(a*b)` until inputs were forced distinct per call via heap pool with non-constant addressing).
- **NEW: `m4t/tests/test_m4t_ternary_matmul_neon.c`** — bit-exact regression test (19/23 configurations: K boundary cases, trit distributions, bulk shapes, alias check via SIGABRT) + perf bench (workload-shape-declared per CONTRIBUTING scope-match rule).
- **T-G1 (throughput characterization):** vmlal_s32 measures **0.84 calls/cycle** on the kernel's actual pattern (two-accumulator chains). Single-chain dependency floor: 0.42/cycle. Independent ceiling: 1.43/cycle.
- **T-G4 (bit-exact, 23 configs):** all PASS — K boundary cases (0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65), trit distributions (all-zero, all-+1, all--1, balanced, sparse 10%), bulk shapes (M=64 K=4096 N=64 = 4096 cells).
- **T-G7 (disasm):** inner loop emits 4× `smlal.2d` + 4× `smlal2.2d` per block (compiler split each `vmlal_s32` source call into low/high pair) plus `ldp q18, q19` paired loads. Clean.
- **T-G8 (bench discipline, min-of-5):** BATCHED (M=64, K=4096, N=64): scalar_ref 10996 ns/cell → bsl-NEON 766 → **vmlal 657** (1.17× over bsl, 16.7× over scalar). TIGHT-LOOP (M=4, K=64, N=4): scalar_ref 24.75 → bsl-NEON 12.25 → **vmlal 5.00** (2.45× over bsl, 5.0× over scalar).
- **T-G9 (productionized):** the `_vmlal` prototype wrapper removed; production `m4t_mtfp_ternary_matmul_bt` now dispatches to vmlal path. Test file renamed `_vmlal.c` → `_neon.c`. **19/19 ctest PASS** (was 18, +1 for `m4t_ternary_matmul_neon`). The bsl-NEON code is preserved in git history per "DELETE = never" project rule.
- **T-G10 (no regression):** `bench_m4t_tier2_perf`, `gesh_confidence_probe`, `gesh_expr_routing_probe` produce identical outputs.

### Methodology lifted from ternary MAC cycle
- **Cycle-level pre-emption of the productionization-invalidates-bit-exact pattern** (lifted from shift3 remediation). When a cycle will replace the function under test, expose the reference oracle as the FIRST gate (here T-G2, before T-G3 prototype). Then post-productionization verification (T-G9 + re-run T-G4) compares production-NEON against an independent scalar oracle, not against itself.
- **Constant-folding defense for throughput microbenchs.** `__attribute__((noinline))` + heap-pool inputs with non-constant addressing + distinct inputs per call. The compiler will factor `acc += K*(a*b)` if any of these are missing. Two iterations needed in this cycle's T-G1 before measurements were valid.

### Added — shift3 NEON cycle remediation (100/100, 12 R-G gates PASS)
Per `journal/shift3_neon_redteam.md` (3 critical, 2 high, 4 medium, 4 low findings) and `journal/shift3_neon_remediation_{precommit,closeout}.md`. The original cycle's 8 gates passed but G6 (productionization) silently invalidated G1's bit-exact verification — replacing the function under test against itself produced a NEON-vs-NEON tautology. Remediation closes all 13 findings.

- **STRUCTURAL FIX (C1, C2, C3, foundation):** `m4t/src/m4t_mtfp.h` declares `m4t_mtfp_shift3_scalar_ref` (always-scalar oracle, test-only). `m4t/src/m4t_mtfp.c` refactored: `static shift3_div_scalar` and `static shift3_div_neon` helpers; `m4t_mtfp_shift3` dispatches; new `m4t_mtfp_shift3_scalar_ref` always uses scalar. LTO can no longer DCE the scalar path (test references it).
- **HELPER EXTRACTION (M2):** NEON path lifted out of `m4t_mtfp_shift3` body into its own static helper with extensive comment explaining the vqrdmulhq → vmull pivot.
- **TEST REWRITE (C1, H1, R-G3, R-G6):** `m4t/tests/test_m4t_shift3_neon.c` (renamed from `_proto.c` per R-G8) now compares production `m4t_mtfp_shift3` against `m4t_mtfp_shift3_scalar_ref`. Prototype kernel copy deleted — single NEON kernel in repo.
- **PERF BENCH FIX (C2, R-G4):** `perf_compare` uses `m4t_mtfp_shift3_scalar_ref` for the scalar measurement.
- **EXHAUSTIVE VERIFY (R-G5):** 22.08 × 10⁹ test points (1.16e9 × 19 k) bit-exact production NEON ≡ scalar_ref. Invokable via `./build/m4t/test_m4t_shift3_neon x` (~25s).
- **HONEST RE-MEASURED SPEEDUP (R-G7, H2):** Production NEON vs scalar_ref, min-of-5: **BATCHED (n=4096): 9.2–9.6×** speedup across k ∈ {1, 7, 13, 19} — confirms the original headline claim. **TIGHT-LOOP (n=4): 1.6×** — corrects the prior 6.3× claim (which was an inlining-asymmetry artifact between the prototype's inlinable copy and the substrate-boundary call). Both numbers reported with workload shape per CONTRIBUTING.md scope-match rule.
- **DOC UPDATES (M4, R-G10; L1, R-G11):** `journal/shift3_neon_closeout.md` framing softened ("divide direction is NEON; multiply direction partly auto-vectorized but has further headroom" — was overstated as "no slow direction"). `m4t/docs/M4T_SUBSTRATE.md` tree updated for `m4t_pow3_magic.h`, `tools/gen_pow3_magic.c`, `test_m4t_shift3_neon.c`, `bench_m4t_lto.c`.
- **PIVOT DOCUMENTED IN PRODUCTION CODE (L2, R-G12):** `m4t_mtfp.c::shift3_div_neon` now contains the inline rationale for the vmull-over-vqrdmulhq choice plus journal pointers.
- **18/18 ctest PASS** (renamed test entry, no count change).

### Methodology lifted from shift3 NEON remediation (cycle-level lesson)
- **When a productionization gate replaces the implementation under test, the bit-exact verification gate must:** (a) run AFTER productionization, AND (b) compare against a separately-preserved reference oracle that productionization does NOT replace. Pre-productionization "bit-exact" claims do not transfer to post-productionization unless one of these conditions holds.
- **Concrete pattern:** when productionizing an optimization, expose the original reference implementation as a permanent test-only oracle in the public API (e.g., `m4t_mtfp_shift3_scalar_ref`). LTO can't DCE it because the test references it externally; future modifications can re-verify against it.

### Added — shift3 NEON divide path (full LMM cycle: prototype → 8-gate productionization)
- `journal/shift3_neon_{raw,nodes,reflect,synthesize,closeout}.md` — full Lincoln Manifold Method cycle on the proposed `m4t_div_3pk_neon` (substitute for custom silicon for shift3 divide direction). RAW dump → 31 atomic NODES → REFLECT cold-eye → SYNTHESIZE 8 pre-committed gates → CLOSEOUT all 8 PASS.
- **STRUCTURAL CHANGE:** `m4t_mtfp_shift3` divide-direction path (k < 0) now uses NEON magic-multiply when `M4T_HAS_NEON && abs_k ∈ [1, 19]`. Pipeline: `vmull_s32 → vaddq_s64(bias) → vshlq_s64 → vmovn_s64`. Compiler fuses vmull+vaddq into `smlal.2d`. Scalar reference path retained as fallback + bit-exact oracle.
- **NEW: `m4t/src/m4t_pow3_magic.h`** — committed magic table (`M4T_POW3_DIV_M[20]`, `M4T_POW3_DIV_N[20]`). Single source of truth (G7); both production substrate and prototype test include it.
- **NEW: `m4t/tools/gen_pow3_magic.c`** — generator that derives `(M, N)` per `k ∈ [1, 19]` and exhaustively verifies bit-exact against the substrate's `m4t_pow3_round_div` reference across 1.16 × 10⁹ × 19 = 2.2 × 10¹⁰ test points (~25s runtime).
- **NEW: `m4t/tests/test_m4t_shift3_neon_proto.c`** — bit-exact NEON-vs-scalar property test, alias test (`dst == src`), workload-shape-explicit perf bench. Wired as ctest `m4t_shift3_neon`.
- **G1 (exhaustive bit-exact):** 22.08 × 10⁹ test points across 19 k values, NEON output matches scalar `m4t_mtfp_shift3` bit-by-bit. Run on demand via `./test_m4t_shift3_neon x`.
- **G2 (saturation):** worst |x*M+bias| = 2⁶¹ (vs INT64_MAX = 2⁶³, 2-bit headroom); worst |result| = 2²⁷·⁵ (vs INT32_MAX = 2³¹, 3.47-bit headroom). `vmovn_s64` narrowing safe.
- **G3 (aliasing):** 12/12 cases (k ∈ {1,10,19} × n ∈ {4,5,64,65}) — `dst == src` works.
- **G4 (disasm):** scalar uses `sdiv x11, x10, x8` (hardware divide, NOT auto-vectorized for the divide direction); NEON uses `smlal.2d + sshl.2d`. Comparison is fair. (Side finding: AppleClang DOES auto-vectorize the multiply direction k > 0 — a future optimization opportunity.)
- **G5 (bench discipline):** workload-shape declared per measurement, min-of-5 sampling. BATCHED (n=4096): **9.5–9.6× speedup** across k ∈ {1, 7, 13, 19}. TIGHT-LOOP (n=4 per call): **6.3–6.5× speedup**. Per V4-residual-3 methodology.
- **G6 (productionized):** `m4t/src/m4t_mtfp.c` divide-direction path branches into NEON. 18/18 ctest PASS (was 17, +1 for `m4t_shift3_neon`).
- **G7 (single source):** magic table in `m4t/src/m4t_pow3_magic.h`; production substrate, prototype test, and generator all coordinate around it. No drift possible.
- **G8 (no regression):** `bench_m4t_tier2_perf`, `gesh_confidence_probe`, `gesh_expr_routing_probe` all produce identical outputs before and after.
- **REFRAMING from earlier session estimate:** the original "~40× speedup" estimate was wrong — assumed scalar used hardware `sdiv` only (~12 cycles); the substrate's actual scalar uses `sdiv` plus round-to-nearest logic and runs faster than the naive estimate. Real speedup is ~9.5× BATCHED, ~6.5× TIGHT-LOOP. The number is honest; the original framing was anchored on the wrong baseline.

### Methodology lifted from shift3 NEON cycle
- **For magic-multiply division: 64-bit intermediate (`vmull + bias + arith-shift`) over 32-bit-with-rounding (`vqrdmulh + vrshl`).** The latter has compound rounding error that's hard to make bit-exact; the former is one rounding step end-to-end. Trade ~1.5× perf for bit-exactness simplicity.
- **Always exhaustively verify the NEON kernel against the scalar reference, not just the emulator.** Emulator is one bridge; NEON intrinsics are another. Both must match.
- **`vmovn_s64` narrowing requires a written saturation argument.** The compiler doesn't insert saturation; if the int64 value exceeds int32, you get garbage. Bound the worst case explicitly.
- **Generated constants live in committed headers, regenerable via committed tools.** No copies in test source. Drift is impossible if the pattern is followed (G7 source-of-truth).

### Added — Outstanding-concerns sweep (4 of 4 closed)
Post-V4-residuals, four outstanding concerns surfaced in conversation review. All four closed methodically; each got its own commit.

- **Concern #1** (`b24c90e`): `journal/tier2_residuals_v4_closeout.md` framed "LTO has nothing to add" too generally. Added header forward-pointer + bottom "Update (V4-residual-3)" section explicitly correcting the framing. Original analysis preserved; correction appended with methodology lesson on adversarial-variant testing of "no delta" findings.
- **Concern #2** (`bcc01f0`): `.github/workflows/build.yml` now runs a 2-job matrix with `GESH_LTO=ON` and `GESH_LTO=OFF`. Without this, the OFF path was only exercised by hand. Verified locally and confirmed in CI: both jobs green (`build-test (LTO=ON)` 28s, `build-test (LTO=OFF)` 17s).
- **Concern #3** (`19c5e73`): `m4t/tests/check_assert_coverage.sh` + ctest entry `m4t_assert_coverage`. Compares the set of substrate `.c` files with `assert(` call sites against `cases[].source_file` in `test_m4t_assert_live.c`. Symmetric — catches both missing cases (silent coverage loss) and stale cases (no asserts left). Caught my own regex bug on first run (`[a-z_]+` excluded digits, missed `m4t_mtfp4.c`); fixed and negative-tested. 16/17 ctest now (was 16; +1 for coverage gate).
- **Concern #4** (this commit): `m4t/README.md` "Reading perf measurements" section + pointer in `m4t/tests/bench_m4t_tier2_perf.c` header. Names the workload-shape dependency explicitly: substrate perf claims rest on carry-dependent, single-pass accumulation; pipelined / batched-independent shapes have a different bottleneck profile and may show very different numbers for the same kernel. Points to `bench_m4t_lto` and the V4-residual-3 closeout as the controlled demonstration.

### Methodology lifted from outstanding-concerns sweep
- **"No delta" perf findings should always carry a workload-shape caveat in the doc that ships them.** Without it, future readers generalize from one shape's measurements.
- **Hand-enumerated coverage lists need automated drift detection.** When a test exhaustively lists "every X with property Y," add a check that compares the list against the actual set. The list will drift; the check catches it.
- **CI matrix on optional build flags catches "only-the-default-is-tested" silent regressions.** Cheap to add, expensive to skip.

### Added — Outstanding-concerns residual sweep
The two small residuals from the prior sweep (concerns #3 and #4) closed:
- **Concern #3 residual** — `m4t/tests/check_assert_coverage.sh` regex tightened from `(^|[^A-Za-z_])assert\(` to `^[[:space:]]*assert\(`. Eliminates false positives from `assert(` mentions in comments or string literals (verified by injecting both: a comment with `assert(...)` text, and a string literal `"...assert(...)..."`, in known-zero-assert files; both correctly ignored). Audit confirmed every current substrate `assert(` call is at line-start modulo whitespace, so no real call sites are missed. Inverse failure mode (mid-line `assert()` would now be missed) handled via a safety-net `warn_mid_line_asserts` function that emits a stderr warning without breaking the gate; verified by injecting `if (n) assert(x);` into a substrate file and confirming the warning fires.
- **Concern #4 residual** — audit of all `*.md` docs outside `journal/` and `01MAY26_archived/` for substrate-perf-claim language. THESIS, FINDINGS, M4T_SUBSTRATE all already appropriately scoped (THESIS's falsification criterion names "on at least one realistic workload"; FINDINGS line 44 says substrate-claim measurements require a consumer; M4T_SUBSTRATE is design spec). Only `m4t/README.md` and `bench_m4t_tier2_perf.c` needed the caveat; both were addressed in concern #4's main commit. Added a project-level rule to `CONTRIBUTING.md`'s scope-match checklist: "Single workload-shape → cannot claim general kernel performance. Name the workload shape when reporting kernel timings."

### Added — V4 residual #3 closure: LTO microbench reveals 3× speedup is achievable
- `journal/v4_residual_3_lto_microbench_closeout.md` — full cycle: design → discover -fno-lto silently overridden → fix CMake gating → measure → discover variant A no delta (matching V4-G5) → red-team → add variant B → discover 3× LTO speedup → red-team again → document.
- **STRUCTURAL FIX:** `CMakeLists.txt` (top-level) — gated `add_compile_options(-flto)` and `add_link_options(-flto)` behind `option(GESH_LTO "Enable link-time optimization" ON)`. Default behavior unchanged. Build a no-LTO comparison tree via `-DGESH_LTO=OFF`. Surfaced because my first attempt at no-LTO via `-DCMAKE_C_FLAGS=-fno-lto` was silently overridden — CMake prepends user flags but the project's own `add_compile_options` appends, so the compile line ended up with both `-fno-lto` and `-flto` and clang took the later (LTO won).
- **NEW MICROBENCH:** `m4t/tests/bench_m4t_lto.c` with two workload variants targeting `m4t_mtfp_block_add` (small cross-TU function, ~6 NEON ops). Variant A: carry-dependent (single dst accumulated). Variant B: pipelined (round-robin across 64 independent dsts). Min-of-3 sampling per variant. Build target only (perf, not regression).
- **HEADLINE FINDING:** Variant A LTO ≈ no-LTO at 1.36 ns/call (matches V4-G5: data-dep-bound workload, LTO has nothing to fix). Variant B LTO 0.23 ns/call vs no-LTO 0.68 ns/call → **3× LTO speedup** on pipelined workload (call-overhead bound). LTO IS doing useful cross-TU inlining; the original V4-G5 finding was right narrowly but not a general statement about LTO.
- **DISASM PROOF:** `otool -tv` confirms LTO build inlines block_add (no `bl _m4t_mtfp_block_add` in main; symbol absent from binary). no-LTO build retains `bl _m4t_mtfp_block_add` and the function symbol. Cross-TU inlining is real, just hidden by data dependency in (A)-shaped workloads.
- **RED-TEAM:** Five findings examined. RT-1 variant B may not represent any real consumer — DOCUMENTED in bench source. RT-2 variant B's 0.8 cycles/iter — VERIFIED via disasm to be Apple Silicon's 8-wide superscalar issue, not unexpected unrolling. RT-10 the 3× could come from cross-TU OR intra-TU LTO contributions — DOCUMENTED as honest concern (cross-TU inlining is the proven first suspect; mixed configs would isolate further).
- **CONSEQUENCE for V4-G5:** the V4 finding "LTO produces no observable bench delta" is correct narrowly (for the substrate's actual carry-dep consumers) but the V4 closeout's framing of "LTO has nothing to add" is wrong as a general claim — LTO does add 3× on pipelined workloads, just none currently exist in the substrate's hot path.
- **16/16 ctest PASS** with no collateral damage.

### Methodology lifted from V4 residual #3
- **Always prove LTO actually applied.** Verbose make output is the source of truth. CMake user-flag prepending vs project-flag appending bit me; fix by gating with `option()` so the `add_compile_options` is conditional, not silently overridden.
- **Compiler flags can lie about their effect; disasm cannot.** Always cross-check optimization claims with `otool -tv`.
- **Workload shape determines bottleneck, not the compiler.** A workload's bottleneck (data dep, memory BW, call overhead, etc.) determines what optimizations CAN help. Measuring in only one shape under-determines the conclusion.
- **"No delta" findings should be tested with at least one adversarial variant.** If a workload designed adversarially in favor of optimization X ALSO shows no delta, the finding generalizes. If it shows a delta, the original "no delta" was scoped to that workload shape.

### Added — V4 residual #1 closure: parameterized assert-live meta-test
- `journal/v4_residual_1_assert_live_closeout.md` — full cycle: remediate → red-team → fix → non-tautology validation → document.
- **STRUCTURAL FIX:** `m4t/tests/test_m4t_assert_live.c` rewritten from single-case to parameterized. Five cases, one per substrate source file with asserts (`m4t_route.c`, `m4t_mtfp.c`, `m4t_mtfp4.c`, `m4t_ternary_matmul.c`, `m4t_trit_pack.c`). Each case violates a DIFFERENT precondition pattern — T-overflow (route), negative-size (mtfp/mtfp4), aliasing (ternary_matmul), out-of-range trit (trit_pack) — for variety beyond just "negative everywhere."
- **HARNESS:** `assert_case_t` struct binds source-file → label → violate function pointer. `run_case` forks-and-verifies-SIGABRT for each case. Distinguishes "assert silenced" (child exits cleanly with `EXIT_ASSERT_SILENCED = 42` sentinel) from "child crashed for other reason" (different signal/exit), so a regression points to the actual cause.
- **RED-TEAM (1 fix, 6 verified/documented):** RT-1 stdio buffer duplication (child inherits parent's unflushed stdout, re-emits on exit) FIXED via `fflush()` before `fork()`. RT-2 verified each violation reaches the intended assert (output names file:line). RT-3 verified earlier preconditions don't short-circuit. RT-4 verified -UNDEBUG uniformly applied (5/5 PASS test variant, 0/5 PASS production). RT-5/6/9 documented as honest concerns.
- **NON-TAUTOLOGY CHECK:** built the parameterized test against production `libm4t.a` (NDEBUG). All 5 cases reported `assert SILENCED (child returned cleanly)`. Inversion is the proof: 5/5 against m4t_test, 0/5 against m4t. Each case is not vacuous; each depends on -UNDEBUG actually being applied.
- **COVERAGE NOTE:** `m4t_trit_ops.c` and `m4t_trit_reducers.c` have 0 asserts by design (pure compute, no preconditions). All 5 source files with asserts are now runtime-verified.
- **16/16 ctest PASS** with no collateral damage.

### Methodology lifted from V4 residual #1
- **Parameterized meta-tests cover surface area; per-case assertions verify each cell of that surface.** A single case proves a mechanism; an enumerated set of cases proves uniform application. Use the latter when the claim is "X holds across N entities."
- **Non-tautology checks belong in red-team.** Building the same test source against the inverse build configuration should produce the OPPOSITE result. If both paths PASS, the test is vacuous.
- **Stdio buffering and fork interact badly.** `fflush()` before `fork()` whenever either side writes to stdio.

### Added — V4 residual #2 closure: tight bound now data-derived
- `journal/v4_residual_2_tight_bound_closeout.md` — full cycle: remediate → red-team → fix → validate → document.
- **STRUCTURAL FIX:** `gesh/tests/test_image_canon.c` replaced hardcoded `tight_bound = 10*dim` (pinned to the specific synthetic pixel pattern) with `derive_tight_bound()` — a per-image bound derived from the actual data's pre-normalize standard deviation. New `test_isqrt64` helper (Newton iteration) for the sd computation.
- **MATH:** `bound = 2 * dim * (1 + scale_over_sd_ub)` where `scale_over_sd_ub = floor(SCALE/sd) + 1`. Walks through the two integer-truncation layers (var/dim then isqrt; SCALE/sd) explicitly. For dim=16, sd ≈ SCALE/5: bound = 224 (vs. previous hardcoded 160). Observed drift ≤ 76; headroom ≈ 2.95×.
- **RED-TEAM (4 findings, all addressed):** R-A off-by-one in formula (code had extra `+1` term unjustified by math; tightened from 256 → 224). R-B comment didn't explain the `+1` as an upper-bound trick. R-C integer `var/dim` ALSO truncates (extra source of conservativeness, now documented). R-D loose bound is now redundant for realistic data (kept as backstop against future `derive_tight_bound` bugs, with explicit role doc).
- **POSITIVE CONTROL:** scratch test injects +15 per pixel → post-normalize sum = 245 > bound 224, tight check correctly fires. Bound is meaningful, not vacuous.
- **RECALIBRATION VALIDATION:** scratch test computes bounds across four data shapes. Low-sd (slowly varying): bound = 472,448. High-sd (alternating): bound = 128. Uniform (sd=0): bound = 32 (edge case via `2*dim` floor). Original synthetic: bound = 224. Bound varies by orders of magnitude — formula is data-shape-aware.
- **16/16 ctest PASS** with no collateral damage.

### Methodology lifted from V4 residual #2
- **Test bounds tied to specific synthetic data are landmines.** Replace hardcoded constants with bounds derived from the actual data when the math is tractable. Future test-data changes auto-recalibrate.
- **Walk through every integer-truncation source in derived bounds.** Each integer divide silently shifts the bound; document the direction (lower/upper) and why the resulting computed bound is conservative.
- **Validate derived bounds three ways: positive injection (does the check fire?), recalibration across data shapes (does the bound move when data changes?), full regression (no other test broken?).**

### Added — V4 remediation: closed all 4 -UNDEBUG residual threats from V3
- `journal/tier2_residuals_v4_{precommit,closeout}.md` — methodical closure of the four threats inherent in V3's `-UNDEBUG` residual.
- **STRUCTURAL FIX (T1):** added parallel `_test` library variants — `m4t_test`, `gesh_test`, `gesh_bench_test`, `gesh_image_canon_test` — compiled from the same sources as the production libs but with `-UNDEBUG` applied. Test executables now link against the `_test` variants. Substrate-internal asserts (e.g., `m4t_route_topk_abs`'s `T <= M4T_ROUTE_MAX_T` precondition) actually fire when triggered from tests. Pre-V4, these asserts were silenced — `libm4t.a` was compiled with `-DNDEBUG` and the test executable's own `-UNDEBUG` only affected its own .o files, not the lib's.
- **RUNTIME PROOF (V4-G2):** `m4t/tests/test_m4t_assert_live.c` — deliberate-abort meta-test. Forks a child, calls `m4t_route_topk_abs(decisions, scores, T=200, k=4)` (T > `M4T_ROUTE_MAX_T = 64`), `waitpid`s, asserts `WIFSIGNALED && WTERMSIG == SIGABRT`. Distinguishes "assert silenced" (child exits cleanly with code 42, parent reports FAIL) from "child crashed for unrelated reason" (different signal/exit). Sanity-verified by linking the meta-test against production `libm4t.a`: it correctly reports `substrate asserts are SILENCED` (exit 1).
- **VERIFICATION UPGRADE (T2):** replaced grep-based assert auditing with `nm` symbol verification on the actual built libraries. Production libs (`libm4t.a`, `libgesh.a`, `libgesh_bench.a`, `libgesh_image_canon.a`): 0 references to `___assert_rtn` each. Test variants (`libm4t_test.a` / `libgesh_test.a` / `libgesh_bench_test.a` / `libgesh_image_canon_test.a`): 5 / 8 / 4 / 1. Concrete structural proof, not regex.
- **TIGHT BOUND (T3):** `gesh/tests/test_image_canon.c` mean-drift check now has TWO bounds. Loose `dim*SCALE/10` (≈ 94K for dim=16) catches order-of-magnitude bugs; tight `10*dim` (= 160 for dim=16) catches 2× regressions. Tight bound derived from worst case `≤ 5*dim` (residual after centering ≤ dim, amplified by SCALE/sd ≤ 4 for typical pixel data) plus 2× safety. Observed drift on synthetic test data ≤ 80; tight bound holds with 2× headroom.
- **LTO MEASUREMENT (T4):** built parallel no-LTO scratch tree (`-DCMAKE_C_FLAGS="-fno-lto"`) and compared to the production LTO build. Bench timings within ±5% noise across 10 measurement points. Binary size byte-identical (50936 vs 50936). `bl _m4t_route*` call counts identical (6 vs 6). LTO is enabled and applied (verified in verbose make output: `-flto` in compile + link commands), but produces no observable optimization on this bench. Substrate's hot paths are already aggressively per-TU-optimized at `-O3 -mcpu=native`; LTO has nothing to add at the bench's measurement granularity. Honest finding, not a fix.
- **`CMakeLists.txt`** (top-level): expanded `gesh_test_undebug()` comment to document the V4 library-variant expansion.
- **`m4t/CMakeLists.txt`**: added `m4t_test` STATIC library; relinked all 9 m4t test executables. `bench_m4t_tier2_perf` keeps linking against production `m4t` (asserts add overhead irrelevant to perf measurement).
- **`gesh/CMakeLists.txt`**: added `gesh_test`, `gesh_bench_test`, `gesh_image_canon_test` STATIC libraries; relinked all 6 gesh test executables. The 22 gesh bench/probe binaries stay linked against production libraries.
- **`m4t/README.md`**: ten ctest binaries (was eight); added rows for `test_m4t_elemental_floor` and `test_m4t_assert_live`; added test-build-discipline section.
- **`m4t/docs/M4T_SUBSTRATE.md`**: refreshed `tests/` listing.
- **16/16 ctest binaries PASS** (was 15; +1 for `m4t_assert_live`) under full LTO with substrate asserts now structurally live in test builds.

### Methodology lifted from V4
- **Substrate asserts must be live in tests.** When a substrate library has internal `assert()` precondition checks, ship a `_test` library variant compiled with `-UNDEBUG` and link tests against it. The test executable's own `-UNDEBUG` is not sufficient — that flag only affects code compiled directly into the executable.
- **"Asserts are live" claims must include a runtime meta-test, not just build flags.** Build assertions ("we passed `-UNDEBUG`") and source assertions ("the source has an `assert(EXPR)`") are not the same as runtime assertions ("EXPR actually evaluates and aborts on failure"). The deliberate-abort meta-test is the runtime check.
- **`nm` symbol verification beats source-level grep.** When auditing whether a build flag actually changed the binary, look at the symbols, not the source. Greps can miss; symbols can't lie.
- **Pair loose + tight principled bounds where possible.** Loose catches catastrophic bugs; tight catches 2-3× regressions. Document derivations.
- **Build-system audits should compare LTO-on vs LTO-off binaries.** Identical sizes/call-counts is a signal LTO is having no effect — flag the finding even when it's not a bug.

### Added — vision claim #2 P0 (expression routing) + 100/100 remediation
- `docs/PLAN_EXPRESSION_ROUTING.md` — plan for closing vision claim #2 (math expressions as signatures via routing), with P0 (4 pieces) and P1 (2 pieces). Plan was rewritten post-LMM-cycle to reflect the equivalence-class-lookup framing rather than the original drop-in-bank framing.
- `journal/expression_routing_{raw,nodes,reflect,synthesize,closeout,redteam,remediation_precommit,remediation_closeout}.md` — full LMM cycle on the plan, plus red-team and 100/100 remediation cycle. Cycle surfaced the conceptual fix: data signatures are learned distributions; expression signatures are defined evaluations; the right primitive is equivalence-class lookup, not bank substitution.
- **P0-2** `gesh/src/expr.{h,c}` — small expression tree types (var, const, neg, add, sub, mul, max, min) over substrate primitives. exp/log explicitly absent (P1-1). int64 evaluator.
- **P0-1** `gesh/src/expr_signature.{h,c}` — behavior-based signature derivation. Evaluate at fixed test inputs, ternarize via `m4t_route_threshold_extract` at tau=0. Substrate-discipline preserved (no open-coded sign step).
- **P0-3** `gesh/src/expr_bank.{h,c}` — equivalence-class bank constructor. Detects byte-equal signatures, picks first-in-order as representative, exposes candidate→class map. Reuses `gesh_bank_t`; only label semantics change.
- **P0-4** `gesh/bench/expr_routing_probe.c` — probe binary with two pre-committed gates. Result: **EASY 60/60 + HARD 18/18 = PASS.**
- **Red-team remediation** `gesh/src/expr_random.{h,c}` and `gesh/bench/expr_routing_remediation.c` — addresses 13 of 15 red-team findings (2 deferred with explicit rationale). Five gated sections: subagent-blind probes (C1, C2), scale-collapse (H3), multi-input-set sweep (H1), random-bank multi-seed (H2, M2), inter-class diagnostic (M1). All five sections PASS pre-committed gates. **OVERALL REMEDIATION: PASS (100/100).**
- **Subagent-blind result is the load-bearing finding.** A subagent that never saw the signature math, test inputs, or any code designed 30 probes from mathematical intuition. Routing matched 29/30 (96.7%). Non-tautological evidence the rule aligns with independent math reasoning. The single MISS was the subagent's own self-flagged ambiguous probe.

### Methodology lifted to project-wide rules
- Pre-commit ALL gates in SYNTHESIZE before any code or run (per `journal/expression_routing_remediation_precommit.md`, addressing red-team finding H4). The pattern "easy gate passed → add hard gate after seeing results" should be flagged as a gate-revision event in the closeout, not normal tightening.
- Subagent-blind probes are a high-value discipline tool for any cycle where the probe-author has full access to the system being tested. ~30 minutes of subagent time is enough to produce non-tautological evidence.

### Deferred to P1
- M3 (substrate's third state barely used by sign-only signatures): P1-1 design must let the third state carry information beyond exact-zero.
- M5 (cost-blindness latent in compose-equivalence decision D5): P1 work using the bank for anything beyond identity-lookup must address.

### Honest concerns surfaced by the remediation
- Subagent's prompt was cooperative, not adversarial. The 96.7% is evidence that cooperative intuition aligns with the rule; adversarial intuition is untested.
- 5-seed 100%-with-0pp-stddev on internal-consistency is partly artifact of test design (relative, not absolute discrimination).
- Arity-1 inter-class minimum distance is 3 (below 4-trit headroom). Two of 10 arity-1 classes are close; future bank growth at this dim likely to start colliding.

### Added — concerns remediation R2 plan + R1 cycle (PASS) + R1 red-team + R1 remediation (FAIL)
- `docs/PLAN_EXPRESSION_ROUTING_R2.md` — three-track plan (R1, R2, R3) addressing 5 of 9 concerns surfaced after the original P0 PASS.
- `journal/r1_signature_rule_{raw,nodes,reflect,synthesize,closeout,redteam,remediation_precommit,remediation_closeout}.md` — full LMM cycle on R1, plus red-team and 100/100 remediation.
- **R1** `gesh/src/expr_signature.{h,c}::expr_to_signature_dual` and `gesh/src/expr_bank.{h,c}::expr_bank_dual_t` — per-expression-tau dual-threshold rule using `m4t_route_threshold_extract_dual` + `m4t_route_confidence_weighted_dist`. Initial verdict **R1 PASS** (R1-A 96.7%, R1-B 92%, R1-C by construction).
- **R1 red-team** surfaced 13 findings (2 critical, 4 high, 4 medium, 3 low). The critical findings: R1-B's "information gain" gate is satisfied by ANY use of the conf channel (not by useful information); R1-A's backward-compat probe set doesn't exercise the rule's new behaviors.
- **R1 100/100 remediation** ran 8 gated/diagnostic sections under sharper pre-committed gates. **Verdict: FAIL.** Two FAILs: §1 partition-change (mean 4.2% pair-change across 5 seeds, gate ≥30%) and §6 inter-class distance (arity-1 min=1 under dual vs 3 under sign-only). One WEAK: §4 granularity (76.7%, gate ≥80%). Two PASS, three diagnostics.
- **What the FAIL teaches:** the dual rule's "information" is largely cosmetic (signatures look 92% different but the equivalence partition only changes 4.2%); discrimination got WORSE for arity-1 (min distance 3 → 1); arity-1 zero band dominates at 66.5% of cells; the rule is 5.68x slower than sign-only popcount path. Original R1 PASS was structurally weak because its gates didn't bite.
- **R1 PASS verdict from `journal/r1_signature_rule_closeout.md` is functionally OVERTURNED for arity-1.** Arity-2 may be salvageable. R1 next step is open: revert to sign-only for arity-1 (Option A, ~2 days), redesign rule (Option B, ~2 weeks), or proceed to R3/R2 anyway accepting the broken foundation (Option C, risky).
- **Discipline lesson:** the pattern PASS → red-team → remediation → honest FAIL is the discipline working. The original PASS was wrong; the remediation surfaced it; the verdict is honest. Per project norms (P0-4 negative result precedent), the dual-rule code remains in the codebase with FAIL documented at the verdict level.

### Added — R1 path-forward LMM cycle + R1 fork experiment (F3 wins)
- `journal/r1_path_forward_{raw,nodes,reflect,synthesize}.md` — LMM cycle on the post-FAIL question. Reduced 8 surface options (A–H) to 3 structural framings of the failure (F1 wrong rule, F2 wrong axis, F3 wrong layer). Synthesized: a focused 3-day fork experiment to distinguish the framings empirically.
- `docs/PLAN_R1_FORK.md` — pre-committed plan for the fork experiment with three framing thresholds.
- `gesh/bench/expr_routing_r1_fork.c` — fork experiment binary. Runs both signature rules at sig_dim ∈ {16, 32, 64} on curated arity-1 and arity-2 banks plus 100 random expressions × 3 seeds.
- `journal/r1_fork_closeout.md` — **F3 wins (wrong layer).** Arity-1 sign-only inter-class min stays stuck at 3 across all dims (no improvement from sig_dim 16 → 64). Random-bank class count stuck at 27 for sign-only. Dual at sig_dim=64 reaches min=4 (only +1 over sign-only). The arity-1 expression set is intrinsically signature-saturated — the bank's discriminability ceiling is set by the EXPRESSION SET, not the SIGNATURE RULE.
- **Concerns 2/3/7/8 from the R1 red-team era are recategorized:** substrate kernels work at the substrate layer; the expression-routing consumer doesn't have a problem they solve. Vision claim #3 must be demonstrated by a different consumer.
- **R-track is closed.** R1 dual rule should be reverted for arity-1 (kept in codebase per ship-with-FAIL discipline). Original R3 (sig_dim sweep) and R2 (scale experiment) cycles cancelled as planned; replanning needed.

### Added — P1-1 design cycle (full LMM, loop-back fired in SYNTHESIZE)
- `journal/p1_1_primitives_floor_{raw,nodes,reflect,synthesize,closeout}.md` — full LMM cycle on closing the primitives floor with exp/log; closeout marks the cycle as superseded by the elemental-floor reframing.
- **SYNTHESIZE surfaced that the substrate has no division operation.** Without division, true Taylor exp/log aren't expressible from the existing seven (`add, sub, mul, neg, max, min, eq`). Range reduction (`exp(x/2)^2`) requires division too. The cheap "Path B prototype" plan that REFLECT had endorsed was built on an assumption that didn't hold.
- **Loop-back trigger fired per LMM.md.** Owner conversation then reframed twice: (a) the "no consumer demand" framing was wrong for foundational research and was retired with extreme prejudice; (b) mul itself is composite (iterated conditional add via shift), so the right question wasn't "add exp/log" but "what's actually elemental and what's missing from the substrate's elemental floor?"

### Added — Elemental floor audit + close (PASS, ~5 ops floor)
- `journal/elemental_floor_{raw,nodes,reflect,synthesize,closeout}.md` — LMM cycle on the audit. Established the cell-level elemental floor as **5 ops + 3 constants**: `add, neg, shift3, sign, select` plus `{-1, 0, +1}`.
- **Substrate had add, neg, sign already; missing shift3 and select.** All other ops (mul, sub, div, max, min, eq, exp, log, sin, cos, sqrt, ...) are composite — derivable from elementals + iteration. Performance kernels for hot composites stay; documentation will name them as composite (deferred follow-on).
- `m4t/src/m4t_mtfp.{h,c}::m4t_mtfp_shift3` — base-3 positional shift. `dst = src * 3^k`, saturation on positive overflow, base-3 round-to-nearest-even on negative k (reuses cross-exponent accumulator's `m4t_pow3_round_div` and the odd-divisor lemma).
- `m4t/src/m4t_route.{h,c}::m4t_route_select` — trit-controlled cell-level mux. Pure routing, no arithmetic, may alias inputs.
- `m4t/tests/test_m4t_elemental_floor.c` — three property tests (G1 shift3 correctness, G2 select correctness, G3 composite re-derivation). All PASS.
- 15/15 ctest binaries green (was 14; +1 for `m4t_elemental_floor`). All prior probes still PASS.
- **Vision claim #1 substantively addressed for the first time** with a defensible audited floor.

### Methodology lifted from this turn
- **Owner pressure on foundational analysis was the unlock** for catching mul's composite status. Without "Mul, if made of two conditions, is composite," the prior cycle would have shipped a plan treating mul as elemental. Foundational claims benefit from explicit owner pressure to test irreducibility.
- **Iteration is not an operation, it's program structure.** Anything derivable by iterating elemental ops is composite, regardless of whether the substrate provides a fast kernel for it.
- **The retirement of "no consumer demand" as a research blocker** unblocked the analysis and the implementation. The original rule (measurement-integrity: route consumer code through kernels) survives; the over-generalized form (don't build primitives without a consumer using them) does not apply to foundational research and was misapplied for cycles before this turn.

### Added — Tier 2 NEON underuse remediation (2 of 3 PASS, 1 reverted)
- `journal/tier2_perf_{precommit,closeout}.md` — methodical pass on three places where existing NEON hardware was underused. Pre-committed gates ahead of code; partial PASS verdict.
- **T2-A NEON `m4t_route_select` — PASS.** Replaced scalar per-cell loop with NEON path using `vceqq_s32` mask construction + bit-select cascade. Measured **2.55x speedup** (9.258ms scalar vs 3.626ms NEON over 100K iterations on 64-cell vectors). G1 correctness PASS, G2 speedup PASS.
- **T2-B branchless `confidence_weighted_dist` — REVERTED.** Branchless per-byte version with bitwise indicator + popcount initially measured 2.9x slower; reverted to branchy original. Post-revert red-team revealed the perf harness was unfair (inlined reference vs lib-call) — the apparent slowdown was largely function-call overhead artifact. True per-call speed at substrate scale unknown; revert is conservative. Inline note in `m4t_route.c` documents the revert reasoning.
- **T2-C `accum_aligning` same-exp branch — PASS.** When `flags == NULL`, the same-exponent path now calls `m4t_mtfp_vec_add_inplace` (NEON-vectorized via `m4t_mtfp_block_add`) instead of per-cell scalar add+clamp. Saturation-tracking path preserved when flags non-null.
- `m4t/tests/test_m4t_tier2_perf.c` — perf harness for G2/G4. Build target only (not ctest) since perf measurements aren't correctness regressions.
- All 15 ctest binaries still PASS. No regression at any step.

### Methodology lifted from Tier 2
- **Perf measurement is itself a discipline that needs gates.** The pre-commit named WHAT to measure (speedup ratios) but didn't specify HOW (inlined vs lib-boundary). Cross-boundary comparisons are unreliable. Future perf gates should specify identical call mechanics for both versions.
- **Pre-committed gates work even when they expose flaws in the gate itself.** T2-B's gate fired correctly; the failure mode it surfaced was the gate-design quality, not a real algorithmic regression. Net: code is in a defensible state, methodology is improved for next time.

### Added — Tier 2 100/100 red-team remediation (PASS, with major finding overturning T2-B)
- `journal/tier2_perf_redteam.md` — 13 findings (1 critical, 3 high, 6 medium, 3 low) on the prior Tier 2 closeout. Critical finding: T2-A's "2.55× speedup" measurement had the same unfair-comparison flaw the closeout flagged for T2-B but treated as fair.
- `journal/tier2_remediation_{precommit,closeout}.md` — methodical remediation with pre-committed gates that explicitly red-team the gate design itself (R-G7).
- **Fair-comparison perf harness** (`m4t/tests/bench_m4t_tier2_perf.c`, renamed from `test_*` per L2): both versions of each candidate now go through identical lib-call boundaries. Three data distributions (random, structured, sparse-zero), pool of 8 cycled arrays, `clock_gettime(CLOCK_MONOTONIC)`, median of 5 trials.
- **Lib reference variants:** `m4t_route_select_scalar_ref` and `m4t_route_confidence_weighted_dist_branchless` added to libm4t for fair benchmarking. Documented "for benchmarking, NOT production use."
- **R-G3 path-exercise test:** `test_accum_same_exp_flags_null` in `test_m4t_elemental_floor.c` — verifies T2-C's flags=NULL fast path against int64 reference. PASS.
- **MAJOR FINDING (R-G2 diagnostic):** the branchless `confidence_weighted_dist` is **1.81–2.56× FASTER** than branchy across all three distributions. The original Tier 2 "branchless is 2.9× slower" was the artifact, not the truth. The substrate currently runs the slower (branchy) production version because of the original bad measurement. Fair re-measurement overturns the T2-B revert decision; production flip pending owner authorization (per H4: gate was diagnostic, not PASS/FAIL).
- **R-G1 update:** T2-A's true select speedup is **1.82–5.57×** depending on data distribution (random shows largest NEON win), not 2.55× as the original closeout stated. Original number was a lower bound under unfair comparison.
- All 16 ctest binaries PASS (was 15; +1 for new T2-C path test). No regression at any step.

### Honest residual gap from this remediation
- R-G5 cache-defeat verification has a design flaw (consecutive-runs check is uninformative). The pool-of-8 mitigation IS real; we can't verify how much it helped. Documented as partial fix.

### Methodology lifted from Tier 2 remediation
- **Reference variants for fair benchmarking go in the lib, not in the test file.** Benchmark harness compares two lib functions through equivalent call paths.
- **Cache-defeat verification needs adversarial design, not naive consecutive-runs comparison.** Fresh-process-per-trial or explicit cache-invalidation are the real options.
- **Diagnostic gates that surface major findings require explicit owner-action protocols.** R-G2 was diagnostic-only but produced a finding that overturns prior production code. Cycle delivers the data; substrate change requires owner authorization. Naming this protocol avoids future ambiguity.

### Added — Tier 2 residuals closure (PASS, with stronger T2-B evidence)
- `journal/tier2_residuals_{precommit,closeout}.md` — closes the three residuals (cache-defeat verification, adversarial distributions, LTO for accurate-AND-fair timings) from the Tier 2 remediation closeout.
- **RES-1 cache-trashing:** explicit 32 MB buffer walk between trials. Result for select: warm/cold ratio 1.00× — workload fits in L1 even after eviction; steady-state numbers are honest for THIS workload size. Mechanism in place for larger-workload tests.
- **RES-2 adversarial distributions:** subagent designed 6 distributions blind; 4 implemented (LFSR-cycled trits, sparse-zero bursts, sparse-opposite needle, triple-period resonance). **Two of four predictions of "vectorized loses to scalar/branchy" FAILED — NEON select still wins on sparse-zero (1.51× vs predicted 1.2-2× loss); branchless conf-dist still wins on sparse-opposite needle (5.23× vs predicted branchy-wins 3-5×).** The vectorized/branchless implementations are robustly faster across both cooperative AND adversarial inputs.
- **RES-3 per-target LTO:** global `-flto` broke `gesh_image_canon` test (segfault, root-cause deferred). Working approach: per-target LTO on `bench_m4t_tier2_perf` only. All 15 ctest binaries unchanged.
- **RES-4 no regression:** 15/15 ctest binaries PASS through every step.
- Strengthened T2-B production-flip recommendation: **no tested distribution favors branchy.** Standard 1.89-2.75× faster, adversarial-designed-to-favor-branchy still 5.23× faster. Substrate currently runs the slower version because of bad measurement + wrong theoretical prior.

### Honest residuals from this cycle
- LTO global-vs-per-target: root cause of image_canon segfault under global LTO not investigated.
- 2 of 6 adversarial distributions not implemented (cache-aliasing patterns; require careful aligned allocation).
- Cache-defeat mechanism in place but its effect untested for L2/L3-stressing workload sizes.

### Methodology lifted from residuals
- **Subagent-blind adversarial distributions are a strong test of perf claims.** Two of four predicted inversions failed when measured — meaning the conventional intuitions ("branchy is better for sparse mismatches" / "branchy is better for predictable branches") don't hold under measurement. Pattern worth applying to any perf claim where cooperative-author bias is a risk.

### Added — Tier 2 residuals v2 (atomics: LTO root-cause, adversarial completion, cache-defeat saturation)
- `journal/tier2_residuals_v2_{precommit,closeout}.md` — addresses three honest residuals from the prior cycle.
- **V2-G1 (LTO root-cause):** full `-flto` triggers SEGV in `image_canon_normalize` via SIMD-loop reading from a stale stack-string-buffer pointer (lldb confirmed: register x16 contained the path string content). Workarounds tested (`-fno-strict-aliasing`, `-fno-vectorize`, `-fno-inline`) didn't fix; only `-flto=thin` did. **Production solution: ThinLTO globally.** All 15 ctest binaries PASS under ThinLTO. The exact pointer-aliasing pattern in image_canon under aggressive full-LTO inlining is a documented residual — ThinLTO sidesteps it; future investigation could narrow it.
- **V2-G2 (adversarial completion):** PASS-PARTIAL. Branch-pattern portions of subagent dists 2 and 5 implemented (run-length trap A3, confidence-stripe thrasher B3); cache-aliasing engineering (page-aligned conflicts on multiple buffers, verified L1 set-index collisions) deferred with rationale.
- **V2-G3 (cache-defeat saturation):** DATA-PRODUCING; gate's hypothesis was wrong. Tested warm/cold across n_cells ∈ {64, 4096, 65536, 524288} — all show ratio ~1.00× regardless of working set size. The cache-defeat mechanism (32MB walk) works but isn't observable for select because the workload is bandwidth-bound (sequential access, prefetcher hides cache effects fully). RES-1's earlier "steady-state honest" finding generalizes further than originally claimed.

### MAJOR FINDING from V2: T2-B production-flip recommendation REVERSED
- Under ThinLTO with both functions inlined into the bench, branchy and branchless `m4t_route_confidence_weighted_dist` are **equivalent in speed across all 7 distributions tested** (3 standard + 4 adversarial). The previous "branchless 1.81-2.56× faster" finding was a function-call-overhead artifact; when both versions can be inlined, compiler-generated code is equally good for both source forms.
- The substrate now ships with ThinLTO globally → the choice between branchy and branchless is cosmetic, not performance-driven. **No production flip needed.** Substrate keeps branchy as the original.
- Without LTO, branchless WOULD have been faster (prior measurements were correct for that configuration). The project's switch to ThinLTO obviates the question.

### Pinpointed: the "full-LTO bug" was a TEST CODE BUG (not substrate, not LTO)
- `journal/tier2_residuals_v2_pinpoint.md` — full diagnostic trace, with V3 update section appended after red-team.
- **Root cause:** three places in `gesh/tests/test_image_canon.c` put side-effecting `image_canon_load_mnist(&ds, IDX_DIR)` inside `assert()`. Under `-DNDEBUG` (CMake Release default), `assert(EXPR)` becomes `((void)0)` and EXPR is never evaluated. The call was eliminated; `ds` was uninitialized; reading garbage stack memory caused the SEGV under aggressive LTO optimization.
- **Verified independently** via a 10-line test program: `assert((x = 42) == 42)` under NDEBUG leaves x=0 (assignment never runs).
- **And empirically** (during red-team): the original broken test under no-LTO ran to "PASS" exit 0 with `zero rate nan%` — proof the test was running with uninitialized ds.
- **Fixed in test code:** three side-effecting asserts replaced with explicit `if (rc != 0) { ...; exit(1); }`.
- **Hidden second bug surfaced:** test's mean-drift tolerance was `±dim` (too tight; real drift is ~5×dim due to rescaling step).
- **Full LTO now works globally.** Reverted from `-flto=thin` to `-flto`. 15/15 ctest binaries PASS.

### Added — V3 remediation: closed all 11 V2-pinpoint-redteam findings (100/100)
- `journal/tier2_residuals_v3_{precommit,closeout}.md` — methodical closure of red-team's 11 findings.
- **STRUCTURAL FIX (C1, C2, M3):** added `gesh_test_undebug()` helper to top-level CMakeLists; applied to all 15 test executables. Tests now compile with `-UNDEBUG` after the substrate's `-DNDEBUG`, so all test asserts actually run in Release. Substrate code unchanged (still NDEBUG, production behavior).
- **PRINCIPLED FIX (H1):** mean-drift tolerance derived from `dim * SCALE/10` ("post-normalize mean within 10% of unit scale"). Documented derivation. Replaces eyeballed `±10×dim`.
- **AUDIT (H2 + M1 + M3):** wide grep across m4t/src, m4t/tests, gesh/src, gesh/bench, gesh/tests. Zero side-effecting asserts anywhere in the codebase. All asserts are pure precondition checks (substrate-internal, NDEBUG-disabled in production = correct).
- **VERIFICATION (L1):** `otool -tv` shows 4 `bl _image_canon_load_mnist` calls in test binary — function is called (not eliminated). LTO chose external linkage; the bug was about ELIMINATION, not inlining-or-not.
- **DOCS (H3, H4, L2, M2):** V2 pinpoint amended with: "undefined behavior" framing instead of "no-op for months"; full-LTO measurements re-confirming branchy ≈ branchless (T2-B flip remains unnecessary); methodology lesson on early-hypothesis anchoring; exit(1) vs abort() note.
- **15/15 ctest binaries PASS** under full LTO with -UNDEBUG on all test executables.

### Methodology lifted from V3
- **Test executables should always compile with `-UNDEBUG`.** Codified via `gesh_test_undebug()` helper. Future tests should call it after `add_executable`.
- **Side-effecting expressions inside `assert()` are forbidden** even with `-UNDEBUG` (Debug gets the side effect, Release doesn't — ambiguous behavior). Use `if (!cond) { ...; exit(1); }` for control flow that must execute.
- **Test tolerance bounds should be derived, not eyeballed.** Document the derivation.
- **Wide-grep audits before declaring "0 bugs."** A targeted fix can leave the same anti-pattern elsewhere.

### Methodology lifted from V2
- **Compiler optimization profile can completely change perf comparison results.** Future perf claims should specify the optimization profile they assume; conclusions only generalize within that profile.
- **Cache-defeat verification needs workload-aware design.** Cache-trash is necessary but insufficient for prefetcher-friendly workloads.
- **Subagent designs including cache-aliasing imply allocator-level engineering work.** Budget for the engineering, not just the pattern.

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

## [2026-05-02 — SDOT fix: route ternary × ternary through m4t_mtfp4_sdot_matmul_bt]

The substrate-discipline cleanup wired `gesh_project` through `m4t_mtfp_ternary_matmul_bt`, which decodes packed-trit weights via ~30 NEON ops per 16 trits. For our ternary × ternary input class, the right kernel is **`m4t_mtfp4_sdot_matmul_bt`** — it accepts both X and W as `int8_t*`, uses `vdotq_s32` (1-cycle 16-lane signed-int8 SDOT), and skips packing/widening entirely.

Both kernels are substrate-legal. The packed-trit one is for general MTFP19×packed-trit; the SDOT one is for int8×int8 (which ternary trivially fits — |trit| ≤ 1 ≪ MTFP4's max 40). Cleanup wired the wrong kernel for our input class.

### Changes
- `gesh/src/gesh_project.c` — replaces `m4t_mtfp_ternary_matmul_bt` with a thin alias `ternary_matmul_sdot` that calls `m4t_mtfp4_sdot_matmul_bt(Y, (m4t_mtfp4_t*)x, (m4t_trit_t*)R, M, K, N)`. Drops R-packing scratch, drops X-widening scratch.
- `gesh/bench/denoise_probe.c` — same swap; drops `R_packed`, `X_train_mtfp`, `P_mtfp` allocations.

### Speedup (bit-equivalent, all numbers unchanged)

| measurement | open-coded* | packed-trit kernel | SDOT kernel |
|-------------|--------------|---------------------|--------------|
| MNIST ablation total | 210s | 1740s | **156s** |
| MNIST Cell A | 7.5s | 84.6s | **5.8s** |
| MNIST Cell B (10× budget) | 124s | 1033s | **90.4s** |
| MNIST Cell C (10× n_train) | 12s | 121s | **11.4s** |
| MNIST Cell D | 64s | 499s | **48.2s** |
| sig_dim sweep total | 515s | 2658s | **631s** |
| `test_gesh_train` | 0.84s | 3.5s | 1.14s |
| `gesh_denoise_probe` | ~1s | ~1s | **0.3s** |

*The "open-coded" column was measured **before** the substrate-discipline cleanup, on a code path that no longer exists. The "packed-trit" and "SDOT" columns are measured on the same checkout; the "open-coded ↔ SDOT" comparison is **cross-branch** and approximate. To make it strictly apples-to-apples we would have to revive the open-coded path temporarily for benchmark — not done.

**Regime split (per Phase-B-redteam H5):** the SDOT path is faster than open-coded on **matmul-large** workloads (MNIST D=784, sig=128, batch=128: 1.3× faster). On **matmul-small** workloads (synthetic-test scale: D=64, sig=16, batch=64), the per-call threshold-extract widen + unpack overhead can flip the comparison — `test_gesh_train` is 1.4× *slower* than the pre-cleanup open-coded measurement. Both regimes are bit-equivalent and substrate-routed. A scratch-aware threshold-extract would close the small-scale gap; not done since small scale isn't a substrate-claim hot path.

### Bit-equivalence verified
- `test_gesh_project`: 33/33 cells pass — kernel and reference open-coded loop produce zero differing trits.
- All probes (Gate 2 denoise, MNIST ablation A/B/C/D, sig_dim sweep through 512+ cells) produce numbers identical to the prior packed-trit-kernel path and the prior open-coded path.

### Key insight (atomics)

The matmul kernel decision is **input-class-driven, not arbitrary**:
- MTFP19 activations × packed-trit weights → use `m4t_mtfp_ternary_matmul_bt` (the kernel pays decode cost to compress weights).
- int8 activations × int8 weights → use `m4t_mtfp4_sdot_matmul_bt` (SDOT crushes this in 1 cycle per 16 elements).

For ternary × ternary, both inputs are int8 already; SDOT applies. The "packed-trit" kernel was overkill — it spent ~30 NEON ops per 16 trits decoding a format that didn't need decoding.

## [2026-05-02 — Finding 3 high-seed measurement (capacity floor at sig_dim ≤ 4)]

Original sweep ran 5 seeds per cell. At sig_dim ∈ {2, 4} the seed stddev was 1.6–3.1 pp on a 15–27% point estimate — wide enough that the "capacity floor" framing was directionally clean but quantitatively soft. Re-ran with **30 seeds per cell** at sig_dim ∈ {2, 4, 8} via new `gesh/bench/finding3_probe.c`.

### Results (30 seeds, permille precision)

| sig_dim | trained mean ± stddev | 95% CI on mean | gain over random |
|---------|------------------------|------------------|---------------------|
|       2 | **19.3% ± 3.26 pp**    | ±1.17 pp         | **+3.5 pp**         |
|       4 | **27.0% ± 3.22 pp**    | ±1.15 pp         | **+4.6 pp**         |
|       8 | **35.9% ± 3.39 pp**    | ±1.21 pp         | **+5.1 pp**         |

### Hardening of the capacity-floor claim
- Monotone climb 19.3 → 27.0 → 35.9 with non-overlapping 95% CIs (gap between cells >10× CI width). Capacity-bounded behavior at low sig_dim is now a finding, not a hypothesis.
- Lattice-update gain remains positive at all three cells with CI excluding zero. C1 holds at the capacity floor.

### Methodology refinement: integer-percent rounding bias in sweep_dims

The original sweep tool's `eval_test_accuracy` returns `(correct * 100) / n_test` — int division, flooring each seed's percent. Across 5 seeds, this systematically under-reports by ~0.5 pp; for the trained mean at sig_dim = 2, the bias was **+1.7 pp** (5-seed sweep claimed 21.0% vs 30-seed permille 19.3%).

The finding3 probe uses permille (`(correct * 1000) / n_test`, divided by 10 at print time). Future sweep tools should default to permille or higher precision. Surfaced by the cross-check that the high-seed sub-mean disagreed with the published 5-seed numbers despite identical seeds and data.

### Documentation updated
- `gesh/docs/sweep_dims_results.md` § Finding 3: corrected magnitudes; cross-references to high-seed doc.
- `gesh/docs/finding3_high_seed_results.md`: full 30-seed results, methodology note, capacity-ceiling discussion.

### Added
- `gesh/bench/finding3_probe.c` — 30-seed probe at sig_dim ∈ {2, 4, 8}.
- `gesh/docs/finding3_high_seed_results.md` — results doc.

## [2026-05-02 — Substrate-discipline cleanup: every MAC and sign-threshold runs through libm4t kernels]

A kernel-use audit caught that the gesh consumer library and bench code re-implemented ternary projection and sign-threshold by hand in **9 places**, when libm4t had `m4t_mtfp_ternary_matmul_bt` and `m4t_route_threshold_extract` for exactly those operations. Every prior substrate-claim measurement was running through hand-written loops, not the substrate kernels we built and tested. **Substrate-claim integrity required full kernel routing**; this cycle delivers it.

### What was open-coded → now kernel-routed

| Site | Was | Now |
|------|-----|-----|
| `gesh/src/gesh_forward.c` | open-coded MAC + sign threshold | `gesh_project_one_packed` |
| `gesh/src/gesh_train.c` rebuild | open-coded MAC + sign | `gesh_project_batch_unpacked_scratch` |
| `gesh/src/gesh_train.c` count_errors | open-coded MAC + sign per query | `gesh_project_batch_unpacked_scratch` per call |
| `gesh/src/gesh_bank.c` | open-coded sign threshold | `gesh_threshold_int32_to_trit` |
| `gesh/bench/sweep_dims.c` | open-coded MAC + sign | `gesh_project_batch_unpacked` |
| `gesh/bench/mnist_probe.c` | open-coded MAC + sign | `gesh_project_batch_unpacked` |
| `gesh/bench/denoise_probe.c::project_train_acc` | open-coded MAC | `m4t_mtfp_ternary_matmul_bt` direct |
| `gesh/bench/denoise_probe.c::prototype_alignment` | open-coded dot product | matmul + `col_stddev` |
| `gesh/bench/image_canon.c::quantize` | open-coded threshold | `m4t_route_threshold_extract` |

### What was added

- **`gesh/src/gesh_project.{c,h}`** — three substrate-routed wrappers plus a scratch-aware variant for hot loops:
  - `gesh_project_batch_unpacked` — batch project unpacked → unpacked. Allocates per call.
  - `gesh_project_one_packed` — single query → packed. Allocates per call.
  - `gesh_threshold_int32_to_trit` — int32 array → unpacked ternary via `m4t_route_threshold_extract`.
  - `gesh_project_batch_unpacked_scratch` + `gesh_project_scratch_init/free` — caller-managed scratch for hot loops (no per-call malloc).
  - All wrappers internally call `m4t_pack_trits_1d` + (for matmul cases) `m4t_mtfp_ternary_matmul_bt` + `m4t_route_threshold_extract` + `m4t_unpack_trits_1d`. Zero open-coded multiply-accumulate or sign threshold.

- **`gesh/tests/test_gesh_project.c`** — bit-equivalence property test:
  - 7 shapes × 3 seeds = 21 batch-projection equivalence checks vs reference open-coded loop. **Zero differing trits** across all 21 cells.
  - 4 sizes × 3 seeds = 12 threshold-extract equivalence checks. **Zero differences**, plus emission coverage holds (all three ternary states present per call).

- **`journal/gesh_substrate_discipline_cleanup.md`** — full cycle record.

- **CONTRIBUTING.md** — new methodology rule: *"Kernel use gates the substrate-claim."* Companion to the prior multi-seed and multi-config rules. Pattern: code red-teams catch kernel issues, doc red-teams catch ensemble drift, measurement red-teams catch methodology drift, **kernel-use audits catch substrate-claim drift**.

### Bit-equivalence verification

All measurements re-run post-cleanup produce **bit-identical** results to the pre-cleanup open-coded path:

- **Gate 2 (denoise probe):** Pearson r = +0.8921, t = 157.89 — identical. Stratification 3,649 / 7,451 / 11,404 — identical.
- **`test_gesh_project`:** 33/33 equivalence checks pass with zero differing trits.

The kernel and the open-coded loop produce identical ternary outputs because both are deterministic integer operations with the same arithmetic semantics. The cleanup is semantically neutral and substrate-claim-positive.

### Implications

- **Substrate-claim integrity:** every measurement supporting "base-3 routing-first" claims now runs through the libm4t kernels we cite as the substrate. Prior hand-written loops are gone from runtime paths.
- **Performance dishonesty corrected:** prior timing claims (e.g., sweep_dims at sig_dim=1024 in ~515s) were for hand-written loops, not kernels. Future timing claims will be on-substrate.
- **Discipline rule reciprocated:** principle 5 ("no primitive without named consumer demand") was satisfied when kernels were built. The flip-side (consumer actually uses the kernel) is now also satisfied.

### What was NOT cleaned up (justified non-kernel sites)

- **`image_canon::normalize_one`** — per-image preprocessing with integer isqrt. One-shot, not runtime; sanctioned per `M4T_SUBSTRATE.md` §12.
- **`gesh_init_random_projection_balanced`** — uniform-random ternary sampling, not a sign threshold.
- **`image_canon::cmp_i64`** — qsort comparator returning C-convention -1/0/+1. Not a ternary trit emitter.
- **Float in `compute_stats_pm`, Pearson r, etc.** — reporting only, not runtime.

### Build
14/14 ctest binaries green from clean rebuild. The new `test_gesh_project` is registered alongside the existing 13.

## [2026-05-02 — Phase B red-team remediation: ablation falsifies original closeout narrative]

The Phase B red-team's C1+H1+H2 critique flagged the original closeout's "consumer architecture is the bottleneck" claim as unsupported by the 2-cell single-config measurement. A 4-cell ablation isolating budget × n_train at sig_dim=128 plus a 5-cell C2 multi-config sweep was run.

### Ablation results (sig_dim=128)

| cell                | config                       | random        | trained       | gain        |
|---------------------|------------------------------|---------------|---------------|-------------|
| A: baseline         | n_train=2000,  budget=20K    | 50.7% ± 1.9pp | 51.6% ± 2.6pp | +0.8 pp     |
| B: 10× budget       | n_train=2000,  budget=200K   | 50.7% ± 1.9pp | 52.8% ± 2.8pp | **+2.0 pp** |
| C: 10× n_train      | n_train=20000, budget=20K    | 51.0% ± 1.9pp | 51.2% ± 1.9pp | +0.2 pp     |
| D: 10× both         | n_train=20000, budget=200K   | 51.0% ± 1.9pp | 52.0% ± 1.8pp | +1.0 pp     |

**Causal verdict: original FAIL was undertraining-dominated.** Cell B (10× budget) doubles the gain to +2.0pp — exactly the original gate's +2pp threshold. Cell C (10× n_train) adds +0.2pp (within noise; sample-size starvation was NOT the cause). The original closeout's claim *"lattice-update mechanism does not transfer to MNIST"* is **falsified**; C1 transfers at smaller magnitude (+2pp on MNIST vs +8pp on synthetic).

### Architecture ceiling — now properly supported

Trained accuracy across the 4 ablation cells caps at ~52–53%. 100× the original probe's compute budget (10× × 10×) does not move trained accuracy above 53%. The Phase A consumer's expressivity ceiling on MNIST is real — but this claim was **previously asserted from 1 cell, now demonstrated from 4**.

### C2 multi-config sweep — faithful test, +13.9pp on MNIST

| sig_dim | random          | gap vs identity@784 |
|---------|------------------|----------------------|
|     64  | 45.2% ± 5.0pp  |  +1.8 pp             |
|    128  | 50.7% ± 1.9pp  |  +7.3 pp             |
|    256  | 54.2% ± 1.7pp  | +10.8 pp             |
|    512  | 56.6% ± 0.6pp  | +13.2 pp             |
|    784  | **57.3% ± 1.1pp** | **+13.9 pp** |

**C2 in its faithful regime (random@D vs identity@D): +13.9pp on MNIST**, vs +7.4pp on the synthetic. Synthetic was structurally rigged (clean K-vs-(D-K) split); MNIST's more diffuse signal benefits the denoising mechanism more. The original "+7.3pp" claim was a regime-conflated comparison (compression random vs full-D identity); the faithful comparison is +13.9pp.

### Updated Phase B verdict

- **Original FAIL on absolute-accuracy bar still stands** (52.8% < 95%). Right verdict, partly wrong reason.
- **Gain bar PASSES at 10× budget** (+2.0pp ≥ original +2pp threshold).
- **Path A (richer consumer) still the right move,** for a refined reason: lattice-update *does* contribute small gains; richer consumer should let it contribute proportionately more.

### Path A pre-committed Gate 1.A (M4 fix)

Specified explicitly per the red-team's methodology critique:
- **PASS:** Gesh + multi-table LSH consumer ≥ **92% MNIST** AND beats `mnist_routed_bucket_multi` (random R, identical consumer config) by ≥ **+1pp**.
- **FAIL:** trained < 88% OR no measurable delta over random-R baseline with same consumer.
- The +1pp delta is strict — forces Gesh to demonstrate substrate-claim contribution rather than just consumer upgrade.

### Methodology lesson promoted

CONTRIBUTING.md gains a new rule: **multi-config gates the story; multi-seed gates the cell.** Single-seed → seed-noise narrative artifact (caught in Phase A.2 red-team). Single-config → config-confound causal artifact (caught here). Pattern: single-N supports a verdict at N; the *interpretation* requires N>1 along the dimension being attributed.

### Code remediations applied

- **M1** — `mnist_probe.c::subsample` comment corrected (was claiming Floyd's algorithm; actually with-replacement uniform). Function renamed `subsample_with_replacement` for honest naming.
- **M2** — Aliasing assertion added to `image_canon_quantize_unpacked_batch`.
- **M3** — `gesh/tests/test_image_canon.c` smoke test added; verifies IDX load, normalize invariants, quantize density. Registered in ctest.
- **M4** — Path A's pre-committed Gate 1.A specified in closeout (above).
- **CONTRIBUTING.md** — multi-config rule added to the post-commit doc-currency checklist.

### Added
- `gesh/tests/test_image_canon.c` — smoke test covering IDX load + normalize + quantize.
- Ablation result rows in `phase_b_gate1_results.md` and `gesh_phase_b_probe_closeout.md`.
- Pre-committed Gate 1.A in the closeout (M4 fix).

### Changed
- `gesh/bench/mnist_probe.c` — full rewrite for ablation design (Cells A–D + C2 multi-config sweep). Original 2-cell version is in git history at commit 500ddaf.
- `gesh/bench/image_canon.c` — aliasing assertion in `image_canon_quantize_unpacked_batch`.
- `gesh/CMakeLists.txt` — registered `test_image_canon`; reordered `gesh_image_canon` library declaration to come before the test block.
- `gesh/docs/phase_b_gate1_results.md` — full rewrite with corrected causal narrative.
- `journal/gesh_phase_b_probe_closeout.md` — revision banner + post-red-team revised reads + Gate 1.A.
- `CONTRIBUTING.md` — multi-config methodology rule added.

### Build
13/13 ctest binaries green from clean rebuild. Remediated probe runs in ~210s on Apple Silicon.

## [2026-05-02 — Gesh Phase B probe: Gate 1 FAIL, Gate 2 PASS]

Executed the two pre-committed gates from `journal/gesh_findings_synthesize.md`.

### Gate 1 — image canon parity (MNIST): FAIL

Lifted the canonical ternary-pixel-quantization pipeline from `01MAY26_archived/` into `gesh/bench/image_canon.{c,h}` (substrate-legal: per-image normalize → direct ternary quantization at calibrated tau, no random projection of pixels). Built `gesh/bench/mnist_probe.c` running Gesh forward + lattice-update against MNIST.

| sig_dim | random          | trained         | gain    |
|---------|------------------|------------------|---------|
|     128 |  50.7% ± 1.9pp |  51.6% ± 2.6pp |  +0.8 pp |
|     256 |  54.2% ± 1.7pp |  54.7% ± 1.6pp |  +0.5 pp |

Identity at sig_dim = 784: 43.4%. Pre-committed PASS bar was ≥95% with ≥+2pp gain; trained Gesh sits **far below the 90% inconclusive floor** with gain within seed noise.

**What survives:** random R at sig_dim=128 beats identity at sig_dim=784 by **+7.3pp** — the same magnitude C2 showed on the synthetic. Substrate-level finding (random ternary projection > identity) transfers cleanly to MNIST.

**What fails:** the lattice-update mechanism's compression-regime gain (synthetic +5–8pp) does not transfer. The Phase A consumer architecture (single class-mean bank, top_k=1) is the bottleneck — its expressivity ceiling on MNIST is 50–55%, leaving lattice-update no informative loss surface.

### Gate 2 — H1 mechanism test: PASS

Built `gesh/bench/denoise_probe.c` testing the "implicit denoising via random ternary projection" mechanism. For each output dim *j* of random R, scored x[j] = stddev_c(R[j] · P_c) (prototype-subspace alignment) and y[j] = max-min spread of per-class projection-mean. 100 random R samples × 64 output dims = 6400 observations.

- **Pearson r = +0.8921**, t = 157.89 (df = 6398), **p << 0.001**.
- Stratification monotone: low alignment (n=7) → mean spread 3,649; mid (n=1267) → 7,451; high (n=5126) → 11,404.

H1 upgraded from hypothesis to **demonstrated mechanism** within the synthetic benchmark's domain. The C2 finding now has a measured story.

### Pre-commit-and-honor methodology held

Gate 1 failed honestly; no post-hoc tuning to push MNIST accuracy up. The clean Gate 1 / Gate 2 split (Gate 1 fails, Gate 2 passes) localizes the failure to the consumer architecture, not the substrate or the projection. Loop-back to NODES recorded in `journal/gesh_phase_b_probe_closeout.md`.

### Added
- `gesh/bench/image_canon.{c,h}` — MNIST IDX loader + per-image normalize + direct ternary quantization to unpacked m4t_trit_t. Lifted from `01MAY26_archived/src/{glyph_dataset.c, glyph_sig.c}`; trimmed to MNIST + normalize + quantize (deskew, gradients, CIFAR loader deferred).
- `gesh/bench/mnist_probe.c` — Phase B Gate 1 probe.
- `gesh/bench/denoise_probe.c` — Phase B Gate 2 mechanism test.
- `gesh_image_canon` static library + `gesh_mnist_probe`/`gesh_denoise_probe` executables in `gesh/CMakeLists.txt`.
- `gesh/docs/phase_b_gate1_results.md` — Gate 1 results, FAIL verdict, mechanism analysis.
- `gesh/docs/phase_b_gate2_results.md` — Gate 2 results, PASS verdict, H1 demonstrated.
- `journal/gesh_phase_b_probe_closeout.md` — gate verdicts, loop-back-to-NODES action, re-evaluation of prior NODES against MNIST data, recommended next cycle (Path A: richer consumer; Path B: different lattice-update objective).

### Changed
- `gesh/docs/sweep_dims_results.md` — H1 marked DEMONSTRATED MECHANISM with Phase B Gate 2 reference.
- `gesh/README.md` — Phase B probe status block added; gate verdicts surfaced.

### Findings table

| Prior node | Status post-MNIST |
|------------|-------------------|
| C1 — lattice update earns +4–8pp in compression | **Synthetic-specific.** |
| C2 — random > identity at sig_dim=D by +7pp | **Transfers** (+7.3pp on MNIST). |
| H1 — implicit denoising mechanism | **Mechanism demonstrated** (Gate 2). |

## [2026-05-02 — Gesh Phase A.2 sweep extended to sig_dim = 1024]

Re-ran the multi-seed sweep with four additional expansion ratios: 384, 512, 768, 1024. Confirms expansion saturation is monotone — random and trained accuracies converge tightly at every extreme dim, with gain pinned to ≤ +0.2pp.

| sig_dim | random         | trained        | gain    |
|---------|------------------|------------------|---------|
|     384 |  96.8% ± 0.4pp |  96.8% ± 0.4pp |  +0.0 pp |
|     512 |  97.4% ± 0.5pp |  97.6% ± 0.5pp |  +0.2 pp |
|     768 |  98.2% ± 0.4pp |  98.2% ± 0.4pp |  +0.0 pp |
|    1024 |  98.6% ± 0.5pp |  98.6% ± 0.5pp |  +0.0 pp |

At 16× the input dimensionality, random ternary projection asymptotes to 98.6% on this benchmark. There is no inflection upward at any expansion ratio — training does not start beating random again. The expansion-regime saturation is stable, not a transient.

Per-seed stddev tightens to ~0.4pp as accuracy approaches the test-set ceiling. Total sweep runtime: 515s (12 sig_dims × 5 seeds × 2 variants).

### Changed
- `gesh/bench/sweep_dims.c` — `dims[]` extended to `{2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024}`.
- `gesh/docs/sweep_dims_results.md` — table extended; new finding #5 ("Expansion saturation is monotone all the way to sig_dim = 1024").
- `gesh/README.md` — Phase A.2 status block notes 16× expansion saturation.

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
