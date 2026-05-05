# CLOSEOUT: ternary MAC routing — productionized

Per `journal/ternary_mac_routing_synthesize.md`. All 10 T-G gates PASS. The vmlal_s32-routed ternary MAC is now the production NEON path for `m4t_mtfp_ternary_matmul_bt`.

## Verdict: PASS — all 10 gates closed

```
T-G1  (vmlal_s32 throughput characterization)         : PASS — 0.84 calls/cycle (pattern C)
T-G2  (m4t_mtfp_ternary_matmul_bt_scalar_ref exposed) : PASS — public API; LTO can't DCE
T-G3  (vmlal path prototype)                          : PASS — compiles + ctest passes
T-G4  (bit-exact vs scalar_ref)                       : PASS — 23 configs across K boundary, distributions, bulk shapes
T-G5  (saturation argument)                           : PASS — int64 acc bound holds for K up to ~1.59e10
T-G6  (aliasing assertion)                            : PASS — Y==X correctly aborts via SIGABRT
T-G7  (disasm verification)                           : PASS — 4× smlal.2d + 4× smlal2.2d in inner loop, ldp paired loads
T-G8  (bench discipline)                              : PASS — workload shape declared, min-of-5
T-G9  (productionized)                                : PASS — 19/19 ctest with NEON path active
T-G10 (no regression in production binaries)          : PASS — bench_m4t_tier2_perf, 2 gesh probes identical
```

## Headline result

The user named ternary MAC as "software doing the work of hardware" earlier in the session. The substrate's prior `m4t_mtfp_ternary_matmul_bt` used a bsl + conditional-negate pipeline (~57 NEON ops per 16-trit block, dominated by mask-widening). The new `vmlal_s32`-routed path uses the multiply-by-trit shortcut: trit ∈ {-1, 0, +1} means multiplying by the trit IS the MAC, with the int64 widening absorbing the accumulator semantics. The mask-widening cost (~40 of the original 60 ops) collapses entirely.

**Measured speedup (T-G8, before T-G9 productionization swap):**

| Shape | scalar_ref | bsl-NEON | vmlal | bsl gain | vmlal gain | vmlal vs bsl |
|-------|-----------:|---------:|------:|--------:|----------:|-------------:|
| BATCHED (M=64, K=4096, N=64) | 10996 ns/cell | 766 | **657** | 14.3× | 16.7× | **1.17×** |
| TIGHT-LOOP (M=4, K=64, N=4) | 24.75 ns/cell | 12.25 | **5.00** | 2.0× | 5.0× | **2.45×** |

## Per-gate disposition

| Gate | What was done | Artifact |
|------|--------------|----------|
| **T-G1** | New `m4t/tools/bench_vmlal_throughput.c` (3 patterns: independent, two-chain, single-chain). Two iterations of constant-fold defense (first attempt got compiler-folded to `add+branch`; fix: distinct inputs per call from heap pool). Pattern C (matches kernel) measured 0.84 vmlal/cycle. | `m4t/tools/bench_vmlal_throughput.c` |
| **T-G2** | Added public `m4t_mtfp_ternary_matmul_bt_scalar_ref`. Lifted `ternary_dot_scalar` static helper. Production never calls scalar_ref; tests do. Same shift3-remediation pattern: separately-preserved oracle survives productionization. | `m4t/src/m4t_ternary_matmul.{h,c}` |
| **T-G3** | Added `static int64_t ternary_dot_vmlal(...)` in m4t_ternary_matmul.c. Initial wrapper `m4t_mtfp_ternary_matmul_bt_vmlal` (later removed at T-G9). | (intermediate; fold at T-G9) |
| **T-G4** | New `m4t/tests/test_m4t_ternary_matmul_neon.c` (originally `_vmlal.c`; renamed at T-G9). Coverage: 11 K boundary cases × 6 distributions × bulk shapes × seeds = 23 configurations. All bit-exact. | `m4t/tests/test_m4t_ternary_matmul_neon.c` |
| **T-G5** | Inline saturation argument in source comment. Bound: \|acc\| ≤ K × MAX_VAL = K × 5.81×10⁸; for K ≤ 1.59×10¹⁰, fits int64. Same K-bound as the existing bsl-NEON path; no new constraint. | inline in `m4t_ternary_matmul.c` |
| **T-G6** | Fork-and-verify-SIGABRT pattern (lifted from V4-G2 meta-test discipline). Y==X correctly aborts on assert(...). | inline in this cycle's bash |
| **T-G7** | `otool -tv` verified inner loop has `ldp q18, q19` paired loads + 4× `smlal.2d` + 4× `smlal2.2d`. No scalar mul/madd. Compiler split each source `vmlal_s32` into the proper low/high pair. | `otool` output preserved here |
| **T-G8** | Min-of-5 across two workload shapes (BATCHED, TIGHT-LOOP). Per CONTRIBUTING scope-match rule. Numbers above. | inline in test source |
| **T-G9** | Replaced `ternary_dot()` body with `#if M4T_HAS_NEON ? ternary_dot_vmlal : ternary_dot_scalar`. Removed `m4t_mtfp_ternary_matmul_bt_vmlal` from public API (was prototype-only). Renamed test file `_vmlal.c` → `_neon.c`. The prior bsl-NEON code is preserved in git history per project rule. | `m4t/src/m4t_ternary_matmul.{h,c}`, `m4t/tests/test_m4t_ternary_matmul_neon.c` |
| **T-G10** | Smoke-tested 3 production-linked binaries (`bench_m4t_tier2_perf`, `gesh_confidence_probe`, `gesh_expr_routing_probe`) before/after; outputs identical. | inline in this cycle's bash |

## What shipped

- `m4t/src/m4t_ternary_matmul.h` — declared `m4t_mtfp_ternary_matmul_bt_scalar_ref`. Removed transient `_vmlal` declaration.
- `m4t/src/m4t_ternary_matmul.c` — refactored: `ternary_dot_scalar`, `ternary_dot_vmlal` (NEON-only), `ternary_dot` dispatcher. New `m4t_mtfp_ternary_matmul_bt_scalar_ref` outer-loop variant. Bsl-NEON code removed (preserved in git history).
- `m4t/tests/test_m4t_ternary_matmul_neon.c` — bit-exact regression test + perf bench. Compares production NEON path against scalar_ref oracle.
- `m4t/tools/bench_vmlal_throughput.c` — vmlal_s32 throughput characterization microbench.
- `m4t/CMakeLists.txt` — new ctest entry `m4t_ternary_matmul_neon`.
- `journal/ternary_mac_routing_{raw,nodes,reflect,synthesize,closeout}.md` — full LMM cycle docs (SYNTHESIZE includes a header note about the user-caught consumer-demand drift early in the session).

## What's structurally true now

**The substrate's `m4t_mtfp_ternary_matmul_bt` ternary MAC routes through vmlal_s32 on Apple Silicon.** Production NEON path is now the closest existing hardware analog to a "ternary MAC at int32 width." The mask-widening + bsl pattern that emulated the multiply-by-trit is replaced with the multiply itself. Both forms are bit-exact (verified across 23 configurations); the multiply form is faster (1.17× to 2.45× depending on workload shape).

**The bit-exact verification gate is structurally sound from the start** (lesson lifted from shift3 remediation). `m4t_mtfp_ternary_matmul_bt_scalar_ref` was exposed BEFORE the prototype, so the post-productionization verification (T-G4 + T-G9 re-run) compares production-NEON against an independent scalar oracle — not against itself.

## Methodology lifted from this cycle

**1. Cycle-level pre-emption of the shift3 productionization invalidation.** The shift3 remediation taught us: when productionization replaces the function under test, the bit-exact gate compares two copies of the new code instead of new vs. reference. This cycle exposed `_scalar_ref` at T-G2 (BEFORE the prototype at T-G3), so productionization at T-G9 doesn't break verification. Apply this pattern: **expose the reference oracle as the FIRST gate, before any prototype work.**

**2. Constant-folding defense for throughput microbenchs.** Compiler will factor `acc += K * (a*b)` into `acc + K*(a*b)` if a, b are constants OR repeated across calls within a loop. Defenses needed: (a) read inputs from a heap pool with non-constant addressing per iteration, (b) make each call use distinct inputs (not just the loop-invariant pair), (c) `__attribute__((noinline))` to prevent caller-side folding. Took two iterations to get clean throughput numbers in this cycle's T-G1.

**3. The cross-cycle reification catch worked.** The synthesize doc was rewritten in-place after the user caught me re-introducing consumer-demand framing as a "lesson from shift3." Memory updated to add the catch trigger; this cycle's execution stayed clean of the drift. The contaminated NODES/REFLECT remain as the cycle's process record.

## Honest concerns from this cycle

**1. The BATCHED 1.17× speedup is modest.** Most of bsl-NEON's cost was already optimized away by the compiler at large K (memory bandwidth dominates compute). The vmlal path's structural simplification (mask-widen-and-bsl → multiply) yields a small relative gain at BATCHED. The TIGHT-LOOP 2.45× is more compelling. Both numbers are workload-shape-dependent; CONTRIBUTING.md scope-match rule applies.

**2. The bsl-NEON code is preserved only in git history.** Per project rule "DELETE = never," superseded code is preserved; git log counts. If a future cycle needs to reference the bsl approach (e.g., for a different cell width that doesn't fit vmlal's int32×int32→int64 shape), it's recoverable via `git show`. Not archived to a separate file.

**3. No cross-exp accumulator analog.** The cross-exponent accumulator (`m4t_mtfp_vec_accum_aligning`) does per-cell-varying division, not packed-trit MAC; this cycle's vmlal_s32 routing doesn't apply there. Different kernel, different optimization story. Out of scope.

**4. Case W (SDOT-direct via MTFP4 activations) remains the strategically larger move.** When activations fit in int8, `m4t_mtfp4_sdot_matmul_bt` runs at SDOT-native throughput (~1 NEON op per 16 elements) — vastly faster than even vmlal at int32 width. This cycle is about Case S (int32 activations), where SDOT can't apply. Consumer-side audit of which workloads NEED int32 activations vs which could use MTFP4 is a separate, structurally larger conversation.

## Status

CLOSED — production substrate now routes the ternary MAC through `vmlal_s32` on Apple Silicon NEON. Bit-exact verified against the scalar reference across 23 configurations + boundary classes. 19/19 ctest. No regression in 3 production binaries. The user-named "ternary MAC = software doing the work of hardware" gap is now closed at int32 width via the closest existing hardware analog.

Followups (deferred per scope): cross-exp accumulator per-cell-varying-k variant; consumer audit for Case W migration eligibility; any further tuning of the trit decode itself.
