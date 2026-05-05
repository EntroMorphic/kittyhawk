# SYNTHESIZE: ternary MAC routing

**REWRITE NOTE (2026-05-04):** the first version of this doc gated the cycle on a "consumer audit" — exactly the consumer-demand framing the user has explicitly disclaimed for foundational substrate work. The drift came from over-fitting an observation from the shift3 NEON benchmark ("no current consumer touches it") into a constraint for future cycles. User caught it; I've recorded the catch in memory. This version drops the consumer-audit gating. The contaminated NODES (N16, N20–N22, N25, N26) and REFLECT ("audit-first," "cross-cycle observation: same shape as shift3") stand as part of the cycle's record, but the SYNTHESIZE here is the corrected output.

The directive ("ternary MAC is software doing hardware's work; we cannot develop silicon; what existing M4/NEON features can we lean on?") IS the justification. Prototype the vmlal_s32 path. Verify bit-exact. Productionize.

## Decision

**Prototype the vmlal_s32 routing for `m4t_mtfp_ternary_matmul_bt`.** The technical analysis is sound (RAW, NODES N1–N15): vmlal_s32 is the closest existing hardware analog to the int32×trit MAC, multiply-by-trit subsumes the bsl pattern, theoretical ~3× kernel-level reduction.

Apply the methodology lessons from prior cycles where they're actually load-bearing:
- **Bit-exactness must survive productionization** (shift3 remediation lesson): expose `m4t_mtfp_ternary_matmul_bt_scalar_ref` BEFORE the prototype, not as remediation after. Get it right the first time.
- **Workload-shape declared per measurement** (V4-residual-3 lesson): perf claims always name the shape they're under.

## Pre-committed gates

These are sequential. Each gate's pass condition gates the next.

### T-G1 — Throughput characterization microbench

**Artifact:** new bench file (~50 lines) timing `vmlal_s32` in a tight loop matching the proposed kernel's dependency structure (8 calls accumulating into one int64 pair per block). Measures realistic ns/instr or ops/cycle on the target hardware.

**Pass:** measurement complete; estimated kernel-level speedup updated from theoretical ~3× to empirical-throughput-derived number.

**Why:** the ~3× estimate is theoretical; vmlal_s32 throughput on M-series is unknown to me. Real number could be 2× or 4×. We need it to size the prototype's expected return.

**Budget:** ~30 min.

### T-G2 — Expose `m4t_mtfp_ternary_matmul_bt_scalar_ref`

**Artifact:** add a public scalar-only reference function to the substrate API (`m4t_mtfp_ternary_matmul_bt_scalar_ref` in m4t_ternary_matmul.h). Same semantics, always uses the existing scalar tail loop, never NEON. Lifts the shift3 remediation pattern preemptively — production never calls it; tests use it as the bit-exact oracle.

**Pass:** function exposed; symbol present in `nm libm4t.a`; documented in header as test-only.

**Budget:** ~15 min.

### T-G3 — Prototype the vmlal_s32 path

**Artifact:** new `static void ternary_matmul_bt_vmlal_path(...)` inside m4t_ternary_matmul.c (NOT yet wired into the production dispatcher). Pipeline:
- Decode 16 packed trits → 16 int8 signs (~6 ops, reuse existing TBL pipeline)
- Sign-extend int8 → int32 (~4 ops via vmovl_s8 → vmovl_s16)
- 8× vmlal_s32 multiply-accumulate into int64x2 pair
- Plus loop control + final horizontal reduce

**Pass:** compiles; standalone callable; placeholder in production dispatcher gated off.

**Budget:** ~1 hour.

### T-G4 — Bit-exact verification (vs scalar reference)

**Artifact:** new ctest entry `m4t_ternary_matmul_vmlal_bitexact`. Compares `ternary_matmul_bt_vmlal_path` against `m4t_mtfp_ternary_matmul_bt_scalar_ref` across:
- Random samples per (M, K, N) shape
- Boundary cases: K=0, K=15 (no NEON path used), K=16 (one NEON block + 0 tail), K=17 (one NEON block + 1 tail), K=4096 (many blocks)
- All 4 trit codes (0b00, 0b01, 0b10, 0b11) at sampled positions
- Random sign distributions, sparse-zero distributions, all-positive, all-negative
- Activation extremes: ±MAX_VAL, 0, ±1

**Pass:** zero mismatches across the test set. Not exhaustive (matmul state-space is too large) but covers the boundary classes.

**Budget:** ~1 hour.

### T-G5 — Saturation argument

**Artifact:** written argument (in source comment + journal) that the int64 accumulator never overflows for valid inputs. Per existing substrate spec: int64 acc + per-block sum ≤ 16 × MAX_VAL ≈ 9.3 × 10⁹ << INT64_MAX = 9.2 × 10¹⁸. Hugely safe.

**Pass:** argument written; bound is general (not table-specific).

**Budget:** ~10 min.

### T-G6 — Aliasing test

**Artifact:** test cases for the typical aliasing patterns (Y == X is forbidden by the existing substrate contract; verify the kernel still asserts that, AND that legitimate non-aliasing patterns work).

**Pass:** assert fires for forbidden alias; non-aliased patterns produce correct results.

**Budget:** ~15 min.

### T-G7 — Disasm verification

**Artifact:** `otool -tv` output showing the prototype emits `smlal.2d`, sign-extension chains, and no scalar `mul`/`madd` in the inner loop.

**Pass:** disasm matches expectation; LTO inlines the helper.

**Budget:** ~15 min.

### T-G8 — Bench discipline

**Artifact:** perf comparison `ternary_matmul_bt_vmlal_path` vs `m4t_mtfp_ternary_matmul_bt_scalar_ref` (the actual scalar oracle, not the current bsl-NEON path — that's a separate question). Workload shapes declared:
- Shape A: BATCHED matmul (M=64, K=4096, N=64) — typical bulk shape
- Shape B: TIGHT-LOOP (M=4, K=64, N=4) — small dims, per-call overhead bound

Min-of-5 sampling each. Plus a third comparison for completeness: `ternary_matmul_bt_vmlal_path` vs the CURRENT `m4t_mtfp_ternary_matmul_bt` (bsl-NEON path) — that tells us how much of the current ~60 ops we actually saved.

**Pass:** numbers reported with shape declared. No PASS/FAIL bar; just measurement.

**Budget:** ~30 min.

### T-G9 — Productionize

**Artifact:** wire `ternary_matmul_bt_vmlal_path` into `m4t_mtfp_ternary_matmul_bt` (production dispatcher), gated by `M4T_HAS_NEON`. The existing bsl-NEON path becomes the explicit non-vmlal alternative or is removed if vmlal beats it cleanly. The scalar reference (T-G2) remains untouched.

**Pass:** production substrate dispatches to vmlal path when available. 18/18 ctest still PASS. Bit-exact verification (T-G4) re-runs AFTER productionization against the scalar reference.

**Budget:** ~45 min.

### T-G10 — No regression in production binaries

**Artifact:** smoke-test the production binaries that exist:
- `bench_m4t_tier2_perf`
- `gesh_confidence_probe`
- `gesh_expr_routing_probe`

before/after, confirming output is unchanged.

**Pass:** outputs identical.

**Budget:** ~10 min.

## Order of execution

T-G1 → T-G2 → T-G3 → T-G4 → T-G5 → T-G6 → T-G7 → T-G8 → T-G9 → T-G10. Each gate stops the cycle on FAIL.

## Risk register

- **R1 (T-G1 throughput disappoints):** vmlal_s32 turns out to be half-rate. Speedup estimate degrades. **Action:** continue anyway — the substrate is still better with the cleaner pipeline (multiply-by-trit is structurally clearer than the mask-widen-and-bsl pattern), even if the speedup is smaller.
- **R2 (T-G4 bit-exact fails):** vmlal path produces different output than the scalar reference. Math says it shouldn't (multiply-by-trit ≡ conditional-negate-or-zero), but check empirically. **Action:** debug to root cause; abandon if not resolvable.
- **R3 (T-G8 shows bsl-NEON beats vmlal):** the current path is actually faster than vmlal for some shapes. Possible if the bsl pattern parallelizes differently. **Action:** keep both, dispatch by shape, OR pick whichever wins on the substrate's typical workloads. Document.
- **R4 (T-G9 productionization breaks something):** shift3 remediation showed this can happen. **Mitigation:** T-G2 already exposed the scalar reference, so bit-exact gate (T-G4) re-run post-productionization is meaningful.

## What's NOT in this cycle

- **Custom silicon design.** Out of scope (constraint).
- **Case W consumer migration.** Different decision; substrate-design-level.
- **Modifying `m4t_mtfp4_sdot_matmul_bt`.** Already SDOT-direct; nothing to add.
- **General matmul optimization.** Specifically routing the int32×trit MAC through vmlal_s32; not matmul as a class.
- **Trit decode tuning.** Already tight (~6 ops); unchanged.

## Done when

T-G1 through T-G10 PASS. CLOSEOUT documents per-gate verdict and any methodology lifted.

## Status

Pre-committed (rewritten). Ready to begin T-G1.
