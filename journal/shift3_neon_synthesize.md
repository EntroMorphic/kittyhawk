# SYNTHESIZE: NEON magic-multiply divide-by-3^k prototype

Pre-committed plan + gates derived from `shift3_neon_reflect.md`. This document is what the next cycle will execute against and verify against.

## Decision

**Productionize, but only after closing the four substantive gaps from REFLECT.** Do not ship the kernel as-is. The prototype's 10× speedup and bit-exact constant verification are credible foundations but the kernel-level evidence is sample-based and several engineering checks are missing.

The cross-exp accumulator question (per-cell-varying k) is deferred to a future cycle — not blocking THIS cycle's productionization. This cycle ships the same-k batched divide path inside `m4t_mtfp_shift3`. The cross-exp variant is a separate, structurally larger piece of work.

The original `vqrdmulhq + per-k-specialized vrshrq_n_s32` path is also deferred. The 64-bit-intermediate path is good enough; chasing a further 1.5–2× before having a consumer that benefits is premature.

## Pre-committed gates (must ALL pass before productionization commit)

These gates lock in the verdict before any production code is written. Each gate names the artifact that proves it.

### G1 — Exhaustive bit-exact NEON-vs-scalar verification
**Artifact:** add an exhaustive verification mode to `test_m4t_shift3_neon_proto.c` that runs every x in `[-MAX_VAL, +MAX_VAL]` for every k ∈ [1, 19], comparing the NEON kernel's output against `m4t_mtfp_shift3`'s scalar output bit-by-bit.
**Pass:** zero mismatches across 19 × 1.16 × 10⁹ = 2.2 × 10¹⁰ test points.
**Runtime budget:** ~5 minutes (acceptable for a one-time verification).

### G2 — Saturation proof
**Artifact:** a written argument (in the productionization commit message or journal) that for all `x ∈ [-MAX_VAL, +MAX_VAL]` and all k ∈ [1, 19], the int64 intermediate `x*M + (1 << (N-1))` after arithmetic-right-shift by N produces a value within `[INT32_MIN, INT32_MAX]`. The argument should be derivable from the table values, not empirical.
**Pass:** the argument is sound and the bound is verified empirically by G1.

### G3 — Aliasing test
**Artifact:** a test case in the property test that exercises `m4t_shift3_div_neon(buf, buf, abs_k, n)` (dst == src) and confirms the result matches the scalar `m4t_mtfp_shift3(buf, buf, -abs_k, n)`.
**Pass:** bit-exact for at least k ∈ {1, 10, 19} (the boundaries + middle), n covering NEON-aligned + tail cases (n=4, n=5, n=64, n=65).

### G4 — Disasm verification
**Artifact:** `otool -tv` output for both the scalar `m4t_mtfp_shift3` and the NEON `m4t_shift3_div_neon` under `-O3 -mcpu=native -flto`, with a brief annotation of what's emitted.
**Pass:** scalar's hot loop is NOT auto-vectorized to NEON ops (i.e., the perf comparison is fair); NEON kernel inlines as expected with `vmull` / `vmlal` / `vshl` / `vmovn` ops visible.

### G5 — Bench discipline applied
**Artifact:** rerun the perf bench with workload-shape declaration (carry-free vector divide, n=4096), min-of-5 runs, plus at least ONE adversarial-shape variant (e.g., n=4 calling shift3 in a tight loop, simulating per-call overhead).
**Pass:** speedup factor reported with workload shape named explicitly; both min-of-5 numbers reported. Per the project rule from concern #4, the speedup claim must specify the shape.

### G6 — Productionization wires kernel into the existing primitive
**Artifact:** modified `m4t/src/m4t_mtfp.c` where `m4t_mtfp_shift3`'s divide-direction path uses `m4t_shift3_div_neon` (renamed to `m4t_mtfp_shift3_div_neon` for the substrate naming convention) when `M4T_HAS_NEON` and `abs_k ∈ [1, 19]`. The scalar reference path remains, gated to non-NEON / abs_k ≥ 20.
**Pass:** 17/17 ctest binaries (existing) still PASS. The new property test (`test_m4t_shift3_neon`) passes ctest. No collateral damage to consumer probes.

### G7 — Magic table source-of-truth
**Artifact:** committed magic constants live in exactly one place. Either `m4t/src/m4t_pow3_magic.h` (committed, regenerable via `gen_pow3_magic`) OR generated at build time via a CMake custom_command. The test source includes the same header, no copies.
**Pass:** no two files contain the M_table or N_table values.

### G8 — No regression in production binaries
**Artifact:** smoke-test 3 production-linked binaries (`bench_m4t_tier2_perf`, `gesh_confidence_probe`, `gesh_expr_routing_probe`) before-and-after, confirming output is unchanged.
**Pass:** outputs match.

## Order of execution

1. **G1 first.** Cheapest to add (extend the existing test), highest information value. If G1 fails, the prototype is wrong and everything downstream is moot. Run G1, see the result, decide.
2. **G2 + G3** in parallel. Both are local to the prototype; ~30 min each.
3. **G4** quick disasm check (~10 min).
4. **G5** bench rerun with discipline (~30 min including variant).
5. **G7** factor magic table to its own header (~30 min).
6. **G6** productionization edit (~1 hour including ctest pass).
7. **G8** smoke test (~10 min).

If any gate fails, STOP and re-enter LMM (RAW for the failure, etc.). Don't push past failures.

## Risk register

- **R1 (G1 risk):** NEON intrinsic semantics differ from C emulator on some edge case. Mitigation: G1's exhaustive run surfaces this. If found, options are (a) adjust the kernel to match the emulator behavior, or (b) regenerate the table against the actual NEON behavior. (b) is preferable.

- **R2 (G2 risk):** the int64-intermediate overflows int32 for some boundary x. The bound `|q| ≤ MAX_VAL/3 ≈ 1.94e8` is well below INT32_MAX = 2.15e9, so unlikely. But not formally proved. Worst case: the proof reveals a saturation gap and the kernel needs explicit clamp logic. Adds 1-2 cycles, takes the speedup from 10× to ~9×.

- **R3 (G4 risk):** AppleClang IS auto-vectorizing the scalar path. If true, the 10× evaporates because both paths are NEON. Honest update to the speedup number; productionization may not be worth shipping. Expected probability: low (the substrate's scalar uses runtime-variable divisor, harder to auto-vectorize) but not zero.

- **R4 (G5 risk):** the speedup is shape-dependent in ways the original bench didn't catch. Per-shape numbers reported honestly. If pipelined (no carry) shape shows 10× and tight-loop (per-call overhead bound) shape shows 2×, ship both numbers and let the consumer pick.

- **R5 (G7 risk):** committing generated constants creates regeneration discipline (constants drift if not regenerated when generator changes). Mitigation: a CMake custom_target that regenerates the header on every configure, with a CI check that the committed header matches what the generator currently produces. Slightly heavier.

## What's NOT in scope this cycle

Explicitly deferred:
- **Cross-exp accumulator (per-cell-varying k).** Separate, larger work. New cycle when there's a perf reason.
- **Multiply direction (k > 0).** Currently scalar `(int64_t)src[i] * scale`. Likely already efficient; investigate only if a consumer measures it as a bottleneck.
- **Original vqrdmulhq + per-k-specialized path.** ~1.5–2× headroom over THIS cycle's kernel. Pursue only if a consumer measures the 64-bit path as still-bottlenecked.
- **Energy / power measurement.** Not in the substrate's current measurement vocabulary.

## Methodology lifted from this cycle (provisional, lock in at CLOSEOUT)

- **Magic-multiply for fixed-divisor division: 64-bit intermediate (`vmull + bias + arith-shift`) over 32-bit-with-rounding (`vqrdmulh + vrshl`).** The latter has compound rounding that's hard to reconcile bit-exact across all magnitudes; the former is one rounding step end-to-end. Trade ~1.5× perf for bit-exactness simplicity.

- **Always exhaustively verify the NEON kernel against the scalar reference, not just the emulator.** The emulator is one bridge; the NEON intrinsics are another bridge. Both must match. Exhaustive at table-generation is necessary but not sufficient; runtime exhaustive on the kernel itself closes the loop.

- **Bound the smart-set test count explicitly.** Don't let `step = d/8` become 1 for small d.

- **Sanity-check generator bounds.** N_max and similar derived parameters benefit from `assert` statements that catch off-by-one before the verification phase wastes minutes failing.

## Closeout criteria

CLOSEOUT will be written after all 8 gates pass (or after a deliberate decision to abandon, with reasoning). The CLOSEOUT will include:
- Verdict per gate.
- The lifted methodology, finalized.
- A forward pointer to the cross-exp accumulator cycle if/when it starts.
- An updated CHANGELOG entry.

## Status

This document is pre-committed. Beginning G1 next.
