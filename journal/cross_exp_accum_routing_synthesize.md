# SYNTHESIZE: cross-exp accumulator routing

Pre-committed plan + gates derived from `cross_exp_accum_routing_reflect.md`. Locks in execution shape BEFORE writing kernel code. Applies methodology lessons from shift3 remediation, ternary MAC remediation, V4-residual-3, and the new throughput-microbench-discipline checklist (CONTRIBUTING).

**REWRITE NOTE (2026-05-05):** the first version of this doc had two pre-commitments the user explicitly rejected: (1) "stop the cycle if speedup estimate is <3×" — wrong; function correctness is the goal, speed can be tuned later; (2) "(c) hybrid flag tracking that falls back to scalar for ROUNDED" — wrong; production NEON path must handle all flag work, no scalar fallback in production dispatch. Memory updated with both lessons. Corrections folded below.

## Decision

**Route `m4t_mtfp_vec_accum_aligning`'s align step through the existing shift3 NEON vmlal-magic-multiply pipeline.** No new technique; no new magic constants. Compose existing primitives (shift3-divide + block-add) into a fused accumulator inner loop. **Function correctness is the gate; speed is informational.**

Pre-committed design decisions:

- **Flag-tracking strategy: (b) full reconstruction via NEON.** NEON path computes BOTH SATURATED (via post-add clamp comparison) AND ROUNDED (via `quotient × divisor != original` comparison) entirely in NEON. Per-block compare-and-set; ~5-8 cycles overhead per block. **No scalar fallback in production for any flag-tracking path.**
- **No scalar in production dispatcher.** The CMake configure check already requires aarch64+NEON; the `#if !M4T_HAS_NEON` fallback in production code is non-load-bearing for any actually-shipping target. Production `m4t_mtfp_vec_accum_aligning` becomes NEON-only. The `_scalar_ref` function (added at A-G1) is test-only verification infrastructure — NOT a production fallback. Distinct concept.
- **Kernel shape: fused.** Custom inner loop (decode shift constants once, then 16-cell-block loop doing divide-magic-multiply + add-into-running + clamp + per-block SATURATED + per-block ROUNDED bit reconstruction). Fused over composed because shift3's API is dst-not-accumulate.
- **Reuse the magic table from `m4t/src/m4t_pow3_magic.h`.** No new constants. Validates the table as a substrate-foundational primitive (second consumer).

## Pre-committed gates

Sequential. Each gate's pass condition gates the next.

### A-G1 — Scalar reference exposed first

**Artifact:** add `m4t_mtfp_vec_accum_aligning_scalar_ref` to the public API (`m4t/src/m4t_mtfp.h`). Same semantics as the production function; always uses the existing scalar inner loop, never NEON. Production never calls it; tests use it as the bit-exact oracle.

**Pass:** symbol present in `nm libm4t_test.a`; declared in header.

**Why first (per shift3 remediation lesson):** when productionization replaces the function under test, the bit-exact gate must compare against a separately-preserved scalar oracle, not against itself.

**Budget:** ~15 min.

### A-G2 — Baseline measurement for context (informational only)

**Artifact:** standalone microbench in `m4t/tools/bench_accum_baseline.c` measuring the current scalar `m4t_mtfp_vec_accum_aligning` per-cell ns at a couple of shapes. Records the pre-cycle baseline.

**Pass:** measurement complete and recorded. **No stop-condition based on speedup magnitude** — function correctness is the gate (per user directive); speed is informational. The cycle proceeds regardless of whether the baseline reveals 12× or 2× headroom.

**Budget:** ~20 min.

### A-G3 — Prototype the fused NEON path

**Artifact:** new `static int64_t accum_aligning_div_neon(...)` style helper in `m4t_mtfp.c` implementing the fused pipeline:
- For each 16-cell block:
  - Load running[block] (4 int32x4 vectors)
  - Load addend[block] (4 int32x4 vectors)
  - Apply vmlal-magic-multiply to whichever side needs alignment (running or addend, depending on which exp is larger)
  - Add aligned + un-aligned via vector add + clamp via min/max
  - Compute SATURATED bit per 4-cell chunk via post-clamp comparison (NEON compare + bit pack)
  - Write back to running
- Scalar tail for n not multiple of 16

**Pass:** compiles; standalone callable via test API. Wired into a prototype-only API `m4t_mtfp_vec_accum_aligning_neon` (later removed at A-G7 productionization, but per ternary MAC lesson the prototype helper stays in the .c file as a static function).

**Budget:** ~1 hour.

### A-G4 — Bit-exact verification (with full apparatus from G1)

**Artifact:** new ctest entry `m4t_accum_aligning_neon`. Compares production NEON path against `_scalar_ref` across:
- Curated boundary cases (n=0, n=1, n=15, n=16, n=17, n=63, n=64, n=65, n=4095, n=4096)
- All exponent-delta cases (delta=0 same-exp; delta=1 to 19; delta=20 degenerate)
- All branch shapes (addend>running, running>addend, same)
- Saturation-edge constructed cases (post-add overflow → SATURATED bit set; both paths same output, same SATURATED flag)
- 1000 random (n, delta, seed) configurations

**Pass:** all configurations bit-exact (output AND SATURATED flag bits). ROUNDED bits intentionally NOT compared on the NEON path per (c) flag strategy.

**Budget:** ~1 hour.

### A-G5 — Aliasing test

**Artifact:** verify `running == addend` correctly fires the existing assertion (forbidden by current contract). Plus verify legitimate non-aliased patterns work.

**Pass:** SIGABRT on alias; correct on non-alias. Apply ternary MAC pattern (fork-and-verify-SIGABRT).

**Budget:** ~15 min.

### A-G6 — Disasm verification + multi-shape bench

**Artifact:**
- `otool -tv` confirms the production inner loop emits `smlal.2d` (from shift3's vmlal pipeline) AND `sqadd + smin/smax` (from block_add) AND post-clamp compare for SATURATED.
- Min-of-5 perf bench across 5 shapes spanning different (n, delta) combinations:
  - n=64, delta=1 (typical small-rescale)
  - n=64, delta=10 (mid)
  - n=64, delta=19 (max-non-degenerate)
  - n=4096, delta=5 (large-n)
  - n=16, delta=5 (small-n, per-call overhead bound)
- Apply throughput microbench discipline (CONTRIBUTING checklist: noinline, distinct inputs per iter, etc.).

**Pass:** disasm correct; perf measured with workload-shape declared; report speedup as a range across shapes per CONTRIBUTING scope-match rule.

**Budget:** ~45 min.

### A-G7 — Productionize

**Artifact:** wire the NEON helper into `m4t_mtfp_vec_accum_aligning`'s production dispatcher. **Single NEON path; no scalar fallback in production.**

The production function:
- Same-exp branch: unchanged (already NEON-fast via `vec_add_inplace`).
- Cross-exp branch (addend>running OR running>addend): NEON helper. Computes ALL flag bits (ROUNDED + SATURATED) via NEON compare-and-set per block.
- `flags == NULL` skips the flag-OR write but still does the per-block compare (cheap; the compare result is just discarded). No separate code path for "fast no-flags" — the flag work is already cheap enough on the NEON pipeline.
- Tail loop for n < 16: scalar (no NEON path possible for sub-block n; this is implementation detail, not a "scalar fallback" — the tail handles cells the NEON path geometrically can't touch).
- Degenerate-delta branches (delta >= 20): unchanged, branch out before the NEON path.

The `_scalar_ref` (A-G1) remains in the public API as test-only verification infrastructure. It's NOT a production code path.

**Pass:** 19/19 ctest still PASS. Bit-exact verification (A-G4) re-runs against scalar reference and still PASSes (both flag bits match). Same-exp branch unchanged. NEON-only production code; the only scalar paths are: (1) the test oracle `_scalar_ref`, and (2) the n<16 tail (geometrically necessary).

**Budget:** ~30 min.

### A-G8 — No regression in production binaries

**Artifact:** smoke-test `bench_m4t_tier2_perf`, `gesh_confidence_probe`, `gesh_expr_routing_probe` before/after — outputs identical.

**Pass:** outputs match.

**Budget:** ~10 min.

## Order of execution

A-G1 → A-G2 → A-G3 → A-G4 → A-G5 → A-G6 → A-G7 → A-G8.

A-G2 is informational only — does NOT gate the cycle. Function correctness is the goal.

If any correctness gate (A-G4, A-G5) fails, stop and diagnose. Don't push past correctness failures.

## Risk register

- **R1 (A-G4 bit-exact fails):** NEON pipeline produces different output OR different flag bits than scalar reference. Action: debug to root cause. NO scalar fallback as escape hatch — the NEON path must be made bit-exact.
- **R2 (A-G4 ROUNDED reconstruction fails):** the `quotient × divisor != original` check has an edge case (overflow? signed-vs-unsigned ambiguity?) that produces wrong ROUNDED bits. Action: debug; revisit the reconstruction approach (e.g., compare `quotient * divisor + remainder == original` invariant differently). Still no scalar fallback.
- **R3 (A-G3 fused vs composed turns out wrong):** fused proves harder than composed. Action: fall back to composed (two-pass: divide-into-scratch then add). Both are NEON paths; this is structure choice, not scalar-vs-NEON.

## What's NOT in this cycle

- **Same-exp branch optimization.** Already NEON-fast via `vec_add_inplace` (T2-C). Untouched.
- **`vec_add_aligning` / `vec_sub_aligning` wrappers** (per N32 from NODES). They delegate to the accumulator; the accumulator's productionization is sufficient.
- **New magic constants.** Reuse `m4t_pow3_magic.h`.
- **Flag tracking via NEON for ROUNDED.** Documented as dropped-on-NEON-fast-path; consumers who need it call `_scalar_ref`.

## Pre-committed methodology applications

- **Throughput microbench discipline (CONTRIBUTING):** apply the 7-point checklist to A-G6.
- **Scope-match rule:** report perf as a range across 5 shapes (A-G6).
- **shift3 remediation lesson:** `_scalar_ref` exposed at A-G1 (BEFORE prototype).
- **ternary MAC remediation lesson:** bigger sample (curated + 1000 random + saturation-edge) at A-G4 from the start, not as remediation.
- **No consumer-demand gating.** The user's framing of "per-block exponent management is software doing hardware's work" IS the directive.

## Done when

A-G1 through A-G8 PASS. CLOSEOUT records per-gate verdict, the actual speedup-range measured, the flag-tracking decision, and the second-consumer-of-magic-table validation.

## Status

Pre-committed. Beginning A-G1 next.
