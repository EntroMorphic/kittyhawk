# `audit/` — substrate-claim measurement tools

Science deliverables. NOT in `ctest` — these are measurement tools producing journal evidence, not tests.

## What's here

| File | Purpose |
|---|---|
| `tristate_audit.c` | Two-gate audit of third-state utilization across substrate layers L1-L4 + L6 on a 2-layer ternary GEMM workload. Per `journal/tristate_op_synthesize.md`. |
| `b2b_matmul.{h,c}` | NEON kernels for the strong-claim test: Path A (base-3 4-in-8 packed), Path B (B2-B sign+mask honest), Path B-skip (B2-B with all-masked-block skip), Path C (B2-B optimal unified-LUT), Path D (base-3 5-in-8 packed). All NEON-only; no scalar reference. |
| `tristate_strong_bench.c` | Strong-claim bench harness. Runs all 5 audit kernels + substrate's `m4t_ternary_dot_matmul_bt` for external grounding; verifies bit-exact equivalence; measures op count + wall-clock across L1-resident, L2-resident, and DRAM-bound regimes. |
| `results.csv`, `results_summary.txt` | Tristate audit per-seed CSV + per-config summary (last run; regenerate by re-running `tristate_audit`). |
| `strong_results.csv`, `strong_summary.txt` | Strong-claim bench per-seed CSV + per-config summary. |

## Running

```bash
cmake --build build --target tristate_audit tristate_strong_bench

# Audit (third-state utilization across substrate layers)
./build/audit/tristate_audit > audit/results.csv 2> audit/results_summary.txt

# Strong-claim bench (Path A vs B vs B-skip vs C vs D vs substrate)
./build/audit/tristate_strong_bench > audit/strong_results.csv 2> audit/strong_summary.txt
```

`results_summary.txt` and `strong_summary.txt` go to stderr (per-config summaries); CSV per-seed measurements go to stdout.

## Disassembly

Verifying NEON op count per inner block:

```bash
otool -tv build/audit/tristate_strong_bench | grep -A 50 "_base3_packed_matmul_neon:"
otool -tv build/audit/tristate_strong_bench | grep -A 50 "_base3_5in8_matmul_neon:"
otool -tv build/audit/tristate_strong_bench | grep -A 50 "_b2b_optimal_matmul_neon:"
```

All kernel functions are `__attribute__((noinline))` so the inner-loop block remains a discrete disassembly unit.

## Verdict (current state)

Cumulative across the audit + strong-claim cycle (post P0-1 + P0-2 + P0-3 with apples-to-apples tiling):

**Audit (intra-substrate utilization, weak claim):**
- L1, L2, L6 LOAD-BEARING per both gates (info-theoretic + algorithmic).
- L3 MIXED — sparsity-dominated; sink by entropy in sparse regimes but algorithmically load-bearing.
- L4 MIXED — least load-bearing of measured layers (mean cos ≈ 0.94), but not invisible.
- L5 DEFERRED — cross-exp accum not exercised by GEMM-only workload.

**Strong claim (comparative, base-3 vs B2-B):**
- At fixed 2 bits/cell density: encoding labels are aliases — Path A ≡ Path C in op shape (verified by disassembly).
- At sub-2-bit density: base-3 wins. **Path D (5-in-8 packing, 1.6 bits/cell) is ~1.8× faster than Path A AND Path C across all tested regimes** (L1-resident through DRAM-bound) when all are register-tiled. B2-B is structurally floored at 2 bits/cell (sign+mask are independent).

## Why these tools live in `audit/` not `m4t/`

The substrate (`libm4t`) is base-3 only by project rule — no base-2 alternatives, no comparison kernels. The strong-claim test requires a base-2 reference implementation (Path B / Path C). To honor the substrate's rule, the comparison kernels live in `audit/`. They are not part of the project's production surface; they exist solely for measurement.

## Cycle cross-references

| Journal cycle | What it produced |
|---|---|
| `journal/tristate_op_*` | Audit cycle (RAW → SYNTHESIZE → CLOSEOUT, with red-team) |
| `journal/tristate_strong_*` | Strong-claim cycle (RAW → SYNTHESIZE → CLOSEOUT, with multi-round red-team) |
| `journal/tristate_strong_5in8_addendum.md` | Sub-2-bit packing (Path D) addendum |
| `journal/tristate_strong_membw_*` | Memory-bandwidth regime addendum + red-team |
| `journal/p0_kernel_opt_redteam.md` | P0-1, P0-2, P0-3 kernel optimizations (pre-permute X + split-LUT decode + register tile) |

## Project rule compliance

- All audit kernels are NEON-only. No scalar fallback. No `_scalar_ref` test oracles (verification is NEON-vs-NEON cross-check + external grounding via substrate's `m4t_ternary_dot_matmul_bt`).
- `K%80==0` enforced via assert for Path D (5-in-8 alignment); `K%16==0` for others.
- `N%4==0` enforced for register-tiled paths (A, C, D).
- All five kernels produce bit-exact identical Y; verification gate is `memcmp` per run.
