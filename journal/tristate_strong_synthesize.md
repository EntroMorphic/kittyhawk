# SYNTHESIZE: strong-claim test setup

Pre-committed plan + gates derived from `tristate_strong_reflect.md`.

## Decision

**Build a B2-B (sign + sparsity bit) reference matmul kernel as comparison to the substrate's base-3 ternary matmul. Run side-by-side on the audit's workload. Measure density (bits/cell), precision (bit-exact equivalence), and kernel cost (NEON op count). Pre-commit the headline regime, kernel structures, and verdict thresholds.**

This is a build cycle (write the B2-B kernel) + science cycle (run the comparison). Output: `journal/tristate_strong_closeout.md` with verdict, plus `audit/b2b_matmul.{h,c}` reference kernel.

## The question being operationalized

**For L1 (weight third-state) on the audit's workload, does the substrate's packed-ternary representation + ternary matmul kernel outperform a B2-B (sign + sparsity bit) representation + matmul kernel on at least one of {density, precision, kernel cost} without losing on the others?**

If yes → STRONG CLAIM SUPPORTED on L1.
If tie everywhere → STRONG CLAIM NOT SUPPORTED on L1 (substrate has no distinctive advantage).
If loss on any axis without compensating wins → STRONG CLAIM FALSIFIED on L1.
If regime-dependent → REPORTED HONESTLY; partial support.

## B2-B kernel design (pre-committed)

**Storage layout:**
- 2 bits per cell: 1 sign bit + 1 mask bit. Sign=0 → +1, sign=1 → -1. Mask=0 → present, mask=1 → masked (encodes 0).
- Pack 4 cells per byte: bits 0-1 cell 0, bits 2-3 cell 1, bits 4-5 cell 2, bits 6-7 cell 3.
- For an [N, K] weight matrix, packed bytes = (K+3)/4 per row.

**Encoding semantics:**
- (mask=0, sign=0) → +1
- (mask=0, sign=1) → -1
- (mask=1, sign=*) → 0

Equivalent functional range to ternary {-1, 0, +1}. Same density (2 bits/cell) as substrate's ternary packing.

**Two NEON kernel variants:**

**B2-B-uniform:** processes all cells regardless of mask. Inner loop:
1. Load 16-bit packed row (4 bytes, 16 cells).
2. Decode signs into int8 vector via TBL.
3. Decode masks into int8 vector via TBL.
4. Multiply signs × activations (signed int8 × int8).
5. Apply mask via vbslq_s8 (where mask=1, zero out the contribution).
6. Widen + accumulate into int32 via vmlal_s32 or similar.

Comparable shape to the substrate's `ternary_dot_vmlal` kernel. The mask application is the extra cost vs SDOT.

**B2-B-skip:** loads the mask byte first; if all 16 cells are masked, skips the entire block; else proceeds as B2-B-uniform.

This is the "skip-aware" alternative that could potentially win in highly sparse regimes.

**Scalar reference** (`b2b_matmul_scalar_ref`): test oracle for bit-exact verification. Per project pattern (substrate has `_scalar_ref` for shift3, ternary_matmul, accum_aligning).

**Production rule:** B2-B kernel is a REFERENCE for comparison, not a substrate primitive. Lives in `audit/`, not `m4t/`. NEON path is required (per project rule "no scalar in production"); scalar reference is for verification.

## Headline regime (pre-committed)

**K=256, w_zero=0.60, a_zero=0.60** — BitNet-typical. Verdict gates apply primarily to this regime. Other configs reported as context.

This regime is where base-2 alternatives are most likely to be competitive (high sparsity → mask-overhead amortizable). If base-3 wins HERE, the strong claim is robustly supported. If base-3 loses HERE but wins in dense regimes, the verdict is regime-dependent.

## Comparison axes (pre-committed gates)

### Axis 1 — Density

Measurement: bits per cell at storage.
- Base-3 substrate: 2 bits/cell (current packing).
- B2-B: 2 bits/cell.
- Sub-gate: SUPPORT_BASE3 if base-3 strictly tighter; PARITY if equal; SUPPORT_B2 if B2-B tighter.
- **Expected: PARITY (tie).** Wins for base-3 only via sub-investigation of theoretically-optimal packing (out of scope this cycle).

### Axis 2 — Precision (algorithmic equivalence)

Measurement: bit-exact equality of Y output across all configs × seeds.

Pre-committed thresholds:
- PASS (both functionally equivalent): identical Y across all 60 runs.
- FAIL: any difference in Y output.

If B2-B and base-3 produce identical outputs for the same inputs (which they should by construction since both encode {-1, 0, +1}), this axis is a TIE. The cycle PROCEEDS to axis 3. If they differ, that's a cycle-blocking finding requiring investigation.

### Axis 3 — Kernel cost (NEON op count)

Measurement: NEON instructions per 16-cell inner block, via `objdump -d` on each kernel function.

Pre-committed thresholds (headline regime only):
- SUPPORT_BASE3: base-3 op count < min(B2-B-uniform, B2-B-skip) for headline regime.
- PARITY: base-3 op count == min(B2-B-uniform, B2-B-skip) ± 2 ops.
- SUPPORT_B2: base-3 op count > min(B2-B-uniform, B2-B-skip) by ≥3 ops.

The "min over both B2-B variants" is the steel-manning move: pick the BEST B2-B variant for the regime, compare base-3 against that.

Per-regime breakdown reported alongside headline.

### Cumulative verdict logic

- AXIS 1 PARITY + AXIS 2 PASS + AXIS 3 SUPPORT_BASE3 → **STRONG CLAIM SUPPORTED on L1.**
- AXIS 1 PARITY + AXIS 2 PASS + AXIS 3 PARITY → **STRONG CLAIM NOT SUPPORTED on L1 (no distinctive advantage).**
- AXIS 1 PARITY + AXIS 2 PASS + AXIS 3 SUPPORT_B2 → **STRONG CLAIM FALSIFIED on L1 (base-2 outperforms).**
- AXIS 2 FAIL → **CYCLE BLOCKED** pending investigation.
- ANY AXIS surface-level support but regime-dependent → **STRONG CLAIM PARTIALLY SUPPORTED**, document regime-by-regime.

## Disassembly methodology (pre-committed)

For each kernel function:
1. Build with `-O3 -flto` (project default).
2. Run `objdump -d build/audit/tristate_strong_bench` (or wherever the binary lives).
3. Locate the inner-loop block — for the substrate's ternary matmul, this is the per-16-trit block in `ternary_dot_vmlal`. For B2-B, this is the per-16-cell block in the B2-B kernel.
4. Count NEON instructions (any instruction with v-prefix register operand, or known NEON mnemonics like `ld1`, `tbl`, `mul`, `mla`, `sdot`, `vbsl`, etc.).
5. Report op count per 16-cell inner block.

If LTO inlines the kernel into the caller, use `__attribute__((noinline))` on the kernel functions to keep them disassemblable as discrete units.

## Order of execution

1. **Build B2-B scalar reference** (`b2b_matmul_scalar_ref`) — test oracle.
2. **Build B2-B NEON uniform** kernel + bit-exact verification against scalar ref.
3. **Build B2-B NEON skip-aware** kernel + bit-exact verification.
4. **Run cross-check on audit workload:** assert Y_base3 == Y_B2B_uniform == Y_B2B_skip across all 12 configs × 5 seeds.
5. **Disassemble** all three kernels; extract NEON op counts per inner block.
6. **Wall-clock benchmark** (informational, not gating) on the audit workload.
7. **CLOSEOUT:** axis-by-axis verdict + per-regime breakdown + methodology lift + forward pointer to L2/L6 follow-on cycles if warranted.

## Risk register

- **R1 (B2-B kernel verification):** if B2-B doesn't produce bit-exact Y match against base-3, the cycle is blocked. Mitigation: scalar reference first, NEON cross-check via byte-by-byte comparison.
- **R2 (LTO inlines kernel beyond recognition):** disassembly of the kernel functions may not show clean inner-loop boundaries. Mitigation: `__attribute__((noinline))` on the kernel functions for measurement; relax for benchmarking.
- **R3 (compiler chooses different inner-loop unrolling for the two kernels):** unfair comparison if base-3 unrolls 4× and B2-B unrolls 2×. Mitigation: inspect disassembly, normalize op counts to per-16-cell-block.
- **R4 (verdict is "tie everywhere"):** the substrate's distinctive advantage on L1 may not exist. Honest finding; does not invalidate the substrate but does redirect attention to other layers (L2, L6) or other operationalizations of claim 3.
- **R5 (op count favors base-3 but wall-clock favors B2-B):** would suggest op count is misleading on this hardware. Honest finding; report both, gate on op count per pre-commit, note wall-clock divergence.

## What this cycle is NOT

- Not testing strong claim broadly. L1 only; L2 / L6 cycles deferred.
- Not modifying production code. B2-B reference lives in `audit/`, not in libm4t.
- Not a perf cycle. Op count is the gate; throughput is informational.
- Not validating the substrate's overall design. The verdict is L1-specific; substrate-broad implications discussed only honestly in CLOSEOUT.

## Done when

B2-B scalar + uniform + skip-aware NEON kernels all built and bit-exact verified. Disassembly complete; op counts tabulated. Cross-check on audit workload PASS. CLOSEOUT records:
- Per-axis verdict (density, precision, kernel cost) with numerical evidence
- Headline-regime verdict + per-regime context
- Methodology lifted (kernel-comparison patterns, disassembly methodology)
- Forward pointer to L2/L6 follow-on if warranted

## Status

Pre-committed. Awaiting user gate before execution.

The execution would be: ~200-400 lines of new C (`audit/b2b_matmul.h`, `audit/b2b_matmul.c`, integration into `tristate_audit.c` or a new `tristate_strong_bench.c`), plus build-system changes, plus disassembly + measurement.

## File layout (pre-committed)

```
audit/
  b2b_matmul.h            — public API for B2-B kernels
  b2b_matmul.c            — implementations: scalar ref, NEON uniform, NEON skip-aware
  tristate_strong_bench.c — strong-claim measurement harness (separate binary from tristate_audit)
  CMakeLists.txt          — extended to build new targets
journal/
  tristate_strong_*.md    — LMM cycle docs (this file + RAW + NODES + REFLECT + CLOSEOUT)
```

The B2-B kernel does NOT enter `m4t/src/` — substrate is base-3 only per project rule.
