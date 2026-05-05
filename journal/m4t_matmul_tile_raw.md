# RAW: register-tile libm4t matmul kernels

Stream of consciousness on bringing the audit's tile-by-4 win into libm4t.

## What's in production today

Two matmul kernels in libm4t:

1. **`m4t_mtfp_ternary_matmul_bt`** — MTFP19 X (int32) × packed-trit W → MTFP19 Y, plus optional SATURATED flag tracking. Inner kernel via `ternary_dot` which dispatches to `ternary_dot_vmlal` (NEON) or `ternary_dot_scalar` (test oracle only).
   - Outer: `for i in 0..M, for j in 0..N`.
   - Inner: single-acc int64 chain through `ternary_dot_vmlal`'s 8 vmlal_s32 calls per 16 trits.

2. **`m4t_mtfp4_sdot_matmul_bt`** — MTFP4 (int8) X × ternary W → MTFP19 Y. Inner uses SDOT directly.
   - Outer: same i-j structure.
   - Inner: 1 vdotq_s32 per 16-trit block, single-acc int32 chain. Plus geometric scalar tail for K%16.
   - Used by `m4t_ternary_dot_matmul_bt` (the ternary-input wrapper).

Both kernels: single-accumulator chain per output cell. The audit demonstrated this leaves 60-80% of SDOT throughput on the table (Path A measured 0.46 SDOTs/cycle vs T2 ceiling 1.52).

## What changes with tile-by-4

Restructure inner loop:
- Compute 4 j-cells per outer iteration.
- 4 parallel accumulator chains (independent, allowing pipelining).
- Shared X load across the 4 j cells.

Per the audit's apples-to-apples comparison: ~1.8× wall-clock gain for free, no spec change, no API change.

## Open design questions

**Q1: How to handle N % 4 ≠ 0?**
Three options:
- (a) Assert N % 4 == 0. Breaks existing API; existing callers might pass arbitrary N.
- (b) Geometric tail: tile body for first N - N%4 cells, untiled NEON path for last 1-3 cells.
- (c) Pad output buffer (no — caller-allocated).

(b) is the right answer per project pattern (geometric tail OK, "fall back to scalar" not OK; the tail here is NEON, not scalar).

**Q2: How to handle K%16 ≠ 0?**
Already handled in m4t_mtfp4_sdot_matmul_bt via geometric scalar tail. Per memory rule, scalar geometric tail is OK. m4t_mtfp_ternary_matmul_bt also has scalar tail in ternary_dot_vmlal.
Tile-by-4 should preserve these existing tail patterns per-cell.

**Q3: Flag tracking for SATURATED in m4t_mtfp_ternary_matmul_bt?**
Currently checks `flags && acc != (int64_t)out` per output cell. With tile-by-4, need to do this for each of the 4 j cells in the tile body. Straightforward but easy to mis-wire.

**Q4: Aliasing assertions?**
Existing kernels assert `Y != X` and `Y != W`. Preserve.

**Q5: API change?**
None. Function signatures stay the same. Tiling is internal restructuring.

**Q6: Per audit, does tile-by-4 work for the vmlal_s32 pipeline (Path A's pattern from audit/, used by m4t_mtfp_ternary_matmul_bt)?**
Yes — the audit's `base3_packed_matmul_neon` was Path A using vdotq_s32. The tile pattern is the same: 4 acc chains, 4 W loads per outer iter, 1 X load shared.
For vmlal_s32 in libm4t, the structure is similar but uses int64 acc and 8 vmlal calls per 16 trits per j cell. Tile-by-4 means 32 vmlal calls per outer iter (4 j cells × 8 vmlal) into 8 acc registers (4 j cells × 2 acc each, since vmlal is split across acc0/acc1).

That's a lot of register pressure (8 acc registers) but should fit in NEON's 32 V regs comfortably with W decode + X loads + LUT.

**Q7: Performance prediction?**
Audit showed Path A wall-clock at K=12800/N=64 went from ~50ms (untiled) to ~30ms (tiled), a 1.7× gain.
For libm4t's m4t_mtfp4_sdot_matmul_bt, similar shape. For m4t_mtfp_ternary_matmul_bt (vmlal-based), gain might be similar or slightly less due to deeper acc chain.

**Q8: Verification?**
- Existing ctest tests must pass bit-exact (no regression in correctness).
- A microbench in m4t/tests/ (or m4t/tools/) demonstrates the wall-clock improvement.
- Disasm verifies tile shape (4 SDOT/vmlal chains, 1 X load shared).

## Risk register

- **R1:** flag-tracking logic mis-wired in tile body → SATURATED bits set wrong per j cell. Mitigation: per-cell unrolled flag check, mirroring untiled logic.
- **R2:** N=1, 2, 3 edge cases (only tail runs, no tiled body). Tail is the original kernel pattern; should work but exercise it in tests.
- **R3:** Compiler register pressure with 8 acc + 4 W + X + decode → spills. Mitigation: inspect disasm post-build, look for spills.
- **R4:** Performance regression at small N (overhead of tile setup > benefit). Mitigation: micro-bench at small N (e.g., N=4) confirms either gain or no-regression.
- **R5:** Flags=NULL vs flags!=NULL paths differ in tile body. Both must be NEON-only per project rule. Avoid scalar fallback in either path.

## What this cycle is NOT

- Not a new packing format (that's Item 2).
- Not a new API (no signature changes).
- Not a substrate spec change.
- Not the SDOT throughput tool (Item 3).

## Where I'd land

Tile both `m4t_mtfp_ternary_matmul_bt` and `m4t_mtfp4_sdot_matmul_bt`. N%4 tail handled with the existing untiled inner loop (NEON, no scalar fallback). Existing ctest tests gate correctness. New micro-bench in m4t/tests/ gates the wall-clock improvement (informational, not a CI gate).

Two kernels to retile, one cycle. Should fit in a focused effort.

## Open question I'd want to settle

Should I also look at whether `m4t_ternary_dot_matmul_bt` (the wrapper) needs any change? It just delegates to `m4t_mtfp4_sdot_matmul_bt`, so tiling that delegated kernel is sufficient. Wrapper unchanged.

Same for any other consumers. Spot-check `gesh/` for matmul callers; they shouldn't need changes (API stable).
