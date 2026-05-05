# SYNTHESIZE: register-tile libm4t matmul kernels

Pre-committed plan + gates derived from `m4t_matmul_tile_raw.md`.

## Decision

**Retile both libm4t matmul kernels (`m4t_mtfp_ternary_matmul_bt` and `m4t_mtfp4_sdot_matmul_bt`) to use 4 parallel accumulator chains per outer iteration. N%4 tail handled by the existing untiled NEON path. No API changes; no spec changes; same bit-exact output.**

This brings the audit's apples-to-apples ~1.8× wall-clock win into production.

## Pre-committed gates

### G1 — Bit-exact preservation

ALL existing ctest binaries must pass without modification:
- `test_m4t_ternary_matmul` (9 tests, includes K=1M long-K + partial-block + reserved-trit-code)
- `test_m4t_ternary_matmul_neon` (NEON-vs-scalar bit-exact verification)
- `test_m4t_mtfp4` (12 tests, includes 10k-sample narrow + K=1M long-K)

Also: 20/20 ctest must pass (no collateral regressions).

This is a HARD gate. Any test failure blocks the cycle.

### G2 — N%4 tail correctness

Add a test case (or extend existing) that exercises N=1, 2, 3, 5, 6, 7, 9, 10, 11 (not multiples of 4 below the body and after-body cases). Verify bit-exact against scalar reference.

This catches off-by-one in the tail logic.

### G3 — Wall-clock improvement

Micro-bench in `m4t/tests/bench_m4t_tier2_perf.c` (or new `bench_m4t_matmul_tile.c`) measures wall-clock at K=12800, N=64 for both retiled kernels.

Pre-committed expectation:
- ≥1.4× wall-clock improvement (lower bound; audit showed 1.8×, allow margin for vmlal vs SDOT differences and libm4t's slightly different inner-loop shape).

If <1.4×: not a hard fail (could be measurement noise on this hardware), but warrants investigation.

If ≥1.0×: tile didn't help. Hard fail; back out.

### G4 — Disasm verification

`otool -tv` on the retiled kernel functions shows:
- 4 distinct accumulator registers materialized (e.g., v4, v5, v6, v7 for SDOT).
- 4 SDOT/vmlal chains in the inner loop (one per j cell).
- Single X load per outer iter (shared).

If disasm shows compiler collapsed the tile (unlikely with `__attribute__((noinline))` and explicit unroll), back out and refactor.

### G5 — Aliasing assertions preserved

Tiled kernels keep:
- `assert(Y != X)`
- `assert(Y != W_packed)`

### G6 — No scalar fallback

Tiled kernels:
- Have NEON-only main body (4 j cells per outer iter).
- N%4 tail uses the existing untiled NEON inner loop (one j cell at a time).
- Existing K%16 geometric scalar tail in `m4t_mtfp4_sdot_matmul_bt` is preserved (the rule allows geometric scalar tails for sub-block n).
- No `#if !M4T_HAS_NEON ... #else scalar ... #endif` patterns introduced.

## Implementation plan

1. **Retile `m4t_mtfp4_sdot_matmul_bt` first.** Simpler (single SDOT per inner block; tile straightforwardly maps to 4 SDOTs).
2. **Retile `m4t_mtfp_ternary_matmul_bt` second.** More complex (vmlal_s32 pipeline with 8 calls per inner block; tile = 32 vmlal calls into 8 acc registers). Higher register pressure.
3. **Add edge-case tests for N%4 tail** (G2).
4. **Add micro-bench** (G3).
5. **Run all ctest** (G1).
6. **Inspect disasm** (G4).
7. **Red-team.** Per LMM discipline.
8. **Address red-team findings.**
9. **Commit + push.**

## Risk register

- **R1 (HIGH):** Flag-tracking mis-wired in tile body. SATURATED bits could be set for wrong j cell. Mitigation: per-j-cell unrolled flag check, mirror exactly the untiled logic for each of the 4 cells.
- **R2 (MEDIUM):** Compiler register pressure with 8 acc registers + W decode + X + LUT in m4t_mtfp_ternary_matmul_bt. Could spill. Mitigation: disasm verify; if spills appear, reduce LUT register footprint or use vqtbl2q if applicable.
- **R3 (LOW):** Performance regression at very small N (N≤4). Tile body might not even run; only tail. Equal-or-worse than untiled. Mitigation: include N=8 (well above tile size) in micro-bench to confirm tile actually fires.
- **R4 (LOW):** Aliasing assertion order. Existing kernels assert before doing work. Tiled kernels keep same order.

## Done when

- G1 (bit-exact tests): PASS.
- G2 (N%4 tail tests): PASS.
- G3 (wall-clock): ≥1.4× improvement measured.
- G4 (disasm): shape verified.
- G5, G6 (project rule compliance): PASS.
- Red-team applied; findings remediated.
- Commit + push; CI green.

## Status

Pre-committed. Beginning implementation next.
