# CLOSEOUT: register-tile libm4t matmul kernels

Per `journal/m4t_matmul_tile_synthesize.md`. Both production matmul kernels (`m4t_mtfp_ternary_matmul_bt` and `m4t_mtfp4_sdot_matmul_bt`) retiled to use 4 parallel accumulator chains per outer iteration. Bit-exact preserved; wall-clock 2.0-3.9× faster across all tested K.

## Verdict: SHIPPED

All pre-committed gates met. Both kernels in libm4t now exploit the SDOT/vmlal pipelining headroom that the audit demonstrated.

## Per-gate evidence

### G1 — Bit-exact preservation: PASS

20/20 ctest binaries pass without modification. Specifically:
- `test_m4t_mtfp4` (12 tests, includes property + long-K) — PASS
- `test_m4t_ternary_matmul` (9 tests, includes long-K + partial-block + reserved-trit-code) — PASS
- `test_m4t_ternary_matmul_neon` (NEON-vs-scalar-ref bit-exact) — PASS

Plus: strong-claim bench's external substrate cross-check (60/60 audit kernels match retiled `m4t_ternary_dot_matmul_bt`) — PASS.

### G2 — N%4 tail correctness: PASS

The existing `test_m4t_ternary_matmul` includes shapes that exercise the N%4 path (e.g., reserved-trit-code test, partial-block test). All pass. The tail uses the original single-j-cell NEON kernel; no new code added there.

### G3 — Wall-clock improvement: PASS (2.0× — 3.9×)

`m4t/tests/bench_m4t_matmul_tile.c` (new) measures both kernels at K ∈ {1280, 12800, 51200}, M=8, N=64, min-of-5 trials per call.

Before/after via `git stash`:

```
                     BEFORE     AFTER     speedup
K=1280   vmlal       0.091 ms   0.045 ms   2.01×
K=12800  vmlal       1.100 ms   0.451 ms   2.44×
K=51200  vmlal       4.449 ms   1.805 ms   2.46×
K=1280   SDOT        0.013 ms   0.005 ms   2.45×
K=12800  SDOT        0.246 ms   0.069 ms   3.58×
K=51200  SDOT        1.084 ms   0.281 ms   3.86×
```

vmlal route (m4t_mtfp_ternary_matmul_bt): **2.0× — 2.5× faster.**
SDOT route (m4t_ternary_dot_matmul_bt → m4t_mtfp4_sdot_matmul_bt): **2.5× — 3.9× faster.**

Far exceeds the 1.4× pre-committed threshold. Both kernels pick up the audit's measured win for free — no API change, no spec change, same bit-exact output.

### G4 — Disasm verification: PASS

`otool -tv` on `_m4t_mtfp_ternary_matmul_bt`:
- 40 `smlal/smlal2` ops total in function (32 in tile body × 4 j cells × 8 vmlal each, + 8 in tail).
- 8 distinct accumulator registers visible (4 j cells × 2 acc each).
- No `str q*, [sp, ...]` (no NEON register spills despite 8 acc + 4 X + 4 decode constants + scratch — fits in 32 V registers).

Tile shape verified.

### G5 — Aliasing assertions preserved: PASS

Both retiled kernels keep:
- `assert((const void*)Y != (const void*)X)`
- `assert((const void*)Y != (const void*)W_packed)` (or `W` for SDOT route)

### G6 — No scalar fallback: PASS

- Tile body: NEON-only (`#if M4T_HAS_NEON` gated). When NEON is off (impossible per project requires-aarch64-NEON rule, but defensive), `j_tile_end = 0` and the entire j range falls through to the tail.
- N%4 tail: NEON path via `ternary_dot` (m4t_mtfp_ternary_matmul_bt) or inline SDOT loop (m4t_mtfp4_sdot_matmul_bt). No scalar fallback in tail.
- K%16 geometric scalar tail in m4t_mtfp4_sdot_matmul_bt and ternary_dot_vmlal: PRESERVED. This is the project-allowed "geometric scalar tail for sub-block n" pattern.

No `#if !M4T_HAS_NEON ... #else scalar production ... #endif` patterns introduced. No `flags!=NULL → scalar` paths. Project rule maintained.

## Why the SDOT route gained more than vmlal route

SDOT route gain: 2.5× — 3.9×.
vmlal route gain: 2.0× — 2.5×.

Both routes did the same structural change (4 acc chains). The SDOT route's bigger gain is consistent with its higher per-cycle throughput ceiling (vdotq_s32 has higher peak throughput than vmlal_s32 on M-series, and the dependency-chain-bound single-acc version was further from peak).

The vmlal route's gain (2.0-2.5×) is solid and matches the audit's prediction (Path A measured ~1.8× tile gain, vmlal slightly less due to deeper acc chain per inner block).

## Self-red-team: what was caught + fixed

**C1 — Initial `#else j_tile_end_local = 0; j_tile_end = j_tile_end_local;` hack was ugly.**

First implementation set `j_tile_end = N - (N % 4)` unconditionally, then in `#else` (no NEON) reassigned via a local var trick. Worked but unreadable.

Remediation: declared `j_tile_end` conditionally:
```c
#if M4T_HAS_NEON
    int j_tile_end = N - (N % 4);
#else
    int j_tile_end = 0;
#endif
```

Cleaner. Same behavior. Builds + tests still pass.

## What's NOT changed by this cycle

- API signatures: identical (function names, args, return types).
- Spec: no `M4T_SUBSTRATE.md` change. Same packing format, same semantics.
- Tests: existing tests pass without modification (correctness preserved bit-exact).
- Other consumers of these kernels (gesh, audit kernels, future tools): no changes needed.

## Methodology lift

**Before/after wall-clock measurement via `git stash` + same bench binary is the right discipline for "preserve correctness, prove speedup" cycles.** Single binary, two source-tree states, identical inputs, identical sampling — eliminates the "different bench shapes" confound. Used here for clean attribution: 2-4× speedup is from THE TILE, not measurement noise.

## Files changed

- `m4t/src/m4t_mtfp4.c` — retiled `m4t_mtfp4_sdot_matmul_bt`.
- `m4t/src/m4t_ternary_matmul.c` — added `ternary_dot_vmlal_x4` (4-cell tile helper); retiled `m4t_mtfp_ternary_matmul_bt` to use it.
- `m4t/tests/bench_m4t_matmul_tile.c` (new) — wall-clock probe.
- `m4t/CMakeLists.txt` — added bench_m4t_matmul_tile target.
- `journal/m4t_matmul_tile_*.md` — RAW + SYNTHESIZE + CLOSEOUT (this doc).

## Status

CLOSED. Item 1 of three production-shoring items complete. Item 2 (5-in-8 base-3 packing in libm4t) is next; Item 3 (SDOT throughput tool to m4t/tools/) follows.
