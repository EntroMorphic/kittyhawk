# M4T — M4 Ternary Extensions

A generic, cache-resident ternary compute substrate for aarch64 + NEON.

## What M4T is

M4T is a library of ternary and multi-trit fixed-point (MTFP) primitives that behave like silicon extensions. From the caller's perspective, each operation is always available, always fast, always cheap, with predictable latency. The implementation is software; the abstraction is hardware.

## Numerical system

M4T operates on four cell widths, all rooted in balanced-ternary fixed point (real value = cell / scale):

| Type | Storage | Trits | Scale | Range | Resolution | NEON lanes |
|---|---|---|---|---|---|---|
| `m4t_mtfp4_t` | int8 | 4 | 3² = 9 | ±4.4 | 0.111 | 16 |
| `m4t_mtfp9_t` | int16 | 9 | 3⁵ = 243 | ±40.5 | 0.0041 | 8 |
| `m4t_mtfp_t` | int32 | 19 | 3¹⁰ = 59049 | ±9842 | 1.69e-5 | 4 |
| `m4t_mtfp_w_t` | int64 | 39 | 3¹⁰ = 59049 | ±3.43e13 | 1.69e-5 | 2 |

No binary floating point. No binary quantization. Integer containers hold ternary fixed-point cells at clean power-of-3 boundaries.

## What's implemented (v0)

### MTFP19 arithmetic (`m4t_mtfp.h`)
Saturating add/sub/mul, mul-by-trit, vector add (NEON with min/max clamp), vector scale, dense MTFP×MTFP matmul, MTFP×packed-trit matmul, bias add, fan-in normalization, LayerNorm (integer isqrt, forward only).

### Trit packing (`m4t_trit_pack.h`)
Pack/unpack between `m4t_trit_t` buffers and 2-bit packed `uint8_t` containers. Popcount routing distance (XOR+VCNT). Decode LUT for `vqtbl1q_s8`.

### Trit operations (`m4t_trit_ops.h`)
Six element-wise ops on packed-trit buffers via TBL lookup: `mul`, `sat_add`, `max`, `min`, `eq`, `neg`. ~28 NEON instructions per 64 trits (binary ops) or ~5 instructions (neg, via bit-swap).

### Trit reducers (`m4t_trit_reducers.h`)
Collapse a packed-trit vector to scalar counts via masked VCNT: `signed_sum` (count(+1) − count(−1)), `sparsity` (count nonzero), `counts` (separate pos/neg). ~14 NEON instructions per 64 trits. Building blocks for weight-derived signature computation.

### Routing primitives (`m4t_route.h`)
Five primitives decomposing k-of-T ternary routing: `sign_extract` (int64 → packed-trit signs), `distance_batch` (batch popcount over T tile signatures), `topk_abs` (top-k by |score| via bitmask selection), `apply_signed` (signed accumulation of tile outputs, NEON for both +1 and −1), `signature_update` (compound: column-sum → mean-subtract → sign-extract, caller-provided scratch).

### MTFP39 wide path (`m4t_mtfp_w.h`)
Int64 parallel of MTFP19 arithmetic. 39 trits, 2 NEON lanes. Saturating add/sub/mul via `__int128`, vec ops with compare+select clamp (no `vminq_s64` on aarch64), dense matmul (`__int128` accumulator, K ≤ 41 documented), ternary matmul. Targets wide accumulation and high-precision paths.

### MTFP4 SDOT routing cell (`m4t_mtfp4.h`)
Int8 cell, 4 trits, 16 NEON lanes. `vdotq_s32` is the hardware-native ternary matmul: 16 int8 multiply-accumulates per instruction. Scalar arithmetic, MTFP19↔MTFP4 conversion with symmetric rounding. W uses raw int8 trits (not packed) for zero SDOT decode overhead.

### Ternary matmul (`m4t_ternary_matmul.h`)
MTFP19 activations × 2-bit packed ternary weights → MTFP19 output. Trit-decode via `vqtbl1q_s8`, sign-select via `vmulq_s32`, int64 accumulator.

## Build

```bash
cd m4t
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

Requires aarch64 + NEON (Apple Silicon or compatible ARM). Non-NEON targets error at compile time.

## Test count

68 test functions across 6 binaries (`test_m4t_smoke`, `test_m4t_trit_ops`, `test_m4t_trit_reducers`, `test_m4t_route`, `test_m4t_mtfp_w`, `test_m4t_mtfp4`), all with hand-derived integer golden values. Zero float in the test suite. Includes an end-to-end mini routing pass and SDOT matmul coverage at multiple K values.

## Tools

| Tool | Purpose |
|---|---|
| `tools/m4t_size_check.sh` | Link-time `.text` budget enforcement (24 KB cap) |
| `tools/m4t_bench.c` | Per-opcode cycle counting via `mach_absolute_time` |
| `tools/m4t_lut_gen.c` | Host-side GELU + exp LUT generator (output committed as `.c`) |
| `tools/m4t_trit_golden.c` | Truth-table enumerator for TBL opcode verification |

## Cache budget

| Region | Budget | Current | Used |
|---|---|---|---|
| L1i (opcode bodies) | 24 KB | 17.7 KB | 71% |
| L1d (LUTs + constants) | 4 KB | ~0.3 KB | 8% |

## Pipeline

Items 1–5 and 7 are done: TBL trit ops, masked-VCNT reducers, routing primitives, MTFP39 wide path, MTFP4 SDOT path, and measurement tools. See `docs/M4T_PIPELINE.md` for remaining items: function-pointer opcode tables and glyph wrapper layer. See `docs/M4T_BEYOND.md` for post-pipeline future work.

## License

MIT.
