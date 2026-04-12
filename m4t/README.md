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

35 tests across 2 binaries (`test_m4t_smoke`, `test_m4t_trit_ops`), all with hand-derived integer golden values. Zero float in the test suite.

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
| L1i (opcode bodies) | 24 KB | 10.3 KB | 42% |
| L1d (LUTs + constants) | 4 KB | ~0.2 KB | 5% |

## Pipeline

See `docs/M4T_PIPELINE.md` for the remaining items: masked-VCNT reducers, routing primitives, MTFP39 wide path, MTFP4 SDOT path, opcode tables, and measurement tools.

## License

MIT.
