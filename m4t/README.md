# M4T — M4 Ternary Extensions

A routing-first ternary/MTFP compute substrate for aarch64 + NEON. Single-threaded at the opcode level; threading is a consumer concern.

Canonical spec: [`docs/M4T_SUBSTRATE.md`](docs/M4T_SUBSTRATE.md).

## Numerical system

MTFP — Multi-Trit Floating Point, base 3. A value is `mantissa × 3^exponent`, with the mantissa in an n-trit signed integer cell and the exponent as sidecar metadata on the block. Four cell widths; all blocks are 16 bytes (one NEON vector):

| Type | Container | Mantissa trits | Mantissa range | Cells per block |
|---|---|---|---|---|
| `m4t_mtfp4_t` | int8 | 4 | ±40 | 16 |
| `m4t_mtfp9_t` | int16 | 9 | ±9 841 | 8 |
| `m4t_mtfp_t` | int32 | 19 | ±581 130 733 | 4 |
| `m4t_mtfp_w_t` | int64 | 39 | ±1.72·10¹⁸ | 2 |

Mantissa bound: `(3^trits − 1) / 2`. No binary floating point at runtime.

## Live surface

All sources under `src/` are routing-first or foundational. Dense matmul, dense nonlinearities, and the MTFP39 wide path were moved to `archive/` in the rebuild; they return when a consumer drives them.

### Numeric core (`m4t_mtfp.h`)
Block-native mantissa primitives for MTFP19: `block_add`, `block_sub` (exactly one NEON vector each), composed into `vec_add_inplace` / `vec_sub_inplace` / `vec_zero` with scalar tails. Saturating clamp `clamp64` for accumulator stores. Case S (§8.5) saturation; same-block contract.

### Trit packing (`m4t_trit_pack.h`)
Pack/unpack between `m4t_trit_t` buffers and 2-bit packed `uint8_t` containers. Popcount routing distance (XOR+VCNT, masked). Decode LUT shared with the ternary matmul.

### Trit operations (`m4t_trit_ops.h`)
Six element-wise ops on packed-trit buffers via 16-byte TBL lookup: `mul`, `sat_add`, `max`, `min`, `eq`, `neg`. ~28 NEON instructions per 64 trits (binary ops); `neg` is bit-swap (~5 instructions).

### Trit reducers (`m4t_trit_reducers.h`)
Masked-VCNT reductions: `signed_sum`, `sparsity`, `counts`. ~14 NEON instructions per 64 trits. Feeds the routing-distance and signature-update paths.

### Routing primitives (`m4t_route.h`)
Five primitives composing into a k-of-T ternary routing pass:
- `threshold_extract` — int64 values → packed-trit signs with a symmetric `tau` band. Emits all three trit states (`+1` when `v > tau`, `−1` when `v < −tau`, `0` when `|v| ≤ tau`). `tau = 0` degenerates to binary sign extraction and is the shape `signature_update` uses internally. Replaces the earlier `sign_extract` per the §18 emission-coverage criterion.
- `distance_batch` — query signature × T tile signatures → T distances (wraps `popcount_dist`).
- `topk_abs` — scores → k (tile, sign) decisions (bitmask uniqueness, T ≤ `M4T_ROUTE_MAX_T` = 64)
- `apply_signed` — decisions × tile outputs → accumulated MTFP19 result (Case S saturation via vec_add/sub)
- `signature_update` — weight-derived signatures (setup-time compound op, heap-allocated row buffer; internally calls `threshold_extract` with `tau = 0`)

### Ternary matmul — MTFP19 (`m4t_ternary_matmul.h`)
MTFP19 activations × 2-bit packed ternary weights → MTFP19. Inner loop uses `vbslq_s32` + `vnegq_s32` over decoded signs in {-1, 0, +1}; no `vmulq_s32`, no dense shape. Int64 accumulator, clamp on store (Case S).

### Ternary matmul — MTFP4 (`m4t_mtfp4.h`)
SDOT-native: `vdotq_s32` computes 16 int8 × int8 → int32 MACs per instruction. This is Case W — the output widens to MTFP19 mantissa exactly by construction (max magnitude 16 × 40 × 40 = 25 600 ≪ int32 max). The one exact hardware-native ternary matmul primitive.

## Build

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

Requires aarch64 + NEON (Apple Silicon or compatible ARM). Non-NEON targets fail at CMake configure. `-Werror` is enabled.

Dev tools (off by default):

```bash
cmake -S . -B build -DM4T_BUILD_TOOLS=ON
```

This builds:
- `m4t_trit_golden` — truth-table enumerator for TBL opcode verification.
- `m4t_lut_gen` — offline GELU/exp LUT generator. The *only* sanctioned binary-float code in the entire ecosystem; runs at build time, emits an integer `.c` file, never linked into `libm4t.a` at runtime.

## Tests

Five test binaries, all with hand-derived integer golden values. Zero float in the test suite.

| Binary | Coverage |
|---|---|
| `test_m4t_mtfp` | clamp64, vec_zero, block_add/sub (NEON + aliasing + saturation), vec_* (NEON-only / scalar-only / NEON+tail) |
| `test_m4t_trit_ops` | all 9 input pairs × all 6 ops; 65-trit NEON+tail case |
| `test_m4t_trit_reducers` | signed_sum, sparsity, counts across zero/pos/neg/mixed inputs |
| `test_m4t_route` | threshold_extract, distance_batch, topk_abs, apply_signed, signature_update, end-to-end mini routing pass |
| `test_m4t_mtfp4` | clamp, SDOT matmul (multiple K values including tail), MTFP19↔MTFP4 conversion |

## What's not here

Deliberately archived (see top-level `archive/README.md`):
- Dense MTFP×MTFP matmul, LayerNorm, bias_add, fan-in normalize.
- LUT-backed GELU/softmax/argmax.
- MTFP39 wide-cell arithmetic.
- Function-pointer opcode dispatch tables.

Each returns only when a named routing consumer drives it.

## License

[MIT](../LICENSE).
