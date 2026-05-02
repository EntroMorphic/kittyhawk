# M4T — M4 Ternary Extensions

A routing-first ternary/MTFP compute substrate for aarch64 + NEON. Single-threaded at the opcode level; threading is a consumer concern.

Canonical spec: [`docs/M4T_SUBSTRATE.md`](docs/M4T_SUBSTRATE.md).

## Status — ground-zero (2026-05-01)

The substrate is being rebuilt in tiers (see [`../docs/REMEDIATION_PLAN.md`](../docs/REMEDIATION_PLAN.md)):

- **Tier 1 — pure base-3 layer — ONLINE.** Trit types, packing, element-wise ops, reductions. Zero MTFP entanglement.
- **Tier 2 — route primitives + MTFP19 mantissa arithmetic — ONLINE.** Five route primitives (`threshold_extract`, `distance_batch`, `topk_abs`, `apply_signed`, `signature_update`) plus same-block-exponent block / vec add/sub on MTFP19 mantissas. Emission-coverage helper (`m4t_route_decisions_emit_coverage`) makes the §18 input-class contract testable at the call site.
- **Tier 3a — cross-exponent accumulator — ONLINE.** `m4t_mtfp_vec_accum_aligning` (canonical) + `m4t_mtfp_vec_add_aligning` (pairwise wrapper). Path A alignment, base-3 round-to-nearest (§8.2), SATURATED+ROUNDED status flags (§14.4). Property-tested bit-exact at 10,000 samples × 6 properties. The substrate is now **genuinely floating in base 3** — the cross-exponent kernel that distinguishes MTFP from fixed-point is built.
- **Tier 3b — MTFP4 SDOT + ternary matmul — pending consumer.** Both return only when a routing consumer demands them.

## Numerical system

MTFP — Multi-Trit Floating Point, base 3. A value is `mantissa × 3^exponent`. Mantissa is an n-trit signed integer cell; exponent is sidecar metadata at the block level. Four cell widths; all blocks are 16 bytes (one NEON vector):

| Type | Container | Mantissa trits | Mantissa range | Cells per block |
|---|---|---|---|---|
| `m4t_mtfp4_t` | int8 | 4 | ±40 | 16 |
| `m4t_mtfp9_t` | int16 | 9 | ±9 841 | 8 |
| `m4t_mtfp_t` | int32 | 19 | ±581 130 733 | 4 |
| `m4t_mtfp_w_t` | int64 | 39 | ±1.72·10¹⁸ | 2 |

Mantissa bound: `(3^trits − 1) / 2`. No binary floating point at runtime.

## Live surface — Tiers 1 + 2

### Trit packing (`m4t_trit_pack.h`) — Tier 1

Pack/unpack between `m4t_trit_t` buffers and 2-bit packed `uint8_t` containers. Popcount routing distance (XOR + masked VCNT). Decode LUT shared with future ternary matmul kernels.

The packing convention (`+1 → 01`, `0 → 00`, `−1 → 10`) is load-bearing: `m4t_popcount_dist` returns a *ternary* Hamming distance with max `2N`, not a binary Hamming with max `N`. The header carries the warning at the call site.

### Trit operations (`m4t_trit_ops.h`) — Tier 1

Six element-wise ops on packed-trit buffers via 16-byte TBL lookup: `mul`, `sat_add`, `max`, `min`, `eq`, `neg`. ~28 NEON instructions per 64 trits (binary ops); `neg` is bit-swap (~5 instructions).

### Trit reducers (`m4t_trit_reducers.h`) — Tier 1

Masked-VCNT reductions: `signed_sum`, `sparsity`, `counts`. ~14 NEON instructions per 64 trits. Building blocks for routing-distance and signature-update paths.

### MTFP19 mantissa arithmetic (`m4t_mtfp.h`) — Tier 2

Block-native mantissa primitives at one shared block exponent: `block_add`, `block_sub` (exactly one NEON vector each), composed into `vec_add_inplace` / `vec_sub_inplace` / `vec_zero` with scalar tails. Saturating clamp `clamp64` for accumulator stores. Case S (§8.5) saturation; **same-block contract** — caller asserts inputs share one block exponent.

### Cross-exponent accumulator (`m4t_mtfp.h`) — Tier 3a

Two functions for combining MTFP19 mantissa buffers carrying different `block_exp` values:

- `m4t_mtfp_vec_accum_aligning(running, &running_exp, addend, addend_exp, flags, n)` — canonical accumulator. `running_exp` is in-out and may grow upward across calls. Path A alignment (max-exponent target); smaller side rescales by `3^Δ` with **base-3 round-to-nearest** (§8.2).
- `m4t_mtfp_vec_add_aligning(dst, &out_e, a, e_a, b, e_b, flags, n)` — pairwise convenience wrapper.

Status flags (`flags` byte per cell, opt-in via non-NULL): `M4T_FLAG_SATURATED` (bit 0) and `M4T_FLAG_ROUNDED` (bit 1). Sticky-OR'd across calls.

`m4t_route_apply_signed` (tier 2) is the same-block-exp degenerate case of this primitive. Architectural reframing per `journal/xexpo_design_closeout.md`.

### Routing primitives (`m4t_route.h`) — Tier 2

Five primitives composing into a k-of-T ternary routing pass:

- `threshold_extract` — int64 values → packed-trit signs with a symmetric `tau` band. Emits all three trit states (`+1` when `v > tau`, `−1` when `v < −tau`, `0` when `|v| ≤ tau`). `tau = 0` degenerates to binary sign extraction; this is the shape `signature_update` uses internally.
- `distance_batch` — query signature × T tile signatures → T distances (wraps `popcount_dist`).
- `topk_abs` — scores → k (tile, sign) decisions (bitmask uniqueness, T ≤ `M4T_ROUTE_MAX_T` = 64).
- `apply_signed` — decisions × tile outputs → accumulated MTFP19 result (Case S saturation via vec_add/sub). Three-state input: sign ∈ {-1, 0, +1}, tile_idx ≥ -1.
- `signature_update` — weight-derived signatures (setup-time compound op; internally calls `threshold_extract` with `tau = 0`).

Plus the §18-testability helper:

- `decisions_emit_coverage` — inspects a `decisions[]` array and reports whether `+1`, `0`, and `−1` sign states all appeared. Consumer integration tests use it to demonstrate the input-class contract is honored at the call site.

## Build

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

Requires aarch64 + NEON (Apple Silicon or compatible ARM). Non-NEON targets fail at CMake configure. `-Werror` is enabled.

## Tests — Tiers 1 + 2 + 3a

Six test binaries. The first five use hand-derived integer golden values. The cross-exponent test uses a bit-exact `int64` reference at 10,000 samples × 6 properties. Zero float in any test path.

| Binary | Coverage |
|---|---|
| `test_m4t_trit_pack` | pack/unpack/popcount_dist round-trip + masked + alignment |
| `test_m4t_trit_ops` | all 9 input pairs × all 6 ops; 65-trit NEON+tail case |
| `test_m4t_trit_reducers` | `signed_sum`, `sparsity`, `counts` across zero/pos/neg/mixed inputs |
| `test_m4t_mtfp` | clamp64, vec_zero, block_add/sub (NEON + aliasing + saturation), vec_* (NEON-only / scalar-only / NEON+tail) |
| `test_m4t_route` | threshold_extract, distance_batch, topk_abs, apply_signed, signature_update, end-to-end mini routing pass, `decisions_emit_coverage` |
| `test_m4t_mtfp_accum_aligning` | accumulator correctness, invariant, aliasing, flags; pairwise wrapper correctness; pairwise roundtrip; all bit-exact vs reference at 10,000 samples each |

## What's not here

Tier 3b surfaces — return only when a routing consumer demands them:

- `m4t_mtfp4.*` — SDOT-native MTFP4 routing cell + ternary matmul.
- `m4t_ternary_matmul.*` — MTFP19 × packed-ternary matmul.

Deliberately archived (see `01MAY26_archived/`):

- Dense MTFP×MTFP matmul, LayerNorm, bias_add, fan-in normalize.
- LUT-backed GELU/softmax/argmax.
- MTFP39 wide-cell arithmetic.
- MTFP9 mid-cell arithmetic — type stays in `m4t_types.h`; no kernels until demanded.
- Function-pointer opcode dispatch tables.

Each returns only when a named, measured routing consumer drives it.

## License

[MIT](../LICENSE).
