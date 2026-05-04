# M4T — M4 Ternary Extensions

A routing-first ternary/MTFP compute substrate for aarch64 + NEON. Single-threaded at the opcode level; threading is a consumer concern.

The build requires aarch64 + NEON (the configure step fails otherwise), and most kernels use NEON intrinsics for the inner loop. A few are scalar — notably the cross-exponent accumulator (no NEON integer-divide on ARM) and the route-primitive scalar paths. Each module's docstring states whether its hot path is NEON or scalar.

Canonical spec: [`docs/M4T_SUBSTRATE.md`](docs/M4T_SUBSTRATE.md).

## Status — ground-zero (2026-05-01)

The substrate is being rebuilt in tiers (see [`../docs/REMEDIATION_PLAN.md`](../docs/REMEDIATION_PLAN.md)):

- **Tier 1 — pure base-3 layer — ONLINE.** Trit types, packing, element-wise ops, reductions. Zero MTFP entanglement.
- **Tier 2 — route primitives + MTFP19 mantissa arithmetic — ONLINE.** Five route primitives (`threshold_extract`, `distance_batch`, `topk_abs`, `apply_signed`, `signature_update`) plus same-block-exponent block / vec add/sub on MTFP19 mantissas. Emission-coverage helper (`m4t_route_decisions_emit_coverage`) makes the §18 input-class contract testable at the call site.
- **Tier 3a — cross-exponent accumulator — ONLINE.** `m4t_mtfp_vec_accum_aligning` (canonical) + `m4t_mtfp_vec_add_aligning` and `m4t_mtfp_vec_sub_aligning` (pairwise wrappers). Path A alignment, base-3 round-to-nearest-even (§8.2; ties impossible due to odd divisors), per-block status flags (§14.4: 1 byte per MTFP19 block carrying SATURATED + ROUNDED bits for each of 4 cells). Property-tested bit-exact across **14 properties**.
- **Tier 3b — MTFP4 SDOT matmul + cell-width conversions — ONLINE.** `m4t_mtfp4_sdot_matmul_bt` (Case W per §8.4: MTFP4 × ternary → MTFP19, exact by construction for K ≤ ~14.5M), `m4t_mtfp4_to_mtfp19` (widen, exact), `m4t_mtfp19_to_mtfp4` (narrow, base-3 round-to-nearest-even + saturate, with flag tracking).
- **Tier 3c — MTFP19 × packed-ternary matmul — ONLINE.** `m4t_mtfp_ternary_matmul_bt` (Case S per §8.5: int64 accumulator, MTFP19 saturating store, optional per-block SATURATED flag tracking).

## Numerical system

MTFP — Multi-Trit Floating Point, base 3. A value is `mantissa × 3^exponent`. Mantissa is an n-trit signed integer cell; exponent is sidecar metadata at the block level. Four cell widths; all blocks are 16 bytes (one NEON vector):

| Type | Container | Mantissa trits | Mantissa range | Cells per block |
|---|---|---|---|---|
| `m4t_mtfp4_t` | int8 | 4 | ±40 | 16 |
| `m4t_mtfp9_t` | int16 | 9 | ±9 841 | 8 |
| `m4t_mtfp_t` | int32 | 19 | ±581 130 733 | 4 |
| `m4t_mtfp_w_t` | int64 | 39 | ±1.72·10¹⁸ | 2 |

Mantissa bound: `(3^trits − 1) / 2`. No binary floating point at runtime.

## Live surface — Tiers 1 + 2 + 3

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

Three functions for combining MTFP19 mantissa buffers that may carry different `block_exp` values. **Scalar implementation** (no NEON path; ARM has no integer-divide instruction and profiling has not yet shown this kernel as a hot path):

- `m4t_mtfp_vec_accum_aligning(running, &running_exp, addend, addend_exp, flags, n)` — canonical accumulator. `running_exp` is in-out and may grow upward across calls. Path A alignment (max-exponent target); smaller side rescales by `3^Δ` with **base-3 round-to-nearest-even** (§8.2; ties impossible because powers of 3 are odd).
- `m4t_mtfp_vec_add_aligning(dst, &out_e, a, e_a, b, e_b, flags, n)` — pairwise add wrapper.
- `m4t_mtfp_vec_sub_aligning(dst, &out_e, a, e_a, b, e_b, flags, n)` — pairwise sub wrapper.

Status flags layout (`flags` opt-in via non-NULL pointer): one byte per MTFP19 block (4 cells per block). Each byte encodes 2 events × 4 cells; bit `(slot * 2 + 0)` = SATURATED for cell `slot`, bit `(slot * 2 + 1)` = ROUNDED. Caller sizes via `M4T_FLAG_BYTES(n)` and reads via `m4t_flag_test(flags, cell, event)`. Sticky-OR'd across calls.

The same-block-exp arithmetic in `m4t_route_apply_signed` (tier 2) is the structurally degenerate case of this primitive — same shape, with `addend_exp == running_exp` always. The cross-exp accumulator generalizes that arithmetic; it does not replicate `apply_signed`'s sign-dispatch / sentinel-skip routing semantics. Per `journal/xexpo_design_closeout.md` and the kernel red-team's L4 finding.

### SDOT MTFP4 matmul (`m4t_mtfp4.h`) — Tier 3b

The substrate's hot path. **NEON-accelerated via `vdotq_s32`** (16 int8 multiply-accumulates per instruction); scalar tail for K not divisible by 16.

- `m4t_mtfp4_sdot_matmul_bt(Y, X, W, M, K, N)` — MTFP4 × ternary → MTFP19. **Case W per §8.4: exact by construction.** Output is MTFP19 (int32 mantissas); since `|X| ≤ 40` and `|W| ≤ 1`, the worst-case sum `K · 40` fits MTFP19 for any K up to ~14.5M.
- `m4t_mtfp4_to_mtfp19(dst, src, n)` — widen MTFP4 → MTFP19 by ×6561. Exact (static-asserted bound).
- `m4t_mtfp19_to_mtfp4(dst, src, flags, n)` — narrow MTFP19 → MTFP4. Base-3 round-to-nearest-even divide, then saturate to ±MAX_VAL_4. Per-block flags (ROUNDED / SATURATED) track precision events; opt-in via non-NULL `flags`.

W layout for SDOT is **unpacked int8** in {-1, 0, +1} (not packed trits — SDOT requires int8 operands). 4× memory of packed; zero decode overhead.

### MTFP19 × packed-ternary matmul (`m4t_ternary_matmul.h`) — Tier 3c

For consumers that need full MTFP19 precision on activations (vs the MTFP4 SDOT path's narrower input). **NEON-accelerated** via 16-trit decode + bit-select + conditional negate; int64 accumulator; saturating clamp to MTFP19 on store.

- `m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, flags, M, K, N)` — MTFP19 × packed-trit → MTFP19. **Case S per §8.5: fixed-output saturate.** Optional per-block SATURATED flag tracking (`flags` non-NULL). Accumulator overflow point: K ≈ 1.59e10.

Inner loop uses `vbslq_s32` + `vnegq_s32` over decoded signs in {-1, 0, +1} — no `vmulq_s32`. Multiplying by a sign through a general-purpose multiply opcode is the base-2 shortcut; the base-3-native expression is mask-and-conditional-negate, which is what TBL + bit-select compute directly.

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

## Tests — Tiers 1 + 2 + 3

Ten ctest binaries. Tier-1/2 tests use hand-derived integer golden values; tier-3 tests use bit-exact `int64` reference implementations as the oracle (no fp in any test path).

| Binary | Coverage |
|---|---|
| `test_m4t_trit_pack` | pack/unpack/popcount_dist round-trip + masked + alignment |
| `test_m4t_trit_ops` | all 9 input pairs × all 6 ops; 65-trit NEON+tail case |
| `test_m4t_trit_reducers` | `signed_sum`, `sparsity`, `counts` across zero/pos/neg/mixed inputs |
| `test_m4t_mtfp` | clamp64, vec_zero, block_add/sub (NEON + aliasing + saturation), vec_* (NEON-only / scalar-only / NEON+tail) |
| `test_m4t_route` | threshold_extract, distance_batch, topk_abs, apply_signed, signature_update, end-to-end mini routing pass, `decisions_emit_coverage` |
| `test_m4t_mtfp_accum_aligning` | 14 properties (10k samples per random property + curated boundary cases): correctness, invariant, determinism, per-block flags, trailing-block-bits-zero, long-sequence (K=256), boundary, n=0, wrapper, roundtrip, dst==a aliasing, NULL out_e, sub-via-negation, sub-self. All bit-exact vs an int64 reference. |
| `test_m4t_mtfp4` | 12 tests: clamp boundaries, SDOT golden 2×4×3, SDOT random vs int64 reference (200 trials, K up to 1024), SDOT high-magnitude (K=4096), SDOT long-K stress (K=1M vs int64 reference), zero-dim, widen exact, narrow round, narrow saturate, narrow flags, **narrow property** (10k random samples + boundary-targeted), widen-narrow roundtrip. |
| `test_m4t_ternary_matmul` | 9 tests: golden 2×4×3, random vs reference (200 trials), long-K stress (K=1M), saturation clamp, saturation flags, **partial-block** (M·N=5 trailing-bits-stay-zero), **invalid trit code** (0b11 reserved → identical NEON/scalar handling), zero-dim, determinism. |
| `test_m4t_elemental_floor` | three property tests on the cell-level elemental floor (`shift3`, `select`, neg-via-select composite re-derivation), plus the R-G3 path test for the cross-exponent accumulator's flags=NULL fast path. |
| `test_m4t_assert_live` | V4 deliberate-abort meta-test: forks a child, calls `m4t_route_topk_abs(T=200)` (T > `M4T_ROUTE_MAX_T = 64`), verifies `WIFSIGNALED && WTERMSIG == SIGABRT`. Proves substrate asserts are actually compiled into `libm4t_test` and fire when triggered. |

### Test-build discipline (V3 + V4)

Tests link against `libm4t_test.a`, a parallel STATIC library compiled from the same sources as `libm4t.a` but with `-UNDEBUG` applied. This makes substrate-internal asserts (precondition checks) actually fire when tests trigger them — without it, `assert(EXPR)` under the substrate's production `-DNDEBUG` becomes `((void)0)` and EXPR is never evaluated. Production binaries (perf benches, gesh probes) keep linking against `libm4t.a` (NDEBUG). The split is enforced via `add_library(m4t_test STATIC ...)` plus `target_compile_options(m4t_test PRIVATE -UNDEBUG)` in `CMakeLists.txt`. The same pattern exists for `gesh_test`, `gesh_bench_test`, `gesh_image_canon_test` in `gesh/CMakeLists.txt`. Verified at three levels: build-time (`nm` shows `___assert_rtn` refs only in `_test` variants), link-time (production binaries link 0 assert symbols), runtime (`test_m4t_assert_live` confirms asserts actually fire).

## Reading perf measurements

Substrate perf claims rest on a specific workload shape: **carry-dependent, single-pass accumulation** (each iteration's output depends on the previous iteration's output). That's how the substrate's actual consumers use these kernels — they accumulate into state. The bench harness `bench_m4t_tier2_perf` measures that shape.

Two consequences for reading bench numbers:

1. **"Substrate is well-optimized" is a property of measurement under that shape, not a global claim.** A future consumer that pipelines block ops across independent buffers (e.g., batched matmul over many small tiles) sees a different bottleneck profile. Separate measurements are warranted; existing numbers don't transfer.

2. **Compiler optimizations that target call overhead can be invisible.** LTO inlines `m4t_mtfp_block_add` cleanly, but on the carry-dep workload the data dependency between iterations dominates and the inlining win is hidden. On a pipelined workload (independent dsts, no carry), LTO produces a **3× speedup** for the same target function — proven by `bench_m4t_lto`. See `journal/v4_residual_3_lto_microbench_closeout.md` for the controlled comparison and `journal/tier2_residuals_v4_closeout.md` (with its V4-residual-3 update) for the corrected framing.

Bottom line: when reading "kernel X takes Y ns/call," check that Y was measured under the workload shape that matches your intended consumer.

## What's not here

Deliberately archived (see `01MAY26_archived/`):

- Dense MTFP×MTFP matmul, LayerNorm, bias_add, fan-in normalize.
- LUT-backed GELU/softmax/argmax.
- MTFP39 wide-cell arithmetic.
- MTFP9 mid-cell arithmetic — type stays in `m4t_types.h`; no kernels until demanded.
- Function-pointer opcode dispatch tables.

Each returns only when a named, measured routing consumer drives it.

## License

[MIT](../LICENSE).
