# M4T — M4 Ternary Extensions

A routing-first ternary/MTFP compute substrate for aarch64 + NEON. Single-threaded at the opcode level; threading is a consumer concern.

The build requires aarch64 + NEON (the configure step fails otherwise). **Production paths are NEON-only** per `feedback_function_over_speed_no_scalar` — the no-scalar audit (2026-05-06, per `journal/no_scalar_audit_2026_05_06.md`) closed the last 9 fully-scalar production functions. Allowed exceptions: (a) `_scalar_ref` test oracles, exposed for bit-exact verification gates; (b) geometric scalar tails for sub-block remainders (e.g., the trailing 0-15 trits of a 16-byte NEON loop). Every public function in `libm4t` either has a NEON inner loop or hard-fails at compile time on non-NEON targets.

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

Three functions for combining MTFP19 mantissa buffers that may carry different `block_exp` values. **NEON-routed** via `vmlal_s32` magic-multiply for the divide step (per `journal/cross_exp_accum_routing_*` LMM cycle). The scalar reference (`m4t_mtfp_vec_accum_aligning_scalar_ref`) remains as a test-only oracle:

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

For consumers that need full MTFP19 precision on activations (vs the MTFP4 SDOT path's narrower input). **NEON-accelerated** via 16-trit decode + `vmlal_s32` widening multiply-accumulate; int64 accumulator; saturating clamp to MTFP19 on store. Plus public scalar reference (`m4t_mtfp_ternary_matmul_bt_scalar_ref`) for bit-exact testing.

- `m4t_mtfp_ternary_matmul_bt(Y, X, W_packed, flags, M, K, N)` — MTFP19 × packed-trit → MTFP19. **Case S per §8.5: fixed-output saturate.** Optional per-block SATURATED flag tracking (`flags` non-NULL). Accumulator overflow point: K ≈ 1.59e10.
- `m4t_mtfp_ternary_matmul_bt_scalar_ref(...)` — same semantics, always-scalar; test-only oracle exposed so bit-exact verification survives productionization.

Inner loop pipeline (per 16-trit block): decode 16 packed trits → 16 int8 signs (via TBL); sign-extend int8 → int32; 8× `vmlal_s32` widening MAC into int64x2 accumulator pair. Multiplying by trit ∈ {-1, 0, +1} subsumes both conditional-negate AND zero-gate — multiply IS the operation, no separate mask plane needed. ~17 cycles per 16-trit block on Apple Silicon. The closest existing M4/NEON hardware analog to a "ternary MAC at int32 width" given there is no native trit-aware silicon op. See `journal/ternary_mac_routing_*.md` for the LMM cycle that arrived at this routing.

### §20 sub-2-bit packed matmul (`m4t_ternary_matmul.h`) — Tier 3c

Two NEON-only matmul kernels using the §20 5-in-8 packing (1.6 bits/cell, sub-2-bit dense format per `M4T_SUBSTRATE.md` §20). Both accept arbitrary `(K, N)` shapes per TD-1 — non-aligned shapes use a per-trit scalar geometric tail for K%80 and a single-acc NEON inner loop for N%4.

- `m4t_ternary_5in8_matmul_bt(Y, X, W_packed, M, K, N)` — unpacked X (8 b/c) × 5-in-8 W (1.6 b/c). Split-LUT decode (1× div-by-9 magic-multiply + 5× vqtbl1q/vqtbl2q lookups per byte), register-tile by 4 j cells, 5 SDOTs per 80-trit block. Per `journal/m4t_5in8_*.md`.
- `m4t_ternary_5in8_matmul_xpacked_bt(Y, X_packed, W_packed, M, K, N)` — X also packed at 1.6 b/c (TD-7). Per row, decodes `X_packed[i, :]` into the same 5 stride-aligned int8 arrays, then runs §20's tile body verbatim. Per `journal/td7_xpacked_bench.md`: **§20-xp consistently 14-26% faster than §20 across all tested (M, K)**, because the X-packed kernel's NEON-vectorized X permutation is faster than §20's scalar X permutation. Recommended packed kernel for new consumers; **§20 is dominated by §20-xp**, kept only for backwards compatibility.

Both have `_scalar_ref` test oracles (`m4t_ternary_5in8_matmul_bt_scalar_ref`, `m4t_ternary_5in8_matmul_xpacked_bt_scalar_ref`).

**Decision tree (consumer guidance):**
- Single-token inference, K ≥ 4480 → §20-xp (fastest at this regime: 0.47-0.86× of `m4t_ternary_dot_matmul_bt`).
- Batched inference / training → unpacked dot (`m4t_ternary_dot_matmul_bt`); xp/dot ≈ 1.05-1.5×.
- Storage- or memory-bandwidth-bound → §20-xp (5× X savings × 5× W savings = 25× total bandwidth reduction).

### BitNet inference primitives (`m4t_mtfp.h`) — Phase 2

Driven by the BitNet b1.58-2B-4T inference harness (`gesh/bitnet/`); each
primitive earned its place by being on the per-token forward path.

- `m4t_mtfp_rmsnorm` / `m4t_mtfp_rmsnorm_bx` — block-int128 SoS, m4t_int32_rsqrt,
  per-cell γ × x × inv >> total_shift with NEON uint96 multiply-and-shift. The
  `_bx` variant is bx-aware: when `gamma_bx > target_bx` it pre-rescales γ to
  target_bx to avoid silent intermediate saturation (per
  `journal/substrate_vs_hf_2026-05-09/RESOLVED.md`).
- `m4t_mtfp_softmax` — exp LUT (z ∈ [-30, 0], 4096 cells) + reciprocal-of-sum
  via `m4t_int32_recip`. V14.G v2 NEON gather restored bit-exactness vs V13.
- `m4t_a8_quantize` / `m4t_a8_dequantize` — per-tensor absmax + int8 quantize
  (matches BitNet W1.58A8 spec).
- `m4t_mtfp_vec_scale` / `m4t_mtfp_bitlinear_scale_bx` — BitLinear output scale
  (`y · α · activation_absmax / 127`). Combined-divisor magic dropped CPU from
  22% → 7% (per `journal/v14f_profile_opt_*`).
- `m4t_mtfp_bitlinear_scale_no_a8_bx` — variant for the bit-faithful path
  (skip A8 quantize, accept int64 raw matmul outputs).
- `m4t_mtfp_relu2_inplace`, `m4t_mtfp_vec_mul_inplace` — FFN sub-pieces.
- `m4t_rope_apply` — RoPE rotation, NEON tile.
- `m4t_mtfp_rescale_bx` — explicit between-bx rescale (consumer composition;
  also used internally by `_bx` kernels).

The `_bx` variants accept explicit per-tensor `block_exp` parameters and
produce output at a caller-chosen `target_bx`, so consumers that store
weights and activations at different precisions (e.g., BitNet's γ at bx=17–21
vs activations at bx=8) can compose without manual rescale stitching.

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

22 ctest binaries (full suite includes gesh consumer tests; m4t-substrate-specific tests are listed below). Tier-1/2 tests use hand-derived integer golden values; tier-3 tests use bit-exact reference implementations (`_scalar_ref` oracles) as the gate (no fp in any test path).

| Binary | Coverage |
|---|---|
| `test_m4t_trit_pack` | pack/unpack/popcount_dist round-trip + masked + alignment. **Plus NEON-vs-scalar_ref bit-exact gates** (per no-scalar audit 2026-05-06): 4-in-8 pack/unpack across 30 N values × 100 random samples each (~6,000 NEON-vs-scalar comparisons); 5-in-8 pack/unpack across 26 N values × 50 samples (~2,600). |
| `test_m4t_trit_ops` | all 9 input pairs × all 6 ops; 65-trit NEON+tail case |
| `test_m4t_trit_reducers` | `signed_sum`, `sparsity`, `counts` across zero/pos/neg/mixed inputs |
| `test_m4t_mtfp` | clamp64, vec_zero, block_add/sub (NEON + aliasing + saturation), vec_* (NEON + tail) |
| `test_m4t_route` | threshold_extract, distance_batch, topk_abs, apply_signed, signature_update, end-to-end mini routing pass, `decisions_emit_coverage` |
| `test_m4t_mtfp_accum_aligning` | 14 properties (10k samples per random property + curated boundary cases): correctness, invariant, determinism, per-block flags, trailing-block-bits-zero, long-sequence (K=256), boundary, n=0, wrapper, roundtrip, dst==a aliasing, NULL out_e, sub-via-negation, sub-self. All bit-exact vs an int64 reference. |
| `test_m4t_mtfp4` | 13 tests: clamp boundaries, SDOT golden 2×4×3, SDOT random vs int64 reference (200 trials, K up to 1024), SDOT high-magnitude (K=4096), SDOT long-K stress (K=1M), zero-dim, widen exact, narrow round, narrow saturate, narrow flags, **narrow property** (10k random samples + boundary-targeted), widen-narrow roundtrip, **conversions NEON-vs-scalar_ref** (no-scalar audit: 20 N values × 100 samples × both directions = 4,000 checks). |
| `test_m4t_ternary_matmul` | 9 tests: golden 2×4×3, random vs reference (200 trials), long-K stress (K=1M), saturation clamp + flags, partial-block, invalid trit code (0b11 reserved), zero-dim, determinism. |
| `test_m4t_elemental_floor` | three property tests on the cell-level elemental floor (`shift3`, `select`, neg-via-select composite re-derivation), plus the R-G3 path test for the cross-exponent accumulator's flags=NULL fast path. |
| `test_m4t_assert_live` | V4 deliberate-abort meta-test: forks a child, calls `m4t_route_topk_abs(T=200)` (T > `M4T_ROUTE_MAX_T = 64`), verifies `WIFSIGNALED && WTERMSIG == SIGABRT`. Proves substrate asserts are actually compiled into `libm4t_test` and fire when triggered. |
| `test_m4t_shift3_neon` | shift3 bit-exact: divide path (k<0; sample + boundary + alias + exhaustive mode `./test_m4t_shift3_neon x` ~25s sweeping 22.08×10⁹ inputs across 19 k values), **multiply path (k>0; no-scalar audit added)** across 19 k values × ~50,012 inputs each (~950K checks), and saturation collapse (k≥20). Plus min-of-5 perf bench. |
| `test_m4t_ternary_matmul_neon` | Production NEON path of `m4t_mtfp_ternary_matmul_bt` (vmlal_s32-routed) vs `m4t_mtfp_ternary_matmul_bt_scalar_ref`. 23 curated configs + 1000 random + 3 saturation-edge (clamp + flag bits match) + alias assertions for both Y==X and Y==W_packed. Plus 5-shape BATCHED perf bench. |
| `test_m4t_accum_aligning_neon` | Cross-exp accum NEON path bit-exact vs `_scalar_ref`. Per `journal/cross_exp_accum_routing_*`. |
| `test_m4t_ternary_5in8_matmul` | §20 5-in-8 W-packed matmul: pack/unpack roundtrip + golden + bit-exact NEON-vs-scalar_ref across aligned configs and **K-tail / N-tail / both-tail paths** (TD-1 arbitrary-(K,N) closure). |
| `test_m4t_ternary_5in8_xpacked` | §20 X-packed sibling (TD-7). Two gates: G1 NEON-vs-scalar_ref bit-exact across aligned + tail regimes; G2 cross-equivalence with §20 (xpacked == bt when X_unpacked unpacks to X_packed). |

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
