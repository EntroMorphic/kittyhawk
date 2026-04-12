# Glyph Remediation Plan

**Status:** Pre-implementation. This plan enumerates the bugs, smells, and policy violations found in the `reference-code/native/src/` kernels from trix-z, and defines what must change before any of it enters `glyph/src/`.

## Ground Rule

Glyph is **ternary / multi-trit / multi-trit floating point only**. No binary floating point (`float`, `double`, `float16`, `bfloat16`) and no small integer quantization types (`int8_t`, `int16_t`, `uint8_t`/`uint16_t` as numeric values). Byte buffers holding **packed trit containers** are fine — they are storage, not numbers. `int32_t` is allowed **only** as an MTFP cell (`typedef int32_t mtfp_t`, `real = value / 59049`), never as a binary integer operand.

Every item below is evaluated against this rule.

---

## Part 1 — `trix_atoms.c` / `trix_atoms.h`

### 1.1 Policy violations (drop entirely)

Everything in the reference atoms layer that takes or returns `float*` is dropped. See `docs/atoms_review.md` (upcoming) for the full function-by-function table. Summary:

- **Drop:** `trix_vec_add/sub/mul/scale/fma`, `trix_dot`, `trix_sum_sq`, `trix_matmul/bt/at`, `trix_bias_add`, `trix_bias_grad`, `trix_gelu`, `trix_gelu_grad`, `trix_softmax`, `trix_layernorm_forward_save`, `trix_layernorm_backward`, `trix_adamw_update`, `trix_sgd_update`, `trix_mtfp21_quantize`, `TrixAtomFFN` (7 functions), `trix_cross_entropy_loss`, `trix_cross_entropy_grad`. That is ~30 symbols, ~80% of the file.

- **Keep structurally:** `trix_pack_ternary`, `trix_ternary_pack_weights_i8`, `trix_popcount_dist_neon` (pure bitwise; no numeric contamination).

- **Rewrite with MTFP interfaces:** `trix_ternary_matvec_i8` becomes `mtfp_ternary_matvec` — MTFP activations in, MTFP output, 2-bit packed weights unchanged.

### 1.2 Bugs

**B-A1. Header/impl parameter-order mismatch on matmul.**
`trix_atoms.h:40-42` declares `trix_matmul(..., int M, int N, int K)`.
`trix_atoms.c:157` defines `trix_matmul(..., int M, int K, int N)`.
C ignores parameter names in declarations, so this compiles, but the header is misleading and any reader who trusts it will build the wrong mental model. Since both functions are being dropped from glyph, this is moot for us — but worth noting because the same inconsistency may lurk elsewhere in the reference.

**B-A2. `trix_sgd_update` defined after first use.**
`trix_atoms.c:406` calls it inside `trix_atom_ffn_sgd_step`; the definition is at `trix_atoms.c:407`. Works because the header declares it, but is a style smell. N/A for glyph (whole FFN dropped).

### 1.3 Smells

**S-A1. Two pack layouts, one encoding.**
`trix_pack_ternary` (signatures, 1D) and `trix_ternary_pack_weights_i8` (weights, 2D row-major with `Kp = K/4` stride) use identical 2-bit codes but different shapes. Callers must not mix them. Glyph should either (a) unify on one layout, or (b) give them names that make the shape obvious (`glyph_pack_trits_1d`, `glyph_pack_trits_rowmajor`).

**S-A2. GELU via `erff` / `expf`.**
Lines 212-220 call `erff` and `expf` per element. This is a reference implementation that the architecture doc flatly contradicts (the real path uses a 708K-entry LUT). Already dropped; the LUT version lives in `trix_mtfp.c`. Flagged so we don't accidentally re-import it.

**S-A3. LayerNorm `dw`/`db` post-parallel serial loop.**
`trix_atoms.c:286-289` accumulates weight/bias gradients in a serial loop after the parallel `dispatch_apply` block to avoid races. Correct but leaves performance on the table. N/A for glyph — LN backward will be rewritten in MTFP anyway.

**S-A4. `MTFP21_SCALE=59049` and `MTFP21_MAX` duplicated.**
Declared in `trix_atoms.c:293-294` and in `trix_mtfp.h:41-45`. Glyph must have exactly one canonical definition, in `mtfp.h`. Delete everywhere else.

---

## Part 2 — `trix_mtfp.c` / `trix_mtfp.h`

This is the layer closest to glyph's target. Most of it is pure integer arithmetic and is structurally portable. The violations are concentrated in three places.

### 2.1 Policy violations

**V-M1. `mtfp_ternary_matmul` and `mtfp_ternary_matmul_bt` use `int8_t*` weights.**
`trix_mtfp.c:94` and `:146` take ternary weights as `const int8_t*` with **one trit per byte**. This is a double violation: int8 as a numeric type is banned, and one-trit-per-byte is 4× wasteful vs. the 2-bit packing already in use elsewhere in the reference. Rewrite to take `const uint8_t* W_packed` and call into a shared trit-decode helper.

**V-M2. Float boundary conversion functions.**
`mtfp_from_float`, `mtfp_to_float`, `mtfp_from_float_batch`, `mtfp_to_float_batch` exist for loading weights and reading outputs. Glyph's default posture: these do not belong in the runtime library at all. **Decision required** — pick one:
- **(a) Delete.** Weights ship as pre-computed MTFP binary blobs; outputs are read as MTFP cells and interpreted by the consumer. Strongest purity.
- **(b) Quarantine in `tools/mtfp_io.c`.** Not linked into `libglyph`; only into a host-side loader/converter CLI.
- **(c) Keep in `mtfp.c` behind `GLYPH_IO_ALLOW_FLOAT` ifdef, off by default.** Weakest; reserved as an escape hatch.

Recommendation: **(b)**. The conversions are useful for bringing external weights across the wall exactly once, but they should not be a library symbol.

**V-M3. LUT initialization calls `expf` / `tanhf`.**
`mtfp_softmax_init` (`trix_mtfp.c:267`) and `mtfp_gelu_init` (`trix_mtfp.c:350`) each iterate the representable range and call `expf` / `tanhf` on floats to populate the table. The **hot path** (`mtfp_exp`, `mtfp_gelu`) is pure integer lookup; only initialization touches float. Options:
- **(a) Precompute offline.** Write a host-side generator that emits `const mtfp_t gelu_table[GELU_TABLE_SIZE] = { ... };` as a `.c` file (or `.bin` blob). Compiled into the library; zero float at runtime or link time.
- **(b) Quarantine init in `tools/lut_gen.c`** same as boundary conversions.

Recommendation: **(a)**. LUTs as compile-time `const` arrays is the cleanest outcome — no runtime allocation, no init race, and the generator is a one-off program that runs on a dev machine and whose output is version-controlled.

**V-M4. `mtfp_exp` uses `mtfp_from_float(403.4f)` in its hot path.**
`trix_mtfp.c:279` returns `mtfp_from_float(403.4f)` when the input exceeds the positive table bound. This is a float call on every clamped lookup. Trivial fix: precompute as `static const mtfp_t MTFP_EXP_CLAMP_HI = 23813074;` (`mtfp_from_float(403.4f)` at `MTFP_SCALE=59049`).

### 2.2 Bugs

**B-M1. `mtfp_ternary_matmul` has no parallelism while `_bt` does.**
`mtfp_ternary_matmul_bt` at `trix_mtfp.c:94` wraps its outer `M` loop in `dispatch_apply` on Apple; the non-transposed `mtfp_ternary_matmul` at `trix_mtfp.c:146` does not. Asymmetric. After rewrite (V-M1), both paths should parallelize identically.

**B-M2. `mtfp_ternary_matmul_bt` compound-literal NEON register build.**
`trix_mtfp.c:112` does `int32x4_t vw = {w0, w1, w2, w3};` per inner iteration. This materializes four scalar loads and a register assembly every 4 trits. Combined with the `vcgtq_s32` / `vcltq_s32` mask-then-and-then-add/sub pattern, the kernel does **more work than a dense int32 matmul would**. The whole approach is wrong for ternary. Rewrite uses the SDOT pattern from `trix_ternary_matvec_i8` but feeds MTFP activations: load packed weights via `vld1q_u8`, decode to `±1/0` via `vqtbl1q_s8` LUT, then either (a) convert to int32 lane masks and widen-accumulate MTFP into int64, or (b) restructure so the accumulator is int64 and the inner kernel uses masked add/sub pairs on int32 lanes with periodic widening. The reference has no prior art for MTFP-with-packed-trits matmul — this is new kernel territory.

**B-M3. LayerNorm `eps` hardcoded.**
`trix_mtfp.c:433` sets `int64_t eps_scaled = 35;` (chosen as `1e-5 * 59049²`). Silently baked in. Fix: parameterize `mtfp_layernorm` to accept `eps` as an `mtfp_t` (or as a raw `int64_t` pre-scaled to `S²` units, with a helper).

**B-M4. `mtfp_softmax` normalization overflow ceiling.**
`trix_mtfp.c:309` computes `((int64)di[c] * MTFP_SCALE + sum/2) / sum`. Upper bound: `di[c] ≤ exp(0) · MTFP_SCALE = 59049` (after max-subtract), so the int64 product is at most `59049 · 59049 ≈ 3.5e9` — fits comfortably. Not currently a bug, but `di[c]` could exceed this if `mtfp_exp`'s clamp is ever loosened. Add a static assert.

### 2.3 Smells

**S-M1. No NEON variant for `mtfp_vec_scale`.**
`trix_mtfp.c:246` is a scalar `mtfp_mul` loop, where `mtfp_mul` itself does an int64 multiply and rescale. Vectorizable with `vmull_s32` (32→64-bit widening multiply) + in-lane rescale, or by keeping the accumulator in int64x2 and narrowing at the end. Performance item.

**S-M2. `mtfp_bias_add` does not fuse.**
`trix_mtfp.c:324` loops `mtfp_vec_add_inplace` per batch row. Fuseable into a single NEON pass that reads the bias vector once. Minor.

**S-M3. No backward pass anywhere in `trix_mtfp.c`.**
LayerNorm, matmul, softmax, GELU — all forward-only. This is acceptable if glyph starts inference-only, but it must be a conscious decision. See §4 below.

**S-M4. `mtfp_softmax_init` / `mtfp_gelu_init` race on first call.**
Both check `if (table) return;` then `malloc`. Not thread-safe at init. N/A once LUTs become compile-time `const` arrays (V-M3 resolution).

**S-M5. `mtfp_from_float_batch` writes a NEON scratch value via `vcvtnq_s32_f32` but does not clamp to `MTFP_MAX_VAL`.**
The scalar `mtfp_from_float` clamps; the NEON batch path does not. Silent divergence on saturating inputs. Either add a NEON `vminq/vmaxq` clamp or drop the NEON path. N/A if boundary conversions get quarantined (V-M2).

---

## Part 3 — What Must Be Rewritten

Glyph's initial `src/` will not port any reference file verbatim. These are the rewrites needed before the first transformer block can run:

### 3.1 `src/glyph_mtfp.h` / `src/glyph_mtfp.c`

The MTFP core. Draws from `trix_mtfp.c` but excludes all float paths and rewrites the ternary matmul.

Must contain:
- `glyph_mtfp_t` typedef (`int32_t` container, `GLYPH_MTFP_SCALE = 59049`, `GLYPH_MTFP_RADIX = 10`).
- Inline scalar arithmetic: `mtfp_add/sub/neg/mul/mul_ternary/scale`.
- Vector ops: `mtfp_vec_add/add_inplace/scale` — NEON int32.
- Integer `isqrt64` + `mtfp_isqrt_inv` (Newton-Raphson, 8 iters).
- LayerNorm with parameterized `eps` (fix B-M3).
- Fan-in normalization.
- Softmax over MTFP with LUT-backed `mtfp_exp` — LUT is a compile-time `const` array (V-M3).
- GELU over MTFP with LUT-backed `mtfp_gelu` — same resolution.
- Bias add.
- `mtfp_matmul` / `mtfp_matmul_bt` (MTFP × MTFP, int64 accumulator).

Must NOT contain:
- Any `float` or `double`.
- Any `int8_t`/`uint8_t` as numeric operands (only as packed-trit containers, and only in dedicated trit-pack modules).
- Any boundary conversion function (those live in `tools/`).

### 3.2 `src/glyph_trit_pack.h` / `src/glyph_trit_pack.c`

Unified trit packing layer. Consolidates `trix_pack_ternary` and `trix_ternary_pack_weights_i8`.

Must contain:
- `glyph_pack_trits_1d(uint8_t* dst, const int8_t* src, int n)` — signature / flat vector packing. (The `int8_t*` input here is a **container** holding per-trit values `{-1, 0, +1}`; it is not a numeric operand. Document this at the call site.)
- `glyph_pack_trits_rowmajor(uint8_t* dst, const int8_t* src, int M, int K)` — weight-matrix packing, row-major with `Kp = (K + 3) / 4`.
- `glyph_popcount_dist(const uint8_t* a, const uint8_t* b, const uint8_t* mask, int packed_dim)` — XOR + AND + VCNT ladder, NEON 16-at-a-time.
- `glyph_trit_decode_lut` — the `{0, +1, -1, 0}` table used by `vqtbl1q_s8`, shared with the matmul kernel.

Decision to resolve: should the pack functions take `int8_t*` or a new `glyph_trit_t` typedef (e.g. `typedef int8_t glyph_trit_t` with a compile-time assertion that values are in `{-1, 0, +1}`)? The typedef makes the intent explicit and stops casual confusion with int8 numerics. Recommendation: **introduce `glyph_trit_t`** in `glyph_trit_pack.h` and use it everywhere unpacked trits appear.

### 3.3 `src/glyph_ternary_matmul.h` / `src/glyph_ternary_matmul.c`

The new kernel that neither `trix_mtfp.c` nor `trix_ternary_matvec.c` currently provides: MTFP activations × 2-bit-packed ternary weights → MTFP output.

Design:
- **Input:** MTFP activations `mtfp_t* X [M, K]`; packed weights `uint8_t* W_packed` with layout `[N, Kp]` (row-major, `Kp = K/4`).
- **Output:** MTFP `mtfp_t* Y [M, N]`.
- **Inner kernel:** for each `(i, j)`, decode a block of 4 trits from `W_packed[j]`, load 4 MTFP cells from `X[i]`, dispatch via masked add/sub into an int64 accumulator. Periodic narrowing / saturation check.
- **Vectorization:** operate on 16 trits / 16 MTFP cells per iteration. Use `vld1q_u8` + shift/mask + `vqtbl1q_s8` to produce sign masks, then use `vaddq_s32` / `vsubq_s32` on conditionally negated int32 lanes, accumulating into two int64x2 lanes via `vaddw_s32`.
- **Parallelism:** `dispatch_apply` over the outer `M` loop on Apple, both for `_bt` and non-`_bt` forms. Fixes B-M1.

This is net new code. Reference files provide idioms (SDOT pattern, VLD4 trit deinterleave, LUT via `vqtbl1q_s8`) but the end-to-end kernel does not exist.

### 3.4 `tools/mtfp_lut_gen.c` (host-side, not linked into `libglyph`)

One-off program that runs on a dev machine, computes the GELU and exp LUTs as `mtfp_t` integer arrays using `tanhf` / `expf`, and emits:

```c
/* Auto-generated by tools/mtfp_lut_gen.c — do not hand-edit. */
#include "glyph_mtfp.h"
const mtfp_t glyph_gelu_table[GLYPH_GELU_TABLE_SIZE] = { ... };
const mtfp_t glyph_exp_table[GLYPH_EXP_TABLE_SIZE] = { ... };
```

Output is committed to the repo as `src/glyph_mtfp_tables.c`. The library links only against the generated `.c` file; the generator is not part of the library build. Resolves V-M3, S-M4, and removes `mtfp_softmax_init` / `mtfp_gelu_init` entirely.

### 3.5 `tools/mtfp_io.c` (host-side)

Boundary conversion CLI. Reads a float32 weight file (e.g. from an existing PyTorch export), clamps and quantizes to MTFP cells, writes a glyph binary blob with a small header (magic, version, dtype=`MTFP10`, shape). Mirror program reads glyph outputs and prints float-decoded values for debugging. Not linked into `libglyph`. Resolves V-M2.

---

## Part 4 — Explicit Open Decisions

Before writing any `glyph/src/` code, these need user sign-off. Each blocks the design of the files above.

**D1. Float boundary posture.** (a) delete / (b) quarantine in tools / (c) ifdef. **Recommendation: (b).**

**D2. LUT generation.** (a) compile-time `const` arrays from a host generator / (b) quarantine runtime init in tools. **Recommendation: (a).**

**D3. Explicit `glyph_trit_t` typedef.** Introduce a dedicated type for unpacked trits so that `int8_t` never appears as a numeric operand in glyph source, only as a container for a value known to be in `{-1, 0, +1}`. **Recommendation: yes.**

**D4. Training in scope?** `trix_mtfp.c` is forward-only. If glyph needs training (MTFP backward pass, MTFP AdamW, MTFP gradient accumulation), that is a substantial rewrite beyond the reference and must be declared up front. **Recommendation: inference-only for v0. Training lands after the forward path is validated end-to-end.**

**D5. Parallelism model.** Reference uses `dispatch_apply` (libdispatch). Glyph can (a) keep libdispatch, (b) move to pthreads for portability, (c) single-threaded for v0. **Recommendation: keep libdispatch on Apple, single-thread fallback. We target M4 and only M4.**

---

## Part 5 — Order of Operations

1. Resolve D1–D5.
2. Write `src/glyph_mtfp.h` (types + inlines + function decls, no float anywhere).
3. Write `tools/mtfp_lut_gen.c` + generate `src/glyph_mtfp_tables.c`.
4. Write `src/glyph_mtfp.c` implementing vector ops, LayerNorm, softmax, GELU, fan-in, MTFP×MTFP matmul.
5. Write `src/glyph_trit_pack.h/.c` with `glyph_trit_t`, pack/popcount helpers.
6. Write `src/glyph_ternary_matmul.h/.c` (the new MTFP-×-packed-trit kernel).
7. Stand up a test harness (pure C, no frameworks) that validates each primitive against a golden reference computed in a throwaway script.
8. Only then begin wiring routed FFN / routed projection / transformer block.

Steps 2–6 are independent of each other once the header in step 2 is fixed, and can be cut as separate commits.

---

## Addendum — Red-Team Findings and Locked Decisions

### Red-team corrections to the body of this plan

- **R1.** "SDOT pattern" in §3.3 was sloppy. `vdotq_s32` is int8×int8→int32 and cannot consume MTFP int32 activations. What glyph reuses from `trix_ternary_matvec_i8` is the **trit-decode idiom** (`vld1q_u8` → shift/mask → `vqtbl1q_s8`), not the accumulator. The MTFP-activation kernel uses `vmulq_s32` on decoded sign lanes and widens to `int64` via `vaddw_s32`.
- **R2.** The "periodic narrowing / saturation check" language in §3.3 was unnecessary. MTFP cells are clamped to `±MTFP_MAX_VAL ≈ 1.07e9`; `int64` accumulates K ternary contributions for K up to ~8.6e9 without overflow. Plain `int64_t` throughout; no periodic narrowing.
- **R3.** `glyph_trit_t` is a documentation-grade `typedef int8_t glyph_trit_t`. It does not create a distinct C type and will not stop casual misuse at the compiler level. Use it consistently so the intent is readable; do not claim it enforces anything.
- **R4.** Golden references for tests must be either hand-computed constants embedded in C test source, or the output of a small committed C program. Python is banned; shell scripts with `awk` are tolerated for glue only.
- **R5.** LUT generator output (when built) is committed to the repo as `src/glyph_mtfp_tables.c`; the generator is run by hand when tables change and its output is version-controlled.
- **R6.** LUT memory footprint: `2 × (2 · 354294 + 1) × 4 bytes ≈ 5.67 MB` of `.rodata`. Acceptable on M4, worth noting.
- **R7.** Scope correction. The first implementation pass does not produce all six files in §3. It produces:
  - `src/glyph_types.h`
  - `src/glyph_mtfp.h` + `src/glyph_mtfp.c` (**without** softmax, GELU, or any LUT-backed function)
  - `src/glyph_trit_pack.h` + `src/glyph_trit_pack.c`
  - `src/glyph_ternary_matmul.h` + `src/glyph_ternary_matmul.c`
  - `CMakeLists.txt`
  - `tests/test_glyph_mtfp_smoke.c`
  Deferred to the next pass: softmax, GELU, `tools/mtfp_lut_gen.c`, `src/glyph_mtfp_tables.c`, `tools/mtfp_io.c`.
- **R8.** LayerNorm `eps` parameterization: caller passes `glyph_mtfp_t eps` in MTFP units representing `eps_real · SCALE`. The function squares it internally (`int64_t eps_var = (int64_t)eps * (int64_t)eps / MTFP_SCALE`) before comparing against variance. No pre-scaling on the caller side.
- **R9.** Trit decode must sign-extend. `vqtbl1q_s8` produces `int8` values in `{-1, 0, +1}`; widening to int32 uses `vmovl_s8` then `vmovl_s16`, never the unsigned variants. Every call site must flag this explicitly.
- **R10.** The "30× over float32" headline from trix-z README applies to the int8-SDOT routing distance kernel, not to MTFP matmul. Glyph's MTFP-packed-trit matmul will be well below that ceiling. Do not import the marketing.
- **R11.** `dispatch_apply` is Apple-only. Non-Apple builds fail at CMake configure time with a clear error. Revisited only if glyph ever needs another target.
- **R12.** Introduce `src/glyph_types.h` at the bottom of the dependency tree. It holds `glyph_mtfp_t`, `glyph_trit_t`, and the scale/radix/max constants. Every other glyph header includes it.

### Locked decisions

| ID | Decision |
|---|---|
| **D1** | Float boundary conversions: **quarantined to `tools/`**, not linked into `libglyph`. Tool itself deferred. |
| **D2** | LUTs: **committed compile-time `const` arrays** produced by a host-side generator. Generator and tables deferred to the follow-up pass. |
| **D3** | **`glyph_trit_t` introduced** as documentation-grade typedef. |
| **D4** | **Inference-only for v0.** No backward pass anywhere. |
| **D5** | **libdispatch on Apple, CMake hard-error on non-Apple.** M4 only. |
