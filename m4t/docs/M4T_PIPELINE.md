# M4T Implementation Pipeline

Sequenced by dependency. Each item gates the ones below it where noted.

Status markers: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked

---

## Phase A — Opcode primitives

- [ ] **1. TBL-based trit ops (`m4t_trit_ops[]`)** — 6 new opcodes: `mul`, `sat_add`, `max`, `min`, `eq`, `neg`. Each is a 16-byte LUT + a ~5-instruction NEON stub. Uniform shape. New files: `src/m4t_trit_ops.{h,c}`, tests in `tests/test_m4t_trit_ops.c`. Exhaustive 3×3 = 9-entry verification per LUT.

- [ ] **2. Masked-VCNT reducers** — `m4t_trit_signed_sum` and `m4t_trit_sparsity`. Masked popcount → signed trit reduction. New files: `src/m4t_trit_reducers.{h,c}`, tests. Dependency: used by item 3 (signature update).

- [ ] **3. Routing primitives (`m4t_route_ops[]`)** — 5 new opcodes: `signature_update`, `distance_batch`, `topk_abs`, `apply_signed`, `sign_extract`. New files: `src/m4t_route.{h,c}`, tests. Depends on items 1 and 2.

## Phase B — Cell-width expansion

- [ ] **4. MTFP39 wide path (`m4t_mtfp_w_*`)** — int64 variants of all MTFP19 ops (vec_add, matmul, layernorm, etc.). New files: `src/m4t_mtfp_w.{h,c}`, tests. Independent of Phase A.

- [ ] **5. MTFP4 SDOT path (`m4t_mtfp4_*`)** — int8 routing cell. `vdotq_s32` as native ternary matmul. New files: `src/m4t_mtfp4.{h,c}`, tests. Independent of Phase A; routing primitives (item 3) should consume this once both land.

## Phase C — Opcode tables and tooling

- [ ] **6. Function-pointer opcode tables** — `m4t_trit_ops[]`, `m4t_mtfp_ops[]`, `m4t_route_ops[]` as extern const arrays of function pointers. Enum indices. New files: `src/m4t_ops.{h,c}`. Depends on items 1–5 for population.

- [x] **7. Size-check and cycle-count tools** — `tools/m4t_size_check.sh` (M1: link-time `.text` budget enforcement, wired into CMake post-build), `tools/m4t_bench.c` (M2: per-opcode cycle counting via `mach_absolute_time`), `tools/m4t_lut_gen.c` (host-side GELU+exp table generator), `tools/m4t_trit_golden.c` (exhaustive truth tables for TBL opcodes). All four built and verified.

## Phase D — Consumer integration

- [ ] **8. Glyph wrapper layer** — thin `glyph/src/glyph_*.h` headers that `#include <m4t_*.h>` and typedef/alias the m4t types. Glyph's `CMakeLists.txt` updated to depend on `libm4t.a`. Existing glyph tests optionally kept as a compatibility check. Independent of Phase C.

---

## Budget tracking

| Region | Budget | Current (.text) | Headroom |
|---|---|---|---|
| L1i (opcode bodies) | 24 KB | 8.4 KB (MTFP19 core only) | 15.6 KB |
| L1d (LUTs + constants) | 4 KB | ~0.1 KB (decode LUT only) | 3.9 KB |

Updated as new opcodes land.
