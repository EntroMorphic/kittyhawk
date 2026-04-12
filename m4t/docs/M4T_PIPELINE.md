# M4T Implementation Pipeline

Sequenced by dependency. Each item gates the ones below it where noted.

Status markers: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked

---

## Phase A — Opcode primitives

- [x] **1. TBL-based trit ops (`m4t_trit_ops[]`)** — 6 opcodes landed: `mul`, `sat_add`, `max`, `min`, `eq`, `neg`. Five use a shared TBL kernel (~28 NEON insns/64 trits); neg uses a bit-swap (~5 insns/64 trits). 13 tests: 6 exhaustive 3×3, 6 NEON+tail (n=65), 1 in-place alias. Added 1.9 KB to .text (total 10.3 KB, 42% budget).

- [x] **2. Masked-VCNT reducers** — `m4t_trit_signed_sum`, `m4t_trit_sparsity`, and `m4t_trit_counts` (separate pos/neg counts). Masked popcount via AND(0x55/0xAA) + VCNT + widen chain. ~14 NEON insns per 64 trits. 8 tests covering zeros, all-pos, all-neg, mixed, NEON+tail (n=65), exact-block (n=256), n=0, n=1. Added 724 bytes to .text (total 11.1 KB, 45% budget).

- [x] **3. Routing primitives (`m4t_route_ops[]`)** — 5 primitives landed: `sign_extract` (int64→packed-trit signs), `distance_batch` (batch popcount over T tiles), `topk_abs` (k-of-T selection by |score|), `apply_signed` (signed tile accumulation), `signature_update` (compound: column-sum → mean-subtract → sign-extract, uses caller-provided scratch). 9 tests including a full end-to-end mini routing pass. Added 4.2 KB to .text (total 15.3 KB, 62% budget).

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
| L1i (opcode bodies) | 24 KB | 14.5 KB (MTFP19 + trit ops + reducers + route) | 9.5 KB |
| L1d (LUTs + constants) | 4 KB | ~0.3 KB (decode LUT + 5 op LUTs) | 3.7 KB |

Updated as new opcodes land.
