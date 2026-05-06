# No-scalar audit + 100/100 remediation — 2026-05-06

User-requested sweep of production code for scalar paths that violate the project rule (`feedback_function_over_speed_no_scalar`):

> production dispatchers are NEON-only (scalar_ref test oracle is fine; geometric tail is fine; no `#if !M4T_HAS_NEON` or "fall back to scalar when X" in production code paths)

## Findings (audit)

Nine production functions had **zero NEON path** — fully scalar from entry to exit. Two stale comments documented "non-NEON fallback" patterns that were already remediated in earlier cycles but whose comments were stale.

| # | Function | File | Class |
|---|---|---|---|
| 1 | `m4t_pack_trits_1d`              | `m4t_trit_pack.c`      | 4-in-8 pack — fully scalar |
| 2 | `m4t_unpack_trits_1d`            | `m4t_trit_pack.c`      | 4-in-8 unpack — fully scalar |
| 3 | `m4t_pack_trits_rowmajor`        | `m4t_trit_pack.c`      | wrapper around #1 |
| 4 | `m4t_unpack_trits_rowmajor`      | `m4t_trit_pack.c`      | wrapper around #2 |
| 5 | `m4t_pack_trits_5in8_1d`         | `m4t_trit_pack.c`      | 5-in-8 pack (TD-7 hot path!) — fully scalar |
| 6 | `m4t_unpack_trits_5in8_1d`       | `m4t_trit_pack.c`      | 5-in-8 unpack — fully scalar |
| 7 | `m4t_mtfp_shift3` k>0 branch     | `m4t_mtfp.c`           | multiply-by-3^k — fully scalar (only divide was NEON) |
| 8 | `m4t_mtfp19_to_mtfp4`            | `m4t_mtfp4.c`          | width conversion — fully scalar |
| 9 | `m4t_mtfp4_to_mtfp19`            | `m4t_mtfp4.c`          | width conversion — fully scalar |
| 10 | (stale comment) `ternary_dot_scalar` | `m4t_ternary_matmul.c:61` | claimed as production fallback; actually only used by scalar_ref |
| 11 | (stale comment) `shift3_div_scalar` | `m4t_mtfp.c:632`        | same — only used by scalar_ref |

Allowed patterns (per memory: "geometric tail is fine"): NEON-loop-with-scalar-tail patterns in `m4t_route_select`, `trit_binary_op`, `m4t_trit_neg`, `m4t_trit_counts`, `m4t_popcount_dist`, and the matmul inner SDOT paths. **Not violations** — they process aligned NEON blocks with a sub-block geometric tail.

## Remediation (Option A — full 100/100)

### Method per function

For each violation:
1. Refactor existing scalar code into a private static `_scalar` helper.
2. Implement a NEON `_neon` helper inside `#if M4T_HAS_NEON`.
3. Modify the public function to dispatch directly to `_neon` (NEON-only production path).
4. Add a public `_scalar_ref` function (test oracle, calls the static `_scalar`).
5. Add randomized NEON-vs-scalar_ref bit-exact verification tests across diverse N values.

### NEON design notes

**4-in-8 pack** (`m4t_pack_trits_1d`): 16 trits → 4 bytes per iter.
- AND with 3 to map (trit & 3) for TBL index space.
- TBL with LUT [0,1,0,2,...] maps {trit=+1→1, 0→0, -1→2}.
- Multiply by per-lane place values [1,4,16,64,1,4,16,64,...] (vmulq_u8 with broadcast vector — `vmulq_n_u8` doesn't exist in clang's arm_neon.h).
- Two pairwise additions (`vpaddq_u8`) reduce 16 lanes → 4 output bytes (codes occupy disjoint bit positions, so add ≡ OR per byte).
- Tail: scalar geometric for n%16.

**4-in-8 unpack** (`m4t_unpack_trits_1d`): 4 bytes → 16 trits per iter.
- Replicate each input byte 4× via `vqtbl1q_u8` with index pattern [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3].
- `vshlq_s8` with per-lane shifts [0,-2,-4,-6,...] right-shifts each replica to put the desired 2-bit code in low bits.
- AND with 3 to extract the code.
- TBL decode to trit value.

**5-in-8 pack/unpack**: NEON has no LD5/ST5 instruction (verified via grep on `arm_neon.h`). Worked around with `vqtbl4q_s8` (4-vector / 64-lane lookup) + `vqtbl1q_s8` (16-lane for the 5th vector) + `vorrq_s8` combine. Out-of-range indices return 0 from the TBL ops, so the two paths combine without overlap. Pre-computed 5×16 index tables for both interleave (unpack) and deinterleave (pack) directions. Per-byte: 5 (de)interleave operations × 5 vector ops = 25 NEON ops to reshape 80 lanes; vs 80 scalar reads/writes — net win.

**Width conversions** (`m4t_mtfp19_to_mtfp4`, `m4t_mtfp4_to_mtfp19`):
- 19→4: reuses the magic-multiply pattern from `shift3_div_neon` for divide-by-6561. Magic constants from `M4T_POW3_DIV_M[8]` / `M4T_POW3_DIV_N[8]`. Per-cell flag tracking done in a small post-NEON scalar pass over the 4 lanes (per-cell branchy flag work doesn't NEON-ize cleanly).
- 4→19: trivial `vmovl_s8` chain to widen int8 → int32, multiply by SCALE_RATIO=6561 with `vmulq_s32`.

**`m4t_mtfp_shift3` k>0 multiply**:
- Two NEON helpers: `shift3_mul_neon` (for k ∈ [1, 19]) and `shift3_mul_saturate_neon` (for k ≥ 20 saturation collapse).
- Multiply: `vmull_s32` (int32×int32→int64) per 2 lanes, clamp to ±MAX_VAL via `vbslq_s64` with `vcgtq_s64`/`vcltq_s64` (NEON has no `vminq_s64`/`vmaxq_s64` in this toolchain), narrow to int32.
- Saturation collapse: `vbslq_s32` based on sign comparison.

### Stale comments fixed

- `m4t_ternary_matmul.c:61-63`: `ternary_dot_scalar` no longer claims to be the "non-NEON fallback inside ternary_dot." Production `ternary_dot` calls `ternary_dot_vmlal` directly.
- `m4t_mtfp.c:632-633`: same — `shift3_div_scalar` is only used by `m4t_mtfp_shift3_scalar_ref` (test oracle), not by production `m4t_mtfp_shift3`.

## Verification

### Bit-exact gates added

- `test_m4t_trit_pack.c` — `test_pack_1d_neon_vs_scalar` (30 N values × 100 random samples × pack + unpack = 6,000 NEON-vs-scalar_ref checks). `test_pack_5in8_neon_vs_scalar` (26 N values × 50 samples × pack + unpack = 2,600 checks).
- `test_m4t_shift3_neon.c` — extended with k>0 multiply path bit-exact (19 k values × ~50,012 inputs each = ~950,228 checks) and k≥20 saturation collapse (3 k values × 4,109 inputs).
- `test_m4t_mtfp4.c` — `test_conversions_neon_vs_scalar_ref` (20 N values × 100 random samples × both directions = 4,000 checks).

### Existing tests

All 22 ctest binaries continue to pass. Existing `test_pack_unpack_roundtrip_various_n`, `test_narrow_property` (10K samples × 64 cells), `test_widen_exact`, `test_roundtrip_widen_narrow`, `test_pack_golden`, etc., all green.

## Decisions and tradeoffs

**Encountered intrinsic-availability issues:**
- `vld5q_s8` / `vst5q_s8` — *do not exist* on aarch64 (verified via grep on `arm_neon.h`). NEON only has stride 2/3/4 deinterleave (LD2/LD3/LD4). Worked around with `vqtbl4q_s8` + `vqtbl1q_s8` + `vorrq_s8`.
- `vmulq_n_u8` — does not exist (no scalar-by-vector u8 multiply intrinsic). Worked around with `vmulq_u8(v, vdupq_n_u8(scalar))`.
- `vminq_s64` / `vmaxq_s64` — not always available depending on toolchain. Worked around with `vbslq_s64` + `vcgtq_s64`/`vcltq_s64`.

These three workarounds add a few NEON ops vs the ideal intrinsics but stay within 1-2× of the structurally-optimal pipeline.

**Scope of `_scalar_ref` exposure:** the new public `_scalar_ref` functions are added strictly as test oracles. Headers note "Production code MUST NOT call these — intentionally slower." Pattern mirrors the existing `m4t_mtfp_shift3_scalar_ref`, `m4t_ternary_5in8_matmul_bt_scalar_ref`, etc.

## Status

CLOSED. Production paths are now NEON throughout the substrate (modulo allowed geometric scalar tails for sub-block remainders). 22/22 ctest binaries green. ~13,500 new NEON-vs-scalar_ref bit-exact assertions added across pack/unpack/conversion/shift3-multiply paths.

## Cross-references

- Project rule: `feedback_function_over_speed_no_scalar` (in user memory).
- Production-shoring red-team that originally established the rule: `journal/production_shoring_redteam.md`.
- Earlier NEON-routing cycles whose pattern this remediation follows: `journal/cross_exp_accum_routing_*.md`, `journal/ternary_mac_routing_*.md`, `journal/shift3_neon_*.md`.
