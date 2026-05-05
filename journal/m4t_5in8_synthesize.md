# SYNTHESIZE: 5-in-8 base-3 packing in libm4t (Item 2)

Per `journal/p0_concern2_l2.md` forward pointer + the post-self-review production-shoring list. Brings the audit-validated sub-2-bit base-3 packing (1.6 bits/cell) into the substrate as an opt-in dense storage format with companion matmul kernel.

This cycle includes a substrate spec amendment (per project rule 7).

## Decision

**Add a 5-trits-in-8-bits packing format to libm4t. Three deliverables:**

1. **Spec amendment** (`m4t/docs/M4T_SUBSTRATE.md`): new §20 documenting the packing format, encoding, and the decode rule.
2. **Pack/unpack primitives** in `m4t/src/m4t_trit_pack.{h,c}`:
   - `m4t_pack_trits_5in8_1d(uint8_t* dst, const m4t_trit_t* src, int n)`
   - `m4t_unpack_trits_5in8_1d(m4t_trit_t* dst, const uint8_t* src, int n)`
   - `M4T_TRIT_PACKED5_BYTES(n)` macro = `((n + 4) / 5)`
3. **Matmul kernel** (`m4t/src/m4t_ternary_matmul.{h,c}`):
   - `m4t_ternary_5in8_matmul_bt(int32_t* Y, const m4t_trit_t* X, const uint8_t* W_packed, int M, int K, int N)` — ternary X (8 bits/cell) × 5-in-8-packed W → MTFP19 Y.
   - `m4t_ternary_5in8_matmul_bt_scalar_ref(...)` — test oracle (per project pattern; allowed by no-scalar rule).
4. **ctest binary** (`m4t/tests/test_m4t_ternary_5in8_matmul.c`):
   - Hand-derived golden values for small cases.
   - Property test: random workloads × 100+ samples, NEON kernel vs scalar_ref bit-exact.
   - Edge cases: K=0, M=0, N=0, K not multiple of 80 (assertion fires).
   - Aliasing assertions per substrate pattern.

## Encoding (formal)

Per audit Path D's verified format:
- Trit-to-unsigned: `-1 → 2`, `0 → 0`, `+1 → 1`.
- Byte = `u_0 + 3·u_1 + 9·u_2 + 27·u_3 + 81·u_4` where `u_i ∈ {0, 1, 2}`.
- Byte range: `[0, 242]`. Codes `[243, 255]` are unused/reserved.
- 5 trits per byte; `((n + 4) / 5)` bytes for n trits.
- Decode: `u_i = (byte / 3^i) mod 3`; `trit_value(u) = {0→0, 1→+1, 2→-1}`.

## NEON decode (for matmul kernel)

Per audit Path D's verified pattern (post-P0-2 split-LUT):
- 1× div-by-9 magic-multiply: `high = (b * 57) >> 9`, `low = b - 9·high`.
- `digit_0 = low % 3` via vqtbl1q_s8 with 16-byte `LUT_LOW_DIGIT0`.
- `digit_1 = low / 3` via vqtbl1q_s8 with 16-byte `LUT_LOW_DIGIT1`.
- `digit_2 = high % 3` via vqtbl2q_s8 with 32-byte `LUT_HIGH_DIGIT2`.
- `digit_3 = (high / 3) % 3` via vqtbl2q_s8 with 32-byte `LUT_HIGH_DIGIT3`.
- `digit_4 = high / 9` via vqtbl2q_s8 with 32-byte `LUT_HIGH_DIGIT4`.
- 5 SDOT calls per outer block (80 trits) against pre-permuted X.
- Tile-by-4 j-cells per outer iter (matches Item 1 pattern).

## Pre-committed gates

### G1 — Spec amendment lands cleanly

- M4T_SUBSTRATE.md gains a new §20 "Sub-2-bit base-3 packing (5-in-8 opt-in)" describing:
  - Encoding (trit-to-unsigned + positional 3^k weighting).
  - Byte range and decode rule.
  - Why this exists (density floor; B2-B can't reach below 2 b/c).
  - When to use it (storage-bandwidth-bound consumers; opt-in, not the default).
  - Cross-reference to the kernel and pack primitives.
- §17 cross-reference table updated to point to new §20.
- Per substrate amendment discipline, this cycle records the decision with traceability.

### G2 — Bit-exact verification (HARD)

- Property test in test_m4t_ternary_5in8_matmul.c: 100+ random samples per K configuration; NEON kernel vs scalar_ref bit-exact for every sample.
- K configurations covered: {80, 320, 1280, 12800} (multiples of 80 — kernel requires K%80==0).
- Cross-check vs audit's Path D kernel at one K (informational; both should produce the same Y for the same inputs).
- All 20+ existing ctest binaries continue to pass.

### G3 — Pack/unpack roundtrip

- `unpack(pack(x)) == x` for random ternary arrays of varying lengths (multiples of 5 + leftover).
- Hand-derived golden values for small cases (n=5 — single byte, n=10 — two bytes, etc.).

### G4 — Aliasing assertions

- Both pack and matmul functions assert `Y != X`, `Y != W` per substrate pattern.

### G5 — No scalar fallback in production paths

- The matmul kernel is NEON-only (assert K%80==0 — no scalar tail per audit pattern).
- Pack/unpack helpers can use scalar (they're storage-format conversions, not hot-path math; consistent with existing `m4t_pack_trits_1d` which is also scalar).
- The `_scalar_ref` test oracle exists for verification ONLY; production code MUST NOT call it (per existing project pattern + comment markers).

### G6 — Audit cross-check (informational, ~1 cell)

After the libm4t kernel exists, modify `audit/tristate_strong_bench.c` to call `m4t_ternary_5in8_matmul_bt` as an external grounding kernel (parallel to current `m4t_ternary_dot_matmul_bt` cross-check). All 80 runs should show bit-exact Y match.

This is the discipline: any new substrate kernel earns its place by being verified against the externally-validated audit kernel.

## Implementation plan

1. Write spec amendment text for §20.
2. Write pack/unpack helpers + `M4T_TRIT_PACKED5_BYTES` macro.
3. Write matmul kernel + scalar_ref oracle.
4. Write ctest binary.
5. Build; ctest gate.
6. Audit cross-check.
7. Disasm verification (kernel inner loop matches expected NEON shape).
8. Red-team.
9. Address findings.
10. Closeout + commit + push.

## Risk register

- **R1 (HIGH):** Encoding mismatch between libm4t and audit kernels. Both must use exactly the same trit-to-unsigned mapping (`-1→2, 0→0, +1→1`) and 3^k positional weighting. Mitigation: cross-check (G6) catches mismatch immediately.
- **R2 (MEDIUM):** Scalar reference must match NEON bit-exact. Easiest source of error: compute mod-3 differently in scalar vs NEON (signedness / division semantics). Mitigation: scalar uses the same direct `u_i = (byte / 3^i) mod 3` formula as the encoding section.
- **R3 (LOW):** ctest binary discovery. Need to add to m4t/CMakeLists.txt + add_test() call.
- **R4 (LOW):** Spec amendment style mismatch with existing M4T_SUBSTRATE.md sections. Mitigation: model the new section after existing §6 Storage Layout and §10 Width Conversions.

## What this cycle is NOT

- Not productionizing a WHOLE matmul family (only ternary-X variant; MTFP19-X variant deferred).
- Not changing default packing (current 4-in-8 stays the default; 5-in-8 is opt-in).
- Not deprecating `m4t_pack_trits_1d` or related (4-in-8 stays for SDOT-friendly path).

## Status

Pre-committed. Beginning implementation next.
