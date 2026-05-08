---
cycle: K%80 fix — closeout
phase: build + test + bench + verify
date: 2026-05-07
scope: execution of the K%80 fix per journal/k80_fix_lmm.md.
       Patched m4t_ternary_5in8_matmul_bt to extend the NEON tile
       body to K_padded with stack-local zero-padded W for the
       boundary tile. Removed the geometric scalar tail.
companions: journal/k80_fix_lmm.md (the design),
            journal/k80_bench_BEFORE.txt (pre-patch timings),
            journal/k80_bench_AFTER.txt (post-patch timings),
            journal/k80_rowskip_after_v2.txt (rowskip after patch).
---

# K%80 fix — closeout

## What was done

Per the LMM plan:

1. **Patched `m4t_ternary_5in8_matmul_bt`** (m4t/src/m4t_ternary_matmul.c):
   - Replaced `k_tile_end = K - K%80` (unused outside fast-path
     bound) with `K_aligned = K - K%80` and added
     `K_padded = ceil(K/80) * 80`.
   - Extended X_strided allocation to `K5_padded * 5` slots; the
     pre-permute loop's `trit_idx < K` check zero-fills past K.
   - Main NEON tile body runs `k = 0..K_aligned` (unchanged when
     K%80 == 0).
   - Added boundary tile path (only when K%80 != 0): processes
     [K_aligned, K_padded) using a 16-byte stack-local W buffer per
     j_cell, populated by memcpy from W_packed with zero-fill past
     Kp. Bit-exact because zero W trits contribute 0 to the dot.
   - Same restructure for the j_tail (single-output) loop.
   - Deleted both geometric scalar tails.

2. **Extended bit-exact tests** (test_m4t_ternary_5in8_matmul.c):
   - Full K%80 sweep ∈ {1..79} at K = 160 + km, 3 random samples each
   - K<80 sweep at K ∈ {1, 5, 17, 33, 40, 64, 79} with both single-
     output and multi-output configs

3. **Benched** before/after on representative shapes.

## Bit-exactness

All tests pass:
  - test_m4t_ternary_5in8_matmul (incl. K%80 sweep + K<80): PASS
  - test_m4t_ternary_matmul_neon: PASS
  - test_m4t_ternary_5in8_xpacked: PASS (uses unrelated kernel)
  - test_m4t_ternary_routed: PASS
  - test_m4t_ternary_routed16: PASS
  - test_m4t_ternary_rowskip: PASS

ASAN+UBSAN clean on test_m4t_ternary_5in8_matmul,
test_m4t_ternary_rowskip, test_m4t_ternary_routed16. (Pre-existing
UB elsewhere in the codebase — m4t_mtfp_ternary_matmul_bt at line
380 — is in a different function and not introduced by this patch.)

## Performance — direct timing

n_iter = 200, mean ± stddev (ms):

  Shape                                    | BEFORE    | AFTER     | Speedup
  -----------------------------------------|-----------|-----------|---------
  K=2560 N=2560  (q/o, K%80=0)             | 0.090     | 0.089     | unchanged ✓
  K=2560 N=6912  (gate/up output, K%80=0)  | 0.208     | 0.212     | unchanged
  K=6912 N=2560  (down_proj, K%80=32)      | 0.264     | 0.212     | +24.5%
  K=2560 N=640   (k/v, K%80=0)             | 0.020     | 0.020     | unchanged ✓
  K=2400 (K%80=0)                          | 0.073     | 0.073     | unchanged ✓
  K=2401 (K%80=1)                          | 0.072     | 0.076     | -5%
  K=2440 (K%80=40)                         | 0.146     | 0.077     | +89%
  K=2479 (K%80=79)                         | 0.223     | 0.077     | +189%
  K=2480 (K%80=0)                          | 0.075     | 0.076     | unchanged
  K=80   (K%80=0)                          | 0.003     | 0.003     | unchanged ✓
  K=40   (K<80)                            | 0.080     | 0.007     | +1043%
  K=1                                      | 0.002     | 0.006     | -67%

Reading:
- **K%80=0 cases unchanged** (within noise). The fast path is
  preserved by the conditional boundary tile.
- **K%80!=0 cases collapse to ~K%80=0 baseline cost**. The scalar
  tail is gone. K=2479 (worst case before, 0.223 ms) is now 0.077 ms
  — same as K=2480 (best case).
- **Down_proj (K=6912 K%80=32)** wins 24.5% — the headline BitNet
  number. Matches the LMM prediction (+18-22%) and slightly exceeds.
- **K=40 wins 11.4×** — the boundary-tile-only path is way faster
  than the old all-scalar path for K<80.
- **K=1 regresses** from 0.002 to 0.006 ms (3× slower in absolute
  terms; tiny in any practical sense). The boundary tile pays NEON
  setup cost where the old scalar-only path was free of it. K=1
  is not a realistic BitLinear shape; not worth special-casing.
- **K=2401 (K%80=1) regresses 5%**: same reason — boundary tile
  fires for 1 real trit + 79 zero trits. Within bench noise; would
  matter only at extreme tail densities.

## Performance — BitNet aggregate (rowskip v2 bench)

The same v2 bench used during the rowskip cycle, run on real BitNet
weights, post-patch:

  Aggregate across 210 calls (30 layers × 7 BitLinears):

                       | BEFORE patch | AFTER patch
  -------------------- | ------------ | ------------
  dense baseline       |   27.98 ms   |   26.24 ms    (-6.2%)
  rs_no_skip vs dense  |   +4.90%     |   -0.80%      (tile-align gone)
  rs (always-on)       |   +6.12%     |   +0.05%
  smart-dispatch (≥5%) |   +1.55%     |   +0.91%

**Dense itself is now 6.2% faster on BitNet aggregate.** The
tile-alignment side effect that used to hide in rowskip's headline
has moved to where it benefits all callers.

Rowskip's residual value is now isolated: +0.91% smart-dispatch
benefit comes from the 4-5 BitLinears with substantial empty-row
fraction (notably L1 down_proj 43.6%, L2 down_proj 27.7%, L29
o_proj 24.0%, L0 o_proj 15.5%).

## Tensions resolution post-execution

Both LMM tensions resolved as planned:

- **T1 (API contract vs internal padding):** went with internal
  padding (option B). Stack-local W per j_cell + larger X_strided
  allocation. No contract change. Per-call cost (16 bytes × 4
  memcpy + 1 extra alloc-size) is well below the savings. Worked
  cleanly.

- **T2 (scope: main + j_tail):** patched both. Same boundary-tile
  pattern applied to the single-output loop with one stack buffer
  instead of four. Code remained tight.

## What this teaches

The bench data confirms what the rs_no_skip control variant
predicted in the rowskip cycle: ~5% aggregate, 24% on the worst
shape (K=6912). The LMM correctly identified this as a higher-value
fix than rowskip itself.

The methodical structure paid off again: the LMM's decision to use
internal padding (not caller-padded) avoided breaking other m4t
consumers; the explicit K%80 sweep test caught no bugs (good — it
means the design was sound), but its existence is a regression
guard against future kernel changes.

The K=1 regression is the only honest blemish. If it ever matters
(it doesn't for BitNet), a small `if (K == 0) return zeros` early
exit + `if (K_aligned == 0 && K < 4) scalar_tiny_fallback` could
patch it without touching the main path. Filed but not addressed —
real workloads have K large enough that this is irrelevant noise.

## Outcome

- m4t_ternary_5in8_matmul_bt is +24% faster on BitNet down_proj,
  unchanged on tile-aligned shapes, and gives +6.2% BitNet
  aggregate per-token compute.
- Rowskip kernel survives but with reduced strategic value:
  smart-dispatch yields +0.9% on top of the patched dense.
- Both kernels are correct, tested, and ASAN-clean.
- Cycle complete.
