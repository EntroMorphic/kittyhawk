# RAW: NEON magic-multiply divide-by-3^k prototype

Stream of consciousness on the shift3 NEON prototype results. No structure, no filtering.

## What happened

User proposed an `m4t_div_3pk_neon` kernel using `vqrdmulhq_s32 + vshlq_s32` as a substitute for custom silicon for shift3 (the #3 elemental-floor op). I prototyped it. Took two iteration cycles to get bit-exactness; the original two-stage rounding pipeline failed and I pivoted to a 64-bit-intermediate approach. The pivoted prototype is bit-exact for all 19 k values across the full substrate input range, with ~10× speedup over the substrate's existing scalar divide.

## What I observed first cycle (vqrdmulh + vrshl pipeline)

Tried the user's exact proposal first. Built a generator that derives `M = round(2^(31+s) / d)` and post-shifts by `s`. Built an emulator. Ran exhaustive verify across [-MAX_VAL, +MAX_VAL].

First emulator had a bug: missed the "doubling" in vqrdmulh's name. After fixing, results were off by ~2× — straight up wrong factor. Found it by inspection.

Second emulator pass: closer. But still mismatches for negative-x values where ref rounded away-from-zero and NEON rounded toward-zero. Diagnosed as compound rounding: vqrdmulh rounds-to-nearest, then vshlq does plain arithmetic right shift which floors. Two different rounding semantics in series. Fix: use `vrshlq_s32` (rounding shift left with negative count = rounding shift right) instead of `vshlq_s32`.

Third pass with vrshl: dramatic improvement, mismatches dropped from ~387M to a small handful per k (e.g., 1 mismatch for k=19, 9 mismatches for k=17). But not bit-exact. Bug: even with two correctly-rounding stages, the compound error can be up to 1.0 (each stage can contribute 0.5).

Tried brute-force search around the theoretical M with ±8 delta. Smart-set verifier had a bug where step=d/8 became 1 for small d, causing the test set to balloon to 8e9 points. Killed it.

## What I observed second cycle (vmull + bias + shift pipeline)

Pivoted to 64-bit intermediate. New pipeline:

```
prod_64 = (int64_t)x * M
adj_64 = prod_64 + (1 << (N-1))   // round-half-up bias
result = (int32_t)(adj_64 >> N)    // arith shift, narrow
```

ONE rounding step end-to-end. M = round(2^N / d). The key insight: with int64 intermediate we can use much larger N (more precision) without the constraint that M fits in 32 bits. M still fits in int32 (so vmull_s32 with broadcast works), but N ranges up to 61 instead of being bounded by 31+log2(d) and limited by quantization error compounding.

Wrote a generator that searches (M, N) per k. Two-stage: smart-set with ~700K samples winnows candidates, then exhaustive 1.16e9 verify on smart-set winners. First run: capped N_max wrong (took 31+13 instead of 31+17), failed for k≥11. Fixed N_max computation. Re-ran: 19/19 bit-exact in 23 seconds.

Generated table:
- k=1: M=1431655765 N=32
- k=10: M=1191700861 N=46
- k=19: M=1983927949 N=61

All M values are big (~1B-2B), all N values straddle 32-61. Pattern: N grows roughly linearly with k.

## NEON kernel correctness

Wrote `m4t_shift3_div_neon` using the constants. Pipeline per 4 lanes:
- `vmull_s32(low_half, M_broadcast)` → int64x2
- `vmull_s32(high_half, M_broadcast)` → int64x2
- `vaddq_s64(prod, bias)` (bias = 2^(N-1))
- `vshlq_s64(adj, neg_N)` (variable right shift via negative shift count)
- `vmovn_s64` × 2 + `vcombine_s32` for narrow-back-to-int32x4

Wrote a property test against the substrate's existing `m4t_mtfp_shift3` for k ∈ [1, 19]. Test set: 12 corners + 50K random per k. All 19 k values bit-exact.

## NEON kernel perf

Bench: n=4096, 200 iters, scalar vs NEON. Results:
- Scalar: 1.67 ns/elem (~5.8 cycles at 3.5 GHz)
- NEON: 0.18 ns/elem (~0.63 cycles)
- Speedup: ~10×

Lower than my original ~40× estimate. The estimate assumed scalar used hardware sdiv (~12 cycles); actual scalar uses the substrate's mul-based `m4t_pow3_round_div` (~5.8 cycles). Real ceiling for this kernel against a sdiv-based scalar would be ~30×.

## Things I'm uncertain about

The 10× kernel-level speedup may or may not translate to anything at the consumer level. I haven't checked if shift3 is on any current consumer's hot path. The cross-exponent accumulator (`m4t_mtfp_vec_accum_aligning`) does base-3 round-divides per cell, but it does them WITH PER-CELL VARYING K (each cell has its own alignment exponent). My kernel takes one k for a whole batch — wrong shape for that consumer.

Multi-cell same-k consumers: image_canon's normalize step? It rescales by `SCALE/sd`, not by 3^k. So no obvious user.

Honest concern: **the kernel might be 10× faster on a workload nobody runs.** Same shape as the V4-residual-3 finding (LTO 3× on pipelined workloads — real but no current consumer).

## Things I haven't proved

- Saturation: vmovn truncates int64 → int32. The result is supposed to fit in int32 by construction (≈ x/d, |x| ≤ MAX_VAL, d ≥ 3 → |result| ≤ MAX_VAL/3 < int32). Should be true but I haven't formally proved it covers ALL x in the input range.
- Aliasing: I read 4 lanes via `vld1q_s32` then write via `vst1q_s32`, all within one inner loop iteration. For dst==src this should work because the load is fully consumed before the store. But I didn't write a test that exercises aliasing.
- Out-of-range k handling: kernel assumes abs_k ∈ [1, 19]. The substrate's scalar handles abs_k ≥ 20 (memset to zero). My kernel doesn't have this guard. If I productionize, the wrapper has to handle it.
- The original vqrdmulh proposal might still be salvageable with per-k specialization (constant-shift vrshrq_n_s32 instead of variable vrshlq_s32). Constant shift would give predictable rounding direction. Could be ~1.5–2× faster than 64-bit path. Didn't pursue because the 64-bit path works and 10× is already compelling.

## Things I touched but didn't carefully think about

The generator's smart-set sampling was buggy in two ways before I got to a working version. The exhaustive verify is slow (~5s per call, 19 calls = 90s). The combination is fine for one-time use but if this becomes a build step the runtime matters.

The magic table currently lives in two places: gen_pow3_magic.c (canonical) and test_m4t_shift3_neon_proto.c (copy for the test). For productionization, one needs to be the source of truth.

## What this is for

The user framed this as "substitute for custom silicon." Apple Silicon doesn't have a divide-by-3 instruction; magic-multiply gives most of the benefit. The 10× over the substrate's already-optimized scalar is the measure of how much the approach reclaims.

## What I would do next if running on instinct

Productionize: replace the divide-direction loop in m4t_mtfp_shift3 with the NEON kernel, gate by M4T_HAS_NEON, keep the scalar path as the bit-exact reference. Add the property test as a ctest entry. Generate the constants at build time so they don't drift.

Followups: investigate the cross-exponent accumulator's per-cell-varying k — could a vector-of-Ms approach (loading per-cell M and N from a small table and using NEON gather-style addressing) give a similar speedup there? If yes, that's a more impactful productionization site.
