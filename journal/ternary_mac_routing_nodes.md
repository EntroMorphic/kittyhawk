# NODES: ternary MAC routing

Atomic claims/findings extracted from `ternary_mac_routing_raw.md`.

## Findings (verified by code or spec)

- **N1.** `m4t_mtfp_ternary_matmul_bt`'s inner loop uses ~60 NEON ops per 16-trit block. Verified by reading `m4t/src/m4t_ternary_matmul.c` lines 82–161.
- **N2.** Of the ~60 ops: ~6 for trit decode, ~24 for mask widening (nz + neg masks per 4 quadrants of int32), ~16 for vbsl conditional negate + zero gate, ~8 for widening accumulate to int64, plus ~6 housekeeping.
- **N3.** SDOT (`vsdot_s32`) computes 16 int8×int8→int32 accumulates per instruction. Available on Apple Silicon, used by `m4t_mtfp4_sdot_matmul_bt`.
- **N4.** SDOT does NOT route Case S (int32 activations × ternary weights → int64) — operand width mismatch.
- **N5.** vmlal_s32 (`vmlal_s32`) computes 2 int32×int32→int64 widening multiply-accumulates per instruction. Available on Apple Silicon NEON.
- **N6.** vmla_s32 (no widening) cannot route the case — even one 16-element block sum is bounded by 16 × MAX_VAL ≈ 9.3 × 10⁹, exceeding INT32_MAX (~2.1 × 10⁹).
- **N7.** Apple Silicon does not expose SVE/SVE2 at user mode.
- **N8.** The substrate already has `m4t_mtfp4_sdot_matmul_bt` (Case W: int8 activations × ternary weights → int32, SDOT-direct, ~1 NEON op per 16 elements).
- **N9.** SMMLA (Armv8.6 int8 matrix multiply) exists on M3+; even where available, doesn't route int32 activations.

## Estimates (theoretical, not measured)

- **N10.** vmlal_s32-routed ternary MAC: ~6 (decode) + ~4 (sign-extend int8→int32) + 8 (vmlal_s32) = **~18 ops per 16 trits**.
- **N11.** Estimated kernel-level speedup of vmlal_s32 path vs current: ~60 / ~18 = **~3.3×**.
- **N12.** Apple Silicon vmlal_s32 throughput is unknown to me; assumed ≥ 1/cycle on M-series. If lower, speedup estimate degrades proportionally.

## Strategic observations

- **N13.** A "ternary MAC" custom-silicon op (int32 × {-1,0,+1} → int64 in 1 cycle) does not exist on M4. Closest analogs: SDOT (int8 only) and vmlal_s32 (int32 but only 2 lanes per instruction).
- **N14.** Multiplication by trit ∈ {-1, 0, +1} subsumes both conditional-negate AND zero-gate. The current vbsl pattern's mask-widening cost (~24 ops) goes away entirely if we use vmlal_s32, because multiply naturally handles zero.
- **N15.** Case W via MTFP4 activations bypasses the int32 problem entirely (SDOT routes it natively). Whether applicable depends on consumer activation precision.

## Concerns (carried forward from prior cycles)

- **N16.** No audit of whether any current substrate consumer calls `m4t_mtfp_ternary_matmul_bt`. Prior shift3 NEON cycle's lesson: optimization without consumer-demand evidence produces kernel wins that don't propagate.
- **N17.** Per shift3 remediation methodology rule: if we productionize the vmlal_s32 path, must expose the current implementation as a separately-preserved scalar reference (`m4t_mtfp_ternary_matmul_bt_scalar_ref` or similar) to maintain bit-exact verification post-productionization.
- **N18.** Per CONTRIBUTING.md scope-match rule (added in concern #4 sweep): any speedup claim must name workload shape. Default bench would be batched-matmul shape; tight-loop or sparse shapes may differ.
- **N19.** vmlal_s32 dependency chains across the 8 calls per block could limit per-cycle throughput below the inverse-throughput. Not characterized.

## Open questions

- **N20.** Which substrate consumers (if any) actually call `m4t_mtfp_ternary_matmul_bt`?
- **N21.** Of those, what fraction of consumer time is spent in this kernel?
- **N22.** Of those, which can move to Case W (MTFP4 activations) and which are structurally fixed at int32?
- **N23.** What is Apple Silicon's actual vmlal_s32 throughput on M4 (or whatever target machine)?
- **N24.** What's the realistic kernel-level speedup once dependency chains and decode overhead are accounted for empirically?
- **N25.** Is the right move (a) prototype-then-measure-consumer-impact, (b) audit-then-decide, (c) document-and-defer?

## Methodology constraints

- **N26.** CONTRIBUTING.md non-negotiable #5 ("no primitive without named consumer demand") technically applies to NEW primitives, not optimizations of existing ones. But the recent shift3 finding shows the same pattern produces kernel wins that don't propagate.
- **N27.** Same workload-shape rule applies (CONTRIBUTING scope-match audit).

## What we're NOT doing this cycle

- **N28.** Not pursuing custom silicon (out of scope per user framing).
- **N29.** Not pursuing Case W migration (different decision; substrate-design-level).
- **N30.** Not modifying `m4t_mtfp4_sdot_matmul_bt` (already optimal).

## Reframing during this cycle

- **N31.** Original user framing was "30 NEON ops per 16-trit block." Verified actual count is ~60. Doesn't change the qualitative story (still much slower than custom silicon would be) but the headline number is corrected.
- **N32.** Original user framing was "trit-decode is software-emulated for lack of a trit-native instruction." Refined: the trit DECODE itself is tight (~6 ops); the multiply-by-{-1,0,+1} pattern is what's expensive (~40 ops via mask+select). The "missing instruction" is more accurately "fused signed-gated MAC" than "trit decode."
