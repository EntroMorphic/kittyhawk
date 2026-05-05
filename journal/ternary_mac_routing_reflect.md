# REFLECT: ternary MAC routing

Cold-eye review of `ternary_mac_routing_nodes.md`. What's load-bearing, what's weak, what's contradictory, what's missing.

## Load-bearing nodes

- **N1, N2** — the cost analysis. Verified by direct code count. Strong evidence; the ~60-op estimate is grounded in the actual inner loop structure.
- **N5** — vmlal_s32 exists and has the right shape (int32×int32→int64 widening). Verified by ARM ARM. Strong.
- **N8** — Case W exists in the substrate. Verified by `m4t/src/m4t_mtfp4.c`. Strong.
- **N14** — multiply-by-trit subsumes the bsl pattern. Algebraically sound: 0×x = 0, ±1×x = ±x. The whole mask-widening cost (~24 ops) genuinely goes away. Strong reasoning.

## Weak nodes (claims that need empirical verification before trusting)

- **N10, N11** — the "~18 ops per 16 trits" and "~3.3× speedup" estimates. Theoretical, derived from instruction counts. Real speedup depends on:
  - Apple Silicon vmlal_s32 throughput per cycle
  - Dependency chains (the 8 vmlal calls accumulate serially into the same int64 accumulator pair)
  - Register pressure and reuse
  - LTO inlining behavior
  Could be 2×, could be 4×; probably between. **Mitigation: prototype + measure.**

- **N12** — vmlal_s32 throughput "≥ 1/cycle assumed." Apple's published throughput tables aren't authoritative for me here; would need empirical measurement on the target M-series.

- **N16** — concern about consumer audit, not a finding yet. The lesson from shift3 is real but needs to be operationalized as an audit step.

## Contradictions and tensions

- **N16 vs N26**: project rule allows optimization without consumer demand for substrate work, but recent learning (shift3) shows it produces kernel wins that don't propagate. Tension. Resolution: optimization is allowed, but the "substrate-claim" framing can only be made when consumer measurement supports it. If we ship a 3× kernel speedup with no consumer evidence, the work is real but the substrate-claim story is weak.

- **N15 (Case W bypasses int32 problem) vs N29 (not pursuing Case W this cycle)**: there's a strategic alternative we're explicitly not exercising. This is fine for one cycle but should be acknowledged in the closeout. Not a contradiction — a deferral.

## Missing information (this cycle would benefit from)

- **M1.** Consumer audit. Who calls `m4t_mtfp_ternary_matmul_bt`? Concrete grep + analysis. Without this, we can't say whether prototyping has consumer-visible value.
- **M2.** Empirical vmlal_s32 throughput on the target M-series. Could measure with a tight microbench.
- **M3.** Empirical kernel-level speedup of the vmlal_s32 path. Requires prototyping (chicken-and-egg with the audit decision).
- **M4.** Consumer activation-precision flexibility. Of the consumers using Case S, are any actually free to use Case W instead? This affects whether Case S optimization matters even in the long run.
- **M5.** What other route candidates exist that I missed. I claim the inventory is complete (SDOT, SMMLA, vmlal, vmla, vbsl, vqrdmulh, TBL, SVE, PMULL); plausibly missed something.

## Errors-and-recovery pattern from prior cycles

The shift3 NEON cycle taught two things:

1. **Consumer-audit-before-prototype.** We shipped a 9.6× shift3 speedup that touches no current consumer. Honest but a missed opportunity to focus the cycle better.

2. **Bit-exact gate must survive productionization.** The shift3 cycle's G1 gate was structurally invalidated by G6 productionization. Remediation exposed `m4t_mtfp_shift3_scalar_ref` as a permanent oracle. If we productionize the vmlal_s32 path, we need the same pattern: `m4t_mtfp_ternary_matmul_bt_scalar_ref` (or similar) preserved.

These lessons apply directly to this cycle. The synthesis should encode them.

## Where I might be wrong

- **The op-count estimate.** I counted ~60 in the inner loop; the user said "about 30." Possible the user was counting fewer ops — maybe by group, maybe ignoring the housekeeping. My count includes everything that emits an instruction. Either count tells the same qualitative story (much slower than 1-cycle silicon).

- **The vmlal_s32 throughput assumption.** If Apple Silicon's vmlal_s32 is half-rate (1 per 2 cycles), the speedup estimate halves to ~1.7×. Less compelling.

- **The "multiply subsumes bsl" claim.** Algebraically sound for trit ∈ {-1, 0, +1}. But there could be an edge case I'm not seeing — e.g., int32 saturation at MAX_VAL × -1 if MAX_VAL = INT_MIN exactly. Need to verify: MAX_VAL = 581130733 < INT32_MAX/2, so -MAX_VAL is well within int32 range, no saturation issue.

## Project-vision alignment

This cycle touches Tier 3c (MTFP19 × packed-ternary matmul). It's substrate-internal optimization, doesn't expand the elemental floor. Fits the substrate's current surface. Doesn't conflict with any of the three vision foundations (six primitives floor, math as routing signatures, base-3 information).

## Cross-cycle observation

This is the SAME shape as the shift3 NEON cycle — kernel-level optimization opportunity, real hardware-routing insight, no current consumer evidence. The risk is replaying shift3's outcome (real win, no consumer visibility) without first checking whether anyone cares.

The strategic correction: do the consumer audit FIRST. Use the shift3 lesson.

## What I'd want before deciding to prototype

In order:

1. **Consumer audit (cheap, ~10 min).** Grep + categorize call sites. If 0 hit `m4t_mtfp_ternary_matmul_bt`: cycle ends here, document the analysis.

2. **vmlal_s32 throughput characterization (cheap, ~30 min).** Microbench. Surfaces whether the ~3× estimate is realistic or if we'd get 1.5×.

3. **Activation-precision flexibility audit (medium, ~30 min).** Of consumers using Case S, how many could move to Case W? If ALL could move, Case S optimization is wasted; document and recommend Case W migration instead.

Only AFTER these three would prototyping be informed.

## Honest framing

The technical analysis is solid. The vmlal_s32 path IS the hardware routing answer. ~3× kernel speedup is a credible estimate. But "should we do it" is downstream of "does it matter" — and we don't know that yet. The shift3 cycle just reminded us this distinction matters.

The right move is audit-first. If audit returns "no consumers": document analysis, no prototype, cycle ends with a clear NO-ACTION decision. If audit returns "consumers exist": then characterize the consumer impact and decide whether the prototype-then-productionize cost is justified.
