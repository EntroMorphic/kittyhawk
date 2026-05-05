# NODES: cross-exp accumulator routing

Atomic claims/findings/concerns extracted from `cross_exp_accum_routing_raw.md`.

## Findings (verified by code)

- **N1.** `m4t_mtfp_vec_accum_aligning` accesses `running_exp` as a SCALAR (one read at line 187, one write at lines 242 and on the addend>running branch). The pointer is for output, not per-block reads.
- **N2.** All N cells in one call use the SAME alignment shift `s = M4T_POW3_TABLE[delta]`. This is one-k-per-call BATCHED divide, not per-cell-varying.
- **N3.** The inner loop calls `m4t_pow3_round_div(value, s, &had_rem)` per cell — the same scalar function whose batched form was productionized to NEON in the shift3 cycle.
- **N4.** Three branches: same-exp (already NEON-fast via `vec_add_inplace`), addend>running (rescale running down), running>addend (rescale addend down). Each align branch is ~25-line loop calling pow3_round_div per cell.
- **N5.** Per-cell scalar cost: ~18-25 cycles (pow3_round_div ~10-15 + add 1 + clamp ~2 + flag handling ~5).
- **N6.** Per-call scalar cost for N=64: ~1100-1600 cycles. User's "hundreds of cycles" framing is accurate.

## My earlier wrong framing — corrected

- **N7.** The shift3 NEON closeout's "Honest concerns" #1 said the cross-exp accumulator does per-cell-varying-k division. That was wrong (per N1, N2). The cross-exp accumulator's per-call shape matches shift3 BATCHED exactly.
- **N8.** Consequence: the shift3 NEON pipeline (vmlal-magic-multiply, productionized in shift3 cycle) routes the cross-exp accumulator's divide step DIRECTLY. No new technique needed.

## What's already routable

- **N9.** **Align step:** `m4t_mtfp_shift3` divide path → ~17 cycles per 16-cell block. Same magic constants table from `m4t/src/m4t_pow3_magic.h`.
- **N10.** **Add step:** `m4t_mtfp_vec_add_inplace` → `m4t_mtfp_block_add` → ~3-5 cycles per 16 cells (single NEON `sqadd + smin/smax` per 4-cell block).
- **N11.** Composed (two-pass): ~20 cycles per 16-cell block. Fused (one pass through magic-multiply + accumulate): probably ~17 cycles.

## Speedup estimate

- **N12.** Current scalar: ~250-400 cycles per 16-cell block.
- **N13.** NEON-routed (composed): ~20 cycles per 16-cell block.
- **N14.** Estimated speedup: **~12-20×** for the divide-and-add work, dependent on shape.

## Flag-tracking trade-off

- **N15.** Current scalar loop sets ROUNDED bit when divide had remainder, SATURATED when post-add clamp triggers. Per-cell, both bits are preserved.
- **N16.** vmlal-magic-multiply gives the rounded result without exposing remainder existence. Reconstructing requires `(quotient × divisor) != original` check — ~5 cycles per block.
- **N17.** Three options for flag tracking on NEON path:
  - (a) Drop entirely → scalar fallback when `flags != NULL`. Cheapest. Loses NEON win for flag callers.
  - (b) Reconstruct via NEON compare-and-set. Adds ~5 cycles per block; full fidelity.
  - (c) Drop ROUNDED only; keep SATURATED via post-add clamp comparison (already needed for the clamp itself). Hybrid.
- **N18.** T2-C precedent (Tier 2 perf cycle) chose (a) for the same-exp branch's `vec_add_inplace` fast path: scalar fallback when `flags != NULL`. Matches existing convention.

## Lessons from prior cycles to apply

- **N19.** **shift3 remediation methodology:** expose `_scalar_ref` BEFORE prototype. Avoid the G1↔G6 invalidation pattern. Here: `m4t_mtfp_vec_accum_aligning_scalar_ref` should be the first gate.
- **N20.** **ternary MAC remediation methodology:** start with the bigger evidence base. 1000 random configs + saturation-edge + multi-shape bench from G1, not as remediation after.
- **N21.** **V4-residual-3 + scope-match rule:** report speedup as a range across shapes, not a single point.
- **N22.** **Throughput microbench discipline (CONTRIBUTING):** apply the 7-point checklist (disasm verification, non-constant inputs, distinct inputs per call, noinline, min-of-N, workload shape declared, range-not-point).

## Concerns

- **N23.** Flag tracking is a substrate-design choice — should SYNTHESIZE pre-commit, or punt? Lean: pre-commit. Going with (c) hybrid (best fidelity/speed balance) by default; if implementation reveals (c) is harder than expected, fall back to (a).
- **N24.** Aliasing: scalar implementation asserts `running != addend`. NEON-routed version needs the same. Plus the divide-write-add pipeline must handle the in-place running buffer correctly (read all 4 cells before writing).
- **N25.** Output exponent: when `addend_exp > running_exp`, the implementation updates `*running_exp = addend_exp` after the loop. NEON path needs the same.
- **N26.** Edge cases: `delta >= 20` collapses one side to zero (memcpy or no-op); n == 0; n with sub-block tail. Existing scalar handles all; NEON must preserve.

## Cross-cycle observation (positive)

- **N27.** This cycle would make `m4t_pow3_magic.h` a SECOND-consumer-validated foundational primitive (was: shift3 only). Validates the magic table as substrate-foundational, not one-off.
- **N28.** Cycle scope is SMALLER than shift3 or ternary MAC. No new technique to develop, no new magic constants. Just applying existing shift3-divide + block-add to a second consumer.

## Open questions

- **N29.** Should the NEON path FUSE divide-and-add, or compose them as two passes (divide-into-scratch, then add)? Fused is faster but more code; composed is cleaner.
- **N30.** Does the existing `m4t_mtfp_shift3` API even fit the use case directly? It writes to dst, doesn't ACCUMULATE. So a fused kernel needs a custom inner loop, not a direct call to shift3.
- **N31.** What about the same-exp branch (already NEON-fast)? Is there anything to improve, or leave alone? Lean: leave alone — it's already at single-block-add throughput.
- **N32.** The `m4t_mtfp_vec_add_aligning` and `m4t_mtfp_vec_sub_aligning` wrappers (lines 274+) probably also benefit. Should the cycle include them, or just the accumulator? Lean: include — they delegate to the accumulator, so no extra work.

## Methodology constraints

- **N33.** No consumer-demand gating per memory directive. The user's framing is the demand for foundational substrate work.
- **N34.** Workload-shape declaration per CONTRIBUTING scope-match rule.
- **N35.** Throughput microbench discipline per CONTRIBUTING new checklist item.
