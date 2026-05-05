# RAW: cross-exp accumulator routing through existing NEON paths

Stream of consciousness on routing `m4t_mtfp_vec_accum_aligning` through the M4/NEON paths the substrate already has. User framing: per-block exponent management is the ternary equivalent of an IEEE-FPU's internal align+round step; today it's hundreds of cycles of scalar align+round logic; a true ternary FPU would do it in one instruction per block.

## What I verified about the current implementation

`m4t_mtfp_vec_accum_aligning(running, running_exp*, addend, addend_exp, flags, n)`:
- `running_exp` is `int8_t*` but accessed only as a SCALAR (line 187: `int8_t e_run = *running_exp;`). The pointer is for OUTPUT (line 242 updates `*running_exp`), not for per-block reads.
- `addend_exp` is a scalar `int8_t`.
- All N cells in one call use the SAME alignment shift `s = M4T_POW3_TABLE[delta]`.
- Three branches: same-exp (no align needed), addend > running (align running down), running > addend (align addend down).
- Each align branch loops over N cells calling `m4t_pow3_round_div(value, s, &had_rem)` per cell.
- Plus per-cell ROUNDED bit set when `had_rem != 0`, SATURATED bit set when sum overflows int32.

Per-cell cost in scalar inner loop:
- `m4t_pow3_round_div`: ~10-15 cycles (sdiv + adjust)
- Add: 1 cycle
- `m4t_mtfp_clamp64`: ~2 cycles
- Flag handling: ~5 cycles when `flags != NULL`
- Total: ~18-25 cycles per cell

For N=64 cells: ~1100-1600 cycles. The user's "hundreds of cycles" framing is accurate.

## My earlier wrong framing — and the correction

In `journal/shift3_neon_closeout.md`'s "Honest concerns" section I wrote: *"the cross-exp accumulator (`m4t_mtfp_vec_accum_aligning`) doesn't benefit from this kernel. It does per-cell-varying-k division."*

THAT WAS WRONG. The cross-exp accumulator does **one-k-per-call** division, identical in shape to shift3's BATCHED workload. The shift3 NEON pipeline applies directly. I conflated:
- "per-block exponents in the running buffer" (which doesn't exist — the buffer has ONE exponent)
- "per-block exponents across multiple calls to vec_accum_aligning" (which does exist — caller chooses the per-call exponent — but is irrelevant within a single call)

This means: the shift3 NEON cycle's productionized vmlal-magic-multiply pipeline could route the cross-exp accumulator's divide step DIRECTLY. No new technique needed. Apply existing.

## What's already routable

The "ternary FPU instruction" (align + round + add + normalize per block, one cycle) decomposes into TWO operations the substrate already has hardware-routed paths for:

1. **Align (divide by 3^Δ).** Productionized: `m4t_mtfp_shift3` divide path → vmlal-magic-multiply. ~17 cycles per 16-cell block (per shift3 closeout). Same magic constants table (`m4t_pow3_magic.h`).

2. **Add (block-aligned sum).** Already exists: `m4t_mtfp_vec_add_inplace` → `m4t_mtfp_block_add` → single NEON `sqadd + smin/smax` per 4-cell block. ~3-5 cycles per 16 cells.

Composed naively (divide-then-add via two passes): ~20 cycles per 16-cell block.
Fused (single pass through magic-multiply pipeline + accumulate into the addend): probably ~17 cycles per block (the add can pipe alongside the vmlal stages).

vs current scalar: ~250-400 cycles per 16-cell block. **Estimated speedup ~12-20×.**

## The flag-tracking wrinkle

The current scalar inner loop sets:
- ROUNDED bit per cell when `had_rem != 0` (the divide had a remainder; data was lost in the round).
- SATURATED bit per cell when the sum exceeded MTFP19's int32 range.

vmlal-magic-multiply gives the rounded result without exposing "remainder existed." We'd have to reconstruct it by checking `(quotient × divisor) != original`. That's an extra NEON multiply + compare + bit set per block — adds ~5 cycles.

OR drop ROUNDED tracking on the NEON fast path. Same precedent as the same-exp branch's `vec_add_inplace` fast path (T2-C from Tier 2 perf cycle): `flags == NULL` → fast path, `flags != NULL` → scalar fallback.

Three options I see:
(a) Drop flag tracking on NEON path; fall back to scalar when `flags != NULL`. Cheapest. Loses NEON win for flag-tracking callers.
(b) Reconstruct flags via NEON compare-and-set. Adds ~5 cycles per block; preserves flag fidelity.
(c) Drop only ROUNDED on NEON (cheap to lose); keep SATURATED via the existing post-add clamp comparison (cheap). Partial flag fidelity.

(a) matches existing substrate convention (T2-C). (b) is the "do it right" option. (c) is hybrid.

## Lessons from prior cycles to apply

**From shift3 remediation:** expose the scalar reference (`_scalar_ref` variant) BEFORE prototype work. Don't repeat the G1↔G6 invalidation pattern. Here: `m4t_mtfp_vec_accum_aligning_scalar_ref` should be the first gate.

**From ternary MAC remediation:** start with the bigger evidence base. 1000 random configs + saturation-edge + multi-shape bench from G1, not as remediation after.

**From V4-residual-3:** report speedup as a range across shapes. Single-shape numbers mislead.

**From the throughput-microbench discipline (just-added CONTRIBUTING.md item):** apply the 7-point checklist to any throughput claim.

## Concerns

**1. The flag-tracking choice is a substrate-design decision, not a kernel-tuning question.** Should the SYNTHESIZE pre-commit to (a) (T2-C precedent) or punt to a separate decision? I lean (a): match existing convention, document the loss, move on.

**2. What about saturation-driven flag tracking?** SATURATED is meaningful — a consumer that loses precision via clamp probably wants to know. Approach (a) loses this for non-NULL flag callers. Approach (b) preserves it. Maybe (c) is the right pragmatic middle.

**3. The "drop ROUNDED on NEON fast path" loses information.** Specifically: when consumers check ROUNDED bits to decide whether to grow the exponent (avoid future precision loss), they'd miss those events. But: the consumer can also just check whether `e_run` changed after the call (delta != 0 implies some cells were rescaled). That's a coarser signal at the buffer level, not per-cell.

**4. Aliasing.** The current implementation asserts `running != addend`. The NEON-routed version needs the same. Plus the magic-multiply pipeline must handle in-place (writing back to `running` after divide-and-add). Need to verify.

**5. Output exponent.** When `addend_exp > running_exp`, the implementation updates `*running_exp = addend_exp` after the loop. The NEON path needs the same update.

**6. Edge cases.** `delta >= 20` collapses one side to zero (memcpy or no-op). Already handled in scalar; NEON should preserve.

**7. n == 0**, n < 16 (sub-block tail), n with mixed full-block and tail. Need scalar tail handling, same shape as shift3 NEON.

## What feels right

This cycle is structurally the SMALLEST of the recent kernel cycles because it's compose-existing-primitives, not invent-new-technique. shift3 NEON was the heavy lift (designed the magic-multiply pipeline + magic table); this cycle just applies it to a second consumer.

The "second consumer" framing has structural value beyond the speedup — it validates `m4t_pow3_magic.h` as a foundational substrate piece, not a one-off helper. Future kernels that need divide-by-3^k (whatever they may be) get the same magic table.

## Where I would land if running on instinct

1. Expose `m4t_mtfp_vec_accum_aligning_scalar_ref` first (lesson from shift3 remediation).
2. Pre-commit to (c) hybrid flag tracking: drop ROUNDED on NEON; keep SATURATED via post-add clamp comparison. Best balance of fidelity and speed.
3. Prototype the routing using the existing shift3 helpers.
4. Bit-exact test with curated + 1000 random + saturation-edge from the start.
5. Multi-shape bench from the start (3-5 shapes).
6. Productionize.
7. Smoke-test consumers (just to confirm no regression — NOT to gate the cycle on consumer evidence).

Cycle should be smaller than ternary MAC because the heavy lift (magic-multiply technique, magic table, bit-exact verification approach) was already done in shift3.

## Lessons codified that bear directly on this cycle

- **CONTRIBUTING throughput microbench discipline:** apply the 7-point checklist.
- **CONTRIBUTING scope-match rule:** sweep 3-5 shapes, report range.
- **shift3 remediation lesson:** scalar ref FIRST.
- **ternary MAC remediation lesson:** bigger sample + edge cases + multi-shape from G1.
- **Memory: no consumer-demand gating.** The user named cross-exp accumulator as the next "software doing hardware's work" gap; that IS the directive.
