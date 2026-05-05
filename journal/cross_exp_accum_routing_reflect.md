# REFLECT: cross-exp accumulator routing

Cold-eye review of `cross_exp_accum_routing_nodes.md`. What's load-bearing, what's weak, what's missing, what contradicts.

## Load-bearing nodes (the cycle's case rests on these)

- **N1, N2** — the cross-exp accumulator's per-call shape. Verified by reading the source. STRONG. The whole cycle's premise is "this is BATCHED divide, like shift3"; without N1/N2 the cycle is misframed.
- **N9, N10** — existing NEON paths exist for both halves. Verified by file presence + prior cycle productionization. STRONG.
- **N7, N8** — my earlier framing was wrong; correction is owed. STRONG (acknowledging the error).

## Weak nodes (claims that need empirical verification)

- **N5, N6** — per-cell scalar cost estimate (~18-25 cycles/cell, ~1100-1600 cycles for N=64). DERIVED from instruction counts, not measured. Could be off by 2x in either direction depending on how aggressively the compiler vectorizes the scalar path or how the cache behaves.
- **N12, N13, N14** — speedup estimate ~12-20×. DERIVED from N5+N11. Same uncertainty inheritance. The shift3 cycle's measured BATCHED speedup ranged 9.2× to 9.7× — similar order. But the cross-exp accumulator has the additional ADD step inside the inner loop, which the scalar path also pays for. Net: speedup estimate is plausible but unverified.
- **N17 (a, b, c)** — flag-tracking cost estimates. ~5 cycles per block for option (b) is a back-of-envelope; could be 3 or 8.

## Contradictions and tensions

- **N23 vs N18:** I want to pre-commit to (c) hybrid flag tracking, but the existing T2-C precedent is (a) drop-or-scalar-fallback. Going with (c) means breaking convention; going with (a) means losing potentially-useful SATURATED bits for callers who track flags.
  - Resolution candidate: pre-commit to (a) by default (matches T2-C convention exactly); revisit if a consumer surfaces with a documented need for SATURATED on the NEON path. Cleaner with existing convention.
  - Counter-argument: (c) keeps SATURATED almost-free (the clamp comparison happens anyway for the clamp itself). (c) is "free fidelity" — drop only the expensive bit (ROUNDED).
  - Going with (c). Document why the deviation from T2-C is justified: (c) doesn't add cycles to the fast path; T2-C's choice was because adding flag work to a same-exp ADD was meaningful relative to the 3-cycle add itself, but here we're adding to a 17-cycle pipeline where the marginal flag work is <10% overhead.

- **N29 vs N30:** Should we fuse divide-and-add or compose? N30 says shift3's API is dst-not-accumulate, so we'd need a custom inner loop anyway. Fused is the natural shape since we're writing the custom loop regardless.

## Missing information

- **M1.** No measurement of the current scalar path's actual cycle count. The estimates in N5/N6 are educated guesses. A microbench would tighten the speedup-estimate prediction.
- **M2.** No survey of whether the existing `vec_add_aligning` / `vec_sub_aligning` wrappers (N32) have additional overhead beyond delegating to the accumulator. If they add something the accumulator doesn't have, that's separate work.
- **M3.** No saturation-edge analysis of the NEON-routed pipeline. The shift3 NEON pipeline was proved bit-exact for the divide alone (the productionized scalar_ref). The CROSS-EXP composition (divide + add + clamp) needs its own bit-exact verification — saturation can happen at the post-add clamp, which the shift3 verification didn't cover.
- **M4.** No analysis of whether the `delta >= 20` degenerate path (N26) has its own NEON-friendly form. Currently it's a per-cell loop checking `addend[i] != 0` for the flag bit. NEON could do the comparison faster but it's an edge case — probably leave scalar.

## What I'd want before declaring "ready to execute"

In rough priority:
1. **Confirm the speedup estimate has the right order of magnitude** via a quick scalar microbench BEFORE committing to the cycle. If it turns out to be 2× instead of 12×, the cycle's value-vs-effort changes substantially. Cheap (~10 min).
2. **Decide flag-tracking pre-commitment** (a) vs (c). I'm leaning (c); REFLECT above resolved the tension.
3. **Decide fused-vs-composed** kernel shape. I'm leaning fused (custom inner loop, since shift3 API isn't directly accumulate-shaped).

## Errors-and-recovery pattern

- **N7's correction.** I had one wrong framing in the shift3 closeout that I'm now correcting. This isn't a unique cycle event — the ternary MAC LMM cycle had a bigger correction (consumer-demand drift in SYNTHESIZE). The fact that THIS cycle's wrong framing was caught by re-reading the source carefully (rather than after committing) is the discipline working — applying lessons from prior cycles to NOT make the same shape of error.

## Project-vision alignment

This work touches Tier 3a (cross-exponent accumulator). It's substrate-internal optimization. Doesn't expand the elemental floor. Doesn't change semantics (bit-exact same answer; just faster). Fits the substrate's existing surface.

The "second consumer of magic table" framing (N27) is the structural payoff beyond raw speedup. After this cycle, the magic table earns its name as substrate-foundational rather than shift3-helper.

## Cross-cycle observation

This cycle is structurally the smallest of the three recent kernel cycles:

| Cycle | New technique? | New constants? | Verification work? |
|-------|----------------|----------------|---------------------|
| shift3 NEON | Yes (vmlal magic-multiply for divide-by-3^k) | Yes (m4t_pow3_magic.h) | High (exhaustive 22e9 sweep) |
| ternary MAC | No (reuse vmlal but for matmul) | No (no per-trit constants) | High (1000+ random configs) |
| cross-exp accum | No (reuse shift3 path) | No (reuse magic table) | Medium (bit-exact at composition level) |

The shrinking work-per-cycle is a sign the foundational cycles paid off. shift3's heavy-lift technique is now reusable; this cycle is application.

## What I might be wrong about

- **The speedup estimate.** Could be lower than 12× if the add step's overhead in the current scalar loop is larger than I estimated, OR if the NEON path's flag-tracking + fused-vs-composed overhead eats more than expected. Cheap to verify with the M1 microbench.
- **The flag-tracking choice.** Going with (c) hybrid feels right; could be wrong if reconstructing SATURATED via clamp comparison is more expensive than I'm estimating, or if there's a subtle correctness issue with computing it in the NEON pipeline.
- **The "second consumer of magic table" framing.** True if this cycle ships. If we discover the cross-exp accumulator can't actually use the magic constants (e.g., wrong saturation behavior at extreme delta), the framing falls apart. Unlikely but possible.

## Honest framing

The cycle is well-scoped. The technique is proven. The application path is clean. The flag-tracking trade-off is the one substantive design decision and I have a defensible position. The cycle should ship in less time than ternary MAC (which itself was less work than shift3).

The "ternary FPU" framing the user opened with is accurate as an ASPIRATION (one instruction per block) but the IMPLEMENTATION reduces to "compose two existing NEON pipelines." Not as headline-worthy as "we built a ternary MAC," but structurally sounder — applies an existing solved technique to a second consumer.
