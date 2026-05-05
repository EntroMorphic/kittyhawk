# RAW: routing a ternary MAC through existing M4/NEON hardware

Stream of consciousness on the conversation about `m4t_mtfp_ternary_matmul_bt`'s inner loop and how to route the int32×trit MAC through hardware.

## What started this

User pointed at `m4t_mtfp_ternary_matmul_bt`'s inner loop and framed the cost as "what custom silicon would do in one cycle." Said "about 30 NEON instructions per 16-trit block." Asked specifically what existing M4/NEON hardware features can route a "ternary MAC" given we can't develop silicon.

I verified the actual inner loop op count is closer to ~60 NEON ops per 16-trit block (user undercount by ~2×). The dominant cost is NOT trit decode (~6 ops, tight) — it's the mask widening (~24 ops) + conditional negate via `vbslq_s32` (~16 ops) + widening accumulate (~8 ops). Decode is fine; the mask-and-select pattern is the cost.

## What I worked through

**Inventory of hardware features:** SDOT (int8×int8→int32, 16 lanes), SMMLA (Armv8.6+, M3+, int8 matrix), vmlal_s32 (int32×int32→int64 widening, 2 lanes), vmla_s32 (no widen, would overflow), vbsl (current), vqrdmulh (fractional, wrong shape), TBL (decode only), SVE/SVE2 (Apple doesn't expose).

**Why SDOT can't route Case S:** SDOT requires int8×int8 operands. Case S activations are int32 (MTFP19, magnitude up to ~5.8e8). Even if trits unpack to int8, the int32 activation side blocks SDOT.

**Why vmla_s32 (no widen) doesn't work:** even one 16-element block sum is bounded by 16 × MAX_VAL ≈ 9.3e9, which overflows int32. Need int64 from the start. So vmlal_s32 (widening to int64) is the right primitive.

**The vmlal_s32 path:** 6 ops (decode trits to int8) + 4 ops (sign-extend int8 → int32) + 8 ops (vmlal_s32, 2 lanes per call × 8 = 16 elements) = ~18 ops per 16 trits. ~3× theoretical reduction from current ~60.

**Why this works (multiply subsumes the bsl):** trit ∈ {-1, 0, +1}. Multiply by the trit handles BOTH conditional negate AND zero-gate at once. 0 × x = 0 (no zero-gate needed); +1 × x = x; -1 × x = -x. The whole vbsl pattern (mask widen + conditional select + zero gate) collapses into one widening multiply.

**The strategic alternative (Case W):** the substrate already has `m4t_mtfp4_sdot_matmul_bt` — int8 activations × ternary weights → int32, using SDOT directly at full hardware throughput. ~1 NEON op per 16 elements. The substrate offers BOTH kernels because activation precision varies by consumer.

## What I'm uncertain about

**Apple Silicon vmlal_s32 throughput.** Probably ≥ 1 per cycle (multiple SIMD pipes), but I haven't characterized it. Could be 0.5/cycle (single pipe), 1/cycle, or higher depending on M-series core. The ~3× kernel-speedup estimate hangs on this.

**Consumer audit.** I have NO IDEA whether any current consumer actually calls `m4t_mtfp_ternary_matmul_bt`. Recent shift3 NEON cycle taught us this matters — we shipped a 9.6× shift3 speedup that touches no current consumer (verified by grep). Same risk here.

**Latency vs throughput.** vmlal_s32 has dependency chains across the 8 calls per block (acc accumulates serially). Could limit per-cycle throughput below the inverse-throughput count. Need to check.

**Dependency on Case W feasibility.** If we discovered that ALL consumers currently using Case S could move to Case W, then optimizing Case S is wasted work. If NO consumers can move (activation precision required), Case S is the only path. Audit needed.

## What I haven't explored

- **Per-byte mask widening tricks.** The current mask widening (~24 ops) might compress via different intrinsic choices (e.g., vsubl + sign-extend vs vmovl + vceqq). Smaller win, harder to get right.
- **Batched accumulator.** Could we batch multiple W rows worth of MAC into one vmlal-chain to amortize the decode? Maybe. Memory access pattern matters.
- **TBL-based "multiply by trit"?** A 256-entry LUT keyed by (trit, low_byte_of_activation) is too big. A smaller LUT keyed only by trit (3 entries: 0, +x, -x) doesn't help because TBL is per-byte, not per-int32.
- **PMULL (polynomial multiply).** Useful for sparse/bit-vector tricks but not for our shape.

## Things I noticed but didn't pursue

- **SMMLA on M4.** I asserted SMMLA exists on M3+; should verify M4. Even if available, doesn't help for int32×trit.
- **Fused vneg + vbsl into vsbcl?** No such instruction; conditional-negate is genuinely a 2-op pattern in NEON.
- **Branch on trit sparsity.** If many trits are zero, could we skip blocks? Branch prediction overhead probably wipes the win for typical density.

## Lessons from prior cycles that bear on this

- **shift3 NEON cycle:** prototyped, productionized, got 9.6× kernel speedup. Honest finding from benchmark: no current consumer touches it. Same risk here.
- **shift3 remediation:** the cycle's bit-exact gate (G1) was structurally broken by productionization (G6) because the gate compared against the function being replaced. Lesson: when productionizing, expose the original implementation as a separately-preserved scalar reference (`m4t_mtfp_ternary_matmul_bt_scalar_ref` if we go this way).
- **V4-residual-3 LTO microbench:** workload-shape-bound speedups. The "no observable LTO benefit" was correct narrowly (carry-dep workload) but not generally (3× on pipelined). Same risk: kernel-level speedup may be invisible at consumer-level depending on workload shape.

## What feels right

The vmlal_s32 path IS the hardware route. It's the closest analog to a "ternary MAC at int32 width" that Apple Silicon offers. ~3× reduction is real and worth pursuing IF a consumer actually exercises this kernel.

The Case W path (SDOT via MTFP4 activations) IS the strategic move. It bypasses the int32 issue entirely. Whether it applies depends on consumer activation precision.

The trap to avoid: prototyping and productionizing without first verifying that ANY consumer actually cares. We just walked through that with shift3. Let's not replay the pattern unconsciously.

## Strategic question

Is this conversation about (a) closing a real consumer bottleneck, (b) inventorying substrate options for future use, or (c) characterizing the hardware-vs-substrate-cost gap as an analytical exercise? Each leads to different next moves:

- (a) → audit consumers first, then prototype if hot
- (b) → document the analysis (this LMM cycle), no code work
- (c) → write the analysis up as a thesis-level note, no prototype needed

The user's framing ("what existing features can we lean on") sounds like (a) or (b). The "we cannot develop our own silicon at the moment" framing suggests (a) is the real question.

## Where I would land if running on instinct

Audit consumers FIRST. If 0 hit `m4t_mtfp_ternary_matmul_bt`: document the analysis, no prototype. If ≥ 1 hit it: characterize hot-path-ness, then decide on prototype. This is the lesson from shift3 applied.
