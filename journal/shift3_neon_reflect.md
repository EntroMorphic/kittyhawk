# REFLECT: NEON magic-multiply divide-by-3^k prototype

Cold-eye review of `shift3_neon_nodes.md`. What's load-bearing, what's weak, what contradicts, what's missing.

## Load-bearing nodes (the cycle's evidence rests on these)

- **N1, N2** — magic constants exist and are exhaustively verified. This is the foundation. Without it, nothing else matters. Strong evidence: 1.16 × 10⁹ × 19 = 2.2 × 10¹⁰ test points, all bit-exact.
- **N3** — NEON kernel matches scalar reference. Tested on 50K + 12 corners per k (951K test points). Sample-based, not exhaustive.
- **N4, N5** — speedup numbers. Single bench run, n=4096, 200 iters. Not multi-seed, no cache-defeat verification, no median-of-N.

## Weak nodes (claims that look strong but rest on thin evidence)

- **N3 weakness**: 50K random + corners is a VERY thin slice of 1.16 × 10⁹. The generator did exhaustive on the EMULATOR; the NEON kernel test relies on the assumption that the NEON intrinsics behave exactly like the emulator. ARM NEON has documented edge cases (saturation behavior on overflow, signedness of shift counts, etc.). If `vshlq_s64` with negative count behaves differently from C's arith shift in some corner I haven't tested, the prototype would silently disagree on that corner. **Mitigation needed: run the NEON kernel through the same exhaustive verify the generator ran on the emulator.**

- **N4 weakness**: single bench run; no noise control. The V4-G5 / V4-residual-3 cycle established that perf measurements need workload-shape declaration and at least min-of-N runs. This prototype's "10×" is one shape (n=4096 carry-free vector) at one moment. The substrate's actual consumer of shift3 (whoever it ends up being) will probably have a different shape.

- **N17 weakness**: I claimed the scalar isn't auto-vectorized but didn't disasm. Could be wrong. If AppleClang IS vectorizing the scalar path, the 10× shrinks because we're comparing NEON to NEON.

## Contradictions and tensions

- **N15 ↔ N26**: N15 says "no current consumer has shift3 on its hot path"; N26 says the user's request itself counts as consumer demand. These don't actually contradict — the user's demand is for the OPTION to have a fast shift3, even if no current call site needs it. But the framing in CHANGELOG / future docs needs to reconcile: ship the optimization, OR keep it as documented headroom. Different commitments.

- **N13 ↔ N16**: The kernel can't help the cross-exponent accumulator (per-cell-varying k), but a variant could. So shipping THIS kernel as the productionization doesn't address the load-bearing call site. This is the same shape as V4-residual-3's pipelined-vs-carry-dep finding: a real win on a workload that's not the current hot path.

- **N22 / N23 reframing**: I quoted "40× speedup" twice in earlier sessions. The actual 10× is honest, but I should walk back the 40× explicitly in any productionization commit to avoid future readers anchoring on the wrong number.

## Missing information

- **M1.** No measurement of how long the kernel actually takes through a real consumer's call path. Pure microbench is per-op latency, not per-op throughput in the consumer's actual pipeline. The 10× could be 2× or 1× when the kernel is mixed with other ops, branch prediction settles, cache effects play in.

- **M2.** No disasm of either the scalar or the NEON kernel under LTO. Without the disasm, I can't verify what the compiler actually emits for either path, can't rule out auto-vectorization of the scalar, can't confirm the NEON kernel inlines into bench main.

- **M3.** No power/energy measurement. The substrate's foundational vision (per project_vision memory) cares about base-3 modeling primarily, but energy is a stated falsification axis in THESIS.md. NEON is wider but doesn't necessarily mean lower energy per element.

- **M4.** No investigation of the multiply direction (k > 0). The substrate's `m4t_mtfp_shift3` for k > 0 does scalar `(int64_t)src[i] * scale` + clamp. If I'm productionizing the divide direction, the multiply direction's opportunity should be characterized too.

- **M5.** No exhaustive test of the NEON kernel itself (only the emulator was exhaustive). N3's 50K samples is the runtime evidence that NEON matches emulator-which-matches-scalar. This is a transitive proof; the missing direct link is "NEON exhaustive vs scalar."

- **M6.** No alias test (`dst == src`).

- **M7.** No saturation proof for `vmovn_s64`'s narrow.

- **M8.** No characterization of whether the prototype helps the cross-exp accumulator (per-cell-varying-k variant). N16 is a hypothesis; needs investigation before claiming the optimization "transfers."

## Errors-and-recovery pattern

N18, N19, N20, N21 — four bugs caught during prototyping. All caught either by my own verification output or by computational runaway (N21). None silent. The pattern is healthy: build-test-fix-iterate. But it's worth noting that N20 (N_max overshoot) was a silent BUILD-time bug — generator reported FAIL for k ≥ 11 but I had to look at WHY rather than just trust the FAIL. Future generators of this shape benefit from sanity-check assertions on the bounds (e.g., `assert(N_max ≤ 31 + ceil(log2(d)))` at table-generation time).

## What I'd want to know before productionizing

In rough priority:

1. **Exhaustive NEON-vs-scalar verification.** Replace the 50K sample with the full 1.16 × 10⁹ run, ONCE. Confirms the NEON intrinsics behave like the emulator across the entire input space. If there's any divergence, it surfaces here.

2. **Saturation proof + aliasing test.** Cheap.

3. **Disasm comparison.** Confirm scalar isn't already vectorized; confirm NEON kernel emits the expected ops.

4. **Bench discipline applied.** Per V4-residual-3 methodology lifted: name workload shape, min-of-N runs, ideally with adversarial workload variants. The current "10×" is provisional.

5. **Cross-exp accumulator audit.** Is per-cell-varying-k tractable? If yes, that's the structurally more important consumer of this technique. If no, this kernel is documented headroom.

## Methodology check (against project rules)

CONTRIBUTING.md scope-match rule: "Single workload-shape → cannot claim general kernel performance." The 10× is currently single-shape. Per the rule (which I just added to CONTRIBUTING in the concern #4 sweep), this number needs a workload-shape caveat or an additional shape measured.

CONTRIBUTING.md rule "Multi-seed gates the cell": single-bench-run isn't multi-seed. Not load-bearing for THIS cycle's verdict, but if a productionization PR ships with the 10× as a perf claim, multi-seed is required.

CONTRIBUTING.md rule "Kernel-use gates the substrate-claim": this kernel is shift3, which is part of the elemental floor. Its only legitimate productionization site is INSIDE m4t_mtfp_shift3 (the existing scalar). If I were to bypass m4t_mtfp_shift3 and call the NEON kernel directly from a consumer, that would be a substrate-bypass; the right move is to wire it into the existing primitive, gated by M4T_HAS_NEON.

## Where I was right vs lucky

- **Right**: pivoting from vqrdmulh to vmull was driven by a clear correctness reason (compound rounding), not by perf. The pivot also gave a path to bit-exact verification, which the original wouldn't have.

- **Lucky**: the 50K + corners sample happened to catch all the corners that mattered. If I'd sampled differently I might have shipped a "PASS" with a real bug. The exhaustive NEON-vs-scalar run is needed to convert luck into evidence.

- **Right**: caught my own bugs (N18-N21) before claiming PASS. The diagnostic outputs (mismatch counts, sample x values) made each bug self-evident.

- **Lucky-ish**: the smart-set sampling almost shipped with `step = d/8 → 1` triggering 8e9 tests for small d. I caught it via timeout, not via design. A better pattern would have been to bound the test count explicitly from the start.

## Project-vision alignment

Per memory's `project_vision`: three foundations are (1) ~6 frozen primitives floor, (2) math as routing signatures, (3) base-3 carries information base-2 collapses. This work touches (1) — shift3 IS one of the elemental-floor primitives. Optimizing it is "make the floor faster," not "expand the floor."

The base-3 specificity matters: this kernel works for divide-by-3^k. The same magic-multiply technique would work for divide-by-2^k (already a single shift), divide-by-5^k, divide-by-7^k, etc. — but the substrate only needs 3^k. So the technique is general; the application is base-3-specific. This is a clean match between technique and substrate philosophy.

## Honest summary

The prototype works. The 10× is real but provisional. Bit-exactness is exhaustively proven for the constants and emulator; sampled-but-not-exhaustive for the NEON kernel itself. There are four substantive missing pieces (exhaustive NEON verify, saturation proof, alias test, disasm) that are each ~30 minutes of work. There are two strategic open questions (cross-exp accumulator applicability, original vqrdmulh-with-specialization pursuit) that deserve answers before deciding productionization scope.

Right now this is a credible prototype with documented gaps. Productionization should run a tight cycle that closes the gaps, not just drop in the kernel.
