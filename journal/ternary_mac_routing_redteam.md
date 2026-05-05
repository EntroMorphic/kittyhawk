# RED-TEAM: ternary MAC routing cycle

Cold-eye review of `journal/ternary_mac_routing_closeout.md` and the productionized state. Ten gates passed cleanly; this red-team examines whether the gates actually proved what they claimed, and whether the framing of the result is accurate.

## Critical findings (real risk to correctness)

### C1: Bit-exact verification is sample-based — 23 configurations

Unlike the shift3 cycle's exhaustive sweep (22.08 × 10⁹ test points across the full input space), this cycle's bit-exact gate covers only 23 configurations: 11 K boundary cases (K ∈ {0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65}), 6 trit distributions (all-zero, all-+1, all--1, balanced, sparse 10%, balanced+sparse), 3 random seeds at K=128, and 3 bulk shapes.

The matmul state space is M × K × N × 2^K (trit patterns) × MAX_VAL^(M*K) (activations) — too large to exhaust. But the SAMPLE coverage isn't tight either:

- **No saturation-edge inputs.** The bit-exact test uses random activations in [-MAX_VAL, +MAX_VAL] with random trits. It doesn't deliberately construct inputs that drive `acc` close to or past the int64 saturation bound. The vmlal pipeline rounds-and-clamps differently from the bsl-bsl pipeline only at edge cases; if those edges aren't sampled, divergence could ship.
- **No K-large coverage.** Maximum K tested is 4096. The saturation argument (T-G5) claims safety up to K ≈ 1.59 × 10¹⁰, but no test goes anywhere near that.
- **No alignment-stress.** The kernel processes 16-trit blocks; tail cases at K=17, 33 ARE tested, but unaligned activation pointers (if a consumer ever passes one) aren't.

**Risk:** medium. The math says vmlal ≡ bsl for ternary ∈ {-1, 0, +1} (multiply-by-trit subsumes the bsl pattern); the sample test passes; but I can't claim "bit-exact across all valid inputs" with confidence — only "bit-exact across a 23-config sample."

**What would tighten:** a longer-running sweep (e.g., 1000 random configs across (M, K, N) ranges + random seeds), plus saturation-edge constructed cases. Could be a ctest with a "slow" label like the shift3 exhaustive.

### C2: T-G1 throughput measurement validity

Two iterations were needed to defeat compiler constant-folding (first attempts measured 0.057 ns/call after the compiler folded `acc += K*(a*b)` to `add+branch`). The third iteration measured 0.84 calls/cycle for pattern C and emits 24 smlal in the binary (8 per noinline function × 3 patterns).

But: I only validated the third iteration via disasm count (24 smlal present) and pattern-relative consistency (A > C > B as expected). I did NOT independently confirm 0.84 calls/cycle via:
- A second measurement method (e.g., Apple Instruments)
- Theoretical comparison against ARM's published throughput tables for M-series
- A different microbench design (e.g., longer call chains, different input patterns)

**Risk:** low. The number IS plausible for Apple Silicon's SIMD throughput, and the kernel-level perf measurement (T-G8 with realistic kernel inner loop) corroborates it indirectly. But "0.84 calls/cycle" is one number from one bench; I treated it as gospel for the cycle's analysis.

## High findings (limit confidence in claims)

### H1: Saturation argument is verbal, not validated at extreme K

T-G5 wrote a saturation argument: |acc| ≤ K × MAX_VAL, fits int64 for K up to ~1.59 × 10¹⁰. Math is sound. But T-G4's bit-exact gate maxes out at K=4096. There's no test at K = 10⁹ (would need ~6 GB of activation data) or any K that approaches the int64 boundary.

The argument is mathematically watertight; the failure mode would only be a code bug (e.g., an int32 intermediate I missed). A static analyzer or extra inline `_Static_assert` could catch a code-vs-argument drift.

**Risk:** low. The math is right; the existing bsl-NEON kernel had the same K-bound and operated correctly (no saturation issues reported in years of use).

### H2: BATCHED 1.17× speedup measured at one (M, K, N) shape

T-G8 reported BATCHED at M=64 K=4096 N=64 (1.17×) and TIGHT-LOOP at M=4 K=64 N=4 (2.45×). Per the CONTRIBUTING.md scope-match rule, workload shape was declared. But:

- Only ONE bulk (M, K, N) tuple was measured. M=8 K=4096 N=8 might give different numbers (less reuse of activation cache lines per W column). M=128 K=512 N=128 (different aspect ratio) might give different numbers.
- Single (M, N) ratio test means we're claiming "BATCHED gain" from a single point.

**Risk:** low. The vmlal vs bsl comparison is structural (multiply-by-trit replaces mask-widen-and-bsl), so the relative ordering should hold across shapes. But the magnitude (1.17× vs e.g. 1.5× or 1.05×) depends on shape; shouldn't extrapolate from one point.

### H3: No consumer benefits from this optimization

Empirical: only the test calls `m4t_mtfp_ternary_matmul_bt`. `grep -rn` confirms zero production gesh probes touch this kernel. Same outcome as the shift3 NEON cycle: real kernel-level work, no consumer-visible speedup.

This is NOT a violation of the consumer-demand rule (the user explicitly disclaimed it for foundational substrate work). But it IS a fact worth flagging: the 1.17× / 2.45× / 16.7× speedups are kernel-microbench numbers; nothing in the substrate's current consumer set surfaces them.

**Risk:** none for correctness. Risk for FRAMING: future readers of CHANGELOG/closeout could overestimate the cycle's project-level impact. Should be flagged explicitly.

## Strategic findings (framing accuracy)

### S1: "Routed through hardware" is accurate; "close to silicon" would overstate

The closeout's headline ("the substrate's `m4t_mtfp_ternary_matmul_bt` ternary MAC routes through vmlal_s32 on Apple Silicon NEON") is accurate. `smlal.2d` IS real silicon; we're using it as intended.

But the CHANGELOG line "the closest existing hardware analog to a 'ternary MAC at int32 width'" should not be read as "we got close to custom silicon's throughput." Custom ternary MAC silicon (NEON-vector-shaped) would do something like 4 int32×trit MACs per cycle (limited by 128-bit register width at int32 lanes). We're at ~0.94 trits per cycle (the kernel does 16 trits in ~17 cycles).

**Custom-silicon-to-vmlal-route ratio: ~4× to ~17×, depending on assumed silicon throughput.** The 1.17× we delivered is over the bsl-NEON path, not over silicon.

This is honest in the prose but easy to misread. Worth a clarifying note in the closeout / CHANGELOG.

### S2: Case W via MTFP4 activations is the strategically larger lever

The closeout's "Honest concerns from this cycle" #4 mentions this. Worth re-emphasizing as a red-team finding because the magnitude difference is large:

| Path | Throughput |
|------|------------|
| Case S vmlal (this cycle) | ~0.94 trits per cycle |
| Case W SDOT (existing m4t_mtfp4_sdot_matmul_bt) | ~16 trits per cycle |

If a consumer can use MTFP4 activations (the int8 cell type, range ±40), they get ~17× more throughput than vmlal at the same matmul size. This cycle's contribution is meaningful (1.17× over bsl) but is dwarfed by the available "switch activation cell type" lever. Not a bug; a strategic frame.

## Medium findings

### M1: Alias test only covered Y==X; assertion exists for Y==W_packed but untested

T-G6 tested `Y == X` (alias-forbidden by the kernel's assert at line 213 in m4t_ternary_matmul.c). The kernel ALSO asserts `Y != W_packed` at line 214, but I didn't add a test for that case. If the second assert ever broke (unlikely; it's a one-liner), a regression could ship.

**Risk:** very low. Trivial fix: add a 5-line test for the second alias case.

### M2: Closeout's per-gate disposition table for T-G3 is inaccurate

The table reads: "T-G3 ... Initial wrapper m4t_mtfp_ternary_matmul_bt_vmlal (later removed at T-G9). | (intermediate; fold at T-G9)"

The "intermediate" framing is wrong. T-G3 produced a PERMANENT artifact: `static int64_t ternary_dot_vmlal(...)` in m4t_ternary_matmul.c. That's the inner-loop helper that the productionized kernel calls. The TRANSIENT artifact was the public `_vmlal` wrapper that was removed at T-G9. The table conflates the two.

**Risk:** documentation accuracy only.

## Low findings

### L1: bsl-NEON code preserved only via git history

Per "DELETE = never" project rule, superseded code must be preserved. The bsl-NEON ternary_dot is gone from the working tree; recoverable via `git log -- m4t/src/m4t_ternary_matmul.c`. The closeout justifies this as "git log counts as preservation," which is defensible but unhelpful for a future reader inspecting the file. A 5-line comment block in m4t_ternary_matmul.c pointing to the prior approach + git SHA would reduce friction.

### L2: Test file rename is clean but header description is slightly stale

`test_m4t_ternary_matmul_neon.c` opens with "bit-exact verification of the vmlal_s32-routed ternary matmul." Should be "bit-exact verification of the production NEON path of m4t_mtfp_ternary_matmul_bt" to match the renamed convention.

### L3: 23-config bit-exact runs in <0.4s; could afford more

The bit-exact ctest entry runs in ~0.4 seconds. Could quintuple the configurations (115) or run 1000 random seeds at no perceptible cost. Cheap upgrade.

## Methodology issues this red-team surfaces

**1. The pre-emptive scalar reference (T-G2) worked exactly as intended.** The shift3 remediation lesson was fully internalized this cycle; productionization at T-G9 didn't invalidate verification at T-G4. The pattern is reusable.

**2. Microbench discipline (T-G1 took two iterations) needs a checklist.** Constant-folding, dead-code elim, register pressure, instruction scheduling can all silently invalidate measurements. A shared "throughput microbench checklist" in CONTRIBUTING.md would prevent this from being relearned every cycle. Not in scope this red-team.

**3. The cycle's framing is accurate within scope, but the scope is small relative to the available lever (Case W).** If a future cycle pursues Case W migration, the kernel-level gain from THIS cycle becomes vanishingly small relative to that move. Worth noting in any forward-looking doc that references this cycle.

## What I'd want before declaring this fully closed

In rough priority:

1. **Bigger sample for bit-exact (C1).** Either add a "slow" ctest variant with 1000+ configs, or document why 23 is sufficient with a probabilistic argument.
2. **Saturation-edge bit-exact case (C1).** Construct one or two configs where acc approaches ±MAX_VAL × K. Cheap.
3. **Multiple (M, K, N) shape measurements (H2).** 3-5 BATCHED shapes instead of 1.
4. **Y==W_packed alias case (M1).** Trivial.
5. **Closeout corrections (M2, L2).** Documentation cleanup.
6. **Custom-silicon-vs-vmlal framing in closeout (S1).** Add explicit "this is hardware-routed but ~17× off silicon ceiling" note.
7. **Reference Case W lever in CHANGELOG (S2).** Re-emphasize.

## Status

10 findings (1 critical, 3 high, 2 strategic, 4 medium/low). None invalidate the cycle's correctness — the productionized vmlal path IS bit-exact against the scalar reference for every configuration tested AND structurally sound (multiply-by-trit ≡ bsl pattern by algebraic identity). The findings are about EVIDENCE COMPLETENESS and FRAMING ACCURACY:

- The "1.17× over bsl, 16.7× over scalar" numbers are real. They're also one-shape measurements that shouldn't be extrapolated.
- The "ternary MAC routed through hardware" framing is accurate. It shouldn't be read as "close to custom silicon" — we're ~17× off that ceiling, with Case W (existing) being a much larger lever.
- The bit-exact verification covers a sample, not the full input space. The risk is structural (the math is right) but the gate is thinner than shift3's.

The cycle's productionization should stand. A small remediation could close C1, H2, M1 in maybe ~30 minutes of work; the others (S1, S2) are documentation amendments.
