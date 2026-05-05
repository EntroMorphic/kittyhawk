# REFLECT: strong-claim test setup

Cold-eye review of `tristate_strong_nodes.md`.

## Load-bearing nodes

- **N1, N2, N3** — the claim restated correctly. Comparative against B2-B; axes from `feedback_substrate_claim_scope.md`.
- **N5** — L1 (weights) as the test layer. Defensible via audit ranking + prior-work alignment (BitNet etc).
- **N9, N11** — B2-B as the canonical comparison. Same density, same functional range. Right call.
- **N12-N14** — pre-committed axes with measurable gates.
- **N25, N26** — the dual-mode B2-B (uniform AND skip-aware) is critical. Without it the verdict is biased against whichever mode we don't measure.

## Weak nodes

- **N12 (density)** — both 2 bits/cell. The "tie expected" framing is correct in our packing, but it elides a fact: information-theoretically, base-3 uses log2(3) ≈ 1.58 bits/cell. Our packing wastes the 0b11 state. A theoretically-optimal base-3 packing (e.g., 5 trits in 8 bits = 1.6 bits/cell, close to log2(3)) would be DENSER than B2-B. Whether to pursue this in the strong-claim cycle vs leave it for a future cycle is a scope choice.
  - Resolution: stick with the substrate's actual packing for this cycle. The strong-claim test is about CURRENT substrate vs base-2, not theoretical-optimal substrate vs base-2. Note this caveat in CLOSEOUT.
- **N20 (disassembly methodology)** — "count NEON instructions per inner block" is well-defined but assumes the compiler doesn't restructure the kernel beyond recognition. -O3 + LTO might inline things. Pre-commit a specific kernel function with `__attribute__((noinline))` or similar to make disassembly clean.
- **N32 (op count vs cycles)** — proxy concern. NEON op count is what we'll measure; cycles are what really matter. Different ops have different latencies and different throughput pipelines on Apple Silicon. Pre-commit op count as the gate but report wall-clock for sanity check.
- **N35 (regime-dependent expectation)** — honest, but the "split verdict" outcome is the LEAST decisive. Could the cycle structure produce a more decisive verdict? Possibly: pick a SINGLE regime that's the most representative (BitNet-typical: high zero-frac, e.g., 0.60) and gate on THAT regime alone. Use other regimes as informational sub-findings.

## Tensions

- **N31 (B2-B "reasonably tight, not heroically optimized") vs N32 (op count fairness)**: choosing how clever B2-B is shapes the verdict. Too lazy → base-3 wins trivially. Too clever → B2-B might win in regimes where base-3 should be ahead.
  - Resolution: pre-commit the B2-B kernel structure. State: "B2-B kernel A: uniform processing, no skip; B2-B kernel B: skip-aware via mask check at start of inner block." Both implementations specified up front; no iterative tuning to bias the verdict.
  
- **N22 (reuse audit's workload) vs the question of which regime matters most**: if we report on all 12 configs, the verdict is regime-dependent. If we pick one canonical regime, we get a decisive answer at the cost of representativeness.
  - Resolution: report all 12 configs honestly. Identify a "headline regime" (BitNet-typical: K=256, w_zero=0.60, a_zero=0.60) for the headline verdict. Other regimes provide context.

- **N27 (no scalar in production) vs the prototype-quality framing**: the B2-B kernel will be NEON, but does it need a scalar reference for verification? The audit-style framework already does bit-exact verification via cross-check between kernels.
  - Resolution: write B2-B with NEON path AND scalar reference (test oracle), per the substrate's pattern (`_scalar_ref` test oracles exist; production is NEON-only). The B2-B reference is for the strong-claim test; doesn't enter production. Scalar test oracle is per-project pattern.

## Missing information

- **M1.** What does the substrate's current ternary matmul look like at the disassembly level? Need to disassemble `m4t_ternary_dot_matmul_bt` (or the underlying SDOT kernel) and count ops per 16-cell block. This is the BASELINE we'll compare against.
- **M2.** Is there an existing wall-clock benchmark harness we can extend, or do we need to write one? `bench_m4t_tier2_perf.c` exists; might be reusable.
- **M3.** What's the right packing format for B2-B in our audit? The substrate packs trits as 2-bits-per-cell with a specific encoding; B2-B should pack similarly (1 sign bit + 1 mask bit per cell, 4 cells per byte). Pre-commit the layout.

## What I'd want before declaring "ready to execute"

In rough priority:

1. **Specify the B2-B kernel structure precisely.** Layout: 1 sign bit + 1 mask bit per cell, 4 cells per byte (or 16 cells per 4 bytes). Inner loop: load both, conditionally apply mask, multiply against activation, accumulate. Pre-commit the NEON op shape.

2. **Specify the disassembly methodology.** Which functions to disassemble; what instruction counts to extract; where to find the inner-loop block.

3. **Specify the headline regime.** I'd pick K=256, w_zero=0.60, a_zero=0.60 as BitNet-typical. Other 11 configs reported as context.

4. **Write the B2-B scalar reference first** — for bit-exact verification. Then write the NEON kernel. Then verify. Then disassemble.

5. **Pre-commit gate values:**
   - Density gate: PASS = both ≤ 2 bits/cell. Sub-gate: SUPPORT_BASE3 if base-3 < B2-B; PARITY if equal.
   - Precision gate: PASS = bit-exact Y match across all configs.
   - Cost gate: SUPPORT_BASE3 if NEON op count for base-3 < B2-B uniform AND base-3 < B2-B skip-aware in headline regime. PARITY if equal. FALSIFY if base-3 > B2-B.

## What I might be wrong about

- **The "B2-B kernel design choice biases the verdict" framing is tractable** — pre-committing the kernel structure upfront avoids most of the bias. But I might be overconfident; reasonable people could disagree on what "reasonably tight" means.
- **The wall-clock vs op-count tradeoff** could be wrong. On Apple Silicon, NEON throughput is so high that op count differences might not matter at the ms scale; cycle count might be more informative. But cycles aren't directly measurable on macOS without instruments. Op count + wall-clock is the practical pair.
- **The regime-dependence framing assumes regime-dependence will appear.** If both kernels perform near-identically across all configs, the verdict collapses to "tie" and the strong claim is NOT SUPPORTED. That's a real possible outcome.

## Honest framing

This is a cycle where the most likely outcome is "regime-dependent verdict" — base-3 wins on dense, B2-B-skip wins on sparse (the BitNet-typical regime). If that holds, it produces a NUANCED finding: base-3 substrate's value is concentrated in dense regimes, and the BitNet-typical sparse regime is where B2-B alternatives are competitive.

This would be a real finding. It would inform: maybe base-3's structural advantage is at LEAST PARTIALLY about handling dense/intermediate sparsity, not extreme sparsity (which is where B2-B-skip can amortize the mask overhead).

## Methodology check (against project rules)

CONTRIBUTING rules I should apply:
- **Substrate-novelty audit:** the cycle IS this. The B2-B comparison is the audit.
- **Multi-seed validation:** 5 seeds per config (per audit's pattern).
- **Multi-config gates the story:** 12 configs from audit; headline + per-regime.
- **Hypothesis vs finding:** each axis produces a verdict, not a hypothesis.
- **Match scope of evidence to scope of claim:** L1 layer only; cycle's claim doesn't extend beyond.

Memory rules:
- No consumer-demand framing: ✓
- No time/speed gating: gate on op count, not throughput. ✓
- No scalar in production: B2-B kernel has NEON path; scalar reference is verification only. ✓
- Substrate utilization vs comparative advantage: cycle IS the comparative test.

The methodology is already in place. The cycle's contribution is the B2-B kernel + the side-by-side measurement.

## Where I'd land

Ready to SYNTHESIZE. The reflection has surfaced:
- B2-B kernel structure pre-committed (uniform AND skip-aware variants)
- Disassembly methodology defined
- Headline regime identified
- Verdict thresholds pre-committed
- Honest expectation: regime-dependent verdict

The synthesis will tighten these into pre-committed gates with specific numerical thresholds.
