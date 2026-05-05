# REFLECT: useful operationalization of the tri-state potential

Cold-eye review of `tristate_op_nodes.md`.

## Load-bearing nodes

- **N1, N2** — the failure-mode diagnosis. R1's third state became a sink because it was DERIVED from quantization rather than produced as a deliberate output. Any next operationalization has to invert this: the third state is an OUTPUT of the algorithm, not a residue.
- **N8** — the substrate currently uses the third state for SPARSITY only, which doesn't distinguish base-3 from "base-2 + sparsity flag." Distinguishing base-3 requires the third state to do something more than be zero.
- **N16, N17** — the tier ranking. TIER A (structural) beats TIER B (semantic) because TIER A is harder to fake and easier to falsify. TIER C is the failure zone.
- **N25, N26, N27** — gate design. R1's gates didn't bite. The fix: information-theoretic gates (entropy, mutual information) where "did it work" is hard to satisfy by accident.

## Weak nodes

- **N9 (balanced ternary accumulation)** — structurally appealing but unclear what's actually MISSING from the existing cross-exp accum. The cross-exp accum already accumulates across mixed-exponent ternary blocks. What would "balanced-ternary native arithmetic" add that the current substrate doesn't? Need to specify the GAP, not just propose another implementation.
- **N15 (trit-state gating)** — ML-relevant but requires a target task. Without a target task, "does it route better" is unmeasurable. With a target task, selection bias is a real risk.
- **N20 / N21 (survey vs direct test)** — both have merit. The survey is informative but doesn't pre-commit a falsifiable test. The direct test pre-commits but risks picking the wrong candidate. The two-cycle plan (N22) splits the cost.

## Tensions

- **N23 (substrate-touching) vs N9 (balanced ternary accum)**: Balanced ternary accum might require new low-level kernels (trit-overflow accumulator), which is heavier than "exercise the existing substrate." The substrate-touching constraint pushes toward operationalizations that USE the kernels we have on existing data shapes, not ones that need new kernels.
  - Resolution: pre-commit the constraint. If the cycle's deliverable requires a new kernel, that's a multi-cycle commitment. The first cycle should produce evidence within existing substrate.
  
- **N20 (survey) vs N21 (direct test)**: the user's prompt — "deploy LMM on how it can be usefully operationalized" — could mean either. Survey produces a directional finding; direct test produces a verdict.
  - Resolution: I'm inclined toward survey FIRST — the R1 cycle's lesson is that picking the wrong test wastes effort. A measurement-first cycle that ranks candidates by evidence is more disciplined than betting on one candidate.

- **N31 (substrate-novelty audit) vs the "use existing kernels" framing**: substrate-novelty would normally ask "is this kernel doing something base-2 can't do." But the existing kernels were built to BE substrate-novel — the question now is whether they're being USED in a way that exercises that novelty.
  - Resolution: the substrate-novelty axis here is at the USER / CONSUMER layer. Are downstream consumers (signature derivation, MoE gating, attention) using ternary in a way that DEPENDS on the third state's distinct semantic? If they're treating ternary as "signed int with extra zero," the substrate's novelty is wasted.

## Missing information

- **M1.** How is the third state distributed in our existing substrate's outputs on real workloads? We don't know without measuring. Audit-style cycle would generate this evidence.
- **M2.** Is there a workload (synthetic but plausible) where we could compare {base-3 with active third state} vs {base-3 with third state collapsed to "off-bit"} vs {base-2 sign+magnitude}? If we can't construct such a workload, the substrate-novelty claim isn't testable on this substrate.
- **M3.** What does "load-bearing" mean operationally? Two candidates:
  - (a) Information-theoretic: entropy of third state's outputs ≈ log2(3) per cell (roughly equal to the other two states), AND mutual information with downstream task output is non-zero.
  - (b) Algorithmic: removing the third state (collapsing it to one of the other two) measurably degrades the algorithm's downstream performance.
  Definition (b) is harder to fake but requires a downstream task. Definition (a) is easier to measure but can be satisfied by uniformly random third states (no actual signal).
- **M4.** What's a representative workload? "Real LLM" is too heavy. "Toy synthetic" is too easy. Middle ground: a small ternary GEMM with weights sampled from a known distribution that mirrors trained 1.58-bit LLM weight stats; activations from a real input sequence.

## What I'd want before declaring "ready to execute"

In rough priority:

1. **A pre-committed definition of "load-bearing."** Probably both (a) and (b) above as separate gates: information-theoretic AND algorithmic. Either gate alone is fakeable.

2. **A specific workload spec.** Multi-config (per CONTRIBUTING N32). At least: {small synthetic GEMM, sampled-from-realistic-distribution weights, realistic activation patterns} × {2-3 sizes for size-scaling check}.

3. **A specific operationalization to measure.** I'm inclined toward an audit-style cycle: measure third-state utilization at every layer of the existing substrate (mtfp, matmul, accum) on the workload, and identify where the third state is load-bearing vs sink-like vs under-exploited.

4. **A clear handoff to the next cycle.** The audit's output is a ranked list of candidates, not a final verdict. The cycle should commit to: "after audit, the highest-leverage candidate becomes the next cycle's target."

## What I might be wrong about

- **The "survey first" framing** could be over-cautious. If I had high confidence in one candidate (say, balanced ternary accum), going direct would be more decisive. But I don't — the R1 cycle's lesson is that confidence without evidence is the failure mode.
- **The "information-theoretic + algorithmic" double gate** might be too rigorous for a first cycle. Could split: this cycle uses information-theoretic only, next cycle uses algorithmic. But information-theoretic alone is fakeable (uniform-random third state passes entropy gate).
- **The "small synthetic GEMM" workload** could be too simple — might not exercise the third state in ways a real workload would. Mitigation: validate the workload's third-state distribution matches a published 1.58-bit LLM weight stats before drawing conclusions.

## Honest framing

The cycle's contribution is going to be: an audit of where the third state is and isn't load-bearing in our existing substrate, with information-theoretic measurements at each layer. The output is a directional finding (which candidate to pursue next), not a final verdict on claim 3.

If the audit shows the third state is load-bearing somewhere unexpected (e.g., in cross-exp accum's exponent transitions), that's a finding. If it shows the third state is uniformly sink-like across the substrate, that's a finding too (suggests claim 3 needs different operationalization, possibly outside the existing kernels).

Either outcome moves the project forward. Audit cycles are low-risk in the sense that they generate evidence regardless of whether the underlying claim holds.

## Methodology check (against project rules)

CONTRIBUTING rules:
- **Substrate-novelty audit:** applies directly. The CYCLE IS a substrate-novelty audit on the existing kernels' actual usage of the third state.
- **Multi-seed validation:** apply to any random workload generation.
- **Multi-config:** at least 2 sizes / distributions.
- **Hypothesis vs finding:** each layer's audit produces a finding (load-bearing / sink-like / under-exploited), not a hypothesis.
- **Match scope of evidence to scope of claim:** the audit measures CURRENT substrate's third-state utilization. Doesn't claim to validate / invalidate operationalizations not yet built.

Memory rules:
- No consumer-demand framing: ✓ (audit is foundational)
- No time / speed gating: ✓ (audit gates on information content)
- No scalar in production: ✓ (audit doesn't produce production code, only journal evidence + measurement scripts)
- Six primitives floor: relevant context — the audit's output should help direct the project toward primitive-level operationalizations
- Math as routing signatures: relevant — third state could be load-bearing in routing specifically
- Base-3 carries information base-2 collapses: the cycle's central testable claim

## Where I'd land

Audit-style cycle. Pre-commit:
- Workload spec (multi-config, multi-seed)
- Information-theoretic gates per substrate layer (entropy, mutual information with downstream output)
- A second algorithmic gate where feasible (does removing the third state degrade the layer's output?)
- Output: a ranked list of layers by third-state utilization gap (potential vs actual). The next cycle pursues the highest-gap candidate.

This is methodically conservative — generates evidence before betting on a candidate. Trade-off: takes longer to reach a verdict on claim 3 broadly. But the user explicitly disclaimed time as a constraint (per memory).
