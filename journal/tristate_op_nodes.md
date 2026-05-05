# NODES: useful operationalization of the tri-state potential

Atomic claims extracted from `tristate_op_raw.md`.

## What R1's failure tells us

- **N1.** R1's third state became a SINK at arity-1 (66.5% zero-band). The failure mode is: deriving the third state from analog quantization makes it the residual, not a meaningful class.
- **N2.** Operationalizations that THRESHOLD analog values to ternary risk this failure mode. The third state has to mean something the algorithm explicitly USES, not just be "below |τ|."
- **N3.** Avoiding quantization isn't an option — the substrate IS ternary at multiple layers (mtfp, matmul, mtfp4). The question is at the CONSUMER layer: do the third state's outputs carry information the consumer's algorithm depends on?

## The substrate's existing third-state utilization

- **N4.** m4t_mtfp / packed trits: 0 trit = "this cell contributes nothing to this block." Sparsity. Structural.
- **N5.** Ternary matmul (vmlal_s32 pipeline): zero trits structurally skip MAC contribution. Same sparsity story as N4.
- **N6.** shift3 / cross-exp accum: divides by 3^k via magic-multiply. Trit-aware in number-system sense; doesn't directly use the third state for semantic distinction.
- **N7.** mtfp4: packed 4-trit format with shared exponent. Sparsity story matches N4/N5.

- **N8.** In all of N4-N7, the third state's "load" is **structural sparsity**. This is real but not unique to base-3 — base-2 sign-bit + sparsity flag matches it. To distinguish base-3 from base-2 with extra bits, the operationalization has to do MORE than sparsity.

## Where base-3 specifically beats base-2

- **N9.** Balanced ternary arithmetic. {-1, 0, +1} as native number representation. Multiplication is sign-aware natively (trit × trit = trit). Addition has different carry structure (sum of two trits = trit + trit-carry). Structural property of the number system; not a quantization choice.
- **N10.** Signed encoding without sign bit. 1.58 bits per signed value vs 2 bits for sign+magnitude. Density advantage at extreme low-bit.
- **N11.** Three-valued logic (Kleene). {True, False, Unknown} where Unknown is computationally meaningful — a deliberate output, not residue. Useful for symbolic reasoning, equivalence checking, control flow.
- **N12.** Don't-care / mask semantics. 0 means "this dimension is masked / not applicable" — distinct from sparsity (=zero contribution). Requires consumer interpretation.
- **N13.** Halt / resolve in tree-structured computation. Routing tree node decision is naturally ternary {go-left, go-right, halt-here}.
- **N14.** GF(3) operations. Galois field arithmetic. Algebraic properties (associativity, distributivity, inverses) over 3 elements. Whether useful for ML is open.
- **N15.** Trit-state gating in MoE / attention. {-1: suppress, 0: pass-through, +1: amplify} per expert / dimension. Distinct from binary {block, pass}.

## Tier ranking of candidates

- **N16.** TIER A (structural; third state required by the algebra): N9 (balanced ternary accum), N13 (halt/resolve in routing trees).
- **N17.** TIER B (semantic; third state has distinct meaning at consumer layer): N12 (don't-care/mask), N15 (trit-state gating).
- **N18.** TIER C (interpretive; third state derived from quantization — R1's failure mode): any thresholding scheme. Avoid.
- **N19.** TIER D (algebraic / specialized): N14 (GF(3)). Off-track for ML primitives.

## Cycle scope options

- **N20.** Option A: SURVEY-style audit. Measure third-state utilization across substrate layers on a representative workload. Identify where third state is load-bearing vs sink-like vs under-exploited. Output: a ranking of operationalization candidates by evidence, not opinion.
- **N21.** Option B: DIRECT test of one TIER A or TIER B candidate. Pre-commit test gates. Output: a verdict on whether that candidate makes the third state load-bearing.
- **N22.** Two-cycle plan: A first (survey) → B (test the highest-leverage candidate).

## Substrate-touching constraint

- **N23.** "Support our substrate" means using the kernels we have (mtfp, matmul, cross-exp accum, magic-multiply). Operationalization should EXERCISE these, not propose entirely new kernels.
- **N24.** Operationalizations that need new kernels (e.g., balanced ternary accumulator with native trit-overflow) are valid in scope but heavier. Pure-software operationalizations on top of existing kernels are lighter.

## Falsifiability constraint

- **N25.** R1 failed because gates didn't bite — "did the rule do something different" is necessary but not sufficient. Gates have to be hard to fake AND hard to satisfy by accident.
- **N26.** "Did this expression work" pre-committed gate must measure: does the third state CARRY information the algorithm depends on? Not "is it present" or "is it different from binary."
- **N27.** Information-theoretic measurement (Shannon entropy near log2(3) for the third state's distribution; mutual information between third-state outputs and a target the algorithm cares about) is one way to make the gate bite.

## Methodology constraints

- **N28.** Per memory: no consumer-demand framing. The cycle is foundational; doesn't gate on measured demand.
- **N29.** Per memory: function over speed. Cycles gate on correctness / information content, not throughput.
- **N30.** Per memory: no scalar in production. If the cycle produces production code, NEON-only.
- **N31.** Per CONTRIBUTING: substrate-novelty audit. The operationalization must demonstrate that base-3 specifically (not base-2 + extra bits) is the right substrate.
- **N32.** Per CONTRIBUTING: multi-config gates the story. At least 2 configurations (e.g., 2 problem sizes, 2 distributions) before drawing conclusions.

## Risk register

- **N33.** Pattern-matching from R1: the natural overcorrection is "don't threshold." But the substrate ALREADY thresholds at multiple layers. The cycle has to address the third state's role at the CONSUMER layer, not just at the storage layer.
- **N34.** Toy-task selection bias: any direct test on a synthetic task can be tuned to make the third state work. Mitigation: pre-commit the task setup and the gate; don't iterate on the task to make the gate pass.
- **N35.** Multi-axis claim trap: "tri-state potential is useful" is bigger than any one operationalization. The cycle should produce a single clean test result, not promise to validate the whole vision.
- **N36.** Substrate-changes-during-cycle: shouldn't happen since this is mostly measurement, but worth noting that any substrate change between measurement and analysis voids the data.

## What this cycle is and is NOT

- **N37.** IS: an LMM cycle on a strategic question — which operationalization of the third state is most worth pursuing next?
- **N38.** IS: pre-committed gates for the chosen operationalization, ready for execution.
- **N39.** IS NOT: validation of the whole vision claim 3.
- **N40.** IS NOT: a perf cycle.
- **N41.** IS NOT: a re-run of R1 in a different guise.

## Open questions for REFLECT

- **N42.** Survey vs direct test? (Two-cycle plan vs single deep cycle)
- **N43.** If direct test, which candidate? N9 (balanced ternary accum) is most structural; N15 (trit-gating) is most ML-direct.
- **N44.** What's the gate-design pattern that won't repeat R1's "didn't bite" failure?
- **N45.** What workload / task setup is the right shape — synthetic enough to be tractable, realistic enough to be informative?
