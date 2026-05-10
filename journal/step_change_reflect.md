# Reflect: next step-change

The nodes are symptoms; the structure beneath them is what matters. Trying to find the structure.

## Asking "why" three times on the gradient-kernels recommendation

**Why train?** Because it unlocks Phase 2/3 of the four-phase plan and lets the substrate be a producer of weights, not just a consumer.

**Why does that matter?** Because the deeper thesis question (Part B: routing is essential) requires consumers that are routing-native by design, and routing-native designs need to be trained.

**Why is testing Part B the load-bearing thing?** Because Part A (base-3 is the natural shape) is partially supported (one strong-claim layer + one real LLM running) and Part B is essentially untested. The thesis stands on both. Without Part-B evidence, the project has built well but hasn't earned its central claim.

So the chain is: training → routing-native consumer → Part-B test → thesis support.

Three hops between training and Part-B evidence. Each hop is months of work. The path is REAL but it's LONG.

## What would this look like if it were easy?

The easy version of "test Part B" wouldn't require traversing the three-hop chain. It would be:

> Find an existing inference workload where the substrate's routing primitives (threshold_extract, distance_batch, topk_abs, apply_signed) STRUCTURALLY win against a base-2 dense alternative at equal compute, and where the gap WIDENS as we add structural complexity (more classes, more modalities, more compositional structure).

This is what NORTH_STAR.md and THESIS.md actually call for. The R1 cycle attempted this and got falsified. But R1's falsification was specific (per-expression-tau dual-threshold doesn't outperform sign-only) — it didn't establish that NO routing operationalization could pass.

The easy version doesn't exist yet, but it's not impossible. It's UNDERTHEORIZED. The project hasn't spent serious time looking for it because:
1. R1's falsification was demoralizing in a small way
2. Other work (BitNet, strong-claim L1) provided clearer paths to demonstrable progress
3. "Find a benchmark" doesn't have the satisfying engineering structure that "build a kernel" does

## What am I assuming that might be wrong?

- **Assumption:** Step-change means "biggest single thing you can build next."
  - **Counter:** Step-change might mean "biggest shift in what's TRUE about the project." Building doesn't always shift truth; sometimes measurement does.

- **Assumption:** The four-phase plan is the right sequence.
  - **Counter:** It was written before the substrate had a real consumer. Now that BitNet works, the question of "does base-3 routing actually win at anything real" is more answerable than it was — and might be more important to ask than "let's build training."

- **Assumption:** Part-B requires a routing-native CONSUMER (a new architecture).
  - **Counter:** Part-B requires routing-native COMPUTATION on a workload where it wins. The architecture and the computation aren't the same thing. A workload could exist where applying the existing routing primitives in an inference-only setup demonstrates the gap.

- **Assumption:** I should optimize for capability extension.
  - **Counter:** The project might be in a state where extending capability ahead of evidence is exactly the wrong move. Build more, test less, claim density falls. The CONTRIBUTING.md scope-match rule warns about exactly this pattern.

## The structural insight (or attempt at one)

Looking at the nodes as a system: there are **two distinct strategic gaps** in the project, and they're confused with each other.

**Gap 1: Capability gap.** The substrate can't train. This limits what kinds of consumers can be built. Closing this gap is what "training" accomplishes.

**Gap 2: Evidence gap.** The substrate has Part-A evidence (some) and zero Part-B evidence. The thesis stands on both parts. Closing this gap is what "Part-B test" accomplishes.

**These are different gaps and they require different responses.**

If you're optimizing for capability, train. If you're optimizing for thesis support, find a Part-B test. Both are valid framings of "step-change," but they answer different questions.

My initial recommendation collapsed them by arguing "training enables Part-B." That's true but it's a 3-hop argument. A more honest framing is: "training is the largest CAPABILITY step-change available; finding a Part-B test is the largest CLAIM step-change available; choose based on what the project actually needs more of."

## Resolved tensions

**T1 (four-phase plan vs L2 strong-claim first):** false dilemma. The plan was a planning artifact under uncertainty. Honoring it isn't a discipline obligation if evidence has shifted. But L2 strong-claim is also not the answer — it's incremental Part-A work that doesn't address the bigger Part-B gap.

**T2 (training enables Part-B vs creates claim drift):** real tension. Training without strengthening Part-A first IS claim drift if we then make Part-B claims with training-derived consumers. The mitigation is sequencing: do enough Part-A confirmation that Part-B claims, when they come, can rest on it. But the Part-A confirmation can be modest — L2 strong-claim or a second model is enough; we don't need to exhaustively prove Part-A at every layer first.

**T3 (Part-B value vs probability):** the right framing is NOT "EV alone." Part-B's low probability is partly because it's UNDEREXPLORED. Spending some cycles to make it more tractable would raise the probability and thus the EV. The lowest-probability options are the ones with least search effort.

**T4 ("step-change" frame might be wrong):** I think the frame is right, but the content was wrong. The step-change isn't a SINGLE THING — it's a shift in what the project is ABOUT. From "build the substrate" to "test the substrate's central claim." That's a big shift in project mode.

## Core insight (one sentence)

**The next step-change is not a build; it's a shift from substrate-building to substrate-testing — specifically, from accumulating Part-A evidence to actively searching for Part-B evidence.**

What that looks like concretely:
- The PRIMARY work becomes "design and execute Part-B-relevant experiments"
- Capability work (training, kernel additions) is justified WHEN it's required by a planned Part-B experiment, not as the default mode
- Strong-claim L2/L4/L6 cycles are valued as Part-A foundations that support eventual Part-B claims, but they're NOT the primary focus
- The four-phase plan (inference → fine-tune → train → productize) is reinterpreted: not a march to productization, but a sequence of capability-building moves whose value comes from what THESIS-RELEVANT EXPERIMENTS each unlocks

The step-change is mode change. Not "what to build next" but "what to optimize for next."

## Remaining questions

1. What are the candidate Part-B experiments? My nodes listed three vague directions; none are scoped. The first concrete work in the new mode would be a focused search: list 5-10 candidate Part-B test designs, score them on tractability and informativeness, pick the most promising.

2. Does the user agree with the mode-shift framing? It's a bigger ask than "let's build gradient kernels."

3. What does the project look like if Part-B turns out to be FALSE? If we can't find a workload where base-3 routing wins, that's a thesis falsification. The honest version of mode-change is willingness to act on negative results.

## What I now understand

My initial recommendation (gradient kernels / training) was a CAPABILITY answer to a CLAIM question. The user asked about step-change for the project. The project's biggest gap is not capability; it's CLAIM SUPPORT for thesis Part B. Training is one capability among many that COULD eventually support that — but it's a long path with multiple hops, and each hop is months. A more direct path exists: spend cycles searching for tractable Part-B experiments, even if the search itself is uncertain.

The clean framing: **capability without claim support is engineering theater; claim work might fail to find a clean answer, but the failure itself is a finding.**

This is the discipline the project's journals already model (R1 was a Part-B test that got falsified — and the falsification is the finding). Returning to that mode is the actual step-change.
