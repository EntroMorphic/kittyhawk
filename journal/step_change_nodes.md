# Nodes: next step-change

Extracted from `step_change_raw.md`. Numbered for reference.

## Node 1: Step-change has multiple possible meanings

The term "step-change" can mean any of:
- **Capability change** (substrate gains a new ability — e.g., training)
- **Claim change** (substrate makes a new thesis-relevant claim — e.g., L2 strong-claim)
- **Generality change** (substrate proves it's not single-model-specific — e.g., second model)
- **Approach change** (substrate gains a new way of being USED — e.g., routing-native consumer)
- **Workflow change** (faster iteration on substrate-claim work — e.g., measurement infra)

Why it matters: the answer depends on which kind of step-change is wanted. My initial recommendation (training) optimized for capability change. The user may have meant something else.

## Node 2: The four-phase plan is prima facie evidence for "training next"

`gesh/bitnet/README.md` lists "inference → fine-tune → train-from-scratch → productize." Phase 1 is done. Phase 2 is fine-tune. So "training" has the structural argument.

Tension with Node 1: the plan was written before BitNet inference actually worked. The state of evidence is different now. Plans written under uncertainty can be sub-optimal once that uncertainty resolves.

## Node 3: Strong-claim L2 might be a real claim move, not just incremental

I dismissed L2/L4/L6 strong-claim cycles as "modest reward." But the L1 strong claim is one data point. L2 doubles the per-layer evidence — if it confirms the structural advantage, the "advantage at L1 generalizes" story strengthens; if it doesn't, that's important negative evidence about thesis Part B.

Tension with Node 2: if training is the named next phase, doing L2 first delays training. Sequence question.

## Node 4: Part-B is the most thesis-relevant question and least-tested

NORTH_STAR + THESIS Part B: "routing is essential, gap widens with task richness." Currently zero direct evidence for or against this on real workloads. R1 was an attempted Part-B test that got falsified — but R1's falsification was a falsification of one specific operationalization, not the broader claim.

Tension with Node 3: L2 strong-claim is Part-A evidence (substrate's hardware fit), not Part-B. Doing L2 doesn't move the Part-B needle.

## Node 5: Training is a prerequisite for serious Part-B work

The most thesis-relevant Part-B test is probably "design a routing-native architecture and show it beats a base-2 baseline at equal compute." That requires training. Inference-only Part-B experiments are possible but limited (would need analytically-derived weights, which is a research project of its own).

Tension with Node 4: Part-B is the right destination but training is on the path to it. Sequencing argument again.

## Node 6: Step-change vs expected value are different optimization targets

Step-changeyness = magnitude of impact if it works. Expected value = magnitude × probability of success. Training is high step-changeyness AND high expected value because the path is clear. Part-B benchmark is the highest step-changeyness but lowest probability (we don't know what the benchmark looks like).

Tension with Node 4: even though Part-B is most thesis-relevant, lower probability means lower EV, which weakens it as a recommendation.

## Node 7: "Second model" addresses a real and underacknowledged gap

The substrate has ONE real consumer (BitNet). Claims like "the substrate runs ternary LLMs" rest on n=1. Porting Mistral or Llama to inference-only on the substrate would convert n=1 to n=2. The kernel surface is mostly shared, so the work is much less than the original BitNet port.

Why this is important: it's evidence about the GENERALITY of the L1 strong claim and the BitNet inference result. Without a second consumer, "substrate runs ternary models" is "substrate runs THIS ternary model."

## Node 8: Workflow / measurement infrastructure has hidden leverage

The current measurement loop is slow: each battery run takes 20 minutes; HF cross-checks take an hour; per-layer ε comparison requires custom Python each time. Investing in faster iteration would compound — every subsequent experiment runs faster.

Tension with calling it "step-change": it's a meta-change, not a thing-change. Step-changes in the conventional sense are about WHAT you can do, not HOW FAST you can do it.

## Node 9: Hidden assumption — "step-change" implies one thing

The project mode is mostly sequential single-track. So "what's the next step-change" naturally reads as "pick one." But this might be a constraint of the project's current operating mode rather than a feature. Could the next step-change be "shift from sequential single-track to parallel"? That's a workflow change.

## Node 10: Hidden assumption — I should recommend the option that maximizes "step-changeyness"

Maybe the user wants the option that maximizes EXPECTED VALUE OF PROGRESS ON THE THESIS. Those are different. By the second criterion, training and L2-strong-claim are closer in value than my initial recommendation suggests.

## Node 11: Thesis Part-A vs Part-B — claim density vs claim breadth

Part A (base-3 is the natural shape of hardware): substantial evidence. L1 strong claim. BitNet inference works. Multiple kernel-level wins.

Part B (routing is essential, gap widens with task richness): essentially no evidence. R1 falsified. No other Part-B-specific experiments since.

Tension: the project has been building Part-A evidence robustly while Part-B sits unaddressed. The thesis stands on BOTH. A step-change that doesn't move Part B doesn't advance the thesis as a whole.

## Node 12: The "claim drift" risk

If we build training infrastructure and then test routing-native architectures with it, we'll be making Part-B claims while Part-A is only validated at one substrate layer (L1). This is the methodology debt the CONTRIBUTING.md repeatedly warns about: "the N (or scope) of the evidence must match the N (or scope) of the claim." A routing-native consumer that beats a baseline would be claimed as Part-B evidence; but if Part-A only has L1 evidence, the routing win could be "L1 efficiency" not "routing essentiality."

Tension with Nodes 5 + 6: training is a prerequisite for Part-B, but doing training without first strengthening Part-A creates claim density debt.

## Node 13: Inference-only Part-B is undertheorized

Could there be a Part-B test that DOESN'T require training? Examples:
- Construct (analytically or by post-hoc routing of a trained model) a routing-native inference architecture; show it matches a dense baseline at equal compute on a task.
- Show that adding routing structure to an existing inference path (e.g., BitNet attention) improves a metric without retraining.
- Find a benchmark where the substrate's existing routing primitives, applied to a specific input distribution, structurally beat a base-2 alternative.

None of these are clearly defined or easy. But IF one were tractable, it would let us touch Part B before training. Worth thinking about.

## Tensions summary

- T1 (Node 2 vs Node 3): four-phase plan says train next; L2 strong-claim says strengthen Part-A first.
- T2 (Node 5 vs Node 12): training enables Part-B but creates claim-density debt if Part-A isn't strengthened first.
- T3 (Node 4 vs Node 6): Part-B is most valuable but lowest EV; training is high EV but doesn't directly address Part-B.
- T4 (Node 9): "step-change" might be a wrong frame — could the next move be a workflow shift to parallel tracks?

## Emerging solution paths (not committing yet)

- A: train next (default; aligns with plan; high EV; defers Part-B test).
- B: L2 strong-claim next (deepens Part-A; supports later Part-B claim density; modest in isolation).
- C: search for a tractable inference-only Part-B test (high impact if found; uncertain scope).
- D: port a second model to inference (validates generality of n=1 result; cheap given existing kernels).
- E: invest in measurement infrastructure (compounding leverage; unclear "step-change" framing).
- F: do A + B in parallel (workflow shift; bigger lift than the project's current mode).
