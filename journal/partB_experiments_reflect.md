# REFLECT: Part-B experiment candidates

Per `cycle1_plan_v2.md`. Primary task: sharpen Part-B operational definition.
Secondary: categorize candidates by what they test, resolve tensions, surface
assumptions.

## Sharpening Part B (the load-bearing part)

THESIS.md Part B as written:
> Routing is essential, not optional. As task complexity rises, the gap
> [between routing-native and dense] should widen, not close.
> Falsification: routing-native matches dense at *equal* compute, AND the
> routing advantage does not widen with task structure.

Three sub-claims need operationalization:

### "Routing is essential"

Operationalization: removing the routing structure from the model degrades
performance more than removing equivalent dense capacity would. Or, more
testable: at fixed compute, routing-native > dense by a margin that exceeds
noise.

What "routing" means in this project: the substrate's specific primitives
(threshold_extract, distance_batch, topk_abs, apply_signed). Not "any
conditional computation." This narrows the test — generic MoE-vs-dense
results don't validate Part B; substrate-routed-vs-dense does.

### "Equal compute"

The thesis statement leaves this ambiguous. Three interpretations:
1. **Matched FLOPs**: same number of multiply-adds at inference. Cleanest;
   most common in ML literature; favors substrate (its routing primitives
   skip computation rather than approximating it).
2. **Matched parameter count**: same number of stored weights. Less natural
   for routing comparisons (routing decisions consume few params but
   skip many weights at inference).
3. **Matched wall-clock or energy**: hardware-realistic. Hardest to
   measure cleanly because of cache effects, parallelism, etc.

For Cycle 2, **matched FLOPs at inference** is the right operationalization.
It's measurable, it's the literature standard, and it isolates the routing
question from hardware-specific concerns.

### "Gap widens with task structure"

What's a "task structure" axis? Candidates:
- Number of classes
- Sequence length
- Compositional depth
- Number of modalities
- Number of distinct rules / functions / operations

Each makes the trajectory test slightly different. The trajectory must be
on a SINGLE axis where increasing the value clearly increases task richness.
Multi-axis trajectories conflate variables.

For the Part-B claim to land, we need a trajectory test where: at low
complexity, routing ≈ dense; at high complexity, routing > dense. NOT
"routing always wins by a constant margin" — that would be a substrate-
efficiency claim, not a routing-essentiality claim.

## What makes a Part-B experiment useful — categorization

Three axes:

**EXISTENCE.** Does routing-native achieve > dense on at least one
workload at compute-parity? (One data point.) Necessary but weak alone.

**TRAJECTORY.** Does the gap widen along a complexity axis on at least one
workload? (Multi-data-point pattern.) Sufficient if the workload is a
fair test.

**MECHANISM.** Is the win actually about substrate-routing-essentiality,
or about something else (sparsity efficiency, regularization effect,
quantization noise)? Requires controls.

The strongest Part-B candidate supports all three. Most candidates only
support one or two.

## Surfaced assumptions and challenges

- **Assumption:** "Routing" = substrate's specific primitives, not generic
  conditional computation. **Challenge:** this is the right narrowing for
  THIS thesis but it means generic MoE wins don't transfer.

- **Assumption:** "Dense" = no conditional computation. **Challenge:** dense
  in what sense? Same params? Same activation pattern? Same FLOPs? The
  test design must specify which.

- **Assumption:** Part B is binary (true or false). **Challenge:** the
  honest answer might be partitioned — true on workload class X, false on
  class Y. The test design should allow for this finding.

- **Assumption:** The substrate's primitives are "natural" so workloads
  that fit them will look favorable. **Challenge:** this is tautology
  risk. The substrate-novelty audit is the defense.

- **Assumption:** R1's falsification doomed the signature-based bucket
  (N5, N6) entirely. **Challenge:** R1 falsified ONE specific signature
  rule (per-expression-tau dual-threshold). Other signature rules might
  pass — but the burden of proof is now higher.

## Resolved tensions

### T1 (tractability vs informativeness) — RESOLVED via parallel tracking

**Don't pursue both at once with one project.** Cycle 2 picks the
highest-tractability candidate that is also reasonably informative. In
parallel, design (not execute) the highest-informativeness candidate as
a future Cycle 3, with capability prerequisites listed. This way the
short-term work is tractable AND the long-term path is mapped.

### T2 (training vs inference bucket sizes) — the synthesis's framing SURVIVES if N4 is strong

The inference-only bucket has 4-6 candidates; only one needs to be
strong enough to be Cycle 2's choice. **N4 (post-hoc sparse attention)
qualifies.** It's compute-parity definable, trajectory testable,
substrate-distinct, not R1-vulnerable. The synthesis's mode-shift
framing survives this test.

If N4 turns out (during Cycle 2) to be unworkable for some currently-
unknown reason, the framing is then in question and we'd loop back to
`step_change_synth.md`.

### T3 (existence vs trajectory vs mechanism) — strong candidates support 2-3

- **N4** supports existence + trajectory + (with controls) mechanism. ★
- **N7** (compression measurement) supports existence + trajectory + (with
  bits-per-cell as the mechanism axis) mechanism. ★
- **N1, N2, N3** support existence (with training); trajectory requires
  more design.
- **N5, N6** mostly support existence; trajectory unclear.
- **N15** (image) supports existence with high probability of negative
  result.

### T4 (substrate-distinctiveness vs accessibility) — honest scoring

- N5, N6 (signature-based on trit-shaped tasks): tautology risk → score
  substrate-distinctiveness HIGH but workload-fairness LOW.
- N15 (image): workload doesn't fit substrate → score workload-fairness
  HIGH (it's a real workload) but substrate-distinctiveness LOW
  (the routing primitives don't have a natural fit here).
- N4: workload (BitNet attention) is real LLM inference; substrate-routing
  is being USED (not catered to) → score both HIGH. ★
- N1, N2: routing-native architectures on a real LLM workload → score
  both HIGH. ★

### T5 (compute-parity definability) — N4 and N7 are unusually clean

These are the cleanest comparison candidates. N1-N3 require designing
matched-FLOPs baselines, which adds friction.

### T6 (R1's failure mode) — excludes signature-only candidates

N5 and N6 are R1-vulnerable. They're not auto-disqualified, but their
prior probability of producing Part-B evidence is reduced. To pursue
them, we'd need a stronger argument that the third state is load-bearing
in the specific signature rule used.

## Structural insight (one sentence)

**The strongest Part-B candidate is post-hoc sparse attention via the
substrate's route_topk_abs primitive (N4): it's the only candidate that
combines compute-parity-definable, trajectory-testable, substrate-distinct,
mechanism-testable-with-controls, AND inference-only.**

What that means concretely:
- Take BitNet inference as shipped.
- At each attention step, compute Q·K signature distances (substrate
  primitives), use route_topk_abs to pick the top-k K positions per Q,
  dot-product only on those.
- Compare three controls:
  1. Dense attention (the current substrate baseline at gate1+fudge2)
  2. Substrate-routed top-k attention (the experimental arm)
  3. Random top-k attention (sanity-check baseline — if substrate routing
     is no better than random, the routing primitive isn't earning its
     place)
- Plus an oracle control: post-hoc best top-k (selected with knowledge
  of the actual attention weights). Bounds the BEST possible top-k can do.

Trajectory: vary k from 1 to head_dim. The Part-B-relevant question:
- At small k (high routing aggressiveness), does substrate-routed sparse
  attention degrade more gracefully than random top-k?
- Does the gap between substrate-routed and random GROW as k decreases?

Mechanism: if substrate-routed sparse attention sits between random and
oracle in quality, AND the gap to random grows as k decreases, then the
substrate's routing primitive is doing real work — Part-B EVIDENCE.

If substrate-routed sparse attention is indistinguishable from random,
that's Part-B FALSIFICATION for this workload.

## Cycle 3 design (parallel work, not Cycle 2)

The highest-informativeness candidate is **N1 (routing-native attention)**.
It's the architectural test of Part B — can we DESIGN attention to use
substrate routing natively, train it, and beat dense BitNet attention at
matched FLOPs?

Cycle 3 prerequisites:
- Training capability (gradient kernels for substrate primitives)
- A baseline routing-native attention design

Cycle 3 design work in parallel with Cycle 2 execution: enumerate the
gradient kernels needed, sketch the routing-native attention architecture,
identify which kernels are reusable from Cycle 2.

This makes the synthesis's mode-shift more honest: Cycle 2 is INFERENCE-
ONLY Part-B testing; Cycle 3 is TRAINING-ENABLED Part-B testing; both
are sequenced rather than competing.

## What I now understand

Part B is testable with current substrate capabilities. N4 is the candidate
that lets us test it without first building training. The synthesis's
mode-shift framing was right but the SCOPE of what's tractable in Cycle 2
is narrow (essentially one strong candidate, N4). That narrowness isn't a
problem — Part-B work has one good experiment to start with, which is more
than the project had yesterday.

The R1 falsification is more useful than I initially thought: it gives
us a calibrated example of what "Part-B falsification on a specific
operationalization" looks like. The discipline is to bring that same
willingness-to-falsify to N4.

## Remaining questions to take into SYNTH

1. What's the right scoring rubric for the SYNTH table? I have 5 axes from
   v2 plan: tractability range, informativeness positive, informativeness
   negative, substrate-distinctiveness, operationalizability. Add: trajectory-
   testable, mechanism-testable.
2. How should I score "needs more research" candidates? They can't be
   compared on the same axes as scored-confidently candidates. Maybe a
   separate column listing what's needed before scoring is possible.
3. The substrate-novelty audit per CONTRIBUTING.md — should it be a
   PASS/FAIL gate or a graded score? I lean toward graded, with a note
   that anything failing it strictly is on weaker thesis-evidence ground.
