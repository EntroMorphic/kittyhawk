---
cycle: gesh_design
phase: REFLECT
date: 2026-05-01
scope: find structure beneath the nodes; resolve tensions; name the core insight
---

# Reflect — gesh_design

## Core insight (one sentence)

**Gesh is designed against attention's surface area, but the test of Gesh is whether routing-first base-3 beats base-2 attention on a specific task — and the design has no specific task.**

The architectural-unification claim ("attention-plus-FFN collapsed into routed retrieval") is the substrate's THESIS Part B demonstration. The three Gs are mechanism. The unification doesn't require three Gs; it requires one working example on a deliberately-chosen benchmark. The current design specifies three Gs because attention has three surface features (multiscale, learned manifold, gradient-trained) — that's reasoning from the wrong direction.

## Asking why three times

**Why three Gs?** Because pure LSH has three named failure modes (no global structure, random projections, no end-to-end training).

**Why those three failure modes?** Because they're the gaps between LSH and attention. Attention has hierarchy (transformers stack), learned embeddings (manifold awareness), gradient-based training (end-to-end optimization).

**Why is the design space shaped by attention's gaps?** Because the unification claim implicitly framed Gesh as "make LSH look like attention." But the substrate's pitch is the opposite — make attention's *function* run on a routing-first substrate, where many of attention's design pressures (joint training of dense weights with quadratic attention) don't apply.

The third "why" reveals the design has the wrong target. The substrate-claim is task-shaped: "routing-first beats dense on this benchmark." The Gesh design is attention-shaped: "make LSH match attention's surface." Those are different things.

## Structure beneath the nodes — the laundry method

Three buckets emerge:

### Bucket A — Mathematically/architecturally forced (no consumer dependency)
- **`apply_signed` as Gesh's accumulator** (Node 10) — correct as far as it goes, but the design's claim that Gesh exercises the cross-exp generalization is contingent on call pattern, which makes it a Bucket B item, not Bucket A.
- Nothing else here. The design has very few items that survive without consumer/task dependency.

### Bucket B — Consumer-shape contingencies (the cycle must validate)
- Three Gs orthogonal vs one principle (Node 2) — ablation tests it.
- Frozen bank vs joint training (Node 3) — task-dependent.
- Benchmark choice (Node 4) — load-bearing.
- MTFP4 vs ternary projections (Node 6) — measurement question.
- Kernel placement substrate vs libglyph (Node 7) — profile-dependent.
- PCA prior on training-data vs bank (Node 8) — geometry-dependent.
- Per-region vs shared local projection (§6.3) — capacity-vs-cost.
- Refresh frequency (§6.4) — dynamics-dependent.
- Cross-exp validation reality (Node 10) — bank-construction-dependent.

### Bucket C — Methodology / discipline / framing
- Six stacked provisionals (Node 5) — pre-commit to gates.
- All-three vs incremental (Node 11) — incremental is discipline-default.
- Research probe vs production infra (Node 12) — scope decision.
- Designed-against-attention vs designed-against-task (Node 13) — framing reset.
- Three training mechanisms simultaneously (Node 9) — sequencing decision.

### The boundary items (where mistakes hide)

**Node 13 sits on the C/B boundary.** "Designed against attention vs against task" reads as a methodology issue (Bucket C), but it's actually a consumer-shape question (Bucket B): the task determines what's useful. Treating it as Bucket C ("just reframe the design exercise") misses that it requires a Bucket B answer ("pick the task").

**Node 1 (unification claim) sits at the A/B boundary.** As an abstract claim ("routing-first beats attention"), it's Bucket A — the substrate's thesis. As a measurement, it's Bucket B — depends on what task. The design conflates them: it treats the unification claim as motivating *all three Gs*, when the claim only motivates *one working test*.

**Node 11 (build-all vs incremental) sits at the C/B boundary.** Methodology says incremental (Bucket C). The design's §7 framing ALSO says incremental — but the build plan says all-three. The contradiction is between methodology recognition and execution. The incremental path's sequencing (which G to add first?) is Bucket B — task-dependent.

The recurring pattern: **the design has many Bucket B questions but no Bucket B answers.** Because the benchmark hasn't been picked, the consumer-shape questions are open, and the design fills them with provisionals shaped by attention's surface area.

## Resolved tensions

### T1: Unification vs three-Gs mechanism — RESOLVED
The unification claim is the test. The three Gs are a hypothesis about what mechanism the unification needs. Build the simplest Gesh that could work; let ablation determine which Gs earn their place.

### T2: Frozen bank — SURFACED, NOT RESOLVED
The frozen-bank constraint is the design's central architectural commitment. Surfacing it as such is the resolution; whether it's correct depends on whether a frozen bank can match a joint-trained bank on the chosen task. Bucket B; benchmark-dependent.

### T3: Benchmark — SHIFTED TO ACTION
This is the cycle's first decision and gates everything else. Selecting it requires answering: "what's the smallest task that genuinely tests the unification claim?" Synthesize phase makes this concrete.

### T4: Six provisionals — RESOLVED IN STRUCTURE
Convert each §6 question to a pre-committed gate with a falsifiability measurement on the chosen benchmark. The benchmark turns provisionals into testable hypotheses.

### T5: MTFP4 default — RESOLVED
Flip default to ternary-first. Discipline-aligned; cheaper to test; escalates to MTFP4 only on measured insufficiency.

### T6: Substrate territory — RESOLVED IN DIRECTION
Build Gesh in libglyph end-to-end. Promote primitives to substrate when profile shows hot path or when libglyph composition is expensive. `threshold_extract2` is the only candidate; even that's empirical.

### T7: PCA prior — SURFACED
PCA-on-training-data and PCA-on-bank give different geometries. The cycle should make the choice explicit and measure both if cheap; default to PCA-on-bank if the bank is the routing target.

### T8: Three training mechanisms — RESOLVED IN SEQUENCING
Test STE alone first (with hard top-k, no Gumbel, no refresh). Add Gumbel only if STE fails to train. Add periodic refresh only if the projections drift too far from the bank's signatures. Each mechanism earns its place.

### T9: Cross-exp validation — DOWNGRADED
Gesh may or may not exercise the cross-exp generalization, depending on bank construction. Don't claim Gesh is the cross-exp kernel's first measured consumer until measurement shows the call pattern is genuinely cross-exp.

### T10: All-three vs incremental — RESOLVED
Incremental. The §7 ablation framing in the design IS the resolution; the build plan should commit to it.

### T11: Research probe vs production — RESOLVED PROVISIONALLY
Treat the first cycle as a research probe. The unification claim is the substrate-claim demonstration. Production-infra concerns (deployment, scaling, integration) are for later cycles, gated on the probe's success.

### T12: Designed-against-attention vs against-task — RESOLVED
Reset framing. The design starts from "what's the smallest task that tests the unification?" and asks what mechanism that task demands. Attention's surface area is reference, not template.

## Hidden assumptions surfaced

1. **The data manifold and the bank manifold are the same thing.** Often false; depends on bank construction.
2. **All three Gs are independently necessary.** Hypothesis, not derivation; ablation tests it.
3. **The frozen bank can match joint-trained attention.** The design's central architectural bet, unexamined.
4. **MTFP4 projections are needed for capacity.** Inverted; ternary should be tested first.
5. **Three training mechanisms compose stably.** Three novel mechanisms with unstudied interaction.
6. **MNIST is a reasonable validation.** Only if the unification claim is being tested at the level of "does it work at all"; insufficient for the substrate-claim narrative.
7. **`signature_match` deserves substrate residency.** It's a libglyph composition; promote on profile, not preemptively.
8. **The benchmark is interchangeable with the design.** Backwards; the benchmark determines the design.

## What I now understand

Gesh's most useful framing is: **a hypothesis test about whether attention's function survives the substrate-shape transformation.** The three Gs are the design's bet about *what mechanism* the surviving attention-substitute needs. The bet might be right (all three contribute), or partial (one or two contribute), or wrong (a simpler Gesh suffices). The cycle's job is to test the bet, not assume it.

The architectural-unification claim is what makes Gesh worth building. The three Gs are mechanism-hypotheses, ablation-testable. The benchmark is the decision that turns every other open question into a measurable gate.

The design as written is a research-probe-scoped artifact dressed in production-infra clothing. Strip the production framing back to research-probe; pick the smallest unification-test benchmark; build the simplest Gesh that could work; let measurement drive the build outward.

## Remaining questions for the synthesize phase

- What's the smallest task that genuinely tests the unification claim? (Concrete proposal needed.)
- What does "the simplest Gesh that could work" look like at the kernel/code level?
- What are the pre-committed gates for adding each G?
- How does the cycle handle the frozen-bank question — measure against a joint-trained baseline, or take the constraint as given?
