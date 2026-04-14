---
date: 2026-04-14
phase: SYNTHESIZE
topic: How ternary routing helps M4T kernels and the training of models
---

# Synthesize

## Short answer

**Kernels:** Routing doesn't "help" M4T kernels — routing *is* the natural shape of M4T kernels. A base-3 substrate's kernels are routing-shaped by construction (conditional negate-add, TBL dispatch, masked VCNT, SDOT zero-skip). Asking how routing helps kernels in this substrate is like asking how indentation helps Python — it's the syntax, not a modifier.

**Training:** Routing does not yet "help" training in any demonstrated sense. Training in a ternary substrate is an **unsolved architectural problem**. Gradient descent is base-2-shaped; straight-through estimators reintroduce float; prototype accumulation is natively ternary but hits a capacity ceiling. Three ternary-native *refinement* primitives — sign-flip, exponent-shift, anti-signatures — surfaced in reflection as candidates, but none are built, none are measured, and none are proven to reach transformer-scale capability.

---

## Reframe

The original question presupposes a base-2 separation: kernels, routing, training are three independent artifacts. Under that framing routing is an optimization bolted onto each. Under the base-3 reframe:

- **Kernels ARE routing compositions.** `ternary_matmul` = conditional negate-add = routing on sign trit per weight cell. `distance_batch` = popcount over packed trits = routing on trit agreement. `apply_signed` = routing on tile decisions into MTFP result. There are no "non-routing kernels" in the substrate; every kernel that matters is a composition of routing-shaped primitives.

- **Training is an open question shaped by the substrate, not a paradigm the substrate inherits.** What a ternary-native training loop looks like is research, not engineering.

This is a substantive reframe. It means the thesis "routing will outperform dense" is not about adding routing to dense computation — it's about whether base-3-natural architectures (which are routing-native) can match or exceed base-2-natural architectures (which are dense-native) at equivalent task scope.

---

## Where ternary routing concretely helps the M4T kernels (once the substrate is clean)

These are measurable, if small.

**Depth composition.** N-layer routing networks amortize routing overhead over N matmuls/reductions. The regime where routing-overhead dominates (N=1, tiny tile count) is not the regime the substrate is built for. The break-even point has not been measured. **Empirical test:** vary N and tile count, measure overhead vs savings ratio.

**Structural zero-skip.** SDOT already delivers this at the primitive level. Routing at the tile level extends it: whole tiles are skipped when the routing decision excludes them, not just individual lanes. The multiplicative savings compound: if each layer routes to k of T tiles, compute scales with k/T per layer. **Empirical test:** compare full-T matmul vs k-of-T routing on representative tile sizes.

**Three-way dispatch.** Genuinely three-state routing (once sign_extract is replaced) distinguishes active-positive, inactive, and active-negative paths. The inactive path isn't just "compute zero" — it's "this computation is not present." That distinction matters in architectures where the zero-arm does something different (e.g., skip connection, identity, fallback). **Empirical test:** implement a three-way dispatcher kernel and measure against a two-way MoE-style analog.

---

## Where ternary routing could help training (speculative, candidates only)

None of these are built. All three emerged during reflection. Each is a substrate-level primitive that doesn't exist.

1. **Sign-flip refinement.** When a training example is misclassified, flip specific trits in the winning-but-wrong signature. Local, discrete, no gradients. Candidate primitive: `m4t_route_signature_refine(sig, example, label, learning_rate_in_trits)`. Open question: which trits to flip?

2. **Exponent-shift refinement.** Adjust the *block exponent* of a signature to re-weight its total influence without touching the trit pattern. Uses per-block-exponent metadata we've specified but not yet implemented. Coarser than trit-flip but cheaper. Candidate primitive: `m4t_route_signature_scale(sig, delta_exponent)`.

3. **Anti-signatures.** Accumulate positive prototypes and negative prototypes per class. Classify by net affinity (positive distance − negative distance). `apply_signed` supports this directly. Candidate primitive: `m4t_route_signature_update_signed(pos_sig, neg_sig, examples, labels)`.

These are sketches. The project doesn't have infrastructure to evaluate them yet. They're proposals for where research could go — not answers to "how does routing help training."

---

## What blocks concrete answers

In order of what's actionable:

1. **`sign_extract` is binary-shaped.** Until replaced with a ternary-native extractor, every "ternary routing" experiment tests binary routing underneath. Substrate-level fix.
2. **Hardware utilization is undischarged.** Claims about SDOT, TBL, VCNT being "native" have not been measured. Without this, "routing helps" stories lack empirical grounding.
3. **No benchmark bed.** MNIST's magnitudes defeat sign-based routing by construction. Thesis-relevant measurement requires a task where trit structure is load-bearing. `docs/THESIS.md` §4.
4. **No refinement primitive.** Prototype accumulation is static. Until a refinement operation exists in the substrate, "training" means "one-shot signature construction" — a research dead-end beyond trivial scale.

---

## Required edits / actions (what this cycle implies)

- [ ] **Substrate:** Design and implement a ternary-native extraction primitive. Deprecate (or retire) `m4t_route_sign_extract`. Name and API TBD — candidates discussed in the prior turn's response.
- [ ] **Benchmark:** Identify a task where trit/sign structure is naturally load-bearing. Candidates noted: near-duplicate detection, char-ngram text classification, sparse-signal tasks. Pick one by named consumer.
- [ ] **Hardware discharge:** Before claiming "routing helps" quantitatively, measure SDOT/TBL/VCNT utilization on the kept kernels. Minimal benchmark suite in `m4t/tools/` (deferred in `REMEDIATION_PLAN.md`).
- [ ] **Substrate spec update:** `M4T_SUBSTRATE.md` currently names routing primitives without naming what "training" means on this substrate. Add a §18 or a separate `docs/TRAINING.md` articulating that training is an open architectural problem, surveying the three refinement candidates (or whatever ones survive scrutiny), and marking it as research scope, not deliverable scope.
- [ ] **Thesis update:** `docs/THESIS.md` should name training as a separate open item, distinct from benchmark bed. "Can a ternary-native training paradigm match dense-trained capability at scale?" is a falsifiable claim that deserves explicit tracking.

---

## Success criteria for the cycle

- [x] Named the base-2 framing assumption in the original question.
- [x] Resolved the kernel-help question into a reframe (kernels ARE routing compositions).
- [x] Left the training-help question honestly open, with three concrete candidate paths.
- [x] Identified substrate-level blockers to empirical answers.
- [x] Produced actionable items that don't require committing to unproven research directions.
