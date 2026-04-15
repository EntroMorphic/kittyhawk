---
date: 2026-04-15
scope: LMM cycle — online, inline meta-router
phase: REFLECT
---

# REFLECT

## Core insight

**The observer is not a predictor of failure. It is a routing primitive whose *keys* are routing-context signatures and whose *values* are routing decisions, and whose bank grows from the residue of cascade execution itself.**

Everything in NODES tangles until the question shifts from "how does the observer know?" (an epistemology question — how can you predict failure without labels) to "how does the observer route?" (a routing question — what's the signature, what's the bank, what's the lookup). Once reframed, the answer is the same primitive used everywhere else in Glyph: a signature, a bank, a k-NN, a decision. The only architectural novelty is that the bank grows over time.

The observer reads context that *local routing already computed as a byproduct* — H1 top-1 distance, tied-count, H2/H3 disagreement, H1-H2 rank correlation. These are emitted by the cascade's own state during normal execution. Packing them into a short trit vector produces the *routing-context signature*. The meta-router's lookup is: given this context, what did past queries like this one need?

Self-supervised "ground truth" dissolves as a problem because the meta-router's update rule is not "label this query correct/wrong." It is "append the context signature, tagged with whatever action path the cascade actually took and whatever self-supervised signal was observed at the end (final margin, final agreement, final K needed to stabilize)." The bank accumulates *behaviors*, not *truths*. At inference, k-NN over the bank surfaces the behavior patterns most similar to the current query, and the routing decision is a majority-of-behaviors among the neighbors.

This is a radical move worth naming: **the meta-router learns which cascade-execution traces are neighbors of which, not which queries are failures.** Failure never enters the update rule. Drift via self-confirmation is eliminated because there's nothing to confirm — the bank stores observed cascade state, not opinions about correctness.

## Resolved tensions

**T1 resolved (anticipation vs correction).** The meta-router is *correction-shaped* by default: local cascade runs first, emits its intermediate state, the meta-router reads that state and decides whether to commit the local answer or escalate. Anticipation reduces to the special case where the "state" is just the query signature with empty cascade history — useful for a fast-exit decision before the cascade even starts, but not the primary mode.

**T2 resolved (online learning without labels).** Self-supervised labels are not required because the bank stores context signatures, not labeled examples. The update rule is append-on-execution, not append-on-error. Closed-loop drift is avoided structurally.

**T3 resolved (learning target vs action vocabulary).** Start with a binary action vocabulary: accept-local vs escalate-global. The bank entry schema is `(routing_context_sig, action_taken, self_supervised_signal_at_end)`. k-NN lookup returns the dominant action among the nearest M bank entries. Expand vocabulary later if the binary case measures positively.

**T4 resolved (self-confirmation vs exploration).** Because the bank stores behaviors not correctness, there is nothing to self-confirm. Exploration is still needed — occasionally forcing the non-dominant action to populate the bank with diverse behaviors — but the motivation is coverage, not bias correction. Decay is still useful to shed stale entries under distributional shift, but not load-bearing.

**T5 still binding (prerequisite measurement).** The cycle produces a design. The design is only worth executing if global *is* better than local on local-failures. Prerequisite measurement is still load-bearing — it's just now a go/no-go filter on the synthesize phase, not an obstacle in the design phase.

**T6 resolved (is this different from a learned threshold).** Yes, structurally. A learned threshold over a scalar self-supervised signal can route to only the actions that signal was designed to trigger. A meta-router over routing-context signatures can route based on arbitrary combinations of signals — margin AND tied-count AND disagreement AND OOD — without pre-specifying which combinations matter. The bank learns combinations from experience. That is genuinely more expressive than "escalate when margin < tau."

## Prediction

**The meta-router is worth building if and only if the prerequisite measurement passes.** Conditional on it passing, I expect:

1. Aggregate accuracy between pure local (~83.86% at N_PROJ=16 quadruple) and pure global (~91.55% equivalent at N_PROJ=16×4). Rough interpolation: around 87-89% if ~30-40% of queries escalate. Actual depends on how well the context signature separates easy from hard queries.

2. Cost between 1.0× and 1.5× of pure local, dominated by the primary H1 pass plus meta-router k-NN lookup. Global escalation fraction determines the tail.

3. The bank will saturate after a few thousand queries. Growing beyond that adds little because context signature space is small (a few bits per signal × a few signals ≈ 20-30 bits of context, ~1M possible contexts, most never reached).

4. Exploration will matter more than decay on MNIST because MNIST is stationary. On drifting distributions (streaming from a camera, etc.) the opposite.

## What the prior work already produced

Two pieces are already on disk and can be pulled into this:

1. **The atomics tool** (`mnist_cascade_atomics.c`) already computes tied-count, rescue/damage, and per-partition statistics per query. The self-supervised signal machinery is already there. We can extract the per-query signal vector with a small addition to that tool.

2. **The quadruple-hash resolver sweep** (`mnist_resolver_sweep.c`) has H1/H2/H3/H4/H_D50/H_D20 signatures per query. Ensemble disagreement is free from this data.

So the meta-router can be built as a thin adapter around existing tools plus a new bank data structure — not a ground-up rewrite.

## The real question for SYNTHESIZE

Not "how do we build an online meta-router" — the design is above. The real question is: **what is the minimum-viable experiment that produces a measurable meta-router accuracy number with honest cost accounting, on MNIST, in under ~500 lines of C?** Synthesize phase has to produce an executable plan at that scope.
