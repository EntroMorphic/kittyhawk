---
date: 2026-04-15
scope: LMM cycle — online, inline meta-router as a global observer over local routing
phase: RAW
---

# RAW: can a routing substrate watch itself and learn from doing so?

The proposal: a global "observer" that runs *inline* with query processing (part of the same forward pass, not a separate batch job) and *online* (state evolves from live experience, not a one-shot trained predictor). It watches the local cascade — H1 filter plus H2+H3+H4 local fusion — and either anticipates failures before they happen or corrects them after. "Learn from prior failures" implies some memory that accumulates over time. That memory has to live inside the routing surface or the whole architecture violates NORTH_STAR.

Let me grapple honestly before reaching for any plan.

**What does "anticipate failures" even mean without ground truth?** At inference time we don't know which queries the local router just got wrong. The observer can't read labels because labels aren't there. So the observer has to use self-supervised proxies: margin, ensemble agreement, distributional OOD-ness, quantization-regime flags. Each of these is a *predictor* of failure, not a detector. If the predictors are accurate enough to be useful, they might also be accurate enough to be deployed as the final classifier — collapsing the distinction. I'm unsure whether there's a genuine gap where the observer adds value the local can't extract itself.

**Online learning in a stateless substrate is a contradiction unless we introduce state.** K-NN over a fixed prototype set has no memory. To "learn" online we have to add a new store: either a growing failure-signature bank (a k-NN index of flagged queries), or a per-region confidence map (quantized regions of signature space with rolling success/fail counters), or something weirder like a streaming histogram of H1 top-1 distances. All of these have cold-start problems: the first N queries run with an empty memory and the observer is useless. Does the cold start burn the accuracy advantage before the bank fills up?

**The amplification trap haunts this.** The earlier `amplification_negative_result.md` showed that routing hard cases to a "stronger" classifier fails when the strong classifier isn't actually stronger on hard cases — they're hard for everyone. If our observer flags low-margin queries and escalates them to global fusion, and global fusion is also bad on low-margin queries, the escalation is a cost-multiplier with zero accuracy gain. The prerequisite question from the prior response is load-bearing: on queries where local fails, is global actually better? If yes, observer routing has headroom; if no, the whole architecture is a dressed-up version of amplification.

**Self-confirmation bias is a real failure mode.** If the observer flags queries X and updates its memory with "X was flagged," and the update rule is "flagged queries are hard," the observer will progressively flag more queries that look like X even if they're not actually hard. Online learning without ground truth is a closed feedback loop that can run away from the true distribution. Some kind of grounding signal is needed — either a trusted self-supervised proxy, or periodic comparison against a fixed reference, or a structural constraint that forces exploration.

**Ground truth substitute: ensemble disagreement.** If H2 and H3 disagree on top-1 within the filtered pool, that's a statement about query hardness that doesn't require labels. Ensemble disagreement is a classical variance estimate. It correlates with error rate under mild assumptions. Using disagreement as the failure signal is self-supervised and doesn't drift because it's computed fresh per query. This feels like the cleanest substitute for ground truth.

**But disagreement alone doesn't give the observer anything to learn.** If the update rule is "when H2 and H3 disagree, do X" and X is static, there's no learning. Learning would mean the observer's decision function itself evolves — maybe the threshold for "disagreement" moves, or the set of downstream strategies expands, or the region-confidence map sharpens. The learning has to bite something that matters.

**Meta-routing framing.** Let me try this: a meta-router is a router that routes routers. It reads the query AND the local router's state (its top-K, its margins, its H2/H3 disagreements) and outputs a decision: which local path to use, what K to use, which resolver family. In routing terms, the meta-router is a k-NN lookup into a "routing-context signature" bank — each entry is a tuple of (query signature, observed local router state, observed correct action). Online update means appending new entries as observed actions succeed or fail. Correctness feedback comes from self-supervised proxies: margin, consistency, agreement, eventual downstream signal if present.

This is clean because it stays entirely inside the routing primitives. No gradients. No dense ops. Just k-NN over routing-context signatures, with a bank that grows.

**Cold start solution: seed the bank from a one-shot offline pass.** Before going live, run local + global on the training set, record disagreements + self-supervised signals, and preload the bank. At inference, the bank is already dense and updates are fine-grained refinements. This sidesteps the cold-start problem at the cost of a one-time offline phase — arguably a violation of "pure online," but maybe an acceptable hybrid.

**Inline cost budget.** The observer must cost <10-20% of the primary H1 pass to be architecturally worth it. A k-NN over a bank of N_bank routing-context signatures costs O(N_bank × signature_bytes). If N_bank is bounded (say 1024-4096 entries) it's dominated by the primary H1 cost at N_PROJ=16 (60K × 16 trits ≈ 1M ops; bank k-NN at 4096 × 16 trits ≈ 64K ops — 6% of primary). Feasible.

**The experiment shape.** A build-and-measure in two phases: (1) prerequisite — is global-on-local-failures actually better? (2) if yes, prototype the meta-router as an online k-NN bank with self-supervised update and measure aggregate vs baseline.

Unresolved tensions I'm feeling:
- "Anticipate" vs "correct." Anticipating means acting *before* the local router runs; correcting means acting *after*. Only correcting is well-defined here — anticipating implies a query classifier that predates local execution, which requires features the query already has (signature, OOD proxies) without running local. Feasible but narrower.
- "Global observer" was the user's framing. What's actually global here? The bank is global in the sense that it accumulates across all queries. The routing decisions are per-query but they read a global state. Is this what the user meant?
- Is meta-routing a genuinely different architecture from "conditional escalation with learned threshold"? In practice they might be the same thing. I'm not sure the distinction survives scrutiny.

The RAW phase should end here. I've exposed enough tensions. Next: extract discrete nodes and dependencies.
