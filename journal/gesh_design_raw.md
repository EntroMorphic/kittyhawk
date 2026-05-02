---
cycle: gesh_design
phase: RAW
date: 2026-05-01
scope: LMM applied to GESH_DESIGN.md (the Gesh routing proposal)
status: priors from the prior-cycle archive deliberately set aside per owner direction
---

# Raw — gesh_design

## Stream of consciousness

The "three Gs" framing is rhetorically too clean. Global, Geometric, Gradient — they all happen to start with G. Are these three independent dimensions, or three names for one underlying principle (the substrate-native form of attention) that happens to factor along three syllables? The design treats them as orthogonal failure-mode fixes; that's a hypothesis the cycle should test, not a derivation.

The architectural-unification claim is what's actually load-bearing here. "Attention-plus-FFN collapsed into routed retrieval from a frozen ternary store" — that's a structural claim about what attention IS. If Gesh works, it suggests attention's success is mostly the routing+retrieval shape, not the specific dot-product-softmax-MLP machinery. That claim doesn't need three Gs to be true. It needs ONE working example.

I keep coming back to the benchmark question. Without picking the task, every other question in §6 is unanswerable. Bank size, projection capacity, region count, refresh frequency — all of these are conditional on what Gesh is being asked to do. The design proposes MNIST as the toy. MNIST has almost no global structure for stage 1 to exploit; the global stage might do nothing on MNIST and still Gesh "works." That's not a test of the unification claim, it's a test of stage 2 alone. Wrong shape.

What's the smallest task that genuinely exercises hierarchical retrieval? Probably sequence modeling — character-level on something tiny — where local patterns (stage 2) and global context (stage 1) are both needed. Or an algorithmic task like induction-head where attention is canonically known to work via routing. Algorithmic tasks have ground-truth retrieval — you can verify routing decisions, not just final accuracy.

The "frozen bank" choice is the design's central architectural commitment, and the doc treats it as an implementation detail. Most attention literature trains keys+values jointly with everything else; Gesh freezes the bank. That's either a real insight (the bank can be installed once, like a knowledge base) or a self-imposed constraint that Gesh has to spend complexity to overcome. The doc doesn't surface it as the choice it is.

I worry about training dynamics: STE through ternary signature, Gumbel-softmax over Hamming distances, periodic refresh of region signatures. Three mechanisms each with stability stories; their interaction is unstudied. Each one in isolation has known failure modes; together they could compound or they could stabilize each other. No way to know without measuring, but I'd want to test one mechanism at a time before stacking.

The PCA initialization makes me uncertain. PCA on what data — training samples, or the bank's tile signatures? The design picks training-data PCA, but the bank's geometry is what queries route against. There's a quiet assumption that the data manifold and the bank manifold are the same thing. They might be, if the bank was built from training data. They might not be, if the bank is a designed prototype set.

The kernel surface is smaller than the design claims. `signature_match` is `popcount_dist` + `topk_abs` — both already shipping. That's a libglyph-level utility, not a substrate primitive. `threshold_extract2` (asymmetric two-threshold) is genuinely new but might not even be needed if a single symmetric threshold (already shipping) is sufficient — that's an empirical question. The design proposes new substrate territory before measuring whether the existing surface composes.

What scares me: I can imagine Gesh "working" on MNIST in a way that doesn't actually test the architectural-unification claim. We'd ship a routing layer that hits a number on a benchmark, and the substrate-claim narrative would advance, but we wouldn't know whether the unification is real or whether stage 2 alone happened to be enough.

What's exciting: with the verified substrate, this is the first time the architectural claim is testable on hardware that isn't lying about its own arithmetic. The cross-exp accumulator exists; the SDOT path is exact and property-tested; the flag tracking exposes saturation/rounding events. Whatever Gesh measures will be a real measurement, not noise from a broken kernel.

The design's §7 self-awareness ("if Geometric drops 2 points and Gradient drops 30, defer Geometric") is the right framing. The build plan should respect it: build the simplest Gesh that could work, measure where it fails, add the G that addresses the specific failure. The current plan builds all three simultaneously, which is six provisional design choices stacked.

## Questions arising

1. What's the benchmark? Without it, the rest is speculation.
2. Are the three Gs orthogonal or one principle? Ablation tests; the design assumes.
3. Is the frozen bank load-bearing or self-imposed? Worth surfacing.
4. PCA on training data or on the bank? Different geometries.
5. Does Gesh need substrate primitives, or can it live entirely in libglyph?
6. What's the smallest Gesh that could work on the chosen task?
7. Do the three training mechanisms compound their instabilities, or stabilize each other?

## First instincts (now suspicious)

- "Build all three Gs from the start" — too many simultaneous commitments.
- "MTFP4 projections by default" — discipline-flip; ternary first.
- "MNIST validation" — wrong shape for testing the unification claim.
- "`signature_match` as new substrate primitive" — should be libglyph until proven hot.
- "PCA initialization is a strong prior" — strong but the wrong-prior risk is unsurfaced.
- "Periodic refresh every 1000 steps" — guess, not derived.

## What scares me

I designed-by-committee against attention's surface area, not against a specific task's demand. The Gesh design reads like "what does attention have that LSH doesn't, and how do we add those things to LSH?" — which inherits attention's design pressures whether or not those pressures apply to a substrate where you're not training the bank jointly.
