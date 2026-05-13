# REFLECT — Glyph gaps

## Core insight

**The eviction arc is a corollary I've been treating as the main
event.** The three foundational claims sit in a structure: claim 1
(primitives floor) supplies the operations, claim 2 (math-as-
signatures) supplies the bridge, claim 3 (base-3 IS the graph)
supplies the substrate. Eviction is an *application* of claim 3's
substrate metric. Even a clean qsigdist win doesn't substantiate
the foundation — it substantiates a downstream use.

I've been measuring increasingly-narrow corollaries with
increasingly-elaborate infrastructure while the FOUNDATIONAL test
of claim 3 (path-graph structure earning its keep beyond eviction)
hasn't been designed. Claims 1 and 2 have no measurement loop at
all. This is the structural gap that explains why so much work has
produced so little forward motion on the vision: I've been
sharpening one tool while leaving two of three foundations
unbuilt.

## Resolved tensions

**T1 (eviction vs foundational pivot)** — RESOLVED toward pivot.
The eviction trend is "+6pp, CI straddles zero." Settling it from
inconclusive-positive to significant-positive moves a corollary,
not a foundation. The unit of work that moves the project is
"start a measurement loop for claim 1 or claim 2," not "add 30
prompts to a corollary's battery." Background the long battery if
hardware time is cheap; foreground cycles go elsewhere.

**T2 (c_dump_v3 trust)** — RESOLVED toward verify. The cost is ~10
minutes: find the script or commit that generated c_dump_v3, decode
the prompts, confirm natural language. This is a cheap audit with a
high payoff: if c_dump_v3 is gibberish, ALL of Phase α-ε's oracle
numbers need re-running on natural language. Cannot in good
conscience write a new measurement loop while this unverified gap
sits.

**T3 (qsigdist tests vs cycle speed)** — RESOLVED toward defer. The
production default is no eviction. Sigdist and qsigdist are
research modes. If qsigdist eventually becomes production-default,
THEN unit tests become load-bearing. Until then the harness IS the
integration test. Adding unit tests right now is premature.

**T4 (vision vs corollaries)** — RESOLVED toward the vision. The
project's center of mass moves when foundational claims earn
evidence. Corollaries that are well-measured but downstream don't
move it. The next measurement loop should attack claim 2 or claim
1, NOT a new corollary of claim 3.

**T5 (documentation accuracy)** — RESOLVED toward periodic audit.
Build an audit step into the pivot. Don't let it block the pivot,
but do it before the next round of new work.

## Hidden assumptions challenged

1. **"Eviction is the showcase for claim 3."** False. Eviction is
   one application. The path-graph metric could be measured in
   many other contexts (similarity search, clustering, routing
   decisions, the routed-attention path itself). Eviction was
   accessible because production code exists. That's a HEURISTIC
   advantage, not a vision-level priority.

2. **"More measurement = more validation."** False. Eleven journals
   on eviction have produced one retracted negative, one retracted
   mechanism, and an inconclusive trend. The marginal information
   per journal is approaching zero. Adding more isn't "more
   rigorous"; it's diminishing returns.

3. **"The vision's three claims are separable test targets."**
   Half-false. Claim 2 (bridge) is the connective tissue between
   1 and 3. Testing 3 in isolation (as I've been doing) measures
   only the substrate's *intrinsic* properties — not whether the
   substrate is good at REPRESENTING THE MATH the project needs.
   Without claim 2's bridge, you can't ask "does the path-graph
   metric serve the routing that the math demands?" — because you
   can't even derive the routing from math.

4. **"Match-rate vs no_evict is the right harness metric."**
   Suspect. If no_evict's generation is itself low-coherence
   (likely on some prompts), match-rate measures agreement with a
   potentially-bad baseline. Better metrics: NLL/perplexity against
   a tokenized reference; KL divergence between policy and
   no_evict logit distributions; human evaluation of coherence.
   I've been treating match-rate as authoritative without
   stress-testing it.

5. **"Journals are persistent context."** Partially false. They
   are persistent, but their TRUTH STATUS changes as later
   journals retract earlier claims. Without a top-level index that
   reflects current state, the journal directory is an unsorted
   pile of correct-and-incorrect.

## What I now understand

The gaps for Glyph are not "things left to do on the eviction arc."
They are **the unbuilt measurement loops for two of three
foundational claims**, plus a thin layer of hygiene on the
substrate-claim arc itself.

**Priority ordering by foundational weight:**

1. (Highest) Claim 2's first measurement loop — math-expression → signature bridge.
2. Claim 1's first audit — does the current primitive set close over
   the math the project actually uses?
3. Claim 3's harness territory — settle the qsigdist trend.
4. Hygiene — c_dump_v3 provenance, journal index, README sync,
   memory cleanup.

The temptation is to do #3 first because it's familiar and has
existing infrastructure. The right move is to do hygiene (#4) in
parallel cheap cycles, design #1 deliberately, and either run #3
as a background battery or accept "inconclusive with positive
trend" as a stopping point.

## Remaining questions

- What does a "math-expression-to-signature bridge" look like in
  code? The vision says it exists in principle; is there a minimal
  toy implementation that would test the claim?
- For claim 1's primitive-set closure: what set of math operations
  should the project be able to express? Softmax? RMSNorm? Both
  already exist in the harness as fused MTFP ops. Are those
  expressible as compositions of the 6 frozen primitives, or do
  they require additional primitives (exp/log)? Is that gap
  actually measurable?
- Is there a way to test claim 3 outside of eviction that doesn't
  require building a whole new pipeline? Maybe the routed-attention
  path itself (which uses substrate sigs to pick top-k) is a
  cleaner showcase.
