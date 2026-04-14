---
date: 2026-04-14
phase: REFLECT
topic: How ternary routing helps M4T kernels and the training of models
---

# Reflect

## Core insight

**The question "how does ternary routing help kernels and training" is framed from a base-2 architecture worldview that the substrate rejects.**

It assumes kernels and routing are separable artifacts that get combined, and that training is a known procedure that routing decorates. Both assumptions are base-2 artifacts — in a genuinely base-3 substrate, the routing *is* the kernel (N10), and training is an open architectural question that can't be "helped by" routing because it has to be reinvented in routing's shape.

The useful reframe: instead of "how does routing help kernels and training," ask "**what are the kernels and training loops that are natural to ternary routing, and what can they do?**"

Under that framing the answers open up:

---

## Resolved tensions

### T1 (overhead vs savings) → RESOLVED via compositional thinking
Routing overhead is only an issue if routing wraps a computation that could be done more cheaply in dense form. In composed multi-layer ternary architectures (N9), every layer routes; the routing decision is amortized across the entire downstream computation, not the one matmul it dispatches. The overhead-beats-savings regime shrinks as depth grows.

### T2 (accumulation vs refinement) → PARTIALLY RESOLVED
Signature accumulation (one-shot, ternary-native) and gradient refinement (iterative, base-2-native) look opposed. The ternary-native path for refinement doesn't exist yet — but it's not impossible. Candidates surfaced:
- **Sign-flip refinement.** For misclassified examples, flip individual trits in the relevant signature. Discrete, local, base-3.
- **Exponent-shift refinement.** Adjust the block exponent of an entire signature to re-weight its influence without touching the trit pattern. Uses the per-block-exponent metadata we already planned.
- **Anti-signatures.** Accumulate positive and negative prototypes per class; classify by ratio. The `apply_signed` primitive supports this natively.
None of these are gradient descent. All are natively discrete. This is where the research lives.

### T3 (ternary purity vs binary surface) → RESOLVED by fixing the substrate
The current sign_extract is binary. It has to be replaced — not decorated with a ternary sibling. "Keep both and let consumers choose" re-invites the base-2 default. The replacement is `m4t_route_trit_extract` (k-th trit of a mantissa) or equivalent three-state-producing primitive. Until this lands, "ternary routing" claims inherit a binary substrate.

### T4 (kernel vs architecture) → RESOLVED by dropping the separation
Routing is not a modifier; routing is a *shape*. Base-3 kernels are routing-shaped by default. The question "does routing help" is like asking "does indentation help Python" — it's not an optimization, it's the syntax.

### T5 (thesis vs empirical state) → STATED, NOT RESOLVED
The thesis is still aspirational. Empirical validation requires: (a) a ternary-native routing primitive, (b) a benchmark bed, (c) a training loop that doesn't cheat with float. We have (0) none of these. The reflection can't resolve this; work can.

---

## Hidden assumptions challenged

1. **"Routing is added to a kernel."** False. In base-3, routing is the kernel's shape.
2. **"Training is gradient descent."** Contingent, not universal. Base-3 invites other paradigms; we haven't built them.
3. **"Kernel speed is the metric."** Hardware utilization on native ops (SDOT, TBL, VCNT) may be more informative than FLOPS. Unmeasured.
4. **"MoE is what we mean by routing."** No. MoE is one special architecture where routing is a binary on/off gate bolted onto dense. Ternary routing is more primitive.
5. **"Transformer-level capability requires transformer-like training."** Unproven. The transformer's training paradigm was shaped by what was easy on base-2 silicon. Base-3 substrates might admit different architectures that don't need gradient refinement to reach high capability.

---

## What I now understand

The kernel-help question collapses once we stop imagining ternary routing as a base-2-compute optimizer. In a base-3 substrate the kernels ARE routing-shaped: `distance_batch` is popcount over packed trits, `apply_signed` is signed accumulation over selected tiles, `ternary_matmul` is conditional negate-and-add. These don't "benefit from routing"; they ARE routing, in different compositions.

The training question is harder. There is no established routing-native training paradigm at scale. Signature accumulation is the natural starting point but has a ceiling; refinement primitives exist conceptually (sign-flip, exponent-shift, anti-signatures) but aren't implemented and aren't proven to reach high capability. This is where the thesis has real empirical risk, and it's the reason NORTH_STAR is honest that "the end-game is unknowable."

The useful framing isn't "how does routing help training" but **"what is the ternary-native training paradigm, and can it reach the capability ceiling of gradient-trained dense networks?"** That's an open empirical question for the entire project, not something I can reflect into an answer.

---

## What remains uncertain

- Whether a ternary-native training paradigm can match gradient-descent capability at scale. If not, the thesis is capped at small-to-medium models where prototype methods work.
- Whether multi-layer routing compose to transformer-equivalent expressiveness. The combinatorial math says yes (N9); whether the optimization landscape is navigable without gradients is open.
- Whether fixing `sign_extract` substantially changes MNIST routed accuracy. Possible: the zero-state might matter even on MNIST. Worth testing.
- Whether the hardware-utilization claim holds under benchmark. Still completely unmeasured.

---

## What the method surfaced

- The "kernel vs routing" dichotomy is a base-2 artifact. Routing IS the kernel shape in base-3.
- The training question is deeper than I'd been treating it. It's not "gradients in ternary" — it's "what does learning look like when the substrate is discrete?"
- My own framings (MoE, prototype learning) are narrowings. Watch the narrowings.
- Three ternary-native refinement candidates (sign-flip, exponent-shift, anti-signatures) emerged. None were in my mental model before the cycle.
