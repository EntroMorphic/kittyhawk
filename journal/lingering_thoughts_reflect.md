# Reflections: Lingering Thoughts

Working from `lingering_thoughts_nodes.md`.

---

## The "why" ladder

1. **Why am I uneasy?** Because the substrate is done but nothing uses it.
2. **Why does that matter?** Because correctness of pieces doesn't prove correctness of composition.
3. **Why haven't we composed yet?** Because we followed the pipeline religiously and the pipeline was about primitives, not systems.

The pipeline was the right thing to build. It was also the *safe* thing to build. Every item was self-contained: write a kernel, test it, red-team it, commit. The risk was bounded. The validation was local.

The next step — composing primitives into a forward pass — is the *unsafe* thing. The validation is global. A bug in composition could implicate any primitive. The feedback loop is longer. The golden values can't be hand-derived (they depend on trained weights, which are opaque).

That's why I'm uneasy. I've been doing the safe work. The unsafe work is next.

---

## Core insight

> **The substrate is a hypothesis. The forward pass is the experiment. Until the experiment runs, the hypothesis is untested.**

Everything we've built — the MTFP types, the NEON kernels, the routing primitives, the opcode tables — is a hypothesis that says: "ternary fixed-point arithmetic on M4 is sufficient to reproduce the behavior of a trained model." The only experiment that tests this hypothesis is: load real weights, run a real input, compare the output to a known-good reference.

MNIST with trix-z's trained weights is the cheapest such experiment.

---

## Resolved tensions

### Tension A — substrate vs. system

**Resolution: the substrate is done. Stop perfecting it. Start using it.**

The diminishing returns are clear: eight red-team passes found seven bugs in the first three passes, zero in the last five. The substrate is converged. Any remaining bugs are integration bugs that only reveal themselves under composition. The only way to find them is to compose.

### Tension B — purity vs. pragmatism

**Resolution: the line was already crossed correctly.** The policy says: float is banned in `libm4t.a`. Tools that run on a dev machine can use float. The LUT generator uses float; its output (MTFP cells in a `.c` file) does not. The IO tool uses float; its output (MTFP binary blobs) does not. Committing `m4t_mtfp_tables.c` to `src/` puts float-derived values in the library's `.rodata`, but the runtime code never touches float. The boundary is between *arithmetic* and *data*. Float arithmetic is banned. Float-derived data — pre-computed offline, version-controlled, auditable — is permitted. This is the same distinction hardware makes: a ROM lookup table was computed by an engineer with a calculator, but the circuit that reads it is pure digital logic.

### Tension C — contract vs. shipping

**Resolution: document the failure, ship anyway, fix it.** The LayerNorm p99 jitter at small N is a known, measured, documented contract violation. It's in `M4T_BEYOND.md`. Shipping with it is honest because it's disclosed. The fix (cap Newton-Raphson iterations at a fixed count, or pre-compute isqrt for common denominators via a small LUT) is tractable and can land alongside the MNIST work.

### Tension D — inference vs. training

**Resolution: inference first. Training is research, not engineering.**

The inference path validates the substrate. If MTFP19 can reproduce trix-z's MNIST accuracy, the numerical system works. Training in MTFP is genuinely novel — it requires MTFP gradients, MTFP optimizer state, and a ternary-native update rule. That's a paper, not a sprint. Attempting it before inference works would be premature. But it should be *designed* in parallel, even if the implementation is deferred.

---

## Hidden assumptions surfaced

### Assumption 1: trix-z's trained weights are available and convertible

The MNIST model (d=128, T=4, 398K params) was trained in float32 with STE through ternary quantization. The float shadow weights exist in trix-z's checkpoint format. Converting them to MTFP cells requires: (a) reading the float values, (b) multiplying by MTFP_SCALE, (c) rounding, (d) clamping. The ternary weights ({-1, 0, +1}) don't need conversion — they're already trits. What needs conversion: biases, LayerNorm weight/bias, the projection weight, the classification head. These are float in trix-z.

If the float values are outside MTFP19 range (±9842 real), they'll clamp. For a well-trained small model, this is unlikely — biases and LN parameters are typically O(1) in magnitude.

### Assumption 2: the MNIST forward pass is FFN-only

Looking at trix-z's `test_ternary_mnist.c`: it's a projection layer (input → hidden), a series of routed FFN blocks (ternary tiles with k-of-T routing), and a classification head (hidden → 10 logits). No attention. No embedding. No positional encoding. This is the simplest possible path through M4T.

### Assumption 3: MTFP precision is sufficient for classification

MNIST is a classification task where the model only needs to produce the right argmax over 10 logits. Small numerical differences (from MTFP rounding vs. float) are unlikely to flip the argmax unless the model is operating right at a decision boundary. We should expect accuracy within 0.1–0.5% of the float reference.

### Assumption 4: the SDOT path isn't needed for MNIST

trix-z's MNIST path uses the ternary routing kernel (popcount distance + k-of-T) with float32/int32 activations, not int8. The MTFP4 SDOT path is a performance optimization for the routing distance computation; the MTFP19 path can do everything MNIST needs. SDOT becomes important at GPT-2 scale where routing throughput matters.

---

## What I now understand

The lingering unease is healthy. It's the correct emotion for the transition from "building tools" to "using tools." The tools are good. The next risk is composition, not construction.

The single most useful next action is: **commit the LUT tables, write the IO tool, port the MNIST forward pass, and compare against trix-z's 98.22%.** Everything else — SDOT bridge, LayerNorm fix, training design, external users — is downstream of that result.

---

## What would make me feel at peace

One number: **MNIST accuracy ≥ 97.5% using M4T primitives with trix-z's trained weights.** That's the substrate hypothesis, tested. If it passes, M4T is validated. If it fails, the failure mode tells us exactly what to fix.

The number doesn't need to match 98.22% exactly (MTFP rounding will cause small differences). But it needs to be in the same ballpark. If accuracy drops below 95%, something fundamental is wrong. If it's above 97.5%, the substrate works and the rounding is tolerable.

That's the experiment. Everything else is preparation for it.
