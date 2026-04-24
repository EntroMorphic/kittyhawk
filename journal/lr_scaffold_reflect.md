---
date: 2026-04-21
scope: LMM cycle — would logistic regression benefit Glyph, and if so, where?
phase: REFLECT
---

# REFLECT

## The core insight

**The question is not "LR or not" — it's "what is Glyph's answer to the class of problems LR solves?"**

Stripped of the base-2 scaffolding, LR is a specific answer to a generic question: *given signatures that correlate with class membership, produce a per-dim-per-class weighting that maps signature → class score.* That question is already live in Glyph. Pair-IG is one answer. LR is another. The thesis-relevant question is which answer the base-3 substrate *prefers* — not which one is culturally familiar.

Framed this way, the LMM cycle pops a layer up from the naive "should we adopt LR?" to the substrate-level "what is the per-dim-per-class weighting primitive on a routing-first base-3 machine?" The latter is the NORTH_STAR-aligned question. The former is the scaffolding question whose answer is probably "eh, maybe, as one experiment."

## Why this matters more than "yes run Slot A"

The user's working-style memory says *derive from hardware, revise spec not thesis*. "Run LR as a quantized re-ranker because pair-IG is already learned weights" is a perfectly reasonable scaffolding experiment. But it doesn't *derive*. It imports.

If instead we ask: *what does the hardware already know how to do that serves the weighting purpose?*
- SDOT is the per-output-cell weighted sum of int8 inputs → int32, where weights are ternary int8. Shape: `score_c = Σ_d W_c[d] × sig[d]` with `W_c[d] ∈ {-1, 0, +1}`. That is already the scoring architecture. No gradient descent required yet; we just need to *choose* the weights.
- TBL is the per-dim three-way dispatch. Shape: `sig[d] → action_d[class]` where `action` is a LUT per class — dim votes for class based on its own trit value. That is also a weighted sum shape, but with two-bit weights implied by the trit.
- Masked-VCNT is popcount over agreement masks. Shape: `score_c = popcount(sig XOR W_c AND mask)` — ternary Hamming distance to a class prototype. A single per-class prototype IS a weight vector, trivially.

These are three structurally equivalent ways to write `score_c = f(sig, W_c)` in base-3. The current Hamming k-NN is "masked-VCNT against all training prototypes and softmax by k-NN vote." A per-class prototype classifier is "masked-VCNT against 10 class prototypes and argmin distance." That's one prototype per class — `W_c ∈ {-1, 0, +1}^D` — and the substrate has had the kernel for this all along.

**This is the base-3 shape of LR: one ternary prototype per class, scored via SDOT or VCNT, argmax/argmin over class scores. No gradient descent in the inner loop. No binary float. No dense matmul over the training set at inference — only over K=10 class weights.**

## Resolved tensions

**T1 (pair-IG learned or not?):** pair-IG is learned *in the statistical-derivation sense*. Each weight has an entropy computation behind it and survives the "I can tell you why this weight is what it is" test. The NO RANDOM WEIGHTS rule is about **weights without derivation story**, not weights without structural prior. Pair-IG passes. LR-derived-and-quantized weights pass IF the derivation is named and the final artifact is the quantized integer table, not the float gradients. The rule's spirit is: every dim/weight must be nameable.

**T2 (scoring headroom on CIFAR-10?):** resolvable by one quick measurement — oracle over candidate union on CIFAR-10. The current `direct_lsh` sweep already outputs `union_sum[si]` and would only need the "does the correct class exist in the union" counter to be the headroom ceiling. If oracle > 50%, scoring has headroom. If oracle ≈ 46.63%, scoring is at ceiling and Slot A cannot help. **This measurement is pre-commitment work, not experimental work — it decides whether to commit.**

**T3 (scaffold LR vs base-3-native formulation?):** The reflection collapses this. The base-3-native formulation — **per-class ternary prototypes scored via SDOT / VCNT** — already exists in the substrate's kernel surface. "Scaffold LR" is then a misnamed move: the actual experiment is "compute 10 class prototypes, score against them via the SDOT kernel, argmax." That is *not* LR adapted to ternary; it is the substrate's own shape for the per-class-weighting problem. The LR framing would be the training rule that produces those prototypes.

**T4 (training location):** If the final artifact is a per-class ternary prototype vector of D trits (D = total_dim for direct_lsh, so ~9024 trits × 10 classes = 90K trits = 22.5 KB of weights), the *training* of these prototypes can be one-shot integer: (a) class centroid in signature space with sign-thresholding — one pass over training data, integer arithmetic throughout, no gradient descent; (b) perceptron-update with integer accumulation and sign-at-the-end; (c) offline float training → quantize once → commit the trit table to the repo as a .c file (exact `m4t_lut_gen.c` precedent). Option (a) is *base-3-native*. Option (b) is perceptron-style, integer. Option (c) is what base-2 does and quantizes. The cleanest first experiment is (a): class centroid on the direct signature, sign-thresholded to a trit prototype per class.

**T5 (rage against the trodden):** partially resolved. The **LR gradient-descent training loop** is the trodden path — a base-2 shape with float weights and log-likelihood loss. The **ternary class prototype classifier** is a different primitive that happens to solve the same class of problem. Pursuing the prototype classifier is substrate-derived, not LR-adapted. The reframe converts the question from "yes scaffold LR" to "let's exercise the per-class prototype shape the substrate's been waiting to use."

**T6 (structured signature + learned weights coherent?):** yes, but the structure relation is multiplicative, not subsumptive. Each trit's input meaning is preserved (horizontal gradient at (3,7) is still horizontal gradient at (3,7)); the class prototype's trit at that position says "class c expects this specific input trit value." The prototype doesn't redefine what the input means; it records which values the class is characterized by. This is structurally identical to pair-IG's per-dim weight, just with a per-class instead of per-class-pair shape.

## Hidden assumptions

- **A1: We need a drop-in re-ranker.** Not necessarily. If the prototype classifier can *replace* the Hamming k-NN as the primary classifier (not the re-rank), it's a cleaner architectural story. One SDOT kernel call scores all 10 classes in one pass; argmax gives the prediction directly. No union, no probe, no resolver. This changes the question from "fix the re-rank" to "do we need the probe+union+resolver dance at all?"

- **A2: Pair-IG is the scoring ceiling.** Only true if class-pair entropy captures all discriminative structure. Not obviously true — pair-IG is pairwise (C(10,2) = 45 weight tables) while a per-class prototype is unary (10 prototypes). The geometries are different; which wins is empirical.

- **A3: The interesting question is CIFAR-10 numbers.** Actually probably false. The more interesting question is whether a production consumer can live on SDOT without a k-NN dance in the middle. That would shrink the cascade, exercise the fastest primitive on the substrate, and produce a classifier that's O(class_count × sig_bytes) per query instead of O(n_train × sig_bytes) worst case.

## What I now understand

1. **The "LR or not" frame was too narrow.** The right frame is "what is the base-3-native per-class scoring primitive, and what kind of training produces its weights?"

2. **The substrate's answer is already present:** per-class ternary prototypes, scored via SDOT (or VCNT), argmax. The kernel exists. No production consumer uses it. This is a gap in the consumer surface, not a missing primitive.

3. **LR in classical form is the wrong import.** Binary float weights, gradient descent, softmax — all carry base-2 ergonomics that don't align with the substrate. The reflex rejection was correct for the *surface* reason (wrong ergonomics), but the *deeper* reason is that a better substrate-aligned shape already exists.

4. **Training is a consumer concern.** The substrate is inference-only; the pair-IG LUT is precedent for startup-time float confined to one-shot LUT builds. A prototype-training consumer tool would fit the same pattern — and integer training rules (class centroid + sign, or perceptron) let us avoid float entirely for the first pass.

5. **Two distinct experiments drop out cleanly.**
   - **E1 (cheap, falsifiable):** class-centroid + sign-threshold prototype per class on the direct signature. Compare to Hamming k-NN and to pair-IG-Selective on all three datasets. No training loop, no float, one pass over training data. Measures whether the substrate's own SDOT-shape classifier beats the scanning k-NN it currently runs.
   - **E2 (iterative, if E1 shows signal):** perceptron-update training on class prototypes, integer arithmetic, bounded epochs. Only runs if E1 produces usable prototypes; converges them further without introducing float.

6. **The CIFAR-10 ceiling matters.** If oracle-over-union on CIFAR-10 is at or near 46.63%, the scoring stage is saturated and E1 cannot help regardless of how elegant. This measurement is pre-work.

7. **The thesis-aligned claim.** If E1 shows SDOT-prototype classification matches or beats Hamming k-NN on any of the three datasets at equal signature dimensions, that's a direct instance of the thesis: *routing-native primitives beat the dense-shape scaffolding when we stop using the scaffolding's shape*. The comparison is as clean as the Axis 6 bucket-vs-scan comparison was for speed.

## Open residuals

- **R1:** need a name. "LR on Glyph" misnames it. Candidates: *ternary prototype classifier*, *SDOT classifier*, *class-signature argmax*. The first is descriptive; the last is the sharpest. I'll use **Class-Signature Argmax (CSA)** in the synthesize phase because it names the operation without importing LR vocabulary.

- **R2:** what if the class centroid is too coarse and perceptron is needed? That's the E1 → E2 handoff. E1 answers "does the primitive shape work at all?"; E2 answers "how much training buys how much accuracy, while staying integer?"

- **R3:** is the right scope MNIST first (cleanest baseline), or CIFAR-10 first (where pair-IG already helps +1.95pp)? MNIST first, because bit-identical fallback checks are easiest there and the signature dim is smallest — fastest iteration.
