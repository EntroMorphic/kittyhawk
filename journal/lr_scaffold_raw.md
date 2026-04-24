---
date: 2026-04-21
scope: LMM cycle — would logistic regression benefit Glyph, and if so, where?
phase: RAW
---

# RAW: logistic regression as scaffolding (or not) for Glyph

The user asked whether LR would benefit Glyph. I gave a fast take ("probably not as first-class; possibly as a narrow scaffold experiment") and now they want an LMM pass on the idea. I should be honest about what I don't know, what I'm pattern-matching from base-2 habits, and where the real leverage might live.

## What I think I know

Classical LR: score each class c as `s_c = w_c · x + b_c`, where `w_c ∈ ℝ^D` is a learned weight vector, `x ∈ ℝ^D` is the input, `b_c ∈ ℝ` is a bias. Output is `softmax(s)` — the probability vector over classes. Training is gradient descent against cross-entropy loss. Inference is one dense matrix-vector multiply plus an argmax (for classification; the softmax is only needed if you want probabilities).

Glyph's three durable invariants make this uncomfortable:

1. **No binary float in runtime kernels.** Classical LR weights are real-valued; softmax is a nonlinearity over floats.
2. **No random or meaningless weights anywhere.** User has corrected this TWICE — pixels first, FFN bridge second. A learned weight isn't technically *random* but it also isn't *structurally assigned* — gradient descent says "this column's weight is 0.073 because the training set told it to be," which is semantically different from "this column is the horizontal gradient at pixel (3,7)."
3. **Routing over dense.** LR at inference is a dense matmul. That's the exact shape NORTH_STAR rages against.

So the naive read is: LR violates all three invariants and we should pass.

## But wait — pair-IG is already a learned weighting

This is the uncomfortable observation. `direct_lsh::build_pair_ig` computes `pw[d]` for every dimension and every class pair from information gain — statistics of the training set. These are *learned* integer weights derived from data. They are structurally indistinguishable from the weights an LR head would produce, except:

- Pair-IG weights are derived from frequency counts via entropy — interpretable, non-gradient.
- LR weights would be derived from gradient descent against a loss — less interpretable, more accurate in expectation.

On CIFAR-10, pair-IG gives +1.95pp over pure Hamming. The selective combination (Hamming when LSH+GSH agree, pair-IG when they disagree) is what produces 46.63%.

**If pair-IG is fine, why would LR not be fine?** The honest answer is probably "LR would be fine, if trained and quantized consistently with Glyph's invariants." The gut rejection of LR is partly base-2 allergy (LR is the archetype of what the thesis wants to avoid) and partly well-founded (training loops require float; inference requires dense matmul).

## Where LR could plausibly fit

Two slots in the current architecture:

**Slot A: replace pair-IG scoring.** Instead of entropy-derived `pw[d] ∈ [1,16]`, use LR-derived per-dim weights quantized to a similar range. Same re-rank mechanism: score each candidate by `dig = Σ_d pw_c1,c2[d] × (q[d] ≠ t[d])`. The only difference is where `pw` comes from. This is a drop-in experiment — one new `build_pair_lr` function, same data flow.

**Slot B: direct classifier on the signature.** Train a ternary LR head on the packed-trit signatures themselves. At inference, compute `score_c = sdot(sig, W_c)` per class — this is literally what `m4t_mtfp4_sdot_matmul_bt` was designed for. Pure SDOT, one kernel call for all 10 classes in MNIST. Could replace the Hamming k-NN entirely as the primary classifier (not the re-rank). The argmax over class scores is the prediction.

Slot B is much more interesting from a substrate standpoint — it's a direct use of the SDOT kernel that's been sitting there without a production consumer. But it's also a much bigger architectural swing (replaces the whole classifier, not just a scoring step).

## What scares me

- **I'm pattern-matching.** "LR = dense ML = bad" is a NORTH_STAR reflex. But NORTH_STAR also explicitly says "we may temporarily model some base-2-native ML systems in base-3 — transformers, routing variants of existing architectures, ternary adaptations of known methods. This is scaffolding. It lets us exercise the substrate and calibrate against known baselines." So a ternary LR head is *exactly* sanctioned scaffolding. My reflexive rejection is unsafe.

- **Training is float.** The Glyph repo is C-only, NEON-optimized, no Python. Training LR weights requires gradient descent — which requires a training loop, which requires iterative float arithmetic. The only way to do LR in pure Glyph is one of:
  (a) Train externally in Python/numpy, dump quantized int8/trit weights, load at startup.
  (b) Train in C with float gradients at startup, quantize, discard float.
  (c) Train via integer updates only (e.g., Rosenblatt-style perceptron updates, which are integer-safe but less accurate than gradient descent).
  Each has tradeoffs.

- **CIFAR-10 gap is probably representational, not scoring.** The `cifar10_nproj_ceiling.md` journal concluded the distance-metric gap is in the *input representation*, not the ranking. If that's right, LR as a re-ranker (Slot A) is wasted effort. LR as a direct classifier (Slot B) might do better because it bypasses Hamming distance entirely, but it's still bottlenecked by the signature's information content.

- **The "structured signature" tools already exist.** `structured_lsh`, `structured_gsh` etc. are already current. Their design is "each trit has a specific meaning" — which is exactly the antithesis of learned weights. Dropping an LR head on top of these feels like introducing meaninglessness right after establishing meaning.

## What I'd want to measure before committing

1. Does *any* per-dim weighting (even uniform `pw = 1`) beat pair-IG on CIFAR-10? If no, pair-IG is at ceiling and LR won't help.
2. What is the upper-bound accuracy on the CIFAR-10 direct signature achievable by *any* scorer? (Oracle over union already measures this.)
3. What would an LR head quantized to {-1, 0, +1} × signed-8bit scales look like — is it runnable on SDOT in the existing kernel?
4. Does the training artifact (float weights) need to stay in the repo, or can it be produced externally and only int8/trit quantized weights checked in?

## Open questions

1. Is the "no learned weights" reading of user feedback too strict? Pair-IG is technically learned from data — if that's OK, LR-style learned weights should also be OK, provided they land as structurally-meaningful integers at runtime.
2. Which slot (re-rank vs direct classifier) is the right experiment to run first? The cheaper is Slot A (replaces only pair-IG); the more thesis-interesting is Slot B (exercises SDOT).
3. Where does the training live? If training is excluded from M4T by design (substrate is inference-only), and libglyph should be runtime-only, training is a *consumer-builder* tool — maybe `tools/lr_train.c` at startup (one-shot integer gradient descent on signatures, produce quantized weights), or maybe offloaded to a one-time Python script.
4. Is there a ternary formulation of LR that's natively base-3 (not a quantized adaptation of a base-2 design)? E.g., scores in {-1, 0, +1} per class, decision via signed popcount of dim votes? That would be base-3-native instead of base-2-ported.

## First instincts, to be checked in later phases

- The right experiment is Slot A (replace pair-IG with LR weights) because it's cheap, measurable, and the comparison is apples-to-apples. If it wins, Slot B becomes a natural next step.
- The training should be one-shot integer arithmetic (perceptron-style or quantized SGD at startup) so the repo stays C-only per the project scope.
- The user's "NO RANDOM WEIGHTS" rule should be interpreted as "no *meaningless* weights" — data-derived integer weights (pair-IG, LR-quantized) satisfy the spirit if each weight's derivation is legible. Pair-IG is legible because entropy is interpretable; LR weights would need a story.
- The routing-vs-dense tension for Slot B (SDOT is literally the SDOT-native kernel) is actually *aligned* with the substrate, not against it — the kernel exists, has no current consumer, and was built for exactly this.
- But question 4 above (is there a base-3-native LR shape?) is the more interesting NORTH_STAR move. "Adapt LR to ternary" is scaffolding; "find what's LR-shaped natively in base-3" is the thesis.
