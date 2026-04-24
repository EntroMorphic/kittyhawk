---
date: 2026-04-21
scope: LMM cycle — would logistic regression benefit Glyph, and if so, where?
phase: SYNTHESIZE
---

# SYNTHESIZE: Class-Signature Argmax (CSA) — the LR question reframed

## The reframe

The original question — "should Glyph adopt logistic regression?" — is the wrong question. LR is a specific base-2-native answer to a generic question that is already live in the substrate: *given signatures that correlate with class membership, produce a per-dim-per-class weighting that maps signature → class score.*

Glyph already answers this question in two ways (Hamming k-NN over signatures; pair-IG re-rank on the union). The reflection found a third answer, **substrate-native, latent in the kernel surface since day one**: one ternary prototype per class, scored via SDOT or masked-VCNT, argmax over class scores. The SDOT-native matmul kernel `m4t_mtfp4_sdot_matmul_bt` has no current consumer and sustains 55–60 Gops/s on M3 — the fastest primitive in the substrate. Class-Signature Argmax (CSA) is the first consumer that exercises it for a classifier, not for re-ranking.

This is not "LR adapted to ternary." This is the base-3 hardware's own shape for the per-class-weighting problem. LR is a naming trap; CSA is the operation.

## Decision

**Pursue CSA as a scaffolding experiment sanctioned by NORTH_STAR §4.** Two staged experiments, E1 cheap and falsifiable, E2 conditional on E1.

Do NOT pursue classical LR (float weights, gradient descent, softmax). That path imports base-2 ergonomics for no substrate-aligned gain.

## Success criteria

**Pre-commit gate (measurement, not experiment):**
- [ ] Compute oracle-over-union accuracy on CIFAR-10 from the current `direct_lsh` sweep. If oracle ≤ 47.5% (≈ 46.63% + 1pp), the scoring stage on CIFAR-10 is at ceiling and CSA cannot help there. Run E1 on MNIST only.
- Cost: instrument one counter in `direct_lsh` main classification loop; runtime ≈ single CIFAR-10 sweep (~75 min).

**E1: class-centroid CSA.**
- [ ] Implement: one pass over training signatures, accumulate per-class sum, sign-threshold with density-calibrated τ into `W_c ∈ {-1, 0, +1}^D`. Pure integer arithmetic. One-shot consumer operation, same shape as pair-IG LUT build.
- [ ] Score: `score_c = m4t_mtfp4_sdot_matmul_bt(W_c, sig)` per query per class. 10 SDOT calls per query, argmax.
- [ ] Report: accuracy, per-class breakdown, confusion matrix. Compare to Hamming k-NN and pair-IG-Selective on MNIST, Fashion-MNIST, CIFAR-10 (the latter only if pre-commit gate passes).
- **Success:** CSA matches Hamming k-NN within 0.3pp on at least one dataset, at measurably lower per-query cost (10 SDOT calls vs probe+union+resolver).
- **Unambiguous win:** CSA beats Hamming k-NN on any dataset — direct evidence that the substrate-native shape outperforms the scan shape at equal information.
- **Unambiguous loss:** CSA is >2pp below Hamming k-NN on MNIST (the cleanest benchmark). Stop; centroid is too coarse and E2 is the only remaining path.

**E2: perceptron-update CSA (conditional on E1 lossy-but-not-broken).**
- [ ] Only run if E1 is within 0.5-2pp of Hamming k-NN on MNIST. Integer-only perceptron updates: for each misclassified example, `W_{y}[d] += sig[d]`, `W_{y_pred}[d] -= sig[d]`. Sign-threshold at the end of each epoch. Bounded at k_epochs ≤ 5.
- [ ] Report: does epoch-by-epoch training close the gap? How much training buys how much accuracy?

## Implementation specification

### New consumer tool: `tools/csa_classifier.c`

```
Flags (reuses glyph_config where possible):
  --data DIR                MNIST / Fashion-MNIST / CIFAR-10 (auto-detect)
  --density F               signature density (same as direct_lsh)
  --gradients               append gradient channels (CIFAR-10 / Fashion)
  --normalize               per-image contrast normalization
  --no_deskew               skip deskew
  --train_mode MODE         "centroid" (E1) | "perceptron" (E2)
  --perceptron_epochs N     k_epochs for E2 (default 3)
  --emit_prototypes PATH    dump learned W_c trits to .c (for commit)
  --load_prototypes PATH    load prebuilt W_c (skip training)
```

### Data flow

```
load + deskew + normalize → MTFP signatures (unchanged from direct_lsh)
         │
         ▼
direct trit quantization (glyph_sig_quantize) → packed-trit signatures
         │
         ▼
TRAIN:                                  INFER:
  centroid per class                      for each query:
  ↓ (integer accumulate)                    for c in classes:
  sign(centroid − τ)                           score[c] = sdot(W_c, sig)
  → W_c ∈ {-1, 0, +1}^D                     pred = argmax(score)
         │
         ▼
  per-class prototype store (N_CLASSES × D trits)
```

### Kernel use

- Training: integer accumulation into per-class `int32_t cents[N_CLASSES][D]`, then `sign(cents[c][d] - tau)` per dim. No libm4t kernel needed — this is scalar.
- Inference: `m4t_mtfp4_sdot_matmul_bt(Y=scores, X=sig_mtfp4, W=W_packed, M=1, K=D, N=N_CLASSES)` — one call per query, 10-way score in parallel. Signatures convert to MTFP4 once at load time (precedent: `m4t_mtfp19_to_mtfp4`).
- Alternative kernel (fallback if SDOT alignment is awkward): `m4t_popcount_dist(sig_packed, W_c_packed, mask, sig_bytes)` per class — 10 calls per query, ternary Hamming distance, argmin. Less novel; already-measured kernel speed.

### What stays out

- Float anywhere at runtime. No softmax, no probability output — argmax only. Training is integer centroid or integer perceptron; sign-threshold lives in the same one-shot window as the pair-IG LUT build.
- Gradient descent. Class centroid + sign is the E1 rule; perceptron is the E2 rule. Neither is gradient descent.
- Bucket + multi-probe + union resolver. CSA short-circuits all of this: it does not filter candidates, it computes per-class scores directly.
- External training (Python / numpy). All training fits in C-only, integer, one-shot startup.

## Handling the major tensions

- **T1 (pair-IG learned vs LR learned — is the rule bent?):** the NO RANDOM WEIGHTS rule is read as **no weights without a derivation story**. Class centroid is derived (integer mean of signatures per class, sign-thresholded). Pair-IG is derived (entropy of class-conditional frequency). Both pass. Float-trained LR quantized to int8 would pass if the final artifact is the int8 table — but we don't take that path because we don't need to.

- **T2 (scoring headroom on CIFAR-10?):** explicitly gated by the pre-commit measurement. If no headroom, CSA runs only on MNIST/Fashion. We decide to commit or defer based on data, not speculation.

- **T3 (scaffold LR vs substrate-native?):** resolved by reframing. CSA is substrate-native. LR was a misidentification of the underlying operation.

- **T4 (training location):** fits the §12 fourth-exception pattern (startup one-shot LUT build) with integer-only arithmetic, so no new float site. If E2 perceptron turns out to need float momentum, the decision is: abandon E2 rather than introduce float.

- **T5 (rage against the trodden):** CSA is not LR-adapted; it's SDOT-shape-discovered. The trodden path (gradient descent + softmax + float weights) is explicitly declined.

- **T6 (learned weights on structured signatures):** CSA stacks class-level learned weights on top of structured per-dim input meaning. Multiplicative composition: input trit at dim d means "horizontal gradient at (x,y)"; CSA weight `W_c[d]` says "class c expects that gradient trit to have value [+1, 0, -1]." Both meanings coexist.

## Quality check

- **Could someone else execute this?** Yes — the implementation spec names the kernel, the data flow, the training rules, and the fallback kernel. The pre-commit measurement tells them whether to run E1 on all three datasets or just MNIST.
- **Does it address all nodes and tensions?** Yes — six tensions resolved explicitly; Nodes 1–16 map to either the E1/E2 design, the pre-commit gate, or the explicit "declined" list.
- **Is it simpler than the starting point?** Yes — the RAW page had two slots, three training options, four invariant questions, and a hovering "should we?" The synthesis is: one kernel (SDOT or VCNT), one training rule (centroid-then-sign), one gate (CIFAR-10 oracle), one measurement.
- **Surprised?** Yes. Coming in I expected the answer to be "don't bother; LR violates the rules." The real answer is "LR was the wrong name for an operation the substrate has been set up for since day one; do this now." That's a thesis-relevant win regardless of how the numbers come in.

## Immediate next action

1. Run pre-commit CIFAR-10 oracle measurement. **Do not build CSA yet.** If oracle ≤ 47.5%, scope CSA to MNIST/Fashion only for E1. If oracle > 50%, scope CSA to all three.
2. Report oracle result; proceed to E1 implementation based on gate outcome.

Estimated cost: oracle measurement ~75 min (one CIFAR-10 sweep). E1 implementation ~2–3 hours (new tool, test harness, three-dataset measurement). E2 implementation ~2 hours if needed.

## What this LMM pass actually produced

Not an LR integration. Not "run a quantized ML classifier." A **named primitive** — Class-Signature Argmax — that was latent in the substrate's kernel surface, unused by any consumer, and structurally the right answer to the question LR would have been a base-2-shaped answer for. The reframe is the work product; E1 is the falsifiable follow-through.
