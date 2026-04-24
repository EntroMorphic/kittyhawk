---
date: 2026-04-23
scope: LMM cycle — routed autodiff engine in NEON-optimized C. Thesis-aligned substrate extension. Enables S7 and future learned-encoder experiments without leaving the substrate.
phase: RAW
---

# RAW: routed autodiff in NEON C

User said: autodiff in NEON-optimized C, routed not dense. This is a thesis-level move. NORTH_STAR: "routing is essential, and will naturally outperform dense, in a base-3 environment." Every autodiff library I know is dense: PyTorch/TF/JAX/tinygrad all think in terms of `Y = X @ W`, dense matmul forward, dense matmul backward. Routing is absent from their primitive set.

This cycle is about what routed autodiff even LOOKS like.

## What autodiff usually means

Standard (dense) forward: `Y = X @ W + b`.
Standard backward: given dY (gradient of loss w.r.t. Y), compute `dX = dY @ W.T`, `dW = X.T @ dY`.

Every modern deep net is chains of these, plus nonlinearities whose backwards are known. PyTorch builds the chain dynamically; autograd traces ops and replays them backward.

**This is dense from end to end.** Every weight participates in every forward pass, every weight receives a gradient, every gradient flows through every connection.

## What routing means in Glyph

Current substrate has `m4t_route` primitives:
- `m4t_route_threshold_extract`: activations → packed-trit signatures (sign-at-threshold).
- `m4t_route_distance_batch`: query-sig × T tile-sigs → T Hamming distances.
- `m4t_route_topk_abs`: T scores → k (tile_idx, sign) decisions.
- `m4t_route_apply_signed`: k (tile_idx, sign) × T tile-outputs → accumulated result.
- `m4t_route_signature_update`: setup-time weight signature computation.

A routing forward pass, structurally:
```
1. signature_update(weights) → sig_tiles[T]          // once, setup
2. threshold_extract(query_act) → query_sig          // per token
3. distance_batch(query_sig, sig_tiles) → scores[T]  // per token
4. topk_abs(scores, k) → decisions[k]                // per token
5. per-selected-tile matmul via m4t_mtfp_ternary_matmul_bt  // per token per decision
6. apply_signed(decisions, tile_outs) → result       // per token
```

Only k of T tiles contribute to the output. 1/3 of weights are 0. Signature match is sparse. Decision boundaries are non-smooth (argmax / topk).

**This is fundamentally sparse dispatch over dense pieces**, not dense over everything.

## Why dense-autodiff backward doesn't port trivially

The routing pipeline has four non-differentiable ops:

1. **`threshold_extract`** — piecewise sign function. Zero gradient almost everywhere. Classical STE workaround: backward is identity (or clipped identity).

2. **`topk_abs`** — discrete selection. Gradient is zero for unselected tiles and "how much the decision was made" for selected tiles. Classical approaches: soft top-k (differentiable relaxation); STE (pretend it's identity backward for the selected tiles); score-function estimator (REINFORCE).

3. **`route_apply_signed`** — signed accumulation gated by k binary decisions. Gradient flows only through selected tiles; must be zero into unselected tiles. STE treats this as transparent.

4. **Trit weights themselves** — `W ∈ {-1, 0, +1}`. Not differentiable in weight space. Standard: maintain a float "latent" weight, backward updates the float, periodic re-quantization to trit.

So the autodiff engine needs:
- A sanctioned STE backward for each routing primitive.
- Latent-float weight storage separate from the deployed trit weights.
- Gradient routing that mirrors the forward routing — dL/dW only gets contributions at the selected tiles.

## What makes this NEON-optimized

- Forward SDOT uses vdotq_s32. Backward dW would be `sum_over_tokens (outer(x, dy))` restricted to selected tiles. This is also SDOT-shaped on selected tile blocks, but needs packing into int32 tile buffers.
- Forward distance is popcount_dist. Backward gradient through the distance: STE says "identity"; implementation is just pass-through of dY into the sign positions.
- Backward accumulation of dL/dW is the main hot loop. For ternary weights stored in packed-trit form, the backward accumulates INTO FLOAT latents. So we have:
  - Float latent weights: `float* W_latent[T][H][D]` (or int32 fixed-point for no-float runtime)
  - Trit weights packed: `uint8_t* W_trit_packed[T][H][Dp]`

Wait, "no float at runtime" per substrate discipline. Training is not runtime in the production sense, but it IS running a forward+backward loop repeatedly. Does this count?

NORTH_STAR §4 and §12 explicitly sanction build-time LUT generation as a "float once, integer forever" pattern. Training is **build-time for model weights** — float iteration, integer deployment.

So: fixed-point latents with integer accumulation might be an option. But SGD with fixed-point needs careful bit-width management — the gradient step `W += lr * dW` rounds hard. Either:
- (a) Accept float latents during training; export integer/trit after.
- (b) Use INT32 fixed-point latents with wider accumulators for dW.

Option (a) is classical. Option (b) is substrate-purist but harder and potentially less expressive. First pass should (a); revisit.

## What's the MVP?

Minimum viable routed autodiff that answers the S7 question on CIFAR:

**Single-layer routed ternary encoder.**
- Input: MTFP pixels + gradients (as direct_lsh builds them).
- Layer: T ternary weight tiles of shape [H, D_out/T] (T tiles, H rows each, total output dim D_out).
- Routing: per-image signature derived from input; top-k tile selection.
- Output: sign-quantized per-dim value → ternary signature of dim D_out.
- Loss: auxiliary classifier head on pre-sign activations (like I sketched for PyTorch S7).
- Training: integer or float latent weights, STE backward through routing gates, export packed trits.

T maybe = 16 tiles, k = 4 per query, H = 256, D_out = 11000-ish (match MS4+R4).

## What scares me

- **Autodiff engine is hard.** Even a simple one needs: forward ops emitting pairs of (op, inputs, output); backward traversal; gradient accumulation; parameter updates. In C, without a language-level tape, this is manual bookkeeping. Getting it right is not trivial.

- **Routing gradients are bespoke.** Every routing primitive needs its own backward. STE is a choice, not a derivation — the wrong STE can make training diverge or train to wrong solutions.

- **No Python escape valve.** If I hit a numerical issue, I can't quickly prototype in PyTorch to check. Everything has to be C, debugged in C.

- **Substrate pollution risk.** If the autodiff code lives in libm4t, suddenly libm4t has training-concern state. NORTH_STAR §13: "Training artifacts live in the consumer." So autodiff lives in a CONSUMER TOOL, not libm4t. That's the right place.

- **Scope explosion.** A full autodiff engine in C is a multi-week project. The user said "let's autodiff in NEON-optimized C, routed." That could mean:
  - (a) Build a minimal routed trainer for S7. ~1-2 weeks.
  - (b) Build a general-purpose base-3 autodiff engine. ~1-3 months.
  - (c) Build something in-between that handles a small family of routed ops but is extensible.

Narrow interpretation: MVP = ONE routed ternary-encoder trainer. Answers S7, demonstrates the pattern, proof-of-concept for a larger engine. User can extend later.

## What base-3 actually does differently

Dense autodiff: every weight contributes to every gradient step. Gradient signal is maximally informative but computation is O(P) per step where P is parameter count.

Routed autodiff: only k-of-T tiles contribute per token. Gradient sparsity matches forward sparsity. Computation per step is O(k/T × P) — 4/16 = 25% of dense. At matched training budget, you train 4× more steps or 4× bigger models.

The thesis claim: base-3 routing is substrate-aligned at BOTH forward AND backward. Dense autodiff wastes gradient budget on weights that didn't contribute. Routed autodiff concentrates the gradient on what actually fired.

## What a "routed gradient" MIGHT look like (scratch)

Forward per token (simplified):
- signature(x) → s ∈ trit^D
- distance_batch(s, W_sig) → scores ∈ int32^T
- topk(scores, k) → decisions = [(t_1, sign_1), ..., (t_k, sign_k)]
- for each decision (t_i, sign_i): partial_result += sign_i × (x @ W_tiles[t_i])
- Loss = L(partial_result, y)

Backward:
- dL/d(partial_result) = dy
- For each decision (t_i, sign_i):
    dL/dW_tiles[t_i] += sign_i × outer(x, dy)   // SDOT-shaped
    dL/dx += sign_i × (dy @ W_tiles[t_i].T)
- For unselected tiles t ∉ decisions: dL/dW_tiles[t] = 0 (no update this step)
- dL/dW_sig: zero (STE — signature matching is non-differentiable)
- Parameter update: W_tiles += lr × dL/dW_tiles (on latent float), periodically re-quantize to trit

The signature matrix W_sig is updated INDIRECTLY via the signature_update routine applied to new W_tiles. Not through gradient flow.

## Open questions

1. **Float latents or integer latents during training?** Float is pragmatic; integer is substrate-pure. First pass float, revisit.

2. **How to handle STE for top-k?** Simplest: during backward, treat top-k as transparent — gradient flows through the k selected paths as if the selection were identity. Harder: use soft-top-k (softmax over scores, differentiable). Start with STE.

3. **How much of a general autodiff engine to build?** If I build ONLY the routed-ternary-encoder, it's one specific training loop, not an engine. If I build a general engine, it's months. Compromise: build a minimal forward-backward framework with 4-5 primitive operations (signature_extract, tile_matmul, topk, apply_signed, sign_quantize), all with hand-derived backwards. Composable but narrow.

4. **Where does it live in the tree?** New directory `train/` or `tools/autodiff/`? Not libm4t (too low-level). Not libglyph (consumer infrastructure, not training). Fresh directory.

5. **What are the specific kernels NEON needs for backward?** SDOT backward — accumulate dW from (x, dy) pairs. VCNT backward — pass-through STE. TBL backward — bespoke per trit op. This has to be designed before coding.

6. **Does this LIVE in the repo or is it an external branch?** Given scope, the user may want to see a spec before a 2-week commit. I propose: spec in synthesize phase, confirm before coding.

## First instincts

- **Scope: tightest possible MVP.** One consumer tool `tools/routed_trainer.c` that implements a single-layer routed ternary encoder + STE backward + latent float weights + packed-trit export. Not a general engine. ~1 week if I'm careful.

- **Math first, code second.** Write down the backward for each routing primitive on paper (in the reflect phase). Check the STE choices are sound. Only then implement.

- **Reuse forward kernels.** libm4t's forward primitives are already NEON. The trainer tool composes them for forward. Backward is new NEON code in the tool, not in libm4t.

- **Gate: routed-ternary-encoder trained on CIFAR hits ≥50%.** If it beats direct quantization by ≥2pp, the autodiff works and the encoding story is validated. If it underperforms direct quantization, training is buggy or STE choice is wrong. If it matches (48-49%), ambiguous — need more training or different architecture.
