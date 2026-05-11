# #7 Phase A.1 — Gumbel-softmax test (pre-registration)

Cites:
- `journal/td27_7_phase_a_2026-05-11.md` — original Phase A pre-reg,
  which named Gumbel-softmax as the STE-fallback estimator.
- `journal/td27_7_phase_a1_result_2026-05-11.md` — Phase A.1
  INCONCLUSIVE result, identifying H1 (capacity) vs H2 (optimization
  surface) as the load-bearing distinction.

**Written BEFORE running.** Per the discipline lesson (pre-verdict
overclaim pattern caught 4× this session): pre-register criteria so
the verdict is checkable against constraints defined before any data
is observed.

## Purpose

Distinguish H1 from H2:
- **H1 (capacity gap):** substrate needs more dimensions/depth than
  dense to encode the position-to-position mapping under the
  discrete-routing constraint.
- **H2 (optimization-surface gap):** STE through `gather` produces
  poor gradients for variable-length where the right K varies across
  sequences. The substrate routing's GRADIENT SURFACE is the
  bottleneck, not the model's capacity.

Gumbel-softmax CHANGES the gradient surface WITHOUT changing
capacity. If Gumbel-substrate converges where STE-substrate
plateaued, H2 is supported. If both plateau, H1 is more likely.

## Hypothesis (pre-registered)

At 2 layers, variable-N copy, with Gumbel-softmax replacing STE:
- **Dense**: unchanged from Phase A.1 (Gumbel-STE doesn't apply;
  dense doesn't use top-k selection). Borderline pass.
- **Substrate-Gumbel**: hypothesis being tested. Predict:
  - If H2 (optimization surface): substrate-Gumbel converges
    where substrate-STE plateaued.
  - If H1 (capacity): substrate-Gumbel also plateaus.

## Implementation design

`SubstrateGumbelAttention` — new variant. Mechanism:

1. **Hard forward selection** by substrate signature distance
   (sign-based), same as `SubstrateRoutedAttention`. Top-k=4
   indices via `argsort` of distances. Non-differentiable.
2. **Soft backward signal** via a differentiable relaxation:
   - Relaxed signatures: `tanh(Q * scale)`, `tanh(K * scale)`. As
     `scale` grows, tanh approximates sign; gradient flows.
   - Pairwise relaxed match: `sum_d tanh(Q*s)_d * tanh(K*s)_d`.
     High when relaxed signs agree (analog of "small signature
     distance").
3. **Gumbel noise** added to the relaxed match scores during
   training to permit stochastic exploration of selection.
4. **STE mask** via standard Gumbel-STE: `hard + (soft - hard).detach()`.
5. **Attention** uses full Q·K scores (dense compute for this test).
   Mask is applied via `scores + log(mask + eps)` before softmax —
   this is dense for the experiment but isolates the gradient-surface
   question from the sparse-compute question.

Critical design choice: **this variant uses DENSE compute** for
the experiment. The sparse-compute substrate (Phase A's
SubstrateRoutedAttention) is the production target; Gumbel-substrate
is the experimental tool that lets gradients flow through the
selection decision. If this passes, we know H2; if it fails, H1.

## Success / failure criteria (FROZEN)

**Outcomes:**

| outcome | dense passes | substrate-Gumbel passes | interpretation |
|---------|--------------|------------------------|----------------|
| **H2 SUPPORTED** | ≥1/3 seeds | ≥1/3 seeds | optimization-surface was the issue; Gumbel fixes it |
| **H1 SUPPORTED** | ≥1/3 seeds | 0/3 seeds | capacity is the bottleneck regardless of gradient |
| INCONCLUSIVE-A-FAILS | 0/3 seeds | n/a | dense still undercapacity; need 3-layer rerun |
| SUSPICIOUS | 0/3 seeds | ≥1/3 seeds | unexpected; investigate before claiming H2 |

**The dense baseline is reused from Phase A.1** (no need to re-run;
already shows 1/3 pass at 2 layers with seed 43). Only the new
variant (substrate-Gumbel) is added to the experiment.

**For substrate-Gumbel to "pass":** ≥95% test accuracy within 10000
steps on at least 1 of 3 seeds (same as Phase A.1's dense bar).

**For substrate-Gumbel to "fail":** all 3 seeds plateau below 95%.

## What this experiment IS

- A targeted test of H1 vs H2 for the Phase A.1 plateau.
- A check on the original Phase A pre-reg's STE-fallback (Gumbel-
  softmax) — is it the right tool when STE fails?

## What this experiment is NOT

- A production proposal. Gumbel-substrate is DENSE compute; not a
  candidate for the production sparse-routing path.
- A re-test of Phase A's fixed-N pass. The PASS there with STE
  stands; Gumbel isn't needed at fixed-N.
- A claim about substrate-everywhere. Activations are still float.

## Hyperparameters (FROZEN)

Same as Phase A.1:
- 2 layers, 4 heads, head_dim 16, model_dim 64, FFN inner 128
- RoPE
- variable-N ∈ {4..12}
- AdamW lr 3e-4, batch 32, cosine schedule to 1e-5 over 10000 steps
- Grad clip 1.0
- Eval every 100 steps, 1024 test sequences
- 3 seeds: 42, 43, 44

**Gumbel-specific:**
- `tanh` relaxation scale: 5.0 (approximates sign smoothly without
  saturating gradients at common Q/K magnitudes)
- Gumbel noise temperature τ: 1.0 (standard initialization; no
  annealing — keep simple for the test)
- Soft mask normalization: softmax over the relaxed match scores

## Compute budget

Roughly same as Phase A.1: ~1.5 hours for 3 seeds on M-series CPU.

## Sign-off

This pre-registration commits BEFORE the implementation or experiment
runs. The verdict criteria are FROZEN. Modifications require explicit
justification in the modifying commit.

If H2 is supported (substrate-Gumbel passes), it tells us:
- Substrate routing IS architecturally compatible with variable-length
  attention.
- The STE-through-gather estimator is the bottleneck for harder
  regimes, not the substrate itself.
- Production-bound substrate routing would benefit from either (a)
  better gradient estimators for the sparse path, or (b) train with
  Gumbel-dense → distill to sparse at inference.

If H1 is supported (both plateau), it tells us:
- Substrate routing has an architectural ceiling at this capacity
  for variable-length tasks.
- 3-layer (or wider) is the next test.

Either outcome moves the substrate's claim forward.
