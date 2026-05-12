# #7 Phase A.1 Gumbel result — H1 supported (per pre-reg) with H2 also partially supported

Cites:
- `journal/td27_7_phase_a1_gumbel_prereg_2026-05-11.md` — pre-registration.
- `journal/td27_7_phase_a1_result_2026-05-11.md` — the STE result that
  this experiment was designed to disambiguate.

## Result table (3 seeds, 2 layers, RoPE, variable-N, 10000-step limit)

Reuses Phase A.1 dense / substrate (STE) / random results; adds
substrate-Gumbel as the new variant.

| variant         | seed | pass_step | final_acc | final_loss |
|-----------------|------|-----------|-----------|------------|
| dense           | 42   | None      | 0.9268    | 0.031      |
| dense           | 43   | **7800**  | 0.9521    | 0.017      |
| dense           | 44   | None      | 0.9092    | 0.035      |
| substrate (STE) | 42   | None      | 0.0039    | 2.015      |
| substrate (STE) | 43   | None      | 0.0020    | 2.064      |
| substrate (STE) | 44   | None      | 0.0020    | 2.356      |
| **substrate-Gumbel** | 42 | None  | **0.0205** | **1.483** |
| **substrate-Gumbel** | 43 | None  | **0.0215** | **1.446** |
| **substrate-Gumbel** | 44 | None  | **0.0166** | **1.412** |
| random          | 42   | None      | 0.0029    | 3.022      |
| random          | 43   | None      | 0.0010    | 2.995      |
| random          | 44   | None      | 0.0000    | 3.007      |

## Pre-registered verdict: H1 SUPPORTED (with H2 nuance)

Per the FROZEN outcome table in
`journal/td27_7_phase_a1_gumbel_prereg_2026-05-11.md`:

> | outcome | dense passes | substrate-Gumbel passes | interpretation |
> |---------|--------------|------------------------|----------------|
> | H1 SUPPORTED | ≥1/3 seeds | 0/3 seeds | capacity is the bottleneck regardless of gradient |

Dense reaches 95% in 1/3 seeds. Substrate-Gumbel reaches 95% in
0/3 seeds. By the letter of the pre-registration: **H1 SUPPORTED**.

## What the data also shows (the H2 nuance)

The pre-reg's binary PASS / FAIL framework misses an important
quantitative pattern. Substrate-Gumbel makes **substantial progress
over substrate-STE** that's invisible at the PASS threshold:

| metric | substrate-STE | substrate-Gumbel | gap |
|--------|--------------|------------------|-----|
| final loss (mean) | 2.145 | **1.447** | -0.70 nats (32% reduction) |
| final accuracy (mean) | 0.0026 | **0.0195** | ~7× better |
| improvement over random | loss -0.86 | loss **-1.56** | substrate-Gumbel pulls FURTHER from random than substrate-STE does |

Interpretation:
- **Gumbel-softmax DOES improve the gradient surface for substrate
  routing.** Even though it doesn't unlock PASS at 2 layers, it
  consistently produces lower loss and higher accuracy than STE.
- **But 2-layer capacity is insufficient regardless of gradient
  estimator.** Substrate-Gumbel still plateaus far from dense's
  converged loss (1.4 vs 0.03 — two orders of magnitude).

**Both H1 and H2 are partially supported.** The clean H1-vs-H2
framing in the pre-reg was too binary; the actual data shows a
**combined bottleneck**.

## What this commits the project to

1. **Substrate routing has TWO known bottlenecks at variable-N copy:**
   gradient surface (H2; Gumbel helps but doesn't fully fix) AND
   capacity (H1; 2 layers insufficient).

2. **The path to substrate convergence at variable-N requires
   addressing BOTH:**
   - Better gradient estimator (Gumbel-style or beyond) for the
     discrete top-k decision.
   - More capacity (3+ layers, or wider).

3. **For production use at small scale, sparse-compute substrate
   routing (Phase A's `SubstrateRoutedAttention`) trains at
   fixed-N with STE.** Variable-length needs the combined fix.

4. **The 1.29× step ratio at fixed-N is the BEST CASE.** At harder
   regimes, the substrate's training cost vs dense grows
   non-linearly (and currently exceeds the model's effective budget).

## Updated H1/H2 framework

Strict binary: one or the other.
- H1: capacity is the bottleneck.
- H2: optimization surface is the bottleneck.

Refined (per data): both are bottlenecks; the data tells you which
contributes more:
- **Gumbel reduces loss by 0.7 nats** (vs STE) → H2 contributes
  meaningfully but not decisively at this scale.
- **Loss remains far from convergence** (1.4 vs dense's 0.03) → H1
  is the larger immediate barrier.

Next test: substrate-Gumbel + 3 layers (or substrate-Gumbel + wider
model). If THAT converges, both bottlenecks are tractable. If it
plateaus, there's a third bottleneck not yet identified.

## What this experiment validates

- **The original Phase A pre-reg's STE-fallback prescription was
  partially right.** Gumbel-softmax IS useful when STE plateaus —
  not as a magic fix, but as one of two parallel improvements needed.
- **Pre-registration discipline (split-commit) catches nuance.**
  Without the pre-reg's outcome table, I would either:
  - Cherry-pick the "Gumbel improves loss" framing → declare H2 win.
  - Default to "neither passes" → declare both fail.
  Neither is fully honest. The H1-supported-with-H2-nuance verdict
  IS honest.

## What this experiment does NOT validate

- That substrate routing CAN converge at variable-N. The combined
  fix (Gumbel + 3 layers) remains untested.
- The full architectural Part-B claim. Still Phase B/C territory.
- Production-bound substrate viability. Gumbel-substrate uses dense
  compute (full Q·K) — not the production sparse path.

## Red-team

### C1: 0.02 accuracy isn't "real progress" — both are essentially 0
2% accuracy is well below 95%. From a PASS perspective both
substrate-STE and substrate-Gumbel fail equally. The relative
improvement (0.2% → 2%) is a 10× ratio but absolute remains tiny.
Is this signal or noise?

Counter: the LOSS gap (2.1 → 1.4) is large and consistent across
seeds (variance ~0.05 within each variant, gap ~0.7 across variants).
That's signal, not noise. The accuracy is just downstream of loss;
it's binary at the per-position level and only becomes nonzero when
loss is low enough that the argmax aligns with truth.

### C2: tanh(x * 5.0) saturation
At Q magnitudes typical for this model (~1.0 by initialization),
tanh(5x) saturates near ±1 for |x| > 0.3. This makes the
"relaxation" very close to sign() in most cases, which means the
Gumbel-relaxed score is close to the actual signature match. Could
explain why Gumbel doesn't fully unblock — the relaxation is too
tight. Looser scale (e.g., 2.0) might give smoother gradients.
Recorded as a follow-up.

### C3: Gumbel τ=1.0 with no annealing
Standard Gumbel-softmax papers anneal τ from high (smooth) to low
(sharp) during training. I used fixed τ=1.0 for simplicity. With
annealing, the gradient signal might be much smoother early in
training. Recorded; not a verdict-changer at this scale.

### C4: Single-task evaluation
Variable-length sequence copy is one task. Substrate's plateau here
might be specific to this task structure. A different task (e.g.,
key-value lookup at the same N range) might give different results.
Recorded as future follow-up.

### C5: I'm interpreting "H1 SUPPORTED with H2 nuance" — is that
honoring the pre-reg or post-hoc reframing?

The pre-reg's outcome table maps cleanly to H1 SUPPORTED. That
verdict stands. The "H2 nuance" is ADDITIONAL information observable
in the data that the binary framework didn't anticipate. Reporting
both honestly is correct; it's not post-hoc reframing.

If I were to claim "H2 supported" by emphasizing the loss
improvement and downplaying the PASS failure, THAT would be
post-hoc reframing. I'm not doing that. The verdict from the
pre-reg's letter is H1; the data also says H2 contributes
non-trivially.

## Sign-off

Per pre-reg discipline, this is H1 SUPPORTED with explicit H2
nuance recorded. The substrate's architectural Part-B claim survives
in its Phase A form (fixed-N PASS); variable-N requires combined
capacity + gradient-estimator fixes. Phase A.1.b (3-layer + Gumbel)
is the natural next test.

For this cycle, this commit is the methodical closure on the
H1 vs H2 question.
