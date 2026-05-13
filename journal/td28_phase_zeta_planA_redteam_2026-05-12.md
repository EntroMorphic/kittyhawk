# Plan A red-team — strengthens the negative finding

User directive: "Red-team and remediate A. Then proceed with B."

Plan A measured Spearman ρ between production sigdist's K-K
direction proxy and Phase ε's Q-K oracle. Across M ∈ {1,2,4,8,16},
ρ stayed in [+0.03, +0.06]. Conclusion as written: K-K proxy is
inadequate, not fixable by M.

Red-team adds three checks. The first two reproduce or refine; the
third dramatically *strengthens* the negative finding.

## R1 — Calibration baselines

Compute ρ vs Q-K oracle for the non-K-K policies that plan A
implicitly relied on as "what ρ ≈ 0 looks like":

| ranking      | mean ρ   | median   | p10      | p90      |
|--------------|---------:|---------:|---------:|---------:|
| sigdist K-K  | +0.0548  | +0.0741  | -0.355   | +0.434   |
| fifo         | +0.0059  | +0.0144  | -0.478   | +0.474   |
| random       | +0.0008  | +0.0007  | -0.243   | +0.245   |

Sigdist's +0.055 is real but very small. Fifo is ~zero. Random is
~zero by construction. The "+0.055 is essentially uncorrelated"
framing in plan A is fair; if anything, sigdist has *more* signal
than fifo, but the signal doesn't translate (see R2).

## R3 — Per-layer breakdown reveals a sign flip

Aggregate ρ ≈ +0.055 hides structure across the 30 layers:

| layer band   | K-K ρ range          | fifo ρ range         | interpretation              |
|--------------|----------------------|----------------------|------------------------------|
| 0-3 (input)  | +0.15 .. +0.21       | +0.19 .. +0.33       | recency ≈ relevance          |
| 4-13         | mixed, mostly weak + | mixed, mostly weak + | unclear                      |
| 14-27 (deep) | -0.03 .. -0.14       | **-0.07 .. -0.36**   | recency anti-aligns          |
| 28-29        | mixed                | mixed                | terminal                     |

**fifo is sharply positive in early layers and sharply negative in
deep layers.** Deep layers attend to *non-recent* content based on
semantic processing — fifo's "keep recent" rule actively kills the
right answer. K-K follows a weaker version of the same pattern,
because K is "smeared" position information in early layers and
diverges from position in deeper layers.

This explains why aggregate ρ ≈ 0: the early-layer positive signal
and deep-layer negative signal cancel. Production sigdist is
*sometimes* doing the right thing (input layers) and *sometimes*
doing the wrong thing (deep layers), and on net it's zero. That's
worse than "uniformly random in direction" — it's "systematically
inverted in the half of the network that matters most for output."

## R2 — Attention-output L2 error (the headline upgrade)

Plan A measured ranking correlation. Red-team R2 measures the
*consequence*: attention-output L2 error vs no-eviction, using
each policy's eviction selection. This is on the same infrastructure
Phase ε used.

| policy                       | k_keep=8 | k_keep=16 | k_keep=32 |
|------------------------------|---------:|----------:|----------:|
| oracle Q-K                   | 0.150    | 0.066     | **0.016** |
| hamming                      | 0.242    | 0.129     | 0.043     |
| **K-K (production sigdist)** | **1.476**| **1.365** | **1.181** |
| fifo                         | 1.512    | 1.436     | 1.351     |
| random                       | 1.354    | 1.032     | 0.585     |

**At k_keep=32, K-K's L2 error is 2.02× random's.** Plan A's
conclusion was understated. The K-K proxy is not "uncorrelated with
the oracle" — it's actively *anti-correlated with attention
relevance* in single-shot terms.

**Mechanism of K-K's anti-correlation:** sigdist *keeps* K's most
similar to current K (smallest L1 distance) and evicts the most
distant. Keeping near-duplicates of current K means the cache
becomes redundant — multiple K's pointing in the same direction in
trit-sig space. Random keeps a diverse sample, which preserves more
information per kept slot. The result is K-K throws away exactly
the diverse K's the attention needs.

Fifo is even worse (1.35 at k_keep=32) because it aggressively
keeps the most-recent positions, exactly the wrong move in deep
layers where attention is non-recent.

**Sanity:** random's L2=0.585 reproduces Phase ε's reported 0.584;
oracle Q-K's L2=0.016 reproduces Phase ε's reported 0.016. Same
infrastructure, same numbers. The new K-K number 1.181 is on the
same scale and directly comparable.

## Why the harness Phase ζ result was only "≈ random," not "2× worse"

The single-shot k_keep=32 oracle drops 18+ positions at once,
amplifying K-K's bias. The harness drops one position per step;
each step's bias is small, and cumulative damage over 16-24 steps
in the eviction window doesn't fully amplify to the 2× single-shot
result. Phase ζ showed sigdist match-rate **slightly worse** than
random (5 vs 0 argmax flips on bos_only at window=16, 5 vs 3 on
medium_11). The mechanism is the same direction; the magnitude is
muted by per-step amortization.

## Updated plan A verdict

- Code match between Python M-mean and production
  `bitnet_harness.c:541-568`: verified line-by-line.
- ρ vs Q-K with M sweep: still flat at ~0.05 across M ∈ {1..16},
  decreasing toward zero. No env-var fix.
- ρ baselines: sigdist not meaningfully better than fifo at
  approximating Q-K; both are dominated by per-layer sign flips
  that average out.
- **L2-error consequence: K-K eviction is 2× worse than random.**
  Production sigdist isn't just "non-improving" — it's
  *attention-harmful* in single-shot oracle terms.

The remediated conclusion is stronger and more actionable: production
sigdist as currently implemented should not be enabled. The path
forward is plan B (defer eviction to Q-aware step) or plan C
(deprecate). Plan A is closed.

## Files

- `experiments/phase_zeta/m_sweep_correlation.py` — original plan A.
- `experiments/phase_zeta/redteam_a.py` — R1/R2/R3 augmentation.
- Per-layer table above derived from R3 output (run 2026-05-12).

## Discipline log

Red-teaming plan A produced a stronger negative finding than plan A
itself. The first version stopped at "ρ ≈ 0, no fix exists."
Red-team R2 went one layer deeper and asked "what's the L2-error
consequence?" — and got 2× worse than random. Pattern: when a
negative result is reported, ask whether the metric used understates
or overstates the negative. Here it understated.

Adding to `feedback_spot_check_before_verdict`: the discipline
applies symmetrically to negative findings, not just positives. A
mild-negative finding can still be incorrectly mild.
