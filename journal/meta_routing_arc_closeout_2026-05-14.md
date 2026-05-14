# Meta-routing arc closeout — three-layer architecture under iteration

**Date:** 2026-05-14
**Companions:** `meta_routing_arch_proposal_2026-05-14.md` (the
proposal and iter 1 falsification); `experiments/phase_zeta/results/
meta_iterate/anchors.json` (the 13-anchor store); commits b974a51 →
HEAD.

## What was claimed

Tripp's 2026-05-14 framing: a three-layer routing architecture
where layer 1 is the trit primitives, layer 2 is parameterized
programs (compositions of primitives), and layer 3 is meta-routing
— a search over layer-2 programs evaluated against a loss signal.
Two claims:

**A (architectural).** Layer 3 produces falsifiable predictions that
can be tested in hours.

**B (empirical).** Layer 3 can DISCOVER a layer-2 program beating
hand-coded qsigdist (+6.4pp Δ vs random on N=100 prompts, N=100
closeout closure).

The layer-2 program family: KV-eviction policies scored by

    score(slot) = w_r·age + w_kk·KK_sim + w_qk·QK_sim    (Python convention)
    evict argmin

with trit weights `(w_r, w_kk, w_qk) ∈ {-1, 0, +1}^3` (27 cells).
The four existing hand-coded modes (random, fifo, sigdist, qsigdist)
are fixed points in this space.

## What happened (iterations 1–8 + δ probe)

Layer 3 was implemented twice. The first version (regression-based)
fit linear coefficients to anchor Δs and predicted the highest-Δ
untested cell. The second version (kernel retrieval) followed the
Entropy Walker article "The Prior Should Be a Voice, Not a Verdict"
(shared by Tripp mid-arc): an append-only anchor store with a fixed
exp(-α·d) kernel over L1-trit distance, no learnable parameters, a
literal structural wall between prior holder and evidence reader.

| iter | architecture | candidate | predicted | observed | error | direction |
|---|---|---|---|---|---|---|
| 1 | regression | (0, -1, 1) | +13.3 | -3.0 | 16.3pp | wrong |
| 2 | regression | (0, 1, 1) | +15.8 | +2.0 | 13.8pp | right |
| 3 | regression | (1, 1, 1) | +9.8 | +4.25 | 5.55pp | right |
| 4 | regression | (0, -1, 0) | +7.0 | -13.88 | 20.9pp | wrong |
| 5 | **kernel + wall** | (-1, 1, 1) | +4.3 | +3.54 | **0.8pp** | right HIT |
| 6 | **kernel + wall** | (1, 0, 1) | +2.0 | +5.0 | **3.0pp** | right HIT |
| 7 | **kernel + wall** | (-1, 0, 1) | +0.8 | +5.83 | 5.1pp | right (hair-miss) |
| 8 α′ | **kernel + wall** | (1, -1, 1) | +0.4 (iso) / +2.9 (anis) | -4.21 | 4.6 / 7.1pp | wrong |
| δ | **kernel + wall** | (1, -1, 0) | -4.7 (iso) / -6.3 (anis) | -13.67 | 9.0 / 7.3pp | right MISS |

Regression: 4 iters, 2 directional wrong, mean error 14.1pp.
Kernel + wall: 5 iters, 1 directional wrong, mean error 4.3pp (iso) / 4.5pp (anis).

The structural-separation principle — wiring the prior holder
(anchor store) apart from the evidence reader (kernel retrieval) —
produced calibrated predictions where the regression version had
chased its own tail.

## The death zone

Iter δ was a targeted probe motivated by iter 4's outlier. `(0,-1,0)`
came in at -13.88pp — far worse than any neighbor predicted. Was that
local to that specific cell, or regional?

`(1,-1,0)` came in at -13.67pp — within 0.2pp of `(0,-1,0)`. The
w_kk=-1 ∧ w_qk=0 region is a **death zone**: keeping K-K-similar
slots without any Q-K signal drops eviction quality ~14pp below
random. The kernel had no way to predict this from neighbors because
no neighbor was *in* the death zone; the discontinuity is sharp on
the w_qk axis (going from w_qk=0 to w_qk=±1 inside w_kk=-1 changes
Δ by ~10pp), and L1-trit averages across the discontinuity to a
meaningless midpoint.

This is real surface structure that the chosen distance metric
cannot model. Without an embedding richer than ternary coordinates,
sharp regions like the death zone are invisible to neighborhood
predictors.

## The anisotropic-metric overfitting trap

After iter 7's hair-miss (5.1pp), I swept the kernel's hyperparameter
space looking for an anisotropic per-axis bandwidth `(α_r, α_kk,
α_qk)` that would close the magnitude gap. The sweep produced
α=(0.25, 0.25, 3.0): w_qk axis sharp (only same-w_qk neighbors
contribute), w_r and w_kk axes diffuse. LOO MAE dropped from 4.52pp
to 3.59pp; the iter 5/6/7 errors all came in under 5pp under the
refined metric. I declared the refinement validated.

It wasn't. On the genuinely-new cell `(1,-1,1)`:

```
isotropic α=1.0   predicted +0.39   error 4.60pp  HIT
anisotropic       predicted +2.90   error 7.11pp  MISS
```

The anisotropic kernel was wider-of-the-mark than the isotropic
baseline. It had retrofit the smooth qsigdist-family gradient into a
shape that didn't survive crossing the w_kk axis. **Classic
in-sample overfit signature**: lower LOO error, worse out-of-sample.

The lesson: a hyperparameter sweep that improves LOO is not metric
validation. Validate on a held-out point or accept that you're
fitting noise. This is now saved as memory
`feedback_in_sample_overfit_trap.md` for future arcs.

## Red-team of the harness (clean)

Before drawing architectural conclusions, ran a red-team on the
parameterized harness mode (`BITNET_KV_EVICT_META`) to rule out
that the calibration result was a measurement artifact:

1. **Score-function math** (sign convention, argmax↔argmin,
   sim↔dist): PASS by inspection.
2. **Fixed-point identicality** — `meta(-1,0,0)` ≡ `fifo`,
   `meta(0,+1,0)` ≡ `sigdist`, `meta(0,0,+1)` ≡ `qsigdist` —
   bit-identical tokens across 10 prompts × 3 pairs = **30/30 PASS**.
3. **Determinism** — same env, same prompt, same weights → same
   tokens: **6/6 PASS** across 3 prompts × 2 candidates × 2 repeats.

Latent bug noted but not affecting the arc: `meta(0,0,0)` is NOT
random — it picks the first alive slot deterministically. The
`(0,0,0)=0.0pp` anchor in the store came from the fixed `random` mode,
not `meta(0,0,0)`, so the bug never touched the iteration loop.

Files: `experiments/phase_zeta/redteam_meta_fixedpoints.py` (script
+ results), `redteam_meta_determinism.py` (script + results).

## Verdicts

| claim | verdict |
|---|---|
| A (architectural — Layer 3 makes falsifiable predictions in hours) | ✓ confirmed |
| Structural separation improves predictor calibration | ✓ confirmed (regression 14.1pp mean err → kernel+wall 4.3pp) |
| B (empirical — Layer 3 finds a policy beating qsigdist) | ✗ refuted (no untested cell predicted > +6.4pp; 13/27 cells anchored; observations confirm ceiling at qsigdist) |
| L1-trit distance is sufficient for this response surface | ✗ refuted (death zone invisible) |
| Hyperparameter-tuned anisotropic kernel generalizes | ✗ refuted (in-sample improvement was overfit) |

The architecture works AS AN EPISTEMIC MECHANISM. The empirical claim
fails: the linear-score program family `score = w_r·age + w_kk·KK +
w_qk·QK` over `{-1,0,+1}³` does not contain a policy strictly better
than qsigdist. The response surface has sharp structure (death zone)
that the chosen metric can't see; the kernel underfits this region
but the underfit is irrelevant for finding a champion because the
champions cluster near qsigdist.

## What's in the codebase

- `experiments/phase_zeta/meta_routing.py` — original prototype (regression).
- `experiments/phase_zeta/meta_iterate.py` — rebuilt as kernel retrieval + structural wall, with CLI: `status`, `propose`, `iterate`.
- `experiments/phase_zeta/run_one_candidate.py` — one-shot runner for arbitrary `meta(w_r, w_kk, w_qk)`.
- `experiments/phase_zeta/metric_refinement.py` — anisotropic bandwidth sweep + LOO analysis (refuted refinement).
- `experiments/phase_zeta/redteam_meta_fixedpoints.py` — anchor-mode identicality test (30/30 PASS).
- `experiments/phase_zeta/redteam_meta_determinism.py` — same-env reproducibility (6/6 PASS).
- `experiments/phase_zeta/meta_policy_battery.py` — iter 1 empirical runner (initial falsification).
- `experiments/phase_zeta/results/meta_iterate/anchors.json` — 13 anchors with full provenance.
- `gesh/bitnet/bitnet_harness.c` — `BITNET_KV_EVICT_META` mode with `_W_R`, `_W_KK`, `_W_QK` env vars.

## Why this matters

The arc was a controlled architectural experiment: Tripp proposed
three-layer meta-routing as scaffolding for learning over composed
primitives; I built it, tested it, refuted the strong empirical
claim, and validated the underlying structural principle. Two
takeaways carry forward beyond this specific problem:

1. **Structural separation of prior and evidence** is a real
   engineering pattern, not just an article-philosophy point. The
   regression vs kernel-with-wall comparison on the same data
   produced a measurable improvement in calibration; the
   contaminated-witness antipattern is detectable from error
   trajectories alone (regression errors high and high-variance;
   kernel errors lower with disagreement signal visibly recorded).

2. **L1 on coordinates is not enough** when the response surface
   has sharp boundaries between regimes. The death zone observation
   is a concrete instance of "the geometry that's natural to the
   substrate (path graph on trit alphabet) is not always the right
   geometry for the application." Future Glyph work that uses
   neighborhood predictors over routing-derived embeddings should
   be cautious about this — the trit alphabet's native geometry is
   load-bearing in many places, but a learned or
   application-specific metric may be needed where the response
   surface has sharp regimes.

## Forward

The next exploration is **β** from the earlier menu: enrich the
program family with a fourth feature (e.g., slot hit-count, attention
mass over time, or recency-of-last-attention). That expands the
search space from 27 to 81 cells and gives Layer 3 a genuinely new
family to explore — with the lessons of this arc (structural
separation YES, anisotropic-LOO tuning NO, validate on held-out)
already in hand.

The substrate's vision (ternary, routed, six-primitives floor) is
undisturbed by this work: the META mode is *built on* the six
primitives via the existing harness; this arc tests one specific
research question on top of an unchanged substrate.
