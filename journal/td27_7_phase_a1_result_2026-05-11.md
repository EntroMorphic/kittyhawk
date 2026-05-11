# #7 Phase A.1 — INCONCLUSIVE-A-FAILS; substrate gap widens at variable-N

Cites pre-registration: `journal/td27_7_phase_a1_prereg_2026-05-11.md`.

## Result table (3 variants × 3 seeds, 2 layers, RoPE, var-N ∈ {4..12}, 10000-step limit)

| variant | seed | pass-step | final_acc | final_loss |
|---------|------|-----------|-----------|------------|
| dense | 42 | None | 0.9268 | 0.031 |
| dense | 43 | **7800** | 0.9521 | 0.017 |
| dense | 44 | None | 0.9092 | 0.035 |
| substrate | 42 | None | 0.0039 | 2.015 |
| substrate | 43 | None | 0.0020 | 2.064 |
| substrate | 44 | None | 0.0020 | 2.356 |
| random | 42 | None | 0.0029 | 3.022 |
| random | 43 | None | 0.0010 | 2.995 |
| random | 44 | None | 0.0000 | 3.007 |

## Pre-registered verdict: INCONCLUSIVE-A-FAILS

Dense reaches ≥95% in only 1 of 3 seeds at 10000 steps. By the
pre-reg's INCONCLUSIVE protocol, the substrate comparison is
muddled by dense's own capacity issues; the prescribed next step
is "move to 3 layers and re-pre-register."

**Not declaring PASS or FAIL on substrate per pre-reg discipline.**
The data is informative but not enough for the architectural-claim
verdict.

## What the data does show

1. **Dense is on the edge of capacity.** Two seeds reach 91-93%
   accuracy at 10000 steps — clearly converging but not crossing the
   95% bar within budget. One seed (43) crosses at step 7800. With
   either more steps or slightly more capacity, dense likely
   converges consistently.

2. **Substrate plateaus far behind dense** at loss 2.0-2.4. This is
   QUALITATIVELY different from the fixed-N PASS regime (where
   substrate trained in 1.29× dense's steps). At variable-N + RoPE
   + 2 layers, substrate doesn't just need "more steps" — it appears
   to plateau, suggesting the optimization surface has changed.

3. **Random is consistently worst** (loss ~3.0). The substrate >
   random ordering from fixed-N is preserved (substrate loss 2.0 <
   random loss 3.0), but both are far from solving the task.

4. **The gap between substrate and dense WIDENED dramatically.**
   At fixed-N: 1.29× step ratio. At variable-N: substrate doesn't
   converge in 10000 steps while dense reaches 91-95%. This is the
   most significant finding — substrate's PASS on fixed-N may not
   generalize to "any harder task with sufficient capacity."

## Why this matters for the architectural claim

Phase A's PASS on fixed-N established **substrate-routed attention
is trainable** as the gradient-estimator gate. Phase A.1 was
designed to test whether that PASS generalizes to harder regimes
when capacity is sufficient.

What A.1 actually reveals: at 2× capacity + variable-N + RoPE,
substrate plateaus where dense converges. Two hypotheses:

**H1: Capacity gap.** Substrate needs MORE capacity than dense to
solve harder tasks, not less. Substrate's discrete top-k restricts
the Q/K representation to "sign-coherent direction" patterns; at
variable-length, the model needs more dimensions to encode the
position-to-position mapping under that constraint.

**H2: Optimization surface.** Substrate's signature-distance routing
creates a non-smooth optimization surface for variable-length
problems. The STE through gather works for fixed-N (where each
query has one obvious right key) but produces poor gradients when
the right key varies across sequences.

Both hypotheses predict the observed behavior (substrate lags
dramatically at harder regime). They make different predictions for
3-layer or wider:
- H1 predicts substrate converges with more capacity.
- H2 predicts substrate plateaus regardless of capacity.

The 3-layer rerun (or 128-dim rerun) discriminates.

## Per-pre-reg next step

The pre-reg's INCONCLUSIVE-A-FAILS protocol: "Move to 3 layers and
re-pre-register."

The honest amendment: 3 layers ALONE may not be enough. The
substrate gap is large enough that the issue might be wider
(model_dim) or harder (gradient estimator alternative) than just
deeper. The next pre-registration should specify which axis of
capacity is being explored:

- Phase A.1.b: 3 layers, same model_dim — tests H1 vs H2.
- Phase A.1.c: 2 layers, model_dim 128 — tests another capacity axis.
- Phase A.1.d: 2 layers, model_dim 64, Gumbel-softmax (per pre-reg
  fallback) — tests whether the gradient estimator is the issue.

## What stays true

- Substrate-routed attention IS trainable on fixed-N copy
  (Phase A `2188337`).
- Substrate's signature-based routing IS specifically load-bearing
  (random FAILS the same task; Phase A remediation `2188337`).
- Substrate has SOME signal even at variable-N (loss 2.0 vs random's
  3.0 in this experiment).

## What now requires investigation

- Whether substrate's training cost RATIO (vs dense) is stable across
  task difficulty, or whether it grows with difficulty until
  substrate fails to train.
- Whether substrate's plateau at variable-N is a CAPACITY issue (H1)
  or an OPTIMIZATION-SURFACE issue (H2).

## Verdict — INCONCLUSIVE; not a substrate fail, not a substrate pass

Per pre-reg discipline, this is INCONCLUSIVE-A-FAILS, not a verdict
on substrate. The data is honest evidence that substrate's
fixed-N PASS doesn't trivially generalize to variable-N at this
capacity level. Either H1 or H2 is true; the 3-layer or
Gumbel-softmax rerun distinguishes.

The pre-registration earned this clean verdict. Without writing
criteria before running, I'd have been tempted to either:
- Declare PASS on dense's seed 43 and ignore the substrate gap, or
- Declare FAIL on substrate's plateau and ignore that dense is also
  on the edge.

Neither would be honest. INCONCLUSIVE-A-FAILS is.

## Files

- Logs: `experiments/phase_a/logs/{dense,substrate,random}_{42,43,44}_a1.json`
- Pre-reg: `journal/td27_7_phase_a1_prereg_2026-05-11.md`
- This result: `journal/td27_7_phase_a1_result_2026-05-11.md`
