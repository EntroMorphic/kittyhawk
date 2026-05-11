# #7 Phase A.1 — variable-length with proper capacity (pre-registration)

Per `journal/td27_7_phase_a_remediation_2026-05-11.md` finding that
1-layer 51K-param model was undercapacity for variable-length copy.
Phase A.1 tests whether substrate's PASS on fixed-N generalizes to
the harder variable-length regime when capacity is sufficient.

**Written BEFORE running, per the discipline lesson from Phase A
(commit messages noting 4 caught overclaims; the antibody is to
pre-register the verdict criteria before the experiment).**

## Hypothesis (pre-registered)

At 2 layers with RoPE and variable-length copy:
- Dense will converge to ≥95% test accuracy.
- Substrate will converge to ≥95% in ≤2× dense's pass-step.
- Random will FAIL to converge (≥95% threshold not reached within
  10× dense's pass-step OR 10000 steps, whichever is smaller).

## Success / failure criteria (FROZEN)

**Phase A.1 PASS:** all of:
1. Dense reaches ≥95% in ≤10000 steps.
2. Substrate reaches ≥95% in ≤2× dense's pass-step.
3. Random does NOT reach ≥95% (counter-test).

**Phase A.1 FAIL — substrate side:** substrate plateaus below 95% at
2× dense pass-step OR substrate diverges. Recorded as Phase A.1
failure with explicit mode (PLATEAU / DIVERGE / SLOW).

**Phase A.1 INCONCLUSIVE — dense side:** if dense itself doesn't
reach 95% at 10000 steps, model is still undercapacity. Move to 3
layers and re-pre-register.

**Phase A.1 SUSPICIOUS — random side:** if random ALSO reaches 95%,
substrate's load-bearing claim from fixed-N doesn't generalize to
variable-length. This would be a partial RETRACTION of the random-
fails-on-fixed-N result; would need deeper investigation.

## What changes from Phase A

Only the architecture's `n_layers` (1 → 2). Everything else held
constant:
- Same TinyGPT, same heads (4), same head_dim (16), same model_dim
  (64), same FFN inner (128).
- Same RoPE position encoding (Phase A.1's variable-length test).
- Same task (variable-N ∈ {4..12} sequence copy).
- Same hyperparameters (AdamW lr 3e-4, batch 32, cosine schedule).
- Same eval (every 100 steps, 1024 test sequences, ≥95% pass).

Expected param count: ~80K (the second layer adds attention + FFN
weights). Still tiny.

## Seeds: 3 per variant (42, 43, 44)

Total 9 runs. Wall-clock expected: 2-layer is roughly 2× per step,
plus variable-N runs to 7000+ steps potentially. Estimate ~20 min
total on M-series CPU.

## What this commit IS

The pre-registration. No experiment code change yet beyond reading
this document.

## What this commit IS NOT

The result. The result will be in a separate commit
(`journal/td27_7_phase_a1_result_2026-05-11.md`) AFTER the experiment
runs, citing this pre-registration.

The split is intentional: it makes the verdict checkable against the
criteria defined before any data was observed. This is the
discipline antibody for the pre-verdict overclaim pattern — write
the criteria, commit them, THEN run. Even if the user didn't ask me
to codify the pattern, applying it here is the local fix.
