# phase_alpha/ — categorical-Hamming substrate measurements

**Status:** superseded for vision-claim tests by Phase β/γ. Still
valid for any measurement that intentionally uses categorical
Hamming on substrate signatures.

## Key files

| file | what it is | status |
|---|---|---|
| `m1_estimator.py` | v1 Macocco implementation | **demonstrably broken** — 38% bias on synthetic, calibration halted (see `journal/td27_phase_alpha_calibration_fail`). Kept as reference for the bug pattern. |
| `m1_estimator_v2.py` | v2 with ARCH-A (Macocco fixed-radii) and ARCH-B (corrected TwoNN order stats) | **correct under categorical Hamming.** Reused by Phase γ for Hamming-baseline measurements. |
| `calibrate.py` | v1 calibration (fails by design) | reference only |
| `calibrate_v2.py` | v2 calibration | passes 4/6 d targets within 1% on uniform synthetic |
| `load_k_signatures.py` | ACTV2 dump loader + signature/baseline builders + fast vectorized Hamming | load-bearing; reused by all phases |
| `run_phase_alpha.py` | original run script | findings superseded by `run_phase_alpha_v2.py` |
| `run_phase_alpha_v2.py` | remediated v2 with stricter rules, structured baselines, τ sweep, bootstrap CIs | **the M1 "reversal" finding here is correct under categorical Hamming, but Hamming is the wrong metric for the vision claim — see Phase β.** |
| `regenerate_dumps_v2.sh` | regenerates `data/c_dump_v2/` from `bitnet_harness` | needs harness binary + weights |

## Findings (under categorical Hamming, supersaded for the vision)

- v1 calibration: FAILED (38% bias). Stopped per pre-reg.
- v2 calibration: PASSED on synthetic uniform-ternary.
- v2 Phase α run on K-cache: "VALIDATED 2/3" originally; red-team
  exposed M3 degenerate rule + M2 fails both criteria; remediation
  reversed M1 under stricter rules. Final: **MIXED 1/3 under
  categorical Hamming, with the M1 reversal driven by unit-of-
  measure mismatch (different ambient D between substrate and B2)**.

## Why this is here at all

Phase β/γ revealed that **categorical Hamming on substrate destroys
the alphabet's path-graph structure** (the third state's role as
the natural center is invisible to Hamming). So the Phase α
verdicts don't speak to the vision claim. But the *machinery*
built here (ACTV2 loading, signature construction, Macocco
estimator, B2/B3/B4 baselines, bootstrap CIs, regime stratification)
all carries forward unchanged. Only the metric needed to change.

See `journal/td28_phase_alpha_methodology_pivot_2026-05-12.md` for
the framing pivot.
