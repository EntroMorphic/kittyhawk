# phase_beta/ — L1 cell-graph substrate measurements

**Status:** superseded for verdict-labeling by Phase γ. Estimator
math and infrastructure remain load-bearing.

The cell-graph framing (substrate as a product of path graphs with
0 as the natural center) is correct; Phase β is where it gets
implemented. The single-metric verdict claim from this directory
(`run_phase_beta.py` → "VALIDATED 3/3") was an artifact of the
`d̂/Dmax` normalization choice and collapses under `d̂/D_ambient`
(see `journal/td28_phase_beta_redteam_2026-05-12.md`). Phase γ
replaces single verdicts with a robustness matrix.

## Key files

| file | what it is | status |
|---|---|---|
| `m1_l1_estimator.py` | L1 Macocco estimator with shell-volume DP convolution from per-cell PMF | **correct on uniform/structured synthetic. Biased ~45% LOW on correlated synthetic** (γ-G finding). Use absolute d̂ as conservative; relative comparisons are valid. |
| `calibrate_L1.py` | calibration on structured-ternary (p_nonzero=0.62) | passes 4/4 d targets within 4%; doesn't cover correlated data |
| `run_phase_beta.py` | full Phase β pipeline | findings superseded by Phase γ. The B5 scrambled-ternary (P3 control) implementation here is reused by γ. |
| `results/phase_beta_results.json` | Phase β numeric results | "VALIDATED 3/3 under d̂/Dmax" |
| `results/run_log.txt` | archived log | reference |

## Methodology contributions that survive into Phase γ

- **L1 pairwise distance** on int8 ternary signatures
  (`pairwise_L1_int8`).
- **Shell-volume DP** under L1 with per-cell PMF (uniform,
  K-cache-marginal, scrambled variants).
- **Macocco fixed-radii MLE** with arbitrary cell PMF
  (`estimate_id_L1`).
- **Scrambled-ternary cell-graph** (`pairwise_scrambled_int8`,
  +1 as center) — control for P3 (centrality of 0).

## What's wrong with the Phase β verdict label

The FROZEN spec used `d̂/Dmax` as the normalization (Dmax = max
possible pairwise distance). L1 has Dmax=2D while Hamming has
Dmax=D, so substrate's d̂/Dmax is halved relative to Hamming
even before measuring any structure. The `d̂/D_ambient`
normalization (intrinsic dim ÷ ambient cell count) keeps units
consistent. Under `d̂/D_ambient`:
- P1 (L1 < Hamming): gap shrinks from +43pp to +5pp, CIs overlap.
- P2 (substrate < B4_pca): reverses — substrate is *higher*.
- P3 (centrality of 0): unchanged direction, larger gap.

Phase γ reports both normalizations explicitly and adds the
correlation-dim estimator as a cross-check. See γ for the honest
robustness picture.
