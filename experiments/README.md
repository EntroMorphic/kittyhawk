# experiments/ — substrate-claim measurement arc

Phase α/β/γ test the project's vision claim 3 ("base-3 carries
information base-2 collapses") on real BitNet K-cache data. Each
phase corrects a methodology flaw from the previous one.

**Read these journals in order to follow the arc:**

1. `journal/td27_geometric_prereg_v2_2026-05-12.md` — Phase α FROZEN pre-reg.
2. `journal/td27_phase_alpha_calibration_fail_2026-05-12.md` — v1 estimator failed calibration; halted per pre-reg.
3. `journal/td27_phase_alpha_synthesis_2026-05-12.md` — Phase α: VALIDATED 2/3 under categorical Hamming.
4. `journal/td27_phase_alpha_redteam_2026-05-12.md` — Red-team: M3 degenerate, M2 fails both criteria, M1 partly unit-of-measure.
5. `journal/td27_phase_alpha_remediation_2026-05-12.md` — Stricter rules: M1 REVERSES, MIXED 1/3.
6. `journal/td28_phase_alpha_methodology_pivot_2026-05-12.md` — Categorical Hamming was the wrong test; L1 metric named.
7. `journal/td28_phase_beta_prereg_2026-05-12.md` — Phase β FROZEN pre-reg with P1/P2/P3 on L1.
8. `journal/td28_phase_beta_synthesis_2026-05-12.md` — Phase β: VALIDATED 3/3 under L1 + d̂/Dmax normalization.
9. `journal/td28_phase_beta_redteam_2026-05-12.md` — Red-team: VALIDATED collapses to MIXED 1/3 under d̂/D_amb.
10. **`journal/td28_phase_gamma_robustness_2026-05-12.md` — current** — full remediation, robustness matrix.

## **READ FIRST — what the arc actually measured**

The arc compared Python `pairwise_hamming_int8` (categorical Hamming,
0/1 per cell) vs Python `pairwise_L1_int8` (L1, 0/1/2 per cell) on
substrate signatures. **Neither corresponds to a switchable production
choice.** The production substrate distance kernel
(`m4t/src/m4t_trit_pack.c::m4t_popcount_dist`) computes XOR popcount
on packed 2-bit trit codes, which equals **L1 path-graph distance**.

So "L1 vs Hamming" across these phases is actually:
- **Hamming-substrate (the Python baseline)**: never in production. A
  strawman that captures what substrate eviction would look like
  with 1-bit-per-cell sign-only packing.
- **L1-substrate (the Python target)**: equivalent to what
  production's `m4t_popcount_dist` already computes.

The headline +37-62% L2-error reduction from Phase ε re-casts as:
**the substrate's 2-bit code design choice is load-bearing for
attention-output quality, and the safety margin over a hypothetical
1-bit-sign-only encoding is 38-62% L2 error.** No production switch
is implied or needed.

See `journal/td28_l1_already_in_production_2026-05-12.md` for the full
misalignment write-up.

## Final status (after Phase γ)

The substrate-distinctive claim is not a single PASS/FAIL — it's a
**robustness matrix** across six methodologies
({Macocco, correlation-dim} × {abs, /Dmax, /D_amb}):

| Claim | Robustness |
|---|---|
| Centrality of 0 in cell-graph (P3a/b) | **ROBUST** (6/6) |
| Close-regime substrate compression (γ-D new) | **LARGE** (47pp gap to PCA-binary) |
| L1 metric reveals structure (P1) | PARTIAL (4/6) |
| Substrate beats structured binary at equal capacity (P2) | PARTIAL (5/6) |

**Critical methodology caveat (γ-G):** the Macocco fixed-radii
estimator is ~45% biased low on correlated synthetic. Real K-cache
has correlations. **Absolute d̂ values across the arc are
conservative (true intrinsic dim is roughly 2× reported).**
Relative comparisons (P-rules) remain valid because all
representations are biased similarly.

## Directory map

| dir | purpose | key files | status |
|---|---|---|---|
| `phase_alpha/` | Categorical-Hamming Macocco estimator + K-cache run | `m1_estimator_v2.py`, `run_phase_alpha_v2.py` | superseded for vision-claim tests by Phase β/γ; valid for Hamming-substrate measurements |
| `phase_beta/` | L1 (cell-graph) Macocco estimator | `m1_l1_estimator.py`, `run_phase_beta.py` | superseded for verdict-labeling by Phase γ; estimator + machinery still load-bearing |
| `phase_gamma/` | Robustness matrix; multi-normalization; null controls | `run_phase_gamma.py`, `correlation_dim.py` | **current** |

Each phase directory has its own `README.md` with detailed
file-level status.

## Reproducing

1. **Calibration:** every phase has a `calibrate*.py` script that
   exits 0 on pass, 1 on fail. Phase γ adds correlated-synthetic
   calibration via `run_phase_gamma.py`'s γ-G section.
2. **Real-K runs:** require ACTV2 dumps at `data/c_dump/` and
   `data/c_dump_v2/`. To regenerate v2 dumps:
   `bash experiments/phase_alpha/regenerate_dumps_v2.sh`
3. **Phase γ end-to-end:**
   `python experiments/phase_gamma/run_phase_gamma.py`
   Writes results to `experiments/phase_gamma/results/`.
