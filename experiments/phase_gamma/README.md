# phase_gamma/ — robustness matrix, null controls, current verdict

**Status: current.** This is the honest reading of the substrate-
distinctive claim after the full red-team and remediation arc.

Phase γ replaces single VALIDATED/FALSIFIED verdicts with a
**robustness matrix** — each claim is evaluated under all
defensible methodologies, and findings are classified ROBUST
(passes all), PARTIAL (passes most), or NOT ROBUST (passes one
specific choice).

## Key files

| file | what it is |
|---|---|
| `run_phase_gamma.py` | full remediation pipeline. Adds B5m mirror, correlation-dim estimator, shuffled-K null control, correlated-synthetic calibration, close-regime under L1, τ sweep, multi-normalization robustness grid. |
| `correlation_dim.py` | Grassberger-Procaccia correlation-dimension estimator. Independent second method for cross-validation. Known to have edge-effect bias at higher d, used for direction agreement not absolute. |
| `results/phase_gamma_results.json` | full numeric grid |
| `results/run_log.txt` | archived log |

## Headline findings (after Phase γ)

| Claim | Test | Status |
|---|---|---|
| **Centrality of 0** (P3a, P3b) | substrate L1 < scrambled (+1-center AND −1-center) | **ROBUST** (6/6 methodologies) |
| **Close-regime compression** (γ-D new) | within (layer, kv_head, site) pooled | **LARGE** — substrate 0.259 vs B4_pca 0.724 d̂/D_amb (47pp gap) |
| **L1 reveals structure Hamming hid** (P1) | substrate L1 < Hamming-substrate | PARTIAL (4/6) |
| **Substrate beats structured binary** (P2) | substrate L1 < PCA-binary at equal capacity | PARTIAL (5/6) |

## Critical methodology caveats

1. **Macocco estimator calibration FAILS on correlated synthetic
   (~45% rel err).** Both Macocco and corrdim are biased low on
   data with non-trivial cell correlations. Real K-cache has
   correlations. **Absolute d̂ values across the whole arc should
   be treated as conservative; true intrinsic dim is roughly 2×
   reported.** Relative comparisons (P-rules) remain valid because
   all representations are biased similarly.

2. **Shuffled-K null control (γ-F)** distinguishes
   "marginal-statistics" effects from "learned-structure" effects:
   - P2 collapses on shuffled K → substrate's advantage over B4
     REQUIRES learned structure.
   - P3 persists on shuffled K → centrality-of-0 is a metric-on-
     marginals property (38% zeros vs 31% each of ±1), NOT
     learned-semantic-silence.

3. **The verdict-flipping pattern** through Phase α/β finally
   stops with Phase γ — by reporting the robustness matrix
   honestly instead of picking a single methodology and declaring
   victory.

## Honest restatement of the vision claim

Substrate signatures under L1 distance capture **close-range**
geometric similarity that binary baselines lose. The cell-graph
metric with the more-common cell value at center is load-bearing.
Effect is SMALL in pooled measurements (mixture-averaging across
heterogeneous regimes), LARGE in close-regime measurements (where
local manifold structure is preserved).

For downstream operations that exploit local similarity (KV-eviction,
soft routing, retrieval), substrate L1 has measurable advantages
over binary at equal capacity. **The qualified vision claim is
load-bearing; the strong vision claim is not robust across
methodologies.**

## What to do next

Don't run another pooled-d̂ comparison. Build a downstream-
application benchmark that exploits the close-regime advantage and
measure on the application's quality metric (e.g., "does
L1-substrate KV-cache eviction beat Hamming-substrate eviction by
X% on inference coherence on a held-out prompt set?"). The
close-regime structure is where substrate's advantage actually
lives; demonstrate it on the operation that benefits.
