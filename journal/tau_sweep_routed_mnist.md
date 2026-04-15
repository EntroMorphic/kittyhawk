---
date: 2026-04-14
scope: τ sweep of fully-routed MNIST classifier; first empirical discharge of §18
type: experiment
tool: tools/mnist_routed_lattice.c
---

# τ sweep on the fully-routed MNIST classifier

First experiment that empirically distinguishes a §18-failing deployment from §18-passing deployments on the same routing primitive (`m4t_route_threshold_extract`). The rest of the pipeline — projection, centroid build, distance, top-k — is unchanged.

## Result

```
N_PROJ = 256
  L1-over-mantissa:           80.12%
  Routed tau=0       [FAIL §18]:  57.47%
  Routed tau=10000   [pass §18]:  57.10%
  Routed tau=50000   [pass §18]:  55.32%
  Routed tau=200000  [pass §18]:  50.99%
  Routed tau=1000000 [pass §18]:   9.80%

N_PROJ = 512
  L1-over-mantissa:           80.46%
  Routed tau=0       [FAIL §18]:  58.38%
  Routed tau=10000   [pass §18]:  56.97%
  Routed tau=50000   [pass §18]:  56.29%
  Routed tau=200000  [pass §18]:  53.19%
  Routed tau=1000000 [pass §18]:   9.80%

N_PROJ = 1024
  L1-over-mantissa:           81.14%
  Routed tau=0       [FAIL §18]:  57.54%
  Routed tau=10000   [pass §18]:  56.25%
  Routed tau=50000   [pass §18]:  54.31%
  Routed tau=200000  [pass §18]:  54.49%
  Routed tau=1000000 [pass §18]:   9.80%

N_PROJ = 2048
  L1-over-mantissa:           81.40%
  Routed tau=0       [FAIL §18]:  58.37%
  Routed tau=10000   [pass §18]:  57.45%
  Routed tau=50000   [pass §18]:  56.68%
  Routed tau=200000  [pass §18]:  57.30%
  Routed tau=1000000 [pass §18]:   9.80%

Wall clock per N_PROJ (L1 + 5 tau values, 10K test images): 393–3332 ms.
```

## What this tells us

**Three-state routing loses to sign-only routing on MNIST.** At every N_PROJ, the §18-passing deployments (τ > 0) produce *worse* accuracy than the §18-failing deployment (τ = 0). The gap is small for small τ and grows with τ until the band swallows the projection magnitudes entirely (τ = 1M → 9.80% chance-level).

**§18 is a utilization criterion, not a quality criterion.** A primitive can be properly base-3-deployed (its three-way semantic actually exercised on realistic inputs) AND produce worse downstream accuracy than a base-2-degenerate deployment, if the task's discriminative signal lives in a representation the three-way primitive throws away. §18 says "your three-way semantic is being utilized." It does not say "this primitive will produce good accuracy on this task."

**MNIST classification's signal is in projection magnitudes.** The L1-over-mantissa baseline (which uses full magnitude) achieves 81%. Sign-only routing (τ=0, no band) keeps the dominant magnitude→sign correlation and hits 58%. As τ widens, more projection magnitudes get suppressed to zero, and the sign-only signal gets corrupted by zero-trits where the projection was just below threshold. Information loss exceeds information gain. By τ=1M, all projection magnitudes fall inside the band, all signatures collapse to all-zeros, and classification is random.

**The compass held one more time.** NORTH_STAR §4: "Running routing-native on [MNIST] is a test of adapter efficiency, not the thesis." The τ sweep is the most explicit test of this we've run. Even the §18-correct routing deployment loses to dense by ~25 points. MNIST is not a base-3-native task; its decision shape (nearest-centroid by L1) doesn't reward three-state representations.

## Submonotonic blip at τ=200000

For τ=200000 alone, accuracy is non-monotonic in N_PROJ:
- N_PROJ=256 → 50.99%
- N_PROJ=512 → 53.19%
- N_PROJ=1024 → 54.49%
- N_PROJ=2048 → 57.30%

Increasing projections at fixed τ recovers accuracy. This is consistent: more projections give more dims to discriminate over, so the band-induced information loss is partially compensated by parallel signature dims. None of the tried combinations beat τ=0, but the sweep at τ=200000 + N_PROJ=2048 (57.30%) approaches τ=0 (58.37%).

If we extrapolated: there might exist an (N_PROJ, τ) configuration where three-state routing matches τ=0 on MNIST. There is no plausible configuration where it *beats* L1 — the magnitude information loss is structural.

## What this NOT-RESULT does for the project

This is a clean negative empirical discharge. Negative on the experimental hypothesis (three-state routing on MNIST), positive on the methodological commitment:

1. The §18 criterion is now empirically distinguishable from "make accuracy go up." We have an example of §18-passing producing *worse* accuracy. This protects the criterion from being reinterpreted as a performance prediction.

2. The thesis "routing outperforms dense in a base-3 environment" is not contradicted — MNIST is not a base-3 environment for nearest-centroid classification. It is a base-2 environment for that decision shape, and our routing primitives lose accordingly.

3. The candidate experiment named in `journal/fully_routed_mnist.md` ("Want me to do it?" — yes, done) is closed with a documented outcome. Time to move to a different bed.

## What this experiment doesn't address

- **Whether routing wins on a different task.** Open. Requires a non-MNIST bed where sign/band structure is load-bearing.
- **Whether a smarter consumer pipeline (e.g., MTFP4-native signatures, multi-projection-bank routing) closes the gap on MNIST.** Possibly, but the question is what we'd be measuring — adapter efficiency on a task that doesn't reward routing.
- **Hardware utilization of the routed path.** Still unmeasured. The `~3.3s per N_PROJ=2048 sweep` is not isolated to the routed step.

## Pointers

- Tool: `tools/mnist_routed_lattice.c` (now sweeps τ ∈ {0, 10K, 50K, 200K, 1M}).
- §18 contract: `m4t/docs/M4T_SUBSTRATE.md` §18.
- Prior measurement: `journal/fully_routed_mnist.md` (the τ=0-only run).
- NORTH_STAR §4 — the prediction this confirms.
