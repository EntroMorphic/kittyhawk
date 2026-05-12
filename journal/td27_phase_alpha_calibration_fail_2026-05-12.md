# Phase α calibration FAILED — implementation halted per v2 pre-reg

Per the FROZEN calibration protocol in
`td27_geometric_prereg_v2_2026-05-12.md`:
> If calibration fails (any synthetic d exceeds 20% relative error):
> the estimator implementation has a bug, OR the ternary-Hamming
> TwoNN formulation is wrong. Phase α implementation **stops**
> until resolved.

The stop-criterion fired on first attempt. **Implementation halted.**
This document records the failure honestly and the path forward.

## Calibration results

Implementation: `experiments/phase_alpha/m1_estimator.py` and
`experiments/phase_alpha/calibrate.py`.

Setup: synthetic ternary data of known intrinsic dimension d_true,
N=500 (subsampled from 5000), ambient D=128. M1 estimator applied
under the v2-derived ternary-categorical-Hamming likelihood.

Results:

| d_true | d_hat | rel err | verdict |
|--------|-------|---------|---------|
| 2 | nan | n/a | DEGENERATE (intrinsic-dim too small; only 9 configurations) |
| 5 | nan | n/a | DEGENERATE (243 configurations; many r1=r2 ties) |
| 10 | 10.00 | 0.00% | PASS |
| 20 | 14.00 | -30% | FAIL |
| 50 | 31.00 | -38% | FAIL |
| 100 | 62.00 | -38% | FAIL |

**3 of 6 target d values exceed 20% relative error.** Systematic
underestimation (~60% of true) for d ≥ 20.

## Diagnosis

The implementation used the pairwise likelihood:
```
P(r1, r2 | d) ∝ V(r1, d) · V(r2, d)
```
where `V(r, d) = C(d, r) · 2^r` is the ternary categorical Hamming
shell volume.

This is **too simple**. The Macocco I3D framework for binary
Hamming uses a more careful conditional likelihood that accounts for:
1. The ordering r2 > r1 (here treated as independent, which is wrong).
2. The fact that r1 is the MINIMUM over N-1 candidate distances,
   not a random draw — extreme-value statistics matter.
3. The local density correction at the boundary of the manifold.

My substitution of `C(d, r) → C(d, r) · 2^r` for shell volume is
mathematically correct as a substitution, but the likelihood
function around it needs more care.

## What this teaches us

The discipline antibody worked. The v2 pre-reg said "stop if
calibration fails," and stop is what happens.

In v1 (the original pre-research that the red-team caught), I claimed
the ternary extension of Macocco was "straightforward." It is not.
The calibration failure makes that concrete: the simple substitution
of shell volumes does not preserve the estimator's accuracy.

This is the 9th caught overclaim of the session, but caught by a
PRE-REGISTERED CRITERION rather than by user red-team. The
calibration step earned its place in the v2 pre-reg.

## Path forward (options)

### Option A: Fix the estimator

Re-derive the ternary categorical Hamming I3D likelihood from
Macocco's framework with correct extreme-value treatment. Specifically:

1. Read Macocco's PRL paper carefully. Reproduce the binary case as
   a sanity check (synthetic binary data → recover known d).
2. Generalize the conditional likelihood to ternary alphabet,
   preserving the extreme-value structure (not just substituting
   shell volumes).
3. Re-run calibration. If it passes, proceed to Phase α
   implementation.

**Estimate: 3-5 days of focused mathematical work + implementation.**

### Option B: Drop M1, proceed with M2 + M3 only

M2 (k-NN topology) and M3 (Betti-0 persistence) don't require an
intrinsic-dimension estimator. They compute graph-theoretic and
topological properties that are well-defined under any discrete
metric.

The substrate-claim verdict rules become: ≥ 1 of 2 measures (instead
of ≥ 2 of 3) must clear the substrate-vs-B2 threshold. Weaker
evidence, but tractable today.

**Estimate: ~1 week. The downside: M1 was the load-bearing
measurement because it's the most directly substrate-distinctive
(intrinsic dimensionality is what the vision claim implicitly
references).**

### Option C: Defer Phase α entirely

Phase α requires methodology that doesn't currently exist in
published literature for ternary categorical Hamming. Honest
deferral until either (i) someone publishes the ternary extension
or (ii) we commit research-grade effort to derive it.

**Estimate: indefinite.**

## My recommendation

**Option A**, properly scoped. The substrate's vision claim is
load-bearing for the project; the M1 measurement is load-bearing for
testing it; the estimator's correctness is load-bearing for the
measurement. We should fix this rather than work around it.

But honestly: the math here is the kind of thing that benefits from
a focused work-unit with proper math infrastructure (Mathematica or
sympy for symbolic derivation), not vibes-coded in Python at 10pm.

The right next step is to **commit this calibration failure
honestly, push, and not pretend we're ready to implement Phase α**.

## What stays valid from the v2 pre-reg

All other FROZEN sections still apply:
- Equal-bits B2 baseline (D=128 trits vs D=203 sign bits)
- Citations (Macocco-Glielmo-Grilli-Laio; Ruppik et al.; Loki; HashEvict)
- Operational close/far prototype regime for K-cache
- M2 and M3 measures and thresholds
- Bootstrap CIs and verdict rules
- Honest N (3600 from ACTV2 dumps)

Only M1's specific implementation is broken; the rest of the
infrastructure is sound.

## Code committed

- `experiments/phase_alpha/m1_estimator.py` — Python implementation
  of ternary categorical Hamming distance + estimator (currently
  biased).
- `experiments/phase_alpha/calibrate.py` — synthetic-data calibration
  test (exit code 1 on failure).

Both are committed as the demonstrably-not-working version, with
this journal documenting the failure. The frozen pre-reg
(td27_geometric_prereg_v2_2026-05-12.md) remains the spec.

## Sign-off

The discipline worked. Pre-registered calibration caught the
estimator bias before any K-cache measurement contaminated the
record. The substrate's quality-side wins (Phase A, #9, #10) stand
unchanged.

Phase α is paused pending estimator fix (Option A) or scope reduction
(Option B). The pre-reg's stop-criterion was correctly triggered.
