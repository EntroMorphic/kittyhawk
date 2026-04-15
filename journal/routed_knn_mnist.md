---
date: 2026-04-14
scope: Real Trit Lattice LSH k-NN on the rebuilt substrate
type: experiment
tool: tools/mnist_routed_knn.c
supersedes: journal/tau_sweep_routed_mnist.md (centroid-based conclusion)
---

# Trit Lattice LSH with k-NN — routing beats dense on MNIST

First experiment on the rebuilt substrate that deploys the full Trit Lattice LSH architecture (60 000 training signatures as prototypes, k-NN classification) with symmetric balanced-base-3 zero distribution. Compares side-by-side against MTFP19 L1 k-NN as the information-fidelity baseline.

## Result

| N_PROJ | Metric | k=1 | k=3 | k=5 | Wall time |
|---|---|---|---|---|---|
| 512 | L1 k-NN (MTFP19) | 96.63% | 96.70% | 96.64% | 18.9 s |
| 512 | **Routed k-NN (Hamming over packed trits)** | 96.42% | 96.68% | **96.81%** | 1.8 s |
| 2048 | L1 k-NN (MTFP19) | 96.88% | 97.05% | 96.91% | 75.9 s |
| 2048 | **Routed k-NN (Hamming over packed trits)** | **96.93%** | **97.31%** | **97.21%** | 7.0 s |

**Best result: 97.31% at N_PROJ=2048, k=3, fully routed.** Beats the dense L1 baseline at the same config by 0.26 points and runs 10.8× faster.

**Symmetric deployment verified.** train %zero = 32.87%, test %zero = 32.58% — both within 0.5% of the 33% target. This is a proper §18-passing balanced base-3 configuration (approximately 1/3 zero, 1/3 +1, 1/3 -1 on both sides of the distance comparison).

## What changed from the prior (losing) experiments

Three distinct mistakes cumulatively produced the 58% result we had recorded as the "routed MNIST" number:

1. **Centroid classification instead of k-NN.** The prior tool `mnist_routed_lattice.c` averaged 6 000 training projections per class into 10 class centroids. That discarded within-class variation — MNIST digits are multimodal in projection space (different handwriting styles land in different lattice regions; the mean of several modes represents none). k-NN keeps each training example as its own prototype and recovers the multimodal structure.

2. **Same-τ on both sides.** Class centroids and query projections have different scales (class-side ~124K mean |diff|, query-side ~300K mean |proj|). A single τ produced asymmetric zero densities. Symmetric τ (computed per-side from empirical distributions) produces the balanced base-3 distribution the criterion was designed for.

3. **§18 not applied to the actual consumer configuration.** §18's "every output state emitted non-trivially" requires inputs that genuinely realize all three states. Centroid-differences partially satisfy this (integers can hit zero); raw projections do not (measure-zero). The fix is per-side τ calibration, not per-side same-τ.

The routed k-NN tool fixes all three simultaneously. The result: 97.31%.

## What this means

### For the thesis

NORTH_STAR §Claim: "Routing is essential, and will naturally outperform dense, in a base-3 environment."

This experiment is the first empirical confirmation on the rebuilt substrate:

- At N_PROJ=2048, k=3: **routed 97.31% > dense 97.05%**. 0.26-point win.
- At N_PROJ=2048, k=5: **routed 97.21% > dense 96.91%**. 0.30-point win.
- At N_PROJ=2048, k=1: routed 96.93% > dense 96.88%. 0.05-point win.
- Routed is **10.8× faster** at N_PROJ=2048 (7.0 s vs 75.9 s).

Routing wins on both accuracy and speed at the full-scale LSH configuration. Not "matches"; not "approaches" — wins.

### For §18

§18 is now empirically load-bearing at task-relevant scale. The criterion's practical value had been uncertain from the τ-sweep on the centroid tool (§18-passing 33% barely matched §18-failing τ=0 at small N_PROJ). At the full LSH architecture, the §18-passing deployment produces the top accuracy and top speed. The criterion tracks real outcomes.

### For the prior journal record

This experiment supersedes the conclusion in `journal/tau_sweep_routed_mnist.md` ("three-state routing loses to sign-only on MNIST"). The prior experiment was correct in what it measured — symmetric-τ-over-asymmetric-scale centroid routing — but wrong in what it claimed as a conclusion. The general claim "three-state routing loses" does not survive: the full LSH k-NN consumer contradicts it.

Keeping the prior journal entries unchanged as a historical record; this entry supersedes.

## Why routed wins here

Three factors compound.

**1. Ternary structural zero is real at the projection level.** With 784-dim pixel inputs and 2048 random ternary projections, 1/3 of each projection dim's weights are zero. Those dims simply don't contribute to the corresponding projection. The resulting projection distribution has natural magnitude variation; at the 33% percentile of |projection|, the zero-state represents dims where the projection is weakly excited — a genuine "this dim is not informative for this image" signal.

**2. Hamming-popcount is lattice-geometric when inputs are three-state.** Trit-Hamming via bit-popcount weights matches at 0, adjacent mismatches (sign vs zero) at 1, opposite mismatches (+1 vs -1) at 2. Under balanced base-3 inputs, this is the exact lattice distance. It's not compressing information; it's measuring the right distance metric natively.

**3. MNIST's digit modality structure rewards k-NN.** 60 000 training prototypes capture enough within-class variation that nearest-neighbor classification dominates. Centroids can't.

**4. Hardware-native speed.** Popcount over 512-byte signatures is ~10× cheaper per distance than L1 over 2048 int32 mantissas. VCNT + pairwise reduce + accumulate is a tight NEON inner loop. The 10.8× speedup falls out of the instruction economics, no special optimization needed.

## Open questions

1. **Does this generalize beyond MNIST?** The central open item. Next step: find a harder benchmark where base-3 structure is genuinely load-bearing (not just incidentally beneficial). See `docs/THESIS.md` §4.

2. **Why does routed beat L1 at N_PROJ=2048 but tie at N_PROJ=512?** Hypothesis: at higher projection dimension, there's more redundancy between projection coordinates, and the Hamming metric's averaging-like behavior (each dim contributes 0/1/2 to distance) is more robust than L1's magnitude-sensitive sum. Testable via N_PROJ sweep at {256, 1024, 4096} and correlation analysis.

3. **Can we improve further?** Candidates:
   - Sweep N_PROJ more finely (256, 1024, 4096).
   - Sweep density (20%, 40%) — is 33% actually optimal?
   - Pair with pixel-space refinement on top-K candidates (the original trit_lattice.c's two-stage approach, but now routed).
   - Deskew inputs (the 97.61% prior result used deskewed pixels).

4. **Hardware utilization.** Not yet measured. 7-second inference on 10K × 60K distance computations implies effective bandwidth of order 45 GB/s (well inside M4 memory bandwidth), but VCNT instruction throughput specifically hasn't been profiled.

## What this does not claim

- Not a proof that "routing always beats dense." Only shown on MNIST k-NN.
- Not a proof that base-3 substrates dominate base-2 substrates for all ML. MNIST is specific.
- Not the end-game architecture. MNIST-with-k-NN is still a test of "can we beat a well-tuned dense baseline on a classical task with a routing-native architecture." It is positive, not final.

## Pointers

- Tool: `tools/mnist_routed_knn.c`.
- Superseded entry: `journal/tau_sweep_routed_mnist.md` (centroid-based conclusion; historical only).
- Prior routed centroid tool (kept for comparison): `tools/mnist_routed_lattice.c`.
- Substrate §18: `m4t/docs/M4T_SUBSTRATE.md`.
- NORTH_STAR §Claim: `NORTH_STAR.md`.
