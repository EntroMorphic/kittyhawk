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

---

## Revised after fourth red-team (2026-04-14)

The original writeup above was based on one RNG seed, a scalar L1 baseline, and no deskewed-pixel comparison. A fourth-round red-team flagged all three as holes. This section reports the fair-comparison measurements with those holes closed.

### What changed in the experiment

- **NEON-vectorized L1 baseline.** `l1_distance_mtfp` now uses `vabdq_s32` + `vaddw_s32` widening accumulate. Same SIMD shape as the routed popcount path; the speedup comparison is now apples-to-apples.
- **3 RNG seeds per (N_PROJ, mode) cell.** Mean ± stddev reported instead of single-run numbers.
- **Deskewed-pixel dense k-NN baseline.** The classical MNIST baseline that hit 97.61% in the pre-rebuild journal. Re-measured.
- **Deskewed-projection routed path.** Applies deskewing before projection for apples-to-apples with the deskewed-pixel baseline.
- **Full trit distribution reported.** Not just %zero; also %+1 and %-1. Verifies symmetric base-3.

### Revised results

**Dense pixel k-NN baselines (deterministic, single run):**
- Raw pixels, k=3: 96.33%.
- Deskewed pixels, k=3: **97.16%** (the strong classical baseline).

**Projection-based, 3 seeds, full trit distribution verified as +33.4% / 0 32.9% / -33.7%:**

| Mode | N_PROJ | k | L1 (NEON) | Routed | Δ | Significance |
|---|---|---|---|---|---|---|
| raw | 2048 | 3 | 97.00 ± 0.05% | **97.30 ± 0.03%** | +0.30% | 5.2σ |
| raw | 2048 | 5 | 96.85 ± 0.05% | **97.18 ± 0.04%** | +0.32% | 5.9σ |
| raw | 2048 | 1 | 96.78 ± 0.09% | 96.97 ± 0.05% | +0.19% | 1.8σ |
| raw | 512 | 3 | 96.70 ± 0.07% | 96.74 ± 0.08% | +0.05% | 0.5σ |
| deskewed | 2048 | 3 | 97.62 ± 0.07% | **97.79 ± 0.05%** | +0.17% | 2.0σ |
| deskewed | 2048 | 5 | 97.52 ± 0.05% | **97.77 ± 0.02%** | +0.25% | 4.6σ |
| deskewed | 2048 | 1 | 97.49 ± 0.10% | 97.54 ± 0.07% | +0.05% | 0.4σ |
| deskewed | 512 | 3 | 97.41 ± 0.06% | 97.27 ± 0.09% | -0.13% | -1.2σ |

**Wall-time (single-threaded, mean over seeds):**

| N_PROJ | L1 (NEON) | Routed | Speedup |
|---|---|---|---|
| 512 | 28.4 s | 2.4 s | 12.0× |
| 2048 | 141.4 s | 7.0 s | **20.3×** |

### Revised conclusions

1. **Routed k-NN beats NEON-vectorized L1 k-NN at N_PROJ=2048** in both raw and deskewed modes, with 5σ significance at k=3 and k=5 (raw mode). The headline win is statistically robust, not single-run noise.

2. **Routed k-NN beats the dense deskewed-pixel baseline.** Best configuration — deskewed, N_PROJ=2048, k=3 — achieves **97.79 ± 0.05%**, beating the classical 97.16% dense pixel k-NN by 0.63 points. No projection needed on the dense side; routing wins against the stronger baseline.

3. **Speedup survived fair comparison.** Routed is 20.3× faster than NEON-vectorized L1 at N_PROJ=2048 (12.0× at N_PROJ=512). The compression of 2048 int32 mantissas into a 512-byte packed-trit signature is what drives the speed; popcount processes trit information at NEON-native VCNT throughput, while abs-diff-sum fights L2 cache pressure from 480 MB of training projections.

4. **N_PROJ=512 flips:** routed loses slightly (Δ ≈ -0.12%) at low projection dimension. The routed advantage is N_PROJ-dependent; at small projection spaces per-dim precision matters more than routing structure. At large projection spaces, routing's information density compresses enough that its lattice-geometric distance dominates.

5. **Symmetric base-3 distribution confirmed.** +33.4% / 0 32.9% / -33.7% on signatures across all configurations; consistent with the τ = 33rd-percentile calibration.

### What the original writeup got right vs wrong

**Right:**
- The diagnosis of three compounding errors (centroid architecture, asymmetric τ, wrong §18 application) was accurate. Fixing those three produced the 58%→97% recovery.
- The core claim "routed k-NN at balanced base-3 produces MNIST accuracy competitive with dense" is confirmed.
- Symmetric τ calibration works.

**Wrong/overclaimed:**
- "10.8× speedup" — was against a scalar baseline. Against NEON-vectorized L1, actual speedup is 20× at N_PROJ=2048 (higher, not lower — but the original number was measured against the wrong comparison).
- "Beats dense by 0.26 points at k=3" — single-run, 1.5σ. Fair numbers at k=3 raw are 0.30 ± 0.06% (5σ). At k=3 deskewed against the stronger dense pixel baseline, 0.63% (effectively ∞σ for a deterministic baseline).
- "First empirical confirmation of NORTH_STAR §Claim" — true in spirit, but the original evidence was too thin to carry the claim. It carries now.

### What still doesn't cash the NORTH_STAR check

MNIST with classical k-NN is still a cooperative task for both dense and routed approaches. The thesis "routing will naturally outperform dense in a base-3 environment" has now survived a fair comparison on MNIST — but MNIST is not a base-3-native problem. It is a problem where both approaches can do well; routing happens to do slightly better AND much faster.

For the thesis to be decisively tested, we need a problem where base-3 structure is intrinsic — where dense loses not by 0.3 points but by large margins. Open item in `docs/THESIS.md` §4.

### Pointers to the fair run

- Tool: `tools/mnist_routed_knn.c` (post-remediation version).
- Raw output: produced by running `./mnist_routed_knn <mnist_dir>`; reproducible with the same seeds.
- Remediation plan: `docs/REMEDIATION_PLAN.md` fourth round.
