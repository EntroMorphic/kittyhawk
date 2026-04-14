---
date: 2026-04-13
scope: MNIST zero-float k-NN past the 97.61% pixel-L2 ceiling
---

# Where we are

Pixel-space k-NN on deskewed MNIST (MTFP19, scalar L2, k=3) sits at **97.61%**. Everything in this document is about trying to get past that number without reintroducing float, and what we've learned in the attempt.

---

## What was tried and what it taught

### 1. Multi-channel with naive L2 (Phase 5) — DEAD END
Concatenated pixel + horizontal gradient + vertical gradient + flood-fill topology into a 2353-dim vector, ran k-NN with equal-weight L2.

- Result: **96.86%** (−0.75 from baseline)
- Why it failed: gradient has 2× the dimensions of pixel, so unweighted L2 is dominated by gradient matching. Channels aren't commensurate.

### 2. Per-channel weighted L2 — STILL DEAD
Swept six weight combinations over (w_pix, w_grad, w_topo).

- Best: **97.55%** at (w_pix=8, w_grad=1, w_topo=0). All combos worse than 97.61%.
- Diagnostic: topo feature *doubles* the 4↔9 confusion (16 → 31). Flood-fill hole count flips between 0 and 1 on thin-stroke 4s and 9s, creating cliffs in feature space that L2 reads as definite class difference.
- Lesson: gradient info is already implicit in pixel L2 after deskewing; topology via flood-fill is too brittle on noisy handwriting.

### 3. Signed distance field as a **complementary** classifier — REAL SIGNAL
Not a channel — a separate k-NN in SDF feature space, combined at the decision layer. Integer multi-threshold Chamfer 3-4 distance transform at three binarization thresholds ({¼, ½, ¾} × SCALE), summed to smooth the topology cliff.

- Standalone: **96.59%** (k=3 L2) — weaker than pixel
- **But errs differently.** Overlap analysis (scalar MTFP):
  - both right: 9566
  - only pixel right: 195
  - only SDF right: **93**  ← decorrelated gain
  - both wrong (agree): 108
  - both wrong (disagree): 38
  - **Oracle ceiling: 98.54%**
- Interpretation: pixel sees interior ink density; SDF sees boundary geometry. Genuinely complementary.

### 4. Naive weighted-distance fusion of pixel + SDF — INCREMENTAL WIN
`dist(ref) = w_p · d_pix(q, r) + w_s · d_sdf(q, r)`; swept weights, top-K by combined distance.

| Config | Accuracy |
|---|---|
| pixel-only | 97.61% |
| pix=1 sdf=4 (best of this sweep) | **97.76%** (+0.15) |

Every combination beat pixel-only. Monotonic trend: the more weight on SDF, the better. Naive fusion extracts ~16% of the 98.54% oracle gain. The other 84% requires a smarter adjudicator (margin gating, pair-triggered SDF).

### 5. Kernel rewire via ternarization — **MISSTEP**
To run faster we ternarized the references ({-1, 0, +1}) and used `m4t_mtfp_ternary_matmul_bt` for dot products, reconstructing distance via `‖r‖² − 2·q·r`.

- First attempt: pixel = 68.65%. Bug: queries were raw MTFP (0..SCALE) vs trit refs (±1), the L2 identity needs q and r in the same space.
- Centered-query fix: pixel = 97.10%, SDF = 96.95%, combined = 97.14%. Oracle ceiling collapsed from **98.54% → 97.58%**.
- **Why:** ternarizing observations throws away 18 of 19 trits per cell. Both pixel-sign and SDF-sign collapse onto the same boundary geometry; the decorrelation that gave us the oracle headroom came from MTFP *magnitude*. We destroyed it by quantizing the data.
- Deeper mistake: conflated "ternary routing substrate" with "ternary observations." The Trit Lattice LSH framework uses ternary *projections* applied to MTFP data, not ternarized data itself. `m4t_mtfp_ternary_matmul_bt` is for applying sparse routing matrices, not for data quantization.

### 6. Current: NEON MTFP L2 (both sides full width) — IN FLIGHT
Reverted ternarization. Rewrote the hot inner loop with hand NEON:
```c
int32x4_t d = vsubq_s32(q, r);
int64x2_t sq = vmull_s32(vget_low_s32(d), vget_low_s32(d));
```
Keeps all 19 trits of MTFP on both sides. Expected ~3-5× over pure scalar. Running now — will establish the cost of NEON-accelerated honesty.

---

## Empirical laws confirmed / added

1. **Distance beats direction** (established).
2. **The projection IS the intelligence** for centroids (established).
3. **The classifier is the ceiling**, not the representation (established).
4. **Fix the geometry first** — deskewing adds 0.71% (established).
5. **Incommensurate channels poison L2** (Phase 5 naive multi-channel).
6. **Complementary classifiers decorrelate only if they see different magnitudes, not just different sign patterns** (Phase 5-ternary result). Ternarizing observations collapses the complementarity.
7. **The substrate's ternary kernels are for projections, not for observations.** Quantizing data to use a kernel designed for weights is a category error.

---

## The 98.54% oracle ceiling (scalar MTFP)

This is the practical upper bound for any classifier of the form "pixel-k-NN OR SDF-k-NN, pick the right one":
- both wrong: 146
- test set: 10,000
- ceiling: 9854/10000 = **98.54%**

Naive distance fusion extracted ~14% of the 93 "only-SDF-right" samples (97.76%). A smarter adjudicator could reach much closer to 98.54%. Candidates:

1. **Margin gating.** Trust pixel by default; defer to SDF only when pixel's top-2 margin is small AND SDF's is large. Computable per-query.
2. **Pair-triggered SDF.** When pixel k-NN returns {4, 9} or {3, 5, 8}, re-vote with SDF. Targets the structural confusions.
3. **Per-pair binary classifiers.** A small ternary projection trained to separate only confused pairs, applied only on disagreement.

---

## Performance accounting

- Scalar MTFP L2 (pre-NEON): ~1 GOP/s effective. 10k × 60k × 784 = 470 GMAC per config → ~470 s per config. 6 configs ≈ 47 min. Matches observed.
- Hand NEON int32×int32 → int64 squared-L2: 2 int64 squares per vector, expected ~3-5× over scalar = ~10-15 min for 8 configs.
- True fast path would be SDOT (`vdotq_s32`): 16 int8×int8 → int32 MACs per cycle, ~8× over int32 NEON. Requires MTFP4 quantization of queries — a separate substrate path, not yet built.
- M4T does **not** currently have a dense MTFP-vs-MTFP squared-L2 kernel. The existing `m4t_mtfp_ternary_matmul_bt` is for applying ternary weights to MTFP activations (routing), which is structurally different. A proper dense L2 primitive is a legitimate M4T contract addition worth doing if we stay on this problem.

---

## What's still unknown

1. Can tangent distance (invariant to small rotation/scale/thickness) close the gap between 97.76% and the 98.54% oracle — or push past it? Classical route to 98.5%+ on k-NN MNIST.
2. Will margin-gating or pair-triggered SDF (items 1-3 above) recover more than the 16% of oracle gain naive fusion gets?
3. Are there features whose *sign patterns* differ more than pixel-sign vs SDF-sign? Orientation maps, multi-scale SDF, per-pixel skeleton distance. These are candidates for a true trit-lattice decorrelation that survives quantization.
4. Is an SDOT-accelerated MTFP4 path worth building to make exploration fast enough to matter? The research question is empirical: how many tries past 97.61% do we still want?

---

## Next decision points

- If NEON MTFP run reproduces 97.61% / 97.76% at ~3-5× speed: commit this as the tooling baseline, then pick one of {tangent distance, margin gating, per-pair classifier} to attack the oracle ceiling.
- If it falls short of scalar (shouldn't, but NEON arithmetic can surprise): debug before proceeding.
- If fast exploration becomes the bottleneck (more weight sweeps, more feature variants needed), build the MTFP4 SDOT path as a proper M4T primitive.
