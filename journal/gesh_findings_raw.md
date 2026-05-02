---
cycle: gesh_findings
phase: RAW
date: 2026-05-02
scope: dump observations from Phase A.2's multi-seed sig_dim sweep through sig_dim=1024 without interpretation
companions: gesh/docs/sweep_dims_results.md · gesh/bench/sweep_dims.c · journal/gesh_phase_a2_redteam.md
status: capture
---

# RAW — gesh_findings

Observations only. No interpretation. No mechanism claims. No "this means."

## The data

Multi-seed sweep, 5 seeds per (sig_dim, variant), synthetic prototype-classification benchmark (D=64 with K=16 informative + 48 noise; C=10 classes; per-trit noise p=0.10; n_train=2000; n_test=500; top_k=1):

| sig_dim | random          | trained         | gain    |
|---------|------------------|------------------|---------|
|       2 |  15.6% ± 3.1pp |  21.0% ± 2.4pp |  +5.4 pp |
|       4 |  21.2% ± 1.6pp |  26.8% ± 2.3pp |  +5.6 pp |
|       8 |  31.8% ± 3.1pp |  36.2% ± 0.8pp |  +4.4 pp |
|      16 |  43.4% ± 3.8pp |  51.4% ± 4.6pp |  +8.0 pp |
|      32 |  59.0% ± 2.5pp |  67.2% ± 2.4pp |  +8.2 pp |
|      64 |  76.4% ± 2.1pp |  78.2% ± 2.3pp |  +1.8 pp |
|     128 |  90.0% ± 1.7pp |  89.2% ± 1.5pp |  −0.8 pp |
|     256 |  95.4% ± 0.9pp |  95.4% ± 0.5pp |  +0.0 pp |
|     384 |  96.8% ± 0.4pp |  96.8% ± 0.4pp |  +0.0 pp |
|     512 |  97.4% ± 0.5pp |  97.6% ± 0.5pp |  +0.2 pp |
|     768 |  98.2% ± 0.4pp |  98.2% ± 0.4pp |  +0.0 pp |
|    1024 |  98.6% ± 0.5pp |  98.6% ± 0.5pp |  +0.0 pp |

Identity at sig_dim = D = 64: **69%** (deterministic, single trial).

## Shape of the curves

- **Random R, accuracy vs sig_dim:** monotonic increasing across the entire range; concave; asymptotes to but does not reach 100%. Knee around sig_dim = 32–64.
- **Trained R, accuracy vs sig_dim:** same shape as random's, shifted up at sig_dim ≤ 32, indistinguishable from random's at sig_dim ≥ 64.
- **Gain (trained − random) vs sig_dim:** rises from +5pp at sig_dim=2 to a plateau of +8pp at sig_dim ∈ {16, 32}, drops back to +1.8pp at sig_dim=64, oscillates around 0pp from sig_dim=128 to 1024.
- **Stddev:** rises from ~3pp at very low sig_dim through ~4.6pp at sig_dim=16 (the highest-variance cell), then collapses monotonically to ~0.4pp at sig_dim ≥ 384.

## Identity baseline

- One number: 69%.
- Sits below random ternary projection at the same sig_dim (76.4% ± 2.1pp at sig_dim=64).
- Gap: −7.4pp ± 2.1pp.
- Identity is deterministic (no projection seed); random and trained have init+train seed.

## What a perfect classifier would do

The benchmark has a closed-form upper bound: a Bayes-optimal classifier on the K=16 informative dims with p=0.10 per-trit flip noise. With per-trit flip rate p, the prob a sample exactly matches its prototype on the K informative dims is `(1-p)^K = 0.9^16 ≈ 18.5%`. But classification is by majority over K dims, so the error rate is much lower than 1 − 0.185. A rough estimate: per-dim agreement is `(1-p) = 0.9`; with K=16 dims voting, the binomial concentration around 0.9·K = 14.4 agreements vs the next class's expected ~5.3 agreements gives near-perfect separation. Expected ceiling is **~99–100% on the informative dims**, dropped slightly by tie-break failures with C=10 classes.

The ceiling on this benchmark is essentially 100%. The 98.6% achieved at sig_dim=1024 is within striking distance.

## What changed during the cycle

- Original sweep was single-seed; reported "+15pp peak at sig_dim=16," "+13pp at sig_dim=32," "−2pp anomaly at sig_dim=64."
- Multi-seed corrections: +8.0pp, +8.2pp, +1.8pp respectively. Anomaly evaporates.
- Standard deviations make the corrections legible: ±2–4pp seed noise was hiding inside the single-seed numbers.

## What's NOT measured

- **Mechanism for the +7pp gap (random > identity at sig_dim=D).** The doc proposes "implicit denoising via random ternary projection mixing noise dims into incoherent signal." This is a hypothesis. No measurement of per-dim contribution to inter-class Hamming distance, no test of the CLT-concentration story, no ablation that isolates the proposed mechanism.
- **Mechanism for the +8pp gain at sig_dim=K=16.** The doc proposes "compression sweet spot near the informative-dim count." Hypothesis. No K-sweep that varies the informative-dim count to see if the peak follows.
- **Mechanism for expansion saturation.** Random ternary projection preserves discriminative signal at any sig_dim ≥ ~32 on this benchmark. No measurement of *why* (Hamming-space concentration argument? JL-style analysis?).
- **Performance on data with different signal/noise structure.** The benchmark has a clean K-vs-(D-K) split with uniform-random noise dims. Real data has correlated noise, structured signal, varying class balance.
- **Comparison to base-2 baselines.** No PyTorch attention / softmax-classifier numbers. Substrate-claim is unaddressed.

## Phase context

- Phase A is **mechanism-validation only.** The substrate-claim demonstration ("routing-first base-3 matches base-2 attention") is scoped for Phase B+ on a richer benchmark. This is documented in `gesh/README.md`.
- Phase A.2 produced a working training mechanism + measurement methodology (multi-seed, intra-epoch refresh, hypothesis/finding distinction).
- The sweep is the only quantitative result Phase A.2 can cite. Everything else is structural (kernel correctness, test coverage, doc currency).

## Free-floating observations

- The seed-noise stddev (~2–4pp at low sig_dim) is comparable in magnitude to the gain itself (~5–8pp). The gains are real but not enormous relative to noise.
- At sig_dim=16 specifically, both random and trained have the highest stddev in the sweep (3.8pp and 4.6pp). The "peak gain" cell is also the noisiest cell.
- The expansion regime's near-zero stddev (0.4pp at sig_dim ≥ 384) is consistent with both signatures saturating against the test-set ceiling — there's no headroom for variance.
- "Random ternary at sig_dim = D beats identity" is the only finding where the magnitude (+7pp) is larger than the seed noise (±2pp) by a comfortable margin.
- The benchmark has a known optimal projection. Any mechanism that finds the K=16 informative dims should win. The interesting question is whether differences between mechanisms reflect the mechanism or reflect how easy the benchmark made the search.
