---
cycle: gesh_findings
phase: NODES
date: 2026-05-02
scope: extract discrete claims, hypotheses, and anchors from the RAW dump; map dependencies between them
companions: gesh_findings_raw.md
status: structuring
---

# NODES — gesh_findings

Discrete units extracted from the data. Each one is a candidate for being right, wrong, conditional, or load-bearing in the next phase. Tagged by status: **C** (claim — supported by the data within stated conditions), **H** (hypothesis — proposed but not measured), **A** (anchor — fixed reference frame the cycle stands on), **D** (dependency — what other nodes lean on).

## Claims (supported by the data)

### C1 — Lattice update earns positive gain in the compression regime
**Statement:** at sig_dim ≤ 32, multi-seed gain is +4.4pp to +8.2pp, with stddev 0.8–4.6pp. The lower bound of the 1σ confidence interval is positive at every cell in this range.
**Evidence:** five-seed averages with reported stddev. Independent (init, train) seeds per trial.
**Conditions:** synthetic prototype benchmark; D=64; K=16; n_train=2000; flip budget 5×sig_dim×D.
**Sensitivity:** if flip budget were halved, gain would shrink (from M6 — undertraining warning). If n_train were smaller, bank-derived statistics would have higher variance and the gain might fall under noise.

### C2 — Random ternary projection at sig_dim = D outperforms identity at sig_dim = D
**Statement:** at sig_dim = 64 = D, random ternary R achieves 76.4% ± 2.1pp; identity achieves 69%. Gap: +7.4pp ± 2.1pp.
**Evidence:** five seeds for random; one deterministic trial for identity (no projection seed).
**Conditions:** the same as C1, plus the noise dims being uniform-random ternary in samples.
**Sensitivity:** if noise dims had structure correlated with the class, identity might be competitive. If informative dims had noise correlated across dims, random projection's CLT-style averaging argument might fail. Neither is tested.

### C3 — Lattice update adds nothing in the expansion regime
**Statement:** at sig_dim ≥ 64, multi-seed gain is between −0.8pp and +1.8pp, with stddev 0.4–2.3pp. The 1σ confidence interval includes zero at every cell in this range.
**Evidence:** five-seed averages from sig_dim ∈ {64, 128, 256, 384, 512, 768, 1024}.
**Conditions:** same benchmark.
**Sensitivity:** if benchmark were harder (more noise, more classes, structured signal), expansion regime might not saturate as cleanly and training could re-acquire room.

### C4 — Expansion saturation is monotone through 16× input dim
**Statement:** random R accuracy increases monotonically with sig_dim from sig_dim=64 (76.4%) through sig_dim=1024 (98.6%); concave; no inflection upward; trained tracks random within seed noise across this range.
**Evidence:** twelve-cell extended sweep.
**Conditions:** same benchmark.
**Sensitivity:** the synthetic noise floor (per-trit p=0.10 flip on informative dims) caps achievable accuracy below 100%; this cap may be the real ceiling, not anything specific to ternary projection.

### C5 — Single-seed measurements produced narratives that didn't survive multi-seed
**Statement:** the original single-seed sweep reported a "+15pp peak," "+13pp," and "−2pp anomaly" that became +8.0pp, +8.2pp, and +1.8pp respectively under five-seed averaging. The peak's magnitude was a single-seed artifact; the anomaly evaporated.
**Evidence:** direct comparison.
**Conditions:** holds for this benchmark and this training surface; the lesson generalizes (now in CONTRIBUTING.md).
**Sensitivity:** the qualitative shape (compression helps, expansion saturates) survived. The corrected numbers are smaller and less dramatic.

## Hypotheses (proposed but not measured)

### H1 — Implicit denoising via random ternary projection
**Statement:** random ternary projection of the 48 noise dims produces signature contributions that are uncorrelated with class identity, while informative dims produce class-correlated contributions; class-mean banks built in projected space therefore have higher signal-to-noise than banks built in input space.
**Predicts:** per-dim contribution to inter-class Hamming distance, after random projection, is higher for projected dims that received more weight from informative input dims than from noise input dims. Equivalently: reweighting random R toward informative dims improves accuracy; reweighting toward noise dims degrades it.
**Mechanism test:** sample random R; for each output dim, compute |sum of R weights on informative input dims| − |sum of R weights on noise input dims|; bin output dims by this score; measure per-bin contribution to inter-class Hamming distance. If H1 is right, the bins are positively correlated.
**Cost:** cheap. Reuses sweep harness. ~20 lines of measurement code.

### H2 — Compression sweet spot tracks K (the informative-dim count)
**Statement:** the gain peak at sig_dim = 16 is not arbitrary; it tracks the informative-dim count K. Re-run with K=8 should peak at sig_dim ≈ 8; with K=24, at sig_dim ≈ 24.
**Predicts:** linear (or near-linear) shift of peak gain location with K.
**Mechanism test:** synth_proto already has `informative_dim` in its config. Vary K ∈ {4, 8, 16, 24, 32}; sweep sig_dim ∈ {2, 4, 8, 16, 32, 64} per K; tabulate.
**Cost:** ~5× the runtime of the existing sweep (one K per sweep). Unattended.

### H3 — Bayes-optimal ceiling is ~99% on this benchmark
**Statement:** a classifier given the K=16 informative dims directly should achieve ~99% accuracy on this benchmark, capped by p=0.10 per-trit noise plus C=10 tie-break failures.
**Predicts:** sig_dim=1024 random's 98.6% is at saturation; nothing on the substrate-claim path can push this number meaningfully higher on this benchmark.
**Mechanism test:** closed-form Bayes computation (no measurement needed; just an analytical ceiling). Compare to sig_dim=1024 random.
**Cost:** trivial; ~30 minutes of analysis.

## Anchors (fixed reference frames the cycle stands on)

### A1 — The benchmark is synthetic with a known optimal projection
The K-informative-vs-(D-K)-noise structure is a designed-in property. PCA on training data (with class supervision) recovers the K informative dims directly. Any mechanism that approximates this recovery wins.

### A2 — Phase A is mechanism-validation, not substrate-claim
The substrate-claim ("base-3 routing matches base-2 attention") is scoped for Phase B+ on richer benchmarks. Phase A's only job is "does the mechanism we built actually train?" The sweep results address that question, not the substrate-claim question.

### A3 — Discipline rules constrain the next move
- `feedback_no_synthetic`: synthetic results don't count as evidence for closing real-data gaps; next-step scoping must cite real-data paths.
- `feedback_no_random_projections`: not violated here (random projections are the *baseline being beaten* on a non-image task; image canon is the rule's domain).
- `project_benchmark_pivot`: Go position evaluation is the primary substrate-claim benchmark; image canon is the regression-guard.

### A4 — Multi-seed methodology is now load-bearing
CONTRIBUTING.md was updated to require multi-seed validation for any directional measurement claim. Future work in this lineage cannot retreat to single-seed.

## Dependencies

- C1, C2, C3, C4, H1, H2, H3 all depend on **A1**. Drop A1 (use a different benchmark) and every claim/hypothesis must be re-tested.
- C2's +7.4pp gap is the only finding whose magnitude exceeds seed noise by a comfortable margin. C1 and C3 are within a few stddevs of zero. **The strongest observed effect is the one the docs are most uncertain about** (mechanism-wise).
- H1 is testable cheaply and would upgrade C2 from "robust correlation" to "demonstrated mechanism." If H1 fails, C2 becomes "we observe a robust gap with no known mechanism."
- A2 means none of these findings are evidence for or against the substrate-claim. They're evidence about the *training mechanism we built*, not about *base-3 vs base-2*.

## Things not in any node

- **Real-data measurement.** Nothing in this dataset speaks to behavior on natural distributions. The benchmark is a probe, not a proxy.
- **Comparison to base-2 baselines.** No softmax / attention numbers exist for this benchmark in the current cycle.
- **Mechanism story for C4.** Why does saturation happen at this rate? Could be a JL-style argument (random projection preserves pairwise distances with high prob); could be Hamming-space concentration. Unmeasured.
