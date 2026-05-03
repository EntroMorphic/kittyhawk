---
cycle: gesh_kmeans_findings
phase: NODES
date: 2026-05-02
scope: extract discrete claims, hypotheses, anchors from the RAW dump
companions: gesh_kmeans_findings_raw.md
status: structuring
---

# NODES — gesh_kmeans_findings

Discrete units. **C** = claim (supported by data within stated conditions). **H** = hypothesis (proposed but mechanism not directly tested). **A** = anchor (fixed reference frame). **D** = dependency.

## Claims (data-supported)

### C1 — `top_k > 1` on a single-prototype bank silently collapses to "predict class 0"
**Statement:** With T=C tiles (one per class), top_k > 1 always produces top_k distinct class labels (one each), tally is k-way 1-1-...-1 ties, argmax-with-lower-index-wins picks class 0. Accuracy degrades from 50% (top_k=1) toward 10% (top_k=9).
**Evidence:** Direct measurement: single-proto top_k ∈ {1, 3, 5, 9} → {50.0, 30.9, 25.0, 19.0}.
**Conditions:** Holds for any bank where `bank.labels[]` is dense and unique (one tile per class label). Multi-prototype banks with shared labels per class are exempt.
**Severity:** Architectural. Not a tuning issue; it's a structural defect of top_k > 1 with single-prototype banks.

### C2 — Bank capacity dominates training as the accuracy lever
**Statement:** Going from T=10 single-proto → T=80 k-means at constant random R yields **+14.1pp** (50.0 → 64.1). Adding training to T=10 single-proto yields **+6.8pp** (50.0 → 56.8). Bank-architecture change is **2.1× larger than training** in the regimes measured.
**Evidence:** Side-by-side measurements at fixed sig_dim=64.
**Conditions:** Holds for sig_dim=64 on MNIST; magnitude may shift at other sig_dim.
**Sensitivity:** The +14.1pp bank gain depends on the dataset having intra-class structure that k=8 prototypes can capture. Synthetic benchmarks with unimodal classes wouldn't show this gap.

### C3 — Training **hurts** at high bank capacity (load-bearing for the cycle)
**Statement:** With T=80 (k=8 k-means, kmeans refresh in training loop), trained R + k-means scores **−2.7pp** worse than random R + k-means (61.4 vs 64.1) at the same bank shape. The lattice update mechanism is anti-helpful in this regime.
**Evidence:** Direct measurement, same seeds, same bank constructor, only training vs no-training varied.
**Conditions:** Single seed, single config (sig_dim=64, n_train=60K, 250K flip budget, refresh n_flips/4, k_per_class=8). **By the recently-promoted "match scope of evidence to scope of claim" rule, this is OUTCOME, not yet FINDING — needs multi-seed and multi-config validation.**

### C4 — k-means k=1 reproduces single-prototype class-mean exactly
**Statement:** `gesh_bank_build_kmeans_per_class` with k_per_class=1 produces a bank bit-identical to `gesh_bank_build_class_mean`. k=1 measurement (50.0%) matches single-proto baseline (50.0%) exactly.
**Evidence:** Determinism cross-check in the k-sweep probe.
**Conditions:** None — algorithmic identity by construction.

### C5 — Top_k > 1 on multi-prototype (T=80) bank degrades but does not collapse
**Statement:** Unlike single-prototype, multi-prototype with top_k>1 doesn't collapse to ~10%. It degrades from 64.1% (top_k=1) to 49.9% (top_k=9) — a smooth 14pp slope rather than a structural collapse.
**Evidence:** Direct measurement at T=80.
**Conditions:** Multi-prototype with sufficient T such that top_k tiles can come from the same class. With T=80 and top_k=9, ~9 of 80 tiles vote — at most ~one per class (worst-case spread); accuracy degradation is from vote dilution, not pigeonhole-forced ties.

### C6 — k-sweep curve is concave, knee around k=8, plateau approaching k=32
**Statement:** Per-doubling gain is +3.4 (1→2), +3.8 (2→4), **+6.9 (4→8)**, +3.7 (8→16), +2.3 (16→32). Knee at the 4→8 doubling.
**Evidence:** k-sweep at fixed sig_dim=64 random R.
**Conditions:** Single-seed measurement; magnitudes may shift with seed but the concave shape is structural (more clusters → more intra-class structure → diminishing returns once major modes are captured).

## Hypotheses (proposed; mechanism not tested)

### H1 — Loss signal disconnects from generalization at high T
**Statement:** The lattice update minimizes per-batch classification error using the current bank. With k-means, refreshes (k-means rebuilds) move the bank topology between flip-evals. Training optimizes R for transient bank states, gets fragile signal that doesn't generalize.
**Predicts:** Reducing flip budget should reduce overfitting harm. Switching to a non-classification-error objective (e.g., inter-class signature separation, computed once from training data, no refresh) should help.
**Mechanism test:** Run with budget ∈ {25K, 50K, 100K, 250K}. Predict: smaller budgets give larger trained-R accuracy, possibly converging back above random R baseline at some budget.

### H2 — k-means is more R-sensitive than class-mean
**Statement:** Small perturbations in R can flip k-means cluster assignments, causing centroid shifts. Single-prototype class-mean only changes via per-class summation differences — small R changes give small bank changes. The lattice update's signal-to-noise is worse with k-means refresh.
**Predicts:** A "frozen-bank" variant (k-means built once at init, never refreshed during training) should give more stable training. Same final accuracy if our hypothesis is right; better if k-means refresh is actively destabilizing.
**Mechanism test:** Run trained-R with `bank_refresh_every = n_flips × n_epochs` (effectively never refresh). Measure final accuracy.

### H3 — Per-batch optimization overfits the routing pattern
**Statement:** Each flip-eval evaluates "does flipping this trit reduce batch errors?" With k=8 prototypes per class and top_k=1, each query routes to 1 of 80 tiles. Training pushes R toward the batch's specific routing topology, which doesn't generalize.
**Predicts:** Larger batch sizes should reduce overfitting. So should optimization on validation rather than batch.
**Mechanism test:** Run trained-R with batch_size ∈ {128, 512, 2048, full}. Predict: larger batches narrow the train/test gap.

### H4 — The k-sweep accuracy ceiling is near 75% with single-bank multi-prototype
**Statement:** Going from k=24 (69.5%) to k=32 (70.1%) is +0.6pp; the asymptote with this consumer architecture (single bank, no multi-table) is likely 72–75%. The gap to the archive's 97% comes from multi-table composition, not from k.
**Predicts:** k=64 or k=128 should give marginal gain over k=32, ≤ 3pp.
**Mechanism test:** Sweep k ∈ {32, 48, 64, 96, 128}; map plateau.

## Anchors

### A1 — Phase A consumer architecture: single bank, ternary projection R, Hamming top-k vote
The substrate-claim path commits to this consumer in Phase A. The MNIST measurements above stay within this architecture; multi-table composition would be a different consumer (Phase B+ scope).

### A2 — MNIST as the regression-guard, not the substrate-claim primary
Per `project_benchmark_pivot` memory: image canon is the regression-guard for substrate-discipline; Go positions are the substrate-claim primary. MNIST measurements here are diagnostic ("is the consumer working?"), not the substrate-claim measurement.

### A3 — Single-seed measurements need multi-seed validation per the meta-rule
Per the recently promoted CONTRIBUTING.md rule ("Match the scope of evidence to the scope of claim"), single-seed measurements support cell-level outcomes but not directional claims. C3 (training hurts) is a directional claim from a single-seed cell — needs multi-seed before it's a finding.

### A4 — k_per_class=1 collapses to class-mean (verified by C4)
The k-means constructor cleanly subsumes the class-mean constructor. This means the new code is strictly more general than what it replaces; backward compatibility is structural.

## Dependencies

- C1 (top_k>1 broken) depends on `bank.labels` having unique values. If a future bank constructor uses non-unique labels (multi-prototype within class), C1 doesn't apply and top_k>1 becomes meaningful.
- C3 (training hurts at high T) depends on A1 (single-bank consumer) and on H1/H2/H3's mechanism story. If any of those mechanisms is the cause, the inversion is the bank-refresh interaction with the lattice update — *specific to this consumer's training loop shape*, not a substrate-claim-level finding.
- C2 (bank > training as lever) depends on having enough capacity headroom in the bank to make the comparison. At very low T (T=2 with 10 classes), no bank-only solution can beat 50% by much; training would be needed. This is the same Finding 3 capacity-floor regime in different clothing.
- All claims depend on **MNIST** as the dataset. Generalization to Go positions or other domains is unmeasured.

## What's not in any node

- A direct test of whether the lattice-update mechanism *can* help at high T. The mechanism may need a different loss objective (per H1 mechanism test).
- Whether multi-table LSH composition changes the bank-vs-training dominance relation. That's a different consumer; not in this cycle's scope.
- Substrate-discipline cleanup of the new k-means orchestration code. The k-means constructor uses scalar assignment loops, scalar sum updates, scalar argmin — all of which fall under task #14's purification scope.
