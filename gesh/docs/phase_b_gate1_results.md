---
title: Phase B Gate 1 — image canon parity probe (MNIST)
date: 2026-05-02
benchmark: MNIST canonical pipeline (substrate-legal: per-image normalize + direct ternary quantization, no random projection of pixels)
status: FAIL — consumer pipeline does not transfer
---

# Phase B Gate 1 — MNIST canonical pipeline probe

**Pre-committed gate (per `journal/gesh_findings_synthesize.md`):**
- **PASS:** trained Gesh ≥ 95% MNIST AND beats untrained random R ≥ +2pp avg across seeds.
- **FAIL:** trained Gesh < 90% MNIST OR trained ≤ random within seed noise.
- **INCONCLUSIVE:** 90–95% range, marginal gain.

**Verdict: FAIL.**

## Setup

- MNIST IDX from `01MAY26_archived/data/mnist/`. 60k train / 10k test loaded; subsampled to 2k/2k for runtime parity with the synthetic sweep.
- Pipeline: IDX → MTFP-encode → per-image normalize (zero-mean, unit-variance, integer arithmetic) → direct ternary quantization at tau = 26687 (60th percentile of |x| across a 1k training sample).
- **No random projections of pixels.** Each input trit corresponds to one pixel.
- Identity baseline: ternary-quantized pixels fed directly as a 784-trit signature; class-mean bank built on 10 classes; top-1 Hamming nearest-neighbor.
- Gesh forward: ternary projection R (sig_dim × 784) → top-1 vote against class-mean bank.
- Gesh trained: random init R + lattice-update with intra-epoch refresh per Phase A.2 H1/H2 remediations. Flip budget capped at 20k, 50 epochs, batch=128.
- 3 seeds per (sig_dim, variant). Independent (init, train) seed pairs.

## Results

**Identity (sig_dim = 784, no projection): 43.4%** (deterministic, single trial).

| sig_dim | random           | trained          | gain        |
|---------|------------------|------------------|-------------|
|     128 |  50.7% ±  1.9pp |  51.6% ±  2.6pp |  +0.8 pp     |
|     256 |  54.2% ±  1.7pp |  54.7% ±  1.6pp |  +0.5 pp     |

Total probe runtime: 30s.

## What the data says

### 1. Random projection beats identity by +7pp (C2 transfers)
Random ternary R at sig_dim=128 hits 50.7% ± 1.9pp; identity at sig_dim=784 hits 43.4%. Gap: **+7.3pp**, larger than the 1σ noise (±1.9pp) by ~4×. **The C2 finding from the synthetic transfers cleanly to MNIST.** Random ternary projection at much-lower-than-input dim *does* extract more discriminative signal than the identity-passthrough at full input dim.

### 2. Lattice update adds essentially nothing (C1 does NOT transfer)
Trained R outperforms random R by +0.8pp (sig_dim=128) or +0.5pp (sig_dim=256), both within the ±1.6–2.6pp 1σ noise. The compression-regime gain that earned the lattice-update mechanism its place on the synthetic (+8pp at sig_dim ∈ {16, 32}) does not appear here. Training is doing nothing detectable.

### 3. Absolute accuracy is far below the gate floor
Gate 1's PASS bar is 95%; INCONCLUSIVE floor is 90%. Trained Gesh at sig_dim=256 hits 54.7%. **The pipeline does not produce a competent MNIST classifier in this configuration.**

## What this means

The consumer pipeline (forward + class-mean bank + Hamming top-1 vote + lattice update) does not transfer to MNIST scale and complexity. Two findings need separating:

- **Substrate-level claim survives:** ternary pixel quantization + random ternary projection captures more signal than identity pixels (+7pp), exactly as on the synthetic. The substrate primitives are doing their job.
- **Consumer-architecture claim fails:** the Gesh-Phase-A consumer (single class-mean bank, top_k=1) is not strong enough to extract MNIST-level structure. Lattice update has nothing meaningful to optimize because the consumer's loss surface is uninformative at this configuration.

The prior cycle's `mnist_routed_bucket_multi M=32 SUM` reached 97.24% on MNIST (per `LSH architecture` memory). That cascade had multi-table composition, multiple buckets per signature, and a richer ranker. Gesh-Phase-A is intentionally a single-table k-NN; that simplicity is a strength on the synthetic and a liability here.

## Loop-back action (per synthesize pre-commit)

> *FAIL action:* the consumer pipeline does not transfer. Loop back to NODES — what about the synthetic was over-fit to. Specifically check: bank construction may need refresh schedules tuned to the real-data scale, top_k may need to grow with class count, projection budget may need re-thinking when D = 784 (MNIST pixels) instead of 64.

Recorded in `journal/gesh_phase_b_probe_closeout.md`. The next-cycle scope is **NODES re-examination** of which Phase A claims are synthetic-specific, NOT incremental tuning of MNIST hyperparameters until they pass. Tuning until pass is post-hoc model selection — exactly the kind of methodology drift the multi-seed rule was promoted to prevent.

## What's still on the substrate-claim path

Three observations from this probe survive to inform the next cycle:

1. **C2 (random > identity) transfers.** The substrate-level routing-first observation is real on MNIST.
2. **The 50–55% accuracy ceiling at sig_dim ∈ {128, 256}** suggests the bank's expressivity is the limit, not the projection. A richer ranker (multi-table, multi-bucket, or learned bank) would be the surface to test next.
3. **+7pp at sig_dim=128 over identity, with random R** suggests the right move is *more banks*, not *better R*. If 10 class-mean tiles can extract +7pp with a random projection, more tiles per class (subclass bands, prototype clusters) might extract proportionately more.

These are hypotheses for the next cycle, not findings of this probe. Recorded for traceability, not for citation.

## Reproduction

```bash
cmake --build build -j --target gesh_mnist_probe
./build/gesh/gesh_mnist_probe
```

Default MNIST path: `01MAY26_archived/data/mnist/`. Override by passing the directory as the first argument.
