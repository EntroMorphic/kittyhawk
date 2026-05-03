---
cycle: gesh_kmeans_findings
phase: RAW
date: 2026-05-02
scope: dump observations from the multi-prototype bank investigation chain — top_k breakage, k-sweep, training-hurts-at-high-T
companions: gesh/bench/{mnist_full_run.c, mnist_kmeans_run.c, mnist_kmeans_trained.c} · gesh/src/gesh_bank.{c,h} · gesh/src/gesh_train.{c,h}
status: capture
---

# RAW — gesh_kmeans_findings

Observations only. No interpretation. No mechanism claims.

## The data

All measurements: full MNIST (60K train, 10K test), sig_dim=64, single seed (init=0xc0ffeebb, train=0xa5a5a5a5), random R unless trained, top_k=1 unless noted, post-permille-precision.

### Single-prototype baselines

| Config | Accuracy |
|---|---:|
| Random R, T=10, top_k=1 | 50.0% |
| Random R, T=10, top_k=3 | 30.9% |
| Random R, T=10, top_k=5 | 25.0% |
| Random R, T=10, top_k=9 | 19.0% |
| Trained R (250K flips, 64 epochs), T=10, top_k=1 | 56.8% |
| Trained R (250K flips, 64 epochs), T=10, top_k=3 | 34.0% |

### k-means k_per_class sweep (random R, top_k=1)

| k | T | Accuracy | Δ vs single | Per-doubling gain |
|---:|---:|---:|---:|---:|
| 1 | 10 | 50.0% | +0.0pp | (sanity check) |
| 2 | 20 | 53.4% | +3.4pp | +3.4 |
| 4 | 40 | 57.2% | +7.2pp | +3.8 |
| 6 | 60 | 60.9% | +10.9pp | — |
| 8 | 80 | 64.1% | +14.1pp | +6.9 |
| 12 | 120 | 65.2% | +15.2pp | — |
| 16 | 160 | 67.8% | +17.8pp | +3.7 |
| 24 | 240 | 69.5% | +19.5pp | — |
| 32 | 320 | 70.1% | +20.1pp | +2.3 |

### Trained R + k-means refresh (T=80, k=8, top_k=1)

| Variant | Accuracy | Runtime |
|---|---:|---:|
| Random R + k-means | 64.1% | 0.1s |
| Trained R + k-means | **61.4%** | 98.3s |
| Gain | **−2.7pp** | |

Per-epoch batch error trajectory (trained run):
- epoch 1: 88/128 (69%)
- epoch 32: 42/128 (33%)
- epoch 64: 34/128 (27%)
- 2672 flips accepted out of 250,880 (1.06% acceptance rate)

### Determinism / cross-checks

- k-means with k=1 returns identical accuracy to single-prototype class-mean (both 50.0%). Bank constructor sanity verified.
- Single-prototype top_k=1 (50.0%) reproduces across all probes that measure it.
- `test_gesh_train` and all 13 other tests pass after the k_per_class config addition.

## Side observations

- Top_k > 1 on single-prototype bank: accuracy collapses **toward 1/n_classes ≈ 10%**. Pattern: top_k=3 → 30.9%, top_k=5 → 25.0%, top_k=9 → 19.0%. With 1 tile per class and ties at "1 vote each," the argmax-with-lower-index-wins rule biases predictions toward class 0.
- Top_k > 1 on multi-prototype (k=8, T=80) bank degrades gracefully but doesn't help: top_k=1 → 64.1%, top_k=3 → 56.6%, top_k=5 → 54.4%, top_k=9 → 49.9%.
- k-means build cost is small (20–70ms for k ∈ {1, ..., 32} on 60K samples), even cumulative across training-loop refreshes (~250 refreshes × 30ms ≈ 7.5s of bank-rebuild time per training run).
- Trained R + k-means runtime (98.3s) is longer than trained R + single-proto (87.8s) primarily due to the extra k-means cost in refreshes plus larger T affecting Hamming distance loops in the forward pass.

## What's NOT measured

- Multi-seed at any of these cells. Every result is single-seed.
- Sig_dim variation with k-means. Only sig_dim=64 measured.
- Different training budgets at multi-prototype. Only 250K flips measured.
- Different refresh cadences with k-means. Only n_flips/4 measured.
- Validation accuracy during training (we have batch errors per epoch, not held-out test/val).
- Different k_per_class with training. Only k=8 trained measured.
- Multi-table LSH composition (separate consumer architecture, not measured here).
