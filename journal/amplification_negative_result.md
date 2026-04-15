---
date: 2026-04-14
scope: Amplification experiment (ensemble + audit-triggered fallback)
type: experiment
status: PREDICTION WRONG — negative result on fallback, marginal positive on ensemble
tool: tools/mnist_routed_amplified.c
---

# Amplification: ensemble + fallback — predictions failed honestly

Tested the two amplification paths proposed in the prior turn: (1) K=5 independent ternary projections with cross-projection majority vote, and (2) audit-triggered pixel-k-NN fallback for queries where the ensemble is uncertain. Predicted +0.3 to +0.5% from (1) and +0.1 to +0.2% from (2). Actual results: +0.04% from (1) and *negative* from (2).

## Result (3 master seeds, K=5 projections each, deskewed MNIST N_PROJ=2048)

| Configuration | Accuracy | Delta vs prior best (rank-k=5 single) |
|---|---|---|
| Best solo projection | 97.91 ± 0.04% | +0.05% |
| **Ensemble (no fallback)** | **97.90 ± 0.02%** | +0.04% |
| Ensemble + FB (agree≥5) | 97.75 ± 0.02% | **−0.15%** |
| Ensemble + FB (agree≥4) | 97.86 ± 0.05% | flat |
| Ensemble + FB (agree≥3) | 97.90 ± 0.01% | tied with ensemble |

Fallback trigger rates and recovery rates across master seeds:
- agree≥5 triggered 1.95% of queries; pixel-k-NN correct on 46.7% of triggers
- agree≥4 triggered 0.88% of queries; pixel-k-NN correct on 42.5% of triggers
- agree≥3 triggered 0.04% of queries; pixel-k-NN correct on 30.6% of triggers

## Why ensemble amplification was smaller than predicted

**Failure modes don't decorrelate across random projections on MNIST.**

I predicted +0.3 to +0.5% based on the assumption that errors would be decorrelated between random projection matrices. That turned out to be wrong. The reason:

The failure modes of random ternary projection are mostly driven by the *input image* (structurally ambiguous MNIST digits — a 4 that looks like a 9), not by the *projection choice*. When the underlying digit is ambiguous, every random projection carries that ambiguity into the signature. Different random seeds don't cure it.

Per-seed solo accuracies vary meaningfully WITHIN a seed (0.3-point spread), but the *failure sets* across projections mostly overlap. Ensembling smooths the per-projection variance but doesn't change the aggregate error set.

The ensemble's actual value is **deterministic stability**:
- Best-of-5 solo accuracy varies: 97.95 / 97.92 / 97.87 across seeds (stddev 0.04).
- Ensemble accuracy: 97.90 / 97.92 / 97.89 across seeds (stddev 0.02).

Ensemble matches best-of-5 performance with half the variance. But picking "best of 5" requires oracle knowledge; ensemble is a deterministic substitute. That's a real but small win — reliability, not accuracy ceiling.

## Why fallback was a net loss

**Hard cases are hard for pixel-k-NN too.**

The audit signal (K-agreement < threshold) correctly identifies hard cases — the subset where the ensemble is uncertain. But on that subset, pixel-k-NN is wrong most of the time:
- 46.7% correct at agree≥5 (1.95% of queries triggered).
- The overall pixel-k-NN baseline is 97.16% correct. On the *hardest* 1.95% of queries, pixel-k-NN drops to 46.7%.

Meanwhile, the ensemble was already getting ~50% of these right. Swapping to pixel-k-NN on the trigger set *loses* roughly 15 cases per seed (0.15 points of accuracy).

The structural reason: both the routing classifier and the pixel classifier struggle on the same queries — those with inherent visual ambiguity. Pixel representation isn't a second opinion with new information; it's a different view of the same underlying structural difficulty. The digit that looks ambiguous in trit-space usually looks ambiguous in pixel-space too.

## What this tells us about MNIST at this performance level

1. **~2.1% error is near the floor for ternary-LSH k-NN on this data.** Different random projections all see the same underlying ambiguities; aggregating them doesn't recover information that isn't there.

2. **The routing surface beats dense pixel k-NN even on hard cases.** The fallback experiment was designed to help by tapping pixel representation on uncertain queries. It hurts instead — which means on those hard cases, the routing classifier is actually doing *better* than pixel k-NN, not worse. This is a stronger statement of NORTH_STAR's "routing outperforms dense" claim than we'd established previously: routing isn't just faster, it's at or above the representational limit for this data.

3. **Audit-based adaptation is a detection tool, not a correction tool.** The K-agreement signal reliably flags hard cases. But flagging ≠ fixing. Substituting another classifier on flagged cases only helps if the alternative classifier has information the primary lacks. For MNIST at ~97.9%, the pixel classifier doesn't.

## What amplification would actually need

Given the above, meaningful further amplification would require changing the *representation*, not aggregating more of the same kind of classifier. Candidates:

- **Multi-stage routing.** First-stage coarse classification, second-stage fine-grained classifier on specific confusion pairs. Uses different discriminative features per stage.
- **Deeper projections.** N_PROJ = 2048 may be saturated. N_PROJ = 4096 or 8192 might pick up residual signal, though memory and time costs grow linearly.
- **Multi-metric routing.** Combine packed-trit Hamming with MTFP L1 on a different projection — not to substitute, but to add truly independent information.
- **Data augmentation during training.** The 2% residual error is partly data-level — certain MNIST digits have labeling noise or intrinsic ambiguity. Augmenting training data (rotations, shears, elastic deformations) might reduce the irreducible-looking subset.

None of these fit in a 90-minute follow-up. Each is a several-hour experiment at minimum.

## What this cycle taught me about prediction

Both my amplification predictions were wrong. Why?

**Ensemble prediction (+0.3 to +0.5%):** I assumed errors would be independent across projections, which assumed projections carry independent information. They don't — most of each projection's signal is shared with other projections' signal. Only the residuals differ, and on this data, even the residuals overlap.

**Fallback prediction (+0.1 to +0.2%):** I assumed pixel representation would be better on hard cases than ternary representation, because the 97.16% pixel baseline is "reasonable." What I failed to see: the 97.16% baseline is an AVERAGE. On the hardest subset, pixel-k-NN performs *much worse* than its average. Averaging-over-all-queries numbers don't predict performance on the hard subset.

Both errors are from reasoning about averages when the relevant thing is the distribution. The fair-comparison remediation earlier also taught a version of this lesson — predictions based on "reasonable-looking headlines" miss the important facts about where the signal lives.

**Lesson for future adaptation experiments:** look at the performance distribution on the subset you plan to route to, not the whole-set average. The audit trail already produces this information; use it.

## The honest headline for this cycle

The best routed MNIST classifier on the rebuilt substrate is **97.90 ± 0.02% (K=5 ensemble, rank-weighted k=5 per projection, no fallback)**. Marginal improvement over the 97.86 ± 0.01% single-projection rank-k=5. Fallback doesn't help and can hurt. The residual ~2.1% error rate appears to be at or near the floor for this representational family on this data.

This is a smaller win than I predicted. It's also the *truth* of what the substrate can do here, and it closes several further-amplification directions by ruling them out.

## Pointers

- Tool: `tools/mnist_routed_amplified.c`.
- Raw output: `/tmp/routed_amplified.txt` (or re-run with same seeds to reproduce).
- Prediction that failed: `journal/mechanism_that_worked_synthesize.md` (end section, "Three cheap follow-ups").
- Prior best: `journal/weighted_voting_adaptation.md` (97.86 ± 0.01%).
- Mechanism of the prior best: `journal/mechanism_that_worked_{raw,nodes,reflect,synthesize}.md`.
