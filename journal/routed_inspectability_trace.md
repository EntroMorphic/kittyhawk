---
date: 2026-04-14
scope: Inspectability demo for routed k-NN — per-classification audit trail
type: demonstration
tool: tools/mnist_routed_trace.c
---

# Routed k-NN: per-classification audit trail

Demonstration of the third axis of the routing thesis — not accuracy, not speed, but **inspectability by construction**. Dense k-NN produces scalar L1 distances that can't be decomposed. Routed Hamming is a literal sum of per-trit {0, 1, 2} costs; every classification has a readable audit trail for free.

## Setup

Single deterministic run, deskewed MNIST, N_PROJ=2048, k=3, density=0.33 (the best-performing configuration from `journal/routed_knn_mnist.md`). Accuracy: 9779 / 10 000 = 97.79%, matching the multi-seed mean.

For each of the 221 misclassified test images, the tool records:
- Top-5 nearest training prototypes (index, label, Hamming distance).
- Vote composition at k=3.
- Per-trit breakdown vs the top-1 prototype (agreements split by trit value; disagreements split by sign-flip vs zero-vs-sign).
- Per-class nearest-prototype distance across all 60 000 training signatures.
- A failure classification (NARROW MISS / VISUAL CONFUSION / SEPARATED / OUTLIER) derived from those numbers.

## Aggregate failure distribution

| Failure type | Count | Fraction of misclassified | Criterion |
|---|---|---|---|
| NARROW MISS | 74 | 33.5% | correct-class prototype within 10 bits of winner |
| VISUAL CONFUSION | 65 | 29.4% | both classes have near-lattice prototypes |
| SEPARATED | 82 | 37.1% | correct class genuinely far from query |
| OUTLIER | 0 | 0.0% | no class has a close prototype |

The absence of OUTLIER cases means the lattice has reasonable coverage for the test distribution; every query lands somewhere recognizable. The 1/3 NARROW MISS share suggests specific classifier-parameter improvements (distance-weighted voting, higher k) have visible leverage — not speculated from the aggregate accuracy number, but traceable to specific cases in the trace output.

## Case-level observations

Selected traces (full output in `/tmp/routed_trace.txt` from the run):

### Test #435 — k-sensitivity on a single case
```
True 8, predicted 9. Top-5: {9, 8, 9, 8, 8} at distances {616, 622, 627, 631, 632}.
Vote at k=3: [8]=1 [9]=2 → predicted 9 (wrong).
Vote at k=5 (hypothetical): [8]=3 [9]=2 → would predict 8 (correct).
```
The correct class is the majority at k=5, minority at k=3. This is visible at the case level. Dense k-NN produces the same ranking but without the "would it flip at k=5" framing made immediate.

### Test #247 — voting failure with correct top-1
```
True 4, predicted 6. Top-5: {4, 6, 6, 6, 6} at distances {835, 854, 863, 867, 872}.
The single nearest prototype IS the correct class. But the next 4 form a cluster of 6s.
Unweighted vote lets the cluster beat the closest example.
```
A distance-weighted vote (e.g., weight = 1/distance) would likely recover this case. The trace shows where that modification would help — not as a hypothetical, as a visible pattern.

### Test #321 — same voting-failure pattern
```
True 2, predicted 7. Top-1 is correct 2 at d=579; next 4 are 7s at d∈{610, 619, 619, 627}.
31-bit gap between true-class top-1 and wrong-class top-2.
```
Another distance-weighting candidate. The pattern repeats.

### Test #445 — genuine visual confusion
```
True 6, predicted 0. Top-5 all 0, at distances {870, 881, 890, 892, 895}.
Class 6's nearest prototype is at d=918 — 48 bits FURTHER than the winner.
In trit-space, this specific handwriting of 6 genuinely resembles 0s.
```
Not a classifier failure. A data-level ambiguity. The trit-space distances make the ambiguity quantitative.

### Per-trit breakdown structure
Across the traces, a consistent pattern:

- Typical near-miss has ~70% trit agreement with its top-1 prototype.
- Of the ~30% disagreements, sign-flips (cost 2, full opposition) are a small minority (5-10%).
- Zero-vs-sign mismatches (cost 1, threshold-boundary noise) are the large majority (90-95% of disagreement-bits).

**Observation:** errors cluster at the quantization boundary, not at semantic opposition. The router rarely says "+1" where the correct class says "-1"; it more often says "0" where the correct class says ±1 (or vice versa). This is a statement about where the information loss lives in the signature encoding.

Dense L1 over MTFP mantissas cannot produce this observation. The distance is a sum of absolute magnitudes; the "kind of disagreement" isn't preserved.

## What this demonstrates about the substrate

The routing surface was designed (per NORTH_STAR and the original README) as glass-box. This trace tool is the first end-to-end demonstration of what "glass-box" actually delivers:

1. **Every decision carries its reasoning.** The top-k list, the per-trit decomposition, and the per-class distance spectrum are all present by construction. Nothing was added for this tool; the substrate already produced them as intermediate values.

2. **Failure modes are quantitatively distinguishable.** NARROW MISS / VISUAL CONFUSION / SEPARATED are derived from integer thresholds on the distance values. No learned classifier; the structure was already in the numbers.

3. **Modification targets are visible at the case level.** "This case would flip at k=5" and "this case would flip under distance-weighted voting" are read directly off the trace, not inferred from aggregate accuracy.

Dense k-NN produces the same rankings but no decomposition. If you want to know why dense's L1-distance of 2847.3 ranked above 2847.5, you can't — L1 is a scalar sum and the contributing dims aren't individually separable without recomputing the distance per-dim.

## What this does NOT claim

- **Not a universal inspectability win.** We haven't shown that routed inspectability is useful for every task. On MNIST classification, it's interesting; for something like medical image diagnosis, it would be load-bearing. Scope is context-dependent.
- **Not a replacement for formal model explanations.** Our trace says "72% of trits agreed with this prototype"; it doesn't say "this prototype represents concept X." Concept-level explanation needs more than per-trit counting.
- **Not new in principle.** Hamming-LSH has always been compositional. What we're demonstrating is that the substrate's primitives preserve that compositionality all the way through to the consumer — nothing collapses to a scalar without reason.

## What to do next with inspectability

Now that the audit-trail primitive exists, several downstream experiments are cheap:

1. **Distance-weighted voting.** Trace shows 3 of 8 sample traces have correct top-1 but wrong cluster. Weighted voting likely converts a significant share of NARROW MISS cases. Small code change; measurable accuracy impact.
2. **Per-prototype coverage analysis.** Aggregate: which training signatures are "useful" (frequently a k-nearest neighbor) vs "dead weight" (never picked)? Could enable prototype pruning — 60K signatures trimmed to a smaller effective set without accuracy loss.
3. **Sign-flip vs zero-vs-sign attribution.** The observation that errors cluster at the threshold boundary suggests τ calibration has a second-order effect on classification quality — not the 33%-density target itself, but the placement of specific trit boundaries. Could motivate per-dim τ calibration (each projection dim gets its own threshold).
4. **Confusion-pair prototype sharing.** For VISUAL CONFUSION cases (3/8, 4/9, etc.), look at which prototypes are close to *both* classes. Candidates for discriminative features: dims where those prototypes split cleanly in trit-space.

Each of these is an experiment enabled by inspectability; none are possible from a scalar-L1 baseline.

## Pointers

- Tool: `tools/mnist_routed_trace.c`.
- Underlying measurement: `journal/routed_knn_mnist.md` "Revised" section.
- The structural argument for inspectability: this file.
- Substrate primitive: `m4t_popcount_dist` in `m4t/src/m4t_trit_pack.c` — the sum is per-byte popcount, which is why decomposition at the trit level works at all.
