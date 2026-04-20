# Routing Footprint as Primary Metric — SYNTHESIZE

**Date:** 2026-04-20

---

## The thesis

The multi-table LSH architecture produces a routing footprint for every image — an M-dimensional vector of bucket-level routing decisions, enriched with probe-depth and co-candidate structure. This footprint carries classification signal that is independent of trit-level Hamming distance. The current architecture uses the routing footprint as a filter (LSH) and a confidence signal (GSH) but scores candidates with a metric (Hamming) that cannot access routing-topology information. Promoting the routing footprint from filter to scorer is the next phase-shift.

## Design: Routing-Footprint k-NN

### Representation

For each image, define the routing footprint as the M-dimensional co-bucket membership vector:

```
footprint[m] = bucket_key_in_table_m    (uint32, the 16-trit spatial summary key)
```

Two images are routing-similar if they share bucket keys in many tables.

### Distance

**Tier 1 distance (co-bucket Hamming):**
For images A and B, compute:
```
d_route(A, B) = number of tables m where footprint_A[m] != footprint_B[m]
```
Range: [0, M]. Cost: M integer compares. At M=64, this is a 64-element vector — can be packed as a bitmask (1 bit per table: match/mismatch) and computed with popcount.

This is the simplest routing distance. It counts how many tables agree that A and B belong in the same spatial neighborhood.

### Scoring

Replace or augment the Hamming resolver with routing-footprint scoring:

**Option A — Routing-only:** Score candidates by `d_route` alone. Ignore trit distance. This tests whether the routing footprint is sufficient for classification.

**Option B — Routing-then-Hamming:** First rank by `d_route` (how many tables co-route). Among ties, break with Hamming distance. This tests whether routing provides coarse ranking that Hamming refines.

**Option C — Weighted combination:** `d_combined = alpha × d_route + (1 - alpha) × d_hamming_normalized`. Sweep alpha. This tests the independence of the two signals.

### Sparsity-aware trit distance (parallel experiment)

Independently test whether shared zeros carry positive evidence:

```
d_sparsity(A, B) = Hamming(A, B) - beta × shared_zero_count(A, B)
```

If beta > 0 improves accuracy, shared sparsity is discriminative. This is orthogonal to the routing footprint — it can be tested on brute-force k-NN without any LSH infrastructure.

## Experiment plan

### Experiment 1: Routing-footprint independence measurement (diagnostic, no new scorer)

For each CIFAR-10 test query where Hamming k-NN is WRONG but the correct class IS in the union:
- Record `d_route` to the Hamming-nearest wrong prototype
- Record `d_route` to the nearest correct prototype
- If `d_route(correct) < d_route(wrong)` for a majority of these cases, the routing footprint carries independent signal

**Expected output:** fraction of oracle-hit/resolver-miss queries where routing correctly discriminates.

**Falsification:** If this fraction ≤ 50%, routing footprint is not independently informative.

### Experiment 2: Routing-only k-NN (Option A)

Score all candidates by co-bucket overlap only. No Hamming distance.
- Measure accuracy on CIFAR-10, Fashion-MNIST, MNIST
- Compare to Hamming-only k-NN at same M

**Purpose:** Establish the ceiling of routing-only classification. Even if it's lower than Hamming, the question is whether the errors are DIFFERENT — if so, combination will help.

### Experiment 3: Combined scoring (Option C)

Sweep alpha in `d_combined = alpha × d_route + (1 - alpha) × d_hamming_normalized` at alpha ∈ {0, 0.1, 0.2, ..., 1.0}.

**Purpose:** Measure whether any nonzero alpha improves over alpha=0 (pure Hamming). If the optimal alpha is 0, routing footprint adds nothing. If optimal alpha is 1, Hamming adds nothing to routing. If 0 < alpha < 1, the two signals are complementary.

### Experiment 4: Sparsity-aware distance (independent)

On brute-force k-NN (no LSH), test:
```
d(A, B) = hamming(A, B) - beta × count(A[i]==0 && B[i]==0)
```
Sweep beta ∈ {0, 0.1, 0.2, ..., 1.0}.

**Purpose:** Test whether shared zeros carry class-discriminative information. Independent of the routing hypothesis.

## Implementation notes

- Experiment 1 requires only analysis code in direct_lsh.c's existing query loop. The footprints are already computed (bucket keys per table per image). Need to add: for each resolver-miss, compare d_route to correct vs wrong prototype.

- Experiment 2 requires a new resolver variant (or inline scoring in direct_lsh) that ranks by co-bucket count instead of Hamming distance.

- Experiment 3 requires normalizing the two distance scales (Hamming range [0, 2×total_dim], routing range [0, M]) before combining.

- Experiment 4 is independent of the LSH pipeline. Can be added to direct_lsh.c's brute-force scoring path or built as a standalone tool.

## What emergence looks like

If Experiment 1 shows >65% of resolver-miss cases where routing correctly discriminates, and Experiment 3 shows optimal alpha in [0.2, 0.5], the routing footprint is a genuine second channel of classification information. The phase-shift is then: redesign the resolver to be a multi-channel scorer (routing + trit distance + sparsity), not a single-metric ranker.

If Experiment 1 shows ≤50%, the routing footprint is redundant with Hamming, and the path forward is the block-encoded pattern distance (trit-level, not routing-level).

Either outcome advances the thesis. The LMM makes the prediction before the measurement.

## Predictions (recorded before running)

1. Experiment 1 (independence): **60-70%** of resolver-miss cases where routing correctly discriminates. Basis: GSH agreement already lifts accuracy by ~10pp on the agreeing subset, suggesting routing carries independent signal, but the signal is noisy.

2. Experiment 2 (routing-only): **35-40%** on CIFAR-10 (below Hamming's 46.6% but above the 10% floor). Routing alone is coarse — 16-trit spatial summary keys can't discriminate fine texture — but it should capture gross category structure.

3. Experiment 3 (combined): Optimal alpha at **0.15-0.30** on CIFAR-10. Routing contributes but Hamming dominates. On Fashion-MNIST, optimal alpha may be higher (routing-similar garments are the failure mode that Hamming can't resolve).

4. Experiment 4 (sparsity): **Positive beta** on CIFAR-10 and Fashion-MNIST. Shared zeros correlate with class identity because natural images have structured sparsity (backgrounds, uniform regions). MNIST: neutral (deskewed digits have similar sparsity patterns across classes).
