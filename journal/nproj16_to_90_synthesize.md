---
date: 2026-04-15
scope: LMM cycle — can N_PROJ=16 reach >90%?
phase: SYNTHESIZE
---

# SYNTHESIZE: the cascade experiment

## Thesis

The 16-bit hash is a **filter**, not a classifier. Currently we use it as both and hit ~62% aggregate. Reframed as filter with a cheap local resolver on the top-K candidate pool, the same 16-bit hash should reach 85-91% — with 90% attainable depending on K and resolver choice. The ceiling under this architecture is 91.47% (the "correct in top-10" fraction at K=10, rising with K).

## Architecture

```
Query q
  ↓
[ N_PROJ=16 signature hash ]          ← PRIMARY INDEX (does 60K × 16 bits of work)
  ↓
[ Rank 60K prototypes by Hamming distance to q ]
  ↓
[ Take top-K candidates ]              ← K ∈ {10, 20, 50}
  ↓
[ Resolver: pixel L1 (or alt) over K candidates ]   ← SECONDARY (~K × 784 ops)
  ↓
[ Vote among K_resolved using majority or 1-NN ]
  ↓
Predicted label
```

Hash cost: 60000 × 16 = 960K trit-ops.
Resolver cost: K × 784 ≈ 8-40K scalar-ops.
Ratio: resolver is 1-4% of hash cost. Hash dominates.

## Concrete experiment plan

### E1. Baseline cascade (primary target)
- K = 10.
- Resolver: pixel L1 distance between query and each of the K candidates.
- Final prediction: 1-NN in the resolved set (label of candidate with smallest pixel L1).
- **Predicted: 87-89% accuracy.**

### E2. Top-K sweep
- Same as E1 but K ∈ {5, 10, 20, 50, 100}.
- Goal: find where resolver ceiling meets resolver confusion.
- Expected: K=5 limited by 85-86% ceiling; K=20 peaks near 90%; K=50+ starts confusing resolver with far candidates.

### E3. Resolver sweep
- K=20 fixed.
- Resolvers: pixel L1, pixel L2, per-class centroid L1 (precomputed), k'-NN with k'=3 within top-K.
- Identifies which cheap signal best discriminates on the filtered pool.

### E4. Partition-aware cascade
- Only apply resolver when `tied_count ≥ 2` (tied-min partition).
- For singleton top-1: skip resolver, take top-1's label.
- Goal: confirm cascade gains come from tied-min resolution, not from perturbing already-correct singletons. Tests mechanistic story.

### E5. Secondary-hash variant (for architectural purity)
- Replace pixel resolver with a *second* 16-bit ternary hash (different seed).
- Re-rank top-K by secondary Hamming distance.
- **Predicted: 75-80%** — better than single-hash k-NN but worse than pixel resolver, because secondary hash has same structural limitations as primary.
- Purpose: quantify how much of the cascade gain comes from pixel access vs just "more bits."

## Implementation plan

New tool: `tools/mnist_cascade_nproj16.c`.

Structure (copy from `mnist_probe_nproj16.c`):
1. Load deskewed MNIST train + test.
2. Build ternary projections at N_PROJ=16, density=0.33, seed=42.
3. Compute all signatures.
4. For each test query:
   - Compute 60K Hamming distances (existing code).
   - Sort prototypes by Hamming distance; take top-K.
   - Compute pixel L1 between query and each top-K candidate.
   - Prediction = label of top-K candidate with smallest pixel L1.
5. Report accuracy for K ∈ {5, 10, 20, 50, 100} in one pass.

Rough size: ~200 lines of C (mostly duplicated from probe tool).

## Success criteria

- **Primary:** cascade E1 reaches ≥ 85%. This proves the filter-vs-classifier reframe.
- **Stretch:** E2 at K=20 reaches ≥ 90%. This proves the user's hypothesis.
- **Honest reporting:** even if E1 hits 85%, name the trade: we've added pixel access to a signature-only architecture. The win is real but the cost-accounting changed.

## Fallback if cascade underperforms

If pixel resolver only gets us to 75-80% (not 85-90%), two diagnoses:
1. Top-K misses correct more often than probe suggests — widen K.
2. Pixel L1 is too weak on MNIST — try L2 or centroid distance.

If both fail, the 8.53% "nowhere" ceiling is probably higher than predicted (some queries have correct at rank 11-50), and the architecture needs a richer primary filter — which means N_PROJ > 16. That would falsify the hypothesis cleanly.

## Note on architectural honesty

This experiment uses 28×28 = 784 pixel values per query at resolver time. That *is* external signal beyond the 16-bit hash. The architecture is fairly called "16-bit LSH with pixel tie-break," not "pure 16-bit LSH." If we want the pure reading to hit 90%, E5 (secondary hash only) is the test — and it will almost certainly fail, confirming that pure-hash 16-bit cannot reach 90%.

**Reframing target:** "What's the cheapest cascade built on a 16-bit hash that reaches 90%?" The answer is what E1-E3 will tell us.

## Next action

Implement `tools/mnist_cascade_nproj16.c` (E1 + E2 + E3 in one pass). Run. Report. Update journal with actual vs predicted. If it hits 90%, declare the filter-vs-classifier reframe validated and add cascade to the substrate.
