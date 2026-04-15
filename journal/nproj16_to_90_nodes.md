---
date: 2026-04-15
scope: LMM cycle — can N_PROJ=16 reach >90%?
phase: NODES
---

# NODES

## Discrete ideas

1. **Ceiling is 91.47%, not 100%.** 8.53% of queries have correct class nowhere in top-10; no strategy using this hash set recovers them. Target of 90% is within 1.5% of ceiling — a near-optimal extraction problem.

2. **Vote rules are near-Bayes for tied-min.** 77.65% rank-wt on tied-min partition is close to the Shannon ceiling of "pick dominant class from a mixed tied set." Further gains require signal beyond the hash.

3. **Vote rules are far below ceiling for elsewhere.** 24.65% majority on elsewhere partition. A top-1 wrong-class prototype dominates; majority can't dethrone it without external signal.

4. **Tied-min set is small.** Avg ~10 prototypes per query (from histogram: mean ~4-6). Cheap to re-rank.

5. **Cascade: hash-filter then pixel-resolve.** Standard LSH pattern. 16-bit hash narrows 60K → ~10 candidates. Pixel L1 or per-class centroid distance resolves the top-10. Cost: 10×784 ops per query = negligible.

6. **Cascade subverts the "N_PROJ=16" framing.** Depends on whether "16-bit hash" means the only primitive, or the primary index in a hash+resolve architecture. Both readings legitimate; must declare which.

7. **Multi-probe: M independent 16-bit hashes.** Each hash uses a different seed for the ternary projection. Candidates are intersection or union. M=4 is effectively 64-bit but distributed.

8. **Secondary-hash resolver.** Within tied set, apply a second 16-bit hash with different projection; pick candidate with smallest secondary distance. Pure signature, no pixel access.

9. **Weighted trit match (not popcount).** Uses ternary values directly instead of collapsing via XOR-popcount. Might help elsewhere partition where trit distances differ but Hamming underresolves.

10. **Learned 16-bit hash.** Replace random ternary projections with projections optimized for class separation on MNIST train set. Preserves "N_PROJ=16" literally but adds training. Ceiling likely >> 90% if trained well.

11. **Bayesian per-class prior in tied set.** Weight tied candidates by log(train class prior) or by empirical per-class mass. Small effect; stays within hash-only domain.

12. **Elsewhere-partition rescue by top-1 rejection.** When tied_count=1 (singleton top-1) but cascade says pixel-distance is actually large, demote that prototype and fall back to rank-2+. A guard against bad top-1.

13. **Architectural framing: the hash is a FILTER, not a classifier.** This is the unifying reframe. Current accuracy is bottlenecked because we asked the hash to do both jobs.

## Tensions

**T1. Spirit of "N_PROJ=16" vs reaching 90%.**
Cascade uses pixels at the resolver stage. Multi-probe is really N_PROJ=64 with structure. Learned hash is still 16 bits but supervised. Each reading trades architectural purity for accuracy. Need to pick which definition the exercise honors.

**T2. Cost-accounting.**
The hash is cheap (16 bits, popcount). Pixel L1 on 10 candidates is 8K ops — 10× the hash cost. Still far cheaper than N_PROJ=4096 (4K ops × 60K = 240M), but no longer "signature-only." If cost is the real objective (not literal N_PROJ), cascade is unambiguously correct.

**T3. Per-partition strategies already win separately.**
Adaptive voting (tied_count ≥ 2 → rank-wt else majority) gains +0.75% — small. Because vote rules alone can't break the 77% tied-min ceiling. Getting to 90% is fundamentally a different regime: not "pick a better vote rule" but "introduce a new signal on the resolver stage."

**T4. Elsewhere partition may be unsalvageable by voting alone.**
If correct is at rank 7 and wrong is at rank 1 with 3 prototypes, voting cannot rescue. Only external re-ranking (pixel L1 or secondary hash) can. So elsewhere partition is the pure case for cascade — no voting trick will work.

**T5. Learned hash lives in a different phase of work.**
Training a hash introduces an ML pipeline not currently in Glyph. The substrate is routing-first, training-agnostic. Learned hash is a *research* answer; cascade is an *engineering* answer. For the current phase, engineering wins.

## Dependencies

- Cascade resolver needs cached train set in pixel form (already have — MNIST images).
- Multi-probe needs N_SEEDS × projection matrices — trivially scalable.
- Learned hash needs training loop — not currently in substrate.
- All strategies need the tied-set enumeration already computed by `mnist_probe_nproj16.c`.
