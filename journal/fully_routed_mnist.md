---
date: 2026-04-14
scope: First genuinely-routed MNIST classifier on the rebuilt substrate
type: experiment
tool: tools/mnist_routed_lattice.c
---

# Fully routed MNIST classifier — result and reading

First MNIST consumer that exercises the **full routing surface** (`m4t_route_sign_extract` + `m4t_route_distance_batch` + `m4t_route_topk_abs`) on the rebuilt substrate. Same projections and training as `mnist_trit_lattice.c`; only the decision step differs.

## Pipeline

```
  Input image (784 int32 MTFP19 mantissas)
    ↓  random ternary projection (m4t_mtfp_ternary_matmul_bt)
  Projection (N_PROJ int32 MTFP19 mantissas)
    ↓  sign-extract (m4t_route_sign_extract, widening to int64)
  Query signature (N_PROJ packed trits)
    ↓  popcount-Hamming distance × 10 class signatures (m4t_route_distance_batch)
  Per-class distances (10 int32)
    ↓  score = MAX_DIST - distance; topk_abs(k=1)
  Predicted class
```

Class signatures were built offline as `sign(centroid[c][p] - mean_over_classes(centroid[*,p]))`. The mean-subtract normalizes dims where every class is consistently positive or negative (those contribute zero bits to discrimination).

Mask is all-ones; every trit participates.

## Numbers (same projections, same training, N=10 000 test)

```
N_PROJ   L1-over-mantissa     Routed (sign + VCNT)    Combined-loop ms
  256        80.12%                57.47%                   382
  512        80.46%                58.38%                   784
 1024        81.14%                57.54%                  1636
 2048        81.40%                58.37%                  3215
```

Routed path loses ~23 accuracy points across every N_PROJ.

Timing note: both paths run in the same inference loop (shared projection cost). "Combined-loop ms" includes the projection step twice-amortized. Isolating classification-only time would require separate loops.

## Reading

### The routing surface works
Every `m4t_route_*` primitive we built is exercised and produces the arithmetically-correct result. Sign-extract preserves the sign bit per dim; distance_batch computes the masked popcount-Hamming distance; topk_abs selects the class with the largest score (= smallest distance). This is the first consumer that actually invokes the full surface. Substrate-correctness ✓.

### The routing surface *loses* on this task
Sign-extract compresses 19 trits of mantissa per dim into 1 trit of sign. That's 19:1 information loss at the representation boundary. On MNIST — where the classification signal is distributed across *magnitudes* (ink density, stroke thickness, relative pixel brightness contrasts after ternary projection) — throwing magnitude away is a structural defeat.

The routed path's accuracy *barely* responds to more projections (57→58% going from N_PROJ=256 to N_PROJ=2048). That's the tell: once you've sign-extracted, doubling the projection dim doesn't meaningfully add information. The bit-budget of the signature is what's binding, not the projection width.

### This is consistent with NORTH_STAR, not against it

NORTH_STAR §4:
> "Running routing-native on [MNIST] is a test of adapter efficiency, not the thesis. MNIST is posed in base-2: scalar intensities, Euclidean distance, one-hot labels. The real test is problems whose structure is base-3 from the start."

This experiment measured adapter efficiency. The answer is: an MNIST nearest-centroid classifier adapted to the routing surface loses ~23 points vs. the L1-over-mantissa decision. That is *not* evidence against the thesis "routing will naturally outperform dense in a base-3 environment." It is evidence that **MNIST is not a base-3 environment** for this decision shape.

### What this does NOT say
- It doesn't say routing is slower or less hardware-native. The popcount-distance path is clearly simpler per operation; we didn't measure it cleanly against the L1 path (both paths ran in one loop).
- It doesn't say the routing primitives are wrong. They implement their spec correctly.
- It doesn't say sign-based representation is universally weaker. On tasks where the decision *is* about sign patterns (LSH for set similarity, sparse-activation classification, certain signal detection tasks), sign-based distance can match or beat magnitude-based distance.

## Recovery paths (if we wanted to close the gap on MNIST)

Each of these is a consumer-level modification, not a substrate change:

1. **Multi-bit signatures.** Instead of one sign bit per projection dim, use 2-bit quartile buckets per dim. Distance becomes a Hamming over a 2-bit code. Twice the signature size, much more information retained. Still routing-compatible.
2. **Multi-projection banks.** Project with K independent random matrices; aggregate K routing decisions via vote or `apply_signed`. The substrate's `apply_signed` becomes the combiner.
3. **Per-class tile banks.** Instead of one signature per class (current), give each class M signatures (clustered sub-centroids). Route query → k-of-(N_CLASSES·M) tiles, accumulate class votes. This fully uses `topk_abs` + `apply_signed` for what they were designed for: k-of-T expert routing.

None of these are worth doing on MNIST *for thesis reasons*. They'd only close an adapter-efficiency gap, not test the thesis. They'd be worth doing on a task where sign-structure is native to the problem.

## Empirical take-aways

1. The substrate is empirically sound: routing primitives produce correct results when invoked end-to-end.
2. MNIST at 81.40% L1 is the adapter ceiling for this consumer shape. Routing on MNIST underperforms by construction.
3. **The thesis is still untested.** To test "routing outperforms dense in a base-3 environment," we need a problem whose structure is natively base-3 — sparse, sign-patterned, or lattice-geometric in a way MNIST's pixel intensity distribution is not.
4. The next move is therefore NOT to improve MNIST accuracy. It is to identify or construct a benchmark where the base-3 shape is load-bearing.

## Open items this leaves on the table

- **Isolated classification-only timing.** Current tool reports combined-loop ms. A separate pass per path would give us "dense L1 vs. routed Hamming" in microseconds-per-query.
- **SDOT utilization.** Still not measured. The ternary projection step uses MTFP19 × ternary via bit-select, not MTFP4 × ternary via SDOT. Exercising the SDOT path would require quantizing activations to MTFP4 first — another consumer experiment.
- **Benchmark bed** (`docs/THESIS.md` §4). Still the biggest open. This experiment reinforces why.

## Pointers

- Tool: `tools/mnist_routed_lattice.c`.
- Baseline tool (L1-only): `tools/mnist_trit_lattice.c`.
- First-light note: `journal/rebuilt_substrate_first_light.md`.
- Thesis and benchmark-bed openness: `docs/THESIS.md`.
