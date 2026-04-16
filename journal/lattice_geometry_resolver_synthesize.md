---
date: 2026-04-16
phase: SYNTHESIZE
topic: Lattice Geometry Resolver — reading the routing pass's own measurements
---

# Lattice Geometry Resolver — SYNTHESIZE

Executable specification.

---

## What to build

A **margin-weighted SUM resolver** that reads each table's own
geometric confidence. One new function in libglyph. Wire into
the dynamic cascade tool for multi-stage testing.

## Why this, not k-NN or PCA

Margin-weighting tests the thesis that the lattice's own
geometry contains usable information the current resolver
discards. k-NN and PCA are valid engineering improvements but
don't test the thesis. REFLECT showed that the mechanism is
sound: the decisive subset's accuracy must merely exceed the
full set's accuracy, and since the full set includes 75% pure
noise (tied tables), any directional signal in the decisive
subset is an improvement.

## Implementation

### Step 1: glyph_resolver_sum_marginweighted

```c
int glyph_resolver_sum_marginweighted(
    const glyph_union_t* u,
    int                  m_active,
    int                  sig_bytes,
    uint8_t* const*      table_train_sigs,
    const uint8_t* const* query_sigs,
    const uint8_t*       mask)
{
    /* Phase 1: compute per-table margin.
     * For each table m, find the two smallest distances to any
     * candidate in the union. margin[m] = d_2nd - d_1st. */
    int32_t margins[m_active];  /* VLA; m_active ≤ 256 */
    int32_t total_margin = 0;
    for (int m = 0; m < m_active; m++) {
        int32_t d1 = INT32_MAX, d2 = INT32_MAX;
        for (int j = 0; j < u->n_hit; j++) {
            int idx = u->hit_list[j];
            int32_t d = popcount_dist(query_sigs[m],
                            train_sigs[m] + idx * sig_bytes,
                            mask, sig_bytes);
            if (d < d1) { d2 = d1; d1 = d; }
            else if (d < d2) { d2 = d; }
        }
        margins[m] = (d2 == INT32_MAX) ? 0 : (d2 - d1);
        total_margin += margins[m];
    }

    /* Fallback: if all tables are tied, use unweighted SUM. */
    if (total_margin == 0) {
        return glyph_resolver_sum(u, m_active, sig_bytes,
                                  table_train_sigs, query_sigs, mask);
    }

    /* Phase 2: weighted score per candidate.
     * score(c) = Σ_m margin[m] × dist(q_m, c_m). */
    int64_t best_score = INT64_MAX;
    int     best_label = -1;
    for (int j = 0; j < u->n_hit; j++) {
        int idx = u->hit_list[j];
        int64_t score = 0;
        for (int m = 0; m < m_active; m++) {
            int32_t d = popcount_dist(query_sigs[m],
                            train_sigs[m] + idx * sig_bytes,
                            mask, sig_bytes);
            score += (int64_t)margins[m] * d;
        }
        if (score < best_score) {
            best_score = score;
            best_label = u->y_train[idx];
        }
    }
    return best_label;
}
```

**Cost:** 2 × O(n_hit × m_active) popcount_dist calls. Phase 1
scans the union once per table to find top-2 distances. Phase 2
scans again to compute weighted scores. Total: 2× the cost of
scalar SUM. Acceptable for the measurement phase.

**Optimization path (deferred):** fuse Phases 1 and 2 into a
single pass by computing margins on-the-fly during the first
partial scan, then weighting during the remainder. Requires
careful bookkeeping but halves the popcount_dist calls.

### Step 2: add to glyph_config

Add "marginweighted" to the --resolver_sum validation.

### Step 3: wire into dynamic_nproj.c

At each stage of the cascade, compute both SUM and margin-
weighted SUM predictions. Report both accuracy columns.

Alternatively (simpler): add margin-weighted as a standalone
resolver option in the multi-table consumer, test at N_PROJ=16
M=64 first, then add to the cascade.

**Decision:** standalone first (multi-table consumer at N_PROJ=16),
cascade second. Isolate the mechanism before integrating.

### Step 4: diagnostic — margin-correctness correlation

Before the full sweep, measure whether the decisive subset has
signal. For each (query, table) pair on CIFAR-10:

    is_decisive = (margin > 0)
    is_correct  = (per-table 1-NN label == true label)

Report:
    P(correct | decisive) vs P(correct | tied)
    P(correct | decisive) vs P(correct | all)

If P(correct | decisive) > P(correct | all), the mechanism is
sound and margin-weighting should help. If P(correct | decisive)
≈ P(correct | all), the decisive tables aren't more accurate
than average and margin-weighting is reweighting noise.

This diagnostic uses the fashion_atomics tool (extended) or a
one-off measurement. It's the go/no-go gate for the resolver.

## Go / no-go criteria

**Go:** P(correct | decisive) ≥ 1.2 × P(correct | all) on
CIFAR-10. The decisive tables are at least 20% more likely to
be correct than the full set. Margin-weighting should produce
a measurable accuracy gain.

**Marginal:** P(correct | decisive) between 1.0× and 1.2× of
P(correct | all). Decisive tables are slightly better but the
margin is thin. Margin-weighting might produce a small gain
(<1pp) that's hard to distinguish from noise.

**No-go:** P(correct | decisive) ≤ P(correct | all). Decisive
tables are no more accurate than average. The mechanism is
broken — confidence doesn't correlate with correctness in the
trit lattice at this configuration. Pivot to k-NN resolver.

## Testing plan

1. **Diagnostic** (Step 4): measure P(correct | decisive) on
   CIFAR-10 at N_PROJ=16 M=64. Go/no-go gate. (~2 minutes.)

2. **Standalone** (Step 3 first option): run multi-table
   consumer on CIFAR-10 with --resolver_sum marginweighted at
   N_PROJ=16 M=64. Compare to scalar SUM baseline (35.32%).
   (~5 minutes.)

3. **Cascade integration** (Step 3 second option): add margin-
   weighted scoring to each stage of the dynamic cascade. Report
   per-stage accuracy and cascade accuracy at various thresholds.
   Compare to the scalar cascade results. (~15 minutes.)

4. **Cross-dataset validation**: run Fashion-MNIST and MNIST
   with margin-weighted resolver. Confirm no regression on MNIST,
   check for improvement on Fashion-MNIST upper-body cluster.

## What the LMM cycle changed

The design document proposed three progressively richer variants
(V1 global, V2 per-candidate, V3 agreement-weighted). The LMM
cycle found:

1. V1 is sufficient for the first test. V2/V3 add complexity
   without changing the fundamental mechanism.
2. The go/no-go gate should be P(correct | decisive) — measured
   BEFORE building the resolver, not after.
3. The right first deployment is the standalone multi-table
   consumer (N_PROJ=16), not the cascade. Isolate the mechanism
   before integrating.
4. At N_PROJ=16, continuous weighting naturally degenerates to
   binary table selection. The same code works at wider N_PROJ
   with richer margins.

## Files to create / modify

| file | action |
|---|---|
| `src/glyph_resolver.{h,c}` | add glyph_resolver_sum_marginweighted |
| `src/glyph_config.c` | add "marginweighted" to valid resolver_sum modes |
| `tools/mnist_routed_bucket_multi.c` | wire dispatch |
| `tools/fashion_atomics.c` | add decisive-subset diagnostic (Step 4) |

## Estimated effort

- Step 4 diagnostic: ~30 lines in fashion_atomics.c + one run.
- Steps 1-3: ~50 lines across resolver + config + tool.
- Step 4 full sweep: ~20 minutes of measurements.
- Total: ~80 lines of code + ~25 minutes of measurements.

## Execution order

1. **Diagnostic first** (Step 4). Measure P(correct | decisive).
   If no-go, skip the resolver build entirely and pivot to k-NN.
2. If go: build resolver (Steps 1-2), wire into tool (Step 3).
3. Measure on CIFAR-10, Fashion-MNIST, MNIST.
4. If the mechanism works: integrate into the dynamic cascade.
