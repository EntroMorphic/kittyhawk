---
date: 2026-04-16
phase: SYNTHESIZE
topic: Routing the CIFAR-10 gap — from 37.90% to 53%+ without leaving the lattice
---

# Routing the gap — SYNTHESIZE

Executable specification.

---

## What to build

A **dimension-subsetted multi-table consumer** where each table
projects a random subset of D input dimensions instead of all
input_dim dimensions. The rest of the pipeline (bucket index,
multi-probe, union, resolve) is unchanged.

## Why this closes the gap

The 37% CIFAR-10 ceiling is caused by 1:192 per-table compression
(3072 dims → 16 trits). MNIST reaches 97% at 1:49 compression.
Reducing per-table input to D=256 gives 1:16 compression —
denser than MNIST. The projection can capture local discriminative
structure that 1:192 compression washes out.

## Why this is routing-native

Each table's dimension subset is selected by the same RNG that
generates projection weights. No spatial knowledge, no pixel-
space computation, no float. The routing architecture is routing
through random input subspaces — each table takes a different
random slice of the input and projects it independently.

## Implementation

### New tool: tools/subsetted_multi.c

Standalone tool (not modifying the existing multi-table consumer).
Focused, single-purpose, no backward-compatibility concerns.

```
Architecture:
  M_filter tables, each with:
    - dim_subset[D] — random subset of D input indices
    - sig_builder at N_PROJ over D dims (not input_dim)
    - bucket index on the D-dim signatures
  
  Per query:
    - Extract each table's D-dim subset from the query vector
    - Probe each table's bucket index
    - Build union (same as multi-table)
    - Resolve by SUM or k-NN across all tables
    
  Optional: multi-resolution re-rank with wider N_PROJ
  per spatial subset
```

### Key parameters

| parameter | value | rationale |
|---|---|---|
| D | 256 | 3072/256 = 12 disjoint subsets possible. 1:16 compression at N_PROJ=16. |
| M_filter | 16 | 12 subsets cover the full image; 4 extra for redundancy. |
| N_PROJ | 16 | Same as current — tests the subsetting hypothesis in isolation. |
| Subset selection | random (RNG-based) | Routing-native. No spatial knowledge. |
| Resolver | SUM 1-NN and k=5 | Compare both to establish whether k-NN compounds with subsetting. |
| Re-rank | N_PROJ=32 over same D=256 subset | 1:8 compression for re-rank — 12× denser than current 1:96. |

### Per-table subset selection

For each table m:
1. Seed the RNG with table m's seed (same derive_seed).
2. Generate a random permutation of [0, input_dim).
3. Take the first D indices as the subset.
4. Sort the subset for cache-friendly access.

Random permutation via Fisher-Yates shuffle on the index array.
~10 lines of code.

### Per-query subset extraction

For each query q and table m:
1. Extract dims[subset[0]], dims[subset[1]], ..., dims[subset[D-1]]
   from the query vector into a contiguous D-length buffer.
2. Pass the buffer to glyph_sig_encode.

The extraction is O(D) per table per query. With D=256 and
M=16, that's 4096 MTFP copies per query — negligible vs the
popcount_dist work.

### Training-set pre-extraction

For the training data, pre-extract each table's D-dim subset
at build time:

```c
m4t_mtfp_t* subset_train[M]; /* M × n_train × D */
for (int m = 0; m < M; m++) {
    subset_train[m] = malloc(n_train * D * sizeof(m4t_mtfp_t));
    for (int i = 0; i < n_train; i++)
        for (int d = 0; d < D; d++)
            subset_train[m][i * D + d] =
                x_train[i * input_dim + subset_indices[m][d]];
}
```

Then pass subset_train[m] as the calibration and encoding
input to glyph_sig_builder_init with input_dim=D.

## Testing plan

1. Run subsetted_multi on CIFAR-10 with D=256, M=16, N_PROJ=16.
   Compare to full-image baseline (37.90% combined k=5).

2. If improvement > 5pp: add multi-resolution re-rank at
   N_PROJ=32 per subset. Expect further gains since 1:8
   compression is far richer than the current 1:96.

3. If improvement > 10pp: test spatial blocks (8×8×3 = 192 dims)
   vs random subsets at matched D to measure whether spatial
   coherence adds value.

4. Run on MNIST and Fashion-MNIST to verify no regression
   (subsetting should be neutral-to-positive on lower-dim
   datasets since D=256 > 784/M is already most of the image).

## Go / no-go

**Go:** CIFAR-10 accuracy ≥ 42% (+4pp over 37.90% combined
baseline). Subsetting provides real information-density gain.

**Strong go:** CIFAR-10 accuracy ≥ 48% (+10pp). Approaching
SSTT territory. Spatial blocks and routing-learned subsets
are worth pursuing.

**No-go:** CIFAR-10 accuracy ≤ 39% (+1pp). Compression ratio
is not the bottleneck — the problem is deeper than per-table
information density.

## Files

| file | action |
|---|---|
| `tools/subsetted_multi.c` | new tool |
| `CMakeLists.txt` | add build target |

No library changes. The existing glyph_sig_builder_init already
accepts any input_dim.

## Estimated effort

- Subset selection + extraction: ~40 lines
- Tool main (based on mnist_routed_bucket_multi skeleton): ~200 lines
- Build and test on CIFAR-10: ~10 minutes
- Total: ~250 lines, ~30 minutes.
