---
date: 2026-04-17
phase: SYNTHESIZE
topic: Closing the CIFAR-10 gap to >50%
---

# CIFAR-10 >50% — SYNTHESIZE

Executable specification.

---

## What to build

A **projection selection pass** that generates N_cand random
ternary projection directions, scores each by class separability
on the training set, and keeps the top N_keep for use as the
projection matrix. The selected projections replace the random
ones in the existing multi-table LSH consumer.

## Why this is routing-native

- Candidate directions: random ternary weights via RNG.
- Scoring: ternary matmul (m4t_ternary_matmul_bt) over training
  vectors → integer projection outputs → class-conditional means
  → separability score. All integer arithmetic.
- Selection: sort by score, keep top N_keep.
- Downstream: unchanged (bucket index, multi-probe, SUM/k-NN).

No float. No gradient. No pixel-space distance. The routing
measures its own random experiments and keeps what works.

## Implementation

### New tool: tools/selected_projections.c

Single-purpose tool. Steps:

**Step 1: generate N_cand random directions.**

Each direction is a ternary vector of length input_dim (3072 for
CIFAR-10), packed via m4t_trit_pack. At density=0.33, ~1024
non-zero weights per direction.

```c
for (int d = 0; d < N_cand; d++) {
    generate random ternary direction (same as sig_builder_init)
    store as packed trits
}
```

**Step 2: project all training vectors through each direction.**

For each direction d, compute proj_d[i] = w_d ⋅ x_train[i] for
all i ∈ [0, N_train). This is a single ternary matmul at N_PROJ=1.

```c
for (int d = 0; d < N_cand; d++) {
    for (int i = 0; i < n_train; i++) {
        proj[d][i] = ternary_dot(direction_d, x_train[i]);
    }
}
```

**Step 3: score each direction by class separability.**

For each direction d:
- Compute class means: μ_c = mean(proj_d[i] for i in class c)
- Compute separability: sep = Σ_{c≠c'} |μ_c - μ_c'|

```c
for (int d = 0; d < N_cand; d++) {
    int64_t class_sum[N_CLASSES] = {0};
    int     class_count[N_CLASSES] = {0};
    for (int i = 0; i < n_train; i++) {
        class_sum[y_train[i]] += proj[d][i];
        class_count[y_train[i]]++;
    }
    int64_t sep = 0;
    for (int c = 0; c < N_CLASSES; c++)
        for (int c2 = c+1; c2 < N_CLASSES; c2++)
            sep += abs(class_sum[c]/class_count[c]
                     - class_sum[c2]/class_count[c2]);
    scores[d] = sep;
}
```

**Step 4: sort and select top N_keep.**

Sort directions by separability score descending. Keep top N_keep.

**Step 5: build M tables using selected directions.**

Group the N_keep selected directions into M tables of N_PROJ
trits each (N_keep = M × N_PROJ). Build each table's sig_builder
with the selected directions as the projection matrix instead
of random ones. Proceed with standard bucket build, multi-probe,
resolve.

### Parameters

| parameter | value | rationale |
|---|---|---|
| N_cand | 1000 | Cheap to generate and score. Scale to 10000 if gain is marginal. |
| N_keep | 256 | M=16 × N_PROJ=16 = 256 directions. Uses existing uint32 bucket keys. |
| M | 16 | Matches the filter M used in the dynamic cascade experiments. |
| N_PROJ | 16 | Per table. Uses existing 4-byte sig infrastructure. |
| k | 5 | Rank-weighted k-NN (sweep later if needed). |

### Interface with glyph_sig_builder

Currently, `glyph_sig_builder_init` generates its own random
projection matrix internally. For selected projections, we need
to INJECT a pre-computed projection matrix.

Two options:
(a) Add `glyph_sig_builder_init_with_proj(sb, proj_packed, ...)`.
(b) Build the tool's own encoding path that bypasses the builder.

Option (b) is simpler for a proof-of-concept: compute projections
manually (ternary matmul → threshold_extract → trit_pack), using
the selected directions directly. The builder's calibration (τ)
is still needed — compute τ per table from the selected
directions' projection distribution.

Actually, the easiest approach: after generating the projection
matrix inside `glyph_sig_builder_init`, REPLACE `sb->proj_packed`
with the selected directions. The rest of the builder (τ
calibration, encoding) works unchanged.

```c
glyph_sig_builder_init(&builders[m], n_proj, input_dim, density,
                        seeds[0], ...);
/* Replace the random projection with selected directions. */
memcpy(builders[m].proj_packed, selected_proj_packed[m],
       n_proj * packed_row_bytes);
/* Re-calibrate τ with the new projection. */
recalibrate_tau(&builders[m], x_train, n_calib);
```

This requires exposing a τ recalibration function, or just
re-running the calibration logic after replacing proj_packed.

Simpler: init the builder with dummy seeds, then overwrite
proj_packed and re-call the internal calibration. The builder
struct has all fields public — direct access is fine for a
proof-of-concept tool.

## Testing plan

1. **Diagnostic: projection separability distribution.** Before
   building the full tool, measure the distribution of class
   separability across 1000 random directions. If the distribution
   is uniform (all directions equally separable), selection can't
   help. If it has a long right tail (a few directions much more
   separable than average), selection should help significantly.

2. **Build tool and run on CIFAR-10.** Compare selected-projection
   accuracy to random-projection baseline at matched M=16 N_PROJ=16.

3. **If gain > 3pp:** scale to N_cand=10000, N_keep=1024 (M=64 ×
   N_PROJ=16). Run with k-NN k=5.

4. **Cross-dataset validation:** run on MNIST and Fashion-MNIST
   to verify no regression (selected projections might over-specialize
   to CIFAR-10's class structure, but since each dataset gets its
   own selection pass this should be fine).

## Go / no-go

**Go:** CIFAR-10 accuracy ≥ 42% (+4pp over 38.14% brute-force
random baseline). Projection selection provides real gain.

**Strong go:** ≥ 48% (+10pp). Selection finds genuinely
discriminative directions. Scale aggressively.

**No-go:** ≤ 39% (+1pp). All random directions are equally
uninformative. The discriminative structure is not in the
random ternary space at all. Would need non-random feature
design (SSTT's approach).

## Estimated effort

- Separability diagnostic: ~40 lines, ~2 minutes to run.
- Full selected-projection tool: ~200 lines.
- Measurement runs: ~10 minutes per dataset.
- Total: ~250 lines, ~30 minutes.

## What the LMM cycle changed

Started with five candidate approaches (structured features,
cross-query geometry, two-layer routing, subsetting, projection
selection). RAW explored all five. NODES narrowed to two
(selection vs two-layer). REFLECT resolved to projection
selection as the simpler, more direct attack on the diagnosed
bottleneck. SYNTHESIZE produced an executable spec with a
clear diagnostic gate (separability distribution) before the
full build.

The critical REFLECT insight: the N_PROJ=64 peak applies to
RANDOM projections only. Selected projections may shift the
optimal width — the N_PROJ sweep must be re-run after selection.
This prevents over-committing to N_PROJ=64 before knowing
whether selection changes the landscape.
