---
cycle: bitnet dead columns (P0 follow-up)
phase: investigation + verdict on mechanism
date: 2026-05-07
scope: investigate the empty-K-row finding from the routed16 weight
       structure analysis. ~10% of o_proj input dims and up to 44%
       of down_proj input dims contribute nothing to their respective
       projections. Determine whether this is real model structure,
       a conversion artifact, or quantization information loss.
companions: commit (the analysis), commit f8a9e32 (atomics analysis
            where this question arose), commit (routed16 weight
            structure where the empties first surfaced).
            scripts/empty_rows_audit.py, /tmp/check_bf16_columns.py.
---

# BitNet's dead columns — what they are and where they come from

## Question

The routed16 weight structure analysis found ~10% empty K-rows in
layer 0 o_proj. Per-K-row variance was 25× random, peaking with
~10% of rows entirely zero. The data was striking enough to warrant
its own investigation.

## Phase 1: per-layer quantification

Decoded all 30 layers' BitLinears. Counted empty K-rows per (layer,
BitLinear). Cross-tabulated overlap.

  Layer | o_proj  empty | down_proj empty
  ------|---------------|-----------------
    0   |  397 (15.5%)  |  333  (4.8%)
    1   |   86  (3.4%)  | 3016 (43.6%)  ← striking
    2   |    3  (0.1%)  | 1917 (27.7%)
    3   |    9  (0.4%)  | 1029 (14.9%)
    4   |    1  (0.0%)  |  626  (9.1%)
    5   |    4  (0.2%)  |  324  (4.7%)
   ...  |    0  (0.0%)  | 23-100 ( <2%)
   28   |   10  (0.4%)  |  154  (2.2%)
   29   |  614 (24.0%)  |  244  (3.5%)

**Empty rows are concentrated at boundary layers** (early + final).
Middle layers (10-25) have essentially none. q/k/v/gate/up are
clean (≤6 empty rows total across all 30 layers each).

**Cross-layer overlap = 0** for every BitLinear. No K-index is
empty in all layers. Empty rows DRIFT — different layers have
different sets of dead input dims.

## Phase 2: provenance — model structure or conversion artifact?

Decoded HF's original W1.58 safetensors directly (bypassing our
substrate blob). Compared empty-row counts.

  EXACT match across all sampled layers:
    Layer 0 o_proj:    397 in HF, 397 in blob ✓
    Layer 1 down_proj: 3016 in HF, 3016 in blob ✓
    Layer 29 o_proj:   614 in HF, 614 in blob ✓

**Verdict: real BitNet W1.58 model structure.** Our conversion
pipeline (4-in-8 → 5-in-8) is a pure repacking with no
quantization step; faithful by construction and confirmed by
matching counts.

## Phase 3: why does it happen?

Loaded the bf16 (unquantized) BitNet weights. For each empty
K-column in W1.58, checked the bf16 magnitudes at the same indices.

W1.58 quantization is per-tensor absmean: scale s = mean(|W|),
then q = round(W/s).clamp(-1, 1). A column rounds entirely to 0
iff every cell satisfies |W[j,k]| < s/2.

**Two distinct mechanisms produce the dead columns:**

### Mechanism A: bf16 was already essentially zero

  Layer 1 down_proj (3016 dead): 3013 of 3016 cols have
    col_max(bf16) < 0.01 × absmean
  Layer 0 down_proj (333 dead):  330 of 333 same pattern
  Layer 1 o_proj (86 dead):      76 of 86 same pattern

  These are TRULY dead channels — the bf16 weights themselves
  are vanishingly small. W1.58 quantization preserves the
  structure; it doesn't create it. These input dims have no
  contribution to their projections at any precision.

### Mechanism B: moderate magnitudes rounded to zero

  Layer 0 o_proj (397 dead): all 397 cols have col_max(bf16)
    in [0.28, 0.50] × absmean

  These columns have nontrivial bf16 magnitudes (up to ~0.48 in
  absolute units, with absmean ~0.97). Each cell is below the
  rounding threshold (s/2 = 0.48), so all round to zero. The
  bf16 model would use these columns; the W1.58 quantized model
  cannot.

  This is quantization-induced "shoulder rounding." Quantization-
  aware training encourages weights to cluster at lattice points
  {-s, 0, +s}; the boundary cases just below s/2 collectively
  vanish. BitNet's training has pushed ~16% of layer-0 o_proj
  input dims into this rounded-to-zero band.

### What this teaches

1. **Real structural sparsity exists in BitNet inference**, but
   it's at the column (input dim) level, not random per-cell. It's
   concentrated at boundary layers.

2. **The bf16 model has ~5-44% structurally-dead channels in
   down_proj** depending on layer. These are model features, not
   quantization losses.

3. **W1.58 quantization adds a few percent more dead columns** to
   o_proj at layer 0 specifically (mechanism B). The bf16 model
   would use those dims; the substrate cannot.

4. **The amount of compute "lost" to quantization is small
   per-layer** (≤16% of o_proj's input dims at layer 0). Whether
   this matters depends on whether the bf16 model performs better
   than the W1.58 model in absolute terms — that's a separate
   measurement (per-token KL, generation quality, etc.).

## Compute opportunity

A **row-skip dense kernel** (skip K-rows where W is empty) would
save proportional to empty-row fraction:

  Per-layer savings on each BitLinear's compute:

  Layer | o_proj | down_proj
  ------|--------|----------
    0   | 15.5%  |  4.8%
    1   |  3.4%  | 43.6%   ← biggest single win
    2   |  0.1%  | 27.7%
    3   |  0.4%  | 14.9%
   ...  | <1%    | <2%
   29   | 24.0%  |  3.5%

Average across 30 layers (ignoring layer-0 q/k/v/gate/up which
are negligible):
  - o_proj total empty rows / 30×2560 ≈ 1.6%
  - down_proj total empty rows / 30×6912 ≈ 4.7%

Per BitLinear's contribution to total inference compute:
  - down_proj is ~14% of per-token compute (largest single
    BitLinear due to K=6912)
  - 4.7% × 14% ≈ 0.7% total inference speedup from down_proj
    row-skip alone
  - All BitLinears combined: ~1-2% inference speedup

This is modest. Not nothing, but not the dramatic win the
single-layer numbers (43.6% on layer 1!) might suggest, because
the savings are heavily layer-dependent and most layers are dense.

## Disposition

**Mechanism understood. Three follow-up directions, ranked:**

1. **Don't build the row-skip kernel right now.** The 1-2% total
   inference speedup is small relative to the engineering cost
   (new kernel, encoder, tests, benches, integration). Below the
   bar where I'd recommend the work without explicit user direction.

2. **Document the quantization-induced loss for o_proj layer 0.**
   ~16% of input dims have moderate bf16 magnitudes that round to
   zero. If we ever investigate "why does the substrate's W1.58
   inference have degraded fact-recall vs bf16" (a real prior
   finding at 5/10 vs 7/10), this is a candidate mechanism. File
   for that future investigation.

3. **The truly-dead channels are permanent features of BitNet
   inference.** No optimization changes that. They are
   exploitable for compute (skip them) or just accepted as
   structural cost.

## What I would propose

Given the user's pattern of preferring methodical, value-aligned
work over speculative optimization: **close this investigation
cleanly with the finding recorded.** The three mechanisms are
identified, the magnitude is measured, the optimization opportunity
is bounded.

If at some point we find that the W1.58 substrate underperforms
the bf16 baseline in a way that matters (and we want to close that
gap), the o_proj quantization-induced dead columns are the most
likely first place to look.
