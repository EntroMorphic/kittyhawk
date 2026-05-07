---
cycle: routed16 weight structure (P0)
phase: measurement + verdict on group-wise hypothesis
date: 2026-05-07
scope: settle whether real BitNet weights have tile-pattern sharing
       across output columns. The atomics analysis showed routed16
       loses 2× to dense because dense register-tiles 4 columns and
       shares the X-load. The proposed fix (group-wise routed16)
       requires column-pair correlation to be useful. This cycle
       measures whether that correlation exists.
companions: commit f8a9e32 (atomics analysis where the question
            arose), scripts/analyze_weight_structure.py.
---

# routed16 weight structure — group-wise hypothesis closed

## Question (rephrased from atomics close-out)

The dense kernel beats routed16 because it register-tiles 4 output
columns and shares the X-load across them (0.24 ops/cell vs
routed16's 0.63 ops/cell). The hypothetical fix: group-wise routed16
that processes 4 columns simultaneously, sharing X-loads.

For this to work, the 4 columns being processed together must share
enough nonzero positions to make the shared X-load worthwhile. If the
columns are essentially independent (random sparsity), every position
in the union needs processing — and the union of 4 random columns
covers ~all positions, eliminating the sharing benefit.

So: **does real BitNet have column-pair correlation worth exploiting?**

## Method

Decoded all 7 layer-0 BitLinears from the substrate blob into dense
ternary matrices. Sampled 2000 random column pairs and 2000 random
4-column groups per BitLinear. Measured:

  1. Per-column nnz distribution — is sparsity uniform across cols?
  2. Per-row (K-position) nnz distribution — uniform across rows?
  3. Pairwise jaccard similarity (sampled column pairs)
  4. 32-trit window all-zero rate (single col → block-skippable)
  5. 32-trit window all-zero rate (4-col group → group-block-skippable)
  6. 4-col group lanes-covered per window — measures union coverage

For each metric, computed expected value under random-bernoulli
sparsity at the observed cell density, and reported the ratio
observed/random. Ratios >>1 = structure to exploit.

## Result

### Pairwise column correlation: weak

  BitLinear  | jaccard mean | random expectation | ratio
  -----------+--------------+--------------------+-------
  q_proj     |     0.31     |      0.34          | 0.92×
  k_proj     |     0.34     |      0.37          | 0.94×
  v_proj     |     0.41     |      0.40          | 1.02×
  o_proj     |     0.47     |      0.36          | 1.33×
  gate_proj  |     0.42     |      0.44          | 0.97×
  up_proj    |     0.43     |      0.45          | 0.97×
  down_proj  |     0.48     |      0.45          | 1.08×

**Six of seven BitLinears are within ±10% of random expectation.**
The outlier (o_proj at 1.33×) suggests modest column correlation,
but still not enough to drive group-wise sharing.

### 4-col group block-skip rate: essentially zero

  BitLinear  | mean block-skip% | random expectation
  -----------+------------------+-------------------
  q_proj     |    0.000000      |   1.1e-37
  k_proj     |    0.000000      |   2.5e-41
  v_proj     |    0.000000      |   1.2e-45
  o_proj     |    0.006875      |   4.1e-40
  gate_proj  |    0.000000      |   7.0e-51
  up_proj    |    0.000000      |   1.8e-52
  down_proj  |    0.000000      |   3.3e-52

For a 4-column group, a "block-skip" means a 32-trit window where
ALL 4 columns are entirely zero. Random expectation: vanishingly
small at typical sparsity. Real BitNet: also essentially zero.
o_proj's 0.007% mean (max 1.25% across some specific groups)
suggests SOME structure but not enough to drive a kernel.

### Lanes-covered per 32-trit window in a 4-col group

  BitLinear  | covered lanes | random expectation
  -----------+---------------+-------------------
  q_proj     |     30.00     |   30.06
  k_proj     |     30.48     |   30.51
  v_proj     |     30.56     |   30.91
  o_proj     |     25.95     |   30.37   ← meaningfully fewer!
  gate_proj  |     31.24     |   31.25
  up_proj    |     31.31     |   31.33
  down_proj  |     29.95     |   31.32

For 6 of 7 BitLinears, ~30 of 32 lanes per window are touched by at
least one of any 4 columns. **The union of any 4 columns covers
nearly every position.** This means a group-wise X-load wouldn't
share any work — every lane would be processed for at least one of
the 4 columns.

o_proj is the exception: ~26 of 32 lanes covered (vs random ~30).
4-of-32 lanes per window are zero across all 4 columns of a random
group. That's a 12.5% potential skip — meaningful, but localized to
o_proj only.

### Row-position structure: STRONG

  BitLinear  | nnz/row sd ratio vs random | nnz/row range
  -----------+-----------------------------+----------------
  q_proj     |     2.20×                   | 1074..1798
  k_proj     |     1.93×                   | 245..521
  v_proj     |     4.57×                   | 82..535
  o_proj     |    25.33×                   | **0..2084 (10% empty rows!)**
  gate_proj  |     2.30×                   | 3850..5323
  up_proj    |     2.84×                   | 2370..4839
  down_proj  |    14.96×                   | **0..1930**

The variance in nnz-per-row is far higher than random would predict,
especially for o_proj and down_proj. **o_proj has ~10% of K-rows
that are entirely zero across all 2560 output columns.** Some
hidden-dim channels feeding the attention output projection
contribute nothing. down_proj has the same feature less pronounced.

## What this answers

**Group-wise routed16 — the original P0 prototype hypothesis — would
not deliver a meaningful win on real BitNet weights.**

- Pairwise column jaccard is at random for 6/7 BitLinears.
- Random 4-col group block-skip rate is ~0% across the board.
- The union of 4 random columns covers ~30/32 lanes per window —
  nothing to share.
- The structure that DOES exist is per-row, not per-column-pair.

The dense kernel's win over routed16 (the X-load amortized across 4
columns) is structural to the *random* aspect of BitNet sparsity.
You cannot beat that with a kernel optimization unless you find
correlation that isn't there.

## What this opens

The row-position structure is a real, surprising finding:

  - 10% of o_proj's K-rows are entirely empty. Dense compute on
    those rows is wasted memory bandwidth (W bytes loaded for
    zero contribution).
  - down_proj has empty rows too, less pronounced.
  - All BitLinears show higher-than-random row-density variance.

This points to a different (separate) optimization opportunity:
**row-skip dense kernel** that bypasses K-positions where W is
empty. Concretely:

  - Per BitLinear, compute a row-empty bitmap at pack time.
  - Dense kernel skips W loads + SDOTs for empty rows.
  - For o_proj: ~10% bandwidth/compute saved.
  - For down/up/gate: <1% saved (most rows have nonzeros).

This is a different cycle. It doesn't help routed16. It helps
dense, specifically on o_proj.

## Decision

**Group-wise routed16 prototype: do not build.** The hypothesis is
falsified by the data. Building it would be substantial engineering
that the bench data already says wouldn't pay off.

**Routed16 cycle: closed cleanly with this final negative result.**

  - Original premise: "math as signatures via routing" buys speed
    at high sparsity. → measured: only above ~92% sparsity.
  - BitNet weight sparsity (38-50%): far below crossover.
  - BitNet activation sparsity (peak 87%): also below crossover.
  - Atomic-level optimization: ≤5% headroom on existing kernel.
  - Group-wise restructure: structure doesn't exist in real
    weights to make it pay off.

The kernel stays in libm4t as documented infrastructure. It does
what it does correctly. Its win condition does not exist in BitNet.

## What I would propose next (separate from routed16)

The row-structure finding is too clean to ignore. If we want to
keep extracting compute wins from BitNet's inference path:

  1. **Row-skip dense kernel** — primary candidate. ~10% win on
     o_proj specifically; trivial on others. Small implementation,
     measurable benefit.
  2. **Investigate WHY o_proj has 10% empty rows.** Is this an
     artifact of the absmean/scale path? Of the convert_weights.py
     decoding? Of legitimate model structure (dead channels)?
     If artifact → fix the artifact, sparsity drops; if legit
     model structure → exploit it.
  3. **Look at attention masks during inference.** Decode-time
     causal masks zero out future positions in attention scores
     — that IS a high-sparsity operation. Could routed16 fit
     attention score computation? Different question, separate
     measurement.

These are next-cycle questions, not part of this routed16 close.
