---
title: Dynamic N_PROJ — resolution-adaptive routing cascade
status: Design proposal (2026-04-16)
companion: journal/fashion_mnist_atomics.md · journal/fashion_mnist_density_sweep.md
evidence: cifar_seed_overlap.c cross-seed overlap measurements
---

# Dynamic N_PROJ — resolution-adaptive routing cascade

## The idea

Instead of projecting every query at a fixed N_PROJ, start cheap
(N_PROJ=16) and escalate to wider projections (N_PROJ=32, 64) only
for queries where the first pass lacks confidence. The entire
cascade is routing-native — same signature-as-address, bucket index,
multi-probe, SUM resolver at every stage. The only variable is the
lattice resolution per query.

## Empirical motivation

Three measurements converge on the same conclusion: the lattice
knows when it's right and when it's blind, and those two populations
are seed-invariant.

### 1. Cross-seed overlap (CIFAR-10, N_PROJ=16, M=64, 3 seeds)

```
always wrong (0/3):  6161   61.6%
swing (1/3 or 2/3):   586    5.8%
always right (3/3):  3253   32.5%
```

94.2% of queries have a fixed fate regardless of projection seed.
The failure set is input-driven, not projection-driven.

### 2. Confidence separation (CIFAR-10, seed-0 metrics)

|                          | always-right (3253) | always-wrong (6161) |
|--------------------------|---------------------|---------------------|
| per-table votes for true | 24.4%               | 15.3%               |
| per-table votes for win  | 24.4%               | 15.5%               |
| chance baseline (10 cls) | 10%                 | 10%                 |

Clean separation: always-right queries achieve 24% per-table vote
agreement; always-wrong are at 15%, barely above chance. A threshold
on per-table vote count is a computable, O(1) confidence gate that
distinguishes the two populations.

### 3. Fashion-MNIST atomics

Same structural pattern at a different point on the scaling curve.
Per-table vote on failing queries: true 30.5%, winner 35.1%. The
lattice is less blind on Fashion-MNIST than CIFAR-10 (closer to
discriminative) but still shows the same binary "sees it / doesn't"
structure.

### 4. Amplification experiment (MNIST, earlier work)

Three N_PROJ=2048 projections produce correlated errors. More of
the same doesn't help. But wider projections (N_PROJ doubling) have
always produced clean accuracy gains on the scaling curve. Different
resolution is different information; different seeds are not.

## Architecture

```
query → Stage 1: N_PROJ=16, M tables
              │
              ├─ votes_for_winner ≥ K₁  →  accept (cheap, ~32% on CIFAR-10)
              │
              └─ uncertain  →  Stage 2: N_PROJ=32, M₂ tables
                                    │
                                    ├─ votes_for_winner ≥ K₂  →  accept
                                    │
                                    └─ uncertain  →  Stage 3: N_PROJ=64
                                                          └─ accept (final)
```

### Properties

**Self-similar.** Every stage is the same mechanism (projection →
signature → bucket → multi-probe → SUM resolve) at a different
lattice resolution. No dense fallback, no pixel math, no change
in architectural shape.

**Cost-adaptive.** On MNIST where 97% are easy, Stage 1 handles
nearly everything. On CIFAR-10 where 32.5% are easy, Stages 2-3
fire frequently. The system automatically allocates compute
proportional to input difficulty.

**Routing-native end-to-end.** The confidence gate is a count
(votes_for_winner out of M) — already computed during the probe
pass. The escalation is a fresh projection-encode-bucket-resolve
pass at wider N_PROJ. No new primitives required.

**Monotone accuracy.** Escalation can only help: a query that was
correctly classified at Stage 1 is accepted early; a query that
was wrong at Stage 1 gets a second chance at Stage 2. The final
accuracy is at least as good as the widest stage alone, and the
average latency is lower because easy queries exit early.

### Key design decisions

1. **Per-stage M (number of tables).** Stage 2 may not need M=64.
   If N_PROJ=32 is discriminative enough, M=16 or M=32 may suffice,
   cutting the cost of escalation.

2. **Confidence threshold K.** The operating point on the
   speed/accuracy tradeoff. Higher K → fewer queries accepted at
   Stage 1 → more escalations → higher accuracy, slower average
   latency. The cross-seed overlap data provides the calibration:
   set K so that ~32% of CIFAR-10 queries (the always-right set)
   clear Stage 1.

3. **Fresh tables or shared tables?** Each stage builds its own
   tables with its own projection width. Tables are NOT shared
   across stages (different N_PROJ = different signature size =
   different bucket keys). But the training data, seed derivation,
   and calibration method are the same.

4. **Confidence metric.** Three candidates:
   - `votes_for_winner / M` — fraction of tables whose per-table
     1-NN matches the SUM winner. Simple, O(1) after the resolve
     pass.
   - `(winner_sum - runner_up_sum)` — SUM margin. Requires
     tracking the runner-up during the resolve pass.
   - `votes_for_winner - votes_for_runner_up` — vote margin.
   
   The cross-seed data was measured with per-table votes. SUM
   margin is also available (mean 13.88 on always-right).

5. **Bucket key width at N_PROJ=32.** Packed-trit signature is
   8 bytes → uint64 bucket key. Requires generalizing
   `glyph_bucket` from uint32 to uint64 keys. ~200 lines of
   plumbing (key type, hash function, lower_bound).

6. **What about the union?** Each stage builds its own candidate
   union from scratch. The Stage-1 union does NOT carry over to
   Stage 2 — the wider projection assigns different bucket keys
   to the same training prototypes, so the union membership
   changes entirely.

## Expected outcomes

### CIFAR-10

Stage 1 (N_PROJ=16): 35.32%. Handles ~32.5% of queries at
current speed (~6 ms/query).

Stage 2 (N_PROJ=32): should push into the 40-50% range based on
the scaling curve (the steep-climb regime from 16→32 trits on
CIFAR-10 should produce a large gain since we're at 35% — deep
in the growth phase).

Stage 3 (N_PROJ=64): should push further. At N_PROJ=64 pure
(no cascade), the scaling curve on MNIST was 92%. CIFAR-10 is
harder but the cascade specifically targets the hard residual.

### MNIST

Stage 1 (N_PROJ=16): 97.31%. Handles ~97% of queries.
Escalation fires on ~3% — negligible latency impact.

### Fashion-MNIST

Stage 1 (N_PROJ=16): 85.15-85.54%. Handles ~80-85% of queries.
Escalation should primarily target the upper-body cluster
(classes 0/2/4/6) where the 16-trit projection is blind.

## Infrastructure requirements

1. **uint64 bucket keys** (glyph_bucket generalization).
2. **Per-stage builder + table arrays** (tool-level orchestration).
3. **Confidence gate computation** (in the probe/resolve pass).
4. **Per-query routing decision** (if-else in the query loop).

No new library modules. No API changes to glyph_sig, glyph_multiprobe,
or glyph_resolver — they already accept any sig_bytes.

## Risks and tensions

1. **Memory.** Each stage's M tables hold N_train signatures at
   sig_bytes per prototype. Stage 2 at N_PROJ=32: 50000 × 8 × M₂
   bytes per table set. At M₂=32: 12.8 MB. Manageable on M-series
   (16+ GB) but scales linearly with N_train and M.

2. **Build time.** Projection + encode + bucket-build at N_PROJ=32
   over 50K×3072 will take ~40s (extrapolating from N_PROJ=16 at
   38s, roughly 2× for 2× wider projections). Multi-stage builds
   are independent and could be parallelized.

3. **Threshold sensitivity.** The confidence gate K determines the
   speed/accuracy tradeoff. Too low → accepts wrong answers that
   could have been corrected; too high → escalates everything,
   defeating the purpose. Need an empirical sweep of K at each
   stage.

4. **Marginal queries.** The 5.8% "swing" queries (1/3 or 2/3
   correct across seeds) have ambiguous confidence — they may be
   misrouted by any fixed threshold. These are the hardest to serve.

5. **Will N_PROJ=32 actually help on CIFAR-10?** The scaling curve
   was measured on MNIST (different dataset, different dimensionality).
   CIFAR-10 at 3072 dims may saturate the per-table discriminability
   at a different N_PROJ. This is the key empirical unknown.

## Relationship to prior work

- **Not the old dense cascade.** The pre-Axis-5 cascade tools ran
  routing inside a dense O(N) outer loop. This cascade is routing-
  native at every stage.
- **Not ensemble amplification.** The amplification experiment used
  multiple projections at the same N_PROJ and found errors don't
  decorrelate. This uses different N_PROJ — genuinely different
  resolution, not different random draws of the same resolution.
- **Closest analogue in ML literature:** early-exit networks,
  cascade classifiers (Viola-Jones), anytime prediction. The
  novelty here is that every stage is the same hash-based routing
  mechanism at a different resolution — no neural network, no
  gradient, no float.

## Summary

Dynamic N_PROJ is a resolution-adaptive routing cascade where the
lattice self-selects which queries need more resolution based on a
confidence signal it already computes. The architecture stays
self-similar and routing-native across stages. The cross-seed
overlap data provides the empirical foundation: the lattice's
vision is stable, its confidence is computable, and wider
projections provide genuinely different information.
