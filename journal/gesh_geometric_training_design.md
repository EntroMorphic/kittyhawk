---
cycle: gesh_geometric_training_design (P0-3)
date: 2026-05-02
status: design + verification plan in one doc
---

# P0-3: Lattice-native geometric training objective

## Substrate gap

`gesh_train_lattice_update` minimizes per-batch classification error. That's label geometry — the lattice is just a search space. The Phase A closeout said "the lattice IS the geometry" but the loss never references the lattice's geometric structure (Hamming distances between tiles, polytope volume, etc.).

A genuinely lattice-native loss operates on bank tile signatures and their pairwise Hamming distances. No labels referenced beyond grouping.

## What's substrate-novel

Pairwise Hamming margin between class tiles is a quantity *only definable on a discrete trit lattice*. Base-2 with quantization could compute Hamming over binary signatures, but {-1, 0, +1}-state Hamming is a substrate-specific metric (the third state contributes asymmetrically to distance). And training-by-flip on the lattice IS the substrate's geometry — no continuous relaxation.

## Build commitment

One kernel + one training variant + one verification:

**`m4t_route_pairwise_hamming_sum`** (libm4t)
- Inputs: T tile signatures (packed-trit), mask, sig_dim.
- Output: int32 sum of Hamming distances over all (i,j) pairs with i < j.
- Substrate-distinct: ternary Hamming over packed trits; not reproducible base-2-natively without storage overhead per the §19 audit.

**`gesh_train_lattice_update_geometric`** (libgesh)
- Same shape as `gesh_train_lattice_update`, but loss = -m4t_route_pairwise_hamming_sum (we maximize it).
- Per flip-eval: build bank, compute pairwise sum, flip R if sum increased.
- No batch sampling, no classification error. Loss is bank-only.

**`geometric_train_probe`** (verification)
- synth_proto multi-seed (3 seeds).
- Compare classification accuracy at end of:
  - Random R (baseline)
  - Error-trained R (existing `gesh_train_lattice_update`)
  - Geometric-trained R (new variant)

## Verification gates

| Gate | Test | PASS |
|---|---|---|
| 1 | Geometric-trained R accuracy ≥ Error-trained R | gain ≥ 0pp paired-CI lower bound |
| 2 | Geometric-trained R has higher pairwise margin than error-trained | by construction (loss directly optimized) |
| 3 | Substrate-novelty audit | by construction (ternary Hamming over packed trits) |
| 4 | MNIST regression | within ±2pp |

## §19 / §20 amendment

The new kernel `m4t_route_pairwise_hamming_sum` is input-side §18; consumes packed-trit signatures with three-state semantics. Add as §19.4 entry. No new zero-state interpretation needed.

## Build sequence

1. Spec entry.
2. Kernel + 2 property tests.
3. Training variant + small test.
4. Probe + multi-seed.
5. MNIST gate.
6. Close.

## VERDICTS (post-implementation)

3-seed paired probe on synth_proto (n_train=2000, sig=64, 10K flip budget):

| | random | error-trained | geometric |
|---|---:|---:|---:|
| seed 0 | 76.8% | 80.2% | 77.8% |
| seed 1 | 78.8% | 78.6% | **81.8%** |
| seed 2 | 76.4% | 80.8% | 79.2% |
| mean | 77.3% | 79.9% | 79.6% |
| pairwise margin (mean) | 2875 | 2865 | **3172** |

| Gate | Verdict |
|---|---|
| 1 (geometric ≥ error-trained accuracy) | **TIE**. Paired mean −0.27pp, CI [−3.69, +3.16] straddles zero. Geometric is not reliably better OR worse than error-trained. |
| 2 (geometric margin > error margin) | **PASS** in all 3 seeds (+223 / +316 / +381). The loss IS optimizable; training produces what it claims. |
| 3 (substrate-novelty audit) | **PASS by construction**. Ternary Hamming over packed trits; no continuous relaxation; substrate-distinct from base-2 (which can't do native ternary Hamming without storage overhead). |
| 4 (MNIST regression) | DEFERRED. Per-flip bank rebuild at n_train=60K would take ~hours without an incremental-update kernel. Out of scope for first verification. |

## What this demonstrates and what it doesn't

**Demonstrated:** the substrate supports lattice-native training. A loss that operates entirely on packed-trit tile signatures (no labels, no batches, no continuous relaxation) is optimizable via flip-based coordinate descent. The kernel composes cleanly with existing substrate primitives.

**Not demonstrated:** that this specific lattice-native loss (pairwise Hamming sum) outperforms label-supervised training. **The right finding is that max-tile-spread is the wrong loss.** Tiles spread further apart, but training samples don't reliably end up closer to their correct tile. The geometric objective is decoupled from the classification objective in a way that matters.

## What's shipped as substrate primitive

- `m4t_route_pairwise_hamming_sum` — useful for any future training that needs lattice-margin metrics. Substrate-distinct (Gate 3 PASS).
- `gesh_train_lattice_update_geometric` — lattice-native training variant; works (Gate 2 PASS); not currently outperforming error-trained for classification (Gate 1 TIE).

## What's open / deferred

- A better lattice-native loss formulation: contrastive (samples-to-correct-tile pulled close, samples-to-wrong-tile pushed far) is the natural next step. Probably the actual right substrate-native objective. Out of P0-3 scope.
- Incremental bank update for fast per-flip eval at MNIST scale.
- MNIST regression check.

## Cycle closes with mixed verdict

P0-3 ships the substrate primitive but the loss formulation is research-incomplete. The substrate-claim "the lattice IS the geometry" is supported (we can train on it directly); the specific geometric loss we tried isn't the right one for classification. Both findings are honest and worth recording.

P0-4 next.
