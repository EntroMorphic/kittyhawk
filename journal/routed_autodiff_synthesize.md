---
date: 2026-04-23
scope: LMM cycle — routed autodiff in NEON C
phase: SYNTHESIZE
---

# SYNTHESIZE: routed autodiff MVP — one-layer ternary encoder trainer

## The reframe

Routed autodiff is **not** "autodiff applied to a routing network." It's a different primitive class: (signature, decision, sparse-dispatch) triples whose backward is a **sparse gradient scatter conditioned on the forward decision**, not a tensor reshuffle. That distinction decides where the gradients go, which kernels are hot, and whether the substrate's SDOT/TBL/VCNT advantage carries to the backward pass.

The user's ask "routed, not dense" is the NORTH_STAR alignment: the same routing shape that makes forward substrate-native should make backward substrate-native. A routed backward accumulates gradient INTO the k selected tiles only; unselected tiles receive zero. Dense backward would corrupt the 75% of tiles that didn't fire with irrelevant signal.

## Decision — MVP first, not general engine

**Build a minimum viable routed ternary encoder trainer.** One forward+backward pair, hand-coded, integer-result output, NEON-optimized hot loops. **1-2 weeks.** Output answers the open S7 question: can a learned routed ternary encoder on Glyph's substrate beat direct quantization on CIFAR-10?

If the MVP validates the thesis, a follow-up cycle generalizes — adds more ops, maybe a tape, multi-layer. Not this cycle.

## Success criteria

**MVP-level:**
- [ ] `train/routed_trainer.c` compiles under `-Werror`.
- [ ] All new backward kernels pass numerical gradient check (finite-difference vs analytical) within `1e-3` relative error on a toy problem.
- [ ] Trainer converges on a synthetic 2-class linear-separation toy and produces signatures that classify correctly.
- [ ] Trainer produces packed-trit signature files (`train_sigs.bin`, `test_sigs.bin`) in the same format `glyph_sig_quantize` emits.
- [ ] `direct_lsh --sigs_from_file` loads them and runs normal Selective pipeline.

**Cycle-level (gate on CIFAR):**
- [ ] ≥50% Selective on CIFAR-10 (+2pp over direct MS4+R4 48.05%) — **routed autodiff materially improves signature quality.** Thesis-validated.
- [ ] 48–49% Selective — **inconclusive.** Encoder roughly matches direct; architecture may be too small.
- [ ] <48% Selective — **training is broken OR architecture is too weak.** Diagnose or abandon.

**Thesis-level:**
- Regardless of CIFAR number, the cycle produces: (a) routed backward primitive set, validated; (b) empirical data on whether STE through top-k works; (c) template for future routed-autograd work.

## Implementation specification

### Directory layout

```
train/
  CMakeLists.txt                       — new, sibling to m4t/
  src/
    routed_trainer.c                   — main training loop
    routed_backward.c + .h             — backward NEON kernels
    ste.c + .h                         — straight-through-estimator primitives
    toy_dataset.c + .h                 — synthetic 2-class data for gradient check
  tests/
    test_gradient_check.c              — numerical gradient check on every backward
    test_toy_convergence.c             — synthetic task convergence test
  docs/
    TRAIN_SUBSTRATE.md                 — spec for routed backward primitives
```

**Not on libm4t's build path.** Trainer links libm4t (forward) and libm for float math; produces binary artifacts consumed by `direct_lsh`. Substrate discipline: training is a consumer concern per NORTH_STAR §13.

### Architecture (MVP)

- **Input:** MTFP image features (intensity + gradients) from `glyph_dataset`. Same as `direct_lsh`.
- **Signature derivation:** reuse `m4t_route_threshold_extract` on a learned projection of input. Produces query signature for routing.
- **Tile bank:** T=16 ternary weight tiles, each of shape [H=256, D_out_per_tile=1024]. Total parameters: 16 × 256 × 1024 trits = 4 MB packed.
- **Routing:** `m4t_route_distance_batch` + `m4t_route_topk_abs` over T. k=4 tiles selected per query.
- **Per-tile compute:** `m4t_mtfp_ternary_matmul_bt` produces partial MTFP output per selected tile.
- **Accumulation:** `m4t_route_apply_signed`.
- **Output signature:** sign-at-tau quantize accumulated output → D_out trits per image. D_out = T × D_out_per_tile = 16384 (rounded down to match target ~11000 via sub-dim selection if needed).

Wait — with T=16 and D_out_per_tile=1024, the accumulated output is 1024-dim (all tiles output into the same 1024 slots, with sign weighting). To get 11000-dim signatures, restructure:

**Revised architecture:** tiles are INPUT-SIDE, not output-side. Input D_in=3072 divided across T=16 groups of 192 dims each. Tiles route over input subgroups. Output is flat D_out=11000.

Actually this is getting into architecture design that belongs in the implementation phase, not the spec phase. **Lock the key numbers:** T=16, k=4, D_out ≈ 11000 (match MS4+R4). Detail architecture during implementation; the cycle's gate is CIFAR Selective accuracy, not architecture shape.

### Training hyperparameters (for replicability)

- Dataset: CIFAR-10 (consistent with direct_lsh's flags).
- Preprocessing: same as direct_lsh MS4 (normalize, gradients, multi-scale).
- Loss: auxiliary classifier head (linear layer on PRE-sign activations → 10-way softmax → cross-entropy).
- Optimizer: SGD with momentum 0.9 (simpler than Adam to implement in C), lr 0.01, cosine decay.
- Batch: 128. Epochs: 20.
- STE clipping: `dW_latent *= (|W_latent| < 1.0)` mask to prevent unbounded latents.
- Latent quantization cadence: re-pack tile_trit from W_latent every 10 steps.

### Backward kernel list

1. **`sdot_backward_dX`** — given float dY and packed-trit W, compute float dX. Hot loop; NEON-native.
2. **`sdot_backward_dW`** — given float dY and float X restricted to selected tiles, accumulate into W_latent. Float outer-product into float array.
3. **`routed_dispatch_backward`** — scatter float dY into dX_tile slots by sign for selected tiles; zero for unselected.
4. **`ste_backward_threshold`** — clipped-identity backward for sign-at-tau.
5. **`ste_backward_topk`** — identity on selected positions, zero elsewhere.
6. **`ste_backward_sign_clip`** — identity on |W_latent| ≤ 1, zero otherwise.

Scalar implementations first. Gradient check all. Then NEON hot paths (#1, #2, #3).

### Gradient check contract

For every backward kernel, numerically verify:
- Pick random input X, random ternary W, random dY.
- Compute analytical dX and dW via backward.
- For a sample of elements: perturb by ε=1e-3, recompute forward, finite-difference gradient.
- Assert max relative error < 1e-3.

One test per kernel. Fails immediately if wrong.

### Convergence test

Generate synthetic 2-class linearly separable data (1000 samples, 64 dim, two Gaussian clusters). Train the single-layer trainer on it. Verify:
- Loss decreases monotonically.
- Final classification accuracy ≥ 95%.
- Learned tile weights concentrate on the discriminative dim(s).

If this passes, scale to CIFAR.

### `direct_lsh --sigs_from_file` loader

Add to `tools/direct_lsh.c`:
```
--sigs_from_file TRAIN_PATH TEST_PATH
    Load pre-computed packed-trit signatures from disk; skip
    glyph_sig_quantize and all its predecessors (normalize, gradients,
    multi_scale, region_tau, fstat). File format: header (magic, n_images,
    n_trits), then n_images × ceil(n_trits/4) bytes packed-trit data.
```

### Reuse `--brute_1nn` as the control

Already exists from S7. Compare S7-trained signatures under both Glyph Selective and brute 1-NN. Populates the interpretation grid cleanly.

## Handling the major tensions

- **T1 (MVP vs engine):** MVP, this cycle. Engine follow-up only if MVP validates.
- **T2 (float vs int latents):** float. Sanctioned per §4/§12 scaffolding.
- **T3 (STE vs soft routing):** STE with clipping. Empirically validated or not by MVP.
- **T4 (tree location):** new `train/` directory, sibling to m4t/.
- **T5 (scalar first, NEON after):** yes. Gradient-checked scalar precedes NEON.
- **T6 (testing):** gradient check + convergence test before CIFAR.

## Quality check

- **Executable by someone else?** Yes. Named directory, named file list, named kernels, named tests, named gate. Specific enough to scope and track.
- **Addresses all tensions?** Six resolved, all with explicit paths.
- **Simpler than starting point?** RAW had three architecture choices, four STE options, two tree locations, five loss variants. Synthesis reduces to: one architecture (one-layer routed), one STE (clipped), one location (train/), one loss (aux classifier), one gate (CIFAR ≥ 50%).
- **Surprised?** Yes — entered thinking "build autodiff engine." Left with "build ONE forward+backward, validate the pattern, escalate only if it works." MVP discipline trumps engine ambition.

## Timeline estimate

Week 1:
- Day 1-2: directory scaffolding, CMakeLists, basic `train/` layout.
- Day 2-3: scalar forward+backward for one small ternary linear kernel; gradient check it.
- Day 4-5: extend to routed ternary (tile dispatch, topk); gradient check it.
- Day 6-7: convergence test on 2-class toy.

Week 2:
- Day 1-2: NEON backward kernels, maintain scalar fallback, test-check.
- Day 3-4: scale to CIFAR-sized config; `--sigs_from_file` wiring in direct_lsh.
- Day 5: CIFAR training run.
- Day 6-7: measurement, writeup.

If at end of week 1 the convergence test hasn't passed, halt and reassess the STE and architecture choices before proceeding.

## Immediate next actions (need user sign-off)

This is a 1-2 week commit. Per "confirm large moves" in the user's working style, get explicit go-ahead before starting week 1.

Specific confirmations needed:
1. Scope: MVP, not general engine. ✓
2. Latents float during training, trit deployed. Float sanctioned under §4/§12.
3. Tree location: new `train/` directory.
4. Gate: ≥50% CIFAR Selective = thesis-validated; <48% = diagnose or abandon.
5. 1-2 week commit ok? Or timebox shorter?

## What this cycle produces regardless of MVP success

Even if the MVP's CIFAR number disappoints:

1. **Routed backward primitive set exists.** First autodiff kernels in the substrate. Future consumers can use.
2. **STE-through-routing is empirically tested** (not just assumed from literature).
3. **`--sigs_from_file` pipe** unifies trainer and direct_lsh; any future encoder (gradient-trained, hand-designed, external) plugs in.
4. **Gradient-check test pattern** codified for future substrate extensions.
5. **Baseline for a bigger engine.** If thesis-level work wants a general routed autograd engine, this MVP is the seed.
