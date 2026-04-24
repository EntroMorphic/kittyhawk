# train/ — routed autodiff MVP

Consumer-layer training scaffolding for Glyph's routing-first ternary substrate, per **NORTH_STAR §13** ("training artifacts live in the consumer"). Built in pure C, NEON-optimized for Apple M-series. **Not linked into `libm4t` or `libglyph`** — opt-in via `-DGLYPH_BUILD_TRAIN=ON` (default ON).

## What this is (and is not)

This subtree answers one question: **what does autodiff look like when the forward pass is routing-native ternary, not dense float?** It is the minimum viable demonstration of a backward pass through the substrate's routing primitives.

It is **not**:
- A general-purpose autograd engine. No op tape, no dynamic graph. Hand-coded forward/backward pairs.
- A drop-in for PyTorch. Zero Python, zero float in the hot path.
- A production trainer. MVP-scope primitives; full trainers live in downstream cycles.

## Build surface

```
train/
  src/
    backward_linear.{c,h}   scalar ternary linear forward + backward_dX + backward_dW
    backward_routed.{c,h}   per-token top-k selection, signed dispatch, backward (STE)
    requantize.{c,h}        hysteresis-aware float-latent → int8-trit re-quantization
  tests/
    test_gradient_linear.c    numerical gradient check, tlinear dX/dW vs finite diffs
    test_gradient_routed.c    routed dispatch dX/dW on selected slots (STE-linear portion)
    test_toy_convergence.c    2-class linear-separation convergence + STE behavior monitor
    test_toy_10class.c        10-class harder toy documenting expert-collapse finding
    test_edge_cases.c         k>T, all-zero scores, M=0, k=0, empty requant, zero-hysteresis
```

All five tests ship under the default build and are registered with `ctest`. 14/14 tests pass repo-wide when `GLYPH_BUILD_TRAIN=ON`.

## Primitives

**`tlinear_forward` / `_backward_dX` / `_backward_dW`** — dense ternary linear layer. Forward: `Y = X·W` with `W ∈ {−1, 0, +1}`. Backward receives `dY`, produces `dX` (for upstream layers) and `dW_latent` (accumulated into a float mirror of `W`). Scalar reference; NEON port deferred (see `journal/routed_autodiff_closeout.md`).

**`rroute_forward_select` / `_forward_dispatch` / `_backward_dX` / `_backward_dW`** — routed dispatch. Forward: compute `scores[m,t] = Σ_h X[m,h]·U[t,h]`, pick top-k by |score|, then accumulate per-tile matmul contributions into `Y[m,n]`. Backward is a **straight-through estimator (STE)** through top-k: gradient flows only to the k selected tiles; unselected tiles receive zero.

MVP routing choices (documented in `train/src/backward_routed.c`):
- `U` (gating weights) frozen at random ternary. Training targets only `W` (tile weights).
- Dispatch ignores the per-tile score sign to avoid randomization from random `U` during SGD from random init.
- Selection-only routing; sign-routing is a follow-up once tiles are discriminative.

**`requantize_hysteresis`** — sticky-trit re-quantization. Each trit requires `|W_latent|` to exceed τ·(1+h) to enter a ±1 state and fall below τ·(1−h) to leave it. Fixes the first-attempt oscillation pattern (100% → 1.6% → 100% between epochs) where latents near the threshold flipped on every SGD step.

## Known findings from the MVP cycle

Documented in `journal/routed_autodiff_{raw,nodes,reflect,synthesize,closeout}.md`:

1. **Gradient checks pass** at 1.4e-4 (dW) and 1.6e-4 (dX) relative-or-absolute tolerance.
2. **2-class toy converges**: 96.50% mean / 2.79pp σ across 5 seeds; 95% single-seed.
3. **10-class toy reveals expert collapse**: random-gate selection-only routing at 34% test accuracy vs plain ternary linear at 91%. Frozen `U` + random init causes every tile to see every class, so no tile can specialize. This is the MVP's architectural ceiling — fixing it requires learned routing (soft / differentiable top-k), load-balancing loss, or class-aware gate initialization. Out of MVP scope; queued for a follow-up cycle.
4. **STE behavior monitor** (test_toy_convergence): runtime assertion that per-epoch selection-flip count is 0 under frozen `U`. Guards against regressions that accidentally couple routing to trainable state.

## Hyperparameters (derivations in `tests/test_toy_convergence.c`)

| Parameter | Value | Why |
|---|---|---|
| `REQUANT_DENSITY` | 0.33 | Max-entropy prior for trits over {−1, 0, +1}. τ at 67th percentile of |W_latent| realizes it on deployed weights. |
| `REQUANT_HYSTERESIS` | 0.10 | Dead-zone half-width ≈ 0.1·τ ≈ 5e-3; one order above typical per-cycle latent drift (3e-4), below percentile spacing. |
| `LR` | 5e-4 | Cumulative drift per requant cycle ≈ 3e-4 << τ ≈ 0.05 → percentiles are stable across epochs. |
| `W_latent` init scale | 0.05·N(0,1) | Half-normal |·| mean ≈ 0.04 < τ ≈ 0.049 → W quantizes to all-zero at init; trits must be earned by gradient flow. |
| STE clip | |W_latent| < 1.0 | Mirrors BinaryConnect/XNOR-Net hard-tanh straight-through. Saturated latents add no signal and block re-quantization. |

## Running the tests

```bash
# Build and test (default)
cmake -S . -B build
cmake --build build -j
ctest --test-dir build -R '^train_'

# Single-seed convergence smoke (faster, ~1s)
./build/train/test_toy_convergence --single

# Multi-seed stability (5 seeds, ~5s)
./build/train/test_toy_convergence

# 10-class toy with expert collapse diagnostic
./build/train/test_toy_10class
```

## Discipline checks

- **§12 (no binary float in compute)** — maintained inside libtrain's kernels. Float appears only in `W_latent` accumulators and finite-difference gradient checks, both explicitly sanctioned training-only sites.
- **§13 (training in consumer)** — maintained. libtrain.a is not linked by libm4t or libglyph.
- **Routing discipline** — forward dispatch is selection-only per MVP design; backward is STE through top-k.

## Next (queued)

- `routed_go` trainer cycle (task #41): learned routing on top of `hamming_norm` substrate distance. Applies the expert-collapse lessons from this cycle (needs learned `U`, load-balance loss, or class-centroid init). Gated on hamming_norm's image-pipeline measurement (task #40).
- NEON port of backward kernels (deferred — kernels are correct but slow; scalar is fine until scale matters).
