# Native C/NEON Engine

Zero-dependency C implementation of the trix-z routing and computation primitives. Targets ARM NEON (Apple Silicon, Jetson AGX Thor) with scalar fallbacks for other platforms.

## Source Files

### Core Atoms

**`src/trix_atoms.h`** / **`src/trix_atoms.c`**

29 atomic operations covering vector, reduce, matrix, activation, update, loss, and classification. NEON-optimized with Accelerate (cblas) on Apple. These are the building blocks for all higher-level routines.

### Routed FFN (Training Engine)

**`src/trix_multitrit.h`** / **`src/trix_multitrit.c`**

Full training engine: forward + backward + AdamW. Supports global and signature routing with MTFP21 quantization via STE. This is the **active GPT-2 training path** — called from Python via `src/trix/native_multitrit.py`.

Key features:
- LayerNorm → router logits → argmax dispatch → per-tile FFN → residual
- Full backward pass with per-tile gradient accumulation
- Dropout with seeded PRNG for reproducibility
- Aux loss (Switch-style balance) computed in C

### Ternary Routing (Inference Engine)

**`src/trix_ternary_route.h`** / **`src/trix_ternary_route.c`**

Inference and training engine with weight-derived signatures and Hamming/dot-product routing. Produced **98.22% MNIST** (D=128, T=4, seed=42). Uses packed ternary expert tiles via `trix_ternary_matvec`.

**Backward pass status:** A full backward pass exists in committed code (`930a535`, `f6cc347`) with per-tile gradient routing, sign modulation, MTFP21 quantized weights, LayerNorm backward, and residual gradients. Uncommitted local edits replaced it with stubs during refactoring — these stubs should be reverted or the working backward pass restored. See `git show f6cc347:native/src/trix_ternary_route.c` for the working version.

**MNIST ablation note (7 April 2026):** The trained ternary FFN contributes +0.03-0.11% over a projection+head baseline that already reaches 97.9-98.1%. The routing architecture is real and trainable, but on MNIST the projection does the heavy lifting. See `docs/RED_TEAM_20260407.md` Attack 2 for full ablation results.

Key features:
- Weight-derived signatures via `trix_ternary_route_update_signatures()` — column-sum + mean-subtraction + sign
- Hamming routing via `trix_popcount_dist_neon()` (XOR+POPCNT)
- Packed ternary expert tiles (2-bit weights, int8 activations)
- `dispatch_apply` batch parallelism on Apple

### Ternary Matvec

**`src/trix_ternary_matvec.h`** / **`src/trix_ternary_matvec.c`**

Multiply-free matrix-vector product for ternary weights. Uses NEON VLD4 deinterleaving + SDOT for 64 dimensions per loop iteration. Standalone value for any ternary inference on ARM.

### Transformer Block

**`src/trix_transformer.h`** / **`src/trix_transformer.c`**

Full transformer block: multi-head attention + routed FFN (via `trix_multitrit`). Forward + backward + AdamW. Used by `src/trix/native_transformer.py`.

### CUDA Ports

**`src/trix_ternary_route_cuda.h`** / **`src/trix_ternary_route_cuda.cu`**
**`src/trix_multitrit_cuda.h`** / **`src/trix_multitrit_cuda.cu`**

CUDA ports for Jetson AGX Thor. The ternary route CUDA port was found to have diverged from the NEON reference (April 6 audit) and needs re-verification.

### Utilities

**`src/trix_rng.h`** — Shared xoshiro128+ PRNG (consolidated from per-file duplicates).

**`src/trix_types.h`** — Forward declarations for opaque struct types.

**`src/trix_neon.h`** — NEON intrinsic helpers.

**`src/ternary_pack.h`** — Ternary packing utilities.

**`src/trix_ternary_dot_chip.h`** — Chip-level ternary dot product.

**`src/trix_loop.c`** — Training loop driver.

**`src/trix_train.c`** — Standalone training entry point.

## Two Routing Codepaths

This is the most important architectural distinction in the native engine:

| | `trix_multitrit.c` | `trix_ternary_route.c` |
|---|---|---|
| **Purpose** | Training (GPT-2 path) | Training + Inference (MNIST path) |
| **Backward pass** | Full | Full in committed code; stubbed in uncommitted local edits |
| **Routing** | Global (learned Linear) or signature (learned params) | Weight-derived signatures (from W1 column sums) |
| **Expert compute** | Float32 matmul (with MTFP21 STE) | Packed ternary matvec (int8 activations) |
| **Best result** | 692.05 PPL (GPT-2 d=64, 10M tokens, signature) | 98.22% MNIST (D=128, T=4, k-of-T) |
| **Python binding** | `src/trix/native_multitrit.py` | (direct ctypes in tests) |

The k-of-T ternary routing algorithm in `trix_ternary_route.c` has **never been run on GPT-2** because it lacks a backward pass. Bridging this gap is the single most important next experiment.

## Build

```bash
cd native && mkdir -p build && cd build
cmake .. && make -j
ctest  # 7 test targets
```

Requires: C11 compiler, ARM NEON headers (or scalar fallback). Optional: Accelerate framework (Apple), CUDA toolkit (Jetson).
