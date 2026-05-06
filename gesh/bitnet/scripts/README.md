# gesh/bitnet/scripts — weight conversion + reference dump

Python helpers that bridge HuggingFace's BitNet release format with the substrate's runtime expectations. These run **once** to prepare data; the C harness then reads the prepared files at runtime.

## Prerequisites

```bash
pip install safetensors transformers torch huggingface_hub numpy
```

Approximate model download size: 1.18 GB. Cached under `~/.cache/huggingface/hub/`.

## Scripts

### `inspect.py`

Lists tensor names, shapes, and dtypes from `model.safetensors`. Use this first to verify the file structure matches what `convert_weights.py` expects.

```bash
python scripts/inspect.py
```

Output: human-readable table to stdout, JSON manifest to `inspect_manifest.json`.

### `convert_weights.py`

Reads HF safetensors → emits a single substrate-format binary blob. Layout is fixed and matches `bitnet_config.h`'s expected order.

```bash
python scripts/convert_weights.py --output bitnet_weights_m4t.bin
```

**What it does per tensor type:**

| HF storage | Substrate storage | Notes |
|---|---|---|
| BitLinear weight (U8, 4-in-8) | 5-in-8 packed | 1.25× density advantage; one-time conversion at load |
| BitLinear weight scale α (BF16) | MTFP19 (int32) | Per-tensor scalar; encoding TBD per work-unit gap #6 below |
| Norm γ (BF16, [HIDDEN] or [INTERMEDIATE]) | MTFP19 (int32) per cell | Same encoding question as α |
| Embedding (BF16, [VOCAB × HIDDEN]) | MTFP19 (int32) per cell | Same encoding question |

### `dump_reference.py`

Loads BitNet via HF `transformers`, runs forward pass on a fixed prompt, dumps per-layer activations to disk for the substrate-side comparison driver.

```bash
python scripts/dump_reference.py --prompt-token 1234 --output ref_activations.npz
```

Output: `.npz` with one array per (layer_index, sublayer_name) pair. The substrate harness reads these for per-layer L2-relative-error computation.

## Newly surfaced substrate gaps (per work-unit 1 inspection of safetensors metadata)

Beyond the 4 primitives planned in SYNTHESIZE (rsqrt, RoPE, softmax, A8), inspecting the actual storage layout surfaced two more:

**Gap #5 — Scalar × vector multiply (`m4t_mtfp_vec_scale`).** Each BitLinear's matmul output must be multiplied by `α_w × s_activation / 127` to get the dequantized result. This is `dst[i] = scale × src[i]` with saturating clamp — a new primitive (NEON `vmulq_n_s32` + `clamp64`). Trivial to implement; just wasn't planned for. Add to substrate gap list.

**Gap #6 — BF16 → MTFP19 cell conversion.** Loading embeddings, norm γ vectors, and BitLinear scales requires converting bf16 floats into the substrate's MTFP19 representation. The substrate has no primitive for this. Phase 1 options:

- **(a) Convert offline in Python** — `convert_weights.py` does the bf16→MTFP19 mapping; substrate-side just mmaps int32 buffers. Simplest, but requires picking a per-tensor scale factor in Python (the MTFP19 block exponent equivalent).
- **(b) Convert at C-side load time** — keep bf16 bit patterns in the blob; convert during the harness's load step. More flexible but requires a load-time conversion routine (which is itself a primitive shape).
- **(c) Defer entirely; carry bf16 through the pipeline as opaque bytes** — won't actually work because activations need to be in MTFP19 to feed the substrate's matmul kernels.

For Phase 1: **option (a)**. `convert_weights.py` chooses a per-tensor power-of-3 scale that maps the bf16 magnitude range into MTFP19's int32 range with reasonable headroom. The C harness mmaps int32 buffers directly. Trade-off: precision loss at conversion time (acceptable for the per-layer-tolerance gate), but no runtime conversion overhead.

Both gaps documented in CHANGELOG via the next commit.

## Cross-references

- LMM cycle: `journal/bitnet_phase1_*`
- Architecture source: `huggingface.co/microsoft/bitnet-b1.58-2B-4T`
- C harness: `gesh/bitnet/bitnet_harness.c`
- Substrate config: `gesh/bitnet/bitnet_config.h`
