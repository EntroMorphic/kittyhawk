#!/usr/bin/env python3
"""
convert_weights.py — emit substrate-format binary blob from BitNet's HF release.

Per the LMM cycle in journal/bitnet_phase1_*. Run once per checkpoint.
Output is consumed by the C harness via mmap.

Conversion summary (per gap analysis in scripts/README.md):
  HF source                    →  Substrate target
  -----------                     ----------------
  BitLinear weight (U8 4-in-8) →  5-in-8 packed (1.25× density)
  BitLinear scale α (BF16)     →  MTFP19 mantissa + per-tensor block exp
  Norm γ (BF16)                →  MTFP19 mantissa + per-tensor block exp
  Embedding (BF16)             →  MTFP19 mantissa + per-tensor block exp

Per-tensor block exponent: chosen automatically as the largest k such that
3^k × max(|values|) ≤ MTFP19_MAX. Maximizes precision retention.

Output blob layout (single self-describing file; C side uses no JSON):
  [magic: 4 bytes "M4T1"]
  [version: uint32]
  [lm_head_tied: uint32 (1 if lm_head reuses embedding, 0 if separate)]
  [n_tensors: uint32]
  [block_exps: int32 × n_tensors]      ; one per tensor, in fixed order
  [offsets:    uint64 × n_tensors]     ; from start of blob
  [sizes:      uint64 × n_tensors]
  [tensor_data: contiguous, in fixed order]

The fixed order matches bitnet_config.h's expected layout:
  embedding, layer0_w_q, layer0_w_k, layer0_w_v, layer0_w_o,
  layer0_w_gate, layer0_w_up, layer0_w_down,
  layer0_alpha_q, alpha_k, alpha_v, alpha_o, alpha_gate, alpha_up, alpha_down,
  layer0_gamma_input_norm, gamma_post_attn_norm, gamma_attn_sub_norm, gamma_ffn_sub_norm,
  ... layers 1..29 same shape ...
  final_norm.gamma,
  [lm_head — only if lm_head_tied == 0]

Total tensor count: 1 + 30*18 + 1 + (0 or 1) = 542 or 543.

Also writes a human-readable JSON manifest (`*_manifest.json`) for debugging.
"""

import argparse
import json
import os
import struct
import sys
from collections import OrderedDict

try:
    import numpy as np
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
except ImportError as e:
    print(f"[error] missing dep: {e}\n"
          f"Install with: pip install numpy safetensors huggingface_hub",
          file=sys.stderr)
    sys.exit(1)

MAGIC = b"M4T1"
VERSION = 1

# Substrate constants — match m4t/src/m4t_types.h
MTFP19_MAX = 581_130_733
LOG_3 = np.log(3.0)

# Model-fixed constants — match gesh/bitnet/bitnet_config.h
HIDDEN_SIZE        = 2560
INTERMEDIATE_SIZE  = 6912
NUM_LAYERS         = 30
NUM_KV_HEADS       = 5
HEAD_DIM           = 128
KV_PROJ_DIM        = NUM_KV_HEADS * HEAD_DIM
VOCAB_SIZE         = 128_256

MODEL_REPO = "microsoft/bitnet-b1.58-2B-4T"
MODEL_FILE = "model.safetensors"


def fp_to_mtfp19(values: np.ndarray):
    """
    Convert FP array (any shape) to (int32 mantissas, block_exp).
    value ≈ mantissa × 3^(-block_exp).
    Picks largest k such that 3^k × max(|x|) ≤ MTFP19_MAX.
    """
    arr = np.asarray(values, dtype=np.float64)
    max_abs = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
    if max_abs == 0.0:
        return np.zeros(arr.shape, dtype=np.int32), 0
    k = int(np.floor(np.log(MTFP19_MAX / max_abs) / LOG_3))
    scale = 3.0 ** k
    mantissas = np.round(arr * scale).astype(np.int32)
    np.clip(mantissas, -MTFP19_MAX, MTFP19_MAX, out=mantissas)
    return mantissas, k


def unpack_4in8_ternary(packed_u8: np.ndarray) -> np.ndarray:
    """HF's 4-in-8: byte = trit_0 + trit_1<<2 + trit_2<<4 + trit_3<<6,
    where trit_code: 00→0, 01→+1, 10→-1, 11→reserved."""
    arr = np.asarray(packed_u8, dtype=np.uint8).flatten()
    n = arr.size * 4
    out = np.empty(n, dtype=np.int8)
    for i in range(4):
        codes = (arr >> (2 * i)) & 0x3
        out[i::4] = np.where(codes == 1, 1,
                              np.where(codes == 2, -1, 0)).astype(np.int8)
    return out


def pack_5in8_ternary(trits: np.ndarray) -> np.ndarray:
    """Substrate 5-in-8: byte = u_0 + 3·u_1 + 9·u_2 + 27·u_3 + 81·u_4,
    where u: -1→2, 0→0, +1→1. Trailing trits zero-pad."""
    arr = np.asarray(trits, dtype=np.int8)
    u = np.where(arr == 1, 1, np.where(arr == -1, 2, 0)).astype(np.uint8)
    n = u.size
    nb = (n + 4) // 5
    out = np.zeros(nb, dtype=np.uint8)
    pow3 = [1, 3, 9, 27, 81]
    for byte_idx in range(nb):
        start = byte_idx * 5
        end = min(start + 5, n)
        v = 0
        for d in range(end - start):
            v += int(u[start + d]) * pow3[d]
        out[byte_idx] = v
    return out


def repack_4in8_to_5in8(packed_4in8: np.ndarray, n_trits: int) -> np.ndarray:
    trits = unpack_4in8_ternary(packed_4in8)[:n_trits]
    return pack_5in8_ternary(trits)


def unpack_hf_4in8_weight(packed: np.ndarray, out_features: int, in_features: int) -> np.ndarray:
    """HF's BitNet packing: shape (out_features/4, in_features), uint8.
    Each byte stores 4 trits along the OUT axis: byte[op, i] → trits at
    out positions op*4 + slot for slot ∈ {0,1,2,3}.
    Trit codes: 00→0, 01→+1, 10→−1.
    Returns the unpacked (out_features, in_features) int8 matrix."""
    op = out_features // 4
    if packed.shape != (op, in_features):
        raise ValueError(f"expected packed shape ({op}, {in_features}), got {packed.shape}")
    out = np.empty((out_features, in_features), dtype=np.int8)
    for slot in range(4):
        codes = (packed >> (2 * slot)) & 0x3
        decoded = np.where(codes == 1, 1,
                           np.where(codes == 2, -1, 0)).astype(np.int8)
        out[slot::4, :] = decoded
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local-path", default=None,
        help="Path to local model.safetensors. Default: HF cache download.")
    parser.add_argument("--output", default="bitnet_weights_m4t.bin",
        help="Output binary blob.")
    parser.add_argument("--manifest", default=None,
        help="Output JSON manifest (default: <output>.manifest.json).")
    parser.add_argument("--layers", type=str, default="all",
        help="Layer subset: 'all' or 'N' (= layers 0..N-1) for partial "
             "conversion. Default: all 30.")
    args = parser.parse_args()

    if args.layers == "all":
        layer_count = NUM_LAYERS
    else:
        layer_count = int(args.layers)
        if not (0 < layer_count <= NUM_LAYERS):
            print(f"[error] --layers must be 1..{NUM_LAYERS}", file=sys.stderr); sys.exit(1)

    if args.manifest is None:
        args.manifest = args.output + ".manifest.json"

    # Resolve model file.
    if args.local_path:
        path = args.local_path
        if not os.path.isfile(path):
            print(f"[error] file not found: {path}", file=sys.stderr); sys.exit(1)
    else:
        print(f"[info] downloading {MODEL_FILE} from {MODEL_REPO} (HF cache; ~1.18 GB)",
              file=sys.stderr)
        path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)

    print(f"[info] reading: {path}", file=sys.stderr)

    # Pre-compute all tensors in memory (fits in ~2 GB; OK on dev machines).
    tensor_records = []  # list of (name, bytes, block_exp)

    with safe_open(path, framework="pt") as sf:
        keys = sf.keys()

        # 1. Embedding.
        emb = sf.get_tensor("model.embed_tokens.weight").float().numpy()
        emb_mtfp, emb_k = fp_to_mtfp19(emb)
        tensor_records.append(("embedding", emb_mtfp.tobytes(), emb_k))
        print(f"[ok] embedding: shape={emb_mtfp.shape} block_exp={emb_k}", file=sys.stderr)

        # 2. Per-layer.
        for layer in range(layer_count):
            base = f"model.layers.{layer}"

            # 7 BitLinear weights.
            for proj_name, n_in, n_out in [
                ("self_attn.q_proj",  HIDDEN_SIZE,        HIDDEN_SIZE),
                ("self_attn.k_proj",  HIDDEN_SIZE,        KV_PROJ_DIM),
                ("self_attn.v_proj",  HIDDEN_SIZE,        KV_PROJ_DIM),
                ("self_attn.o_proj",  HIDDEN_SIZE,        HIDDEN_SIZE),
                ("mlp.gate_proj",     HIDDEN_SIZE,        INTERMEDIATE_SIZE),
                ("mlp.up_proj",       HIDDEN_SIZE,        INTERMEDIATE_SIZE),
                ("mlp.down_proj",     INTERMEDIATE_SIZE,  HIDDEN_SIZE),
            ]:
                w_4in8 = sf.get_tensor(f"{base}.{proj_name}.weight").numpy()
                # HF packing: shape (out/4, in) with 4 trits-per-byte along
                # the OUT axis. Unpack to logical (out, in), then re-pack
                # 5-in-8 along the IN axis (substrate's expected layout).
                w_logical = unpack_hf_4in8_weight(w_4in8, n_out, n_in)
                rows_5in8 = [pack_5in8_ternary(w_logical[r]) for r in range(n_out)]
                w_5in8 = np.concatenate(rows_5in8).astype(np.uint8)
                tensor_records.append(
                    (f"layer{layer}.{proj_name}.weight", w_5in8.tobytes(), 0))

            # 7 BitLinear scales α.
            for proj_name in [
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            ]:
                s_name = f"{base}.{proj_name}.weight_scale"
                if s_name in keys:
                    a = sf.get_tensor(s_name).float().numpy().reshape(-1)
                    a_mtfp, a_k = fp_to_mtfp19(a)
                    tensor_records.append(
                        (f"layer{layer}.{proj_name}.scale", a_mtfp.tobytes(), a_k))
                else:
                    # Some BitNet variants may not store scales explicitly.
                    # Insert zero-scalar with block_exp=0 to keep layout fixed.
                    z = np.zeros(1, dtype=np.int32)
                    tensor_records.append(
                        (f"layer{layer}.{proj_name}.scale", z.tobytes(), 0))

            # 4 Norm γ vectors.
            for norm_name, _ in [
                ("input_layernorm",          HIDDEN_SIZE),
                ("post_attention_layernorm", HIDDEN_SIZE),
                ("self_attn.attn_sub_norm",  HIDDEN_SIZE),
                ("mlp.ffn_sub_norm",         INTERMEDIATE_SIZE),
            ]:
                g = sf.get_tensor(f"{base}.{norm_name}.weight").float().numpy()
                g_mtfp, g_k = fp_to_mtfp19(g)
                tensor_records.append(
                    (f"layer{layer}.{norm_name}.gamma", g_mtfp.tobytes(), g_k))

            print(f"[ok] layer {layer}", file=sys.stderr)

        # 3. Final norm.
        if "model.norm.weight" in keys:
            f_g = sf.get_tensor("model.norm.weight").float().numpy()
            f_mtfp, f_k = fp_to_mtfp19(f_g)
            tensor_records.append(("final_norm.gamma", f_mtfp.tobytes(), f_k))

        # 4. LM head (optional; tied if missing).
        lm_head_tied = 1
        if "lm_head.weight" in keys:
            l = sf.get_tensor("lm_head.weight").float().numpy()
            l_mtfp, l_k = fp_to_mtfp19(l)
            tensor_records.append(("lm_head", l_mtfp.tobytes(), l_k))
            lm_head_tied = 0

    # Write the binary blob.
    n_tensors = len(tensor_records)
    # Header: magic(4) + version(4) + lm_head_tied(4) + n_tensors(4)
    # + block_exps(4*n) + offsets(8*n) + sizes(8*n) = 16 + 4n + 8n + 8n = 16 + 20n
    header_size = 16 + 20 * n_tensors
    cur_offset = header_size
    offsets = []
    sizes = []
    block_exps = []
    for name, data, blk in tensor_records:
        offsets.append(cur_offset)
        sizes.append(len(data))
        block_exps.append(blk)
        cur_offset += len(data)

    print(f"[info] writing blob: {args.output}", file=sys.stderr)
    print(f"[info] tensors: {n_tensors}, header: {header_size} bytes, "
          f"total: {cur_offset:,} bytes ({cur_offset/1e9:.2f} GB)",
          file=sys.stderr)

    with open(args.output, "wb") as f:
        # Header.
        f.write(MAGIC)
        f.write(struct.pack("<III", VERSION, lm_head_tied, n_tensors))
        f.write(struct.pack(f"<{n_tensors}i", *block_exps))
        f.write(struct.pack(f"<{n_tensors}Q", *offsets))
        f.write(struct.pack(f"<{n_tensors}Q", *sizes))
        # Tensor data.
        for _, data, _ in tensor_records:
            f.write(data)

    # Manifest.
    manifest = {
        "model": MODEL_REPO,
        "blob_path": args.output,
        "blob_size": cur_offset,
        "magic": "M4T1",
        "version": VERSION,
        "lm_head_tied": bool(lm_head_tied),
        "n_tensors": n_tensors,
        "n_layers_converted": layer_count,
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_layers": NUM_LAYERS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "vocab_size": VOCAB_SIZE,
        },
        "tensors": [
            {"name": name, "offset": off, "size": sz, "block_exp": be}
            for (name, _, _), off, sz, be in zip(
                tensor_records, offsets, sizes, block_exps)
        ],
    }
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] manifest: {args.manifest}", file=sys.stderr)


if __name__ == "__main__":
    main()
