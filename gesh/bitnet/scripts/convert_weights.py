#!/usr/bin/env python3
"""
convert_weights.py — emit substrate-format binary blob from BitNet's HF release.

Per the LMM cycle in journal/bitnet_phase1_*. Run once per checkpoint.
Output is consumed by the C harness via mmap + bitnet_weights_t population.

Conversion summary (per gap analysis in scripts/README.md):
  HF source                    →  Substrate target
  -----------                     ----------------
  BitLinear weight (U8 4-in-8) →  5-in-8 packed (1.25× density)
  BitLinear scale α (BF16)     →  MTFP19 mantissa + per-tensor block exp
  Norm γ (BF16)                →  MTFP19 mantissa + per-tensor block exp
  Embedding (BF16)             →  MTFP19 mantissa + per-tensor block exp

Per-tensor block exponent: chosen automatically as the largest k such that
3^k × max(|values|) ≤ MTFP19_MAX. Maximizes precision retention. Stored
in metadata.json alongside the binary blob.

Output files:
  bitnet_weights_m4t.bin      — concatenated int32 + uint8 buffers
  bitnet_weights_metadata.json — per-tensor offsets, sizes, block exponents
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

# Substrate constants — match m4t/src/m4t_types.h
MTFP19_MAX = 581_130_733  # (3^19 - 1) / 2
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
    Block exp is the per-tensor power-of-3 exponent: value ≈ mantissa × 3^(-block_exp).

    Picks the largest k such that 3^k × max(|x|) ≤ MTFP19_MAX, maximizing
    precision retention within the MTFP19 cell range.
    """
    arr = np.asarray(values, dtype=np.float64)
    max_abs = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
    if max_abs == 0.0:
        return np.zeros(arr.shape, dtype=np.int32), 0
    # Largest k: 3^k ≤ MTFP19_MAX / max_abs  →  k ≤ log_3(MTFP19_MAX / max_abs)
    k = int(np.floor(np.log(MTFP19_MAX / max_abs) / LOG_3))
    scale = 3.0 ** k
    mantissas = np.round(arr * scale).astype(np.int32)
    # Defensive clamp (rounding could push 1 cell over):
    np.clip(mantissas, -MTFP19_MAX, MTFP19_MAX, out=mantissas)
    return mantissas, k


def unpack_4in8_ternary(packed_u8: np.ndarray) -> np.ndarray:
    """
    HF's 4-in-8 ternary packing. Each byte holds 4 trits, 2 bits per trit.
    Trit encoding: 00=0, 01=+1, 10=-1, 11=reserved.

    Returns int8 array of trit values {-1, 0, +1}.
    """
    arr = np.asarray(packed_u8, dtype=np.uint8)
    n = arr.size * 4
    out = np.empty(n, dtype=np.int8)
    for i in range(4):
        codes = (arr >> (2 * i)) & 0x3
        # Map 00→0, 01→+1, 10→-1, 11→0
        trits = np.where(codes == 1, 1,
                         np.where(codes == 2, -1, 0)).astype(np.int8)
        out[i::4] = trits
    return out


def pack_5in8_ternary(trits: np.ndarray) -> np.ndarray:
    """
    Substrate's 5-in-8 packing per m4t/docs/M4T_SUBSTRATE.md §20.
    byte = u_0 + 3·u_1 + 9·u_2 + 27·u_3 + 81·u_4
    where u_i = trit_to_unsigned(trit_i): -1→2, 0→0, +1→1.

    Trailing trits beyond 5*floor(n/5) zero-pad in the final byte.
    """
    arr = np.asarray(trits, dtype=np.int8)
    # Map trit → unsigned code: -1→2, 0→0, +1→1.
    u = np.where(arr == 1, 1, np.where(arr == -1, 2, 0)).astype(np.uint8)
    n = u.size
    nb = (n + 4) // 5
    out = np.zeros(nb, dtype=np.uint8)
    pow3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)
    # Process in groups of 5; pad the last group if needed.
    for byte_idx in range(nb):
        start = byte_idx * 5
        end = min(start + 5, n)
        v = 0
        for d in range(end - start):
            v += int(u[start + d]) * int(pow3[d])
        out[byte_idx] = v
    return out


def repack_4in8_to_5in8(packed_4in8: np.ndarray, n_trits: int) -> np.ndarray:
    """Convert HF 4-in-8 packed weights to substrate 5-in-8 packed."""
    trits = unpack_4in8_ternary(packed_4in8)[:n_trits]
    return pack_5in8_ternary(trits)


class BlobWriter:
    """Append-only writer that tracks per-tensor offsets for the metadata JSON."""

    def __init__(self, path):
        self.f = open(path, "wb")
        self.offset = 0
        self.entries = OrderedDict()

    def append(self, name: str, data: np.ndarray, block_exp: int = 0):
        bytes_data = data.tobytes()
        size = len(bytes_data)
        self.entries[name] = {
            "offset": self.offset,
            "size": size,
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "block_exp": block_exp,  # value ≈ mantissa × 3^(-block_exp)
        }
        self.f.write(bytes_data)
        self.offset += size

    def close(self):
        self.f.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local-path", default=None,
        help="Path to local model.safetensors. Default: HF cache download.")
    parser.add_argument("--output", default="bitnet_weights_m4t.bin",
        help="Output binary blob.")
    parser.add_argument("--metadata", default="bitnet_weights_metadata.json",
        help="Output metadata JSON.")
    args = parser.parse_args()

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
    print(f"[info] writing blob: {args.output}", file=sys.stderr)

    writer = BlobWriter(args.output)

    # Open safetensors.
    with safe_open(path, framework="pt") as sf:
        keys = sf.keys()

        # 1. Embedding.
        emb_name = "model.embed_tokens.weight"
        if emb_name not in keys:
            print(f"[error] expected tensor not found: {emb_name}", file=sys.stderr); sys.exit(1)
        emb_fp = sf.get_tensor(emb_name).float().numpy()  # bf16 → fp32 → np
        emb_mtfp, emb_k = fp_to_mtfp19(emb_fp)
        writer.append("embedding", emb_mtfp, block_exp=emb_k)
        print(f"[ok] embedding: shape={emb_mtfp.shape} block_exp={emb_k}", file=sys.stderr)

        # 2. Per-layer weights.
        for layer in range(NUM_LAYERS):
            base = f"model.layers.{layer}"
            # BitLinear weights: q, k, v, o, gate, up, down.
            for proj_name, n_in, n_out in [
                ("self_attn.q_proj",  HIDDEN_SIZE,        HIDDEN_SIZE),
                ("self_attn.k_proj",  HIDDEN_SIZE,        KV_PROJ_DIM),
                ("self_attn.v_proj",  HIDDEN_SIZE,        KV_PROJ_DIM),
                ("self_attn.o_proj",  HIDDEN_SIZE,        HIDDEN_SIZE),
                ("mlp.gate_proj",     HIDDEN_SIZE,        INTERMEDIATE_SIZE),
                ("mlp.up_proj",       HIDDEN_SIZE,        INTERMEDIATE_SIZE),
                ("mlp.down_proj",     INTERMEDIATE_SIZE,  HIDDEN_SIZE),
            ]:
                # Weight (U8 4-in-8 packed). Stored shape: [n_out, n_in/4] uint8.
                w_name = f"{base}.{proj_name}.weight"
                w_packed_4in8 = sf.get_tensor(w_name).numpy()
                # Repack each row from 4-in-8 to 5-in-8.
                rows_5in8 = []
                for r in range(n_out):
                    row_4in8 = w_packed_4in8[r] if w_packed_4in8.ndim == 2 else \
                               w_packed_4in8[r * (n_in // 4):(r + 1) * (n_in // 4)]
                    rows_5in8.append(repack_4in8_to_5in8(row_4in8, n_in))
                w_5in8 = np.concatenate(rows_5in8).astype(np.uint8)
                writer.append(f"layer{layer}.{proj_name}.weight", w_5in8)
                # Scale α (BF16 scalar).
                s_name = f"{base}.{proj_name}.weight_scale"
                if s_name in keys:
                    alpha = sf.get_tensor(s_name).float().numpy()
                    a_mtfp, a_k = fp_to_mtfp19(alpha.reshape(-1))
                    writer.append(f"layer{layer}.{proj_name}.scale",
                                  a_mtfp, block_exp=a_k)

            # Norm γ vectors: 4 per layer.
            for norm_name, n_dim in [
                ("input_layernorm",                    HIDDEN_SIZE),
                ("post_attention_layernorm",           HIDDEN_SIZE),
                ("self_attn.attn_sub_norm",            HIDDEN_SIZE),
                ("mlp.ffn_sub_norm",                   INTERMEDIATE_SIZE),
            ]:
                g_name = f"{base}.{norm_name}.weight"
                gamma_fp = sf.get_tensor(g_name).float().numpy()
                g_mtfp, g_k = fp_to_mtfp19(gamma_fp)
                writer.append(f"layer{layer}.{norm_name}.gamma",
                              g_mtfp, block_exp=g_k)

            print(f"[ok] layer {layer}", file=sys.stderr)

        # 3. Final norm + (likely-tied) LM head.
        if "model.norm.weight" in keys:
            final_norm = sf.get_tensor("model.norm.weight").float().numpy()
            f_mtfp, f_k = fp_to_mtfp19(final_norm)
            writer.append("final_norm.gamma", f_mtfp, block_exp=f_k)
            print(f"[ok] final_norm: block_exp={f_k}", file=sys.stderr)

        # LM head: in BitNet's HF release, may or may not be tied. Check both.
        if "lm_head.weight" in keys:
            lmh_fp = sf.get_tensor("lm_head.weight").float().numpy()
            l_mtfp, l_k = fp_to_mtfp19(lmh_fp)
            writer.append("lm_head", l_mtfp, block_exp=l_k)
            print(f"[ok] lm_head (untied): block_exp={l_k}", file=sys.stderr)
        else:
            print("[ok] lm_head (tied to embedding; reuse embedding buffer)",
                  file=sys.stderr)

    writer.close()

    # Write metadata.
    metadata = {
        "model": MODEL_REPO,
        "blob_path": args.output,
        "blob_size": writer.offset,
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_layers": NUM_LAYERS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "vocab_size": VOCAB_SIZE,
        },
        "tensors": writer.entries,
    }
    with open(args.metadata, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[done] blob {writer.offset:,} bytes; metadata: {args.metadata}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
