#!/usr/bin/env python3
"""
compare_activations.py — diff C-harness activation dumps against HF reference.

Per the LMM cycle in journal/bitnet_phase1_*. Closes work-unit 1's gate
("layer 0 forward pass produces some output, per-layer comparison runs").

Reads:
  --c-dump:      output of bitnet_harness <weights_blob.bin> <dump_path>
                 (binary; format defined in bitnet_harness.c::dump_activations_to_file)
  --reference:   output of dump_reference.py (.npz with per-(layer, sublayer) tensors)

Computes per-tensor L2 relative error:
  rel_err = ||c_dump - reference||_2 / ||reference||_2

Reports per-tensor + aggregate. Phase 1 fidelity gate (per SYNTHESIZE D2):
ε bounded across layers, growth not exponential, source characterized.

Note: the C harness's dump is in MTFP19 mantissa units (int32); the HF
reference is fp32. Comparison requires un-scaling the C-side mantissas
by their per-tensor block exponents (read from the weights blob's
metadata) — work-unit 6+ wires this. For Phase 1 work-unit 1, this
script computes raw value comparison with a scale-factor warning;
useful for sanity-check ("does the C side produce non-zero output?")
but not yet a fidelity gate.
"""

import argparse
import struct
import sys

try:
    import numpy as np
except ImportError as e:
    print(f"[error] missing dep: {e}\n  pip install numpy", file=sys.stderr)
    sys.exit(1)


# Tensor capture order in the C-side dump (ACTV2 format; see bitnet_harness.c).
CAPTURE_ORDER_V2 = [
    ("input_layernorm.output",          "hidden"),       # 1. x_norm_input
    ("attn.q_pre_rope",                  "hidden"),       # 2. q before RoPE
    ("attn.k_pre_rope",                  "kv_proj"),      # 3. k before RoPE
    ("attn.v",                           "kv_proj"),      # 4. v
    ("attn.q_post_rope",                 "hidden"),       # 5. q after RoPE
    ("attn.k_post_rope",                 "kv_proj"),      # 6. k after RoPE
    ("attn_sub_norm.output",             "hidden"),       # 7
    ("post_attention_layernorm.output",  "hidden"),       # 8. x_norm (overwritten)
    ("ffn.gate_post_relu2",              "intermediate"), # 9. gate (post-relu²)
    ("ffn.up_proj",                      "intermediate"), # 10
    ("ffn_sub_norm.output",              "intermediate"), # 11
    ("block_output",                     "hidden"),       # 12
]

# Legacy ACTV (v1) format.
CAPTURE_ORDER_V1 = [
    ("input_layernorm.output",          "hidden"),
    ("attn.q",                           "hidden"),
    ("attn.k",                           "kv_proj"),
    ("attn.v",                           "kv_proj"),
    ("attn_sub_norm.output",             "hidden"),
    ("ffn.gate_proj",                    "intermediate"),
    ("ffn.up_proj",                      "intermediate"),
    ("ffn_sub_norm.output",              "intermediate"),
    ("block_output",                     "hidden"),
]


def read_c_dump(path: str):
    """Parse the C-harness dump format (auto-detects v1 vs v2 by magic)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:5] == b"ACTV2":
        order = CAPTURE_ORDER_V2
        # ACTV2 header: 5-byte magic + 3-byte pad + int32 layer + int32×3 sizes = 24 bytes.
        layer_idx, hidden, intermediate, kv_proj = struct.unpack("<iiii", data[8:24])
        offset = 24
    elif data[:4] == b"ACTV":
        order = CAPTURE_ORDER_V1
        layer_idx, hidden, intermediate, kv_proj = struct.unpack("<iiii", data[4:20])
        offset = 20
    else:
        raise ValueError(f"bad magic in {path}")
    sizes = {"hidden": hidden, "intermediate": intermediate, "kv_proj": kv_proj}
    out = {"_meta": {"layer": layer_idx, "hidden": hidden,
                      "intermediate": intermediate, "kv_proj": kv_proj}}
    for name, size_key in order:
        n = sizes[size_key]
        arr = np.frombuffer(data[offset:offset + n * 4], dtype=np.int32)
        out[name] = arr.copy()
        offset += n * 4
    out["_capture_order"] = [n for n, _ in order]
    return out


def read_reference(path: str):
    """Load HF reference dumps (.npz)."""
    return dict(np.load(path))


def relative_l2_error(c_arr_int: np.ndarray, ref_arr_fp: np.ndarray) -> float:
    """Compute relative L2 error treating both arrays as raw values.

    Caveat: C side is int mantissa, ref is fp32. Without the per-tensor
    block exp applied, this is unscaled. Reports anyway as a coarse
    "C side produces non-zero in the right ballpark" check."""
    c = c_arr_int.astype(np.float64)
    r = ref_arr_fp.astype(np.float64).flatten()
    if c.size != r.size:
        return float("nan")
    if np.linalg.norm(r) == 0:
        return float("nan") if np.linalg.norm(c) > 0 else 0.0
    return float(np.linalg.norm(c - r) / np.linalg.norm(r))


def relative_l2_error_scale_invariant(c_arr_int: np.ndarray, ref_arr_fp: np.ndarray) -> float:
    """Scale-invariant relative L2: best-fit a constant scale factor s
    minimizing ||c·s − r||₂, then return ||c·s − r||₂ / ||r||₂.

    Closed form: s = (c · r) / (c · c). Captures "the substrate output
    is the right shape, just at the wrong overall scale" — i.e., the
    block_exp tracking question. If s falls in a sane range across
    layers, the substrate is computing correctly modulo block_exp.

    Phase 1 gate (work-unit 6 of bitnet_phase1_synthesize): ε bounded,
    not exponentially growing across layers. */"""
    c = c_arr_int.astype(np.float64).flatten()
    r = ref_arr_fp.astype(np.float64).flatten()
    if c.size != r.size:
        return float("nan"), float("nan")
    cc = float(np.dot(c, c))
    cr = float(np.dot(c, r))
    rn = float(np.linalg.norm(r))
    if cc == 0 or rn == 0:
        return float("nan"), float("nan")
    s = cr / cc
    err = float(np.linalg.norm(c * s - r) / rn)
    return s, err


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--c-dump", default=None,
        help="Single-layer C dump (legacy; pairs with --layer).")
    parser.add_argument("--c-dump-prefix", default=None,
        help="Multi-layer mode: reads <prefix>.layer<N>.bin for each "
             "layer. Pair with --max-layers.")
    parser.add_argument("--reference", required=True,
        help="Output of dump_reference.py (.npz).")
    parser.add_argument("--layer", type=int, default=0,
        help="Layer index for single-layer mode (default 0).")
    parser.add_argument("--max-layers", type=int, default=30,
        help="Layer count for multi-layer mode (default 30).")
    parser.add_argument("--report-csv", default=None,
        help="Write per-layer (tensor, scale, ε) CSV to this path.")
    args = parser.parse_args()

    if args.c_dump_prefix is None and args.c_dump is None:
        parser.error("specify either --c-dump or --c-dump-prefix")

    ref = read_reference(args.reference)

    if args.c_dump is not None:
        layers = [(args.layer, args.c_dump)]
    else:
        layers = [(l, f"{args.c_dump_prefix}.layer{l}.bin") for l in range(args.max_layers)]

    csv_rows = []
    print(f"{'layer':>5} {'tensor':<32} {'C_norm':>14} {'ref_norm':>14}"
          f" {'best_scale':>14} {'sc_inv_ε':>14}")
    print("-" * 100)

    for layer_idx, c_dump_path in layers:
        try:
            c_dump = read_c_dump(c_dump_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"# layer {layer_idx}: {e}")
            continue

        # Map substrate capture site → HF reference key. Keys with no HF
        # analog (e.g., post-RoPE Q/K — HF only hooks pre-RoPE q_proj output,
        # post-relu²-gate) are listed but skip the comparison for those.
        site_to_ref_key = {
            "input_layernorm.output":           f"layer.{layer_idx}.input_layernorm.output",
            "attn.q_pre_rope":                   f"layer.{layer_idx}.attn.q",
            "attn.k_pre_rope":                   f"layer.{layer_idx}.attn.k",
            "attn.v":                            f"layer.{layer_idx}.attn.v",
            "attn.q_post_rope":                  None,
            "attn.k_post_rope":                  None,
            "attn_sub_norm.output":              f"layer.{layer_idx}.attn_sub_norm.output",
            "post_attention_layernorm.output":   f"layer.{layer_idx}.post_attention_layernorm.output",
            "ffn.gate_post_relu2":               None,
            "ffn.up_proj":                       f"layer.{layer_idx}.ffn.up_proj",
            "ffn_sub_norm.output":               f"layer.{layer_idx}.ffn_sub_norm.output",
            "block_output":                      f"layer.{layer_idx}.block_output",
            # Legacy ACTV v1 names — map to whatever HF analog still works.
            "attn.q":                            f"layer.{layer_idx}.attn.q",
            "attn.k":                            f"layer.{layer_idx}.attn.k",
            "ffn.gate_proj":                     f"layer.{layer_idx}.ffn.gate_proj",
        }

        capture_order = c_dump.get("_capture_order", [n for n, _ in CAPTURE_ORDER_V1])
        for c_key in capture_order:
            if site_to_ref_key.get(c_key) is None:
                continue  # no HF analog
            _ = c_key  # used below
            # Iterate by single key now (legacy CAPTURE_ORDER tuple unpacking removed)
            ref_key = site_to_ref_key[c_key]
            c_arr = c_dump.get(c_key)
            if c_arr is None or ref_key not in ref:
                continue
            r_arr = ref[ref_key]
            c_norm = np.linalg.norm(c_arr.astype(np.float64))
            r_norm = np.linalg.norm(r_arr.astype(np.float64))
            best_s, sc_err = relative_l2_error_scale_invariant(c_arr, r_arr)
            print(f"{layer_idx:>5} {c_key:<32} {c_norm:>14.3e} {r_norm:>14.3e}"
                  f" {best_s:>14.3e} {sc_err:>14.3e}")
            csv_rows.append((layer_idx, c_key, c_norm, r_norm, best_s, sc_err))

    if args.report_csv:
        with open(args.report_csv, "w") as f:
            f.write("layer,tensor,c_norm,ref_norm,best_scale,sc_inv_eps\n")
            for row in csv_rows:
                f.write(",".join(str(x) for x in row) + "\n")
        print(f"\n[wrote] {args.report_csv}")

    print()
    print("note: best_scale is the constant multiplier minimizing the L2 gap;")
    print("      a layer's substrate output is correct mod block_exp if its")
    print("      sc_inv_ε stays small while best_scale is roughly stable.")
    print("      Phase 1 gate: per-layer sc_inv_ε does not grow exponentially")
    print("      across depth.")


if __name__ == "__main__":
    main()
