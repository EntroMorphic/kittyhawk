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


# Tensor capture order in the C-side dump (matches bitnet_harness.c).
CAPTURE_ORDER = [
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
    """Parse the C-harness dump format."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 20 or data[:4] != b"ACTV":
        raise ValueError(f"bad magic in {path}")
    layer_idx, hidden, intermediate, kv_proj = struct.unpack("<iiii", data[4:20])
    sizes = {
        "hidden": hidden,
        "intermediate": intermediate,
        "kv_proj": kv_proj,
    }
    out = {"_meta": {"layer": layer_idx, "hidden": hidden,
                      "intermediate": intermediate, "kv_proj": kv_proj}}
    offset = 20
    for name, size_key in CAPTURE_ORDER:
        n = sizes[size_key]
        arr = np.frombuffer(data[offset:offset + n * 4], dtype=np.int32)
        out[name] = arr.copy()  # detach from buffer
        offset += n * 4
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--c-dump", required=True,
        help="Output of bitnet_harness <weights_blob.bin> <dump_path>.")
    parser.add_argument("--reference", required=True,
        help="Output of dump_reference.py (.npz).")
    parser.add_argument("--layer", type=int, default=0,
        help="Layer index to compare (default 0).")
    args = parser.parse_args()

    c_dump = read_c_dump(args.c_dump)
    ref = read_reference(args.reference)

    print(f"# Comparing layer {args.layer}")
    print(f"# C dump:    {args.c_dump}")
    print(f"# Reference: {args.reference}")
    print()
    print(f"{'tensor':<32} {'C_norm':>14} {'ref_norm':>14} {'rel_L2_err':>14}")
    print("-" * 76)

    site_to_ref_key = {
        "input_layernorm.output":    f"layer.{args.layer}.input_layernorm.output",
        "attn.q":                    f"layer.{args.layer}.attn.q",
        "attn.k":                    f"layer.{args.layer}.attn.k",
        "attn.v":                    f"layer.{args.layer}.attn.v",
        "attn_sub_norm.output":      f"layer.{args.layer}.attn_sub_norm.output",
        "ffn.gate_proj":             f"layer.{args.layer}.ffn.gate_proj",
        "ffn.up_proj":               f"layer.{args.layer}.ffn.up_proj",
        "ffn_sub_norm.output":       f"layer.{args.layer}.ffn_sub_norm.output",
        "block_output":              f"layer.{args.layer}.block_output",
    }

    for c_key, _ in CAPTURE_ORDER:
        ref_key = site_to_ref_key[c_key]
        c_arr = c_dump.get(c_key)
        if c_arr is None:
            print(f"{c_key:<32}    <missing in C dump>")
            continue
        if ref_key not in ref:
            print(f"{c_key:<32} {np.linalg.norm(c_arr):>14.3e}  "
                  f"<not in reference: {ref_key}>")
            continue
        r_arr = ref[ref_key]
        c_norm = np.linalg.norm(c_arr.astype(np.float64))
        r_norm = np.linalg.norm(r_arr.astype(np.float64))
        err = relative_l2_error(c_arr, r_arr)
        print(f"{c_key:<32} {c_norm:>14.3e} {r_norm:>14.3e} {err:>14.3e}")

    print()
    print("note: C-side dump is in raw int32 mantissas; this comparison is")
    print("      unscaled. Phase 1 work-unit 6 will apply per-tensor block")
    print("      exponents from the weights blob to produce a calibrated")
    print("      fidelity comparison.")


if __name__ == "__main__":
    main()
