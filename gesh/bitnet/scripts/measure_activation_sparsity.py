#!/usr/bin/env python3
"""
Measure post-ReLU² activation sparsity in BitNet's FFN intermediate path.

Reads ACTV2 activation dumps produced by bitnet_harness --dump, computes
zero-cell fraction at the two relevant capture sites:

  - gate (post-relu²): fraction of cells zeroed by ReLU²
  - ffn_sub_norm:      fraction of zero cells at the down_proj A8-quantize input

Reports per-layer, per-position distributions across the prompt set.

The K=6912 down_proj routed16 crossover is at 92-94% sparsity. If
post-ReLU² sparsity exceeds that threshold for any meaningful fraction
of (layer, position) pairs, routed16 has a real role to play on
down_proj. Otherwise it does not.
"""

import argparse
import os
import struct
import sys
from collections import defaultdict
from glob import glob


HIDDEN = 2560
INTERMEDIATE = 6912
KV_PROJ = 640


CAPTURE_LAYOUT = [
    ("x_norm_input",  HIDDEN),
    ("q_pre_rope",    HIDDEN),
    ("k_pre_rope",    KV_PROJ),
    ("v",             KV_PROJ),
    ("q_post_rope",   HIDDEN),
    ("k_post_rope",   KV_PROJ),
    ("attn_sub_norm", HIDDEN),
    ("x_norm",        HIDDEN),
    ("gate",          INTERMEDIATE),  # POST-relu²
    ("up",            INTERMEDIATE),
    ("ffn_sub_norm",  INTERMEDIATE),
    ("block_output",  HIDDEN),
]


def read_dump(path):
    """Return dict {capture_name: list_of_int32}. Raises on format mismatch."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 24:
        raise ValueError(f"{path}: too small")
    magic = data[:5]
    if magic != b"ACTV2":
        raise ValueError(f"{path}: bad magic {magic!r}")
    layer, hidden, intermediate, kv_proj = struct.unpack("<iiii", data[8:24])
    if (hidden, intermediate, kv_proj) != (HIDDEN, INTERMEDIATE, KV_PROJ):
        raise ValueError(f"{path}: shape mismatch ({hidden}, {intermediate}, {kv_proj})")

    out = {"_layer": layer}
    off = 24
    for name, n in CAPTURE_LAYOUT:
        end = off + 4 * n
        out[name] = list(struct.unpack(f"<{n}i", data[off:end]))
        off = end
    if off != len(data):
        raise ValueError(f"{path}: trailing {len(data) - off} bytes")
    return out


def zero_frac(values):
    """Fraction of cells that are exactly zero."""
    if not values:
        return 0.0
    z = sum(1 for v in values if v == 0)
    return z / len(values)


def percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def parse_dump_path(path):
    """Extract (prompt_prefix, position, layer) from a dump path.

    Patterns:
      <prefix>.pos<P>.layer<L>.bin
      <prefix>.layer<L>.bin           (single position)
    """
    base = os.path.basename(path)
    if not base.endswith(".bin"):
        return None
    parts = base[:-4].split(".")
    if len(parts) >= 2 and parts[-1].startswith("layer"):
        layer = int(parts[-1][len("layer"):])
    else:
        return None
    pos = 0
    prefix_parts = parts[:-1]
    if prefix_parts and prefix_parts[-1].startswith("pos"):
        pos = int(prefix_parts[-1][len("pos"):])
        prefix_parts = prefix_parts[:-1]
    prefix = ".".join(prefix_parts)
    return (prefix, pos, layer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "dumps",
        nargs="+",
        help="Directories or globs containing pos*.layer*.bin files.",
    )
    ap.add_argument(
        "--site",
        choices=["gate", "ffn_sub_norm", "x_norm", "attn_sub_norm",
                 "all_bitlinear_inputs", "both"],
        default="both",
        help="Capture site to analyze. all_bitlinear_inputs covers "
             "every site that feeds a BitLinear: x_norm (q/k/v/gate/up), "
             "attn_sub_norm (o_proj), ffn_sub_norm (down).",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=92.0,
        help="Sparsity threshold (%%) for routed16 viability check.",
    )
    args = ap.parse_args()

    files = []
    for p in args.dumps:
        if os.path.isdir(p):
            files.extend(sorted(glob(os.path.join(p, "*.bin"))))
        else:
            files.extend(sorted(glob(p)))
    files = [f for f in files if "logits" not in f]
    if not files:
        print("no dump files matched", file=sys.stderr)
        return 1

    # data[(site, prompt, position, layer)] = sparsity %
    data = {}
    prompts = set()
    positions = set()
    layers = set()

    for f in files:
        info = parse_dump_path(f)
        if info is None:
            continue
        prefix, pos, layer = info
        try:
            d = read_dump(f)
        except ValueError as e:
            print(f"skip {f}: {e}", file=sys.stderr)
            continue
        prompts.add(prefix)
        positions.add(pos)
        layers.add(layer)
        for site in ("gate", "ffn_sub_norm"):
            data[(site, prefix, pos, layer)] = 100.0 * zero_frac(d[site])

    layers = sorted(layers)
    positions = sorted(positions)
    prompts = sorted(prompts)

    if args.site == "both":
        sites = ["gate", "ffn_sub_norm"]
    elif args.site == "all_bitlinear_inputs":
        sites = ["x_norm", "attn_sub_norm", "ffn_sub_norm"]
    else:
        sites = [args.site]

    # Extend data to include sites the user requested that aren't yet collected.
    for f in files:
        info = parse_dump_path(f)
        if info is None:
            continue
        prefix, pos, layer = info
        try:
            d = read_dump(f)
        except ValueError:
            continue
        for site in sites:
            key = (site, prefix, pos, layer)
            if key not in data and site in d:
                data[key] = 100.0 * zero_frac(d[site])

    for site in sites:
        print(f"\n══ {site} sparsity (% zero cells) ══")
        # Per-layer summary across all (prompt, position).
        print(f"  {'layer':>5}  "
              f"{'min':>6}  {'p10':>6}  {'p50':>6}  {'p90':>6}  {'max':>6}  "
              f"{'frac≥' + str(args.threshold) + '%':>14}")
        for layer in layers:
            samples = [
                data[(site, p, pos, layer)]
                for p in prompts
                for pos in positions
                if (site, p, pos, layer) in data
            ]
            if not samples:
                continue
            n_above = sum(1 for s in samples if s >= args.threshold)
            print(
                f"  {layer:>5}  "
                f"{min(samples):>6.2f}  "
                f"{percentile(samples, 10):>6.2f}  "
                f"{percentile(samples, 50):>6.2f}  "
                f"{percentile(samples, 90):>6.2f}  "
                f"{max(samples):>6.2f}  "
                f"{n_above:>5} / {len(samples):<5}"
            )

        # Aggregate across all (layer, prompt, position).
        all_samples = [
            data[(site, p, pos, layer)]
            for p in prompts
            for pos in positions
            for layer in layers
            if (site, p, pos, layer) in data
        ]
        n_above = sum(1 for s in all_samples if s >= args.threshold)
        print(
            f"\n  Across all (layer, position, prompt): "
            f"min {min(all_samples):.2f}%, "
            f"median {percentile(all_samples, 50):.2f}%, "
            f"max {max(all_samples):.2f}%"
        )
        print(
            f"  Pairs ≥ {args.threshold:.1f}%: "
            f"{n_above} / {len(all_samples)} "
            f"({100.0 * n_above / len(all_samples):.1f}%)"
        )

    print(f"\nSummary: {len(prompts)} prompt(s), {len(positions)} position(s), "
          f"{len(layers)} layer(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
