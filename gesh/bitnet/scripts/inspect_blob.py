#!/usr/bin/env python3
"""
inspect_blob.py — list tensor names, shapes, dtypes from BitNet's model.safetensors.

(Renamed from inspect.py to avoid shadowing stdlib `inspect` when
scripts/ is on sys.path — broke huggingface_hub.)

Per the LMM cycle in journal/bitnet_phase1_*. Used at the start of
work-unit 1 to ground the conversion script against actual storage
rather than assumed names. Run before convert_weights.py.

Output:
- Stdout: human-readable table grouped by category.
- inspect_manifest.json: machine-readable metadata for downstream scripts.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

try:
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
except ImportError as e:
    print(f"[error] missing dep: {e}\n"
          f"Install with: pip install safetensors huggingface_hub", file=sys.stderr)
    sys.exit(1)

MODEL_REPO = "microsoft/bitnet-b1.58-2B-4T"
MODEL_FILE = "model.safetensors"


def categorize_tensor(name: str) -> str:
    """Bucket a tensor name into a category for grouped display."""
    if name.startswith("model.embed_tokens"):
        return "embedding"
    if name == "lm_head.weight" or name.startswith("lm_head"):
        return "lm_head"
    if name == "model.norm.weight":
        return "final_norm"
    if name.startswith("model.layers."):
        # Path: model.layers.<idx>.<rest>
        rest = name.split(".", 3)[3]
        if rest.startswith("input_layernorm"):
            return "layer.input_norm"
        if rest.startswith("post_attention_layernorm"):
            return "layer.post_attn_norm"
        if rest.startswith("self_attn.q_proj"):
            return "layer.attn.q"
        if rest.startswith("self_attn.k_proj"):
            return "layer.attn.k"
        if rest.startswith("self_attn.v_proj"):
            return "layer.attn.v"
        if rest.startswith("self_attn.o_proj"):
            return "layer.attn.o"
        if rest.startswith("self_attn.attn_sub_norm"):
            return "layer.attn_sub_norm"
        if rest.startswith("mlp.gate_proj"):
            return "layer.ffn.gate"
        if rest.startswith("mlp.up_proj"):
            return "layer.ffn.up"
        if rest.startswith("mlp.down_proj"):
            return "layer.ffn.down"
        if rest.startswith("mlp.ffn_sub_norm"):
            return "layer.ffn_sub_norm"
    return f"unknown:{name}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-path", default=None,
        help="Path to a local model.safetensors file. If unset, downloads "
             f"from HuggingFace ({MODEL_REPO}).")
    parser.add_argument(
        "--manifest", default="inspect_manifest.json",
        help="Output JSON manifest path.")
    args = parser.parse_args()

    # Resolve model file path.
    if args.local_path:
        path = args.local_path
        if not os.path.isfile(path):
            print(f"[error] file not found: {path}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"[info] downloading {MODEL_FILE} from {MODEL_REPO} "
              f"(uses HF cache; ~1.18 GB on first run)...", file=sys.stderr)
        path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        print(f"[info] cached at: {path}", file=sys.stderr)

    # Open and enumerate tensors.
    manifest = []
    by_category = defaultdict(list)
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            entry = {
                "name": name,
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "numel": t.numel(),
                "category": categorize_tensor(name),
            }
            manifest.append(entry)
            by_category[entry["category"]].append(entry)

    # Print grouped summary to stdout.
    print(f"# Inspection: {path}")
    print(f"# Total tensors: {len(manifest)}")
    print()

    # Order categories: embedding, layers, lm_head, final_norm, unknown.
    cat_order = ["embedding", "lm_head", "final_norm"]
    layer_cats = sorted({c for c in by_category if c.startswith("layer.")})
    cat_order = ["embedding"] + layer_cats + ["lm_head", "final_norm"]
    cat_order += [c for c in by_category if c.startswith("unknown")]

    for cat in cat_order:
        items = by_category.get(cat, [])
        if not items:
            continue
        # For per-layer categories, show one example + count.
        if cat.startswith("layer."):
            example = items[0]
            print(f"## {cat}")
            print(f"  count: {len(items)} (one per layer)")
            print(f"  example name: {example['name']}")
            print(f"  shape: {example['shape']}")
            print(f"  dtype: {example['dtype']}")
            print(f"  numel: {example['numel']:,}")
            # Sanity-check: all items in this category should have the same shape/dtype.
            shapes = {tuple(it["shape"]) for it in items}
            dtypes = {it["dtype"] for it in items}
            if len(shapes) > 1:
                print(f"  [WARN] shape variance across layers: {shapes}")
            if len(dtypes) > 1:
                print(f"  [WARN] dtype variance across layers: {dtypes}")
            print()
        else:
            for it in items:
                print(f"## {cat}: {it['name']}")
                print(f"  shape: {it['shape']}, dtype: {it['dtype']}, "
                      f"numel: {it['numel']:,}")
                print()

    # Cumulative param totals.
    total_params = sum(e["numel"] for e in manifest)
    by_dtype = defaultdict(int)
    for e in manifest:
        by_dtype[e["dtype"]] += e["numel"]
    print("## Param totals")
    print(f"  total numel: {total_params:,}")
    for dt, n in sorted(by_dtype.items()):
        print(f"  {dt}: {n:,}")

    # Write manifest.
    with open(args.manifest, "w") as f:
        json.dump({"path": path, "tensors": manifest}, f, indent=2)
    print(f"\n[info] manifest written to {args.manifest}")


if __name__ == "__main__":
    main()
