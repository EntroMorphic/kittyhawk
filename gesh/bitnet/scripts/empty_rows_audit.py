#!/usr/bin/env python3
"""
Audit empty K-rows across all BitLinears in all 30 layers of the
substrate blob.

The structure analysis on layer 0 surfaced ~10% empty K-rows in
o_proj (and some empties in down_proj). Question: is this a layer-0
fluke, or systemic across the model? Are the SAME K-indices empty
across layers, or do they drift?

For each (layer, BitLinear), we count exactly:
  - Number of empty rows (K-positions with 0 nonzeros across all N cols)
  - Min/max/median nnz per row (full distribution, not just percentiles)
  - The set of empty-row indices (for cross-layer overlap analysis)

Then we cross-tabulate:
  - Across all 30 o_proj layers: which K-indices are empty in ALL
    layers (suggesting they're structurally dead at every layer)?
  - Same for down_proj.
"""

import argparse
import os
import struct
import sys

import numpy as np


HIDDEN = 2560
INTERMEDIATE = 6912
KV_PROJ = 640


# Per layer, BitLinear slot offsets within the layer:
#   slot 0=q, 1=k, 2=v, 3=o, 4=gate, 5=up, 6=down
# For layer L, BitLinear at slot s has tensor index 1 + L * 18 + s.
# (verified against /tmp/bench_routed_real.c which used slot 0..6 for layer 0).
# Confirm via header inspection.

NUM_LAYERS = 30
BITLINEARS_PER_LAYER = 18  # tensors per layer; slots 0..6 are BitLinears, 7..17 are scales/etc.

POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)


def decode_5in8(W_packed, K, N):
    Kp = (K + 4) // 5
    raw = np.frombuffer(W_packed[: N * Kp], dtype=np.uint8).reshape(N, Kp)
    out = np.zeros((N, K), dtype=np.int8)
    for d in range(5):
        u = (raw // POW3[d]) % 3
        cols = d + 5 * np.arange(Kp)
        valid = cols < K
        cols = cols[valid]
        u = u[:, : cols.size]
        signed = np.where(u == 1, 1, np.where(u == 2, -1, 0)).astype(np.int8)
        out[:, cols] = signed
    return out


def load_blob(path):
    with open(path, "rb") as f:
        data = f.read()
    version, lm_head_tied, n_tensors = struct.unpack("<iii", data[4:16])
    off = 16
    block_exps = struct.unpack(f"<{n_tensors}i", data[off : off + 4 * n_tensors])
    off += 4 * n_tensors
    offsets = struct.unpack(f"<{n_tensors}Q", data[off : off + 8 * n_tensors])
    off += 8 * n_tensors
    sizes = struct.unpack(f"<{n_tensors}Q", data[off : off + 8 * n_tensors])
    return data, list(zip(block_exps, offsets, sizes)), n_tensors


def get_W(data, descriptors, idx, K, N):
    _, off, size = descriptors[idx]
    Wp = data[off : off + size]
    return decode_5in8(Wp, K, N)


def per_row_nnz(W):
    """W is shape (N, K). Returns nnz count per K-position (length K)."""
    return (W != 0).sum(axis=0)


def empty_row_indices(W):
    """Returns array of K-indices where the row is entirely zero."""
    return np.where(per_row_nnz(W) == 0)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blob")
    ap.add_argument("--layers", type=int, default=NUM_LAYERS)
    args = ap.parse_args()

    if not os.path.exists(args.blob):
        print(f"missing: {args.blob}", file=sys.stderr); return 1

    data, descriptors, n_tensors = load_blob(args.blob)
    print(f"blob has {n_tensors} tensor descriptors", file=sys.stderr)

    # Discover the per-layer tensor stride by looking at sizes.
    # Layer L's BitLinear at slot s should be at index 1 + L*18 + s.
    # Sanity: descriptor[1].size should match q_proj (HIDDEN × HIDDEN trits packed).
    expected_size_q = ((HIDDEN + 4) // 5) * HIDDEN  # bytes
    actual = descriptors[1][2]
    print(f"sanity: q_proj layer 0 expected size {expected_size_q}, "
          f"actual {actual}", file=sys.stderr)

    # Per-BitLinear slots:
    bitlinears = [
        ("q_proj",    0, HIDDEN,       HIDDEN),
        ("k_proj",    1, HIDDEN,       KV_PROJ),
        ("v_proj",    2, HIDDEN,       KV_PROJ),
        ("o_proj",    3, HIDDEN,       HIDDEN),
        ("gate_proj", 4, HIDDEN,       INTERMEDIATE),
        ("up_proj",   5, HIDDEN,       INTERMEDIATE),
        ("down_proj", 6, INTERMEDIATE, HIDDEN),
    ]

    # For each BitLinear name, collect per-layer empty-row sets.
    per_bl_empty_per_layer = {bl[0]: [] for bl in bitlinears}

    print(f"\n{'='*70}\nPer-layer empty-row counts (K-positions w/ 0 nonzeros across N cols)\n{'='*70}")
    print(f"{'L':>3}  ", end="")
    for name, _, _, _ in bitlinears:
        print(f"{name:>10}", end="  ")
    print()

    for L in range(args.layers):
        print(f"{L:>3}  ", end="")
        for name, slot, K, N in bitlinears:
            idx = 1 + L * BITLINEARS_PER_LAYER + slot
            if idx >= n_tensors:
                print(f"{'-':>10}", end="  ")
                continue
            try:
                W = get_W(data, descriptors, idx, K, N)
            except Exception as e:
                print(f"{'err':>10}", end="  ")
                continue
            empties = empty_row_indices(W)
            per_bl_empty_per_layer[name].append(set(empties.tolist()))
            print(f"{len(empties):>4} ({100*len(empties)/K:>4.1f}%)", end=" ")
        print()

    # Cross-layer overlap analysis.
    print(f"\n{'='*70}\nCross-layer overlap of empty-row indices\n{'='*70}")
    for name, _, K, _ in bitlinears:
        sets = per_bl_empty_per_layer[name]
        if not sets:
            continue
        if all(len(s) == 0 for s in sets):
            print(f"  {name:<12s}: no empty rows in any layer")
            continue
        intersection_all = set.intersection(*sets) if sets else set()
        union_all = set.union(*sets) if sets else set()
        # Also: how many layers does each empty row appear in?
        counts = {}
        for s in sets:
            for k in s:
                counts[k] = counts.get(k, 0) + 1
        n_layers_present = sorted(counts.values())
        print(f"  {name:<12s}: K={K}")
        print(f"    union of empty rows across layers: {len(union_all)}")
        print(f"    intersection (empty in ALL layers): {len(intersection_all)}")
        if n_layers_present:
            print(f"    distribution of empty-row layer-count: ", end="")
            from collections import Counter
            ctr = Counter(n_layers_present)
            for k in sorted(ctr):
                print(f"{k}L:{ctr[k]} ", end="")
            print()
        if len(intersection_all) > 0 and len(intersection_all) <= 32:
            print(f"    indices empty in ALL layers: {sorted(intersection_all)}")
        elif len(intersection_all) > 32:
            print(f"    first 32 indices empty in ALL layers: {sorted(list(intersection_all))[:32]}")

    # Drill into o_proj specifically: per-position nnz histograms.
    print(f"\n{'='*70}\no_proj: distribution of per-K-position nnz, sampled across layers\n{'='*70}")
    for L in [0, 7, 14, 21, 29]:
        idx = 1 + L * BITLINEARS_PER_LAYER + 3  # o_proj slot
        if idx >= n_tensors:
            continue
        try:
            W = get_W(data, descriptors, idx, HIDDEN, HIDDEN)
        except:
            continue
        nnz = per_row_nnz(W)
        # Histogram: bucket by nnz count
        bins = [0, 1, 10, 100, 500, 1000, 1500, 2000, 2560]
        hist, _ = np.histogram(nnz, bins=bins)
        total = nnz.sum()
        print(f"  layer {L:>2}: nnz/row hist: ", end="")
        for i in range(len(bins) - 1):
            print(f"[{bins[i]:>4}–{bins[i+1]:>4}]={hist[i]:>4}  ", end="")
        print()
        print(f"           min={nnz.min()}, max={nnz.max()}, "
              f"sum={total} (overall sparsity = "
              f"{100*(1 - total / (HIDDEN*HIDDEN)):.2f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
