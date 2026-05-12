"""Phase δ-2 — per-(layer, kv_head, site) close-regime heterogeneity check.

The Phase γ close-regime finding (substrate_L1 d̂/D_amb = 0.259 vs B4 0.724)
pooled within-group pairs across 300 groups. If the substrate manifold
structure varies wildly across (layer, kv_head, site), pooling averages a
heterogeneous mixture and the 47pp gap could be misleading.

This script computes per-group d̂ (where N=8 per group allows it) and
reports the distribution. If d̂ varies wildly, pooling is suspect.
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np

THIS = os.path.dirname(__file__)
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(THIS, "..", "phase_alpha"))
sys.path.insert(0, os.path.join(THIS, "..", "phase_beta"))

from load_k_signatures import (
    HEAD_DIM, THRESHOLD_TAU, collect_all,
    threshold_extract, b2_signature, pairwise_hamming_int8,
)
from m1_l1_estimator import (
    pairwise_L1_int8, cell_pmf_from_data, cdf_at,
)
from scipy.optimize import brentq


DUMP_DIRS = ["data/c_dump", "data/c_dump_v2"]


def per_group_d_hat(K, group_idx, build_sig_fn, distance_fn, pmf_cell):
    """Estimate d̂ within a single group of K-vectors."""
    sigs = build_sig_fn(K[group_idx])
    dist = distance_fn(sigs)
    N = dist.shape[0]
    iu = np.triu_indices(N, k=1)
    flat = dist[iu]
    if len(flat) < 5:
        return float("nan")
    t1 = int(np.quantile(flat, 0.20))  # higher quantile because N is tiny
    t2 = int(np.quantile(flat, 0.50))
    if t2 <= t1: t2 = t1 + 1
    n_total = int(np.sum(flat <= t1)) * 2
    k_total = int(np.sum(flat <= t2)) * 2
    if k_total == 0 or n_total == k_total or n_total == 0:
        return float("nan")
    target = n_total / k_total

    def f(d):
        f1 = cdf_at(pmf_cell, d, t1)
        f2 = cdf_at(pmf_cell, d, t2)
        if f2 <= 0:
            return 1.0
        return f1 / f2 - target

    try:
        return float(brentq(f, 1.0, 500.0, xtol=1e-3))
    except Exception:
        return float("nan")


def main():
    print("=== δ-2: per-group close-regime heterogeneity ===\n")
    K, meta = collect_all(DUMP_DIRS)
    N = K.shape[0]
    print(f"  Loaded N = {N} K-signatures")

    # Build group indices: (layer, kv_head, site) → list of indices
    groups = {}
    for i in range(N):
        key = (int(meta["layer"][i]), int(meta["kv_head"][i]),
               str(meta["site"][i]))
        groups.setdefault(key, []).append(i)
    keys_used = [k for k, v in groups.items() if len(v) >= 6]
    print(f"  {len(keys_used)} groups with N ≥ 6\n")

    # Use a SHARED PMF from all substrate sigs (matches what pooled close
    # regime in Phase γ effectively does)
    all_sub_sigs = threshold_extract(K, tau=THRESHOLD_TAU)
    sub_pmf = cell_pmf_from_data(all_sub_sigs)

    # Compute per-group d̂ for substrate_L1 and B4-Hamming
    from run_phase_alpha_v2 import b4_pca_sign
    sub_d_hats = []
    b4_d_hats = []
    for key in keys_used:
        idx = np.array(groups[key])
        d_sub = per_group_d_hat(
            K, idx,
            build_sig_fn=lambda Kx: threshold_extract(Kx, tau=THRESHOLD_TAU),
            distance_fn=pairwise_L1_int8,
            pmf_cell=sub_pmf,
        )
        # B4 uses Hamming on binary, so per-cell PMF is (0.5, 0.5, 0)
        d_b4 = per_group_d_hat(
            K, idx,
            build_sig_fn=lambda Kx: b4_pca_sign(Kx),
            distance_fn=pairwise_hamming_int8,
            pmf_cell=np.array([0.5, 0.5, 0.0]),
        )
        sub_d_hats.append(d_sub)
        b4_d_hats.append(d_b4)

    sub_arr = np.array(sub_d_hats)
    b4_arr  = np.array(b4_d_hats)
    sub_fin = sub_arr[np.isfinite(sub_arr)]
    b4_fin  = b4_arr[np.isfinite(b4_arr)]

    print("Per-group d̂ distribution:\n")
    def stats(arr, name):
        if len(arr) == 0:
            print(f"  {name}: 0 finite values"); return
        print(f"  {name:18s} n={len(arr):3d}  mean={np.mean(arr):6.2f}  "
              f"median={np.median(arr):6.2f}  std={np.std(arr):5.2f}  "
              f"min={np.min(arr):5.1f}  max={np.max(arr):5.1f}  "
              f"q25={np.quantile(arr, 0.25):.1f}  q75={np.quantile(arr, 0.75):.1f}")
    stats(sub_fin, "substrate_L1 d̂")
    stats(b4_fin,  "B4_pca d̂")

    # Paired comparison: per-group, does substrate < B4?
    paired = []
    for s, b in zip(sub_arr, b4_arr):
        if np.isfinite(s) and np.isfinite(b):
            paired.append((s, b, s - b))
    paired_arr = np.array(paired)
    if len(paired_arr) > 5:
        diffs = paired_arr[:, 2]
        print(f"\n  paired (substrate - B4): mean={np.mean(diffs):+.2f}  "
              f"median={np.median(diffs):+.2f}  "
              f"frac_substrate_lower={np.mean(diffs < 0):.1%}  "
              f"frac_tied={np.mean(diffs == 0):.1%}")

    # Layer-stratified
    by_layer_sub = {}
    by_layer_b4 = {}
    for key, ds, db in zip(keys_used, sub_arr, b4_arr):
        layer = key[0]
        by_layer_sub.setdefault(layer, []).append(ds)
        by_layer_b4.setdefault(layer, []).append(db)
    print("\nLayer-stratified per-group d̂ (mean across kv_heads × sites):\n")
    print(f"{'layer':>5} {'sub_mean':>10} {'B4_mean':>10} {'diff':>10}")
    for L in sorted(by_layer_sub):
        s = np.array(by_layer_sub[L]); s = s[np.isfinite(s)]
        b = np.array(by_layer_b4[L]);  b = b[np.isfinite(b)]
        if len(s) == 0 or len(b) == 0:
            continue
        print(f"{L:>5d} {np.mean(s):>10.2f} {np.mean(b):>10.2f} "
              f"{np.mean(s) - np.mean(b):>+10.2f}")

    out = {
        "n_groups": len(keys_used),
        "substrate_d_hats": [float(x) for x in sub_arr],
        "B4_d_hats": [float(x) for x in b4_arr],
        "substrate_stats": {
            "mean": float(np.mean(sub_fin)) if len(sub_fin) else None,
            "median": float(np.median(sub_fin)) if len(sub_fin) else None,
            "std": float(np.std(sub_fin)) if len(sub_fin) else None,
        },
        "b4_stats": {
            "mean": float(np.mean(b4_fin)) if len(b4_fin) else None,
            "median": float(np.median(b4_fin)) if len(b4_fin) else None,
            "std": float(np.std(b4_fin)) if len(b4_fin) else None,
        },
    }
    os.makedirs("experiments/phase_delta/results", exist_ok=True)
    with open("experiments/phase_delta/results/heterogeneity.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults: experiments/phase_delta/results/heterogeneity.json")


if __name__ == "__main__":
    main()
