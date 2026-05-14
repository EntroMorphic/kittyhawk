"""Metric refinement for the kernel-routing Layer 3.

The current kernel is isotropic L1-trit with global α=1.0:
    w(d) = exp(-α · d)
    pred(c) = sum_a w(|c - a|_L1) · Δ(a) / sum_a w(...)

Pattern from iterations 5, 6, 7:
    iter 5 (-1, 1, 1): pred +4.3,  obs +3.5,  err  0.8 (HIT)
    iter 6 ( 1, 0, 1): pred +2.0,  obs +5.0,  err  3.0 (HIT)
    iter 7 (-1, 0, 1): pred +0.8,  obs +5.8,  err  5.1 (near-miss; underpred)

The kernel is DIRECTIONALLY correct (3/3) but MAGNITUDE-wise it
shrinks too aggressively toward the dataset mean on w_qk=+1 cells.
Hypothesis: per-axis bandwidth — the w_qk axis is more locally
coherent than the w_kk/w_r axes, so neighbors that differ only in
w_kk or w_r should be MORE informative (smaller effective distance)
than neighbors that differ in w_qk.

We sweep:
    1. Global isotropic α
    2. Per-axis (α_r, α_kk, α_qk) — weighted L1 distance
       d_w(c, a) = α_r·|c_r - a_r| + α_kk·|c_kk - a_kk| + α_qk·|c_qk - a_qk|
       (we then use w(d_w) = exp(-d_w); the α's both rescale the
       distance and act as the bandwidth — equivalent to per-axis
       bandwidth on a uniform-L1 metric.)

Scoring:
    - LOO predictions on all 10 anchors → MAE + HIT-count (|err| ≤ 5pp)
    - Targeted: error on (-1, 0, 1) when held out (the magnitude-miss case)
    - Targeted: error on (0, -1, 0) — the BIG miss from iter 4 (-13.9pp;
      our worst-anchored outlier — does the refined metric handle it?)

Usage:
    python experiments/phase_zeta/metric_refinement.py
"""
from __future__ import annotations

import json
import math
import os
import sys

ANCHORS_JSON = "experiments/phase_zeta/results/meta_iterate/anchors.json"


def load_anchors():
    with open(ANCHORS_JSON) as f:
        d = json.load(f)
    return [(tuple(a["w"]), float(a["delta"]), a["source"]) for a in d["anchors"]]


def weighted_l1(c, a, alpha_axes):
    """Distance: alpha_r·|c_r-a_r| + alpha_kk·|c_kk-a_kk| + alpha_qk·|c_qk-a_qk|."""
    ar, akk, aqk = alpha_axes
    return ar * abs(c[0] - a[0]) + akk * abs(c[1] - a[1]) + aqk * abs(c[2] - a[2])


def predict_kernel(candidate, anchors, alpha_axes):
    """Kernel retrieval with anisotropic weighted distance.
       alpha_axes = (α_r, α_kk, α_qk). For isotropic, pass (α, α, α)."""
    weights = []
    deltas = []
    for w, delta, _ in anchors:
        d = weighted_l1(candidate, w, alpha_axes)
        weights.append(math.exp(-d))
        deltas.append(delta)
    total = sum(weights)
    if total == 0:
        return 0.0
    return sum(wt * dt for wt, dt in zip(weights, deltas)) / total


def loo_scores(anchors, alpha_axes):
    """For each anchor, predict its delta from the others. Return (errors,
       per-anchor predictions, MAE, hit_count)."""
    errors = []
    preds = []
    for i, (w_i, delta_i, _) in enumerate(anchors):
        others = anchors[:i] + anchors[i+1:]
        pred = predict_kernel(w_i, others, alpha_axes)
        preds.append(pred)
        errors.append(pred - delta_i)
    mae = sum(abs(e) for e in errors) / len(errors)
    hits = sum(1 for e in errors if abs(e) <= 5.0)
    return errors, preds, mae, hits


def print_loo_table(anchors, alpha_axes, label):
    errors, preds, mae, hits = loo_scores(anchors, alpha_axes)
    print(f"\n{label}    α=({alpha_axes[0]:.2f}, {alpha_axes[1]:.2f}, {alpha_axes[2]:.2f})")
    print(f"  {'w':<18} {'obs':>7} {'pred':>7} {'err':>7}")
    for (w, delta, _), p, e in zip(anchors, preds, errors):
        mark = "" if abs(e) <= 5 else "  miss"
        print(f"  {str(w):<18} {delta:+7.2f} {p:+7.2f} {e:+7.2f}{mark}")
    print(f"  MAE = {mae:.2f}pp   HITs ≤ 5pp: {hits}/{len(anchors)}")
    return mae, hits


def main():
    anchors = load_anchors()
    print(f"Loaded {len(anchors)} anchors\n")

    # ---- Pass 0: status-quo (isotropic α=1.0) baseline ----
    print("=" * 70)
    print("PASS 0: current isotropic kernel (α=1.0)")
    print("=" * 70)
    print_loo_table(anchors, (1.0, 1.0, 1.0), "isotropic α=1.0")

    # ---- Pass 1: global isotropic α sweep ----
    print("\n" + "=" * 70)
    print("PASS 1: global isotropic α sweep")
    print("=" * 70)
    print(f"  {'α':>5}    {'MAE':>6}    {'HITs':>5}")
    isotropic_results = []
    for alpha in [0.10, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]:
        _, _, mae, hits = loo_scores(anchors, (alpha, alpha, alpha))
        isotropic_results.append((alpha, mae, hits))
        print(f"  {alpha:>5.2f}    {mae:>6.2f}    {hits:>2}/10")
    best_iso = min(isotropic_results, key=lambda r: r[1])
    print(f"\nBest isotropic: α={best_iso[0]} → MAE={best_iso[1]:.2f}pp, HITs={best_iso[2]}/10")

    # ---- Pass 2: per-axis bandwidth sweep ----
    print("\n" + "=" * 70)
    print("PASS 2: per-axis bandwidth (α_r, α_kk, α_qk)")
    print("=" * 70)
    grid_vals = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    best_anis = None
    all_anis_results = []
    for ar in grid_vals:
        for akk in grid_vals:
            for aqk in grid_vals:
                _, _, mae, hits = loo_scores(anchors, (ar, akk, aqk))
                all_anis_results.append(((ar, akk, aqk), mae, hits))
                if best_anis is None or mae < best_anis[1]:
                    best_anis = ((ar, akk, aqk), mae, hits)
    print(f"Best anisotropic: α=({best_anis[0][0]}, {best_anis[0][1]}, {best_anis[0][2]})")
    print(f"  MAE={best_anis[1]:.2f}pp, HITs={best_anis[2]}/10")
    print(f"Top 5 by MAE:")
    all_anis_results.sort(key=lambda r: r[1])
    for cfg, mae, hits in all_anis_results[:5]:
        print(f"  α=({cfg[0]:>4}, {cfg[1]:>4}, {cfg[2]:>4})  MAE={mae:.2f}pp  HITs={hits}/10")

    # ---- Pass 3: detailed LOO under best anisotropic ----
    print("\n" + "=" * 70)
    print("PASS 3: LOO table under best anisotropic kernel")
    print("=" * 70)
    print_loo_table(anchors, best_anis[0], "best-anisotropic")
    # And compare against isotropic baseline for the targeted near-miss cells
    print("\nFOCUS: errors on the two magnitude near-misses (anchors held out):")
    for target in [(-1, 0, 1), (1, 0, 1)]:
        i = next(j for j, (w, _, _) in enumerate(anchors) if w == target)
        others = anchors[:i] + anchors[i+1:]
        pred_iso = predict_kernel(target, others, (1.0, 1.0, 1.0))
        pred_anis = predict_kernel(target, others, best_anis[0])
        obs = anchors[i][1]
        print(f"  {target}  obs={obs:+5.2f}  iso pred={pred_iso:+5.2f} (err {pred_iso-obs:+.2f})"
              f"  anis pred={pred_anis:+5.2f} (err {pred_anis-obs:+.2f})")

    # ---- Pass 4: re-rank the untested cells under best anisotropic ----
    print("\n" + "=" * 70)
    print("PASS 4: untested-cell predictions under best anisotropic")
    print("=" * 70)
    anchor_set = {w for w, _, _ in anchors}
    untested = []
    for r in [-1, 0, 1]:
        for kk in [-1, 0, 1]:
            for qk in [-1, 0, 1]:
                if (r, kk, qk) not in anchor_set:
                    p_iso  = predict_kernel((r, kk, qk), anchors, (1.0, 1.0, 1.0))
                    p_anis = predict_kernel((r, kk, qk), anchors, best_anis[0])
                    untested.append(((r, kk, qk), p_iso, p_anis))
    untested.sort(key=lambda u: -u[2])
    print(f"  {'cell':<14} {'iso pred':>10} {'anis pred':>10}   delta")
    for cell, p_iso, p_anis in untested:
        print(f"  {str(cell):<14} {p_iso:+10.2f} {p_anis:+10.2f}   {p_anis - p_iso:+5.2f}")
    print()
    print(f"Top anisotropic-predicted untested cell: {untested[0][0]}")
    print(f"  isotropic predicted {untested[0][1]:+.2f}pp")
    print(f"  anisotropic predicted {untested[0][2]:+.2f}pp")
    if untested[0][2] > 6.4:
        print(f"  ** PREDICTED TO BEAT qsigdist (+6.4pp) **")

    # ---- Save best metric for re-use ----
    out = {
        "best_isotropic": {"alpha": best_iso[0], "mae": best_iso[1], "hits": best_iso[2]},
        "best_anisotropic": {
            "alpha": list(best_anis[0]),
            "mae": best_anis[1],
            "hits": best_anis[2],
        },
        "untested_predictions_anisotropic": [
            {"cell": list(c), "iso_pred": p_iso, "anis_pred": p_anis}
            for c, p_iso, p_anis in untested
        ],
    }
    out_path = "experiments/phase_zeta/results/meta_iterate/metric_refinement.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
