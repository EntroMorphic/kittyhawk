"""Phase α remediated (v2): full red-team remediation per
journal/td27_phase_alpha_redteam_2026-05-12.md.

Changes from run_phase_alpha.py:

  [#1 M3 fix] Replace degenerate "longest_bar > 2× B3.p95" rule with
    Wasserstein distance between bar-length distributions.
    Verdict: substrate's bar distribution differs from B2's by more
    than B3's bar distribution differs from B2's (null = baseline
    variability among random projections).

  [#2 M1 normalized] Add d̂/D normalized metric alongside absolute d̂.
    Require BOTH the absolute gap ≥ 20% AND the normalized gap ≥ 5pp
    to PASS. The original spec was unit-of-measure inflatable.

  [#3 τ sweep] Run substrate-vs-B2 M1 at τ ∈ {2000, 5000, 10000, 20000}.
    Verdict robustness to threshold choice.

  [#4 B4] Add structured binary baseline: sign of top-203 principal-
    component projection (PCA+sign). Replaces "substrate beats random
    projection" with "substrate beats structured-binary at equal capacity."

  [#5 CIs on M2/M3] Bootstrap reciprocity, Gini, longest_bar across
    representations alongside M1.

  [#6 (t1,t2) sensitivity] Run M1 fixed-radii at quantile choices
    (3,7%), (5,15%), (10,30%). Confirm d̂ stable.

  [#7 Structured-marginal calibration] Verify M1 estimator behaves
    correctly on synthetic ternary data with K-cache-like nonzero rate
    (~62%), not uniform 67%.

  [#8 More prompts] Use combined corpus data/c_dump + data/c_dump_v2
    (N = 12300, 7 distinct prompts). Original run was N=2400 from
    ~2 prompts.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from load_k_signatures import (
    HEAD_DIM,
    B2_BITS,
    THRESHOLD_TAU,
    collect_all,
    threshold_extract,
    b2_signature,
    pairwise_hamming_int8,
    pairwise_l2,
)
from m1_estimator_v2 import (
    estimate_id_fixed_radii,
    estimate_id_twonn,
    auto_choose_radii,
    log_cumvol,
)
from run_phase_alpha import (
    m2_topology,
    m3_betti0_persistence,
    m1_l2_twonn,
)
from scipy.optimize import brentq
from scipy.stats import wasserstein_distance


RNG_SEED = 20260512
N_BOOTSTRAP = 200
DUMP_DIRS = ["data/c_dump", "data/c_dump_v2"]


# ============================================================================
# B4: structured binary baseline (sign of top-PC projection)
# ============================================================================

def b4_pca_sign(K_int32: np.ndarray, n_bits: int = B2_BITS, seed: int = 0) -> np.ndarray:
    """Project K to top-n_bits principal directions (SVD), then sign.

    If K has rank < n_bits, pad with random Gaussian projections (rare
    for n_bits=203 vs HEAD_DIM=128: we always pad up by 75 dims).
    """
    K_f = K_int32.astype(np.float64)
    # Center
    K_c = K_f - K_f.mean(axis=0, keepdims=True)
    # SVD
    U, S, Vt = np.linalg.svd(K_c, full_matrices=False)
    # Principal components: each column of Vt.T is a direction; we keep top-D
    n_avail = Vt.shape[0]  # min(N, HEAD_DIM) = HEAD_DIM for N >> 128
    pc_proj = K_c @ Vt.T  # (N, n_avail)
    if n_bits > n_avail:
        # Pad with random Gaussian projections orthogonal to PC space
        rng = np.random.default_rng(seed)
        extra = K_c @ rng.standard_normal((K_f.shape[1], n_bits - n_avail))
        pc_proj = np.concatenate([pc_proj, extra], axis=1)
    pc_proj = pc_proj[:, :n_bits]
    sig = np.where(pc_proj >= 0, 1, -1).astype(np.int8)
    return sig


# ============================================================================
# Building all signatures including B4
# ============================================================================

def build_all_signatures(K_raw: np.ndarray, tau: int = THRESHOLD_TAU,
                         seed_b2: int = 7, seed_b3: int = 13) -> dict:
    return {
        "substrate": threshold_extract(K_raw, tau=tau),
        "B1_raw":    K_raw.astype(np.float64),
        "B2_sign":   b2_signature(K_raw, seed=seed_b2),         # random projection
        "B3_sign":   b2_signature(K_raw, seed=seed_b3),
        "B4_pca":    b4_pca_sign(K_raw),                         # structured
    }


def compute_distances_all(sigs: dict) -> dict:
    return {
        "substrate": pairwise_hamming_int8(sigs["substrate"]),
        "B1_raw":    pairwise_l2(sigs["B1_raw"]),
        "B2_sign":   pairwise_hamming_int8(sigs["B2_sign"]),
        "B3_sign":   pairwise_hamming_int8(sigs["B3_sign"]),
        "B4_pca":    pairwise_hamming_int8(sigs["B4_pca"]),
    }


# ============================================================================
# Ambient D per representation (for d̂/D normalization)
# ============================================================================

AMBIENT_D = {
    "substrate": HEAD_DIM,        # 128 trits
    "B1_raw":    HEAD_DIM,        # 128 float dims (TwoNN on continuous)
    "B2_sign":   B2_BITS,         # 203 bits
    "B3_sign":   B2_BITS,         # 203 bits
    "B4_pca":    B2_BITS,         # 203 bits (matched to B2/B3)
}


# ============================================================================
# M1 ARCH-A with explicit (t1, t2)
# ============================================================================

def m1_archA(dist: np.ndarray, t1: int = None, t2: int = None,
             quantile_pair: tuple = (0.05, 0.15)) -> dict:
    """ARCH-A with explicit or auto radii. Returns {d_hat, t1, t2}."""
    if t1 is None or t2 is None:
        # Pick from data quantiles
        N = dist.shape[0]
        iu = np.triu_indices(N, k=1)
        flat = dist[iu]
        t1 = int(np.quantile(flat, quantile_pair[0]))
        t2 = int(np.quantile(flat, quantile_pair[1]))
        if t2 <= t1:
            t2 = t1 + 1
    d = estimate_id_fixed_radii(dist, t1=t1, t2=t2)
    return {"d_hat": float(d), "t1": int(t1), "t2": int(t2)}


def m1_for(dist: np.ndarray, rep_name: str,
           quantile_pair: tuple = (0.05, 0.15)) -> dict:
    """Dispatch M1: TwoNN-L2 for B1_raw, ARCH-A otherwise."""
    if rep_name == "B1_raw":
        m = m1_l2_twonn(dist)
        return {"d_hat": m.get("d_hat", float("nan"))}
    return m1_archA(dist, quantile_pair=quantile_pair)


# ============================================================================
# [#1] M3 NEW: Wasserstein-distance between bar distributions
# ============================================================================

def bar_distribution(dist: np.ndarray) -> np.ndarray:
    """All persistence bar lengths (diffs between consecutive merge distances)."""
    res = m3_betti0_persistence(dist)
    # We need the raw bar lengths; rerun with same logic but return all bars
    N = dist.shape[0]
    iu = np.triu_indices(N, k=1)
    edge_d = dist[iu]
    order = np.argsort(edge_d, kind="stable")
    from run_phase_alpha import DSU
    dsu = DSU(N)
    n_comp = N
    merge_distances = []
    for d, i, j in zip(edge_d[order], iu[0][order], iu[1][order]):
        if dsu.union(int(i), int(j)):
            merge_distances.append(float(d))
            n_comp -= 1
            if n_comp == 1:
                break
    if len(merge_distances) < 2:
        return np.array([0.0])
    diffs = np.diff([0.0] + merge_distances)
    return diffs


def m3_wasserstein_verdict(dist_dict: dict) -> dict:
    """New M3 verdict: substrate's bar distribution differs from B2's by
    more than B3-vs-B2 baseline. Robust to integer-distance degeneracy.
    """
    bars = {rep: bar_distribution(d) for rep, d in dist_dict.items()
            if rep != "B1_raw"}
    # Substrate vs B2: distance between distributions
    w_sub_B2 = wasserstein_distance(bars["substrate"], bars["B2_sign"])
    # B3 vs B2: baseline variation between two random projections
    w_B3_B2 = wasserstein_distance(bars["B3_sign"], bars["B2_sign"])
    # B4 vs B2: structured-binary vs random-binary
    w_B4_B2 = wasserstein_distance(bars["B4_pca"], bars["B2_sign"])
    return {
        "w_substrate_vs_B2": float(w_sub_B2),
        "w_B3_vs_B2_null":    float(w_B3_B2),
        "w_B4_vs_B2":         float(w_B4_B2),
        "substrate_distinctive": bool(w_sub_B2 > 2 * w_B3_B2),
        "n_bars_substrate":   int(len(bars["substrate"])),
    }


# ============================================================================
# [#7] Calibration on K-cache-matched synthetic (NEW)
# ============================================================================

def synthetic_structured_ternary(N: int, d_true: int, p_nonzero: float = 0.62,
                                 D_ambient: int = 128, seed: int = 42):
    """Like synthetic_ternary_data but with controlled nonzero rate."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(D_ambient)
    real_cells = perm[:d_true]
    pad_cells = perm[d_true:]
    sigs = np.zeros((N, D_ambient), dtype=np.int8)
    # Real cells: Bernoulli(p_nonzero) for nonzero, then ±1 uniform
    nz = rng.random((N, d_true)) < p_nonzero
    signs = rng.choice([-1, 1], size=(N, d_true)).astype(np.int8)
    sigs[:, real_cells] = (nz * signs).astype(np.int8)
    # Padding fixed constants
    pad_constants = rng.integers(-1, 2, size=D_ambient - d_true, dtype=np.int8)
    sigs[:, pad_cells] = pad_constants[np.newaxis, :]
    return sigs


def calibrate_structured(targets=(10, 20, 50, 100), N: int = 500,
                          p_nonzero: float = 0.62):
    """Re-run calibration on structured (non-uniform) synthetic data."""
    rows = []
    for d_true in targets:
        sigs = synthetic_structured_ternary(N=N, d_true=d_true,
                                             p_nonzero=p_nonzero)
        d = pairwise_hamming_int8(sigs)
        m = m1_archA(d)
        d_hat = m["d_hat"]
        err = abs(d_hat - d_true) / d_true
        rows.append((d_true, d_hat, err, m["t1"], m["t2"]))
    return rows


# ============================================================================
# [#3] τ sensitivity sweep on substrate
# ============================================================================

def tau_sweep(K_raw: np.ndarray, dists_baseline: dict, taus=(2000, 5000, 10000, 20000)):
    """For each τ, recompute substrate signature → distance → M1 d̂.
    Compare to B2 (which is τ-independent).
    """
    results = []
    b2_dist = dists_baseline["B2_sign"]
    b2_m1 = m1_archA(b2_dist)
    for tau in taus:
        sig = threshold_extract(K_raw, tau=tau)
        nonzero = np.mean(sig != 0)
        d = pairwise_hamming_int8(sig)
        m1 = m1_archA(d)
        abs_gap = (b2_m1["d_hat"] - m1["d_hat"]) / b2_m1["d_hat"]
        norm_gap = (b2_m1["d_hat"] / AMBIENT_D["B2_sign"]
                    - m1["d_hat"] / AMBIENT_D["substrate"])
        results.append({
            "tau": tau,
            "nonzero_frac": float(nonzero),
            "d_hat_substrate": m1["d_hat"],
            "d_hat_B2": b2_m1["d_hat"],
            "abs_rel_gap": float(abs_gap),
            "norm_pp_gap": float(norm_gap),
            "t1_sub": m1["t1"], "t2_sub": m1["t2"],
        })
    return results


# ============================================================================
# [#6] (t1, t2) quantile sensitivity for M1
# ============================================================================

def tt_sensitivity(dists: dict, quantile_pairs=((0.03, 0.07), (0.05, 0.15),
                                                  (0.10, 0.30))):
    """Run M1 ARCH-A at multiple quantile-pair choices per representation."""
    out = {}
    for qp in quantile_pairs:
        row = {}
        for rep in ("substrate", "B2_sign", "B3_sign", "B4_pca"):
            row[rep] = m1_archA(dists[rep], quantile_pair=qp)
        out[f"q{qp[0]:.2f}-{qp[1]:.2f}"] = row
    return out


# ============================================================================
# [#5] Bootstrap CIs for M1, M2 reciprocity[k=10], M2 Gini[k=10],
#      and a robust M3 statistic (median bar length)
# ============================================================================

def bootstrap_all(K_raw: np.ndarray, idx_pool: np.ndarray, B: int = N_BOOTSTRAP,
                  N_sub: int = 500, rng=None):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    reps = ["substrate", "B1_raw", "B2_sign", "B3_sign", "B4_pca"]
    metrics = ["m1_d_hat", "m1_d_over_D", "m2_recip_k10", "m2_gini_k10",
               "m3_median_bar"]
    store = {r: {m: [] for m in metrics} for r in reps}
    for b in range(B):
        sub = rng.choice(idx_pool, size=min(N_sub, len(idx_pool)), replace=True)
        K_sub = K_raw[sub]
        sigs = build_all_signatures(K_sub)
        dists = compute_distances_all(sigs)
        for rep in reps:
            dist = dists[rep]
            # M1
            if rep == "B1_raw":
                d_hat = m1_l2_twonn(dist).get("d_hat", float("nan"))
            else:
                d_hat = m1_archA(dist)["d_hat"]
            store[rep]["m1_d_hat"].append(d_hat)
            store[rep]["m1_d_over_D"].append(d_hat / AMBIENT_D[rep]
                                              if np.isfinite(d_hat) else float("nan"))
            # M2 reciprocity at k=10
            m2 = m2_topology(dist, k_values=(10,))
            store[rep]["m2_recip_k10"].append(m2["reciprocity"][0])
            store[rep]["m2_gini_k10"].append(m2["degree_gini"][0])
            # M3 median bar
            bars = bar_distribution(dist)
            store[rep]["m3_median_bar"].append(float(np.median(bars)))
        if (b + 1) % 25 == 0:
            print(f"    bootstrap iter {b + 1}/{B} done")
    summary = {}
    for r in reps:
        summary[r] = {}
        for m in metrics:
            arr = np.array(store[r][m])
            arr_fin = arr[np.isfinite(arr)]
            if len(arr_fin) < 5:
                summary[r][m] = {"mean": float("nan"), "ci_lo": float("nan"),
                                 "ci_hi": float("nan"), "n": int(len(arr_fin))}
            else:
                summary[r][m] = {
                    "mean":  float(np.mean(arr_fin)),
                    "ci_lo": float(np.quantile(arr_fin, 0.025)),
                    "ci_hi": float(np.quantile(arr_fin, 0.975)),
                    "n":     int(len(arr_fin)),
                }
    return summary


# ============================================================================
# [#2] Revised M1 verdict: BOTH absolute gap ≥20% AND normalized gap ≥5pp
# ============================================================================

def m1_verdict_revised(boot: dict) -> dict:
    """Revised M1 PASS rule: substrate is significantly LOWER d̂ than B2
    BOTH in absolute relative terms (≥20%) AND in normalized d̂/D terms
    (≥5pp). CIs must be disjoint on the chosen metric.
    """
    s = boot["substrate"]
    b2 = boot["B2_sign"]
    b4 = boot["B4_pca"]
    abs_gap = (b2["m1_d_hat"]["mean"] - s["m1_d_hat"]["mean"]) / b2["m1_d_hat"]["mean"]
    norm_gap = b2["m1_d_over_D"]["mean"] - s["m1_d_over_D"]["mean"]
    abs_disjoint = s["m1_d_hat"]["ci_hi"] < b2["m1_d_hat"]["ci_lo"]
    norm_disjoint = s["m1_d_over_D"]["ci_hi"] < b2["m1_d_over_D"]["ci_lo"]
    # B4 (structured binary) comparison
    abs_gap_B4 = (b4["m1_d_hat"]["mean"] - s["m1_d_hat"]["mean"]) / b4["m1_d_hat"]["mean"]
    norm_gap_B4 = b4["m1_d_over_D"]["mean"] - s["m1_d_over_D"]["mean"]
    abs_disjoint_B4 = s["m1_d_hat"]["ci_hi"] < b4["m1_d_hat"]["ci_lo"]
    norm_disjoint_B4 = s["m1_d_over_D"]["ci_hi"] < b4["m1_d_over_D"]["ci_lo"]
    # PASS only if both abs and norm gaps cleared vs BOTH B2 and B4
    pass_B2_abs = abs_gap >= 0.20 and abs_disjoint
    pass_B2_norm = norm_gap >= 0.05 and norm_disjoint
    pass_B4_abs = abs_gap_B4 >= 0.20 and abs_disjoint_B4
    pass_B4_norm = norm_gap_B4 >= 0.05 and norm_disjoint_B4
    return {
        "abs_gap_vs_B2": float(abs_gap),
        "norm_gap_vs_B2": float(norm_gap),
        "abs_gap_vs_B4": float(abs_gap_B4),
        "norm_gap_vs_B4": float(norm_gap_B4),
        "pass_B2_abs": pass_B2_abs,
        "pass_B2_norm": pass_B2_norm,
        "pass_B4_abs": pass_B4_abs,
        "pass_B4_norm": pass_B4_norm,
        # PASS criterion: clear both absolute AND normalized against BOTH B2 and B4
        "pass": all([pass_B2_abs, pass_B2_norm, pass_B4_abs, pass_B4_norm]),
        # Partial pass: substrate beats RANDOM (B2) on both metrics but not structured (B4)
        "partial_pass_random_only": (
            pass_B2_abs and pass_B2_norm and not (pass_B4_abs and pass_B4_norm)
        ),
    }


def m2_verdict(reps_pooled: dict) -> dict:
    """M2 PASS rule unchanged from FROZEN spec."""
    s = reps_pooled["substrate"]["M2"]
    b2 = reps_pooled["B2_sign"]["M2"]
    pass_per_k = []
    for i, k in enumerate(s["k_values"]):
        gap_recip = s["reciprocity"][i] - b2["reciprocity"][i]
        gap_gini = b2["degree_gini"][i] - s["degree_gini"][i]
        pass_per_k.append(gap_recip >= 0.05 and gap_gini >= 0.05)
    return {
        "pass_per_k": pass_per_k,
        "k_values": s["k_values"],
        "pass": sum(pass_per_k) >= 3,
    }


# ============================================================================
# Driver
# ============================================================================

def main(out_dir: str = "experiments/phase_alpha/results"):
    os.makedirs(out_dir, exist_ok=True)
    print("=== Phase α REMEDIATED ===\n")

    # [#8] Combined corpus
    print(f"Loading K-signatures from {DUMP_DIRS} …")
    K, meta = collect_all(DUMP_DIRS)
    N = K.shape[0]
    print(f"  N = {N}, D = {HEAD_DIM}")
    print(f"  prompts: {sorted(set(meta['prompt_id']))}\n")

    results: dict[str, Any] = {
        "config": {
            "N_total": int(N),
            "HEAD_DIM": HEAD_DIM,
            "B2_BITS": B2_BITS,
            "default_tau": THRESHOLD_TAU,
            "N_bootstrap": N_BOOTSTRAP,
            "dump_dirs": DUMP_DIRS,
            "prompts": sorted(set(meta["prompt_id"])),
        },
    }

    # [#7] Calibration on structured (K-cache-like) synthetic
    print("[#7] Calibration on structured synthetic (p_nonzero=0.62)")
    cal_rows = calibrate_structured(p_nonzero=0.62)
    print(f"  {'d_true':>6} {'d_hat':>8} {'err':>8} {'t1':>4} {'t2':>4}")
    cal_results = []
    for d_true, d_hat, err, t1, t2 in cal_rows:
        verdict = "PASS" if err < 0.20 else "FAIL"
        print(f"  {d_true:>6d} {d_hat:>8.2f} {err:>8.2%} {t1:>4d} {t2:>4d}  {verdict}")
        cal_results.append({"d_true": d_true, "d_hat": d_hat, "rel_err": err,
                            "t1": t1, "t2": t2})
    results["calibration_structured"] = cal_results
    cal_all_pass = all(r["rel_err"] < 0.20 for r in cal_results)
    print(f"  → Structured calibration: {'PASS' if cal_all_pass else 'FAIL'}\n")

    # Build signatures + distances on FULL corpus, default τ
    print("Building signatures (all baselines, full corpus) …")
    t0 = time.time()
    sigs = build_all_signatures(K, tau=THRESHOLD_TAU)
    print(f"  signatures: {time.time() - t0:.1f}s")
    # For N=12300, full pairwise distance matrices are heavy; sample N_eff
    N_eff = 1500  # large enough for stable M1, M2, M3
    rng = np.random.default_rng(RNG_SEED)
    sample_idx = rng.choice(N, size=N_eff, replace=False)
    K_s = K[sample_idx]
    sigs_s = build_all_signatures(K_s, tau=THRESHOLD_TAU)
    print(f"  sampled N_eff={N_eff} for full-corpus measures")
    t0 = time.time()
    dists_s = compute_distances_all(sigs_s)
    print(f"  distances: {time.time() - t0:.1f}s\n")

    # Per-representation full measures
    print("[POOLED N_eff=1500] M1+M2+M3 per representation:")
    pooled_results = {}
    for rep in ("substrate", "B1_raw", "B2_sign", "B3_sign", "B4_pca"):
        t_rep = time.time()
        dist = dists_s[rep]
        if rep == "B1_raw":
            m1 = m1_l2_twonn(dist)
            d_hat = m1.get("d_hat", float("nan"))
        else:
            m1 = m1_archA(dist)
            d_hat = m1["d_hat"]
        m2 = m2_topology(dist)
        m3 = m3_betti0_persistence(dist)
        d_over_D = d_hat / AMBIENT_D[rep] if np.isfinite(d_hat) else float("nan")
        pooled_results[rep] = {"M1": m1, "M2": m2, "M3": m3,
                                "d_over_D": d_over_D,
                                "elapsed_s": time.time() - t_rep}
        print(f"  {rep:12s} d̂={d_hat:7.2f}  d̂/D={d_over_D:.3f}  "
              f"recip[k=10]={m2['reciprocity'][1]:.3f}  "
              f"Gini[k=10]={m2['degree_gini'][1]:.3f}  "
              f"longest_bar={m3['longest_bar']:.2f}  ({time.time() - t_rep:.1f}s)")
    results["pooled"] = pooled_results

    # [#1] M3 Wasserstein verdict
    print("\n[#1] M3 NEW (Wasserstein between bar distributions):")
    m3_new = m3_wasserstein_verdict(dists_s)
    print(f"  W(substrate, B2) = {m3_new['w_substrate_vs_B2']:.3f}")
    print(f"  W(B3, B2) null   = {m3_new['w_B3_vs_B2_null']:.3f}")
    print(f"  W(B4, B2)        = {m3_new['w_B4_vs_B2']:.3f}")
    print(f"  substrate distinctive (W_sub_B2 > 2× W_B3_B2_null): "
          f"{m3_new['substrate_distinctive']}")
    results["m3_wasserstein"] = m3_new

    # [#3] τ sweep
    print("\n[#3] τ sensitivity sweep:")
    sweep = tau_sweep(K_s, dists_s)
    print(f"  {'τ':>6} {'nz_frac':>8} {'d̂_sub':>8} {'d̂_B2':>8} "
          f"{'abs_gap':>8} {'norm_pp':>8}")
    for r in sweep:
        print(f"  {r['tau']:>6d} {r['nonzero_frac']:>8.3f} "
              f"{r['d_hat_substrate']:>8.2f} {r['d_hat_B2']:>8.2f} "
              f"{r['abs_rel_gap']:>+7.1%} {r['norm_pp_gap']:>+7.3f}")
    results["tau_sweep"] = sweep

    # [#6] (t1, t2) quantile sensitivity
    print("\n[#6] (t1, t2) quantile sensitivity:")
    tt = tt_sensitivity(dists_s)
    print(f"  {'qpair':>14} {'subs':>8} {'B2':>8} {'B3':>8} {'B4':>8}")
    for qp, row in tt.items():
        vals = [f"{row[r]['d_hat']:.2f}" for r in ("substrate", "B2_sign",
                                                     "B3_sign", "B4_pca")]
        print(f"  {qp:>14s} {vals[0]:>8s} {vals[1]:>8s} {vals[2]:>8s} {vals[3]:>8s}")
    results["tt_sensitivity"] = tt

    # [#5] Bootstrap CIs for everything
    print(f"\n[#5] Bootstrap (B={N_BOOTSTRAP}, N_sub=500) on M1, M2, M3:")
    t0 = time.time()
    boot = bootstrap_all(K, np.arange(N), B=N_BOOTSTRAP, N_sub=500)
    print(f"  bootstrap elapsed: {time.time() - t0:.1f}s")
    for rep in ("substrate", "B2_sign", "B3_sign", "B4_pca"):
        b = boot[rep]
        print(f"  {rep:12s} "
              f"d̂={b['m1_d_hat']['mean']:7.2f} [{b['m1_d_hat']['ci_lo']:.2f},{b['m1_d_hat']['ci_hi']:.2f}]  "
              f"d̂/D={b['m1_d_over_D']['mean']:.3f} [{b['m1_d_over_D']['ci_lo']:.3f},{b['m1_d_over_D']['ci_hi']:.3f}]  "
              f"recip={b['m2_recip_k10']['mean']:.3f}")
    results["bootstrap"] = boot

    # [#2] Revised M1 verdict (vs both B2 and B4)
    print("\n[#2] Revised M1 verdict:")
    m1v = m1_verdict_revised(boot)
    print(f"  vs B2 (random):     abs_gap={m1v['abs_gap_vs_B2']:+.1%}  "
          f"norm_pp={m1v['norm_gap_vs_B2']:+.3f}  "
          f"abs_pass={m1v['pass_B2_abs']}  norm_pass={m1v['pass_B2_norm']}")
    print(f"  vs B4 (structured): abs_gap={m1v['abs_gap_vs_B4']:+.1%}  "
          f"norm_pp={m1v['norm_gap_vs_B4']:+.3f}  "
          f"abs_pass={m1v['pass_B4_abs']}  norm_pass={m1v['pass_B4_norm']}")
    print(f"  M1 OVERALL PASS (all four required): {m1v['pass']}")
    if m1v["partial_pass_random_only"]:
        print(f"  (partial: substrate beats random B2 but not structured B4)")
    results["m1_verdict_revised"] = m1v

    # M2 verdict (FROZEN)
    print("\n[M2 verdict (FROZEN rule)]:")
    m2v = m2_verdict(pooled_results)
    print(f"  pass_per_k: {m2v['pass_per_k']}")
    print(f"  M2 PASS: {m2v['pass']}")
    results["m2_verdict"] = m2v

    # M3 NEW verdict
    m3_pass = m3_new["substrate_distinctive"]
    print(f"\n[M3 verdict (NEW Wasserstein rule)]: {m3_pass}")
    results["m3_verdict"] = {"pass": m3_pass, **m3_new}

    # Final verdict
    n_pass = int(m1v["pass"]) + int(m2v["pass"]) + int(m3_pass)
    if n_pass >= 2:
        label = "VALIDATED"
    elif n_pass == 1:
        label = "MIXED"
    else:
        label = "FALSIFIED"
    # Note partial-pass on M1 (random-only) lowers confidence
    if m1v.get("partial_pass_random_only"):
        label_note = " (M1 partial: beats random, not structured)"
    else:
        label_note = ""
    print(f"\n>>> SUBSTRATE GEOMETRIC CLAIM (remediated): "
          f"{label}{label_note} ({n_pass}/3 measures clear) <<<")
    results["final_verdict"] = {
        "label": label,
        "n_pass": n_pass,
        "m1_partial_random_only": m1v.get("partial_pass_random_only", False),
    }

    # Persist
    def serial(x):
        if isinstance(x, dict):
            return {k: serial(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [serial(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        return x

    out_path = os.path.join(out_dir, "phase_alpha_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(serial(results), f, indent=2)
    print(f"\nResults written to {out_path}")
    return results


if __name__ == "__main__":
    main()
