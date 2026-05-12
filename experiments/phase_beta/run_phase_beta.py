"""Phase β full run — substrate distance is L1 on trits (cell-graph metric).

Pre-registered tests (FROZEN per td28_phase_beta_prereg_2026-05-12.md):

  P1: L1-substrate has lower d̂ / D_max than B0 (Hamming-substrate).
      Tests whether the L1 metric reveals structure categorical Hamming
      was hiding.

  P2: L1-substrate matches or beats B4 (PCA+sign binary) on d̂/D_max.
      Tests whether substrate with the right metric competes with
      structured binary at equal capacity.

  P3: L1-substrate beats B5 (scrambled-ternary) on d̂/D_max.
      Tests whether the centrality of 0 in the path graph is
      load-bearing.

Verdict: ALL THREE must clear. Bootstrap CIs disjoint on d̂/D_max.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np

THIS = os.path.dirname(__file__)
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(THIS, "..", "phase_alpha"))

from load_k_signatures import (
    HEAD_DIM, B2_BITS, THRESHOLD_TAU,
    collect_all, threshold_extract, b2_signature,
    pairwise_hamming_int8,
)
from run_phase_alpha_v2 import b4_pca_sign
from m1_l1_estimator import (
    pairwise_L1_int8,
    estimate_id_L1,
    cell_pmf_from_data,
    substrate_cell_pmf,
)


RNG_SEED = 20260512
N_BOOTSTRAP = 200
DUMP_DIRS = ["data/c_dump", "data/c_dump_v2"]


# ============================================================================
# Representations
# ============================================================================

def build_phase_beta_signatures(K_raw: np.ndarray, tau: int = THRESHOLD_TAU,
                                  seed_b2: int = 7, seed_b3: int = 13):
    """All Phase β representations + their natural-metric distances.

    Returns dict of (signatures, distance_fn_name).
    """
    sub = threshold_extract(K_raw, tau=tau)
    return {
        "substrate_L1":      (sub, "L1"),
        "B0_Hamming_sub":    (sub, "Hamming"),
        "B2_sign":           (b2_signature(K_raw, seed=seed_b2), "Hamming"),
        "B3_sign":           (b2_signature(K_raw, seed=seed_b3), "Hamming"),
        "B4_pca":            (b4_pca_sign(K_raw), "Hamming"),
        "B5_scrambled_sub":  (sub, "scrambled"),
    }


# Scrambled-ternary cell distance lookup table:
# Under the standard path graph (-1 — 0 — +1) with 0 as center:
#   d_L1(a, b) = |a − b|, giving d(-1,0)=1, d(0,+1)=1, d(-1,+1)=2.
# To remove the centrality of 0, we permute the path so +1 is the center:
#   new path: -1 — +1 — 0
#   d_scrambled(-1, +1) = 1
#   d_scrambled(+1, 0)  = 1
#   d_scrambled(-1, 0)  = 2
# Lookup table indexed by (a+1, b+1):
SCRAMBLED_LUT = np.array([
    # b=-1, b=0, b=+1
    [0,    2,    1],   # a=-1
    [2,    0,    1],   # a= 0
    [1,    1,    0],   # a=+1
], dtype=np.int8)


def pairwise_scrambled_int8(signatures: np.ndarray) -> np.ndarray:
    """L1-style pairwise distance under the scrambled cell-graph (+1 as
    center instead of 0).
    """
    s = signatures.astype(np.int64) + 1  # map {-1, 0, +1} → {0, 1, 2}
    N, D = s.shape
    out = np.zeros((N, N), dtype=np.int32)
    CHUNK = max(1, min(N, 1024))
    for i in range(0, N, CHUNK):
        a = s[i:i + CHUNK, None, :]   # (chunk, 1, D)
        b = s[None, :, :]             # (1, N, D)
        # Index into LUT: shape (chunk, N, D)
        out[i:i + CHUNK] = SCRAMBLED_LUT[a, b].sum(axis=-1)
    return out


def compute_distance(rep_name: str, sigs: np.ndarray) -> np.ndarray:
    """Dispatch per representation/metric pair."""
    if rep_name == "substrate_L1":
        return pairwise_L1_int8(sigs)
    if rep_name == "B5_scrambled_sub":
        return pairwise_scrambled_int8(sigs)
    return pairwise_hamming_int8(sigs)


# ============================================================================
# d_max per representation (for d̂/D_max normalization)
# ============================================================================

D_MAX = {
    "substrate_L1":      2 * HEAD_DIM,   # L1 max on D=128 trits is 2D
    "B0_Hamming_sub":    HEAD_DIM,        # Hamming max on D=128
    "B2_sign":           B2_BITS,         # Hamming max on D=203
    "B3_sign":           B2_BITS,
    "B4_pca":            B2_BITS,
    "B5_scrambled_sub":  2 * HEAD_DIM,    # same L1 range
}


# ============================================================================
# M1 dispatch — L1 estimator for {substrate_L1, B5}, Hamming Macocco for others
# ============================================================================

def m1_for(rep_name: str, dist: np.ndarray,
           pmf_cache: dict, signatures: np.ndarray):
    """Run M1 fixed-radii estimator with the correct shell-volume model."""
    if rep_name == "substrate_L1":
        # L1 cell PMF from substrate signatures' empirical marginals
        if "substrate_emp" not in pmf_cache:
            pmf_cache["substrate_emp"] = cell_pmf_from_data(signatures)
        m = estimate_id_L1(dist, pmf_cell=pmf_cache["substrate_emp"])
        return m
    if rep_name == "B5_scrambled_sub":
        # Scrambled distance has its own per-cell PMF:
        # For two i.i.d. cells with substrate marginals,
        #   P(scrambled_dist = 0) = P(same value)  = p_-² + p_0² + p_+²
        #   P(scrambled_dist = 1) = 2(p_- p_+ + p_+ p_0)  (since (-1,+1)=1, (+1,0)=1)
        #   P(scrambled_dist = 2) = 2(p_- p_0)            (since (-1,0)=2)
        if "scrambled_emp" not in pmf_cache:
            flat = signatures.flatten()
            p_m = float(np.mean(flat == -1))
            p_z = float(np.mean(flat == 0))
            p_p = float(np.mean(flat == 1))
            total = p_m + p_z + p_p
            p_m, p_z, p_p = p_m/total, p_z/total, p_p/total
            pmf = np.array([
                p_m**2 + p_z**2 + p_p**2,
                2*(p_m*p_p + p_p*p_z),
                2*(p_m*p_z),
            ])
            pmf_cache["scrambled_emp"] = pmf
        m = estimate_id_L1(dist, pmf_cell=pmf_cache["scrambled_emp"])
        return m
    # Hamming-Macocco (categorical) — reuse Phase α ARCH-A
    from m1_estimator_v2 import estimate_id_fixed_radii
    N = dist.shape[0]
    iu = np.triu_indices(N, k=1)
    flat = dist[iu]
    t1 = int(np.quantile(flat, 0.05))
    t2 = int(np.quantile(flat, 0.15))
    if t2 <= t1:
        t2 = t1 + 1
    d = estimate_id_fixed_radii(dist, t1=t1, t2=t2)
    n_total = int(np.sum(flat <= t1)) * 2
    k_total = int(np.sum(flat <= t2)) * 2
    return {"d_hat": float(d), "t1": t1, "t2": t2,
            "n_total": n_total, "k_total": k_total}


# ============================================================================
# Bootstrap CI loop
# ============================================================================

def bootstrap_phase_beta(K: np.ndarray, B: int = N_BOOTSTRAP, N_sub: int = 500,
                          rng=None):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    reps = list(D_MAX.keys())
    store = {r: {"d_hat": [], "d_over_Dmax": []} for r in reps}
    pmf_cache: dict = {}
    for b in range(B):
        idx = rng.choice(len(K), size=N_sub, replace=True)
        K_sub = K[idx]
        sig_distance = build_phase_beta_signatures(K_sub)
        for r in reps:
            sigs, _ = sig_distance[r]
            d_mat = compute_distance(r, sigs)
            m = m1_for(r, d_mat, pmf_cache, sigs)
            d_hat = m.get("d_hat", float("nan"))
            store[r]["d_hat"].append(d_hat)
            store[r]["d_over_Dmax"].append(d_hat / D_MAX[r]
                                            if np.isfinite(d_hat) else float("nan"))
        # Reset PMF cache per bootstrap iter (data-dependent)
        pmf_cache.clear()
        if (b + 1) % 25 == 0:
            print(f"    bootstrap iter {b + 1}/{B}")
    summary = {}
    for r in reps:
        out = {}
        for k in ("d_hat", "d_over_Dmax"):
            arr = np.array(store[r][k])
            arr_fin = arr[np.isfinite(arr)]
            if len(arr_fin) < 5:
                out[k] = {"mean": float("nan"), "ci_lo": float("nan"),
                          "ci_hi": float("nan"), "n": int(len(arr_fin))}
            else:
                out[k] = {
                    "mean":  float(np.mean(arr_fin)),
                    "ci_lo": float(np.quantile(arr_fin, 0.025)),
                    "ci_hi": float(np.quantile(arr_fin, 0.975)),
                    "n":     int(len(arr_fin)),
                }
        summary[r] = out
    return summary


# ============================================================================
# Verdict
# ============================================================================

def phase_beta_verdict(boot: dict) -> dict:
    """Apply P1, P2, P3 verdict rules."""
    s = boot["substrate_L1"]["d_over_Dmax"]
    b0 = boot["B0_Hamming_sub"]["d_over_Dmax"]
    b4 = boot["B4_pca"]["d_over_Dmax"]
    b5 = boot["B5_scrambled_sub"]["d_over_Dmax"]

    # P1: substrate_L1 < B0_Hamming, CI-disjoint
    p1_gap = b0["mean"] - s["mean"]
    p1_disjoint = s["ci_hi"] < b0["ci_lo"]
    p1_pass = (p1_gap > 0) and p1_disjoint

    # P2: substrate_L1 ≤ B4, CI not above B4
    # Strict: substrate_L1 mean < B4 mean AND substrate CI upper < B4 CI lower
    p2_gap = b4["mean"] - s["mean"]
    p2_disjoint_strict = s["ci_hi"] < b4["ci_lo"]
    p2_pass_strict = (p2_gap > 0) and p2_disjoint_strict
    # Lenient: substrate_L1 mean ≤ B4 mean (CIs may overlap)
    p2_pass_lenient = p2_gap >= -0.01  # within 1pp tolerance

    # P3: substrate_L1 < B5_scrambled, CI-disjoint
    p3_gap = b5["mean"] - s["mean"]
    p3_disjoint = s["ci_hi"] < b5["ci_lo"]
    p3_pass = (p3_gap > 0) and p3_disjoint

    return {
        "P1": {"gap": float(p1_gap), "disjoint": bool(p1_disjoint),
               "pass": bool(p1_pass),
               "summary": f"substrate_L1 d̂/Dmax={s['mean']:.3f} vs B0_Hamming={b0['mean']:.3f}"},
        "P2": {"gap": float(p2_gap), "disjoint": bool(p2_disjoint_strict),
               "pass_strict": bool(p2_pass_strict),
               "pass_lenient": bool(p2_pass_lenient),
               "summary": f"substrate_L1={s['mean']:.3f} vs B4={b4['mean']:.3f}"},
        "P3": {"gap": float(p3_gap), "disjoint": bool(p3_disjoint),
               "pass": bool(p3_pass),
               "summary": f"substrate_L1={s['mean']:.3f} vs B5_scrambled={b5['mean']:.3f}"},
        "n_pass": int(p1_pass) + int(p2_pass_strict) + int(p3_pass),
    }


# ============================================================================
# Driver
# ============================================================================

def main(out_dir: str = "experiments/phase_beta/results"):
    os.makedirs(out_dir, exist_ok=True)
    print("=== Phase β (substrate L1 cell-graph metric) ===\n")

    K, meta = collect_all(DUMP_DIRS)
    N = K.shape[0]
    print(f"Loaded N = {N} K-signatures, D = {HEAD_DIM}")
    print(f"  prompts: {sorted(set(meta['prompt_id']))}\n")

    # Pooled measurement at N_eff = 1500
    N_eff = 1500
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(N, size=N_eff, replace=False)
    K_eff = K[idx]
    print(f"Pooled measurement at N_eff = {N_eff}\n")

    sig_dist = build_phase_beta_signatures(K_eff)
    pmf_cache: dict = {}

    print("[POOLED] M1 per representation:")
    pooled = {}
    for r in D_MAX:
        sigs, metric = sig_dist[r]
        t0 = time.time()
        d_mat = compute_distance(r, sigs)
        m = m1_for(r, d_mat, pmf_cache, sigs)
        d_hat = m.get("d_hat", float("nan"))
        d_over_Dmax = d_hat / D_MAX[r] if np.isfinite(d_hat) else float("nan")
        pooled[r] = {"d_hat": d_hat, "d_over_Dmax": d_over_Dmax,
                     "Dmax": D_MAX[r], "t1": m.get("t1"), "t2": m.get("t2"),
                     "metric": metric, "elapsed_s": time.time() - t0}
        print(f"  {r:22s} [{metric:9s}]  d̂={d_hat:7.2f}  Dmax={D_MAX[r]:3d}  "
              f"d̂/Dmax={d_over_Dmax:.3f}  ({pooled[r]['elapsed_s']:.1f}s)")

    print(f"\nBootstrap (B={N_BOOTSTRAP}, N_sub=500):")
    t0 = time.time()
    boot = bootstrap_phase_beta(K, B=N_BOOTSTRAP, N_sub=500)
    print(f"  bootstrap elapsed: {time.time() - t0:.1f}s\n")

    print("Bootstrap CIs on d̂/Dmax:")
    for r in D_MAX:
        b = boot[r]["d_over_Dmax"]
        print(f"  {r:22s} mean={b['mean']:.3f}  CI=[{b['ci_lo']:.3f}, {b['ci_hi']:.3f}]  (n={b['n']})")

    # Apply verdict rules
    print("\n[Verdict]")
    v = phase_beta_verdict(boot)
    print(f"  P1: {v['P1']['summary']}  gap={v['P1']['gap']:+.3f}  "
          f"disjoint={v['P1']['disjoint']}  PASS={v['P1']['pass']}")
    print(f"  P2: {v['P2']['summary']}  gap={v['P2']['gap']:+.3f}  "
          f"disjoint={v['P2']['disjoint']}  PASS={v['P2']['pass_strict']}  "
          f"(lenient: {v['P2']['pass_lenient']})")
    print(f"  P3: {v['P3']['summary']}  gap={v['P3']['gap']:+.3f}  "
          f"disjoint={v['P3']['disjoint']}  PASS={v['P3']['pass']}")
    if v["n_pass"] == 3:
        label = "VALIDATED"
    elif v["n_pass"] == 0:
        label = "FALSIFIED"
    else:
        label = "MIXED"
    print(f"\n>>> PHASE β VERDICT: {label}  ({v['n_pass']}/3 P-rules clear) <<<")

    results = {
        "config": {"N_total": int(N), "N_eff": N_eff, "B": N_BOOTSTRAP,
                   "dump_dirs": DUMP_DIRS, "tau": THRESHOLD_TAU},
        "pooled": pooled,
        "bootstrap": boot,
        "verdict": {"label": label, "n_pass": v["n_pass"], "details": v},
    }

    def to_serial(x):
        if isinstance(x, dict):
            return {k: to_serial(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [to_serial(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        return x

    out_path = os.path.join(out_dir, "phase_beta_results.json")
    with open(out_path, "w") as f:
        json.dump(to_serial(results), f, indent=2)
    print(f"\nResults written to {out_path}")
    return results


if __name__ == "__main__":
    main()
