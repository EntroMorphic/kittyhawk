"""Phase γ — full remediation per the concerns journal.

Addresses every concern from journal/td28_phase_beta_redteam:

  γ-A: Multi-normalization grid (d̂/Dmax, d̂/D_ambient, d̂ absolute).
       A finding is "robust" only if it clears under all defensible
       normalizations.

  γ-B: Cross-validate Macocco with correlation dimension (Grassberger-
       Procaccia). Direction agreement matters more than value
       agreement (correlation dim has known low-bias at high d).

  γ-C: B5 mirror (scrambled with -1-as-center) for symmetry control.

  γ-D: Close-regime stratification under L1.

  γ-E: τ sensitivity sweep for substrate under L1.

  γ-F: Shuffled-K-cache control — shuffle cells per K-vector to
       destroy learned correlations while preserving marginals.

  γ-G: Correlated-synthetic calibration — test the Macocco estimator
       on data with non-trivial cell correlations (closer to real
       K-cache structure than uniform synthetic).

Output: a robustness matrix where each finding is reported under
all variations of methodology. Verdict per claim is one of:
    ROBUST  — clears under all normalizations + both estimators
    PARTIAL — clears under some but not all
    NOT ROBUST — clears under one specific methodology only
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
sys.path.insert(0, os.path.join(THIS, "..", "phase_beta"))

from load_k_signatures import (
    HEAD_DIM, B2_BITS, THRESHOLD_TAU,
    collect_all, threshold_extract, b2_signature,
    pairwise_hamming_int8,
)
from run_phase_alpha_v2 import b4_pca_sign
from m1_l1_estimator import (
    pairwise_L1_int8, estimate_id_L1,
    cell_pmf_from_data, substrate_cell_pmf,
)
from correlation_dim import correlation_dimension
from run_phase_beta import (
    SCRAMBLED_LUT, pairwise_scrambled_int8,
)


RNG_SEED = 20260512
N_BOOTSTRAP = 100  # reduced for the expanded grid
DUMP_DIRS = ["data/c_dump", "data/c_dump_v2"]


# ============================================================================
# γ-C: mirror scrambled with -1 as center  (path: 0 — -1 — +1)
# ============================================================================

# Under this path, d(-1, 0) = 1, d(-1, +1) = 1, d(0, +1) = 2.
SCRAMBLED_MIRROR_LUT = np.array([
    # b=-1, b=0, b=+1
    [0,    1,    1],   # a=-1
    [1,    0,    2],   # a= 0
    [1,    2,    0],   # a=+1
], dtype=np.int8)


def pairwise_scrambled_mirror_int8(signatures: np.ndarray) -> np.ndarray:
    s = signatures.astype(np.int64) + 1
    N = signatures.shape[0]
    out = np.zeros((N, N), dtype=np.int32)
    CHUNK = max(1, min(N, 1024))
    for i in range(0, N, CHUNK):
        a = s[i:i + CHUNK, None, :]
        b = s[None, :, :]
        out[i:i + CHUNK] = SCRAMBLED_MIRROR_LUT[a, b].sum(axis=-1)
    return out


# ============================================================================
# γ-F: cell-shuffled K-cache (destroys cross-cell correlations)
# ============================================================================

def shuffle_cells_per_row(K: np.ndarray, seed: int = RNG_SEED) -> np.ndarray:
    """For each row independently, permute the D cells. Preserves marginals,
    destroys cross-cell correlations.
    """
    rng = np.random.default_rng(seed)
    out = np.empty_like(K)
    for i in range(K.shape[0]):
        out[i] = K[i, rng.permutation(K.shape[1])]
    return out


# ============================================================================
# Representations
# ============================================================================

REP_CONFIG = {
    # name           → (sig_builder, distance_fn, ambient_D, Dmax, metric_label)
    "substrate_L1":      None,  # built dynamically
    "B0_Hamming_sub":    None,
    "B2_sign":           None,
    "B4_pca":            None,
    "B5_scrambled":      None,
    "B5m_mirror":        None,  # γ-C
}


def build_representations(K: np.ndarray, tau: int = THRESHOLD_TAU,
                           seed_b2: int = 7):
    sub = threshold_extract(K, tau=tau)
    B2 = b2_signature(K, seed=seed_b2)
    B4 = b4_pca_sign(K)
    return {
        "substrate_L1":   {"sigs": sub, "dist_fn": pairwise_L1_int8,
                            "D_amb": HEAD_DIM,  "Dmax": 2 * HEAD_DIM},
        "B0_Hamming_sub": {"sigs": sub, "dist_fn": pairwise_hamming_int8,
                            "D_amb": HEAD_DIM,  "Dmax": HEAD_DIM},
        "B2_sign":        {"sigs": B2,  "dist_fn": pairwise_hamming_int8,
                            "D_amb": B2_BITS,   "Dmax": B2_BITS},
        "B4_pca":         {"sigs": B4,  "dist_fn": pairwise_hamming_int8,
                            "D_amb": B2_BITS,   "Dmax": B2_BITS},
        "B5_scrambled":   {"sigs": sub, "dist_fn": pairwise_scrambled_int8,
                            "D_amb": HEAD_DIM,  "Dmax": 2 * HEAD_DIM},
        "B5m_mirror":     {"sigs": sub, "dist_fn": pairwise_scrambled_mirror_int8,
                            "D_amb": HEAD_DIM,  "Dmax": 2 * HEAD_DIM},
    }


# ============================================================================
# γ-B: Macocco + correlation-dim estimators per rep
# ============================================================================

def m1_dual_estimators(rep_name: str, dist: np.ndarray, sigs: np.ndarray,
                        pmf_cache: dict):
    """Return Macocco d̂ AND correlation d̂."""
    # Correlation dim (always)
    cd = correlation_dimension(dist)

    # Macocco fixed-radii
    if rep_name in ("substrate_L1",):
        if "sub_pmf" not in pmf_cache:
            pmf_cache["sub_pmf"] = cell_pmf_from_data(sigs)
        m = estimate_id_L1(dist, pmf_cell=pmf_cache["sub_pmf"])
    elif rep_name in ("B5_scrambled",):
        # PMF under scrambled cell-graph (precomputed in Phase β logic, redo here)
        flat = sigs.flatten()
        p_m = float(np.mean(flat == -1)); p_z = float(np.mean(flat == 0))
        p_p = float(np.mean(flat == 1))
        total = p_m + p_z + p_p; p_m, p_z, p_p = p_m/total, p_z/total, p_p/total
        # +1 is center, so d=0 same; d=1 between (-1,+1) or (+1,0); d=2 between (-1,0)
        pmf = np.array([p_m**2 + p_z**2 + p_p**2,
                        2*(p_m*p_p + p_p*p_z),
                        2*(p_m*p_z)])
        m = estimate_id_L1(dist, pmf_cell=pmf)
    elif rep_name in ("B5m_mirror",):
        # -1 is center: d=0 same; d=1 between (-1,0) and (-1,+1); d=2 between (0,+1)
        flat = sigs.flatten()
        p_m = float(np.mean(flat == -1)); p_z = float(np.mean(flat == 0))
        p_p = float(np.mean(flat == 1))
        total = p_m + p_z + p_p; p_m, p_z, p_p = p_m/total, p_z/total, p_p/total
        pmf = np.array([p_m**2 + p_z**2 + p_p**2,
                        2*(p_m*p_z + p_m*p_p),
                        2*(p_z*p_p)])
        m = estimate_id_L1(dist, pmf_cell=pmf)
    else:
        # Hamming Macocco
        from m1_estimator_v2 import estimate_id_fixed_radii
        N = dist.shape[0]
        iu = np.triu_indices(N, k=1)
        flat = dist[iu]
        t1 = int(np.quantile(flat, 0.05))
        t2 = int(np.quantile(flat, 0.15))
        if t2 <= t1: t2 = t1 + 1
        d_hat = estimate_id_fixed_radii(dist, t1=t1, t2=t2)
        m = {"d_hat": float(d_hat), "t1": t1, "t2": t2}
    return {"macocco": m, "corrdim": cd}


# ============================================================================
# γ-A: multi-normalization comparison
# ============================================================================

def normalize_all(d_hat: float, rep: dict) -> dict:
    if not np.isfinite(d_hat):
        return {"abs": float("nan"), "over_Dmax": float("nan"),
                "over_Damb": float("nan")}
    return {"abs": float(d_hat),
            "over_Dmax": float(d_hat / rep["Dmax"]),
            "over_Damb": float(d_hat / rep["D_amb"])}


# ============================================================================
# γ-E: τ sweep
# ============================================================================

def tau_sweep_L1(K: np.ndarray, taus=(2000, 5000, 10000, 20000),
                  N_eff: int = 1500, rng=None):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    sample_idx = rng.choice(len(K), size=N_eff, replace=False)
    K_s = K[sample_idx]
    out = []
    for tau in taus:
        sub = threshold_extract(K_s, tau=tau)
        nonzero = float(np.mean(sub != 0))
        d = pairwise_L1_int8(sub)
        pmf = cell_pmf_from_data(sub)
        m = estimate_id_L1(d, pmf_cell=pmf)
        cd = correlation_dimension(d)
        out.append({
            "tau": tau, "nonzero": nonzero,
            "macocco_d_hat": float(m["d_hat"]),
            "corrdim_d_hat": float(cd["d_hat"]),
            "t1": m["t1"], "t2": m["t2"],
        })
    return out


# ============================================================================
# γ-D: close-regime under L1
# ============================================================================

def close_regime_L1(K: np.ndarray, meta: dict) -> dict:
    """Within-(layer, kv_head, site) M1 under L1 — substrate vs B2/B4."""
    N = len(meta["layer"])
    groups = {}
    for i in range(N):
        key = (int(meta["layer"][i]), int(meta["kv_head"][i]),
               str(meta["site"][i]))
        groups.setdefault(key, []).append(i)
    group_list = [np.array(v) for v in groups.values() if len(v) >= 6]

    # Aggregate within-group pairs for each representation
    all_within = {"substrate_L1": [], "B2_sign": [], "B4_pca": [],
                  "B5_scrambled": [], "B5m_mirror": []}
    for g in group_list:
        K_g = K[g]
        reps = build_representations(K_g)
        for rn in all_within:
            sigs = reps[rn]["sigs"]
            dist = reps[rn]["dist_fn"](sigs)
            iu = np.triu_indices(dist.shape[0], k=1)
            all_within[rn].extend(dist[iu].tolist())

    out = {}
    for rn, vals in all_within.items():
        flat = np.array(vals)
        if len(flat) < 100:
            out[rn] = {"d_hat": float("nan"), "n_pairs": len(flat)}
            continue
        t1 = int(np.quantile(flat, 0.05))
        t2 = int(np.quantile(flat, 0.15))
        if t2 <= t1: t2 = t1 + 1
        n_total = int(np.sum(flat <= t1))
        k_total = int(np.sum(flat <= t2))
        if k_total == 0 or n_total == k_total or n_total == 0:
            out[rn] = {"d_hat": float("nan"), "t1": t1, "t2": t2,
                       "n_total": n_total, "k_total": k_total,
                       "n_pairs": len(flat)}
            continue
        # Use L1 estimator for L1-type reps, Hamming Brent for Hamming reps
        # Both share the same Macocco target-equation form
        if rn in ("substrate_L1", "B5_scrambled", "B5m_mirror"):
            # Build PMF from one example group
            sample_sigs = threshold_extract(K[group_list[0]], tau=THRESHOLD_TAU)
            if rn == "substrate_L1":
                pmf = cell_pmf_from_data(sample_sigs)
            else:
                # scrambled/mirror PMFs are derived in m1_dual_estimators; replicate here
                flat_s = sample_sigs.flatten()
                p_m = float(np.mean(flat_s == -1)); p_z = float(np.mean(flat_s == 0))
                p_p = float(np.mean(flat_s == 1))
                tot = p_m + p_z + p_p; p_m, p_z, p_p = p_m/tot, p_z/tot, p_p/tot
                if rn == "B5_scrambled":
                    pmf = np.array([p_m**2+p_z**2+p_p**2,
                                    2*(p_m*p_p+p_p*p_z), 2*(p_m*p_z)])
                else:  # B5m_mirror
                    pmf = np.array([p_m**2+p_z**2+p_p**2,
                                    2*(p_m*p_z+p_m*p_p), 2*(p_z*p_p)])
            from m1_l1_estimator import cdf_at
            from scipy.optimize import brentq
            target = n_total / k_total
            def f(d): return cdf_at(pmf, d, t1) / max(cdf_at(pmf, d, t2), 1e-12) - target
            try:
                d_hat = float(brentq(f, 1.0, 500.0, xtol=1e-3))
            except Exception:
                d_hat = float("nan")
        else:
            from m1_l1_estimator import cdf_at
            # Build PMF for binary Hamming (each cell agrees w/ prob 0.5, disagrees 0.5)
            pmf = np.array([0.5, 0.5, 0.0])
            from scipy.optimize import brentq
            target = n_total / k_total
            def f(d): return cdf_at(pmf, d, t1) / max(cdf_at(pmf, d, t2), 1e-12) - target
            try:
                d_hat = float(brentq(f, 1.0, 500.0, xtol=1e-3))
            except Exception:
                d_hat = float("nan")
        out[rn] = {"d_hat": d_hat, "t1": t1, "t2": t2,
                   "n_total": n_total, "k_total": k_total,
                   "n_pairs": len(flat)}
    return out


# ============================================================================
# γ-G: correlated-synthetic calibration
# ============================================================================

def correlated_ternary(N: int, d_true: int, D_amb: int = 128,
                       n_factors: int = 5, seed: int = 42):
    """Factor-model synthetic: d_true cells driven by n_factors latent
    factors (correlations), the rest fixed. Reflects K-cache learned
    structure where cells co-vary.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(D_amb)
    real_cells = perm[:d_true]
    pad_cells = perm[d_true:]
    # latent factors per sample
    F = rng.standard_normal((N, n_factors))
    # loadings: each cell loads on factors
    L = rng.standard_normal((n_factors, d_true)) * 0.7
    # cell scores + noise
    scores = F @ L + 0.3 * rng.standard_normal((N, d_true))
    # threshold to ternary (mimics K → substrate signature)
    sig = np.zeros((N, D_amb), dtype=np.int8)
    tau_pct = np.quantile(np.abs(scores), 0.38)
    sig[:, real_cells] = np.where(scores > tau_pct, 1,
                          np.where(scores < -tau_pct, -1, 0)).astype(np.int8)
    pad_const = rng.integers(-1, 2, size=D_amb - d_true, dtype=np.int8)
    sig[:, pad_cells] = pad_const[np.newaxis, :]
    return sig


def calibrate_correlated(targets=(10, 20, 50, 100), N: int = 500):
    rows = []
    print(f"  {'d_true':>6} {'macocco':>8} {'rel_err':>8} {'corrdim':>8}  verdict")
    for d_true in targets:
        sigs = correlated_ternary(N=N, d_true=d_true)
        dist = pairwise_L1_int8(sigs)
        pmf = cell_pmf_from_data(sigs)
        m = estimate_id_L1(dist, pmf_cell=pmf)
        cd = correlation_dimension(dist)
        rel = abs(m["d_hat"] - d_true) / d_true
        verdict = "PASS" if rel < 0.20 else "FAIL"
        print(f"  {d_true:>6d} {m['d_hat']:>8.2f} {rel:>8.2%} {cd['d_hat']:>8.2f}  {verdict}")
        rows.append({"d_true": d_true, "macocco_d_hat": m["d_hat"],
                     "rel_err_macocco": rel, "corrdim_d_hat": cd["d_hat"]})
    return rows


# ============================================================================
# Pooled run with bootstrap
# ============================================================================

def pooled_with_bootstrap(K: np.ndarray, B: int = N_BOOTSTRAP,
                          N_sub: int = 500, rng=None,
                          include_shuffled: bool = False):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    if include_shuffled:
        K = shuffle_cells_per_row(K, seed=RNG_SEED + 1)
    reps = list(REP_CONFIG.keys())

    # Bootstrap
    boot = {r: {"macocco_abs": [], "corrdim_abs": []} for r in reps}
    rep_meta = build_representations(K[:1])  # capture D_amb / Dmax once
    for b in range(B):
        idx = rng.choice(len(K), size=N_sub, replace=True)
        K_sub = K[idx]
        reps_b = build_representations(K_sub)
        pmf_cache: dict = {}
        for rn in reps:
            sigs = reps_b[rn]["sigs"]
            dist = reps_b[rn]["dist_fn"](sigs)
            est = m1_dual_estimators(rn, dist, sigs, pmf_cache)
            boot[rn]["macocco_abs"].append(est["macocco"].get("d_hat", float("nan")))
            boot[rn]["corrdim_abs"].append(est["corrdim"].get("d_hat", float("nan")))
        if (b + 1) % 25 == 0:
            print(f"    iter {b+1}/{B}")
    # Summarize per rep with all three normalizations
    summary = {}
    for rn in reps:
        rep_data = rep_meta[rn]
        out = {"D_amb": rep_data["D_amb"], "Dmax": rep_data["Dmax"]}
        for est_name in ("macocco_abs", "corrdim_abs"):
            arr = np.array(boot[rn][est_name])
            arr_fin = arr[np.isfinite(arr)]
            if len(arr_fin) < 5:
                out[est_name] = {"mean": float("nan"), "ci_lo": float("nan"),
                                  "ci_hi": float("nan"), "n": int(len(arr_fin))}
            else:
                m_d = float(np.mean(arr_fin))
                lo = float(np.quantile(arr_fin, 0.025))
                hi = float(np.quantile(arr_fin, 0.975))
                out[est_name] = {"mean": m_d, "ci_lo": lo, "ci_hi": hi,
                                  "n": int(len(arr_fin)),
                                  "over_Dmax": m_d / rep_data["Dmax"],
                                  "over_Dmax_lo": lo / rep_data["Dmax"],
                                  "over_Dmax_hi": hi / rep_data["Dmax"],
                                  "over_Damb": m_d / rep_data["D_amb"],
                                  "over_Damb_lo": lo / rep_data["D_amb"],
                                  "over_Damb_hi": hi / rep_data["D_amb"]}
        summary[rn] = out
    return summary


# ============================================================================
# Robustness grid for P1, P2, P3
# ============================================================================

def robustness_grid(boot: dict) -> dict:
    """Apply P1, P2, P3 under every combination of:
       - normalization: abs, over_Dmax, over_Damb
       - estimator: macocco, corrdim

    P1: substrate_L1 < B0_Hamming_sub
    P2: substrate_L1 < B4_pca
    P3: substrate_L1 < B5_scrambled  AND  substrate_L1 < B5m_mirror
        (both scramblings should fail to match substrate)
    """
    out = {}
    pairs = {"P1": ("substrate_L1", "B0_Hamming_sub"),
             "P2": ("substrate_L1", "B4_pca"),
             "P3a": ("substrate_L1", "B5_scrambled"),
             "P3b": ("substrate_L1", "B5m_mirror")}
    for ptag, (rep_a, rep_b) in pairs.items():
        out[ptag] = {}
        for est in ("macocco_abs", "corrdim_abs"):
            for norm_field, lo, hi in [
                ("mean",         "ci_lo",       "ci_hi"),
                ("over_Dmax",    "over_Dmax_lo","over_Dmax_hi"),
                ("over_Damb",    "over_Damb_lo","over_Damb_hi"),
            ]:
                a = boot[rep_a][est]
                b = boot[rep_b][est]
                if norm_field not in a or norm_field not in b:
                    out[ptag][f"{est}_{norm_field}"] = {"pass": False,
                                                          "reason": "missing"}
                    continue
                gap = b[norm_field] - a[norm_field]
                disjoint = a[hi] < b[lo]
                out[ptag][f"{est}_{norm_field}"] = {
                    "a_mean": float(a[norm_field]),
                    "b_mean": float(b[norm_field]),
                    "gap": float(gap),
                    "disjoint": bool(disjoint),
                    "pass": bool(gap > 0 and disjoint),
                }
    return out


# ============================================================================
# Driver
# ============================================================================

def main(out_dir: str = "experiments/phase_gamma/results"):
    os.makedirs(out_dir, exist_ok=True)
    print("=== Phase γ — full red-team remediation ===\n")

    print("[γ-G] Calibration on correlated-synthetic ternary:")
    cal_corr = calibrate_correlated()
    print()

    print(f"Loading K-cache corpus from {DUMP_DIRS} ...")
    K, meta = collect_all(DUMP_DIRS)
    N = K.shape[0]
    print(f"  N = {N}\n")

    results: dict[str, Any] = {
        "config": {"N": int(N), "B": N_BOOTSTRAP, "dump_dirs": DUMP_DIRS},
        "calibration_correlated": cal_corr,
    }

    # ===== Pooled, REAL K =====
    print("[POOLED, real K, bootstrap]:")
    t0 = time.time()
    boot_real = pooled_with_bootstrap(K, include_shuffled=False)
    print(f"  bootstrap real: {time.time() - t0:.1f}s")
    results["pooled_real"] = boot_real

    # ===== γ-F: shuffled K control =====
    print("\n[POOLED, SHUFFLED K — γ-F null control]:")
    t0 = time.time()
    boot_shuf = pooled_with_bootstrap(K, include_shuffled=True)
    print(f"  bootstrap shuffled: {time.time() - t0:.1f}s")
    results["pooled_shuffled"] = boot_shuf

    # ===== Robustness grid (γ-A) =====
    print("\n[γ-A] Robustness grid (real K):")
    grid_real = robustness_grid(boot_real)
    for ptag, entries in grid_real.items():
        print(f"  {ptag}:")
        for k, v in entries.items():
            if v.get("pass") is None: continue
            tag = "PASS" if v["pass"] else "fail"
            print(f"    [{k:>30s}]  a={v.get('a_mean',float('nan')):8.3f}  "
                  f"b={v.get('b_mean',float('nan')):8.3f}  "
                  f"gap={v['gap']:+.3f}  disjoint={v['disjoint']}  → {tag}")
    results["grid_real"] = grid_real

    print("\n[γ-A] Robustness grid (shuffled K — null check):")
    grid_shuf = robustness_grid(boot_shuf)
    for ptag, entries in grid_shuf.items():
        sub = entries.get("macocco_abs_over_Damb", {})
        print(f"  {ptag} macocco_over_Damb: gap={sub.get('gap', 0):+.3f}  "
              f"pass={sub.get('pass')}")
    results["grid_shuffled"] = grid_shuf

    # ===== γ-E: τ sweep =====
    print("\n[γ-E] τ sweep on substrate L1:")
    sweep = tau_sweep_L1(K)
    print(f"  {'τ':>6} {'nz':>6} {'mac':>8} {'corrdim':>8} {'mac/Damb':>10}")
    for r in sweep:
        print(f"  {r['tau']:>6d} {r['nonzero']:>6.3f} "
              f"{r['macocco_d_hat']:>8.2f} {r['corrdim_d_hat']:>8.2f} "
              f"{r['macocco_d_hat']/HEAD_DIM:>10.3f}")
    results["tau_sweep"] = sweep

    # ===== γ-D: close-regime =====
    print("\n[γ-D] Close-regime (within layer/kv/site) under L1:")
    close = close_regime_L1(K, meta)
    for rn, r in close.items():
        d_hat = r.get("d_hat", float("nan"))
        D_amb = REP_CONFIG[rn]["D_amb"] if REP_CONFIG[rn] else HEAD_DIM
        # actually pull from rep_meta — REP_CONFIG just has Nones
        D_amb = boot_real[rn].get("D_amb", HEAD_DIM)
        print(f"  {rn:18s} d̂={d_hat:7.2f}  d̂/D_amb={d_hat/D_amb if np.isfinite(d_hat) else float('nan'):.3f}  "
              f"t1={r.get('t1','?')} t2={r.get('t2','?')} n_pairs={r.get('n_pairs','?')}")
    results["close_regime"] = close

    # ===== Final synthesis =====
    print("\n[FINAL SYNTHESIS]")
    # For each P-rule, count pass conditions across normalizations + estimators
    findings = {}
    for ptag in ("P1", "P2", "P3a", "P3b"):
        entries = grid_real[ptag]
        passes = sum(1 for v in entries.values() if v.get("pass"))
        total = len(entries)
        findings[ptag] = {"passes": passes, "total": total,
                          "ratio": f"{passes}/{total}",
                          "robust": passes == total,
                          "partial": 0 < passes < total,
                          "not_robust": passes == 0}
    for ptag, f in findings.items():
        label = "ROBUST" if f["robust"] else ("PARTIAL" if f["partial"] else "NOT ROBUST")
        print(f"  {ptag}: passes {f['ratio']} → {label}")
    results["findings"] = findings

    # Persist
    def to_serial(x):
        if isinstance(x, dict): return {k: to_serial(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [to_serial(v) for v in x]
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (bool, np.bool_)): return bool(x)
        return x

    out_path = os.path.join(out_dir, "phase_gamma_results.json")
    with open(out_path, "w") as f:
        json.dump(to_serial(results), f, indent=2)
    print(f"\nResults: {out_path}")
    return results


if __name__ == "__main__":
    main()
