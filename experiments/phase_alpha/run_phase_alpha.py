"""Phase α full run: compute M1 (intrinsic dim), M2 (k-NN topology),
M3 (Betti-0 persistence) on substrate signatures + 3 baselines, across
close-prototype and far-prototype regimes, with bootstrap CIs.

Per FROZEN spec in journal/td27_geometric_prereg_v2_2026-05-12.md.

>>> SUPERSEDED for vision-claim tests by experiments/phase_beta/ <<<

This run uses categorical Hamming as the substrate distance, which
flattens the ternary alphabet's path-graph structure (the third state
loses its geometric role as the natural center). Under that metric,
the "VALIDATED 2/3" verdict here was downgraded to "MIXED 1/3" by
red-team and remediation (see journal/td27_phase_alpha_*), and finally
re-framed as testing-the-wrong-metric by the methodology pivot
(journal/td28_phase_alpha_methodology_pivot_2026-05-12.md).

The machinery (M2/M3 metrics, bootstrap, regime stratification) is
reused unchanged by Phase β/γ. The verdict label is not citable for
the substrate-distinctive vision claim.

Outputs:
  results/phase_alpha_results.json — all numeric results
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
)


N_BOOTSTRAP = 200       # 1000 is FROZEN ideal; 200 is what fits compute today
RNG_SEED = 20260512


# ============================================================================
# M1: intrinsic dimensionality
# ============================================================================

def m1_substrate(dist: np.ndarray, run_arch_b: bool = False) -> dict:
    """ARCH-A (Macocco fixed-radii) is the primary; ARCH-B for cross-check.

    ARCH-B is expensive (grid search over d × number of pairs); we run it
    only in the spot-check pass, not in bootstrap.
    """
    t1, t2 = auto_choose_radii(dist)
    d_A = estimate_id_fixed_radii(dist, t1=t1, t2=t2)
    d_B = estimate_id_twonn(dist) if (run_arch_b and dist.shape[0] <= 600) else float("nan")
    return {"d_hat_A": d_A, "d_hat_B": d_B, "t1": int(t1), "t2": int(t2)}


def m1_l2_twonn(dist: np.ndarray) -> dict:
    """Continuous TwoNN (Facco et al. 2017) for B1 (float32 K)."""
    N = dist.shape[0]
    mu = []
    for i in range(N):
        row = dist[i].copy()
        row[i] = np.inf
        sorted_d = np.sort(row)
        r1, r2 = sorted_d[0], sorted_d[1]
        if r1 > 0 and np.isfinite(r2):
            mu.append(r2 / r1)
    mu = np.array(mu)
    mu = mu[mu > 1.0]
    if mu.size < 5:
        return {"d_hat": float("nan")}
    # TwoNN: d = -log(1 - F(mu)) / log(mu), MLE via Hill estimator
    # Classic empirical estimate: d = N / sum(log(mu_i))
    d_hat = len(mu) / np.sum(np.log(mu))
    return {"d_hat": float(d_hat)}


# ============================================================================
# M2: k-NN topology divergence
# ============================================================================

def m2_topology(dist: np.ndarray, k_values=(5, 10, 20, 50)) -> dict:
    """Mutual-kNN reciprocity, clustering, degree-distribution Gini."""
    N = dist.shape[0]
    out = {"k_values": list(k_values)}
    # For each k: compute kNN graph
    np.fill_diagonal(dist, np.iinfo(dist.dtype).max if dist.dtype.kind == "i" else np.inf)
    # ranks
    order = np.argsort(dist, axis=1)
    knn = {k: order[:, :k] for k in k_values}

    reciprocities = []
    cluster_meds = []
    cluster_p95s = []
    ginis = []
    for k in k_values:
        nb = knn[k]
        # mutual reciprocity: frac of edges (i->j) where j->i also exists
        nb_set = [set(row) for row in nb]
        recip_count = 0
        total = 0
        for i in range(N):
            for j in nb[i]:
                total += 1
                if i in nb_set[int(j)]:
                    recip_count += 1
        reciprocities.append(recip_count / total if total else float("nan"))

        # local clustering coefficient (symmetrized graph)
        adj = np.zeros((N, N), dtype=bool)
        for i in range(N):
            adj[i, nb[i]] = True
        adj = adj | adj.T
        np.fill_diagonal(adj, False)
        clust = []
        for i in range(N):
            neighbors = np.where(adj[i])[0]
            kk = len(neighbors)
            if kk < 2:
                clust.append(0.0)
                continue
            sub = adj[np.ix_(neighbors, neighbors)]
            edges = sub.sum() // 2
            possible = kk * (kk - 1) // 2
            clust.append(edges / possible if possible else 0.0)
        clust = np.array(clust)
        cluster_meds.append(float(np.median(clust)))
        cluster_p95s.append(float(np.quantile(clust, 0.95)))

        # degree distribution Gini
        deg = adj.sum(axis=1).astype(np.float64)
        deg_sorted = np.sort(deg)
        n = deg_sorted.size
        if deg_sorted.sum() == 0:
            ginis.append(0.0)
        else:
            cum = np.cumsum(deg_sorted)
            gini = (n + 1 - 2 * cum.sum() / cum[-1]) / n
            ginis.append(float(gini))

    out["reciprocity"] = reciprocities
    out["clust_median"] = cluster_meds
    out["clust_p95"] = cluster_p95s
    out["degree_gini"] = ginis
    # restore diagonal (caller may reuse dist)
    np.fill_diagonal(dist, 0)
    return out


# ============================================================================
# M3: persistent Betti-0
# ============================================================================

class DSU:
    __slots__ = ("p", "r")
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True


def m3_betti0_persistence(dist: np.ndarray) -> dict:
    """Component count β_0(r) as r increases; persistence bars are merge
    events. Returns longest two bar lengths (in distance units).
    """
    N = dist.shape[0]
    # Get all edges sorted by distance
    iu = np.triu_indices(N, k=1)
    edge_d = dist[iu]
    order = np.argsort(edge_d, kind="stable")
    sorted_edges = list(zip(edge_d[order], iu[0][order], iu[1][order]))
    dsu = DSU(N)
    n_comp = N
    merge_distances = []
    last_merge_d = 0
    for d, i, j in sorted_edges:
        if dsu.union(int(i), int(j)):
            merge_distances.append(float(d))
            n_comp -= 1
            if n_comp == 1:
                break
    # bar lengths: differences between consecutive merge events; first bar
    # spans from 0 to first merge for the smaller component, etc. We
    # compute the simpler "merge-distance differences" (between consecutive
    # merges) as the bar lengths.
    if len(merge_distances) < 2:
        return {"longest_bar": float("nan"), "second_bar": float("nan"),
                "bar_p95": float("nan"), "n_merges": len(merge_distances)}
    diffs = np.diff([0.0] + merge_distances)
    diffs_sorted = np.sort(diffs)[::-1]
    longest = float(diffs_sorted[0])
    second = float(diffs_sorted[1]) if len(diffs_sorted) > 1 else float("nan")
    p95 = float(np.quantile(diffs, 0.95))
    return {"longest_bar": longest, "second_bar": second, "bar_p95": p95,
            "n_merges": len(merge_distances)}


# ============================================================================
# Driver
# ============================================================================

def build_signatures(K_raw: np.ndarray, seed_b2: int = 7, seed_b3: int = 13):
    """Construct substrate + B1 + B2 + B3 representations from raw K."""
    return {
        "substrate": threshold_extract(K_raw, tau=THRESHOLD_TAU),  # int8 ternary
        "B1_raw": K_raw.astype(np.float64),                        # continuous
        "B2_sign": b2_signature(K_raw, seed=seed_b2),              # int8 binary
        "B3_sign": b2_signature(K_raw, seed=seed_b3),              # int8 binary
    }


def compute_distances(sigs: dict) -> dict:
    return {
        "substrate": pairwise_hamming_int8(sigs["substrate"]),
        "B1_raw": pairwise_l2(sigs["B1_raw"]),
        "B2_sign": pairwise_hamming_int8(sigs["B2_sign"]),
        "B3_sign": pairwise_hamming_int8(sigs["B3_sign"]),
    }


def run_measures(dist_dict: dict, label: str, run_arch_b: bool = True) -> dict:
    print(f"  [{label}] N = {next(iter(dist_dict.values())).shape[0]}")
    results = {}
    for rep_name, dist in dist_dict.items():
        t0 = time.time()
        if rep_name == "B1_raw":
            m1 = m1_l2_twonn(dist)
        else:
            m1 = m1_substrate(dist, run_arch_b=run_arch_b)
        m2 = m2_topology(dist)
        m3 = m3_betti0_persistence(dist)
        results[rep_name] = {"M1": m1, "M2": m2, "M3": m3,
                             "elapsed_s": time.time() - t0}
        d_hat = (m1.get("d_hat_A") or m1.get("d_hat") or float("nan"))
        print(f"    {rep_name:12s} d̂={d_hat:6.2f}  "
              f"recip[k=10]={m2['reciprocity'][1]:.3f}  "
              f"longest_bar={m3['longest_bar']:.2f}  ({results[rep_name]['elapsed_s']:.1f}s)")
    return results


def bootstrap_m1(K_raw: np.ndarray, idx_pool: np.ndarray, B: int = N_BOOTSTRAP,
                 N_sub: int = 500, rng=None) -> dict:
    """Bootstrap M1 d̂ for each representation. Returns 95% CIs.

    Uses ARCH-A only (ARCH-B too expensive for B=200 iters; cross-validated
    in calibration).
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    d_hats = {"substrate": [], "B1_raw": [], "B2_sign": [], "B3_sign": []}
    for b in range(B):
        sub = rng.choice(idx_pool, size=min(N_sub, len(idx_pool)), replace=True)
        K_sub = K_raw[sub]
        sigs = build_signatures(K_sub)
        # substrate
        d_s = pairwise_hamming_int8(sigs["substrate"])
        m = m1_substrate(d_s, run_arch_b=False)
        d_hats["substrate"].append(m.get("d_hat_A", float("nan")))
        # B1
        d_l2 = pairwise_l2(sigs["B1_raw"])
        d_hats["B1_raw"].append(m1_l2_twonn(d_l2).get("d_hat", float("nan")))
        # B2 / B3
        for key in ("B2_sign", "B3_sign"):
            d_ham = pairwise_hamming_int8(sigs[key])
            d_hats[key].append(m1_substrate(d_ham, run_arch_b=False).get("d_hat_A", float("nan")))
        if (b + 1) % 25 == 0:
            print(f"    bootstrap iter {b + 1}/{B} done")
    return {
        rep: {
            "mean": float(np.nanmean(arr)),
            "ci_lo": float(np.nanquantile(arr, 0.025)),
            "ci_hi": float(np.nanquantile(arr, 0.975)),
            "n_samples": int(np.sum(np.isfinite(arr))),
        }
        for rep, arr in d_hats.items()
    }


def m1_within_group(K_raw: np.ndarray, groups: list, build_sigs_fn, dist_fn) -> dict:
    """M1 estimate using within-group pairs only (close-regime).

    Approach: for each group of K vectors, compute pairwise distance matrix.
    Use Macocco fixed-radii: aggregate n_total = sum over groups of (counts
    within t1), k_total = sum over groups of (counts within t2). Solve
    V_cat(t1, d) / V_cat(t2, d) = n_total / k_total for d.
    """
    from m1_estimator_v2 import log_cumvol
    from scipy.optimize import brentq

    # First pass: pool all within-group distances to pick (t1, t2)
    all_within = []
    for g in groups:
        if len(g) < 3:
            continue
        sub_K = K_raw[g]
        sub_sigs = build_sigs_fn(sub_K)
        sub_d = dist_fn(sub_sigs)
        iu = np.triu_indices(sub_d.shape[0], k=1)
        all_within.extend(sub_d[iu].tolist())
    if len(all_within) < 20:
        return {"d_hat_A": float("nan"), "n_pairs": len(all_within)}
    flat = np.array(all_within)
    t1 = int(np.quantile(flat, 0.05))
    t2 = int(np.quantile(flat, 0.15))
    if t2 <= t1:
        t2 = t1 + 1

    # Second pass: count
    n_total = int(np.sum(flat <= t1))
    k_total = int(np.sum(flat <= t2))
    if k_total == 0 or n_total == k_total:
        return {"d_hat_A": float("nan"), "t1": t1, "t2": t2,
                "n_total": n_total, "k_total": k_total}
    target = n_total / k_total

    def f(d):
        lv1 = log_cumvol(d, t1, 3)
        lv2 = log_cumvol(d, t2, 3)
        return math.exp(lv1 - lv2) - target

    try:
        d_hat = float(brentq(f, 1.0, 500.0, xtol=1e-4))
    except Exception:
        d_hat = float("nan")
    return {"d_hat_A": d_hat, "t1": t1, "t2": t2,
            "n_total": n_total, "k_total": k_total, "n_pairs": len(all_within)}


def define_regimes(meta: dict) -> dict:
    """Close-regime: SAME (layer, kv_head, site), pooled across positions
       and prompts.
       Far-regime: pooled across all (layer, kv_head, site, position)."""
    N = len(meta["layer"])
    layer = meta["layer"]
    kv = meta["kv_head"]
    site = np.array(meta["site"])
    # close: indices grouped by (layer, kv, site); we want INDEX POOLS where
    # variation is only across position/prompt
    groups = {}
    for i in range(N):
        key = (int(layer[i]), int(kv[i]), str(site[i]))
        groups.setdefault(key, []).append(i)
    close_idx = [np.array(v) for v in groups.values() if len(v) >= 6]
    far_idx = np.arange(N)
    return {"close_groups": close_idx, "far_pool": far_idx}


def main(dump_dir: str = "data/c_dump", out_dir: str = "experiments/phase_alpha/results"):
    os.makedirs(out_dir, exist_ok=True)
    print("Loading K-signatures …")
    K, meta = collect_all(dump_dir)
    N = K.shape[0]
    print(f"  N = {N}, D = {HEAD_DIM}")

    regimes = define_regimes(meta)
    print(f"  close groups: {len(regimes['close_groups'])} (each N≈8 — too small alone)")
    print(f"  far pool: {len(regimes['far_pool'])}")

    results: dict[str, Any] = {
        "config": {
            "N_total": int(N),
            "HEAD_DIM": HEAD_DIM,
            "B2_BITS": B2_BITS,
            "THRESHOLD_TAU": THRESHOLD_TAU,
            "N_bootstrap": N_BOOTSTRAP,
        },
        "regimes": {},
    }

    # ====================================================================
    # POOLED (all 2400) — primary verdict population
    # ====================================================================
    print("\n[POOLED — all K signatures]")
    sigs = build_signatures(K)
    dists = compute_distances(sigs)
    results["regimes"]["pooled"] = run_measures(dists, "pooled")

    # ====================================================================
    # FAR regime — random sample of 500 across the full population
    # ====================================================================
    print("\n[FAR regime: random 500 from full pool]")
    rng = np.random.default_rng(RNG_SEED)
    far_idx = rng.choice(regimes["far_pool"], size=500, replace=False)
    K_far = K[far_idx]
    sigs_far = build_signatures(K_far)
    dists_far = compute_distances(sigs_far)
    results["regimes"]["far_n500"] = run_measures(dists_far, "far_n500")

    # ====================================================================
    # CLOSE regime — WITHIN-group only (same layer/kv/site, different positions)
    # ====================================================================
    print("\n[CLOSE regime: within-group only (same layer/kv/site)]")
    close_groups_idx = regimes["close_groups"]
    print(f"  {len(close_groups_idx)} groups, each N≈{len(close_groups_idx[0])}")
    close_results = {}
    # substrate
    close_results["substrate"] = m1_within_group(
        K, close_groups_idx,
        build_sigs_fn=lambda Kx: threshold_extract(Kx, tau=THRESHOLD_TAU),
        dist_fn=pairwise_hamming_int8,
    )
    # B2_sign
    close_results["B2_sign"] = m1_within_group(
        K, close_groups_idx,
        build_sigs_fn=lambda Kx: b2_signature(Kx, seed=7),
        dist_fn=pairwise_hamming_int8,
    )
    # B3_sign
    close_results["B3_sign"] = m1_within_group(
        K, close_groups_idx,
        build_sigs_fn=lambda Kx: b2_signature(Kx, seed=13),
        dist_fn=pairwise_hamming_int8,
    )
    for rep, r in close_results.items():
        print(f"    {rep:12s} d̂_A={r.get('d_hat_A', float('nan')):6.2f}  "
              f"(t1={r.get('t1', '?')}, t2={r.get('t2', '?')}, "
              f"n_pairs={r.get('n_pairs', '?')})")
    results["regimes"]["close_within_group"] = close_results

    # ====================================================================
    # Layer stratification (FROZEN: layers 0, 14, 29)
    # ====================================================================
    layer_arr = meta["layer"]
    for L in (0, 14, 29):
        idx = np.where(layer_arr == L)[0]
        if len(idx) < 50:
            continue
        print(f"\n[Layer {L} — N = {len(idx)}]")
        K_L = K[idx]
        sigs_L = build_signatures(K_L)
        dists_L = compute_distances(sigs_L)
        results["regimes"][f"layer{L}"] = run_measures(dists_L, f"layer{L}")

    # ====================================================================
    # Bootstrap CIs for M1 (the primary verdict measure)
    # ====================================================================
    print(f"\n[Bootstrap CIs on M1, B={N_BOOTSTRAP}]")
    t0 = time.time()
    boot = bootstrap_m1(K, regimes["far_pool"], B=N_BOOTSTRAP, N_sub=500)
    print(f"  bootstrap elapsed: {time.time() - t0:.1f}s")
    for rep, stats in boot.items():
        print(f"  {rep:12s} d̂ = {stats['mean']:6.2f}  "
              f"95% CI [{stats['ci_lo']:6.2f}, {stats['ci_hi']:6.2f}]  "
              f"(n_finite={stats['n_samples']})")
    results["bootstrap_m1"] = boot

    # ====================================================================
    # Verdict rules (FROZEN)
    # ====================================================================
    print("\n[Verdict (per FROZEN spec)]")
    pooled = results["regimes"]["pooled"]
    d_sub = pooled["substrate"]["M1"]["d_hat_A"]
    d_b2 = pooled["B2_sign"]["M1"]["d_hat_A"]
    d_b3 = pooled["B3_sign"]["M1"]["d_hat_A"]
    d_b1 = pooled["B1_raw"]["M1"]["d_hat"]
    print(f"  M1 pooled:   substrate {d_sub:.2f} vs B2 {d_b2:.2f} vs B3 {d_b3:.2f} vs B1 {d_b1:.2f}")
    # M1 verdict: substrate < B2 by ≥ 20% relative AND bootstrap CI disjoint
    sub_boot = boot["substrate"]
    b2_boot = boot["B2_sign"]
    m1_rel = (b2_boot["mean"] - sub_boot["mean"]) / b2_boot["mean"] if b2_boot["mean"] else 0
    m1_disjoint = sub_boot["ci_hi"] < b2_boot["ci_lo"]
    m1_pass = m1_rel >= 0.20 and m1_disjoint
    print(f"  M1: substrate vs B2 relative gap = {m1_rel:.1%}, "
          f"CIs disjoint = {m1_disjoint} → {'PASS' if m1_pass else 'FAIL'}")

    # M2 verdict
    s_recip = pooled["substrate"]["M2"]["reciprocity"]
    b2_recip = pooled["B2_sign"]["M2"]["reciprocity"]
    s_gini = pooled["substrate"]["M2"]["degree_gini"]
    b2_gini = pooled["B2_sign"]["M2"]["degree_gini"]
    m2_per_k = []
    for k_idx in range(4):
        gap_recip = s_recip[k_idx] - b2_recip[k_idx]
        gap_gini = b2_gini[k_idx] - s_gini[k_idx]  # substrate Gini lower = less hub-y
        m2_per_k.append(gap_recip >= 0.05 and gap_gini >= 0.05)
    m2_pass = sum(m2_per_k) >= 3
    print(f"  M2: pass k-counts {m2_per_k} → {'PASS' if m2_pass else 'FAIL'}")

    # M3 verdict
    sub_long = pooled["substrate"]["M3"]["longest_bar"]
    b3_p95 = pooled["B3_sign"]["M3"]["bar_p95"]
    m3_pass = math.isfinite(sub_long) and math.isfinite(b3_p95) and sub_long > 2 * b3_p95
    print(f"  M3: substrate longest_bar {sub_long:.2f} vs 2×B3.p95 {2*b3_p95:.2f} → "
          f"{'PASS' if m3_pass else 'FAIL'}")

    n_pass = int(m1_pass) + int(m2_pass) + int(m3_pass)
    if n_pass >= 2:
        verdict = "VALIDATED"
    elif n_pass == 1:
        verdict = "MIXED"
    else:
        verdict = "FALSIFIED"
    print(f"\n  >>> SUBSTRATE GEOMETRIC CLAIM: {verdict}  "
          f"({n_pass}/3 measures clear) <<<")
    results["verdict"] = {
        "m1_pass": m1_pass, "m2_pass": m2_pass, "m3_pass": m3_pass,
        "n_pass": n_pass, "label": verdict,
        "m1_rel_gap": m1_rel, "m1_ci_disjoint": m1_disjoint,
        "m2_pass_per_k": m2_per_k,
    }

    # ====================================================================
    # Persist
    # ====================================================================
    def to_serializable(x):
        if isinstance(x, dict):
            return {k: to_serializable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [to_serializable(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        return x

    out_path = os.path.join(out_dir, "phase_alpha_results.json")
    with open(out_path, "w") as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"\nResults written to {out_path}")
    return results


if __name__ == "__main__":
    main()
