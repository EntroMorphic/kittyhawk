"""Red-team checks for the Phase δ KV-eviction recall@k result.

Three concerns that flip the verdict if any of them holds:

  RT-A: Bootstrap CI is over 2250 trials but trials are not independent
        (7 prompts × ~50 (layer, kv, position) per prompt, intra-prompt
        correlation). Effective N is closer to 7 than 2250. Re-bootstrap
        by prompt-cluster (jackknife-leave-one-prompt-out and prompt-
        bootstrap).

  RT-C: The heterogeneity 300/300 paired finding could be the
        ESTIMATOR's systematic offset (L1-Macocco always reports lower
        d̂ than Hamming-Macocco-on-binary regardless of input). Test on
        UNIFORM-marginal random data: if substrate < B4 still holds
        with no real-data structure, the 300/300 finding is estimator-
        bias, not substrate-property.

  RT-E: L1 distance kernel cost vs Hamming kernel cost. If L1 is much
        slower, the +3pp recall lift might not pay for itself. Time
        both at production-relevant N, D.
"""
from __future__ import annotations

import os
import sys
import json
import time
import numpy as np

THIS = os.path.dirname(__file__)
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(THIS, "..", "phase_alpha"))
sys.path.insert(0, os.path.join(THIS, "..", "phase_beta"))

from load_k_signatures import (
    HEAD_DIM, THRESHOLD_TAU, b2_signature,
    threshold_extract, pairwise_hamming_int8,
)
from m1_l1_estimator import (
    pairwise_L1_int8, estimate_id_L1, cell_pmf_from_data, cdf_at,
)
from run_phase_alpha_v2 import b4_pca_sign
from m1_estimator_v2 import estimate_id_fixed_radii
from scipy.optimize import brentq


RNG_SEED = 20260512


# ============================================================================
# RT-A: prompt-clustered bootstrap of KV-eviction
# ============================================================================

def rt_A_prompt_clustered_bootstrap():
    """Re-bootstrap KV-eviction recall by RESAMPLING PROMPTS, not trials."""
    with open("experiments/phase_delta/results/kv_eviction_results.json") as f:
        data = json.load(f)
    trials = data["trials"]
    print("RT-A: prompt-clustered bootstrap (vs flat trial bootstrap)\n")

    # Group trials by prompt
    by_prompt = {}
    for t in trials:
        by_prompt.setdefault(t["prompt"], []).append(t)
    prompts = sorted(by_prompt.keys())
    print(f"  prompts: {prompts}  ({len(prompts)} clusters)\n")

    for kf in (0.25, 0.5, 0.75):
        print(f"  k_frac={kf}:")
        # Filter trials at this k_frac
        filtered = {p: [t for t in ts if abs(t["k_frac"] - kf) < 1e-6]
                    for p, ts in by_prompt.items()}
        # 1. Flat trial bootstrap (what δ-1 reported)
        all_ts = [t for p in prompts for t in filtered[p]]
        diffs = np.array([t["l1"] - t["hamming"] for t in all_ts])
        rng = np.random.default_rng(RNG_SEED)
        means_flat = [np.mean(rng.choice(diffs, size=len(diffs), replace=True))
                       for _ in range(1000)]
        flat_mean = np.mean(diffs)
        flat_lo, flat_hi = np.quantile(means_flat, [0.025, 0.975])
        # 2. Prompt-clustered bootstrap: resample PROMPTS with replacement
        rng2 = np.random.default_rng(RNG_SEED)
        means_clust = []
        for _ in range(1000):
            sampled_prompts = rng2.choice(prompts, size=len(prompts), replace=True)
            sub_diffs = []
            for sp in sampled_prompts:
                for t in filtered[sp]:
                    sub_diffs.append(t["l1"] - t["hamming"])
            if sub_diffs:
                means_clust.append(np.mean(sub_diffs))
        clust_lo, clust_hi = np.quantile(means_clust, [0.025, 0.975])
        # 3. Leave-one-prompt-out (jackknife)
        loo_means = []
        for p_out in prompts:
            kept_diffs = [t["l1"] - t["hamming"]
                          for p in prompts if p != p_out
                          for t in filtered[p]]
            loo_means.append(np.mean(kept_diffs))
        loo_range = (min(loo_means), max(loo_means))
        print(f"    flat bootstrap:     mean={flat_mean:+.4f}  CI=[{flat_lo:+.4f}, {flat_hi:+.4f}]")
        print(f"    prompt-cluster boot: CI=[{clust_lo:+.4f}, {clust_hi:+.4f}]")
        print(f"    leave-one-prompt:    range=[{loo_range[0]:+.4f}, {loo_range[1]:+.4f}]  "
              f"(n_loo={len(loo_means)})")
        # Verdict
        if clust_lo > 0:
            print(f"    → RT-A passes at k_frac={kf}: clustered CI fully above 0.")
        elif clust_hi < 0:
            print(f"    → RT-A FAILS at k_frac={kf}: clustered CI fully below 0.")
        else:
            print(f"    → RT-A AMBIGUOUS at k_frac={kf}: clustered CI crosses 0.")
        print()


# ============================================================================
# RT-C: estimator offset on uniform-marginal data
# ============================================================================

def rt_C_estimator_offset_uniform():
    """If L1-Macocco always reports lower d̂ than Hamming-Macocco regardless
    of input, the 300/300 heterogeneity finding is estimator-bias, not
    substrate-property. Test on UNIFORM random ternary (no structure).
    """
    print("RT-C: estimator offset on uniform random ternary (no structure)\n")
    rng = np.random.default_rng(RNG_SEED)

    print(f"  {'N':>5} {'d_amb':>6} {'sub_L1_d̂':>12} {'B0_Ham_d̂':>11} {'B4_PCA_d̂':>10}")
    rows = []
    for N in (200, 500, 1000):
        for d_amb in (64, 128, 256):
            # Uniform ternary "K-cache" — no manifold structure
            K_uniform = rng.integers(-1, 2, size=(N, HEAD_DIM), dtype=np.int8) * \
                        rng.integers(1000, 10000, size=(N, HEAD_DIM)).astype(np.int32)
            # Note: K is HEAD_DIM cells (the model dim), d_amb varies in B4 only here.
            # Skip d_amb dimension and just compute on HEAD_DIM=128.
            if d_amb != HEAD_DIM:
                continue
            sub_sig = threshold_extract(K_uniform, tau=THRESHOLD_TAU)
            sub_dist = pairwise_L1_int8(sub_sig)
            pmf_sub = cell_pmf_from_data(sub_sig)
            m_sub = estimate_id_L1(sub_dist, pmf_cell=pmf_sub)
            d_sub = m_sub["d_hat"]
            # B0 = Hamming on substrate
            ham_dist = pairwise_hamming_int8(sub_sig)
            iu = np.triu_indices(ham_dist.shape[0], k=1)
            flat = ham_dist[iu]
            t1 = int(np.quantile(flat, 0.05))
            t2 = int(np.quantile(flat, 0.15))
            if t2 <= t1: t2 = t1 + 1
            d_b0 = float(estimate_id_fixed_radii(ham_dist, t1=t1, t2=t2))
            # B4 = PCA + sign of K_uniform
            b4_sig = b4_pca_sign(K_uniform)
            b4_dist = pairwise_hamming_int8(b4_sig)
            iu = np.triu_indices(b4_dist.shape[0], k=1)
            flat = b4_dist[iu]
            t1 = int(np.quantile(flat, 0.05))
            t2 = int(np.quantile(flat, 0.15))
            if t2 <= t1: t2 = t1 + 1
            d_b4 = float(estimate_id_fixed_radii(b4_dist, t1=t1, t2=t2))
            print(f"  {N:>5d} {d_amb:>6d} {d_sub:>12.2f} {d_b0:>11.2f} {d_b4:>10.2f}")
            rows.append({"N": N, "d_amb": d_amb, "sub_L1": d_sub,
                          "B0_Ham": d_b0, "B4_PCA": d_b4})

    # Verdict
    print("\n  Interpretation:")
    print("  - On uniform random ternary (no structure), the TRUE intrinsic dim is")
    print(f"    {HEAD_DIM} (all cells independent). Each estimator should report ≈{HEAD_DIM}.")
    print("  - If sub_L1 < B0_Ham and sub_L1 < B4 even here, the gap is ESTIMATOR offset,")
    print("    not a substrate-distinctive structural property.")
    print()
    for r in rows:
        sub = r["sub_L1"]
        b4 = r["B4_PCA"]
        gap = b4 - sub
        if gap > 5:
            print(f"  → N={r['N']}: substrate < B4 by {gap:.1f} on UNIFORM data — "
                  f"this is estimator offset, not data structure.")
        else:
            print(f"  → N={r['N']}: gap = {gap:+.1f} small/zero on uniform — "
                  f"the substrate < B4 finding on real data IS data structure, not bias.")
    return rows


# ============================================================================
# RT-E: L1 vs Hamming kernel timing
# ============================================================================

def rt_E_kernel_cost():
    """Time L1 vs Hamming pairwise distance at production-relevant sizes."""
    print("RT-E: L1 vs Hamming kernel cost (Python/NumPy reference impl)\n")
    rng = np.random.default_rng(RNG_SEED)
    print(f"  {'N':>5} {'D':>4} {'Ham_ms':>10} {'L1_ms':>10} {'L1/Ham':>8}")
    rows = []
    for N in (200, 1000, 4096):
        for D in (HEAD_DIM,):
            sigs = rng.integers(-1, 2, size=(N, D), dtype=np.int8)
            # warm
            _ = pairwise_hamming_int8(sigs)
            _ = pairwise_L1_int8(sigs)
            # time Hamming
            n_iter = 3
            t0 = time.perf_counter()
            for _ in range(n_iter):
                _ = pairwise_hamming_int8(sigs)
            ham_ms = (time.perf_counter() - t0) / n_iter * 1000
            # time L1
            t0 = time.perf_counter()
            for _ in range(n_iter):
                _ = pairwise_L1_int8(sigs)
            l1_ms = (time.perf_counter() - t0) / n_iter * 1000
            ratio = l1_ms / ham_ms if ham_ms > 0 else float("inf")
            print(f"  {N:>5d} {D:>4d} {ham_ms:>10.2f} {l1_ms:>10.2f} {ratio:>7.2f}x")
            rows.append({"N": N, "D": D, "ham_ms": ham_ms, "l1_ms": l1_ms,
                          "ratio": ratio})
    return rows


def main():
    out = {}
    print("=" * 70)
    rt_A_prompt_clustered_bootstrap()
    print("=" * 70)
    out["rt_C"] = rt_C_estimator_offset_uniform()
    print("=" * 70)
    out["rt_E"] = rt_E_kernel_cost()
    os.makedirs("experiments/phase_delta/results", exist_ok=True)
    with open("experiments/phase_delta/results/redteam.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults: experiments/phase_delta/results/redteam.json")


if __name__ == "__main__":
    main()
