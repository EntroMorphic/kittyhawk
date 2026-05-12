"""Phase δ-3 — verify the calibration bias is symmetric across reps.

γ-G found the Macocco estimator is biased ~45% low on correlated
synthetic. If that bias is the SAME for substrate-L1, Hamming-B0,
B4-PCA on correlated data, then RELATIVE comparisons (P-rules) survive
the bias — both numerator and denominator shrink by the same factor.

If the bias is asymmetric — e.g., substrate is biased 30% but B4 is
biased 60% — then relative comparisons can flip under correlated data.

This script generates factor-model correlated synthetic K-cache, builds
substrate / Hamming / B4 reps from it, and reports the per-rep bias.
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

from load_k_signatures import HEAD_DIM, B2_BITS, THRESHOLD_TAU, \
    threshold_extract, b2_signature, pairwise_hamming_int8
from m1_l1_estimator import (
    pairwise_L1_int8, estimate_id_L1,
    cell_pmf_from_data,
)
from run_phase_alpha_v2 import b4_pca_sign
from m1_estimator_v2 import estimate_id_fixed_radii


def factor_model_K_cache(N: int, d_true: int, n_factors: int = 5,
                          ambient_K_dim: int = HEAD_DIM,
                          seed: int = 42, scale: int = 10000):
    """Factor-model synthetic K-cache: latent factors → cell loadings →
    int32-scaled values mimicking K-cache mantissa range.
    """
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((N, n_factors))
    L = rng.standard_normal((n_factors, d_true)) * 0.7
    scores = F @ L + 0.3 * rng.standard_normal((N, d_true))
    # Embed in ambient K_dim, fixed padding
    perm = rng.permutation(ambient_K_dim)
    real = perm[:d_true]; pad = perm[d_true:]
    K = np.zeros((N, ambient_K_dim), dtype=np.float64)
    K[:, real] = scores
    pad_const = rng.standard_normal(ambient_K_dim - d_true) * 0.01
    K[:, pad] = pad_const[None, :]
    # Scale to int32 range like real K-cache (median |K| ≈ 5000)
    K_scaled = (K * scale).astype(np.int32)
    return K_scaled


def measure_d_hat_macocco_hamming(dist: np.ndarray) -> float:
    N = dist.shape[0]
    iu = np.triu_indices(N, k=1)
    flat = dist[iu]
    t1 = int(np.quantile(flat, 0.05))
    t2 = int(np.quantile(flat, 0.15))
    if t2 <= t1: t2 = t1 + 1
    return float(estimate_id_fixed_radii(dist, t1=t1, t2=t2))


def main():
    print("=== δ-3: calibration bias symmetry check ===\n")
    print("Generating factor-model correlated K-cache at known d_true ∈ {10, 20, 50, 100}\n")
    targets = [10, 20, 50, 100]
    rows = []
    print(f"{'d_true':>6} {'sub_L1':>10} {'B0_Ham':>10} {'B2':>10} {'B4_pca':>10}  | "
          f"{'sub_bias':>10} {'B0_bias':>10} {'B4_bias':>10}")
    for d_true in targets:
        K = factor_model_K_cache(N=500, d_true=d_true)
        # substrate (L1)
        sub_sig = threshold_extract(K, tau=THRESHOLD_TAU)
        sub_dist = pairwise_L1_int8(sub_sig)
        pmf_sub = cell_pmf_from_data(sub_sig)
        m_sub = estimate_id_L1(sub_dist, pmf_cell=pmf_sub)
        d_sub = m_sub["d_hat"]
        # B0 (Hamming on substrate)
        ham_dist = pairwise_hamming_int8(sub_sig)
        d_b0 = measure_d_hat_macocco_hamming(ham_dist)
        # B2 random sign
        b2_sig = b2_signature(K, seed=7)
        b2_dist = pairwise_hamming_int8(b2_sig)
        d_b2 = measure_d_hat_macocco_hamming(b2_dist)
        # B4 PCA+sign
        b4_sig = b4_pca_sign(K)
        b4_dist = pairwise_hamming_int8(b4_sig)
        d_b4 = measure_d_hat_macocco_hamming(b4_dist)

        # "Bias" = (d_hat - d_true) / d_true (signed, negative = underestimate)
        # Use normalized d̂/Damb to fairly compare across ambient dimensions
        # since different reps have different ambient caps.
        # But the right "bias" is how much each rep underestimates its OWN
        # intrinsic dim relative to d_true. If the data has d_true=50
        # independent factors, every rep on this data should report d̂ ≈ 50
        # (modulo their estimator's bias).
        sub_bias = (d_sub - d_true) / d_true
        b0_bias  = (d_b0  - d_true) / d_true
        b4_bias  = (d_b4  - d_true) / d_true
        print(f"{d_true:>6d} {d_sub:>10.2f} {d_b0:>10.2f} {d_b2:>10.2f} {d_b4:>10.2f}  | "
              f"{sub_bias:>+10.2%} {b0_bias:>+10.2%} {b4_bias:>+10.2%}")
        rows.append({"d_true": d_true, "sub_L1": d_sub, "B0_Hamming": d_b0,
                     "B2": d_b2, "B4_pca": d_b4,
                     "sub_bias": sub_bias, "B0_bias": b0_bias,
                     "B4_bias": b4_bias})

    print("\n=== Interpretation ===")
    # If all reps have similar bias %, relative comparisons survive.
    # If asymmetric, P-rules could flip.
    avg_sub = np.mean([r["sub_bias"] for r in rows])
    avg_b0  = np.mean([r["B0_bias"]  for r in rows])
    avg_b4  = np.mean([r["B4_bias"]  for r in rows])
    print(f"Mean bias: substrate_L1={avg_sub:+.1%}  B0_Hamming={avg_b0:+.1%}  B4_pca={avg_b4:+.1%}")
    spread = max(abs(avg_sub - avg_b0), abs(avg_sub - avg_b4), abs(avg_b0 - avg_b4))
    print(f"Max pairwise bias spread: {spread:.1%}")
    if spread < 0.10:
        print("→ Biases are SYMMETRIC across reps (within 10pp). Relative P-rule directions survive.")
    else:
        print("→ Biases are ASYMMETRIC. Relative P-rule directions could flip under correlated data.")

    out = {"rows": rows, "mean_bias": {"sub_L1": avg_sub, "B0": avg_b0, "B4": avg_b4},
           "max_bias_spread": float(spread),
           "symmetric": bool(spread < 0.10)}
    os.makedirs("experiments/phase_delta/results", exist_ok=True)
    with open("experiments/phase_delta/results/calibration_symmetry.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults: experiments/phase_delta/results/calibration_symmetry.json")


if __name__ == "__main__":
    main()
