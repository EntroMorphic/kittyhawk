"""M1 estimator under L1 distance on the ternary cell-graph.

Phase β substrate distance: d_L1(a, b) = Σᵢ |aᵢ − bᵢ|.
Each cell contributes 0 (agree), 1 (one-step apart, e.g., 0↔±1), or
2 (across the center, +1↔−1).

Shell volume under L1: given per-cell PMF q over contributions {0,1,2}
of two i.i.d. cells with marginals p, V_L1(t, d) = number of d-cell
configurations with total cost ≤ t. Computed via dynamic programming.

Macocco fixed-radii estimator (ARCH-A from Phase α) carries over
unchanged given V_L1; only the shell-volume function differs from
the categorical-Hamming case.

>>> CALIBRATION CAVEAT (γ-G, journal/td28_phase_gamma_robustness):
    Validated within 4% on synthetic with INDEPENDENT cells (uniform
    or K-cache-marginal). FAILS by ~45% on factor-model synthetic
    with non-trivial cell CORRELATIONS. Real K-cache has correlations.
    Absolute d̂ values for K-cache are therefore biased LOW (true
    intrinsic dim is roughly 2× reported). Relative comparisons
    (P-rules) remain valid because all representations are biased
    similarly. Don't cite absolute d̂ as a literal intrinsic
    dimension on real data.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.optimize import brentq


# ============================================================================
# Single-cell PMF of |a - b| for two i.i.d. ternary cells
# ============================================================================

def cell_pmf(p_minus: float, p_zero: float, p_plus: float) -> np.ndarray:
    """Probability mass over contribution k ∈ {0, 1, 2} of a single cell.

    Given marginals p_minus, p_zero, p_plus over {-1, 0, +1} (must sum to 1),
    compute P(|a - b| = k) where a, b are i.i.d.

    |a−b| = 0 ↔ a = b: prob = p_-² + p_0² + p_+²
    |a−b| = 1 ↔ {a,b} = {0,−1} or {0,+1}: prob = 2(p_- · p_0 + p_0 · p_+)
    |a−b| = 2 ↔ {a,b} = {−1, +1}: prob = 2 · p_- · p_+
    """
    assert abs(p_minus + p_zero + p_plus - 1.0) < 1e-9
    p0 = p_minus ** 2 + p_zero ** 2 + p_plus ** 2
    p1 = 2 * (p_minus * p_zero + p_zero * p_plus)
    p2 = 2 * p_minus * p_plus
    return np.array([p0, p1, p2])


def uniform_cell_pmf() -> np.ndarray:
    """Uniform ternary: P(0) = 1/3, P(1) = 4/9, P(2) = 2/9."""
    return cell_pmf(1 / 3, 1 / 3, 1 / 3)


def substrate_cell_pmf(p_nonzero: float = 0.62, p_sign_balance: float = 0.5):
    """K-cache-like marginals: ~62% nonzero, sign balanced.

    p_sign_balance = 0.5 means equal +/- among the nonzero half.
    """
    p_zero = 1 - p_nonzero
    p_plus = p_nonzero * p_sign_balance
    p_minus = p_nonzero * (1 - p_sign_balance)
    return cell_pmf(p_minus, p_zero, p_plus)


def cell_pmf_from_data(signatures: np.ndarray) -> np.ndarray:
    """Empirical cell PMF over contributions {0,1,2}.

    Estimates marginals from the data then derives the PMF.
    """
    flat = signatures.flatten()
    p_minus = float(np.mean(flat == -1))
    p_zero = float(np.mean(flat == 0))
    p_plus = float(np.mean(flat == 1))
    total = p_minus + p_zero + p_plus
    return cell_pmf(p_minus / total, p_zero / total, p_plus / total)


# ============================================================================
# L1 shell volumes via convolution (DP)
# ============================================================================

def l1_cumulative_pmf(pmf_cell: np.ndarray, d: int) -> np.ndarray:
    """Cumulative PMF over d-cell L1 distance ∈ {0, 1, ..., 2d}.

    F_d(t) = P(L1 ≤ t) for two i.i.d. d-cell signatures.
    Returns array of length 2d + 1: result[t] = P(L1 ≤ t).
    """
    # PMF over [0, 2d]
    pmf = np.zeros(1)
    pmf[0] = 1.0
    for _ in range(d):
        pmf = np.convolve(pmf, pmf_cell)
    return np.cumsum(pmf)


def l1_cumulative_pmf_continuous(pmf_cell: np.ndarray, d_real: float,
                                  t_max: int) -> np.ndarray:
    """Cumulative L1 distance PMF for non-integer d via interpolation.

    F(t | d) is needed for Brent's-method root-finding on d. We
    interpolate between the floor and ceiling integer d:
      F(t | d) ≈ (1 - α) F(t | ⌊d⌋) + α F(t | ⌈d⌉), α = d - ⌊d⌋.

    Returns full CDF array of length t_max + 1.
    """
    d_lo = int(math.floor(d_real))
    d_hi = int(math.ceil(d_real))
    if d_lo == d_hi:
        cdf = l1_cumulative_pmf(pmf_cell, d_lo)
        return cdf[:t_max + 1] if len(cdf) > t_max else \
               np.concatenate([cdf, np.ones(t_max + 1 - len(cdf))])
    alpha = d_real - d_lo
    cdf_lo = l1_cumulative_pmf(pmf_cell, d_lo)
    cdf_hi = l1_cumulative_pmf(pmf_cell, d_hi)
    # Pad both to t_max + 1
    def pad(c, n):
        return np.concatenate([c, np.ones(max(0, n - len(c)))])[:n]
    cdf_lo = pad(cdf_lo, t_max + 1)
    cdf_hi = pad(cdf_hi, t_max + 1)
    return (1 - alpha) * cdf_lo + alpha * cdf_hi


def cdf_at(pmf_cell: np.ndarray, d_real: float, t: int) -> float:
    """P(L1 ≤ t | d cells, cell PMF). Single scalar; cheaper than full CDF."""
    if t < 0:
        return 0.0
    cdf = l1_cumulative_pmf_continuous(pmf_cell, d_real, t)
    if t >= len(cdf):
        return 1.0
    return float(cdf[t])


# ============================================================================
# Macocco fixed-radii estimator under L1
# ============================================================================

def pairwise_L1_int8(signatures: np.ndarray) -> np.ndarray:
    """Pairwise L1 distance over int8 signatures (any alphabet).

    For ternary {-1, 0, +1}: d ∈ [0, 2D]. For binary {-1, +1}: d ∈ [0, 2D].
    Vectorized via N×D matrix — memory hot but BLAS-fast.
    """
    s = signatures.astype(np.float32)
    # |a - b|.sum() = Σ_d (a_i_d² + b_j_d² - 2 a_i_d b_j_d)^(1/2) per pair —
    # but that's for L2 with abs values. For L1 over int8 in {-1,0,+1}, we use
    # the identity |a-b| = (a-b)² for values in {-1, 0, +1} only when both are
    # in {-1, +1} (so |a-b| ∈ {0, 2}). For ternary, |a-b| isn't simply (a-b)².
    # Example: |0 - 1| = 1, but (0-1)² = 1. |(-1) - (+1)| = 2, (a-b)² = 4.
    # So (a-b)² gives |a-b| · |a-b|, i.e., the square of L1.
    # The cleanest BLAS form: build one-hot at 3 levels and matmul:
    #   Each cell at value v has indicator at level (v+1) ∈ {0, 1, 2}.
    #   |a_d − b_d| = |v_a − v_b| where v ∈ {-1, 0, +1}.
    # This isn't a simple matmul. Resort to broadcast:
    # (N, 1, D) - (1, N, D) → (N, N, D), |.|.sum
    N, D = signatures.shape
    out = np.zeros((N, N), dtype=np.int32)
    # Loop over chunks to avoid O(N²D) memory peak
    CHUNK = max(1, min(N, 1024))
    for i in range(0, N, CHUNK):
        block = signatures[i:i + CHUNK, None, :].astype(np.int16) - \
                signatures[None, :, :].astype(np.int16)
        out[i:i + CHUNK] = np.abs(block).sum(axis=-1).astype(np.int32)
    return out


def estimate_id_L1(dist: np.ndarray, pmf_cell: np.ndarray,
                    quantile_pair: Tuple[float, float] = (0.05, 0.15),
                    d_min: float = 1.0, d_max: float = 500.0) -> dict:
    """Macocco fixed-radii estimator with L1 shell volumes.

    Args:
      dist: (N, N) L1 distance matrix.
      pmf_cell: 3-element array of P(cell contrib ∈ {0,1,2}).
      quantile_pair: (q1, q2) to choose (t1, t2) radii.

    Returns: {d_hat, t1, t2, n_total, k_total}.
    """
    N = dist.shape[0]
    iu = np.triu_indices(N, k=1)
    flat = dist[iu]
    t1 = int(np.quantile(flat, quantile_pair[0]))
    t2 = int(np.quantile(flat, quantile_pair[1]))
    if t2 <= t1:
        t2 = t1 + 1

    # Per-point counts excluding self
    n_counts = np.zeros(N, dtype=np.int64)
    k_counts = np.zeros(N, dtype=np.int64)
    for i in range(N):
        row = dist[i].copy()
        row[i] = np.iinfo(row.dtype).max
        n_counts[i] = int(np.sum(row <= t1))
        k_counts[i] = int(np.sum(row <= t2))
    n_total = int(n_counts.sum())
    k_total = int(k_counts.sum())
    if k_total == 0 or n_total == k_total or n_total == 0:
        return {"d_hat": float("nan"), "t1": t1, "t2": t2,
                "n_total": n_total, "k_total": k_total}
    target = n_total / k_total

    def f(d):
        f1 = cdf_at(pmf_cell, d, t1)
        f2 = cdf_at(pmf_cell, d, t2)
        if f2 <= 0:
            return 1.0
        return (f1 / f2) - target

    f_lo = f(d_min)
    f_hi = f(d_max)
    if f_lo * f_hi > 0:
        d_hat = d_min if abs(f_lo) < abs(f_hi) else d_max
    else:
        d_hat = float(brentq(f, d_min, d_max, xtol=1e-3))
    return {"d_hat": d_hat, "t1": t1, "t2": t2,
            "n_total": n_total, "k_total": k_total}


# ============================================================================
# Sanity / self-test
# ============================================================================

if __name__ == "__main__":
    pmf_u = uniform_cell_pmf()
    print(f"uniform-ternary cell PMF: {pmf_u}")
    pmf_s = substrate_cell_pmf(0.62)
    print(f"substrate (p_nz=0.62)   : {pmf_s}")

    # Sanity: CDF at d=10 cells
    for t in [5, 10, 15, 20]:
        c_u = cdf_at(pmf_u, 10, t)
        c_s = cdf_at(pmf_s, 10, t)
        print(f"d=10 t={t:2d}: F_uniform={c_u:.4f} F_substrate={c_s:.4f}")
