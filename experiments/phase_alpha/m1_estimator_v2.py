"""M1 v2: corrected local intrinsic dimensionality under ternary
categorical Hamming.

v1 (m1_estimator.py) failed calibration with systematic ~38%
underestimation at d >= 20. Two diagnostic agents identified the root
cause and produced TWO viable corrected derivations. v2 implements
both for cross-validation:

  ARCH-A (Macocco fixed-radii):
    Macocco-Glielmo-Grilli-Laio (PRL 130, 067401, 2023) does NOT use
    TwoNN order statistics. It uses two fixed radii (t1, t2). For each
    point i, count n_i = neighbors with distance <= t1 and k_i =
    neighbors with distance <= t2. Under uniform local density on a
    d-dim manifold with alphabet A:

      P(distance <= t | d, A) = V_cat(t, d, A) / V_cat(d, d, A) = V_cat(t)/A^d

    where V_cat(t, d, A) = sum_{r=0}^{t} C(d, r) * (A-1)^r.

    Conditional: n_i | k_i ~ Binomial(k_i, p(d)) with
      p(d) = V_cat(t1, d, 3) / V_cat(t2, d, 3).

    MLE: maximize sum_i [n_i log p(d) + (k_i - n_i) log(1 - p(d))]
    via 1-D root-finding on p^(-1)(<n>/<k>).

  ARCH-B (corrected TwoNN order statistics):
    The v1 attempt was reaching for the TwoNN architecture but missed
    the survival factor S(R)^(M-2) and tie handling. The corrected
    joint PMF of the two nearest distances (r1, r2) drawn from M-1
    candidate distances:

      r1 < r2:  (M-1)(M-2) * p(r1) p(r2) S(r2)^(M-3)
      r1 = r2:  C(M-1,2)   * p(r1)^2 S(r1)^(M-3)

    where p(r|d) = C(d, r) (A-1)^r / A^d (single-distance PMF on a
    d-dim manifold) and S(R|d) = sum_{r > R} p(r|d).

    MLE: grid search d ∈ [1, D_ambient + buffer].

Both architectures should agree on calibration data (synthetic ternary
of known d). If they disagree, that is itself useful diagnostic info.
"""
import math
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp


# ============================================================================
# Shared kernel: shell volume V_cat(t, d, A) for ternary alphabet A=3.
# Computed in log space for numerical stability at large d.
# ============================================================================

def log_C(d: float, r: int) -> float:
    """log C(d, r) with continuous d via gammaln. Returns -inf if r > d."""
    if r < 0 or r > d:
        return -math.inf
    return gammaln(d + 1) - gammaln(r + 1) - gammaln(d - r + 1)


def log_shell(d: float, r: int, A: int = 3) -> float:
    """log V_shell(r, d, A) = log[C(d, r) * (A-1)^r]."""
    if r < 0 or r > d:
        return -math.inf
    return log_C(d, r) + r * math.log(A - 1)


def log_cumvol(d: float, t: int, A: int = 3) -> float:
    """log V_cat(t, d, A) = log sum_{r=0}^{t} C(d, r) (A-1)^r."""
    if t < 0:
        return -math.inf
    if t >= d:
        # closed form: sum_{r=0}^{d} C(d,r) (A-1)^r = A^d
        return d * math.log(A)
    terms = []
    for r in range(t + 1):
        v = log_shell(d, r, A)
        if v > -math.inf:
            terms.append(v)
    if not terms:
        return -math.inf
    return logsumexp(terms)


def log_p_distance(d: float, r: int, A: int = 3) -> float:
    """log P(D = r | d) under uniform local density on d-dim manifold."""
    # P(D=r) = C(d,r) (A-1)^r / A^d
    if r < 0 or r > d:
        return -math.inf
    return log_shell(d, r, A) - d * math.log(A)


def log_S(d: float, R: int, A: int = 3) -> float:
    """log P(D > R | d). Computed via complement of cumulative for stability."""
    if R < 0:
        return 0.0  # all distances > R
    if R >= d:
        return -math.inf  # no distance can exceed d on a d-dim manifold
    # log(1 - exp(log_cdf)) is unstable when log_cdf ≈ 0; compute by tail sum.
    terms = []
    # tail r ∈ {R+1, ..., floor(d)}
    rmax = int(math.floor(d))
    for r in range(R + 1, rmax + 1):
        v = log_p_distance(d, r, A)
        if v > -math.inf:
            terms.append(v)
    if not terms:
        return -math.inf
    return logsumexp(terms)


# ============================================================================
# Distance matrix
# ============================================================================

def pairwise_categorical_hamming(signatures: np.ndarray) -> np.ndarray:
    """Pairwise categorical Hamming distance over ternary signatures."""
    N = signatures.shape[0]
    dist = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        dist[i] = np.sum(signatures != signatures[i:i + 1], axis=1)
    return dist


# ============================================================================
# ARCH-A: Macocco fixed-radii estimator
# ============================================================================

def estimate_id_fixed_radii(
    distances: np.ndarray,
    t1: int,
    t2: int,
    A: int = 3,
    d_min: float = 1.0,
    d_max: float = 500.0,
) -> float:
    """Estimate d via Macocco fixed-radii. Picks (t1, t2) externally.

    For each point i, n_i = #{j != i : d(i,j) <= t1},
                      k_i = #{j != i : d(i,j) <= t2}.

    Under uniform local density:
      p(d) = V_cat(t1, d, A) / V_cat(t2, d, A) = E[n_i] / E[k_i].

    MLE solves p(d) = <n> / <k> via Brent's method.
    """
    N = distances.shape[0]
    n_counts = np.zeros(N, dtype=np.int64)
    k_counts = np.zeros(N, dtype=np.int64)
    for i in range(N):
        row = distances[i].copy()
        row[i] = np.iinfo(row.dtype).max  # exclude self
        n_counts[i] = int(np.sum(row <= t1))
        k_counts[i] = int(np.sum(row <= t2))

    total_n = int(n_counts.sum())
    total_k = int(k_counts.sum())
    if total_k == 0:
        return float("nan")
    if total_n == total_k:
        # both shells saturate; can't discriminate d
        return float("nan")
    if total_n == 0:
        return float("nan")

    target_ratio = total_n / total_k

    def ratio_minus_target(d: float) -> float:
        # p(d) = exp(log V(t1) - log V(t2))
        lv1 = log_cumvol(d, t1, A)
        lv2 = log_cumvol(d, t2, A)
        if not math.isfinite(lv1) or not math.isfinite(lv2):
            return 1.0  # nudge brentq
        return math.exp(lv1 - lv2) - target_ratio

    # p(d) is monotone decreasing in d for t1 < t2 (small shells become
    # negligible as d grows). Bracket and root-find.
    lo, hi = d_min, d_max
    f_lo = ratio_minus_target(lo)
    f_hi = ratio_minus_target(hi)
    if f_lo * f_hi > 0:
        # no sign change in bracket — return endpoint nearest target
        return d_min if abs(f_lo) < abs(f_hi) else d_max
    return float(brentq(ratio_minus_target, lo, hi, xtol=1e-4))


def auto_choose_radii(distances: np.ndarray, q1: float = 0.05, q2: float = 0.15):
    """Pick (t1, t2) so neither shell saturates nor empties.

    Heuristic from Macocco: choose so that t1 has small but non-zero
    fraction of points (q1) and t2 has a larger fraction (q2). Returns
    integer radii.
    """
    N = distances.shape[0]
    # collect off-diagonal distances
    iu = np.triu_indices(N, k=1)
    flat = distances[iu]
    t1 = int(np.quantile(flat, q1))
    t2 = int(np.quantile(flat, q2))
    if t2 <= t1:
        t2 = t1 + 1
    return t1, t2


# ============================================================================
# ARCH-B: Corrected TwoNN order statistics
# ============================================================================

def log_likelihood_twonn(pairs: list, d: float, M: int, A: int = 3) -> float:
    """Log-likelihood of (r1, r2) pairs under corrected TwoNN.

    M = N - 1 (number of candidate distances per point, excluding self).
    For r1 < r2:  log[(M)(M-1)] + log p(r1) + log p(r2) + (M-2) log S(r2)
    For r1 == r2: log C(M, 2) + 2 log p(r1) + (M-2) log S(r1)
    """
    ll = 0.0
    log_pair = math.log(M) + math.log(M - 1)
    log_tie = math.log(M) + math.log(M - 1) - math.log(2)
    for r1, r2 in pairs:
        if r1 < r2:
            lp1 = log_p_distance(d, r1, A)
            lp2 = log_p_distance(d, r2, A)
            ls = log_S(d, r2, A)
            if not (math.isfinite(lp1) and math.isfinite(lp2)):
                return -math.inf
            ll += log_pair + lp1 + lp2 + (M - 2) * (ls if math.isfinite(ls) else 0.0)
        else:  # r1 == r2
            lp = log_p_distance(d, r1, A)
            ls = log_S(d, r1, A)
            if not math.isfinite(lp):
                return -math.inf
            ll += log_tie + 2 * lp + (M - 2) * (ls if math.isfinite(ls) else 0.0)
    return ll


def estimate_id_twonn(
    distances: np.ndarray,
    d_grid: np.ndarray = None,
    A: int = 3,
) -> float:
    """Corrected TwoNN-style MLE: joint PMF of two nearest distances per point."""
    N = distances.shape[0]
    M = N - 1  # candidates per point

    pairs = []
    for i in range(N):
        row = distances[i].copy()
        row[i] = np.iinfo(row.dtype).max
        sorted_d = np.sort(row)
        r1 = int(sorted_d[0])
        r2 = int(sorted_d[1])
        if r1 == 0:
            continue  # duplicate points
        pairs.append((r1, r2))

    if not pairs:
        return float("nan")

    max_r = max(max(r1, r2) for r1, r2 in pairs)
    if d_grid is None:
        d_grid = np.arange(max(1.0, float(max_r)), 200.5, 0.5)

    best_d = d_grid[0]
    best_ll = -math.inf
    for d in d_grid:
        ll = log_likelihood_twonn(pairs, d, M, A)
        if math.isfinite(ll) and ll > best_ll:
            best_ll = ll
            best_d = d
    return float(best_d)


# ============================================================================
# Backwards-compatible alias (for calibrate.py)
# ============================================================================

def estimate_id_ternary(distances: np.ndarray, d_grid: np.ndarray = None) -> float:
    """Default v2 estimator: ARCH-A (Macocco fixed-radii) with auto radii.

    The fixed-radii estimator is simpler, matches Macocco's published
    paper exactly, and is computationally cheaper than the order-statistics
    MLE. ARCH-B is exposed via estimate_id_twonn() for cross-validation.
    """
    t1, t2 = auto_choose_radii(distances)
    return estimate_id_fixed_radii(distances, t1=t1, t2=t2)


if __name__ == "__main__":
    # smoke
    rng = np.random.default_rng(0)
    sigs = rng.integers(-1, 2, size=(50, 30), dtype=np.int8)
    d = pairwise_categorical_hamming(sigs)
    t1, t2 = auto_choose_radii(d)
    print(f"auto (t1, t2) = ({t1}, {t2})")
    d_hat_a = estimate_id_fixed_radii(d, t1=t1, t2=t2)
    d_hat_b = estimate_id_twonn(d)
    print(f"ARCH-A: d_hat = {d_hat_a:.2f}")
    print(f"ARCH-B: d_hat = {d_hat_b:.2f}")
