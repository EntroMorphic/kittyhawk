"""M1 v1 — DEMONSTRABLY BROKEN. Kept as reference.

This is the v1 implementation that FAILED calibration with ~38%
systematic underestimation for d ≥ 20. The pre-registered stop-
criterion (FROZEN per `journal/td27_geometric_prereg_v2_2026-05-12.md`)
fired on first attempt; implementation halted. See
`journal/td27_phase_alpha_calibration_fail_2026-05-12.md` for the
diagnosis.

Bug: the pairwise likelihood P(r1, r2 | d) ∝ V(r1) · V(r2) treats r1
and r2 as independent draws. They aren't: r1 is the minimum over N-1
candidate distances (extreme-value statistics), and r2 ≥ r1 by
construction. The correct likelihood needs the survival factor
S(R)^(M-2) and tie handling. Fixed in m1_estimator_v2.py (ARCH-B).
Macocco's actual paper uses a different estimator architecture entirely
(fixed-radii Binomial counts, ARCH-A in v2).

USE m1_estimator_v2.py FOR CATEGORICAL-HAMMING WORK.
USE phase_beta/m1_l1_estimator.py FOR L1 CELL-GRAPH WORK
(the metric that respects the ternary alphabet's path-graph structure).

Derivation (FROZEN in journal/td27_geometric_prereg_v2_2026-05-12.md):

For substrate signatures in {-1, 0, +1}^D under categorical Hamming
(d(s1, s2) = count of disagreeing cells):

Shell volume:  V(r) = C(D, r) * 2^r
  (Choose r positions to disagree; at each, 2 non-equal values.)

For TwoNN-like I3D under this discrete metric, the binomial-shell
likelihood (Macocco et al. 2023 framework, generalized to alphabet
A=3) gives an MLE estimate of local intrinsic dimension d.

Estimator: for each point, take r1 ≤ r2 = first two nearest neighbor
distances. Under uniform-density local manifold of dimension d, the
shell-volume ratio gives:

  Volume within radius R on d-dim manifold = sum_{i=0}^R C(d, i) * 2^i

We treat d as a continuous parameter and MLE over (r1, r2) pairs.
"""
import math
import numpy as np


def shell_volume_ternary(D: float, r: int) -> float:
    """V(r) = C(D, r) * 2^r where D is treated as continuous.
    Uses Gamma function for non-integer D.
    """
    if r < 0:
        return 0.0
    # C(D, r) via Gamma; valid for non-integer D
    log_choose = (
        math.lgamma(D + 1) - math.lgamma(r + 1) - math.lgamma(D - r + 1)
    )
    return math.exp(log_choose + r * math.log(2))


def cumulative_volume_ternary(D: float, R: int) -> float:
    """sum_{i=0}^{R} C(D, i) * 2^i for integer R, continuous D."""
    if R < 0:
        return 0.0
    total = 0.0
    for i in range(R + 1):
        total += shell_volume_ternary(D, i)
    return total


def log_likelihood_pair(r1: int, r2: int, d: float) -> float:
    """Log-likelihood of observing (r1, r2) as the 2 nearest neighbor
    distances under a local manifold of intrinsic dimension d, under
    ternary categorical Hamming.

    Generalization of Macocco's I3D binary-shell likelihood: r2 = r1's
    next-shell distance; under uniform local density, the conditional
    probability is ratio of shell volumes.

    P(r2 | r1, d) ∝ V(r2) for r2 ≥ r1, normalized.
    We use the simpler form: P(r2 ≥ R | r1 = r0, d) =
       (sum_{r ≥ R} V(r)) / (sum_{r ≥ r0+1} V(r))

    For MLE we use the pairwise likelihood of seeing the EXACT (r1, r2):
        L = V(r1) * V(r2) / Z (each independent under uniformity)
    But this doesn't account for ordering. The TwoNN trick:
       μ = r2 - r1, and the ratio of cumulative volumes determines d.

    For the continuous case, log-lik(d) = log(d) - (d+1)*log(μ).
    For discrete categorical Hamming, we replace continuous distances
    with shell-volume cumulants:

      Under uniform local density of dim d, P(r1 = R) ~ V(R, d).

    We use a pairwise estimator: for observed r1, r2 pairs, MLE d
    such that P(both observed in shells r1 and r2) is maximized.
    """
    # Robust pairwise likelihood: P(r1) * P(r2 | r2 ≥ r1)
    # P(r1) ∝ V(r1, d) for the first observation
    # P(r2 | r2 ≥ r1, d) ∝ V(r2, d) for r2 ≥ r1
    log_v1 = math.lgamma(d + 1) - math.lgamma(r1 + 1) - math.lgamma(d - r1 + 1) + r1 * math.log(2)
    log_v2 = math.lgamma(d + 1) - math.lgamma(r2 + 1) - math.lgamma(d - r2 + 1) + r2 * math.log(2)
    return log_v1 + log_v2


def estimate_id_ternary(distances: np.ndarray, d_grid: np.ndarray = None) -> float:
    """Estimate intrinsic dimension from a pairwise distance matrix
    under ternary categorical Hamming.

    distances: (N, N) integer matrix of Hamming distances. Diagonal is 0.
    d_grid: candidate d values (default 1.0 to D in steps of 0.5)
    Returns: argmax-likelihood d̂.
    """
    N = distances.shape[0]
    # For each point i, find r1 < r2 (first two nearest neighbors excluding self)
    pairs = []
    for i in range(N):
        row = distances[i].copy()
        row[i] = np.iinfo(row.dtype).max  # exclude self
        sorted_d = np.sort(row)
        # r1 = smallest positive distance; r2 = next
        r1 = int(sorted_d[0])
        r2 = int(sorted_d[1])
        if r1 == 0:  # duplicate points; skip
            continue
        if r2 <= r1:
            continue
        pairs.append((r1, r2))

    if not pairs:
        return float("nan")

    D_ambient = distances.shape[0]  # the data dim used as upper bound
    # Actually D_ambient = sig_dim; for substrate this is HEAD_DIM = 128
    # The d_grid should run from 1 to D_max (the ambient max)
    D_max = max(max(r2 for _, r2 in pairs) + 5, 10)
    if d_grid is None:
        d_grid = np.arange(1.0, min(D_max, 200.0) + 0.5, 0.5)

    best_d = d_grid[0]
    best_ll = -math.inf
    for d in d_grid:
        ll = 0.0
        try:
            for r1, r2 in pairs:
                # Skip pairs where r > d (impossible to have r1 > d in d-dim space)
                if r2 > d:
                    ll = -math.inf
                    break
                ll += log_likelihood_pair(r1, r2, d)
        except (ValueError, OverflowError):
            continue
        if ll > best_ll:
            best_ll = ll
            best_d = d
    return float(best_d)


def pairwise_categorical_hamming(signatures: np.ndarray) -> np.ndarray:
    """Compute pairwise categorical Hamming distance.

    signatures: (N, D) int8 array with values in {-1, 0, +1}.
    Returns: (N, N) int32 distance matrix.
    """
    N = signatures.shape[0]
    dist = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        dist[i] = np.sum(signatures != signatures[i:i+1], axis=1)
    return dist


if __name__ == "__main__":
    # Quick smoke
    sigs = np.array([
        [1, 0, -1, 1, 0],
        [1, 0, -1, 1, 0],   # identical to row 0
        [1, 0, -1, -1, 0],  # 1 disagreement
        [-1, 1, 1, 1, 0],   # 3 disagreements with row 0
    ], dtype=np.int8)
    d_mat = pairwise_categorical_hamming(sigs)
    print("Hamming matrix:")
    print(d_mat)
    print()
    # ID on this trivial example (N=4, D=5) won't be meaningful
    # but the function should not crash
    d_hat = estimate_id_ternary(d_mat)
    print(f"Estimated d (toy data): {d_hat}")
