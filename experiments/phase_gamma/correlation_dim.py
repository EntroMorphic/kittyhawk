"""Correlation dimension estimator — independent second method.

Grassberger-Procaccia correlation integral:
    C(r) = (2 / N(N-1)) · #{(i, j), i<j : d(i, j) < r}

For small r, C(r) ~ r^d on a d-dim manifold. The correlation dimension
d_C is the slope of log C(r) vs log r in the scaling region.

We pick the scaling region as quantiles q1=0.10 to q2=0.40 of the
pairwise distance distribution, fit by ordinary least squares.
"""
from __future__ import annotations

import math
import numpy as np


def correlation_dimension(dist: np.ndarray,
                          q_lo: float = 0.10,
                          q_hi: float = 0.40,
                          n_points: int = 20) -> dict:
    """Estimate intrinsic dim via the Grassberger-Procaccia correlation
    integral.

    Args:
        dist: (N, N) pairwise distance matrix (any metric).
        q_lo, q_hi: quantile bounds for the scaling region.
        n_points: number of radii to fit.

    Returns: {d_hat, r_lo, r_hi, slope_se, n_points_used}.
    """
    N = dist.shape[0]
    iu = np.triu_indices(N, k=1)
    flat = dist[iu].astype(np.float64)
    # avoid r=0 in log
    flat = flat[flat > 0]
    if len(flat) < 100:
        return {"d_hat": float("nan"), "r_lo": float("nan"),
                "r_hi": float("nan"), "slope_se": float("nan"),
                "n_points_used": 0}
    r_lo = float(np.quantile(flat, q_lo))
    r_hi = float(np.quantile(flat, q_hi))
    if r_hi <= r_lo:
        r_hi = r_lo + 1.0
    radii = np.linspace(max(r_lo, 1.0), r_hi, n_points)
    log_r = np.log(radii)
    log_C = []
    n_total = len(flat)
    for r in radii:
        count = float(np.sum(flat < r))
        if count <= 0:
            log_C.append(-math.inf)
        else:
            log_C.append(math.log(count / n_total))
    log_C = np.array(log_C)
    mask = np.isfinite(log_C)
    if mask.sum() < 3:
        return {"d_hat": float("nan"), "r_lo": r_lo, "r_hi": r_hi,
                "slope_se": float("nan"), "n_points_used": int(mask.sum())}
    x = log_r[mask]
    y = log_C[mask]
    # OLS slope + standard error
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    Sxx = np.sum((x - x_mean) ** 2)
    Sxy = np.sum((x - x_mean) * (y - y_mean))
    slope = Sxy / Sxx
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    sse = np.sum((y - y_pred) ** 2)
    se = math.sqrt(sse / max(n - 2, 1) / Sxx)
    return {"d_hat": float(slope), "r_lo": r_lo, "r_hi": r_hi,
            "slope_se": float(se), "n_points_used": int(n)}


if __name__ == "__main__":
    # Smoke: uniform random in [0, 1]^d should give d_hat ≈ d
    rng = np.random.default_rng(0)
    for d_true in (3, 10, 30):
        X = rng.uniform(0, 1, size=(500, d_true))
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))
        r = correlation_dimension(D)
        print(f"d_true={d_true:3d}  d_C={r['d_hat']:6.2f}  ±{r['slope_se']:.2f}")
