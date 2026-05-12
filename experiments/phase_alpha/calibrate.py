"""Phase α calibration: validate M1 estimator on synthetic ternary
data of known intrinsic dimension.

Per FROZEN spec in journal/td27_geometric_prereg_v2_2026-05-12.md:
For each target d ∈ {2, 5, 10, 20, 50, 100}, generate N=5000 random
vectors in {-1, 0, +1}^d, embed into {-1, 0, +1}^128 via random
injection. Estimator must recover d within 20% relative.

If calibration fails, implementation STOPS until resolved.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from m1_estimator import pairwise_categorical_hamming, estimate_id_ternary


D_AMBIENT = 128


def synthetic_ternary_data(N: int, d_true: int, seed: int = 42):
    """Generate N points in a d_true-dim ternary subspace embedded in
    {-1, 0, +1}^D_AMBIENT.

    Per spec: d cells carry the manifold structure (random ternary);
    D-d cells are random padding (uniform ternary). Random injection
    of which cells are 'real' (fixed for the dataset).
    """
    rng = np.random.default_rng(seed)
    # Pick which D_AMBIENT cells are 'real' vs 'padding'
    perm = rng.permutation(D_AMBIENT)
    real_cells = perm[:d_true]
    pad_cells = perm[d_true:]

    sigs = np.zeros((N, D_AMBIENT), dtype=np.int8)
    # Real cells: uniform {-1, 0, +1}, varying per sample
    sigs[:, real_cells] = rng.integers(-1, 2, size=(N, d_true), dtype=np.int8)
    # Padding cells: uniform {-1, 0, +1} also varies (otherwise it's a constant
    # that doesn't contribute to distance, making it equivalent to d_true)
    # CAREFUL: if padding varies independently, the OBSERVED Hamming distances
    # include padding contributions. That makes the apparent d = D_AMBIENT, not d_true.
    # The correct embedding: padding cells should be FIXED (constants) per
    # signature dataset to truly represent a d_true-dim manifold.
    # Pick a fixed value for each padding cell (per dataset).
    pad_constants = rng.integers(-1, 2, size=d_true, dtype=np.int8) if False else np.zeros(D_AMBIENT - d_true, dtype=np.int8)
    pad_constants = rng.integers(-1, 2, size=D_AMBIENT - d_true, dtype=np.int8)
    sigs[:, pad_cells] = pad_constants[np.newaxis, :]  # broadcast constants

    return sigs


def calibration_run():
    """Run calibration across target d values. Returns (d_true, d_hat) tuples."""
    results = []
    for d_true in [2, 5, 10, 20, 50, 100]:
        sigs = synthetic_ternary_data(N=5000, d_true=d_true)
        # For tractability at N=5000, subsample to N=500 for the calibration
        idx = np.random.default_rng(7).choice(5000, size=500, replace=False)
        sigs_sub = sigs[idx]
        dist = pairwise_categorical_hamming(sigs_sub)
        d_hat = estimate_id_ternary(dist)
        rel_err = abs(d_hat - d_true) / d_true if d_true > 0 else float("inf")
        results.append((d_true, d_hat, rel_err))
        print(f"d_true={d_true:3d}  d_hat={d_hat:6.2f}  rel_err={rel_err:.2%}")
    return results


if __name__ == "__main__":
    print("=== Phase α Calibration ===")
    print(f"Ambient D = {D_AMBIENT}")
    print(f"Synthetic N = 500 (subsampled from 5000)")
    print()
    results = calibration_run()
    print()
    # Pass criterion: all rel_err < 20%
    failed = [r for r in results if r[2] >= 0.20]
    if failed:
        print(f"CALIBRATION FAIL: {len(failed)} of {len(results)} target d values exceeded 20% relative error")
        for d, d_hat, err in failed:
            print(f"  FAIL: d_true={d}, d_hat={d_hat:.2f}, err={err:.2%}")
        sys.exit(1)
    else:
        print(f"CALIBRATION PASS: all {len(results)} target d values within 20% relative error")
        sys.exit(0)
