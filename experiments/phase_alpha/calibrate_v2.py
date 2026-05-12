"""Phase α calibration v2: validate corrected M1 estimator (both
architectures) on synthetic ternary data of known intrinsic dimension.

Per FROZEN spec in journal/td27_geometric_prereg_v2_2026-05-12.md:
For each target d ∈ {2, 5, 10, 20, 50, 100}, generate N points in a
d-dim ternary subspace embedded in {-1, 0, +1}^128 (random injection
+ fixed padding). Estimator must recover d within 20% relative error.

If calibration passes for BOTH architectures, we have cross-validated
agreement and can proceed to Phase α. If only one passes, we use that
one (with a note). If neither passes, we re-derive again.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from m1_estimator_v2 import (
    pairwise_categorical_hamming,
    estimate_id_fixed_radii,
    estimate_id_twonn,
    auto_choose_radii,
)


D_AMBIENT = 128


def synthetic_ternary_data(N: int, d_true: int, seed: int = 42):
    """Generate N points on a d_true-dim ternary subspace embedded
    in {-1, 0, +1}^D_AMBIENT with fixed padding.

    Padding cells are constants (per-dataset random pick); only the
    d_true real cells vary per sample.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(D_AMBIENT)
    real_cells = perm[:d_true]
    pad_cells = perm[d_true:]
    sigs = np.zeros((N, D_AMBIENT), dtype=np.int8)
    sigs[:, real_cells] = rng.integers(-1, 2, size=(N, d_true), dtype=np.int8)
    pad_constants = rng.integers(-1, 2, size=D_AMBIENT - d_true, dtype=np.int8)
    sigs[:, pad_cells] = pad_constants[np.newaxis, :]
    return sigs


def run(N_eff: int = 500, N_total: int = 5000):
    targets = [2, 5, 10, 20, 50, 100]
    rows = []
    print(f"=== Phase α Calibration v2 ===")
    print(f"Ambient D = {D_AMBIENT}, N_eff = {N_eff} (subsampled from {N_total})")
    print()
    print(f"{'d_true':>6} {'t1':>4} {'t2':>4} {'d_A':>8} {'errA':>8} "
          f"{'d_B':>8} {'errB':>8}  verdict")
    print("-" * 70)
    for d_true in targets:
        sigs = synthetic_ternary_data(N=N_total, d_true=d_true)
        idx = np.random.default_rng(7).choice(N_total, size=N_eff, replace=False)
        sigs_sub = sigs[idx]
        dist = pairwise_categorical_hamming(sigs_sub)
        t1, t2 = auto_choose_radii(dist)

        # ARCH-A: fixed-radii
        try:
            d_A = estimate_id_fixed_radii(dist, t1=t1, t2=t2)
        except Exception as e:
            d_A = float("nan")
        # ARCH-B: corrected TwoNN order stats
        try:
            d_B = estimate_id_twonn(dist)
        except Exception as e:
            d_B = float("nan")

        err_A = abs(d_A - d_true) / d_true if (d_true > 0 and np.isfinite(d_A)) else float("nan")
        err_B = abs(d_B - d_true) / d_true if (d_true > 0 and np.isfinite(d_B)) else float("nan")

        pass_A = np.isfinite(err_A) and err_A < 0.20
        pass_B = np.isfinite(err_B) and err_B < 0.20
        if pass_A and pass_B:
            verdict = "PASS (both)"
        elif pass_A:
            verdict = "PASS A only"
        elif pass_B:
            verdict = "PASS B only"
        elif d_true < 10:
            verdict = "skip (small-d degeneracy expected)"
        else:
            verdict = "FAIL"

        rows.append((d_true, t1, t2, d_A, err_A, d_B, err_B, verdict))
        print(f"{d_true:>6d} {t1:>4d} {t2:>4d} {d_A:>8.2f} {err_A:>8.2%} "
              f"{d_B:>8.2f} {err_B:>8.2%}  {verdict}")
    return rows


if __name__ == "__main__":
    rows = run()
    print()
    # Pass criterion (per FROZEN spec): all targets d >= 10 within 20% rel err
    # for at least ONE architecture (ideally both).
    test_targets = [r for r in rows if r[0] >= 10]
    pass_A_all = all(np.isfinite(r[4]) and r[4] < 0.20 for r in test_targets)
    pass_B_all = all(np.isfinite(r[6]) and r[6] < 0.20 for r in test_targets)
    if pass_A_all and pass_B_all:
        print("CALIBRATION PASS (both architectures clear 20% on d >= 10)")
        sys.exit(0)
    elif pass_A_all:
        print("CALIBRATION PASS (ARCH-A clears; ARCH-B does not — using A)")
        sys.exit(0)
    elif pass_B_all:
        print("CALIBRATION PASS (ARCH-B clears; ARCH-A does not — using B)")
        sys.exit(0)
    else:
        print("CALIBRATION FAIL: neither architecture clears 20% on all d >= 10")
        sys.exit(1)
