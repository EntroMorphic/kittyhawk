"""Calibrate the L1-Macocco estimator on synthetic structured-ternary
data of known intrinsic dim. Phase β FROZEN gate: |d̂ − d|/d < 0.20
for d ∈ {10, 20, 50, 100}.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase_alpha"))

from m1_l1_estimator import (
    pairwise_L1_int8,
    estimate_id_L1,
    substrate_cell_pmf,
    cell_pmf_from_data,
)


D_AMBIENT = 128


def synthetic_structured_ternary(N: int, d_true: int, p_nonzero: float = 0.62,
                                   seed: int = 42):
    """N points in a d_true-dim ternary subspace embedded in {-1, 0, +1}^128.

    Real cells: Bernoulli(p_nonzero) for nonzero, ±1 uniform among nonzero.
    Padding cells: fixed per-dataset constants (so they don't contribute
    to pairwise distances).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(D_AMBIENT)
    real_cells = perm[:d_true]
    pad_cells = perm[d_true:]
    sigs = np.zeros((N, D_AMBIENT), dtype=np.int8)
    nz = rng.random((N, d_true)) < p_nonzero
    signs = rng.choice([-1, 1], size=(N, d_true)).astype(np.int8)
    sigs[:, real_cells] = (nz * signs).astype(np.int8)
    pad_constants = rng.integers(-1, 2, size=D_AMBIENT - d_true, dtype=np.int8)
    sigs[:, pad_cells] = pad_constants[np.newaxis, :]
    return sigs


def run_calibration(N: int = 500, p_nonzero: float = 0.62):
    targets = [10, 20, 50, 100]
    print(f"=== Phase β L1 calibration ===")
    print(f"  N={N}, D_ambient={D_AMBIENT}, p_nonzero={p_nonzero}")
    print()
    print(f"{'d_true':>6} {'d_hat':>8} {'rel_err':>8} {'t1':>4} {'t2':>4} {'pmf_emp':>26}  verdict")
    print("-" * 80)
    rows = []
    pmf_data = substrate_cell_pmf(p_nonzero)
    for d_true in targets:
        sigs = synthetic_structured_ternary(N=N, d_true=d_true,
                                              p_nonzero=p_nonzero)
        dist = pairwise_L1_int8(sigs)
        pmf_emp = cell_pmf_from_data(sigs)
        m = estimate_id_L1(dist, pmf_cell=pmf_emp)
        rel = abs(m["d_hat"] - d_true) / d_true
        verdict = "PASS" if rel < 0.20 else "FAIL"
        pmf_str = f"[{pmf_emp[0]:.3f},{pmf_emp[1]:.3f},{pmf_emp[2]:.3f}]"
        print(f"{d_true:>6d} {m['d_hat']:>8.2f} {rel:>8.2%} {m['t1']:>4d} {m['t2']:>4d} {pmf_str:>26s}  {verdict}")
        rows.append({"d_true": d_true, "d_hat": m["d_hat"], "rel_err": rel,
                     "t1": m["t1"], "t2": m["t2"], "pmf_emp": pmf_emp.tolist()})
    return rows


if __name__ == "__main__":
    rows = run_calibration()
    print()
    failed = [r for r in rows if r["rel_err"] >= 0.20]
    if failed:
        print(f"CALIBRATION FAIL ({len(failed)}/{len(rows)} targets exceed 20% rel err)")
        sys.exit(1)
    print(f"CALIBRATION PASS ({len(rows)}/{len(rows)} targets within 20% rel err)")
    sys.exit(0)
