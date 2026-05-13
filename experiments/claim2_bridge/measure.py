"""Claim 2 bridge first measurement loop.

Runs both approaches (canonical-form-hash, routing-derived) on a
battery of expressions grouped by equivalence class. Reports
preservation rate per class and collision rate on distinct
expressions.
"""
from __future__ import annotations

import os
import sys
import numpy as np

THIS = os.path.dirname(__file__)
sys.path.insert(0, THIS)

from canonical import signature_from_expr as sig_A
from routing   import signature_from_expr as sig_B


D = 128


# Equivalence classes — each list is a group whose members are
# pairwise considered "the same expression" for the equivalence.
EQUIV_CLASSES = {
    "commutativity_add":      [["x + y", "y + x"], ["a + b + c", "c + b + a"]],
    "commutativity_mul":      [["x * y", "y * x"], ["a * b * c", "c * b * a"]],
    "associativity_add":      [["(x + y) + z", "x + (y + z)"]],
    "associativity_mul":      [["(x * y) * z", "x * (y * z)"]],
    "identity_add":           [["x + 0", "x"], ["0 + x", "x"]],
    "identity_mul":           [["x * 1", "x"], ["1 * x", "x"]],
    "absorbing_zero":         [["x * 0", "0"], ["0 * x", "0"]],
    "additive_inverse":       [["x - x", "0"], ["x + (-x)", "0"]],
    "double_negation":        [["-(-x)", "x"]],
    "distributivity":         [["x * (y + z)", "x*y + x*z"]],
    "diff_of_squares":        [["(x + y) * (x - y)", "x*x - y*y"]],
    "constant_arithmetic":    [["2 + 3", "5"], ["3 + 2", "5"], ["3 * 4", "12"],
                                ["10 - 7", "3"], ["2 * 3 + 1", "7"], ["100 + 23", "123"],
                                ["-5 + 5", "0"], ["2 * 3 * 4", "24"]],
    "numeric_in_expr":        [["x + 2 + 3", "x + 5"], ["x * 2 * 3", "x * 6"],
                                ["x + (10 - 7)", "x + 3"]],
    "division_numeric":       [["12 / 3", "4"], ["12 / 4", "3"], ["100 / 7", "14"],
                                ["-12 / 3", "-4"], ["7 / 7", "1"], ["0 / 5", "0"]],
    "division_in_expr":       [["x + 12 / 3", "x + 4"], ["x * (10 / 2)", "x * 5"],
                                ["x / x", "1"], ["x / 1", "x"], ["0 / x", "0"]],
    # exp/log: pure-numeric folds via Taylor at fixed-point scale 40.
    # Equivalences hold at the FIXED-POINT trit-vector level.
    "exp_log_numeric":        [["exp(0)", "exp(0)"], ["log(1)", "log(1)"],
                                ["exp(log(2))", "exp(log(2))"]],
    "exp_log_consistency":    [["exp(1)", "exp(1)"], ["log(10)", "log(10)"]],
}


# Distinct-expression collision check: expressions that should
# have DIFFERENT signatures.
DISTINCT = [
    "x", "y", "z",
    "x + y", "x - y", "x * y",
    "x + 1", "x + 2", "x - 1",
    "x * x", "x * y * z",
    "1", "2", "3", "0", "-1",
    "x + y + z", "x * y + z", "x + y * z",
    "x * x + y * y", "x*x - y*y",
    "12 / 3", "x / y", "x / 2",
]


def equivalence_preservation(approach_name: str, sig_fn) -> dict:
    """For each class, compute fraction of pairs with L1=0."""
    results = {}
    for cls, groups in EQUIV_CLASSES.items():
        n_pairs = 0
        n_match = 0
        per_pair_l1 = []
        for group in groups:
            sigs = [sig_fn(e, d=D) for e in group]
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    n_pairs += 1
                    l1 = int(np.abs(sigs[i].astype(int) - sigs[j].astype(int)).sum())
                    per_pair_l1.append(l1)
                    if l1 == 0:
                        n_match += 1
        results[cls] = {
            "n_pairs": n_pairs,
            "n_match": n_match,
            "rate":    n_match / n_pairs if n_pairs else float("nan"),
            "l1s":     per_pair_l1,
        }
    return results


def collision_rate(approach_name: str, sig_fn) -> dict:
    """For DISTINCT expressions, count signature collisions."""
    sigs = {e: sig_fn(e, d=D) for e in DISTINCT}
    pairs = []
    n_pairs = 0
    n_collisions = 0
    for i, a in enumerate(DISTINCT):
        for b in DISTINCT[i+1:]:
            n_pairs += 1
            l1 = int(np.abs(sigs[a].astype(int) - sigs[b].astype(int)).sum())
            if l1 == 0:
                n_collisions += 1
                pairs.append((a, b))
    return {"n_pairs": n_pairs, "n_collisions": n_collisions,
            "rate": n_collisions / n_pairs, "collision_pairs": pairs}


def determinism(sig_fn) -> bool:
    """Same expression evaluated twice gives same signature."""
    for e in ["x + y", "x * (y + z)", "0", "(x+y)*(x-y)"]:
        a = sig_fn(e, d=D)
        b = sig_fn(e, d=D)
        if not np.array_equal(a, b):
            return False
    return True


def main():
    print(f"=== Claim 2 bridge first measurement loop ===\n")
    print(f"D = {D}, equiv classes = {len(EQUIV_CLASSES)}, "
          f"distinct exprs = {len(DISTINCT)}\n")

    for name, fn in (("A_canonical_hash", sig_A),
                     ("B_routing_derived", sig_B)):
        print(f"\n--- approach {name} ---")
        print(f"determinism: {'OK' if determinism(fn) else 'FAIL'}")

        eq = equivalence_preservation(name, fn)
        print(f"\nEquivalence preservation rate per class:")
        print(f"  {'class':<22} {'pairs':>6} {'match':>6} {'rate':>7}  mean L1 (over fails)")
        print(f"  " + "-" * 65)
        for cls, stats in eq.items():
            fails = [l for l in stats["l1s"] if l > 0]
            mean_l1 = np.mean(fails) if fails else 0.0
            marker = " ✓" if stats["rate"] == 1.0 else (" ✗" if stats["rate"] < 1.0 else "")
            print(f"  {cls:<22} {stats['n_pairs']:>6} {stats['n_match']:>6} "
                  f"{stats['rate']*100:>6.1f}%   {mean_l1:>6.1f}{marker}")

        col = collision_rate(name, fn)
        print(f"\nDistinct-expression collisions:")
        print(f"  pairs={col['n_pairs']}  collisions={col['n_collisions']}  "
              f"rate={col['rate']*100:.2f}%")
        if col["collision_pairs"]:
            print(f"  collision pairs (expected distinct, got same sig):")
            for a, b in col["collision_pairs"][:10]:
                print(f"    {a!r} == {b!r}")


if __name__ == "__main__":
    main()
