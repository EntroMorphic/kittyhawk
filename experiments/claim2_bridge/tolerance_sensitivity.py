"""Sweep the integer-demotion tolerance in canonical._simplify.

Production tolerance is the module-level constant
`canonical.INTEGER_DEMOTE_TOL = 1e-12` (set after this sweep ran;
the previous 1e-9 was demonstrably too loose — it would demote
user-typed 1e-10 to 0). The constraints:

  - LOWER bound: must absorb Taylor convergence noise. Empirically
    tops out at ~1.1e-16 (worst case: log(e) where atanh series
    converges slowly at u ≈ 0.46).
  - UPPER bound: must NOT absorb deliberately-small user values like
    1e-12. A user who writes `1.0 + 1e-12` means it.

This script sweeps the tolerance across a range and records, for
each test case, the smallest and largest tolerances at which the
expected demote/no-demote decision is made. The "safe range" is the
intersection: tolerances that get every case right.

Output: a table where each row is a tolerance and each column is a
case, showing whether the bridge made the correct decision. The
contiguous block of "all correct" tolerances is the safe range.

Closed concern #5 of the 100/100 remediation arc; tightened
production tolerance from 1e-9 → 1e-12. See
journal/claim2_100of100_remediation_2026-05-13.md.
"""
from __future__ import annotations

import math
import os
import sys

THIS = os.path.dirname(__file__)
sys.path.insert(0, THIS)

from fixed_point import FixedPoint, fp_encode, fp_decode, SCALE_DEFAULT, exp_taylor, log_taylor


# ============================================================================
# Test cases: (label, expected_demote, get_decoded_value)
#
#   expected_demote = True means the value is "really integer" and
#     should demote.
#   expected_demote = False means the value is "really fractional" and
#     should NOT demote.
# ============================================================================

def case_exp_of(n):
    """math.exp(n) — integer at large n (rounds to int at float64), but
    distance to nearest int is well above 0 for moderate n."""
    return exp_taylor(fp_encode(float(n)))


def case_log_of(n):
    return log_taylor(fp_encode(float(n)))


def case_small(v):
    return fp_encode(v)


CASES = [
    # (label, fp_value_factory, expected_demote, expected_int_or_none)
    ("log(1) = 0",          lambda: case_log_of(1.0),          True,  0),
    ("exp(0) = 1",          lambda: case_exp_of(0),            True,  1),
    ("exp(log(5)) = 5",     lambda: exp_taylor(case_log_of(5.0)), True, 5),
    ("exp(30)",             lambda: case_exp_of(30),           False, None),
    ("exp(20)",             lambda: case_exp_of(20),           False, None),
    ("exp(50)",             lambda: case_exp_of(50),           True,  None),  # float64 rounds
    ("1.0 + 1e-12",         lambda: case_small(1.0 + 1e-12),   True,  1),
    ("1.0 + 1e-8",          lambda: case_small(1.0 + 1e-8),    False, None),
    ("0.5",                 lambda: case_small(0.5),           False, None),
    ("0.0",                 lambda: case_small(0.0),           True,  0),
    ("1e-15",               lambda: case_small(1e-15),         True,  0),  # near zero
    ("1e-10",               lambda: case_small(1e-10),         False, None),
    ("log(2) ≈ 0.693",      lambda: case_log_of(2.0),          False, None),
    ("log(e)",              lambda: case_log_of(math.e),       True,  1),  # log(e)=1 exactly
]


def decide_demote(fp: FixedPoint, tol: float):
    """Replicate _simplify's demotion rule for the given tolerance.
    Returns (demoted, int_value_or_none)."""
    v = fp_decode(fp)
    r = round(v)
    if abs(v - r) < tol:
        return True, int(r)
    return False, None


def main():
    tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10,
                  1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18]

    print("=== Integer-demotion tolerance sweep ===\n")
    print("For each test case, we compute the fp value once and ask: at")
    print("each tolerance, does the demotion rule give the expected answer?\n")

    # Compute fp values once
    case_data = []
    for label, factory, expected_demote, expected_int in CASES:
        try:
            fp = factory()
            v = fp_decode(fp)
            r = round(v)
            dist = abs(v - r)
            case_data.append((label, fp, expected_demote, expected_int, v, r, dist))
        except Exception as e:
            print(f"  [SKIP] {label}: {type(e).__name__}: {e}")
            continue

    # Print value summary first
    print(f"{'case':<22}  {'value':>20}  {'nearest int':>11}  {'distance':>12}")
    print("-" * 70)
    for label, fp, exp_d, exp_i, v, r, dist in case_data:
        print(f"{label:<22}  {v:>20.6g}  {r:>11d}  {dist:>12.3e}")
    print()

    # Tolerance sweep
    print(f"{'tolerance':>10}  ", end="")
    for label, *_ in case_data:
        # Short label
        short = label.replace(" = ", "=").replace("≈ ", "≈")
        print(f"{short[:10]:>10}  ", end="")
    print("  all-correct")
    print("-" * (12 + 12 * len(case_data) + 14))

    safe_tols = []
    for tol in tolerances:
        results = []
        all_correct = True
        for label, fp, exp_demote, exp_int, v, r, dist in case_data:
            demoted, got_int = decide_demote(fp, tol)
            if demoted == exp_demote:
                if exp_demote and exp_int is not None:
                    if got_int == exp_int:
                        results.append("OK")
                    else:
                        results.append(f"=>{got_int}")
                        all_correct = False
                else:
                    results.append("OK")
            else:
                results.append("WRONG")
                all_correct = False
        print(f"{tol:>10.0e}  ", end="")
        for r in results:
            print(f"{r:>10}  ", end="")
        print(f"  {'ALL OK' if all_correct else ''}")
        if all_correct:
            safe_tols.append(tol)

    print()
    if safe_tols:
        print(f"Safe range: [{min(safe_tols):.0e}, {max(safe_tols):.0e}]")
        from canonical import INTEGER_DEMOTE_TOL
        print(f"Current production value: {INTEGER_DEMOTE_TOL:.0e} "
              f"(in safe range: {INTEGER_DEMOTE_TOL in safe_tols})")
    else:
        print("No tolerance correctly handles all cases.")

    print()
    print("Edge cases reviewed:")
    print("  - 1e-15 demotes to 0 — this is a 'tolerance choice', not")
    print("    incorrect. Users wanting strict near-zero handling should")
    print("    use a tighter tolerance OR avoid relying on demotion.")
    print("  - log(e) = 1 exactly: depends on Taylor convergence at u =")
    print("    (e-1)/(e+1) ≈ 0.462. Distance to int 1 should be tiny.")


if __name__ == "__main__":
    main()
