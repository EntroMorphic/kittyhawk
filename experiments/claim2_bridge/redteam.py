"""Red-team the claim 2 bridge.

Tests equivalence classes deliberately NOT included in the original
battery, including:
  - Subtraction non-associativity TRAP (should differ; if it matches,
    the bridge is over-simplifying).
  - Polynomial expansion (bridge doesn't have CAS expansion).
  - Mixed-variable division identities.
  - Division distributivity (breaks under integer truncation).
  - Numeric/variable interactions in complex expressions.
  - Variable factoring (bridge can't factor without a CAS).

Plus a "false equivalence" check: a class where the LHS and RHS are
NOT mathematically equivalent. If the bridge says they are, that's a
collision indicating a substantive bug.
"""
from __future__ import annotations

import os
import sys
import numpy as np

THIS = os.path.dirname(__file__)
sys.path.insert(0, THIS)

from canonical import signature_from_expr as sig_A
from routing import signature_from_expr as sig_B


D = 128


# Equivalence classes the original battery did NOT include.
# Each entry: (class_name, [[expr_a, expr_b, expected_match], ...])
# expected_match = True means they ARE algebraically equivalent (under
# integer arithmetic + standard algebra).
# expected_match = False means they are NOT equivalent — a match here
# would be a false-positive collision.
RED_TEAM = {
    "subtraction_non_associativity_TRAP": [
        # (a - b) - c = a - b - c  vs  a - (b - c) = a - b + c
        # These are NOT equivalent in general algebra.
        ("(a - b) - c", "a - (b - c)", False),
    ],
    "subtraction_distributivity": [
        # x * (y - z) = x*y - x*z   should hold by element-wise routing
        ("x * (y - z)", "x*y - x*z", True),
    ],
    "polynomial_expansion": [
        # Mathematical identity that requires algebraic expansion.
        # Bridge has no CAS — expected to FAIL preservation.
        ("(x + y) * (x + y)", "x*x + 2*x*y + y*y", True),
        ("(x + 1) * (x - 1)", "x*x - 1", True),
    ],
    "factoring": [
        # Reverse direction — requires factoring.
        ("x*x - x", "x * (x - 1)", True),
        ("x*x + x", "x * (x + 1)", True),
    ],
    "exact_div_cancellation": [
        # (x * y) / y = x — algebraic cancellation. Mixed div uses SHA
        # fallback in approach B unless _simplify catches it.
        ("(x * y) / y", "x", True),
        ("(x + y) - y", "x", True),  # ADDITIVE cancellation analogue
    ],
    "div_distributivity_under_int_trunc": [
        # Under INTEGER truncation, (a+b)/c and a/c+b/c MAY coincide
        # by accident even when they differ in real arithmetic. The
        # bridge correctly implements integer semantics. Tests check
        # the integer-arithmetic outcome.
        ("(7 + 1) / 3", "7/3 + 1/3", True),   # 8/3=2;  2+0=2.   coincide.
        ("(8 + 1) / 4", "8/4 + 1/4", True),   # 9/4=2;  2+0=2.   coincide.
        ("(5 + 3) / 4", "5/4 + 3/4", False),  # 8/4=2;  1+0=1.   differ.
    ],
    "complex_nested_numeric": [
        # Should fold completely via balt routings.
        ("((2 + 3) * 4) - 1", "19", True),
        ("(10 - 2) * (3 + 1)", "32", True),
    ],
    "deep_distributivity": [
        # (a + b) * (c + d) = ac + ad + bc + bd — element-wise routing
        # should give this on the trit substrate (analogous to
        # distributivity but with more terms).
        ("(a + b) * (c + d)", "a*c + a*d + b*c + b*d", True),
    ],
    "constant_zero_propagation": [
        # 0 * <anything> should equal 0 regardless of complexity.
        ("0 * (x + y + z + 1)", "0", True),
        ("(x - x) * y", "0", True),
    ],
}


def run(approach_name: str, sig_fn) -> None:
    print(f"\n=== {approach_name} ===")
    print(f"{'class':<40} {'expr_a':<22} {'expr_b':<24} {'expected':>8} {'got':>6}  {'verdict':<8}")
    print("-" * 120)
    correct = 0
    wrong = 0
    bugs = []
    for cls, pairs in RED_TEAM.items():
        for expr_a, expr_b, expected_match in pairs:
            sa = sig_fn(expr_a, d=D)
            sb = sig_fn(expr_b, d=D)
            l1 = int(np.abs(sa.astype(int) - sb.astype(int)).sum())
            matched = (l1 == 0)
            if matched == expected_match:
                correct += 1
                verdict = "OK"
            else:
                wrong += 1
                # Note: when expected=False but got matched, that's a
                # false-positive (signature collision). When expected=True
                # but got differ, that's a missed identity.
                if expected_match:
                    verdict = "MISS"
                else:
                    verdict = "FALSE-EQ"
                bugs.append((cls, expr_a, expr_b, expected_match, matched, l1))
            print(f"  {cls:<38} {expr_a:<22} {expr_b:<24}"
                  f" {str(expected_match):>8} {str(matched):>6}  {verdict:<10}")
    print(f"\nresults: correct={correct}  wrong={wrong}")
    if bugs:
        print(f"\nbugs ({len(bugs)}):")
        for cls, a, b, want, got, l1 in bugs:
            if want:
                print(f"  MISS — '{a}' should equal '{b}'   ({cls})  got L1={l1}")
            else:
                print(f"  FALSE-EQ — '{a}' should differ from '{b}'   ({cls})  got L1={l1}")


def main():
    print(f"=== Claim 2 bridge red-team ===")
    print(f"D = {D}, classes = {len(RED_TEAM)}, "
          f"pairs = {sum(len(v) for v in RED_TEAM.values())}")
    run("A_canonical_hash", sig_A)
    run("B_routing_derived", sig_B)


if __name__ == "__main__":
    main()
