"""Red-team the exp/log bridge integration.

Six attack surfaces:
  R1 fp-vs-integer signature mismatch for same mathematical value.
  R2 log of zero / negative.
  R3 overflow at large exp arguments.
  R4 deep nested compositions (precision degradation).
  R5 pure-numeric algebraic identities: exp(a+b)=exp(a)*exp(b),
     log(a*b)=log(a)+log(b).
  R6 mixed-expression algebraic identities for exp/log (expected
     to fail under SHA fallback).
"""
from __future__ import annotations

import math
import os
import sys
import numpy as np

THIS = os.path.dirname(__file__)
sys.path.insert(0, THIS)

from routing import signature_from_expr as sig
from fixed_point import fp_decode, FixedPoint, SCALE_DEFAULT


def decode_fp_sig(s: np.ndarray) -> float:
    return fp_decode(FixedPoint(s.astype(np.int8), SCALE_DEFAULT))


def check_match(expr_a: str, expr_b: str, expect: bool, name: str) -> tuple[bool, str]:
    """Return (ok, message). expect=True means signatures should be equal."""
    try:
        sa = sig(expr_a)
        sb = sig(expr_b)
    except Exception as e:
        return False, f"  CRASH on {expr_a!r} or {expr_b!r}: {e}"
    l1 = int(np.abs(sa.astype(int) - sb.astype(int)).sum())
    matched = (l1 == 0)
    ok = (matched == expect)
    verdict = "OK" if ok else ("MISS" if expect else "FALSE-EQ")
    return ok, (f"  {name:<30} {expr_a:<28} {expr_b:<28}"
                f" expect={expect}  got={matched}  L1={l1}  {verdict}")


def main():
    print("=== R1: fp_const vs integer-const same-value behavior ===")
    # exp(0) vs 1: post-fix the demotion rule recognizes exp_taylor(0)
    # as integer-valued and emits ('const', 1), matching the integer
    # literal's signature. Design tension closed.
    ok, msg = check_match("exp(0)", "1", True, "exp(0)-vs-int1 (closed)")
    print(msg)
    # log(1) vs 0: BOTH encode to all-zero (fp_encode(0) and base_sig_const(0)
    # both produce all-zero by construction). They accidentally bridge the
    # two encodings. Mathematically log(1)=0, so this match is correct.
    ok, msg = check_match("log(1)", "0", True, "log(1)-vs-int0")
    print(msg)
    # exp(log(5)) vs 5: with the exp(log(e)) → e rewrite, both fold to
    # integer 5. The bridge now correctly identifies them as equal.
    ok, msg = check_match("exp(log(5))", "5", True, "exp(log(5))-vs-int5")
    print(msg)
    # fp_const must match itself.
    ok, msg = check_match("exp(0)", "exp(0)", True, "fp-self")
    print(msg)

    print("\n=== R2: log of zero / negative — does the bridge crash gracefully? ===")
    for expr in ["log(0)", "log(-1)", "exp(log(0))"]:
        try:
            s = sig(expr)
            v = decode_fp_sig(s)
            print(f"  {expr:<20} got value = {v:>20}  (no exception)")
        except Exception as e:
            print(f"  {expr:<20} raised {type(e).__name__}: {e}")

    print("\n=== R3: large exp arguments (precision + range) ===")
    # exp_taylor itself is verified in fixed_point.py self-test; here we
    # verify the bridge's signature for exp(N) matches what the integer-
    # equivalent value would produce (with demotion). At large N, math.exp
    # exceeds float64 fractional precision and rounds to an integer at
    # float64 — bridge correctly demotes via the integer-rounding rule.
    for n in [10, 20, 30, 50]:
        try:
            s = sig(f"exp({n})")
            want = math.exp(n)
            # If math.exp(N) is integer-equivalent at float64 precision
            # (which is true for large N), the bridge demotes via
            # _simplify's rounding check. Check signature equality
            # against the corresponding integer literal.
            want_int = round(want)
            if abs(want - want_int) < 1e-9:
                want_sig = sig(str(want_int))
                l1 = int(np.abs(s.astype(int) - want_sig.astype(int)).sum())
                status = "MATCH int" if l1 == 0 else f"L1={l1}"
            else:
                # Decode as fp (true for moderate N where exp is non-integer)
                got = decode_fp_sig(s)
                err = abs(got - want) / max(abs(want), 1e-30)
                status = "OK" if err < 1e-10 else f"err={err:.2e}"
            print(f"  exp({n:>3})  math={want:>22.6g}  bridge: {status}")
        except Exception as e:
            print(f"  exp({n:>3}) raised {type(e).__name__}: {e}")

    print("\n=== R4: deep nested compositions ===")
    # After exp(log(e)) → e and log(exp(e)) → e rewrites, all of these
    # algebraically reduce to plain 5 (or the inner constant). They
    # should ALL match sig('5') at the signature level.
    five_sig = sig("5")
    for expr in ["exp(log(5))", "log(exp(5))",
                  "exp(log(exp(log(5))))", "log(exp(log(exp(5))))",
                  "exp(log(exp(log(exp(log(5))))))"]:
        try:
            s = sig(expr)
            l1 = int(np.abs(s.astype(int) - five_sig.astype(int)).sum())
            verdict = "MATCH integer 5" if l1 == 0 else f"L1={l1} from int(5)"
            print(f"  {expr:<46} {verdict}")
        except Exception as e:
            print(f"  {expr:<46} raised {type(e).__name__}: {e}")

    print("\n=== R5: pure-numeric algebraic identities ===")
    # exp(a + b) == exp(a) * exp(b)  with numeric a, b
    pairs_r5 = [
        ("exp(2 + 3)",       "exp(2) * exp(3)"),
        ("exp(1 + 1)",       "exp(1) * exp(1)"),
        ("log(2 * 3)",       "log(2) + log(3)"),
        ("log(5 * 7)",       "log(5) + log(7)"),
        ("exp(-1)",          "1 / exp(1)"),
    ]
    for a, b in pairs_r5:
        ok, msg = check_match(a, b, True, "pure-numeric-id")
        print(msg)

    print("\n=== R6: mixed-expression algebraic identities (expected miss) ===")
    pairs_r6 = [
        ("exp(x + y)",       "exp(x) * exp(y)"),
        ("log(x * y)",       "log(x) + log(y)"),
        ("exp(log(x))",      "x"),
        ("log(exp(x))",      "x"),
    ]
    for a, b in pairs_r6:
        ok, msg = check_match(a, b, True, "mixed-id")
        print(msg)


if __name__ == "__main__":
    main()
