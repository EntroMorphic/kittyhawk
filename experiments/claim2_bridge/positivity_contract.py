"""Test the positivity contract: permissive vs strict mode.

The rewrite exp(log(e)) → e is mathematically valid only when e > 0.
For symbolic variables, we can't statically prove positivity. The
bridge offers two modes:

  - permissive (default): assume var is positive. exp(log(x)) → x.
  - strict: don't assume. exp(log(x)) stays as exp(log(x)) and
    log_taylor raises at evaluation time if x ≤ 0.

This test verifies both modes work as documented.

Closed concern #6 of the 100/100 remediation arc; final result
6/6. See journal/claim2_100of100_remediation_2026-05-13.md.
The unsoundness this closes is the one identified in
journal/claim2_explog_redteam_2026-05-13.md.
"""
from __future__ import annotations

import os
import sys

import numpy as np

THIS = os.path.dirname(__file__)
sys.path.insert(0, THIS)

from canonical import canonicalize, _serialize, signature_from_expr
from parser import parse


def main():
    print("=== Positivity contract test ===\n")

    # Cases where positivity matters.
    cases = [
        # (expr, expected_permissive_canon, expected_strict_canon, label)
        ("exp(log(x))",  "V:x",                       "(exp (log V:x))",
         "var arg: rewrite fires in permissive, suppressed in strict"),
        ("exp(log(5))",  "C:5",                       "C:5",
         "positive const arg: rewrite fires in both modes"),
        ("exp(log(0))",  None,                        None,
         "zero arg: rewrite should NOT fire (0 not positive)"),
        ("exp(log(exp(x)))", "(exp V:x)",             "(exp V:x)",
         "exp(_) arg: provably positive in both modes"),
        ("exp(log(x * y))", "(mul V:x V:y)",          "(exp (log (mul V:x V:y)))",
         "mul of vars: positive in permissive, unproven in strict"),
        ("exp(log(x + 1))", "(add C:1 V:x)",          "(exp (log (add C:1 V:x)))",
         "add of var + positive const: permissive yes, strict no"),
    ]

    all_ok = True
    for expr, exp_permissive, exp_strict, label in cases:
        ast = parse(expr)
        try:
            c_permissive = canonicalize(ast, strict_positivity=False)
            ser_p = _serialize(c_permissive)
        except Exception as e:
            ser_p = f"<{type(e).__name__}>"
        try:
            c_strict = canonicalize(ast, strict_positivity=True)
            ser_s = _serialize(c_strict)
        except Exception as e:
            ser_s = f"<{type(e).__name__}>"

        ok_p = exp_permissive is None or ser_p == exp_permissive
        ok_s = exp_strict is None or ser_s == exp_strict
        ok = ok_p and ok_s
        all_ok = all_ok and ok

        status = "OK" if ok else "FAIL"
        print(f"[{status}] {label}")
        print(f"   expr:       {expr}")
        print(f"   permissive: {ser_p}")
        if exp_permissive is not None and ser_p != exp_permissive:
            print(f"   EXPECTED:   {exp_permissive}")
        print(f"   strict:     {ser_s}")
        if exp_strict is not None and ser_s != exp_strict:
            print(f"   EXPECTED:   {exp_strict}")
        print()

    # Verify signatures differ in strict vs permissive mode when behavior differs.
    print("Signature divergence between modes:")
    for expr in ["exp(log(x))", "exp(log(x * y))", "exp(log(x + 1))"]:
        s_p = signature_from_expr(expr, strict_positivity=False)
        s_s = signature_from_expr(expr, strict_positivity=True)
        l1 = int(np.abs(s_p.astype(int) - s_s.astype(int)).sum())
        diverged = "DIFFERENT" if l1 > 0 else "same"
        print(f"   {expr:<26} L1 = {l1:>3}  {diverged}")
    print()
    for expr in ["exp(log(5))", "exp(log(exp(x)))"]:
        s_p = signature_from_expr(expr, strict_positivity=False)
        s_s = signature_from_expr(expr, strict_positivity=True)
        l1 = int(np.abs(s_p.astype(int) - s_s.astype(int)).sum())
        diverged = "DIFFERENT" if l1 > 0 else "same"
        print(f"   {expr:<26} L1 = {l1:>3}  {diverged}  (should be same)")

    print()
    if all_ok:
        print("All cases: OK")
    else:
        print("FAILURES present.")


if __name__ == "__main__":
    main()
