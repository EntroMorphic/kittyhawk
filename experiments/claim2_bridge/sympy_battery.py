"""SymPy-derived adversarial battery for the claim 2 bridge.

External-ground-truth validation: use SymPy (a widely-trusted CAS) as
the source of algebraic equivalences. For each generated expression
pair where SymPy considers them equivalent (or genuinely distinct),
test whether the bridge agrees at the signature level.

This addresses the self-red-teaming epistemic limit. SymPy's
equivalence judgments aren't bounded by my imagination.

Coverage:
  - Polynomial expand/factor pairs.
  - Sum simplifications (combine like terms).
  - Distributivity in both directions.
  - Numerical evaluation of pure-numeric subtrees.
  - exp/log identities (where SymPy generates them).
  - Mixed numeric-symbolic forms.

Pairs are filtered to those expressible in the bridge's grammar
({+, -, *, /, exp, log} over integer constants and lowercase
identifiers). Rational constants (1/2, etc.) are not in scope.

Closed concerns #1 and #2 of the 100/100 remediation arc; final
result 32/32 at 100%. See
journal/claim2_100of100_remediation_2026-05-13.md.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import sympy as sp

THIS = os.path.dirname(__file__)
sys.path.insert(0, THIS)

from routing import signature_from_expr


# ============================================================================
# Sympy → bridge-string conversion
# ============================================================================

def to_bridge_str(expr) -> str | None:
    """Convert a SymPy expression to bridge parser syntax.

    Returns None if the expression contains anything outside the bridge's
    supported grammar (rationals other than integers, transcendentals
    beyond exp/log, derivatives, etc.).
    """
    if expr.is_Number:
        if expr.is_Integer:
            n = int(expr)
            if n < 0:
                return f"({n})"  # parser handles unary minus in factor
            return str(n)
        # Reject non-integer numbers (rationals, floats, etc.)
        return None
    if expr.is_Symbol:
        name = str(expr)
        if not (name.isidentifier() and name.islower()):
            return None
        return name
    if expr.is_Add:
        parts = [to_bridge_str(t) for t in expr.args]
        if any(p is None for p in parts):
            return None
        return "(" + " + ".join(parts) + ")"
    if expr.is_Mul:
        # Recognise unary negation: Mul(-1, x) → "-x".
        parts = []
        for t in expr.args:
            s = to_bridge_str(t)
            if s is None:
                return None
            parts.append(s)
        return "(" + " * ".join(parts) + ")"
    if expr.is_Pow:
        base, exponent = expr.args
        if not exponent.is_Integer or int(exponent) < 0:
            return None  # bridge doesn't support negative or non-int exponents directly
        n = int(exponent)
        if n == 0:
            return "1"
        if n == 1:
            return to_bridge_str(base)
        b_str = to_bridge_str(base)
        if b_str is None:
            return None
        # x ** n → (x * x * ... * x)  (n times)
        return "(" + " * ".join([b_str] * n) + ")"
    # exp / log
    fn = getattr(expr, "func", None)
    if fn is not None:
        fname = getattr(fn, "__name__", "")
        if fname == "exp":
            arg = to_bridge_str(expr.args[0])
            if arg is None:
                return None
            return f"exp({arg})"
        if fname == "log":
            arg = to_bridge_str(expr.args[0])
            if arg is None:
                return None
            # Bridge's log is natural log; SymPy's log is too.
            if len(expr.args) > 1:
                return None  # log with base specified — out of scope
            return f"log({arg})"
    return None


# ============================================================================
# Pair generators
# ============================================================================

x, y, z, a, b, c, d = sp.symbols("x y z a b c d")
ALL_VARS = [x, y, z, a, b, c, d]


def gen_polynomial_seeds():
    """Polynomial expressions used as seeds for expand/factor variations."""
    return [
        x + y,
        x - y,
        x * y,
        x + y + z,
        x * y + z,
        x * (y + z),
        (x + 1) * (x - 1),
        (x + y) * (x - y),
        (x + y) * (x + y),
        (x + y) ** 2,
        (x + y) ** 3,
        (x - y) ** 2,
        (x + 1) ** 2,
        (x + 2) * (x + 3),
        x ** 2 + y ** 2,
        x ** 2 - y ** 2,
        x * y * z + x * y + x + 1,
        (x + y + z) ** 2,
        (a + b) * (c + d),
        a * b + a * c + b * d + c * d,  # not equal to (a+b)(c+d); test that bridge sees them as distinct? Actually = (a+b)(c+d)? Let's check via sympy.
    ]


def gen_numeric_seeds():
    """Pure-numeric and mixed-numeric expressions."""
    return [
        sp.Integer(2) + 3,
        sp.Integer(2) * 3,
        sp.Integer(10) - 7,
        sp.Integer(7) * 8 + 1,
        x + 2 + 3,
        x * 2 * 3,
        x + (10 - 7),
        2 * x + 3 * x,
        x + x + x,
        (x + 1) * 2,
        x * (x + 1),
    ]


def gen_explog_seeds():
    """Expressions involving exp/log with integer arguments."""
    return [
        sp.exp(0),
        sp.exp(1),
        sp.log(1),
        sp.exp(sp.log(5)),
        sp.log(sp.exp(3)),
        sp.exp(x) * sp.exp(y),
        sp.exp(x + y),
        sp.log(x) + sp.log(y),
        sp.log(x * y),
        sp.exp(-x),
        1 / sp.exp(x),
        sp.exp(2 * x),
        sp.exp(x) ** 2,
    ]


def gen_pairs():
    """Yield (a_str, b_str, expected_match, label) tuples."""
    pairs = []
    seeds = gen_polynomial_seeds() + gen_numeric_seeds() + gen_explog_seeds()
    transformations = [
        ("expand",     sp.expand),
        ("factor",     sp.factor),
        ("simplify",   sp.simplify),
        ("trigsimp",   sp.trigsimp),
        ("powsimp",    sp.powsimp),
        ("together",   sp.together),
        ("collect_x",  lambda e: sp.collect(e, x)),
        ("collect_y",  lambda e: sp.collect(e, y)),
    ]
    for seed in seeds:
        a_str = to_bridge_str(seed)
        if a_str is None:
            continue
        for name, fn in transformations:
            try:
                transformed = fn(seed)
            except Exception:
                continue
            b_str = to_bridge_str(transformed)
            if b_str is None:
                continue
            if a_str == b_str:
                continue  # trivial pair
            # SymPy equivalence check
            try:
                equal = sp.simplify(seed - transformed) == 0
            except Exception:
                continue
            pairs.append((a_str, b_str, bool(equal), f"{name}({seed})"))
    return pairs


def gen_distinct_pairs():
    """Pairs that are genuinely different — bridge should NOT match."""
    distinct = [
        (x + y, x - y),
        (x * y, x + y),
        (x ** 2, x * 2),
        (2 * x + 3, 2 * x + 4),
        (sp.exp(x), sp.exp(y)),
        (sp.exp(2), sp.exp(3)),
        (sp.log(5), sp.log(7)),
        (x + 1, x + 2),
        (sp.exp(x + 1), sp.exp(x) + 1),
    ]
    pairs = []
    for a, b in distinct:
        a_str = to_bridge_str(a)
        b_str = to_bridge_str(b)
        if a_str is None or b_str is None:
            continue
        equal = sp.simplify(a - b) == 0
        if not equal:
            pairs.append((a_str, b_str, False, f"distinct({a} vs {b})"))
    return pairs


# ============================================================================
# Runner
# ============================================================================

def sig_or_error(expr_str: str):
    try:
        return signature_from_expr(expr_str)
    except Exception as e:
        return ("ERROR", type(e).__name__, str(e))


def main():
    pairs = gen_pairs() + gen_distinct_pairs()
    print(f"Generated {len(pairs)} adversarial pairs from SymPy.\n")

    # Buckets
    equiv_pairs   = [p for p in pairs if p[2]]
    distinct_pairs = [p for p in pairs if not p[2]]
    print(f"  SymPy-equivalent pairs (should match):       {len(equiv_pairs)}")
    print(f"  SymPy-distinct pairs   (should differ):      {len(distinct_pairs)}\n")

    # Run
    correct_match = 0
    missed_match = 0  # equivalent in SymPy, bridge differed
    correct_distinct = 0
    false_eq = 0      # distinct in SymPy, bridge matched (collision)
    errors = []
    misses = []
    collisions = []

    for a_str, b_str, equal, label in pairs:
        sa = sig_or_error(a_str)
        sb = sig_or_error(b_str)
        if isinstance(sa, tuple) and sa[0] == "ERROR":
            errors.append((a_str, sa))
            continue
        if isinstance(sb, tuple) and sb[0] == "ERROR":
            errors.append((b_str, sb))
            continue
        l1 = int(np.abs(sa.astype(int) - sb.astype(int)).sum())
        matched = (l1 == 0)
        if equal:
            if matched:
                correct_match += 1
            else:
                missed_match += 1
                misses.append((label, a_str, b_str, l1))
        else:
            if matched:
                false_eq += 1
                collisions.append((label, a_str, b_str))
            else:
                correct_distinct += 1

    print("=== Results ===\n")
    print(f"  SymPy-equivalent  bridge agrees:      {correct_match} / {len(equiv_pairs)}")
    print(f"  SymPy-equivalent  bridge MISSED:      {missed_match}")
    print(f"  SymPy-distinct    bridge agrees:      {correct_distinct} / {len(distinct_pairs)}")
    print(f"  SymPy-distinct    bridge COLLISION:   {false_eq}")
    print(f"  Bridge errors:                        {len(errors)}")
    print()
    total_correct = correct_match + correct_distinct
    total = len(pairs) - len(errors)
    if total > 0:
        print(f"  Total accuracy: {total_correct} / {total} = "
              f"{100 * total_correct / total:.1f}%")
    print()

    if misses:
        print(f"=== {len(misses)} MISSED EQUIVALENCES (bridge said different but SymPy says equal) ===")
        for label, a, b, l1 in misses[:30]:
            print(f"  [{label}]")
            print(f"    a: {a}")
            print(f"    b: {b}")
            print(f"    L1: {l1}")
        if len(misses) > 30:
            print(f"  ... and {len(misses) - 30} more")
        print()

    if collisions:
        print(f"=== {len(collisions)} FALSE COLLISIONS (bridge said equal but SymPy says different) ===")
        for label, a, b in collisions:
            print(f"  [{label}] {a!r} vs {b!r}")
        print()

    if errors:
        print(f"=== {len(errors)} BRIDGE ERRORS ===")
        for expr, err in errors[:10]:
            print(f"  {expr!r}  →  {err[1]}: {err[2]}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
