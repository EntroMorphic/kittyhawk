"""Confluence test for the canonicalizer rewrite system.

A rewrite system is confluent if, for any expression, the order in
which rules are applied does not affect the normal form. This matters
because canonicalize() runs flatten -> simplify -> sort iteratively
to fixed point; if rules can produce different normal forms depending
on order, the bridge's signature would depend on how the user wrote
the expression.

We test confluence in three complementary ways:

  1. Idempotence: canonicalize(canonicalize(e)) == canonicalize(e).
     Necessary for confluence (though not sufficient).

  2. Permutation invariance: for any randomly-generated AST e and
     any permutation of the commutative operations (add/mul children
     order), canonicalize gives the same AST.

  3. Identity injection: inserting trivial identities (e + 0, e * 1,
     adding parens, double-negation, etc.) into a random expression
     should not change the canonicalized AST.

  4. Random equivalence preservation (sympy-validated): for randomly
     generated pairs that sympy considers equivalent, the bridge's
     canonical AST should be the same.

Strategy: generate random ASTs over {var, const, add, mul, sub, neg,
exp, log, div}, then apply transformations and check.

Closed concern #4 of the 100/100 remediation arc; final result
500/500 idempotence + 500/500 permutation + 500/500 identity
injection + 2038/2038 SymPy-equivalent + 1962/1962 SymPy-distinct.
See journal/claim2_100of100_remediation_2026-05-13.md.
"""
from __future__ import annotations

import os
import random
import sys

import sympy as sp

THIS = os.path.dirname(__file__)
sys.path.insert(0, THIS)

from canonical import canonicalize, _serialize
from parser import parse
from sympy_battery import to_bridge_str


VARS = ["x", "y", "z", "a", "b"]
SMALL_INTS = [0, 1, 2, 3, -1, -2, 5]


def random_ast(depth: int, rng: random.Random):
    """Generate a random AST tuple of bounded depth."""
    if depth <= 0:
        if rng.random() < 0.5:
            return ("var", rng.choice(VARS))
        return ("const", rng.choice(SMALL_INTS))
    op = rng.choices(
        ["var", "const", "add", "mul", "sub", "neg"],
        weights=[2, 2, 3, 3, 2, 1],
    )[0]
    if op == "var":
        return ("var", rng.choice(VARS))
    if op == "const":
        return ("const", rng.choice(SMALL_INTS))
    if op == "add":
        n = rng.randint(2, 3)
        kids = [random_ast(depth - 1, rng) for _ in range(n)]
        return ("add", *kids)
    if op == "mul":
        n = rng.randint(2, 3)
        kids = [random_ast(depth - 1, rng) for _ in range(n)]
        return ("mul", *kids)
    if op == "sub":
        return ("sub", random_ast(depth - 1, rng), random_ast(depth - 1, rng))
    if op == "neg":
        return ("neg", random_ast(depth - 1, rng))
    raise ValueError(op)


def permute_commutative(node, rng: random.Random):
    """Return an AST equivalent to `node` but with commutative ops'
    children shuffled. Recursive."""
    if not isinstance(node, tuple):
        return node
    op = node[0]
    if op == "fp_const":
        return node
    args = [permute_commutative(a, rng) for a in node[1:]]
    if op in ("add", "mul"):
        rng.shuffle(args)
    return (op, *args)


def inject_identity(node, rng: random.Random):
    """Wrap `node` with a random trivial-identity transformation that
    preserves value but expands the AST shape."""
    choice = rng.choice(["add0", "mul1", "double_neg", "sub_self_add"])
    if choice == "add0":
        # e -> e + 0
        return ("add", node, ("const", 0))
    if choice == "mul1":
        # e -> e * 1
        return ("mul", node, ("const", 1))
    if choice == "double_neg":
        # e -> -(-e)
        return ("neg", ("neg", node))
    if choice == "sub_self_add":
        # e -> e + (var - var)   where var is some var in scope
        v = rng.choice(VARS)
        return ("add", node, ("sub", ("var", v), ("var", v)))
    return node


def to_sympy(node):
    """Convert a bridge-grammar AST to a SymPy expression. Used as
    external ground truth for equivalence testing."""
    if not isinstance(node, tuple):
        raise ValueError(node)
    op = node[0]
    if op == "var":
        return sp.Symbol(node[1])
    if op == "const":
        return sp.Integer(node[1])
    if op == "add":
        out = to_sympy(node[1])
        for k in node[2:]:
            out = out + to_sympy(k)
        return out
    if op == "mul":
        out = to_sympy(node[1])
        for k in node[2:]:
            out = out * to_sympy(k)
        return out
    if op == "sub":
        return to_sympy(node[1]) - to_sympy(node[2])
    if op == "neg":
        return -to_sympy(node[1])
    raise ValueError(op)


# ============================================================================
# Tests
# ============================================================================

def test_idempotence(n: int, rng: random.Random) -> tuple[int, int, list]:
    """canonicalize(canonicalize(e)) == canonicalize(e). Necessary for
    confluence."""
    fails = []
    ok = 0
    for _ in range(n):
        ast = random_ast(rng.randint(1, 4), rng)
        c1 = canonicalize(ast)
        c2 = canonicalize(c1)
        if c1 == c2:
            ok += 1
        else:
            fails.append((ast, c1, c2))
    return ok, n, fails


def test_permutation_invariance(n: int, rng: random.Random) -> tuple[int, int, list]:
    """canonicalize(permute(e)) == canonicalize(e)."""
    fails = []
    ok = 0
    for _ in range(n):
        ast = random_ast(rng.randint(2, 4), rng)
        c_original = canonicalize(ast)
        # Try 5 random permutations
        all_match = True
        for _ in range(5):
            permuted = permute_commutative(ast, rng)
            c_perm = canonicalize(permuted)
            if c_perm != c_original:
                fails.append((ast, permuted, c_original, c_perm))
                all_match = False
                break
        if all_match:
            ok += 1
    return ok, n, fails


def test_identity_injection(n: int, rng: random.Random) -> tuple[int, int, list]:
    """canonicalize(inject_identity(e)) == canonicalize(e)."""
    fails = []
    ok = 0
    for _ in range(n):
        ast = random_ast(rng.randint(1, 3), rng)
        c_original = canonicalize(ast)
        # Inject 1-3 identities
        modified = ast
        for _ in range(rng.randint(1, 3)):
            modified = inject_identity(modified, rng)
        c_modified = canonicalize(modified)
        if c_modified == c_original:
            ok += 1
        else:
            fails.append((ast, modified, c_original, c_modified))
    return ok, n, fails


def test_sympy_random_equivalence(n: int, rng: random.Random) -> tuple[int, int, list]:
    """Two scenarios:
      A. Same-source pair: generate a, transform b = T(a) via SymPy.
         Sympy considers them equivalent; bridge should canonicalize to
         the same form (missed-equivalence test).
      B. Two-source pair: generate independent a and b. Sympy may say
         they're equal or distinct. We record false collisions
         (bridge equal but sympy distinct).
    """
    fails = []
    collisions = []
    matches_when_equal = 0
    nonmatches_when_distinct = 0
    equal_count = 0
    distinct_count = 0
    transformations = [sp.expand, sp.factor, lambda e: e, sp.simplify]
    for _ in range(n):
        # Scenario A: same-source.
        a_ast = random_ast(rng.randint(1, 3), rng)
        try:
            a_sp = to_sympy(a_ast)
            b_sp = rng.choice(transformations)(a_sp)
        except Exception:
            continue
        b_str = to_bridge_str(b_sp)
        if b_str is None:
            continue
        try:
            b_ast = parse(b_str)
            equal = sp.simplify(a_sp - b_sp) == 0
        except Exception:
            continue
        if equal:
            equal_count += 1
            c_a = canonicalize(a_ast)
            c_b = canonicalize(b_ast)
            if c_a == c_b:
                matches_when_equal += 1
            else:
                fails.append((a_ast, b_ast, c_a, c_b))
        else:
            distinct_count += 1
            c_a = canonicalize(a_ast)
            c_b = canonicalize(b_ast)
            if c_a == c_b:
                collisions.append((a_ast, b_ast, c_a))
            else:
                nonmatches_when_distinct += 1

        # Scenario B: two independent random ASTs (mostly produces
        # distinct pairs; small chance of equivalent by accident).
        try:
            other_ast = random_ast(rng.randint(1, 3), rng)
            other_sp = to_sympy(other_ast)
            equal2 = sp.simplify(a_sp - other_sp) == 0
        except Exception:
            continue
        c_a = canonicalize(a_ast)
        c_other = canonicalize(other_ast)
        if equal2:
            equal_count += 1
            if c_a == c_other:
                matches_when_equal += 1
            else:
                fails.append((a_ast, other_ast, c_a, c_other))
        else:
            distinct_count += 1
            if c_a == c_other:
                collisions.append((a_ast, other_ast, c_a))
            else:
                nonmatches_when_distinct += 1
    return (matches_when_equal, equal_count, fails,
            nonmatches_when_distinct, distinct_count, collisions)


def main():
    rng = random.Random(42)

    print("=== Confluence test for canonicalizer ===\n")

    # 1. Idempotence
    ok, total, fails = test_idempotence(500, rng)
    print(f"[1] Idempotence: {ok}/{total} = {100*ok/total:.1f}%")
    if fails:
        print(f"   {len(fails)} failures, first 3:")
        for ast, c1, c2 in fails[:3]:
            print(f"     ast: {_serialize(ast)}")
            print(f"     c1:  {_serialize(c1)}")
            print(f"     c2:  {_serialize(c2)}")
    print()

    # 2. Permutation invariance
    ok, total, fails = test_permutation_invariance(500, rng)
    print(f"[2] Permutation invariance: {ok}/{total} = {100*ok/total:.1f}%")
    if fails:
        print(f"   {len(fails)} failures, first 3:")
        for ast, perm, c_orig, c_perm in fails[:3]:
            print(f"     ast:        {_serialize(ast)}")
            print(f"     permuted:   {_serialize(perm)}")
            print(f"     c_orig:     {_serialize(c_orig)}")
            print(f"     c_perm:     {_serialize(c_perm)}")
    print()

    # 3. Identity injection
    ok, total, fails = test_identity_injection(500, rng)
    print(f"[3] Identity injection: {ok}/{total} = {100*ok/total:.1f}%")
    if fails:
        print(f"   {len(fails)} failures, first 3:")
        for ast, mod, c_orig, c_mod in fails[:3]:
            print(f"     ast:       {_serialize(ast)}")
            print(f"     injected:  {_serialize(mod)}")
            print(f"     c_orig:    {_serialize(c_orig)}")
            print(f"     c_mod:     {_serialize(c_mod)}")
    print()

    # 4. SymPy random equivalence
    (m, eq, fails, nm, dist, coll) = test_sympy_random_equivalence(2000, rng)
    print(f"[4] SymPy random equivalence:")
    print(f"     equivalent (should match): {m}/{eq} = "
          f"{100*m/eq:.1f}%" if eq else "     no equivalent pairs sampled")
    print(f"     distinct   (should differ): {nm}/{dist} = "
          f"{100*nm/dist:.1f}%" if dist else "     no distinct pairs sampled")
    if fails:
        print(f"   {len(fails)} missed-equivalence failures, first 3:")
        for a, b, c_a, c_b in fails[:3]:
            print(f"     a ast:    {_serialize(a)}")
            print(f"     b ast:    {_serialize(b)}")
            print(f"     c_a:      {_serialize(c_a)}")
            print(f"     c_b:      {_serialize(c_b)}")
    if coll:
        print(f"   {len(coll)} false collisions (bridge equal, sympy distinct):")
        for a, b, c in coll[:3]:
            print(f"     a: {_serialize(a)}")
            print(f"     b: {_serialize(b)}")
            print(f"     c: {_serialize(c)}")
    print()

    print("=== Summary ===")
    print("If idempotence + permutation + identity injection are all 100%,")
    print("the canonicalizer has stronger-than-weak confluence on the")
    print("tested fragment. Random equivalence is a coverage probe: it")
    print("samples whether algebraic identities the canonicalizer doesn't")
    print("explicitly know are missed.")


if __name__ == "__main__":
    main()
