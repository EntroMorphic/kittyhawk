"""Consumer demo: an expression-equivalence cache that uses the claim 2
bridge as the cache key.

Demonstrates the bridge's substantive value: algebraically-equivalent
expressions (e.g., `(x+y)*(x-y)` and `x*x - y*y`) hit the SAME cache
entry, so an expensive computation runs once for the entire equivalence
class.

This is a stand-in for any downstream consumer that needs to recognize
expression equivalence — a CAS-style memoizer, a symbolic differentiation
cache, an identity database lookup, etc.

Verifies four properties:
  1. Equivalent expressions hit the same cache entry.
  2. Distinct expressions stay separate.
  3. The cache is faithful: cached values are correct (compute_fn
     produces identical results for any expression in the class).
  4. Negative path: cache speeds up the second call (measured).

This is a USE of the bridge, not a TEST of it. The bridge itself was
verified by sibling scripts in this directory:
  - sympy_battery.py       (32/32 adversarial via SymPy)
  - confluence.py          (5000/5000 across 4 axes)
  - measure.py             (17/17 equivalence classes, 0/276 collisions)
  - redteam.py             (16/16 integer red-team)
  - redteam_explog.py      (22/22 exp/log red-team)
  - positivity_contract.py (6/6 strict vs permissive contract)
Run any of them via `python <name>.py`. Here we exercise the bridge
as if we were any other consumer.

Closed concern #9 of the 100/100 remediation arc; final result 4/4
properties with 15× cache speedup. See
journal/claim2_100of100_remediation_pt2_2026-05-13.md. Property 2
of this script discovered the routing-vs-canonical-hash fidelity
gap; see memory entry
feedback_routing_vs_canonical_hash_signature.md.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import sympy as sp

THIS = os.path.dirname(__file__)
sys.path.insert(0, THIS)

# Approach A: SHA over canonical AST. Faithful equivalence detector —
# two expressions hash to the same key iff they have the same canonical
# AST (which is the bridge's algebraic notion of equality).
#
# Approach B (routing-derived) is NOT used here because saturating add
# at the trit level collides distinct values: e.g., `x*x + y*y` and
# `(x+y)^2 = x*x + 2xy + y*y` both saturate to the same trit pattern
# even though their canonical ASTs differ (the latter has the extra
# `2xy` term). For a faithful equivalence cache we need approach A.
from canonical import signature_from_expr


class ExpressionEquivalenceCache:
    """A cache keyed by claim-2-bridge canonical-AST signatures.

    Two expressions that the bridge considers algebraically equivalent
    canonicalize to the same AST → produce the same SHA → hit the same
    cache entry. Distinct ASTs → different signatures → separate
    entries. False collisions are bounded by SHA-256's collision
    resistance (i.e., not a practical concern).
    """

    def __init__(self):
        self._store: dict[bytes, object] = {}
        self._hits = 0
        self._misses = 0

    def _key(self, expr_str: str) -> bytes:
        sig = signature_from_expr(expr_str)
        return sig.tobytes()

    def get_or_compute(self, expr_str: str, compute_fn):
        """Return (value, was_cached). On miss, calls compute_fn(expr_str)
        and stores the result under the bridge signature."""
        key = self._key(expr_str)
        if key in self._store:
            self._hits += 1
            return self._store[key], True
        self._misses += 1
        value = compute_fn(expr_str)
        self._store[key] = value
        return value, False

    @property
    def stats(self) -> tuple[int, int, int]:
        """(hits, misses, unique_entries)"""
        return self._hits, self._misses, len(self._store)


def expensive_compute(expr_str: str) -> float:
    """A pretend-expensive computation: parse via SymPy, fully expand
    AND simplify (forces internal canonical-form work), then evaluate
    at a fixed point. The expand+simplify makes this an order of
    magnitude slower than the bridge's signature path, so the cache
    speedup is dramatic.

    Symbol-name match raises on unbound symbols (rather than silently
    leaving an un-evaluated expression that would fail at float())."""
    e = sp.sympify(expr_str.replace("^", "**"))
    # Force SymPy to do real work — expand and simplify each call.
    e = sp.expand(e)
    e = sp.simplify(e)
    subs_by_name = {"x": 1.7, "y": 0.3, "z": 2.1,
                    "a": 1.1, "b": 0.9, "c": 1.5, "d": 2.3}
    unbound = [s.name for s in e.free_symbols if s.name not in subs_by_name]
    if unbound:
        raise ValueError(f"unbound symbols in {expr_str!r}: {unbound}")
    sub_dict = {sym: subs_by_name[sym.name] for sym in e.free_symbols
                if sym.name in subs_by_name}
    return float(e.subs(sub_dict).evalf())


# ============================================================================
# Property tests
# ============================================================================

def test_property_1_equivalent_expressions_share_entry():
    """Two algebraically-equivalent expressions must hit the same
    cache entry on the second call."""
    cache = ExpressionEquivalenceCache()
    cases = [
        # (expr1, expr2, should_be_equivalent)
        ("(x + y) * (x - y)",   "x*x - y*y"),
        ("x + y",               "y + x"),
        ("(x + 1) * (x + 1)",   "x*x + 2*x + 1"),
        ("(a + b) * (c + d)",   "a*c + a*d + b*c + b*d"),
        ("x * y * z + x",       "x * (y*z + 1)"),
        ("x - x",               "0"),
        ("x + (y - y)",         "x"),
    ]
    failures = []
    for expr1, expr2 in cases:
        cache_2 = ExpressionEquivalenceCache()  # fresh per pair
        v1, cached1 = cache_2.get_or_compute(expr1, expensive_compute)
        v2, cached2 = cache_2.get_or_compute(expr2, expensive_compute)
        if cached2:
            # Hit on the second call — bridge recognized equivalence.
            if abs(v1 - v2) > 1e-9:
                failures.append(f"{expr1!r} vs {expr2!r}: cached but values "
                                f"differ {v1} vs {v2}")
        else:
            failures.append(f"{expr1!r} vs {expr2!r}: bridge did not recognize "
                            f"equivalence (computed both fresh)")
    return failures


def test_property_2_distinct_expressions_stay_separate():
    """Two genuinely distinct expressions must NOT collide in the
    cache (otherwise we'd return wrong values)."""
    cache = ExpressionEquivalenceCache()
    distinct_pairs = [
        ("x + y", "x - y"),
        ("x * y", "x + y"),
        ("(x + 1) * (x - 1)", "(x + 2) * (x - 2)"),
        ("x*x + y*y", "(x + y)*(x + y)"),
        ("a + b + c", "a + b + d"),  # different vars
    ]
    failures = []
    for expr1, expr2 in distinct_pairs:
        cache_2 = ExpressionEquivalenceCache()
        v1, cached1 = cache_2.get_or_compute(expr1, expensive_compute)
        v2, cached2 = cache_2.get_or_compute(expr2, expensive_compute)
        if cached2:
            failures.append(f"COLLISION: {expr1!r} and {expr2!r} hit same "
                            f"cache entry (v1={v1}, v2={v2})")
    return failures


def test_property_3_cached_values_are_correct():
    """When the cache returns a hit, the cached value must equal what
    we'd compute fresh (modulo equivalence)."""
    cache = ExpressionEquivalenceCache()
    # Compute many expressions, then re-issue them and verify cached
    # values match.
    exprs = [
        "x + y", "y + x", "x * y", "y * x",
        "(x + y) * (x - y)", "x*x - y*y",
        "x + 0", "x * 1",
        "(x + 1) * (x + 1)", "x*x + 2*x + 1",
    ]
    for e in exprs:
        cache.get_or_compute(e, expensive_compute)
    # Now hit each expression again and verify
    failures = []
    for e in exprs:
        cached_value, was_cached = cache.get_or_compute(e, expensive_compute)
        if not was_cached:
            failures.append(f"second call to {e!r} did not hit cache")
            continue
        true_value = expensive_compute(e)
        if abs(cached_value - true_value) > 1e-9:
            failures.append(f"{e!r}: cached {cached_value} != fresh {true_value}")
    return failures


def test_property_4_cache_speeds_up_second_call():
    """Time a heavy computation cached vs uncached. Caching should be
    measurably faster (the bridge's signature lookup is O(D) trit-vector
    work vs. a SymPy parse + expand + evaluate)."""
    expr = "(x + y + z + a + b) * (x - y + z - a + b) * (x + y - z + a - b)"
    cache = ExpressionEquivalenceCache()
    # First call — populates cache
    t0 = time.perf_counter()
    v1, was_cached_1 = cache.get_or_compute(expr, expensive_compute)
    t_first = time.perf_counter() - t0
    # Second call — same string, should hit
    t0 = time.perf_counter()
    v2, was_cached_2 = cache.get_or_compute(expr, expensive_compute)
    t_second = time.perf_counter() - t0
    failures = []
    if was_cached_1:
        failures.append("first call should have been a miss")
    if not was_cached_2:
        failures.append("second call should have been a hit")
    if abs(v1 - v2) > 1e-9:
        failures.append(f"values differ: {v1} vs {v2}")
    speedup = t_first / max(t_second, 1e-12)
    if speedup < 2:
        failures.append(f"cache speedup {speedup:.1f}x is suspiciously small "
                        f"(t_first={t_first*1e6:.1f}us, "
                        f"t_second={t_second*1e6:.1f}us)")
    return failures, t_first, t_second


# ============================================================================
# Demo workload (the "consumer" use case)
# ============================================================================

def demo_workload():
    """Simulate a downstream consumer: a stream of math expressions,
    many of which are algebraically equivalent. The cache should
    deliver high hit rate and consistent values."""
    cache = ExpressionEquivalenceCache()
    workload = [
        # Two equivalence classes interleaved
        "(x + y) * (x - y)",   # class A
        "x*x - y*y",           # class A (equivalent)
        "(x + 1) * (x - 1)",   # class B
        "x*x - 1",             # class B
        "y*y - x*x",           # class A negated → distinct
        "x*x - y*y",           # class A (cached)
        "(x - y) * (x + y)",   # class A (cached, commutativity)
        "(a + b) * (a - b)",   # class C
        "a*a - b*b",           # class C
        "x*x - y*y + 0",       # class A (identity)
    ]
    print("=== Demo workload: 10 expressions across ~3 equivalence classes ===\n")
    print(f"{'#':>3}  {'expr':<28}  {'value':>12}  cached?")
    for i, e in enumerate(workload):
        v, cached = cache.get_or_compute(e, expensive_compute)
        status = "HIT" if cached else "miss"
        print(f"{i+1:>3}  {e:<28}  {v:>12.4f}  {status}")
    h, m, u = cache.stats
    print(f"\n  cache: {h} hits, {m} misses, {u} unique entries")
    print(f"  hit rate: {100*h/(h+m):.1f}%")


def main():
    print("=== Consumer demo: bridge as expression-equivalence cache ===\n")

    fails_p1 = test_property_1_equivalent_expressions_share_entry()
    fails_p2 = test_property_2_distinct_expressions_stay_separate()
    fails_p3 = test_property_3_cached_values_are_correct()
    fails_p4_tuple = test_property_4_cache_speeds_up_second_call()
    fails_p4, t_first, t_second = fails_p4_tuple

    def report(name, fails):
        if fails:
            print(f"[FAIL] {name}")
            for f in fails:
                print(f"    {f}")
        else:
            print(f"[OK]   {name}")

    report("Property 1: equivalent exprs share cache entry", fails_p1)
    report("Property 2: distinct exprs stay separate", fails_p2)
    report("Property 3: cached values are correct", fails_p3)
    report("Property 4: cache speedup measurable", fails_p4)
    print(f"       (first call {t_first*1e6:.0f}us, "
          f"second call {t_second*1e6:.0f}us, "
          f"speedup {t_first/max(t_second, 1e-12):.0f}x)")
    print()
    demo_workload()

    total_fails = sum(len(f) for f in [fails_p1, fails_p2, fails_p3, fails_p4])
    if total_fails > 0:
        print(f"\nFAIL: {total_fails} consumer-property failures")
        return 1
    print("\nOK: consumer demo passes all 4 properties")
    return 0


if __name__ == "__main__":
    sys.exit(main())
