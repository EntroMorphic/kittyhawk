"""Approach A: canonicalize the AST then hash to a trit signature.

Pipeline (called from canonicalize()):
  1. _expand_products: a pre-pass that
       - converts sub(a, b) -> add(a, neg(b)) so subtraction unifies with
         the add path
       - distributes mul over add via cartesian product
       - pushes neg through add (neg(add(a, b)) -> add(neg(a), neg(b)))
       - folds pure-numeric integer subtrees into the const_product factor
  2. _rewrite_explog_identities: exp/log inverse, product/sum, and
     reciprocal rewrites. Honors POSITIVITY_MODE (permissive default,
     strict opt-in). See journal/claim2_unsoundness_designtension_fixes_*
     for why this is gated.
  3. Loop to fixed point: _flatten -> _simplify -> _expand_products
     -> _flatten -> _sort_canonical. _expand_products runs inside the
     loop (not just once) because _simplify's neg-pulling rule
     (mul(neg(X), Y) -> neg(mul(X, Y))) can re-introduce mul-of-add
     structures that need redistribution. Without this, canonicalize
     would not be idempotent.

_simplify applies the substrate-level rewrites:
  - drop e+0, e*1; absorb e*0; double-neg cancel; sub/div cancellation.
  - n-ary mul: pull negs and negative-constant factors out (with
    even-pair cancellation).
  - n-ary add: _combine_like_terms — group by monomial shape, sum
    coefficients. Same-shape terms with coefficients summing to 0
    drop out (subsumes the older explicit x+neg(x) cancel loop).
  - Pure-numeric subtrees fold through the substrate routings
    (balanced-ternary integer ops, or fixed-point exp/log via Taylor).
  - fp_const demotes to ('const', n) when the decoded value is within
    INTEGER_DEMOTE_TOL of an integer (closes the fp-vs-int design
    tension; see tolerance_sensitivity.py for the sweep).

Architectural notes:
  - n-copy expansion (the old "2*x -> x+x" path) was retired because of
    asymmetric behavior at the N_MAX boundary. Today's _expand_products
    keeps every coefficient as a mul(C:n, monomial) factor; combine-
    like-terms then coalesces same-shape terms uniformly. See
    journal/claim2_100of100_remediation_2026-05-13.md for the trace.
  - Routing-derived signatures (routing.py, approach B) collide
    distinct values under saturating add (e.g., x*x+y*y vs (x+y)^2).
    For consumer-grade equivalence detection use this module
    (approach A); the SHA over the canonical AST is faithful. See
    memory entry feedback_routing_vs_canonical_hash_signature.md.

Hashing:
  Canonical AST is serialized to a canonical string and SHA-256 is
  used to seed a deterministic PRNG that fills a length-D trit vector
  with sparsity ~62% nonzero (matching K-sig profile).
"""
from __future__ import annotations

import hashlib
import numpy as np
from typing import Any

try:
    from .parser import parse
    from .numeric import encode, balt_add, balt_neg, balt_mul, balt_div, decode
    from .fixed_point import fp_encode, fp_decode, exp_taylor, log_taylor, SCALE_DEFAULT
except ImportError:
    from parser import parse
    from numeric import encode, balt_add, balt_neg, balt_mul, balt_div, decode
    from fixed_point import fp_encode, fp_decode, exp_taylor, log_taylor, SCALE_DEFAULT


D_DEFAULT = 128
TARGET_NONZERO_FRAC = 0.62

# Integer-demotion tolerance for fp_const → ('const', n).
#
# Picked by the sweep in tolerance_sensitivity.py. The constraints:
#   - LOWER: must absorb Taylor convergence noise, which empirically
#     tops out at ~1.1e-16 (worst case: log(e) where atanh series
#     converges slowly at u ≈ 0.462). Tolerance must be > 1.1e-16.
#   - UPPER: must NOT absorb deliberately-small user values. A user
#     who types 1.0 + 1e-12 means it; their value should not collapse
#     to 1. Tolerance must be < 1e-12.
#
# The window (1.1e-16, 1e-12) is wide enough that any value in [1e-13,
# 1e-15] is safe. We pick 1e-12 as the canonical choice: it's right
# at the upper boundary (slightly aggressive on user values, generous
# on Taylor noise), and any value < 1e-12 is preserved.
#
# The earlier value 1e-9 was demonstrably too loose: it would demote
# user-typed 1e-10 to 0. That was a defensible choice for research
# code but is tightened here for principled behavior.
INTEGER_DEMOTE_TOL = 1e-12


def _flatten(node):
    """Flatten nested (add ...) and (mul ...) into n-ary."""
    if not isinstance(node, tuple):
        return node
    op = node[0]
    if op == "fp_const":
        return node  # leaf — never flatten its payload
    args = [_flatten(a) for a in node[1:]]
    if op in ("add", "mul"):
        flat = []
        for a in args:
            if isinstance(a, tuple) and a[0] == op:
                flat.extend(a[1:])
            else:
                flat.append(a)
        return (op, *flat)
    if op == "sub":
        # x - y  becomes  add(x, neg(y))  for canonical purposes; we keep
        # sub as a distinct node only if it doesn't reduce.
        return ("sub", args[0], args[1])
    if op == "div":
        return ("div", args[0], args[1])
    if op in ("exp", "log"):
        return (op, args[0])
    if op == "neg":
        # neg(neg(e)) -> e
        a0 = args[0]
        if isinstance(a0, tuple) and a0[0] == "neg":
            return a0[1]
        return ("neg", a0)
    return (op, *args)


def _sort_canonical(node):
    """Sort children of commutative ops by canonical-form hash."""
    if not isinstance(node, tuple):
        return node
    op = node[0]
    if op in ("add", "mul"):
        kids = [_sort_canonical(a) for a in node[1:]]
        kids = sorted(kids, key=lambda x: _serialize(x))
        return (op, *kids)
    if op == "sub":
        return ("sub", _sort_canonical(node[1]), _sort_canonical(node[2]))
    if op == "div":
        return ("div", _sort_canonical(node[1]), _sort_canonical(node[2]))
    if op in ("exp", "log"):
        return (op, _sort_canonical(node[1]))
    if op == "neg":
        return ("neg", _sort_canonical(node[1]))
    return node


def _is_pure_numeric(node) -> bool:
    """A subtree is pure-numeric if it has no variable references."""
    if not isinstance(node, tuple):
        return False
    op = node[0]
    if op == "var":
        return False
    if op == "const":
        return True
    if op == "fp_const":
        return True
    return all(_is_pure_numeric(a) for a in node[1:])


def _subtree_needs_fp(node) -> bool:
    """Does this subtree contain exp/log/fp_const anywhere? If so the
    pure-numeric fold must use fixed-point arithmetic, not integer."""
    if not isinstance(node, tuple):
        return False
    op = node[0]
    if op in ("exp", "log", "fp_const"):
        return True
    return any(_subtree_needs_fp(a) for a in node[1:])


def _fold_to_fp(node, scale: int = SCALE_DEFAULT):
    """Evaluate a pure-numeric subtree at fixed-point scale, returning
    a FixedPoint. Used for any pure-numeric subtree that contains
    exp/log (or other inherently-fractional ops)."""
    from fixed_point import (FixedPoint, fp_add, fp_sub, fp_neg,
                              fp_mul, fp_div, fp_from_int)
    from numeric import D_DEFAULT
    if not isinstance(node, tuple):
        raise ValueError(f"unexpected non-tuple {node!r}")
    op = node[0]
    if op == "const":
        return fp_from_int(node[1], scale, D_DEFAULT)
    if op == "fp_const":
        # Stored as ('fp_const', trits_tuple, scale)
        import numpy as np
        trits = np.array(node[1], dtype=np.int8)
        if node[2] != scale:
            raise ValueError(f"scale mismatch: {node[2]} vs {scale}")
        return FixedPoint(trits, scale)
    if op == "neg":
        return fp_neg(_fold_to_fp(node[1], scale))
    if op == "sub":
        return fp_sub(_fold_to_fp(node[1], scale), _fold_to_fp(node[2], scale))
    if op == "div":
        return fp_div(_fold_to_fp(node[1], scale), _fold_to_fp(node[2], scale))
    if op == "add":
        acc = _fold_to_fp(node[1], scale)
        for child in node[2:]:
            acc = fp_add(acc, _fold_to_fp(child, scale))
        return acc
    if op == "mul":
        acc = _fold_to_fp(node[1], scale)
        for child in node[2:]:
            acc = fp_mul(acc, _fold_to_fp(child, scale))
        return acc
    if op == "exp":
        return exp_taylor(_fold_to_fp(node[1], scale))
    if op == "log":
        return log_taylor(_fold_to_fp(node[1], scale))
    raise ValueError(f"_fold_to_fp: unknown op {op!r}")


def _fold_numeric(node):
    """Evaluate a pure-numeric subtree using balanced-ternary substrate
    routings. Returns ('const', value). Used when the entire subtree is
    constants — exercises balt_add/balt_neg/balt_mul as the actual
    computation, then decodes back to the integer constant for the
    canonical AST."""
    if not isinstance(node, tuple):
        return node
    op = node[0]
    if op == "const":
        return node
    if op == "neg":
        inner = _fold_numeric(node[1])
        sig = balt_neg(encode(inner[1]))
        return ("const", decode(sig))
    if op == "sub":
        a = _fold_numeric(node[1])
        b = _fold_numeric(node[2])
        sig = balt_add(encode(a[1]), balt_neg(encode(b[1])))
        return ("const", decode(sig))
    if op == "div":
        a = _fold_numeric(node[1])
        b = _fold_numeric(node[2])
        q_sig, _ = balt_div(encode(a[1]), encode(b[1]))
        return ("const", decode(q_sig))
    if op == "add":
        kids = [_fold_numeric(a) for a in node[1:]]
        acc = encode(kids[0][1])
        for k in kids[1:]:
            acc = balt_add(acc, encode(k[1]))
        return ("const", decode(acc))
    if op == "mul":
        kids = [_fold_numeric(a) for a in node[1:]]
        acc = encode(kids[0][1])
        for k in kids[1:]:
            acc = balt_mul(acc, encode(k[1]))
        return ("const", decode(acc))
    raise ValueError(f"_fold_numeric: unknown op {op!r}")


def _partition_fold(kids, op):
    """For an n-ary add/mul, fold the pure-numeric children into a
    single constant using balanced-ternary routings, leave the rest
    untouched. Returns the (possibly shorter) kids list."""
    numeric_vals = []
    mixed = []
    for k in kids:
        if _is_pure_numeric(k):
            n = _fold_numeric(k)
            # n is now ('const', value); extract value
            numeric_vals.append(n[1])
        else:
            mixed.append(k)
    if not numeric_vals:
        return mixed
    if op == "add":
        acc = encode(numeric_vals[0])
        for v in numeric_vals[1:]:
            acc = balt_add(acc, encode(v))
        folded = decode(acc)
    else:  # mul
        acc = encode(numeric_vals[0])
        for v in numeric_vals[1:]:
            acc = balt_mul(acc, encode(v))
        folded = decode(acc)
    # If folded numeric is the identity for this op, drop it.
    if op == "add" and folded == 0:
        return mixed
    if op == "mul" and folded == 1:
        return mixed
    if op == "mul" and folded == 0:
        return [("const", 0)]
    if not mixed:
        return [("const", folded)]
    return mixed + [("const", folded)]


def _term_shape_and_coef(node):
    """Extract (shape, integer_coefficient) from an add-summand.

    A "shape" is the monomial AST stripped of its integer coefficient;
    the coefficient is the product of all integer-const factors. Two
    terms with the same shape can be combined by summing coefficients.

      var x                  -> (var x, 1)
      const(n)               -> (const(1), n)         # n*1
      neg(t)                 -> (shape, -coef)        # propagate
      mul(C:n, ..., other)   -> (mul(...others...), n)
      mul(other, other)      -> (mul(...), 1)
      other op               -> (node, 1)             # treat opaquely
    """
    if not isinstance(node, tuple):
        return (node, 1)
    op = node[0]
    if op == "const":
        return (("const", 1), node[1])
    if op == "neg":
        s, c = _term_shape_and_coef(node[1])
        return (s, -c)
    if op == "mul":
        coef = 1
        non_const = []
        for k in node[1:]:
            if isinstance(k, tuple) and k[0] == "const":
                coef *= k[1]
            else:
                non_const.append(k)
        if not non_const:
            return (("const", 1), coef)
        if len(non_const) == 1:
            return (non_const[0], coef)
        return (("mul", *non_const), coef)
    return (node, 1)


def _combine_like_terms(kids):
    """Group add-children by monomial shape and sum their coefficients.

    Returns a new list of kids with same-shape terms merged. Pure-
    numeric collapses to a single const term; coefficient ±1 yields
    bare shape or neg(shape); other coefficients yield mul(C:n, shape).
    """
    if len(kids) < 2:
        return kids
    groups: dict[str, tuple] = {}
    order: list[str] = []
    for k in kids:
        shape, coef = _term_shape_and_coef(k)
        key = _serialize(shape)
        if key in groups:
            s, c = groups[key]
            groups[key] = (s, c + coef)
        else:
            groups[key] = (shape, coef)
            order.append(key)
    out = []
    for key in order:
        shape, coef = groups[key]
        if coef == 0:
            continue
        if shape == ("const", 1):
            # Pure constant term
            out.append(("const", coef))
            continue
        if coef == 1:
            out.append(shape)
        elif coef == -1:
            out.append(("neg", shape))
        else:
            # Embed coefficient as a const factor in a mul. _flatten /
            # _simplify in subsequent passes will sort and partition.
            if isinstance(shape, tuple) and shape[0] == "mul":
                out.append(("mul", ("const", coef), *shape[1:]))
            else:
                out.append(("mul", ("const", coef), shape))
    return out


def _simplify(node):
    """Apply identity / absorbing element rules."""
    if not isinstance(node, tuple):
        return node
    op = node[0]
    # Pure-numeric subtree → fold through balanced-ternary substrate.
    # If it contains exp/log, fold at fixed-point; else integer.
    if _is_pure_numeric(node):
        if _subtree_needs_fp(node):
            fp = _fold_to_fp(node, SCALE_DEFAULT)
            # If the folded value rounds to an integer within tolerance,
            # demote to an integer const so it aligns with integer-
            # literal signatures (closing the fp-vs-integer design
            # tension). See INTEGER_DEMOTE_TOL above for the rationale.
            v = fp_decode(fp)
            r = round(v)
            if abs(v - r) < INTEGER_DEMOTE_TOL:
                return ("const", int(r))
            return ("fp_const", tuple(int(t) for t in fp.trits), fp.scale)
        return _fold_numeric(node)
    # Recurse into exp/log when arg is mixed: simplify the argument.
    if op == "exp":
        return ("exp", _simplify(node[1]))
    if op == "log":
        return ("log", _simplify(node[1]))
    if op == "neg":
        a = _simplify(node[1])
        if isinstance(a, tuple) and a[0] == "neg":
            return a[1]  # double negation
        if a == ("const", 0):
            return ("const", 0)
        return ("neg", a)
    if op == "sub":
        a = _simplify(node[1])
        b = _simplify(node[2])
        if a == b:
            return ("const", 0)
        # x - 0 -> x
        if b == ("const", 0):
            return a
        # 0 - x -> neg(x)
        if a == ("const", 0):
            return _simplify(("neg", b))
        # (e + b) - b  →  e   (additive cancellation in mixed expr)
        if isinstance(a, tuple) and a[0] == "add":
            for i, k in enumerate(a[1:], start=1):
                if k == b:
                    rest = [c for j, c in enumerate(a[1:], start=1) if j != i]
                    if not rest:
                        return ("const", 0)
                    if len(rest) == 1:
                        return rest[0]
                    return _simplify(("add", *rest))
        return ("sub", a, b)
    if op == "div":
        a = _simplify(node[1])
        b = _simplify(node[2])
        # x / 1 -> x
        if b == ("const", 1):
            return a
        # 0 / x -> 0
        if a == ("const", 0):
            return ("const", 0)
        # x / x -> 1  (when x is not zero — we can't statically tell;
        # assume so for canonicalization purposes)
        if a == b:
            return ("const", 1)
        # (e * b) / b  →  e   (multiplicative cancellation)
        if isinstance(a, tuple) and a[0] == "mul":
            for i, k in enumerate(a[1:], start=1):
                if k == b:
                    rest = [c for j, c in enumerate(a[1:], start=1) if j != i]
                    if not rest:
                        return ("const", 1)
                    if len(rest) == 1:
                        return rest[0]
                    return _simplify(("mul", *rest))
        return ("div", a, b)
    if op == "add":
        kids = [_simplify(a) for a in node[1:]]
        # drop zero terms
        kids = [k for k in kids if k != ("const", 0)]
        # fold pure-numeric children via balanced-ternary substrate
        kids = _partition_fold(kids, "add")
        # log identity: add(log(a), log(b), ...) -> log(mul(a, b, ...))
        if kids and all(isinstance(k, tuple) and k[0] == "log" for k in kids):
            args = [k[1] for k in kids]
            return _simplify(("log", ("mul", *args)))
        # Combine like terms by monomial shape (subsumes the older
        # explicit x + neg(x) cancel loop: shape x with coefficients
        # +1 and -1 sum to 0 and drop out).
        kids = _combine_like_terms(kids)
        if not kids:
            return ("const", 0)
        if len(kids) == 1:
            return kids[0]
        return ("add", *kids)
    if op == "mul":
        kids = [_simplify(a) for a in node[1:]]
        # zero absorbs
        if any(k == ("const", 0) for k in kids):
            return ("const", 0)
        # drop ones
        kids = [k for k in kids if k != ("const", 1)]
        if not kids:
            return ("const", 1)
        # fold pure-numeric children
        kids = _partition_fold(kids, "mul")
        if any(k == ("const", 0) for k in kids):
            return ("const", 0)
        if not kids:
            return ("const", 1)
        # Pull negations and negative-integer constants out of mul.
        # mul(neg(a), b) -> neg(mul(a, b));  mul(C:-n, x) -> neg(mul(C:n, x)).
        # Even count of negs cancels (e.g. mul(neg(x), neg(y)) -> mul(x, y)).
        # This makes (x-y)*(x-y) equivalent to expand((x-y)**2) =
        # x*x - 2*x*y + y*y at the canonical-AST level.
        neg_count = 0
        stripped = []
        for k in kids:
            if isinstance(k, tuple) and k[0] == "neg":
                neg_count += 1
                stripped.append(k[1])
            elif isinstance(k, tuple) and k[0] == "const" and k[1] < 0:
                neg_count += 1
                if k[1] != -1:
                    stripped.append(("const", -k[1]))
                # else: -1 is absorbed entirely into the neg_count.
            else:
                stripped.append(k)
        # Drop +1 consts that may have been left behind.
        stripped = [k for k in stripped if k != ("const", 1)]
        kids = stripped
        # If everything got absorbed into neg_count, the value is +/-1.
        if not kids:
            return ("const", -1) if neg_count % 2 == 1 else ("const", 1)
        if len(kids) == 1:
            if neg_count % 2 == 1:
                return _simplify(("neg", kids[0]))
            return kids[0]
        # exp identity: mul(exp(a), exp(b), ...) -> exp(add(a, b, ...))
        if all(isinstance(k, tuple) and k[0] == "exp" for k in kids):
            args = [k[1] for k in kids]
            result = _simplify(("exp", ("add", *args)))
            if neg_count % 2 == 1:
                return _simplify(("neg", result))
            return result
        result = ("mul", *kids)
        if neg_count % 2 == 1:
            return _simplify(("neg", result))
        return result
    return node


# --- Positivity contract -----------------------------------------------------
#
# The rewrite exp(log(e)) → e is mathematically valid only when e is
# positive (log is undefined for e ≤ 0). _is_definitely_positive is a
# conservative static predicate: it returns True only when the subtree
# is provably positive.
#
# The treatment of `var` is a CONTRACT, not a mathematical fact:
#
#   - Default mode ("permissive"): a bare variable is assumed positive.
#     This matches how symbolic algebra systems handle ambiguous
#     identities (assume the principal-branch interpretation). It
#     means exp(log(x)) canonicalizes to x even when x could be ≤ 0.
#     Convenient for most algebraic manipulation; unsafe if downstream
#     code requires log_taylor to raise on x ≤ 0.
#
#   - Strict mode: a bare variable is NOT assumed positive. The
#     rewrite exp(log(x)) → x only fires when x is structurally
#     provably positive (e.g., x = exp(something), x = positive
#     constant). For arbitrary variables, the rewrite is suppressed,
#     and runtime log_taylor will raise on non-positive values.
#
# Toggle via POSITIVITY_MODE module variable or canonicalize(...,
# strict_positivity=True) parameter.
POSITIVITY_PERMISSIVE = "permissive"  # default: var assumed positive
POSITIVITY_STRICT = "strict"          # var NOT assumed positive
POSITIVITY_MODE = POSITIVITY_PERMISSIVE


def _is_definitely_positive(node, mode: str | None = None) -> bool:
    """Static check: can we prove this subtree always evaluates to
    a positive real? Returns False if we can't prove it (conservative).

    Cases we recognize (mode-independent):
      - integer const n with n > 0.
      - fp_const whose decoded value > 0.
      - exp(_) of anything (exp is always positive in reals).
      - mul/add of provably-positive operands (recursive).

    Mode-dependent:
      - variable: True in permissive mode (default), False in strict
        mode. See contract comments above.

    Everything else: False (conservative; let log_taylor raise at
    runtime if the value turns out non-positive)."""
    if mode is None:
        mode = POSITIVITY_MODE
    if not isinstance(node, tuple):
        return False
    op = node[0]
    if op == "const":
        return node[1] > 0
    if op == "fp_const":
        # decode without importing fp_decode to avoid cycle: trits at scale.
        v = 0
        p = 1
        for c in node[1]:
            v += int(c) * p
            p *= 3
        return v > 0
    if op == "exp":
        return True
    if op == "var":
        return mode == POSITIVITY_PERMISSIVE
    if op == "mul":
        return all(_is_definitely_positive(a, mode) for a in node[1:])
    if op == "add":
        return all(_is_definitely_positive(a, mode) for a in node[1:])
    return False


def _expand_products(node):
    """Distribute products of sums: mul(add(a, b), add(c, d), ...) ->
    sum of monomials by cartesian product. Also:
      - Converts sub(a, b) -> add(a, neg(b)) so subtraction participates
        in distribution (and downstream cancellation rules see it).
      - Folds pure-numeric integer subtrees (e.g., neg(const(2))) into
        the const_product accumulator.
      - Expands n * expr for small positive integer n into repeated
        addition (so 2*x*y matches x*y + x*y after canonical sort).
      - Recurses on each product after distribution so nested
        mul(const, var) gets expanded too (e.g., mul(2, x) becomes
        add(x, x) inside a larger distribution).
    Bounded by a size limit so we don't blow up large expressions."""
    import itertools
    SIZE_LIMIT = 200   # cap on output term count from a single expansion
    N_MAX = 20         # cap on n for "n * expr -> n copies" rewrite

    if not isinstance(node, tuple):
        return node
    op = node[0]
    if op == "fp_const":
        return node
    args = [_expand_products(a) for a in node[1:]]

    # Convert sub(a, b) -> add(a, neg(b)) so subtraction can be
    # distributed and cancelled uniformly through the add path.
    if op == "sub":
        return ("add", args[0], ("neg", args[1]))

    # Push neg through add: neg(add(a, b, ...)) -> add(neg(a), neg(b), ...).
    # This is the additive analogue of distributing mul over add, and
    # makes -1*(x+y) and -(x+y) canonicalize to the same form.
    if op == "neg":
        inner = args[0]
        if isinstance(inner, tuple) and inner[0] == "add":
            negated = [_expand_products(("neg", a)) for a in inner[1:]]
            return ("add", *negated)
        # neg(neg(e)) -> e  (handled here for completeness; _flatten/_simplify
        # also handle it, but folding early helps _expand_products see the
        # underlying structure).
        if isinstance(inner, tuple) and inner[0] == "neg":
            return inner[1]
        return ("neg", inner)

    if op == "mul":
        # Separate integer-valued factors (const or pure-numeric integer
        # subtree like neg(const(n))) from genuinely non-constant
        # factors.
        const_product = 1
        non_const = []
        for a in args:
            if isinstance(a, tuple) and a[0] == "const":
                const_product *= a[1]
            elif (isinstance(a, tuple) and _is_pure_numeric(a)
                  and not _subtree_needs_fp(a)):
                folded = _fold_numeric(a)
                const_product *= folded[1]
            else:
                non_const.append(a)

        # If product is zero, result is zero.
        if const_product == 0:
            return ("const", 0)

        # If no non-constant factors, the result is a constant.
        if not non_const:
            return ("const", const_product)

        # Distribute over any sum factors among non_const.
        # Cartesian product of summands. We collect summands recursively
        # so nested adds (produced by left-associative parsing) all get
        # distributed in a single pass.
        def _collect_summands(f):
            if not (isinstance(f, tuple) and f[0] == "add"):
                return [f]
            out = []
            for elem in f[1:]:
                out.extend(_collect_summands(elem))
            return out

        factor_summands = [_collect_summands(f) for f in non_const]

        # Predicted term count after distribution
        term_count = 1
        for fs in factor_summands:
            term_count *= len(fs)
        if term_count > SIZE_LIMIT:
            # Too big; keep folded
            rebuild = list(non_const)
            if const_product != 1:
                rebuild.append(("const", const_product))
            if len(rebuild) == 1:
                return rebuild[0]
            return ("mul", *rebuild)

        products = []
        for combo in itertools.product(*factor_summands):
            if len(combo) == 1:
                prod = combo[0]
            else:
                prod = ("mul", *combo)
            # Recurse only when the combo contains an integer-valued
            # factor (const or pure-numeric integer subtree); that's
            # the only case where re-expansion can change the structure
            # (n*expr -> add of n copies). Unconditional recursion
            # would infinite-loop on mul(x, y) which returns itself.
            if any(
                (isinstance(c, tuple) and c[0] == "const") or
                (isinstance(c, tuple) and _is_pure_numeric(c)
                 and not _subtree_needs_fp(c))
                for c in combo
            ):
                prod = _expand_products(prod)
            products.append(prod)

        # Apply the const_product. We do NOT expand n*monomial into n
        # copies — that approach causes asymmetry around the N_MAX
        # boundary (3*7 expands to 21 copies via nested calls, but
        # mul(C:21, ...) doesn't, so 3*7*expr and 21*expr canonicalize
        # differently). Instead, every coefficient is preserved as a
        # mul(C:n, ...) factor; combine_like_terms in _simplify-add
        # then groups identical shapes and sums their coefficients,
        # giving a unique normal form regardless of how the user
        # wrote the const factor.
        if const_product == 1:
            if len(products) == 1:
                return products[0]
            return ("add", *products)
        if const_product == -1:
            if len(products) == 1:
                return ("neg", products[0])
            return ("add", *[("neg", t) for t in products])
        # General case: multiply the const into each distributed product.
        # Wrapping in mul(C:n, term) keeps the canonical form uniform;
        # combine_like_terms later coalesces identical shapes.
        if len(products) == 1:
            return ("mul", ("const", const_product), products[0])
        return ("add", *[("mul", ("const", const_product), t) for t in products])

    return (op, *args)


def _rewrite_explog_identities(node, mode: str | None = None):
    """Bottom-up rewrite of exp/log algebraic identities:
       mul(exp(a), exp(b), ...) -> exp(add(a, b, ...))
       add(log(a), log(b), ...) -> log(mul(a, b, ...))
       exp(log(e)) -> e   ONLY when e is provably positive (mode-gated).
       log(exp(e)) -> e   (always safe; exp > 0 always).
       1 / exp(a) -> exp(-a)
    Applied as a pre-pass so pure-numeric folding doesn't short-circuit
    before the rewrite fires."""
    if not isinstance(node, tuple):
        return node
    op = node[0]
    if op == "fp_const":
        return node
    args = [_rewrite_explog_identities(a, mode) for a in node[1:]]
    if op == "mul" and len(args) >= 2 and all(
            isinstance(a, tuple) and a[0] == "exp" for a in args):
        sum_args = [a[1] for a in args]
        return _rewrite_explog_identities(("exp", ("add", *sum_args)), mode)
    if op == "add" and len(args) >= 2 and all(
            isinstance(a, tuple) and a[0] == "log" for a in args):
        prod_args = [a[1] for a in args]
        return _rewrite_explog_identities(("log", ("mul", *prod_args)), mode)
    # exp(log(e)) -> e  only when log(e) is well-defined (e > 0).
    # For e ≤ 0 (or unprovable in strict mode), leave the expression
    # alone so log_taylor raises at runtime.
    if op == "exp" and isinstance(args[0], tuple) and args[0][0] == "log":
        inner = args[0][1]
        if _is_definitely_positive(inner, mode):
            return inner
        # else: keep the exp(log(e)) form; downstream log_taylor will
        # raise ValueError if e ≤ 0 at evaluation time.
    # log(exp(e)) -> e  always safe (exp is always positive in reals).
    if op == "log" and isinstance(args[0], tuple) and args[0][0] == "exp":
        return args[0][1]
    # 1 / exp(a) -> exp(-a)  always safe (exp > 0).
    if op == "div" and len(args) == 2 and args[0] == ("const", 1):
        b = args[1]
        if isinstance(b, tuple) and b[0] == "exp":
            return _rewrite_explog_identities(("exp", ("neg", b[1])), mode)
    return (op, *args)


def canonicalize(node, strict_positivity: bool = False):
    """Run flatten → simplify → expand → sort to fixed point.

    Pre-passes in order:
      1. _expand_products: distribute mul over add (and neg over add).
      2. _rewrite_explog_identities: exp/log inverse and product/sum
         rewrites. Applied AFTER expansion so the rewrites see the
         normalized monomial form. Honors strict_positivity: if True,
         exp(log(var)) → var is suppressed.

    _expand_products is also called inside the loop because
    _simplify's neg-pulling rule can re-introduce mul-of-add structures
    (mul(neg(X), Y) → neg(mul(X, Y)), where X might be an add). Without
    re-running expansion inside the loop, canonicalize would not be
    idempotent.
    """
    mode = POSITIVITY_STRICT if strict_positivity else POSITIVITY_PERMISSIVE
    node = _expand_products(node)
    node = _rewrite_explog_identities(node, mode)
    prev = None
    cur = node
    for _ in range(20):  # bounded iteration
        if prev == cur:
            break
        prev = cur
        cur = _flatten(cur)
        cur = _simplify(cur)
        cur = _expand_products(cur)  # re-distribute after simplify
        cur = _flatten(cur)  # simplify may unflatten via collapses
        cur = _sort_canonical(cur)
    return cur


def _serialize(node) -> str:
    """Deterministic string serialization of a canonical AST."""
    if not isinstance(node, tuple):
        return repr(node)
    op = node[0]
    if op == "var":
        return f"V:{node[1]}"
    if op == "const":
        return f"C:{node[1]}"
    if op == "fp_const":
        return f"FP:scale={node[2]}:trits={node[1]}"
    if op == "neg":
        return f"(neg {_serialize(node[1])})"
    if op == "sub":
        return f"(sub {_serialize(node[1])} {_serialize(node[2])})"
    if op == "div":
        return f"(div {_serialize(node[1])} {_serialize(node[2])})"
    if op == "exp":
        return f"(exp {_serialize(node[1])})"
    if op == "log":
        return f"(log {_serialize(node[1])})"
    if op in ("add", "mul"):
        body = " ".join(_serialize(a) for a in node[1:])
        return f"({op} {body})"
    raise ValueError(f"unknown op {op!r}")


def signature_from_canonical(ast, d: int = D_DEFAULT,
                              target_nonzero: float = TARGET_NONZERO_FRAC,
                              strict_positivity: bool = False) -> np.ndarray:
    """SHA-256 the serialized canonical form, expand to D trits.

    Sparsity targeting: we pick the N most-extreme positions (by their
    underlying SHA-stream value mod 256) to be nonzero, where
    N = round(d * target_nonzero). The sign is determined by another
    byte of the hash stream.
    """
    canon = canonicalize(ast, strict_positivity=strict_positivity)
    ser = _serialize(canon)
    # Produce enough hash bytes to fill d positions × 2 bytes each
    stream = b""
    seed = ser.encode("utf-8")
    counter = 0
    while len(stream) < d * 4:
        stream += hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        counter += 1
    # Magnitudes for sparsity selection (first d bytes)
    mags = np.frombuffer(stream[:d], dtype=np.uint8).astype(np.int32)
    signs = np.frombuffer(stream[d:2*d], dtype=np.uint8).astype(np.int32)

    sig = np.zeros(d, dtype=np.int8)
    n_nonzero = int(round(d * target_nonzero))
    # Pick the n_nonzero positions with largest magnitude
    idx = np.argsort(-mags)[:n_nonzero]
    for i in idx:
        sig[i] = +1 if signs[i] >= 128 else -1
    return sig


def signature_from_expr(expr_str: str, d: int = D_DEFAULT,
                         strict_positivity: bool = False) -> np.ndarray:
    return signature_from_canonical(parse(expr_str), d=d,
                                     strict_positivity=strict_positivity)


if __name__ == "__main__":
    tests = [
        ("x", "x"),
        ("x + y", "y + x"),
        ("x * y", "y * x"),
        ("(x + y) + z", "x + (y + z)"),
        ("x + 0", "x"),
        ("x * 1", "x"),
        ("x * 0", "0"),
        ("x - x", "0"),
        ("x*(y+z)", "x*y + x*z"),       # distributivity — should differ
        ("(x+y)*(x-y)", "x*x - y*y"),    # algebraic identity — should differ
    ]
    print(f"{'A':<22} {'B':<22} {'L1(sigA, sigB)':>16}")
    print("-" * 64)
    for a, b in tests:
        sa = signature_from_expr(a)
        sb = signature_from_expr(b)
        l1 = int(np.abs(sa.astype(int) - sb.astype(int)).sum())
        mark = "  ✓ match" if l1 == 0 else f"  L1={l1}"
        print(f"{a:<22} {b:<22} {l1:>16}{mark}")
