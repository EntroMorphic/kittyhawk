"""Approach A: canonicalize the AST then hash to a trit signature.

Canonicalization rules (semantics-preserving):
  - Flatten nested ('add', ...) and ('mul', ...) into n-ary.
  - Sort children of n-ary ('add', 'mul') by their canonical-form
    hash (commutativity, associativity).
  - Simplify identities and absorbing elements:
      e + 0    -> e
      e * 1    -> e
      e * 0    -> 0
      e - e    -> 0
      neg(neg(e)) -> e
      e + neg(e) -> 0
  - Recurse to fixed point.

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
            # tension). Absolute tolerance 1e-9: tight enough that
            # math.exp(30)=10686474581524.463 won't demote (distance
            # 0.463), loose enough to absorb Taylor convergence noise
            # (much smaller than 1e-9 at scale 3^-40).
            v = fp_decode(fp)
            r = round(v)
            if abs(v - r) < 1e-9:
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
        # cancel x + neg(x)
        out = []
        used = [False] * len(kids)
        for i, k in enumerate(kids):
            if used[i]:
                continue
            paired = False
            for j in range(i + 1, len(kids)):
                if used[j]:
                    continue
                if (isinstance(kids[j], tuple) and kids[j][0] == "neg"
                        and kids[j][1] == k):
                    used[i] = used[j] = True
                    paired = True
                    break
                if (isinstance(k, tuple) and k[0] == "neg"
                        and k[1] == kids[j]):
                    used[i] = used[j] = True
                    paired = True
                    break
            if not paired:
                out.append(k)
        if not out:
            return ("const", 0)
        if len(out) == 1:
            return out[0]
        return ("add", *out)
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
        if len(kids) == 1:
            return kids[0]
        # exp identity: mul(exp(a), exp(b), ...) -> exp(add(a, b, ...))
        if all(isinstance(k, tuple) and k[0] == "exp" for k in kids):
            args = [k[1] for k in kids]
            return _simplify(("exp", ("add", *args)))
        return ("mul", *kids)
    return node


def _is_definitely_positive(node) -> bool:
    """Static check: can we prove this subtree always evaluates to
    a positive real? Returns False if we can't prove it (conservative).

    Cases we recognize:
      - integer const n with n > 0.
      - fp_const whose decoded value > 0.
      - exp(_) of anything (exp is always positive in reals).
      - mul/add of provably-positive operands.
      - variable — pragmatically treat as positive (preserves the
        common algebraic-identity case for symbolic variables; the
        caller asserts positivity by writing log(var)).
    Everything else: False (be conservative; let log_taylor raise at
    runtime if the value turns out non-positive)."""
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
        return True  # pragmatic: caller asserts positivity by using log(var)
    if op == "mul":
        return all(_is_definitely_positive(a) for a in node[1:])
    if op == "add":
        return all(_is_definitely_positive(a) for a in node[1:])
    return False


def _rewrite_explog_identities(node):
    """Bottom-up rewrite of exp/log algebraic identities:
       mul(exp(a), exp(b), ...) -> exp(add(a, b, ...))
       add(log(a), log(b), ...) -> log(mul(a, b, ...))
       exp(log(e)) -> e   ONLY when e is provably positive.
       log(exp(e)) -> e   (always safe; exp > 0 always).
       1 / exp(a) -> exp(-a)
    Applied as a pre-pass so pure-numeric folding doesn't short-circuit
    before the rewrite fires."""
    if not isinstance(node, tuple):
        return node
    op = node[0]
    if op == "fp_const":
        return node
    args = [_rewrite_explog_identities(a) for a in node[1:]]
    if op == "mul" and len(args) >= 2 and all(
            isinstance(a, tuple) and a[0] == "exp" for a in args):
        sum_args = [a[1] for a in args]
        return _rewrite_explog_identities(("exp", ("add", *sum_args)))
    if op == "add" and len(args) >= 2 and all(
            isinstance(a, tuple) and a[0] == "log" for a in args):
        prod_args = [a[1] for a in args]
        return _rewrite_explog_identities(("log", ("mul", *prod_args)))
    # exp(log(e)) -> e  only when log(e) is well-defined (e > 0).
    # For e ≤ 0, leave the expression alone so log_taylor raises.
    if op == "exp" and isinstance(args[0], tuple) and args[0][0] == "log":
        inner = args[0][1]
        if _is_definitely_positive(inner):
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
            return _rewrite_explog_identities(("exp", ("neg", b[1])))
    return (op, *args)


def canonicalize(node):
    """Run flatten → simplify → sort to fixed point."""
    # Pre-pass: rewrite exp/log identities so pure-numeric folding
    # produces the same canonical form regardless of input shape.
    node = _rewrite_explog_identities(node)
    prev = None
    cur = node
    for _ in range(20):  # bounded iteration
        if prev == cur:
            break
        prev = cur
        cur = _flatten(cur)
        cur = _simplify(cur)
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
                              target_nonzero: float = TARGET_NONZERO_FRAC) -> np.ndarray:
    """SHA-256 the serialized canonical form, expand to D trits.

    Sparsity targeting: we pick the N most-extreme positions (by their
    underlying SHA-stream value mod 256) to be nonzero, where
    N = round(d * target_nonzero). The sign is determined by another
    byte of the hash stream.
    """
    canon = canonicalize(ast)
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


def signature_from_expr(expr_str: str, d: int = D_DEFAULT) -> np.ndarray:
    return signature_from_canonical(parse(expr_str), d=d)


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
