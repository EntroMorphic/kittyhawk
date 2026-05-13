"""Approach B: routing-derived signature.

Each variable and constant has a base signature. Each primitive
operation has a routing function on trit vectors. The signature of
an expression is the composed routing applied bottom-up.

Primitive routings on trits {-1, 0, +1}:

  route_add(a, b)  element-wise saturating ternary add.
                   commutative; identity = all-zero.
                   a + 0 = a;  +1 + +1 = +1 (saturated);
                   +1 + -1 = 0.
  route_sub(a, b)  element-wise saturating ternary sub.
                   route_sub(a, a) = 0 (identity inverse).
  route_mul(a, b)  element-wise ternary product.
                   commutative; identity = all-+1;
                   absorbing = all-zero. a * 1 = a; a * 0 = 0.
  route_neg(a)     element-wise negation of trits.

Base signatures:
  s(var(name))   deterministic SHA-derived ~62% nonzero trit vector.
  s(const(n))    deterministic mapping integer → trits.
                   special: 0 -> all-zero; 1 -> all-+1; -1 -> all--1.
                   other integers: SHA-derived.

These are intentionally minimal. The routing implements the path-
graph structure of trits literally: each cell is a path graph
{-1, 0, +1}, and the primitive routings are well-defined on the path.
"""
from __future__ import annotations

import hashlib
import numpy as np

try:
    from .parser import parse
    from .numeric import encode, balt_add, balt_neg, balt_mul, balt_div, decode
    from .canonical import canonicalize as _canonicalize_ast
except ImportError:
    from parser import parse
    from numeric import encode, balt_add, balt_neg, balt_mul, balt_div, decode
    from canonical import canonicalize as _canonicalize_ast


D_DEFAULT = 128
TARGET_NONZERO_FRAC = 0.62


def base_sig_var(name: str, d: int = D_DEFAULT,
                  target_nonzero: float = TARGET_NONZERO_FRAC) -> np.ndarray:
    """Deterministic trit signature for a variable name."""
    seed = ("var:" + name).encode("utf-8")
    stream = b""
    c = 0
    while len(stream) < d * 4:
        stream += hashlib.sha256(seed + c.to_bytes(4, "big")).digest()
        c += 1
    mags  = np.frombuffer(stream[:d], dtype=np.uint8).astype(np.int32)
    signs = np.frombuffer(stream[d:2*d], dtype=np.uint8).astype(np.int32)
    sig = np.zeros(d, dtype=np.int8)
    n_nonzero = int(round(d * target_nonzero))
    idx = np.argsort(-mags)[:n_nonzero]
    for i in idx:
        sig[i] = +1 if signs[i] >= 128 else -1
    return sig


def base_sig_const(n: int, d: int = D_DEFAULT) -> np.ndarray:
    """Integer constant → trit signature.

    Special integers are encoded with mathematical meaning:
      0 → all-zero  (so add(x, 0) = x, mul(x, 0) = 0).
      1 → all-+1   (so mul(x, 1) = x).
     -1 → all--1.
    Other integers: SHA-derived (no special algebraic identity).
    """
    if n == 0:
        return np.zeros(d, dtype=np.int8)
    if n == 1:
        return np.ones(d, dtype=np.int8)
    if n == -1:
        return -np.ones(d, dtype=np.int8)
    # General integer: deterministic ternary representation.
    seed = ("const:" + str(n)).encode("utf-8")
    stream = b""
    c = 0
    while len(stream) < d * 4:
        stream += hashlib.sha256(seed + c.to_bytes(4, "big")).digest()
        c += 1
    mags  = np.frombuffer(stream[:d], dtype=np.uint8).astype(np.int32)
    signs = np.frombuffer(stream[d:2*d], dtype=np.uint8).astype(np.int32)
    sig = np.zeros(d, dtype=np.int8)
    n_nonzero = int(round(d * TARGET_NONZERO_FRAC))
    idx = np.argsort(-mags)[:n_nonzero]
    for i in idx:
        sig[i] = +1 if signs[i] >= 128 else -1
    return sig


def route_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise saturating ternary add.
    +1 + +1 -> +1, +1 + 0 -> +1, +1 + -1 -> 0, etc."""
    s = a.astype(np.int16) + b.astype(np.int16)
    return np.clip(s, -1, 1).astype(np.int8)


def route_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    s = a.astype(np.int16) - b.astype(np.int16)
    return np.clip(s, -1, 1).astype(np.int8)


def route_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise ternary product. Already saturated since
    {-1,0,+1} is closed under multiplication."""
    return (a.astype(np.int16) * b.astype(np.int16)).astype(np.int8)


def route_neg(a: np.ndarray) -> np.ndarray:
    return (-a.astype(np.int16)).astype(np.int8)


def _flatten(node):
    """Flatten nested ('add', ...) and ('mul', ...) into n-ary form so
    that downstream sort-and-reduce gives a canonical commutative
    result regardless of how the parser nested the chain. Also folds
    pure-numeric children via balanced-ternary substrate routings."""
    if not isinstance(node, tuple):
        return node
    op = node[0]
    args = [_flatten(a) for a in node[1:]]
    if op in ("add", "mul"):
        flat = []
        for a in args:
            if isinstance(a, tuple) and a[0] == op:
                flat.extend(a[1:])
            else:
                flat.append(a)
        # Partition pure-numeric children, fold them via balt routings.
        numeric_vals = []
        mixed = []
        for k in flat:
            if _is_pure_numeric(k):
                numeric_vals.append(_fold_numeric(k))
            else:
                mixed.append(k)
        if numeric_vals:
            if op == "add":
                acc = encode(numeric_vals[0])
                for v in numeric_vals[1:]:
                    acc = balt_add(acc, encode(v))
                folded = decode(acc)
                if folded == 0 and mixed:
                    return (op, *mixed) if len(mixed) > 1 else mixed[0]
                if not mixed:
                    return ("const", folded)
                return (op, *mixed, ("const", folded))
            if op == "mul":
                acc = encode(numeric_vals[0])
                for v in numeric_vals[1:]:
                    acc = balt_mul(acc, encode(v))
                folded = decode(acc)
                if folded == 0:
                    return ("const", 0)
                if folded == 1 and mixed:
                    return (op, *mixed) if len(mixed) > 1 else mixed[0]
                if not mixed:
                    return ("const", folded)
                return (op, *mixed, ("const", folded))
        return (op, *flat)
    return (op, *args)


def _is_pure_numeric(node) -> bool:
    if not isinstance(node, tuple):
        return False
    op = node[0]
    if op == "var":
        return False
    if op == "const":
        return True
    return all(_is_pure_numeric(a) for a in node[1:])


def _fold_numeric(node) -> int:
    if not isinstance(node, tuple):
        raise ValueError(f"unexpected non-tuple {node!r}")
    op = node[0]
    if op == "const":
        return node[1]
    if op == "neg":
        return decode(balt_neg(encode(_fold_numeric(node[1]))))
    if op == "sub":
        return decode(balt_add(encode(_fold_numeric(node[1])),
                                balt_neg(encode(_fold_numeric(node[2])))))
    if op == "div":
        a_v = _fold_numeric(node[1])
        b_v = _fold_numeric(node[2])
        q_sig, _ = balt_div(encode(a_v), encode(b_v))
        return decode(q_sig)
    if op == "add":
        acc = encode(_fold_numeric(node[1]))
        for child in node[2:]:
            acc = balt_add(acc, encode(_fold_numeric(child)))
        return decode(acc)
    if op == "mul":
        acc = encode(_fold_numeric(node[1]))
        for child in node[2:]:
            acc = balt_mul(acc, encode(_fold_numeric(child)))
        return decode(acc)
    raise ValueError(f"_fold_numeric: unknown op {op!r}")


def signature(ast, d: int = D_DEFAULT) -> np.ndarray:
    """Bottom-up evaluation of the AST through the routing primitives.

    Pure-numeric subtrees are folded through balanced-ternary
    substrate routings (`numeric.balt_*`) and become single ('const', n)
    leaves. Mixed (variable-containing) subtrees use the element-wise
    abstract routings."""
    if not isinstance(ast, tuple):
        raise ValueError(f"expected tuple AST, got {ast!r}")
    # Algebraic simplification (identities like x/x=1, x*0=0, neg(neg(x))=x)
    # via canonical's _simplify. Substrate routings then derive the
    # signature from the simplified AST. This blends approach B's
    # substrate-routing purity with approach A's algebraic rewriter:
    # the bridge handles both classes of equivalence.
    ast = _canonicalize_ast(ast)
    if _is_pure_numeric(ast):
        ast = ("const", _fold_numeric(ast))
    ast = _flatten(ast)
    op = ast[0]
    if op == "var":
        return base_sig_var(ast[1], d=d)
    if op == "const":
        return base_sig_const(ast[1], d=d)
    if op == "neg":
        return route_neg(signature(ast[1], d=d))
    if op == "sub":
        return route_sub(signature(ast[1], d=d), signature(ast[2], d=d))
    if op == "div":
        # Mixed-variable division: no clean element-wise trit routing
        # preserves algebraic identities here (element-wise trit-div
        # collapses to mul-with-zero-fallback, losing structure). For
        # this iteration: fall back to a canonical-form hash so the
        # signature is deterministic and distinct from related forms.
        # Pure-numeric div is handled via _fold_numeric above.
        import hashlib
        from canonical import _serialize  # type: ignore
        try:
            from .canonical import _serialize  # type: ignore
        except ImportError:
            from canonical import _serialize  # type: ignore
        seed = ("div:" + _serialize(ast)).encode("utf-8")
        stream = b""
        c = 0
        while len(stream) < d * 4:
            stream += hashlib.sha256(seed + c.to_bytes(4, "big")).digest()
            c += 1
        mags  = np.frombuffer(stream[:d], dtype=np.uint8).astype(np.int32)
        signs = np.frombuffer(stream[d:2*d], dtype=np.uint8).astype(np.int32)
        sig = np.zeros(d, dtype=np.int8)
        n_nonzero = int(round(d * TARGET_NONZERO_FRAC))
        idx = np.argsort(-mags)[:n_nonzero]
        for i in idx:
            sig[i] = +1 if signs[i] >= 128 else -1
        return sig
    if op == "add":
        kids = [signature(a, d=d) for a in ast[1:]]
        # n-ary add: sum-then-saturate (associativity by construction).
        # Sequential pairwise saturating add would break deep
        # distributivity ((a+b)*(c+d) ≠ ac+ad+bc+bd) because of
        # intermediate saturations. Computing the full sum first and
        # saturating at the end makes the operation truly
        # commutative-associative on the substrate.
        if not kids:
            return np.zeros(d, dtype=np.int8)
        s = np.zeros(d, dtype=np.int32)
        for k in kids:
            s += k.astype(np.int32)
        return np.clip(s, -1, 1).astype(np.int8)
    if op == "mul":
        kids = [signature(a, d=d) for a in ast[1:]]
        order = sorted(range(len(kids)), key=lambda i: kids[i].tobytes())
        kids = [kids[i] for i in order]
        out = kids[0]
        for k in kids[1:]:
            out = route_mul(out, k)
        return out
    raise ValueError(f"unknown op {op!r}")


def signature_from_expr(expr_str: str, d: int = D_DEFAULT) -> np.ndarray:
    return signature(parse(expr_str), d=d)


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
        ("x*(y+z)", "x*y + x*z"),
        ("(x+y)*(x-y)", "x*x - y*y"),
    ]
    print(f"{'A':<22} {'B':<22} {'L1':>5}")
    print("-" * 50)
    for a, b in tests:
        sa = signature_from_expr(a)
        sb = signature_from_expr(b)
        l1 = int(np.abs(sa.astype(int) - sb.astype(int)).sum())
        mark = '  MATCH' if l1 == 0 else ''
        print(f"{a:<22} {b:<22} {l1:>5}{mark}")
