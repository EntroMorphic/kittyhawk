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
except ImportError:
    from parser import parse
    from numeric import encode, balt_add, balt_neg, balt_mul, balt_div, decode


D_DEFAULT = 128
TARGET_NONZERO_FRAC = 0.62


def _flatten(node):
    """Flatten nested (add ...) and (mul ...) into n-ary."""
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
        return (op, *flat)
    if op == "sub":
        # x - y  becomes  add(x, neg(y))  for canonical purposes; we keep
        # sub as a distinct node only if it doesn't reduce.
        return ("sub", args[0], args[1])
    if op == "div":
        return ("div", args[0], args[1])
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
    return all(_is_pure_numeric(a) for a in node[1:])


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
    if _is_pure_numeric(node):
        return _fold_numeric(node)
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
        return ("div", a, b)
    if op == "add":
        kids = [_simplify(a) for a in node[1:]]
        # drop zero terms
        kids = [k for k in kids if k != ("const", 0)]
        # fold pure-numeric children via balanced-ternary substrate
        kids = _partition_fold(kids, "add")
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
        return ("mul", *kids)
    return node


def canonicalize(node):
    """Run flatten → simplify → sort to fixed point."""
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
    if op == "neg":
        return f"(neg {_serialize(node[1])})"
    if op == "sub":
        return f"(sub {_serialize(node[1])} {_serialize(node[2])})"
    if op == "div":
        return f"(div {_serialize(node[1])} {_serialize(node[2])})"
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
