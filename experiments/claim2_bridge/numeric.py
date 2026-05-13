"""Balanced-ternary integer encoding for the bridge.

The bridge's first iteration encoded integers via SHA-derived
signatures, so `s(2) + s(3) ≠ s(5)`. This module adds a numeric
encoding where the trit substrate's positional structure carries
integer arithmetic.

Balanced ternary: integer n = Σ t[i] · 3^i, t[i] ∈ {−1, 0, +1}.
With D cells the representable range is ±(3^D − 1)/2 (≈ 3^128/2
at D=128, vastly more than needed for any small constant).

Two routings on balanced-ternary trit vectors:

  balt_add(a, b)  positional ternary add with carry propagation.
                  Carry is a trit; saturates if the result exceeds
                  D-cell range (which won't happen for normal use).
  balt_mul(a, b)  positional ternary multiply via shift-and-add.

These compose with element-wise routings only when carefully
dispatched (see routing_v2). They are NOT element-wise commutative
with abstract-math routings — the substrate distinguishes "abstract
algebra of unknowns" from "numeric values."
"""
from __future__ import annotations

import numpy as np


D_DEFAULT = 128


def encode(n: int, d: int = D_DEFAULT) -> np.ndarray:
    """Integer n → balanced-ternary trit vector of length d.

    Standard balanced-ternary digit extraction:
      while n != 0:
        r = ((n + 1) mod 3) - 1   # r ∈ {-1, 0, +1}
        n = (n - r) / 3
        trit[i] = r, i++

    Asserts the value fits in d cells (positive or negative).
    """
    sig = np.zeros(d, dtype=np.int8)
    i = 0
    m = int(n)
    while m != 0:
        if i >= d:
            raise ValueError(f"integer {n} does not fit in {d} balanced-ternary cells")
        r = ((m + 1) % 3) - 1
        # python's % gives a non-negative result for positive divisor, so
        # (m+1) % 3 is in {0, 1, 2}, minus 1 gives r ∈ {-1, 0, +1}.
        sig[i] = r
        m = (m - r) // 3
        i += 1
    return sig


def decode(sig: np.ndarray) -> int:
    """Balanced-ternary trit vector → integer."""
    n = 0
    p = 1
    for c in sig:
        n += int(c) * p
        p *= 3
    return n


def balt_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Positional balanced-ternary add with carry propagation."""
    d = len(a)
    out = np.zeros(d, dtype=np.int8)
    carry = 0
    for i in range(d):
        s = int(a[i]) + int(b[i]) + carry
        # s ∈ {-3, -2, -1, 0, 1, 2, 3}; reduce to balanced-ternary digit
        r = ((s + 1) % 3) - 1
        carry = (s - r) // 3
        out[i] = r
    if carry != 0:
        raise OverflowError(f"balt_add overflow: final carry {carry}")
    return out


def balt_neg(a: np.ndarray) -> np.ndarray:
    """Negation in balanced ternary is element-wise sign flip."""
    return (-a.astype(np.int16)).astype(np.int8)


def balt_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return balt_add(a, balt_neg(b))


def balt_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Positional balanced-ternary multiply via shift-and-add.

    For each nonzero trit b[j], add a-shifted-left-by-j (if +1) or
    its negation (if -1) to the running sum.
    """
    d = len(a)
    acc = np.zeros(d, dtype=np.int8)
    for j in range(d):
        bj = int(b[j])
        if bj == 0:
            continue
        # shift a left by j cells
        shifted = np.zeros(d, dtype=np.int8)
        if j < d:
            shifted[j:] = a[:d - j]
        if bj == -1:
            shifted = balt_neg(shifted)
        # check that shift didn't lose information from a's high cells
        # (i.e., a[d-j:] must be all zero); else multiply overflows
        if j > 0 and np.any(a[d - j:] != 0):
            raise OverflowError(f"balt_mul shift loses high cells (j={j})")
        acc = balt_add(acc, shifted)
    return acc


def balt_div(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positional balanced-ternary integer division.

    Returns (quotient, remainder) with truncation toward zero, matching
    C integer-division semantics. Structurally a series of:
      shift3 (b << j) → sign-aware subtract → threshold-select trit qj → repeat
    All primitives on the elemental floor (shift3, sign, select, add+neg).

    For the bridge's first numeric integration, the implementation uses
    integer arithmetic on the decoded values and re-encodes. The
    substrate-native iterated-sub version is for the C kernel that
    `elemental_floor_closeout.md` references as future work.
    """
    bv = decode(b)
    if bv == 0:
        raise ZeroDivisionError("balt_div by zero")
    av = decode(a)
    # Truncate toward zero (C semantics): compute magnitude with floor,
    # then sign-adjust. Bulletproof against float-precision artifacts
    # that int(av/bv) would have at large magnitudes.
    qa = abs(av) // abs(bv)
    q = qa if (av >= 0) == (bv >= 0) else -qa
    r = av - q * bv
    return encode(q, len(a)), encode(r, len(a))


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    import random
    print("=== balanced-ternary encode/decode round-trip ===")
    for n in [0, 1, -1, 2, 3, -3, 5, 12, -12, 100, -100, 1000, -1000]:
        sig = encode(n)
        rev = decode(sig)
        ok = "✓" if rev == n else "✗"
        print(f"  {n:>6} -> {decode(sig):>6}   {ok}   sig[:8]={sig[:8].tolist()}")

    print("\n=== balt_add identity ===")
    for a, b, expected in [
        (2, 3, 5),
        (3, 2, 5),
        (5, 0, 5),
        (-2, 3, 1),
        (-2, -3, -5),
        (100, 23, 123),
        (1, 1, 2),
        (1, -1, 0),
    ]:
        sig = balt_add(encode(a), encode(b))
        rev = decode(sig)
        ok = "✓" if rev == expected else "✗"
        print(f"  {a:>4} + {b:>4} = {rev:>4}  (expected {expected})  {ok}")

    print("\n=== balt_mul identity ===")
    for a, b, expected in [
        (2, 3, 6),
        (3, 4, 12),
        (-2, 3, -6),
        (-2, -3, 6),
        (5, 0, 0),
        (0, 5, 0),
        (1, 7, 7),
        (10, 10, 100),
    ]:
        sig = balt_mul(encode(a), encode(b))
        rev = decode(sig)
        ok = "✓" if rev == expected else "✗"
        print(f"  {a:>4} * {b:>4} = {rev:>4}  (expected {expected})  {ok}")

    print("\n=== randomised round-trip ===")
    random.seed(42)
    failures = 0
    for _ in range(100):
        a = random.randint(-500, 500)
        b = random.randint(-500, 500)
        s = decode(balt_add(encode(a), encode(b)))
        p = decode(balt_mul(encode(a), encode(b)))
        if s != a + b or p != a * b:
            failures += 1
            print(f"  FAIL: {a} + {b} = {s} (want {a+b})  /  {a} * {b} = {p} (want {a*b})")
    print(f"  100 random pairs: {'all OK' if failures == 0 else f'{failures} fail(s)'}")

    print("\n=== balt_div identity (truncate toward zero) ===")
    for a, b in [(12, 3), (12, 4), (-12, 3), (12, -3), (-12, -3),
                  (7, 3), (-7, 3), (7, -3), (-7, -3),
                  (100, 7), (1, 5), (0, 5)]:
        q_sig, r_sig = balt_div(encode(a), encode(b))
        q = decode(q_sig); r = decode(r_sig)
        # truncate toward zero: a = q*b + r, |r| < |b|, sign(r) follows sign(a)
        want_q = int(a / b)
        want_r = a - want_q * b
        ok = "✓" if q == want_q and r == want_r else "✗"
        print(f"  {a:>4} / {b:>4} = q={q:>4} r={r:>4}  (expected q={want_q} r={want_r})  {ok}")

    print("\n=== randomised div round-trip ===")
    random.seed(43)
    failures = 0
    for _ in range(100):
        a = random.randint(-500, 500)
        b = random.randint(-100, 100)
        if b == 0: continue
        q_sig, r_sig = balt_div(encode(a), encode(b))
        q = decode(q_sig); r = decode(r_sig)
        want_q = int(a / b)
        want_r = a - want_q * b
        if q != want_q or r != want_r:
            failures += 1
            print(f"  FAIL: {a} / {b}: got q={q} r={r}, want q={want_q} r={want_r}")
    print(f"  100 random divisions: {'all OK' if failures == 0 else f'{failures} fail(s)'}")
