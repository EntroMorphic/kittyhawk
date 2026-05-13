"""Fixed-point trit arithmetic and Taylor exp/log.

A fixed-point value is (trits, scale) where the represented real
value is `decode(trits) * 3^(-scale)`. The trits are balanced-ternary
just like in numeric.py; the implicit radix-point shift is encoded
purely in the scale metadata.

Primitive ops:

  fp_encode(value, scale, d)   real -> FixedPoint(trits, scale).
  fp_decode(fp)                FixedPoint -> float (lossy round-trip
                                from 3^-scale precision).
  fp_add(a, b)                 same-scale add; reuses balt_add.
  fp_neg(a), fp_sub(a, b)
  fp_mul(a, b)                 balt_mul then shift3 right by scale
                                to preserve scale. Substrate-native:
                                shift3 is one of the elemental floor
                                primitives.
  fp_div(a, b)                 shift3 numerator left by scale, then
                                balt_div, return quotient.

Taylor series:

  exp_taylor(x_fp, n_terms)    Σ x^k / k!  via fp_mul + fp_div.
  log_taylor(x_fp, n_terms)    2·Σ u^(2k+1)/(2k+1) where u=(x-1)/(x+1).

Both terminate when the new term's magnitude falls below 3^-scale
(below representable precision) or when n_terms is reached.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from .numeric import (encode, decode, balt_add, balt_neg, balt_mul,
                          balt_div, D_DEFAULT)
except ImportError:
    from numeric import (encode, decode, balt_add, balt_neg, balt_mul,
                         balt_div, D_DEFAULT)


SCALE_DEFAULT = 40  # 3^-40 ≈ 8.6e-20, ~19 decimal digits of fractional precision


@dataclass
class FixedPoint:
    trits: np.ndarray   # shape (d,), int8
    scale: int          # k such that value = decode(trits) * 3^(-k)

    def __repr__(self) -> str:
        return f"FixedPoint({fp_decode(self):.6g} @ scale={self.scale})"


# ─────────────────────────────────────────────────────────────────────
# encode / decode
# ─────────────────────────────────────────────────────────────────────

def fp_encode(value: float, scale: int = SCALE_DEFAULT,
              d: int = D_DEFAULT) -> FixedPoint:
    """Convert a real value to FixedPoint(trits, scale).

    The encoded integer is round(value * 3^scale).
    """
    scaled = int(round(value * (3 ** scale)))
    return FixedPoint(encode(scaled, d), scale)


def fp_decode(fp: FixedPoint) -> float:
    """Convert FixedPoint to float."""
    return decode(fp.trits) / (3 ** fp.scale)


# ─────────────────────────────────────────────────────────────────────
# Substrate-level shift3 — multiply/divide trit vector by 3^k.
# Implementation here matches semantics of m4t_mtfp_shift3
# (saturating positive shift, base-3 round-to-nearest-even negative).
# For Python it's just integer encode/decode * 3^k.
# ─────────────────────────────────────────────────────────────────────

def shift3(trits: np.ndarray, k: int) -> np.ndarray:
    """Multiply the integer represented by `trits` by 3^k.

    For k > 0: shift cells up by k positions (low cells become 0).
    For k < 0: shift cells down (truncate toward zero).
    """
    d = len(trits)
    out = np.zeros(d, dtype=np.int8)
    if k >= 0:
        # shift up: trits[i] → out[i+k]
        n_keep = d - k
        if n_keep > 0:
            out[k:k + n_keep] = trits[:n_keep]
        # check for overflow
        if k > 0 and np.any(trits[d - k:] != 0):
            raise OverflowError(f"shift3 by {k}: high cells lost")
    else:
        # shift down: trits[i] → out[i+k] for i >= -k
        kk = -k
        if kk < d:
            out[:d - kk] = trits[kk:]
        # truncation toward zero: cells trits[0..kk-1] are dropped
    return out


# ─────────────────────────────────────────────────────────────────────
# fixed-point primitives
# ─────────────────────────────────────────────────────────────────────

def _require_same_scale(a: FixedPoint, b: FixedPoint) -> None:
    if a.scale != b.scale:
        raise ValueError(f"fp scale mismatch: {a.scale} vs {b.scale}")


def fp_add(a: FixedPoint, b: FixedPoint) -> FixedPoint:
    _require_same_scale(a, b)
    return FixedPoint(balt_add(a.trits, b.trits), a.scale)


def fp_neg(a: FixedPoint) -> FixedPoint:
    return FixedPoint(balt_neg(a.trits), a.scale)


def fp_sub(a: FixedPoint, b: FixedPoint) -> FixedPoint:
    _require_same_scale(a, b)
    return FixedPoint(balt_add(a.trits, balt_neg(b.trits)), a.scale)


def fp_mul(a: FixedPoint, b: FixedPoint) -> FixedPoint:
    """Multiply two fixed-point values, preserving common scale.

    Mathematically: (A * 3^-k_a) · (B * 3^-k_b) = (A·B) * 3^-(k_a+k_b).
    To keep result at the larger of the two input scales, shift3 the
    product trits down by min(k_a, k_b).
    """
    if a.scale != b.scale:
        raise NotImplementedError("fp_mul requires same scale")
    product_trits = balt_mul(a.trits, b.trits)
    # result is at scale 2·a.scale; shift down by a.scale to restore.
    out_trits = shift3(product_trits, -a.scale)
    return FixedPoint(out_trits, a.scale)


def fp_div(a: FixedPoint, b: FixedPoint) -> FixedPoint:
    """Divide two fixed-point values, preserving common scale.

    Mathematically: (A · 3^-k) / (B · 3^-k) = (A / B).
    To preserve the scale-k representation, shift A up by k before
    dividing, so the result trits encode (A · 3^k) / B = (A/B) · 3^k.
    """
    if a.scale != b.scale:
        raise NotImplementedError("fp_div requires same scale")
    a_shifted = shift3(a.trits, a.scale)
    q_trits, _ = balt_div(a_shifted, b.trits)
    return FixedPoint(q_trits, a.scale)


def fp_div_by_int(a: FixedPoint, n: int) -> FixedPoint:
    """Divide a fixed-point by an integer (no scale change)."""
    n_trits = encode(n, len(a.trits))
    q_trits, _ = balt_div(a.trits, n_trits)
    return FixedPoint(q_trits, a.scale)


def fp_from_int(n: int, scale: int = SCALE_DEFAULT,
                d: int = D_DEFAULT) -> FixedPoint:
    """Encode integer n at the given scale (i.e., trits = n · 3^scale)."""
    return FixedPoint(encode(n * (3 ** scale), d), scale)


# ─────────────────────────────────────────────────────────────────────
# Taylor exp
# ─────────────────────────────────────────────────────────────────────

def exp_taylor(x: FixedPoint, n_terms: int = 40) -> FixedPoint:
    """Compute exp(x) via Taylor series: 1 + x + x²/2! + x³/3! + ...

    Terminates when the next term's magnitude falls below 3^-scale
    or after n_terms iterations.
    """
    one = fp_from_int(1, x.scale, len(x.trits))
    acc = FixedPoint(one.trits.copy(), x.scale)
    # term k = x^k / k!. Start with k=1: term = x.
    term = FixedPoint(x.trits.copy(), x.scale)
    acc = fp_add(acc, term)
    fact_k = 1
    for k in range(2, n_terms + 1):
        # term ← term · x   (gives x^k still divided by (k-1)!)
        term = fp_mul(term, x)
        fact_k *= k
        # then divide by k to bring to x^k / k!
        contrib = fp_div_by_int(term, k)
        # accumulate
        acc = fp_add(acc, contrib)
        # convergence: stop when contrib's underlying integer is 0
        if not np.any(contrib.trits):
            break
        # also pin term back to the post-divide value to avoid factorial
        # growing unboundedly in `term` itself:
        term = contrib
        # restore: next iteration multiplies by x again to get x^(k+1)/(k!)
        # then divides by k+1. So term needs to be x^k / k! at this point.
        # That's exactly contrib. ✓
    return acc


# ─────────────────────────────────────────────────────────────────────
# Taylor log via atanh series:
#   log(x) = 2 · Σ u^(2j+1) / (2j+1)   where u = (x-1)/(x+1).
# Converges for x > 0. For x in [1, ~3], converges fast.
# For large x, optionally reduce: log(x) = log(x/3) + log(3).
# First iteration: accept x > 0 in any range, take up to n_terms.
# ─────────────────────────────────────────────────────────────────────

def log_taylor(x: FixedPoint, n_terms: int = 100) -> FixedPoint:
    """Compute log(x) via 2·atanh((x-1)/(x+1)) series."""
    one = fp_from_int(1, x.scale, len(x.trits))
    u = fp_div(fp_sub(x, one), fp_add(x, one))   # (x-1)/(x+1)
    u_sq = fp_mul(u, u)
    acc = FixedPoint(u.trits.copy(), x.scale)    # j=0 term: u/1
    cur_u_power = FixedPoint(u.trits.copy(), x.scale)
    for j in range(1, n_terms):
        cur_u_power = fp_mul(cur_u_power, u_sq)  # u^(2j+1)
        contrib = fp_div_by_int(cur_u_power, 2 * j + 1)
        acc = fp_add(acc, contrib)
        if not np.any(contrib.trits):
            break
    # double
    two = fp_from_int(2, x.scale, len(x.trits))
    return fp_mul(acc, two)


# ─────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math
    print(f"=== fixed-point round-trip (scale={SCALE_DEFAULT}) ===")
    for v in [0.0, 1.0, -1.0, 0.5, math.pi, math.e, 100.0, -0.001]:
        fp = fp_encode(v)
        rev = fp_decode(fp)
        print(f"  {v:>12.6f} -> {rev:>12.6f}   err={abs(rev-v):.2e}")

    print("\n=== fp_add / fp_sub / fp_mul / fp_div ===")
    a = fp_encode(1.5)
    b = fp_encode(2.5)
    print(f"  1.5 + 2.5 = {fp_decode(fp_add(a, b)):.6f}  (want 4.0)")
    print(f"  1.5 - 2.5 = {fp_decode(fp_sub(a, b)):.6f}  (want -1.0)")
    print(f"  1.5 * 2.5 = {fp_decode(fp_mul(a, b)):.6f}  (want 3.75)")
    print(f"  1.5 / 2.5 = {fp_decode(fp_div(a, b)):.6f}  (want 0.6)")

    print("\n=== exp_taylor ===")
    for v in [0.0, 1.0, -1.0, 0.5, 2.0, 5.0]:
        fp = fp_encode(v)
        result = fp_decode(exp_taylor(fp))
        want = math.exp(v)
        err = abs(result - want) / max(abs(want), 1e-30)
        print(f"  exp({v:>5}) = {result:>14.9f}  want {want:>14.9f}  rel_err={err:.2e}")

    print("\n=== log_taylor ===")
    for v in [1.0, 2.0, math.e, 5.0, 10.0]:
        fp = fp_encode(v)
        result = fp_decode(log_taylor(fp))
        want = math.log(v)
        err = abs(result - want) / max(abs(want), 1e-30)
        print(f"  log({v:>5.3f}) = {result:>14.9f}  want {want:>14.9f}  rel_err={err:.2e}")

    print("\n=== composition: exp(log(x)) ≈ x ===")
    for v in [2.0, 3.0, 5.0]:
        fp = fp_encode(v)
        lx = log_taylor(fp)
        elx = exp_taylor(lx)
        result = fp_decode(elx)
        err = abs(result - v) / abs(v)
        print(f"  exp(log({v:>5})) = {result:>12.6f}  rel_err={err:.2e}")
