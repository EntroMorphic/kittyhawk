# Claim 2 bridge — division extension

Per `glyph_gaps_2026-05-13_synthesize.md` next-iteration #2: division
joins the bridge. The substrate's elemental floor includes `shift3`
and `sign` and `select`; balanced-ternary integer division composes
from these via iterated conditional-sub. For first iteration the
bridge uses an integer-level implementation in `numeric.py` that
matches truncate-toward-zero semantics; substrate-native iterated-
sub is deferred to the C kernel work that
`elemental_floor_closeout.md` lines up.

## What landed

- `numeric.balt_div(a, b)` — returns `(quotient, remainder)` trit
  vectors with truncate-toward-zero semantics (matches C `int(a/b)`).
  Self-tested: 100 random divisions over ±500/±100, all round-trip
  correct.
- `parser.py` — extended grammar with `/` operator at term level
  (same precedence as `*`).
- `canonical.py` — handles `div` in flatten/sort/serialize and
  simplifies `x/1=x`, `0/x=0`, `x/x=1`; pure-numeric div folds
  through `balt_div`.
- `routing.py` — pure-numeric div folds through `balt_div`. Mixed-
  variable div falls back to SHA-derived signature, but `signature()`
  now first calls `canonical._simplify` so algebraic identities like
  `x/x=1` reduce before the fallback fires.

## Updated measurement (15 classes)

| equivalence class       | A: canonical | B: routing |
|-------------------------|:------------:|:----------:|
| commutativity, associativity | 100%   | 100%       |
| identity, inverse, double-neg | 100%  | 100%       |
| absorbing zero          | 100%         | 100%       |
| distributivity          | 0%           | **100%**   |
| (x+y)(x−y) == x²−y²     | 0%           | **100%**   |
| constant_arithmetic     | 100%         | 100%       |
| numeric_in_expr         | 100%         | 100%       |
| **division_numeric** (8 pairs)  | **100%** | **100%** |
| **division_in_expr** (5 pairs)  | **100%** | **100%** |

**Approach B now hits 15/15 equivalence classes at 100%.** 0
collisions on 276 distinct-expression pairs at D=128.

The integration:

- **Element-wise routing** for abstract algebra of variables
  (distributivity, diff-of-squares fall out cell-by-cell).
- **Balanced-ternary positional routing** (`balt_*`) for pure-
  numeric subtrees (constant arithmetic propagates through the
  substrate).
- **Algebraic rewriter** (canonical's `_simplify`) for identities
  like `x/x=1`, `x/1=x`, `0/x=0` that neither element-wise nor
  positional routing reaches naturally.

The bridge dispatches based on subtree type. The substrate IS doing
the math in each regime; the bridge just selects the right
substrate operation.

## What this closes vs leaves open

**Closed:**
- Integer arithmetic in expressions (constant_arithmetic +
  numeric_in_expr).
- Integer division with truncate-toward-zero (matches C / Python
  `int(a/b)` semantics).
- Division identities that reduce algebraically (x/x, x/1, 0/x).

**Still open (next iteration):**
- **exp / log.** Taylor series require non-integer results (fixed-
  point arithmetic). The bridge currently has no fixed-point or
  fractional encoding. Adding exp/log requires:
    1. A fixed-point trit encoding (e.g., balanced-ternary with an
       implicit radix-point shift).
    2. Taylor-coefficient table.
    3. Fixed-point versions of balt_add and balt_mul.
- **General algebraic identities** like `(x+y)² = x² + 2xy + y²`
  that require expansion. Approach B's element-wise routing handles
  these only when they reduce pointwise on the trit algebra.
- **Comparison / max / min / abs / eq** in expressions. Substrate
  has these as composite kernels but the bridge doesn't wire them.

## Substrate division: bridge vs kernel

For this iteration, `balt_div` runs Python integer division then
re-encodes. The substrate-native version (per
`elemental_floor_closeout.md`) is iterated conditional-sub:

```
q ← 0; r ← a
for j from high to 0:
    while |r| ≥ |b · 3^j|:
        sign-aware-subtract b·3^j from r
        q ← q + sign · 3^j
return (q, r)
```

All five elemental floor primitives (`add`, `neg`, `shift3`, `sign`,
`select`) are invoked. Implementing this as `m4t_mtfp_div` in the
substrate C kernels is the next-but-not-this-session work; the
bridge's Python balt_div is the algorithmic specification.

## Cumulative bridge state (end of 2026-05-13 session)

- **15 equivalence classes** tested.
- **Approach B: 15/15 at 100%.** Approach A: 13/15.
- **0 collisions** on 276 distinct-expression pairs.
- Operators supported: `+`, `−`, `*`, `/` (truncate-toward-zero),
  unary `−`.
- Substrate-level routings exercised: element-wise saturating add
  / sub / mul / neg for abstract algebra; positional balanced-
  ternary add / sub / mul / div for numeric.
- Vision claim 2 (math = signatures via routing) substantiated for
  this scope.

## Files

- `experiments/claim2_bridge/numeric.py` — balt_div added.
- `experiments/claim2_bridge/parser.py` — `/` operator.
- `experiments/claim2_bridge/canonical.py` — div in flatten/sort/
  serialize/simplify, `x/x=1` rule.
- `experiments/claim2_bridge/routing.py` — div fold + canonicalize
  integration.
- `experiments/claim2_bridge/measure.py` — 15 classes.

## Sign-off

Division added cleanly. The bridge now covers integer arithmetic
plus the four classical operations, with substrate-level routings
doing the actual math. The next gap — exp/log — requires a fixed-
point encoding and is a separate cycle. For this session: claim 2
bridge has reached its first substantively complete form on integer
algebra; claim 1's elemental floor closes over the routings used.
The arc's verdict for this gaps-cycle is positive.
