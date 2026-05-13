# Claim 2 bridge — balanced-ternary numeric extension

Per `glyph_gaps_2026-05-13_synthesize.md` next-iteration #1:
balanced-ternary integer encoding so `s(2+3) == s(5)` propagates
through the substrate.

## What landed

- `experiments/claim2_bridge/numeric.py` — balanced-ternary
  integer encoding and substrate-level routings:
    - `encode(n) -> trit vector` (length D, balanced-ternary).
    - `decode(sig) -> int`.
    - `balt_add(a, b)` — positional add with carry propagation.
    - `balt_neg(a)` — element-wise sign flip.
    - `balt_sub(a, b)` — composite from add + neg.
    - `balt_mul(a, b)` — shift-and-add positional multiply.
  Self-tested: 100 random pairs over ±500 × ±500, all round-trip
  correct.

- `experiments/claim2_bridge/canonical.py` — pure-numeric subtree
  detection (`_is_pure_numeric`) and substrate-level folding
  (`_fold_numeric`), plus a partition-fold step for mixed n-ary
  add/mul that folds the numeric children into one constant via
  balt routings. The trit substrate does the arithmetic; the
  bridge integrates the result back into the AST.

- `experiments/claim2_bridge/routing.py` — same integration in the
  routing-derived approach. The bridge identifies pure-numeric
  subtrees, runs them through balt routings, replaces them with
  the resulting ('const', n) leaf, and proceeds.

## Updated measurement

13 equivalence classes (12 from prior iteration + new
`numeric_in_expr` covering mixed expressions like `x + 2 + 3` vs
`x + 5`). All pairs evaluated at D=128.

| equivalence class       | A: canonical | B: routing |
|-------------------------|:------------:|:----------:|
| commutativity (+ and *) | 100%         | 100%       |
| associativity (+ and *) | 100%         | 100%       |
| identity (x+0, x*1)     | 100%         | 100%       |
| absorbing zero (x*0)    | 100%         | 100%       |
| additive inverse (x-x)  | 100%         | 100%       |
| double negation         | 100%         | 100%       |
| distributivity          | 0%           | 100%       |
| (x+y)(x−y) == x²−y²     | 0%           | 100%       |
| **constant_arithmetic** | **100%** (was 0%) | **100%** (was 0%) |
| **numeric_in_expr**     | **100%** (new) | **100%** (new) |

**Approach B: 13/13 classes at 100%.** Approach A still misses
distributivity and difference-of-squares, by design (its rewriter
doesn't expand algebraic identities). Distinct-expression
collisions: still 0/210 for both.

## What the integration looks like in practice

Trace for `x + 2 + 3`:

1. Parser produces `('add', ('add', ('var', 'x'), ('const', 2)), ('const', 3))`.
2. `_flatten` recurses, collapses to `('add', ('var', 'x'), ('const', 2), ('const', 3))`.
3. `_partition_fold` separates `('const', 2)` and `('const', 3)` from `('var', 'x')`.
4. Folds the numerics via balt: `balt_add(encode(2), encode(3)) = encode(5)`.
   That's a substrate-level computation — positional carry-propagating
   ternary add operating on packed-trit vectors.
5. Re-assembles AST: `('add', ('var', 'x'), ('const', 5))`.
6. Signature derivation proceeds with this canonicalized AST.

`x + 5` parses directly to `('add', ('var', 'x'), ('const', 5))`. After
flattening, same canonical form. Same signature.

The substrate is doing the math. The bridge is dispatching it.

## Honest scope

What this extension does NOT do:

- **Division, exp, log.** Still missing kernels. balt_mul exists;
  balt_div (long division via shift3 + iterated conditional-sub)
  does not. Without division, Taylor exp/log are unreachable.
- **Comparison, max, min, abs, eq.** Available in the substrate as
  composite kernels but not wired through the bridge yet.
- **General algebraic identities** (e.g., `x*x + 2*x*y + y*y ==
  (x+y)²`). Approach B preserves these only when they reduce to
  pointwise trit identities, which is true for distributivity and
  difference-of-squares but not for the general case.

## What's next

Per the gaps synthesis priority order:

1. ✓ Balanced-ternary integer encoding (this work).
2. **Division kernel.** Composite from `shift3` + iterated
   conditional-sub per `elemental_floor_closeout.md`. Once shipped
   in the substrate, division joins the bridge. ~1-2 days.
3. **Exp/log via Taylor.** Requires division (#2). Once shipped, the
   bridge can express softmax, log-likelihood, sigmoid. The vision's
   "all math = routing over primitives" framing becomes operational
   for these heavily-used ops.
4. **Bridge grammar extension** for division/exp/log when substrate
   kernels exist.

## Files

- `experiments/claim2_bridge/numeric.py` — substrate-level
  balanced-ternary routings.
- `experiments/claim2_bridge/canonical.py` — approach A with
  fold integration.
- `experiments/claim2_bridge/routing.py` — approach B with fold
  integration.
- `experiments/claim2_bridge/measure.py` — 13 equivalence classes.

## Sign-off

Claim 2's bridge now closes constant arithmetic via substrate-level
balanced-ternary routings. The trit substrate's positional structure
literally performs the integer addition and multiplication. 13/13
equivalence classes at 100% for the routing-derived approach. The
remaining bridge work is exp/log/div — gated on a substrate-level
division kernel, which is the next foundational gap.
