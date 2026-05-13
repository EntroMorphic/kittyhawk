# claim2_bridge — math-expression → substrate signature

First implementation of claim 2 of Glyph's vision:

> All mathematics can be classified and expressed as signatures via
> routing over the 6 frozen primitives. Different math = different
> routing = different signature. Address-as-identity.

## Quick start

```
python experiments/claim2_bridge/measure.py
```

## What's here

- `parser.py` — recursive-descent parser for arithmetic expressions
  over `+`, `-`, `*`, integers, and identifiers.
- `canonical.py` — Approach A: canonicalize AST, then SHA-derive a
  trit signature. Equivalences are baked into the canonicalization
  rules.
- `routing.py` — Approach B: each variable has a base trit signature,
  each primitive operation has a routing function on trits, the
  signature of an expression is the composed routing bottom-up.
- `measure.py` — battery of equivalence classes with preservation
  rate per class and collision rate on distinct expressions.

## Findings (first iteration, D=128)

|equivalence class       | A canonical | B routing |
|------------------------|-------------|-----------|
| commutativity (+ and *)| 100%        | 100%      |
| associativity (+ and *)| 100%        | 100%      |
| identity x+0, x*1      | 100%        | 100%      |
| absorbing x*0          | 100%        | 100%      |
| additive inverse x-x   | 100%        | 100%      |
| double negation        | 100%        | 100%      |
| **distributivity**     | **0%**      | **100%**  |
| **(x+y)(x-y) == x²−y²**| **0%**      | **100%**  |
| constant arithmetic    | 0%          | 0%        |

Collisions on 21 distinct expressions × 210 pairs: 0 for both.

**Approach B (routing-derived) earns the vision's framing.** Element-
wise ternary multiplication distributes over saturating ternary
addition by construction. The path-graph structure of trits IS
carrying algebraic load. Approach A is a control baseline that only
preserves equivalences encoded into its canonicalization.

See `journal/claim2_first_measurement_2026-05-13.md` for the full
write-up.

## Out of scope (next iterations)

- exp, log, division (claim 1 gap).
- Functions, abstraction, higher-order operations.
- Balanced-ternary integer encoding so `2 + 3 == 5` propagates.
- Production C kernels (research code in Python first).
