# claim2_bridge — math-expression → substrate signature

First implementation of claim 2 of Glyph's vision:

> All mathematics can be classified and expressed as signatures via
> routing over the 6 frozen primitives. Different math = different
> routing = different signature. Address-as-identity.

## Quick start

```
python experiments/claim2_bridge/measure.py            # 17/17 equivalence classes
python experiments/claim2_bridge/sympy_battery.py      # 32/32 SymPy adversarial
python experiments/claim2_bridge/confluence.py         # 5/5 confluence axes
python experiments/claim2_bridge/consumer_demo.py      # 4/4 cache properties
```

## What's here

**Core:**
- `parser.py` — recursive-descent parser for arithmetic expressions
  over `+`, `-`, `*`, `/`, unary `-`, integers, identifiers, and the
  function calls `exp(...)` and `log(...)`.
- `canonical.py` — Approach A: canonicalize AST, then SHA-derive a
  trit signature. **Faithful** for consumer equivalence detection.
- `routing.py` — Approach B: each variable has a base trit signature,
  each primitive operation has a routing function on trits, the
  signature of an expression is the composed routing bottom-up.
  **NOT faithful** for arbitrary consumer use (saturating add can
  collide distinct values, e.g., x²+y² vs (x+y)²); see consumer_demo.py.
- `numeric.py`, `fixed_point.py` — balanced-ternary integer arithmetic
  and MTFP-style fixed-point with Taylor exp/log.

**Test batteries (run independently, no shared state):**
- `measure.py` — 17 equivalence classes (commutativity, associativity,
  identity, inverse, double-neg, absorbing zero, distributivity, diff-
  of-squares, constant arithmetic, division, exp/log). 0 collisions on
  276 distinct expressions. Reports both approaches A and B.
- `sympy_battery.py` — 32 adversarial pairs generated via SymPy
  transformations (expand, factor, simplify, etc.). External-ground-
  truth validation: if SymPy says equivalent, the bridge must agree.
- `confluence.py` — 4 axes of rewrite-system correctness: idempotence
  (canonicalize is a fixed point), permutation invariance (commutative
  ops), identity injection (wrapping with e+0, e*1, etc.), and 2000
  SymPy random pairs.
- `tolerance_sensitivity.py` — sweep of the integer-demotion tolerance
  from 1e-3 to 1e-18 across 14 test cases. Documents the constraint
  window (Taylor noise floor ~1.1e-16 vs user-precision ceiling
  ~1e-12). Sets `INTEGER_DEMOTE_TOL = 1e-12` as the production value.
- `positivity_contract.py` — 6 cases verifying the permissive vs strict
  positivity mode for the `exp(log(e)) → e` rewrite. Documents that
  bare-variable positivity is a CONTRACT, not a mathematical fact.
- `consumer_demo.py` — expression-equivalence cache built on the
  bridge's signature. 4 properties: equivalent exprs share entry,
  distinct exprs stay separate (caught the routing-vs-canonical-hash
  fidelity issue), cached values correct, 15× speedup.
- `redteam.py`, `redteam_explog.py` — integer and exp/log red-teams.
  22+16 adversarial cases all close.

## Findings (2026-05-13/14, D=128)

| gate | result |
|---|---|
| main battery (`measure.py`) | 17/17 at 100%, 0/276 collisions, both A and B |
| integer red-team | 16/16 |
| exp/log red-team | 22/22 |
| SymPy adversarial | 32/32 |
| confluence (4 axes, 5000 cases) | 100% on every axis |
| positivity contract | 6/6 |
| consumer demo | 4/4 (15× cache speedup) |
| **cumulative** | **4604/4604** |

**Approach A vs Approach B:**
- Approach A (canonical-hash, this module's `signature_from_expr` →
  SHA over canonical AST): faithful. Use for content-addressable
  caching, equivalence detection, identity database lookup.
- Approach B (routing-derived, `routing.signature_from_expr`):
  faithful for the substrate's own notion of value (saturated trits),
  not for arbitrary mathematical value. Use for substrate-internal
  work; not for consumer-grade equality.

## Operator coverage

Currently in scope: `+`, `−`, `*`, `/`, unary `−`, integer literals,
lowercase variable identifiers, `exp(·)`, `log(·)` (natural log).

Not yet in scope: trig, derivatives, integrals, rational coefficients
beyond integer division, higher-order/abstract operations.

## Architecture summary

The bridge canonicalizes expressions through:
1. Expansion (cartesian distribution of mul-over-add; sub→add+neg;
   neg pushing).
2. exp/log identity rewrites (positivity-gated).
3. Iterated simplification + sort with `_combine_like_terms` grouping
   same-shape monomials by integer coefficient.

The pipeline reaches a unique canonical AST per algebraic equivalence
class. Approach A SHA-hashes this AST; approach B routes through trit
ops bottom-up. See `canonical.py`'s module docstring for the full
pipeline detail.

## References

- `journal/claim2_first_measurement_2026-05-13.md` — initial result.
- `journal/claim2_balt_numeric_2026-05-13.md` — integer arithmetic.
- `journal/claim2_division_2026-05-13.md` — division added.
- `journal/claim2_exp_log_2026-05-13.md` — Taylor exp/log integration.
- `journal/claim2_explog_redteam_2026-05-13.md` — exp/log red-team.
- `journal/claim2_unsoundness_designtension_fixes_2026-05-13.md` —
  positivity-gated rewrites + integer-rounding demotion.
- `journal/claim2_100of100_remediation_2026-05-13.md` — SymPy battery,
  confluence, tolerance sensitivity, positivity contract,
  combine-like-terms refactor.
- `journal/claim2_100of100_remediation_pt2_2026-05-13.md` — consumer
  demo + the routing-vs-canonical-hash fidelity finding.
