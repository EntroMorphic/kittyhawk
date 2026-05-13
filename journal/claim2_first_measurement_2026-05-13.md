# Claim 2 bridge first measurement — POSITIVE result

Per `glyph_gaps_2026-05-13_synthesize.md` Track A3. First time the
vision's claim 2 ("all math = signatures via routing over the
6 frozen primitives, address-as-identity") has a concrete
measurement.

## Setup

Two approaches implemented in `experiments/claim2_bridge/`:

- **Approach A — canonical-form-hash.** Parse to AST,
  canonicalize via flatten + simplify + sort (the canonicalization
  encodes the equivalences explicitly), then SHA-derive a trit
  signature with ~62% nonzero (matching K-sig sparsity profile).
  Equivalences are baked into A's rewriter, not into the substrate.

- **Approach B — routing-derived.** Each variable and constant gets
  a base trit signature. Each primitive (`+`, `−`, `*`, unary `−`)
  has an element-wise routing on trits:
    - `route_add`: saturating ternary add
    - `route_sub`: saturating ternary sub
    - `route_mul`: ternary product (closed in {−1, 0, +1})
    - `route_neg`: element-wise negation
  Signatures compose bottom-up. The path-graph structure of trits
  carries the algebra.

Canonical sort+flatten happens BEFORE reduction in approach B, so
the result is invariant to the parser's left-association of `a+b+c`.
This converts the operation's non-associativity into a
representation-level associativity.

D = 128 (matches BitNet substrate HEAD_DIM). 12 equivalence classes,
21 distinct expressions, 210 distinctness pairs.

## Results

| equivalence class       | A: canonical | B: routing |
|-------------------------|:------------:|:----------:|
| commutativity (+ and *) | 100%         | 100%       |
| associativity (+ and *) | 100%         | 100%       |
| identity (x+0, x*1)     | 100%         | 100%       |
| absorbing zero (x*0)    | 100%         | 100%       |
| additive inverse (x-x)  | 100%         | 100%       |
| double negation         | 100%         | 100%       |
| **distributivity** (x*(y+z) == x*y+x*z) | **0%**  | **100%**  |
| **(x+y)(x−y) == x²−y²** | **0%**       | **100%**   |
| constant arithmetic (2+3 == 5) | 0%    | 0%         |

Both approaches: **0 collisions on 21 distinct expressions × 210 pairs.**
Determinism: 100% (sanity check).

## What the result says about claim 2

**Approach B substantiates claim 2 stronger than approach A.** A
preserves equivalences only via its hand-coded canonicalization
rewriter. B preserves equivalences as an emergent property of
trit-substrate arithmetic.

Two algebraic identities that A cannot reach (distributivity, the
difference-of-squares identity) are FREE in approach B. Why?

- **Distributivity:** for each cell `i`, `s(x*(y+z))[i] = x[i] *
  sat(y[i] + z[i])` and `s(x*y + x*z)[i] = sat(x[i]*y[i] +
  x[i]*z[i])`. These are equal cell-by-cell across all 9
  combinations of `x[i], y[i], z[i] ∈ {−1, 0, +1}`. The
  path-graph structure of the trit alphabet, combined with
  element-wise routing, naturally distributes.

- **Difference of squares:** for each cell `i`,
  `s((x+y)(x−y))[i] = sat(x+y) * sat(x−y)` and
  `s(x*x − y*y)[i] = sat(x*x − y*y)`. Enumerate all 9 (x[i], y[i])
  pairs — both expressions agree. Saturation interacts with
  ternary multiplication in just the right way.

**This is what "address-as-identity via routing" was supposed to
mean.** The substrate's arithmetic is the math; the signature is
the trajectory; algebraic identities that hold in the trit algebra
hold in the signatures.

## What approach B does NOT capture

- **Constant arithmetic** (`2 + 3 == 5`): in approach B, `s(2)` and
  `s(3)` are SHA-derived signatures bearing no relation to `s(5)`.
  To make `route_add(s(2), s(3)) = s(5)`, integers would need a
  balanced-ternary encoding so that the routing literally performs
  trit-wise addition with carry. That's a real design exercise; not
  in this iteration.
- **Identities outside the trit algebra.** If an algebraic identity
  is true over the integers/rationals but NOT over the saturating
  ternary algebra, it won't hold in approach B. The substrate has
  its own algebra; the bridge measures coincidence between the
  substrate's algebra and the desired math algebra.

## What this changes for the vision

Pre-this-measurement, claim 2 was words. Post-this-measurement:

- **Claim 2 has a falsifiable definition.** Equivalence preservation
  rate per algebraic class is now a measurable quantity. The bridge
  is concrete code (`experiments/claim2_bridge/`).
- **Approach B is the substrate-native realization** — it earns the
  vision's framing. The trit substrate doesn't just store
  signatures; its arithmetic IS the structure being measured.
- **The substrate has its own algebra**, the *saturating ternary*
  algebra, which agrees with the integer algebra on commutativity,
  associativity (via canonical reduction), identity, inverse,
  distributivity, and the difference-of-squares identity, but
  DIVERGES on constant arithmetic. The latter divergence is a
  design choice (current integer encoding) rather than a
  fundamental limit.
- **Claim 1 (six-primitive floor) now has a use case.** The bridge
  uses `add, sub, mul, neg` — four of the named primitives. If the
  bridge is extended to handle `exp/log/div`, claim 1's
  primitive-set audit becomes concrete: "does the named primitive
  set produce a bridge that closes over the math the project
  needs?"
- **Claim 3 (substrate = path graph) is reinforced** as a structural
  fact rather than just a metric. The reason distributivity holds
  on the substrate is the path-graph algebra of trits, not just the
  L1 distance on trits. Claim 3 isn't just about distance — it's
  about the whole arithmetic.

## Anti-overclaim check

- **D=128 may not be load-bearing here.** All algebraic-identity
  checks are pointwise (per-cell); they would hold at D=1 or
  D=4096. Collision-rate measurements DO depend on D, but with 21
  distinct expressions at D=128, the 0% collision rate is
  unsurprising (signature space is 3^128).
- **The expression battery is small** (12 classes, 21 distinct).
  Patterns missed by this battery aren't being tested.
- **B's "100% distributivity" is mathematical, not statistical.**
  It holds by case analysis on the 9 trit-combinations per cell.
  Hand-verified for one cell; the implementation reproduces it.
- **Constant arithmetic 0% is real.** Don't gloss it as "edge case."
  Integer literals in expressions don't currently propagate through
  the substrate. Most non-trivial math has constants. The bridge
  is incomplete here.

## Next iteration

1. **Balanced-ternary integer encoding** so `s(2) + s(3) = s(5)`.
   This makes the bridge handle numerical computation, not just
   variable algebra.
2. **Larger expression battery** including polynomials, nested
   products, and integer-valued operations.
3. **exp / log primitives** for claim 1 closure. Without them, the
   bridge can't express softmax, log-likelihood, or anything
   involving exponential decay.
4. **Production C kernel** for routing primitives if the bridge
   moves toward load-bearing use. Currently Python research code.

## Files

- `experiments/claim2_bridge/parser.py` — recursive-descent parser.
- `experiments/claim2_bridge/canonical.py` — approach A.
- `experiments/claim2_bridge/routing.py` — approach B.
- `experiments/claim2_bridge/measure.py` — battery + reports.
- `experiments/claim2_bridge/README.md` — pointer doc.
- `journal/claim2_bridge_spec.md` — the spec this implements.

## Discipline log

This is the FIRST concrete measurement of vision claim 2 in the
project's history. The substrate-claim arc spent 11+ journals on
a corollary of claim 3 (KV eviction). Claim 2's first measurement
took ~2 hours and produced a positive result (11/12 classes at
100%, including 2 non-trivial algebraic identities that approach
A cannot reach).

The LMM synthesis pivot was correct: foundational work has higher
information density than corollary refinement. The path-graph
substrate naturally implements more algebra than I expected
(distributivity, difference-of-squares are FREE under saturating
ternary). The vision's claim 2 framing earned its first piece of
operational evidence.

## Sign-off

Claim 2 has its first measurement loop. The bridge exists. Approach
B substantiates the vision's "address-as-identity via routing"
framing for 11 of 12 tested equivalence classes. The next
foundational gap — claim 1's primitive-set closure — now has the
inputs it needs.
