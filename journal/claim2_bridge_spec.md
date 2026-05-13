# Claim 2 bridge spec — math-expression → signature

Per `glyph_gaps_2026-05-13_synthesize.md` Track A1.

## What this is

The first concrete specification of how a mathematical expression
becomes a substrate signature. Claim 2 of the vision says:

> All mathematics can be classified and expressed as signatures via
> routing over the 6 frozen primitives. Different math = different
> routing = different signature. **Address-as-identity.**

Until this spec exists, claim 2 is words. Until a toy
implementation exists, the spec is also words. This is the spec.

## Scope (first iteration)

**Input grammar:** arithmetic expressions over

- a finite set of variables (lowercase identifiers, e.g. `x, y, z`),
- integer constants,
- three binary operations: `+`, `-`, `*`.

Grammar in EBNF-ish form:

```
expr   := term (('+' | '-') term)*
term   := factor ('*' factor)*
factor := identifier | integer | '(' expr ')'
```

Out of scope for this iteration (named for later):

- exp, log (foundational claim 1 has these in the named-six but not
  implemented in the project yet — same gap).
- Division (introduces undefined cases; needs careful handling).
- Functions / abstraction.
- Boolean / comparison operators.
- Sequences / matrices.

## Output

A signature `s(e) ∈ {−1, 0, +1}^D` for a fixed `D` (start at `D = 128`
to match the BitNet substrate's HEAD_DIM). The signature is a trit
vector — the project's canonical substrate encoding.

## Derivation: two candidate approaches

### Approach A — canonical form + hash (the baseline)

1. Parse expression to AST.
2. Canonicalize:
   - Sort children of `+` and `*` by their canonical-form hash
     (commutativity).
   - Flatten nested associative operations into a single n-ary node
     (associativity).
   - Simplify `e + 0 → e`, `e * 1 → e`, `e * 0 → 0`, `e - e → 0`.
3. Hash the canonical AST to a trit vector via a deterministic
   keystream: SHA256 → expand to D trits in `{−1, 0, +1}` (use the
   sparsity profile of substrate signatures: target ~62% nonzero,
   matching K-sigs in `feedback_calibrate_on_application_distribution`).

**Equivalences trivially respected** (by canonicalization):

- Commutativity of `+`: `x+y == y+x` → same canonical form → same hash.
- Commutativity of `*`: same.
- Associativity of `+` and `*`: same.
- Simple identities: `e + 0`, `e * 1`, `e * 0`, `e − e`.

**Equivalences NOT respected by A:**

- Distributivity: `x*(y+z)` and `x*y + x*z` are syntactically
  distinct and will hash differently. Canonicalizing this requires
  algebraic expansion / a CAS-like normal form.
- Higher identities (e.g., `(x+y)*(x−y) == x² − y²`).

**Substrate connection:** weak. The signature is a hash; the
path-graph structure of trits isn't carrying semantic load. Approach
A is a "control" baseline that doesn't substantiate claim 2's deeper
claim ("math IS routing") — it just satisfies the address-as-identity
constraint at a shallow level.

### Approach B — routing-derived signature (the deeper claim)

1. Each variable has a base signature (deterministic, e.g. hashed
   from the variable's name).
2. Each constant has a base signature (deterministic mapping from
   integer to trits — small integers via direct trit encoding, large
   via SHA-derived).
3. Each primitive operation has a defined ROUTING that takes two
   signatures and produces one signature:
   - `route_add(s₁, s₂)`: element-wise saturating ternary add.
     `+1 + +1 → +1`, `+1 + −1 → 0`, `+1 + 0 → +1`, etc. **Commutative.**
   - `route_sub(s₁, s₂)`: element-wise saturating ternary sub.
     `+1 − +1 → 0`, `+1 − −1 → +1` (saturated), etc. **Anti-commutative.**
   - `route_mul(s₁, s₂)`: element-wise ternary product. `+1 * +1 → +1`,
     `+1 * −1 → −1`, `0 * anything → 0`. **Commutative.**
4. The signature of an expression is the composed routing applied
   bottom-up over the AST.

**Equivalences respected** (by construction):

- Commutativity of `+` and `*`: routings are commutative.
- Identity: if `s(0)` is the all-zero vector and `s(1)` is the
  all-`+1` vector, then `route_add(s, s(0)) = s`, `route_mul(s, s(1))
  = s`, `route_mul(s, s(0)) = s(0)`.
- Inverse: `route_sub(s, s) = s(0)` (any cell minus itself is 0).

**Equivalences NOT immediately respected by B:**

- Associativity of `+`: saturating add is NOT associative in general
  (`(+1 + +1) + −1 = +1 + −1 = 0`, but `+1 + (+1 + −1) = +1 + 0 = +1`).
  This breaks `(x+y)+z == x+(y+z)`. Approach B's first iteration
  WILL violate associativity. Either accept this as a known
  limitation, or introduce a non-saturating-then-canonicalize step.
- Distributivity: same problem as A.

**Substrate connection:** STRONG. The signature IS the substrate
trajectory through the primitives. Each cell of the signature is a
deterministic function of the input cells under the primitive's
routing rule. The path-graph structure of trits (`−1`-`0`-`+1` as a
path) carries the semantic load directly. This is what claim 2
seems to want.

### Choice for the first iteration

Implement **both** A and B in the toy. They are complementary:

- A is a baseline that respects more equivalences but doesn't
  exercise the substrate.
- B is the substrate-native realization but respects fewer
  equivalences out of the box.

Measuring A and B on the same expression battery tells us:

1. **How often does B's signature collide with A's signature?** If
   often, the routing-derived signature carries similar information
   to the canonical-form hash, suggesting they're capturing related
   structure. If rarely, B and A are measuring different things.

2. **Which equivalences does B preserve in practice?** If saturating
   add turns out to be "associative enough" on the natural
   distribution of expressions (e.g., signs cancelling rarely), B
   may be operationally good even though not mathematically
   associative.

3. **What's the signature-collision rate for distinct expressions?**
   Both A and B map an infinite expression space to a finite
   signature space, so collisions are unavoidable. Measuring their
   rate at D=128 tells us whether D needs to scale.

## Measurement loop (the falsifiable test)

For a battery of expressions:

```
EXPRESSIONS = [
  ("zero_a",  "0"),
  ("zero_b",  "x - x"),
  ("zero_c",  "x * 0"),
  ("zero_d",  "0 * x"),
  ("one_a",   "1"),
  ("one_b",   "x / x"),   # skipped: division out of scope
  ("simple",  "x"),
  ("comm_a",  "x + y"),
  ("comm_b",  "y + x"),
  ("comm_c",  "x * y"),
  ("comm_d",  "y * x"),
  ("assoc_a", "(x + y) + z"),
  ("assoc_b", "x + (y + z)"),
  ("ident_a", "x + 0"),
  ("ident_b", "0 + x"),
  ("ident_c", "x * 1"),
  ("inv_a",   "x - x"),
  ("inv_b",   "x + (-x)"),
  ("distrib_a", "x * (y + z)"),
  ("distrib_b", "x * y + x * z"),
  ("poly_a",  "x*x - y*y"),
  ("poly_b",  "(x + y) * (x - y)"),
  ("constants_a", "2 + 3"),
  ("constants_b", "5"),
  ("constants_c", "3 + 2"),
  # ... more
]
```

For each pair of expressions that are KNOWN to be equivalent, the
expected outcome is `s(e₁) == s(e₂)`. Define:

- **equivalence preservation rate** (per equivalence class):
  `|same signature| / |pairs in class|`.
  Target for commutativity, identity, inverse: 100% (or near).
  Target for associativity and distributivity: best-effort.

- **distinct-expression collision rate**:
  fraction of unrelated expressions that accidentally share a
  signature. Target: ~0 at D=128 (signature space is `3^128`, so
  random collision should be negligible if the hash is well-mixed).

- **determinism**: same expression evaluated twice produces the same
  signature. Target: 100%. Trivial; included as a sanity check.

## Falsifiability

The bridge claim is falsifiable in either direction:

- **Falsified if:** equivalence preservation is so low that the
  signature doesn't track math semantics at all (e.g., `x+y` and
  `y+x` produce different signatures most of the time in approach
  A — would indicate a canonicalization bug or a fundamental issue
  with the encoding).
- **Falsified if:** approach B's routing-derived signatures
  collapse all expressions to the same trit pattern (e.g., all
  saturating-add chains converge to all-zero or all-±1).
- **Validated to first iteration if:** approach A passes the
  commutativity and identity tests at 100%, AND approach B passes
  at >50% on commutativity (saturating arithmetic permitting).

## Files (planned)

```
experiments/claim2_bridge/
  __init__.py
  parser.py             # expr string → AST
  canonical.py          # approach A: canonicalize + hash
  routing.py            # approach B: signatures + primitive routings
  measure.py            # battery + equivalence-preservation metrics
  README.md             # how to run
journal/
  claim2_first_measurement_<date>.md  # results
```

## Out of scope for this spec

- Bridging to claim 1 (do these primitives close over the math the
  project needs?). That's the next measurement after this one
  produces stable signatures.
- Production C implementation. Research code in Python first.
- exp, log, division, comparisons. Add when foundational primitive
  set expands.

## Done when

1. Both approaches implemented in `experiments/claim2_bridge/`.
2. Battery of ≥30 expressions covering the equivalence classes
   above.
3. Measurement report with preservation rates per equivalence class,
   per approach, plus collision rate on distinct expressions.
4. Written conclusion: which approach (if either) substantiates
   claim 2 in its current form, or which redesign is needed.
