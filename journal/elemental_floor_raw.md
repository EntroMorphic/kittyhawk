# Raw Thoughts: Elemental Floor Audit

## Stream of Consciousness

Reframing from the prior cycle: the question isn't "add exp/log to the substrate." The question is "what's actually elemental and what's missing from the substrate's elemental floor."

In the conversation that prompted this cycle I argued the floor is 4–5 operations:

1. **add** — combine two values
2. **neg** — additive inverse
3. **shift3** — multiply or divide by the radix (3)
4. **sign** — read state of a value as a trit
5. **select** — route one of three inputs based on a control trit

Or 4 if we fuse sign+select into "sign-mux" (read sign and route in one op).

Now the doubts I want to surface in RAW.

**Doubt 1: Is select really atomic, or can it be built from arithmetic?**

For trit-control select(c, a, b, d):
- Want: c=+1 → a, c=-1 → b, c=0 → d
- Arithmetic attempt: result = a*(c==+1) + b*(c==-1) + d*(c==0)
- But (c==+1), (c==-1), (c==0) are themselves comparisons. Each is `c == constant`, which is `sign(c - constant) == 0`, which is itself a comparison.
- Or: write it as `result = (a*c*c + a*c)/2 + (b*c*c - b*c)/2 + d*(1 - c*c)` for trit c
  - At c=+1: (a + a)/2 + (b - b)/2 + 0 = a ✓
  - At c=-1: (a - a)/2 + (b + b)/2 + 0 = b ✓
  - At c=0: 0 + 0 + d = d ✓
- So algebraically: select(c, a, b, d) = (a + b)*c²/2 + (a - b)*c/2 + d*(1 - c²)
- This needs: mul, add, sub, division-by-2.
- mul is composite. div-by-2 is composite (well, in ternary, /2 isn't even clean — 2 isn't a power of 3).

So select might be derivable from {add, mul, div}, but mul and div are composite from {add, neg, shift3, sign, select}. **Circular.** Without select (or some equivalent branching primitive), you can't build mul, and without mul you can't build select.

**Conclusion: select is genuinely irreducible.** Or some equivalent branching primitive must exist.

**Doubt 2: Is neg necessary, or is the constant -1 enough?**

If we have +1 as a constant and add as primitive, we can build all positive integers (1, 2, 3, ...). To get to -1 or any negative number, we need EITHER:
- neg as a primitive
- sub as a primitive (but sub = add + neg, so this is just neg-disguised)
- The constant -1 plus a way to get from +1 to -1
- Some other operation that flips sign

If -1 is a constant and we have add, we can compute -1, -2, -3, ... by repeated add (since add(-1, -1) = -2, etc.). So with constants {+1, -1} and add, we have all integers without needing neg.

But: in a substrate where constants are stored values, having both +1 and -1 as constants is fine. Then neg(x) is computable as: well, how? We can't build neg from constants alone. We'd need to do something like: select(sign(x), x_negated_value, 0, x_negated_value). But that needs neg already.

Alternative: neg(x) = mul(-1, x). But mul is composite. So neg = composite of select + add + shift3? Let's see:
- mul(-1, x) where x is a cell: trit-by-trit, multiply each trit by -1, which is "swap +1 and -1, leave 0 alone."
- That's a TBL operation on each trit. Equivalent to a select per trit.
- So neg(x) = trit-wise select per trit position, controlled by the value's bit pattern.
- Which is just select applied to each trit.

So neg IS derivable from select (applied trit-wise). With select + the constant -1 + add, you have neg.

**Hmm. So maybe neg is NOT in the elemental floor.** Maybe the floor is:
1. add
2. shift3
3. sign
4. select
5. The constants {+1, -1} (not operations but available values)

And neg is derived from select.

**Doubt 3: What about sign — is it derivable?**

sign(x) where x is a cell: returns -1, 0, or +1.
- This is essentially "look at x's value and classify into three buckets."
- sign(x) = select(x_is_zero, 0, select(x_is_negative, -1, +1, 0), 0)
- But "x_is_zero" and "x_is_negative" are themselves sign-extractions. Circular.

Can sign be expressed arithmetically? Like: sign(x) = some formula in x using only add/shift?
- For x in [-N, N], sign(x) = x / |x| if x ≠ 0, 0 otherwise.
- Division and abs are both composite.
- For trit x: sign(x) = x (the trit IS its sign).
- For multi-trit cell x: sign needs to look at the highest-magnitude trit. That's a positional read.

**For multi-trit cells, sign requires reading the highest non-zero trit.** This is a kind of "find leading nonzero" operation. Is THAT elemental, or composite?

- Find leading nonzero = iterate from highest position, return position of first nonzero.
- That's iteration over trit positions with a sign-test at each.
- So find-leading-nonzero is composite from {sign-test-on-trit, iteration}.

But sign-test-on-trit is just reading the trit's value (which is its sign at the trit level). So:

**At the trit level, sign(t) = t.** It's a no-op; the trit IS its sign.

**At the cell level, sign(cell) = sign of the highest-magnitude trit, which requires iteration to find.** Composite.

So **sign at the cell level is composite** if you accept iteration. Sign at the trit level is trivial (it's just the trit's value).

What's left elemental at the cell level?
- add: yes
- shift3: yes (positional shift)
- select: yes (mux)
- neg: derivable from select per trit (apply trit-neg to each trit, but trit-neg = select)

Wait, trit-level neg is just "swap +1 and -1, leave 0." That's:
- neg_trit(t) = select(t_is_+1, -1, +1, 0)
- Which uses select with sign-of-trit (but t IS its sign at trit level, so sign is free).
- So trit-neg is derivable from select alone (with trit values as both control and outputs).

So at the trit level: just **add, shift3, select** might suffice. With trit-add, trit-shift, and trit-select, plus iteration over trit positions, you can build all cell-level operations.

That's THREE elemental ops. Even smaller than I thought.

But wait — trit-add returns a 2-trit result (carry). So trit-add isn't quite a closed operation on trits alone. You need to handle the carry, which means... adding more trits, which means add-with-carry, which is its own circuit.

Hmm. Trit-level add: a + b for trits a, b returns a value in {-2, -1, 0, +1, +2}, which doesn't fit in one trit. So trit-add either:
- Saturates (m4t_trit_sat_add does this — returns clamped)
- Returns multi-trit (carry-propagating add)

Saturating add loses information. Carry-propagating add is fundamentally a multi-trit operation (you need to track carries across positions).

So at the trit level, "add" is really "saturating add" or "add-with-carry" — not a clean single-trit op.

For cell-level arithmetic, add is well-defined. And cell-level add is built from trit-level add-with-carry, which is built from trit-level select + trit-level add-modulo-3 + carry-propagation.

This is getting deep. Let me step back.

**The right level of abstraction for the elemental floor question is the CELL level**, not the trit level. We're asking what cell-level ops are atomic, given that the substrate's primitive arithmetic unit is the cell (MTFP4, MTFP19, etc.).

At the cell level:
- **add(a, b)** = cell-level addition. Atomic; the substrate has it.
- **neg(a)** = cell-level negation. Atomic; the substrate has it (or can build it from saturated mul-by-(-1)).
- **shift3(a, k)** = multiply by 3^k. Atomic positional shift. Substrate has this implicit in MTFP per-block exponent but not as a runtime op.
- **sign(a)** = read sign of cell. Atomic read; substrate has m4t_route_threshold_extract at tau=0.
- **select(c, a, b, d)** = route based on control trit. Atomic mux; substrate has this in fragments (m4t_route_apply_signed dispatches based on sign decisions).

**The elemental floor at cell level: 4-5 ops.**

**Doubt 4: Are there ops I'm completely missing?**

Things that aren't arithmetic but might be needed for a complete substrate:
- **Load/store** (data movement) — yes, but those are memory ops, not arithmetic
- **Iteration/loop control** — program structure, not an op
- **Branching control flow** — program structure
- **Constants** — values, not ops
- **I/O** — outside scope

What about COMPARISON between cells (a < b)? 
- a < b ⟺ sign(a - b) = -1
- Composite from {sub, sign} where sub = add + neg.
- Composite, not elemental.

What about MIN of two cells?
- min(a, b) = select(sign(a - b), b, a, a)  // if a-b < 0, return a; else return b (with =a on tie)
- Wait: sign(a-b) = -1 means a < b, so return a. sign(a-b) = +1 means a > b, return b. sign(a-b) = 0 means equal, return either.
- Composite from {sub, sign, select}.

What about MUL by a constant?
- mul(a, k) for known k: build via repeated add and shift3.
- Composite.

What about absolute value?
- abs(a) = select(sign(a), a, neg(a), 0)
- Composite from {sign, select, neg}.

OK I think nothing's missing. The 4-5 ops are it.

## Questions Arising

- Is the right level "trit" or "cell" for the elemental analysis? I argued cell, but should justify.
- Is shift3 really atomic or is it a special case of "shift by k positions" for arbitrary k?
- What's the cleanest API for select at the cell level (it has 3 input cells + 1 control trit + 1 output cell)?
- Does the substrate need both single-trit-control select AND signature-controlled select (which routes based on the sign of a value)?
- How much do we collapse: keep sign and select separate, or fuse into sign-mux?

## First Instincts (suspect; to be challenged)

- The elemental floor is 4-5 cell-level ops: add, neg, shift3, sign, select.
- The substrate has add and neg already; missing shift3 (as a runtime op) and select (as a clean primitive).
- Existing composite kernels (mul, max, min, eq, sub, ternary_matmul) stay for performance but should be documented as composite.
- exp, log, sin, cos, sqrt are consumer-level constructions, not substrate ops.

## Risks I See

- Picking the wrong level of abstraction (trit vs cell).
- Defining shift3 in a way that's not actually general enough (e.g., only by ±1 power vs by arbitrary k).
- Over-engineering select at the cell level when the substrate's existing patterns might already cover it.
- Treating composite kernels as elemental in code or docs after this audit.
