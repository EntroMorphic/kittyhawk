# Synthesis: P1-1 Primitives Floor — Path B Prototype as Falsification Test

## Architecture

**P1-1 starts with a Path B prototype, not a path commitment.** Build Taylor truncations of `exp(x)` as expression trees, route them through the existing expression-routing consumer, and measure whether vision claim #1 (six-atoms expressibility) is operational.

Path A (substrate primitive kernels) is held in reserve and only invoked if the prototype reveals it's necessary — same pattern as the R1 fork experiment that resolved the prior cycle.

`log` is deferred until `exp` Path B viability is established.

## Key Decisions

**D1: Path B prototype is the next cycle, not a Path commitment.** [from REFLECT core insight]
The cheapest test that distinguishes "exp is expressible from six" from "exp needs to be primitive" is to BUILD the expressible version and measure. Same falsification-first discipline that just resolved R1.

**D2: Vision claim #1 is the motivation; drop vision claim #3.** [REFLECT T4]
exp/log enter because the foundation names them. Path B respects the substrate-discipline rule by adding no new substrate primitive until measured demand appears.

**D3: Small test-input range for the prototype.** [REFLECT remaining question 1]
Use test inputs in [-3, 3] where Taylor converges directly. Avoids the range-reduction design problem entirely. 16 inputs densely sampled in this range: `{-3, -2, -1.5 (rounded to -2), ...}` — actually all integer for substrate-discipline. Plan: `{-3, -3, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 3, 3}` is too duplicate. Better: re-purpose the curated 16 inputs but restrict by value: `{-3, -2, -1, 0, 1, 2, 3}` (7 inputs) → small sig_dim, but enough to test discrimination.

Or use 16 inputs spanning [-5, 5] integer, which gives `{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}` (11 inputs) padded to 16 with repeats or duplicates. Cleaner: 16 inputs with some at fractional positions encoded as MTFP (no, the substrate is integer). Cleanest: `{-5, -4, -3, -3, -2, -2, -1, -1, 0, 1, 1, 2, 2, 3, 3, 5}` — 16 inputs, denser near zero, integer-only, range [-5, 5].

Final decision: 16 inputs `{-5, -4, -3, -3, -2, -2, -1, -1, 0, 1, 1, 2, 2, 3, 3, 5}`. The duplicates at 1, 2, 3 are intentional — the signature derives a trit per input position, so duplicate inputs produce duplicate trits (consistency check on signature stability).

Actually the duplicates are wasteful — they always agree by construction. Better: 16 distinct inputs in [-5, 5] including a mix: `{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}` is 11 distinct, then 5 more: `{0, 1, 2, 3, 4}` repeats. No.

Simplest defensible: 11 distinct inputs in [-5, 5]. sig_dim = 11. Yes the existing kernel handles non-multiple-of-4 sig_dim (the mask code handles tail). Sig_dim=11, packed into 3 bytes (11 trits = 22 bits = 3 bytes). Acceptable.

Going with sig_dim=11, inputs `{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}`.

**D4: Taylor depths for the prototype:** depths 3, 4, 5. Depth 3: `1 + x + x²/2 + x³/6` (denominators 1, 1, 2, 6 — integer-divisible). Depth 4: adds `x⁴/24`. Depth 5: adds `x⁵/120`. All denominators integer-divisible at small depth. Beyond depth 5, denominators (720, 5040, ...) start producing precision loss in integer arithmetic.

Note: integer division loses precision. `x³/6` at x=2 gives 8/6 = 1 (truncated). At x=5 gives 125/6 = 20. The Taylor truncation isn't exact; it's an integer-rounded approximation. Document this.

**D5: Bank composition for the prototype:**
- `exp_taylor(x, depth=3)` (= 1 + x + x²/2 + x³/6)
- `exp_taylor(x, depth=4)`
- `exp_taylor(x, depth=5)`
- `exp_taylor(x*2, depth=3)` — different argument
- `exp_taylor(-x, depth=3)` — should NOT merge with positive-x Taylor (different sign profile)
- `exp_taylor(x+1, depth=3)` — shifted argument

That's 6 candidates. Add 2-variable expressions:
- `exp_taylor(x+y, depth=3)` (computed as one tree)
- `exp_taylor(x, 3) * exp_taylor(y, 3)` (product of two single-variable Taylor trees)

If the consumer recognizes vision claim #2's natural equivalence, these last two should land in the same equivalence class.

Total: 6 arity-1 candidates + 2 arity-2 candidates = 8 in a single bank, or per-arity. For simplicity, two banks (one per arity).

**D6: Pre-committed gates:**
- **G1 (depth-equivalence):** Taylor depths 3, 4, 5 of `exp(x)` should merge into a single equivalence class under sign-only signature on [-5, 5] inputs. PASS if all three are in the same class.
- **G2 (natural mathematical equivalence):** `exp_taylor(x+y, 3)` and `exp_taylor(x, 3) * exp_taylor(y, 3)` should merge under sign-only signature on the arity-2 grid. PASS if same class.
- **G3 (sign-distinct):** `exp_taylor(x, 3)` and `exp_taylor(-x, 3)` should NOT merge (they have different sign profiles for x outside [-1, 1]). PASS if different classes.

A P1-1 PASS = G1 AND G2 AND G3. WEAK = 2 of 3. FAIL = ≤1 of 3 OR G3 fails (suggests signatures don't even discriminate sign of input).

**D7: Path A is held in reserve.** Only proceed to Path A if:
- G1 FAILs (Taylor truncations don't merge → consumer can't recognize exp-equivalence at this scale → richer signatures might help OR substrate primitives might be needed)
- G2 FAILs (natural equivalence not recognized → fundamental issue with Path B's expressibility claim)

A G3 FAIL would mean signatures are too coarse to distinguish even the SIGN of an exp argument — that's a sign-only-rule problem, not a Path A problem.

## Implementation Spec

### Files

- `gesh/src/expr_taylor.{h,c}` — `expr_t* expr_exp_taylor(expr_t* arg, int depth)` constructor. Builds the Taylor expansion tree.
- `gesh/bench/expr_routing_p1_1.c` — prototype binary that builds the candidate set, routes them, applies G1/G2/G3 gates.
- CMakeLists target `gesh_expr_routing_p1_1`.

### `expr_exp_taylor` algorithm

```
expr_exp_taylor(arg, depth):
    result = K(1)                      // constant term: 1
    if depth >= 1:
        result = expr_add(result, copy(arg))           // + x
    if depth >= 2:
        x2 = expr_mul(copy(arg), copy(arg))            // x²
        // x²/2: integer divide. Implement via subtree shift: x*x is even-ish for even x; truncate.
        // Substrate has no division. Workaround: build x*x once and accept it represents x²
        // (not divided). The "Taylor" here is approximate by integer constants.
        // Alternative: encode the / 2 by repeating: x²/2 ≈ x² subject to floor.
        // Simplest hand-build: just use x², x³, x⁴, x⁵ as terms with their integer-rounded
        // denominators baked into the LOOK of the tree, not the math.
        ...
```

Wait — substrate has no division. Need to be honest:

Path B's "Taylor" is necessarily approximate in integer arithmetic. Two options:

**Option B.1: integer Taylor with truncation.** Each term's denominator is folded into the integer arithmetic. `x² / 2` becomes some integer-arithmetic equivalent — perhaps just `(x*x) >> 1` if we had shift, or we accept that we can only approximate. Without division, we can compute `x²` exactly but can't divide by 2 in the substrate. So the "Taylor" isn't really Taylor — it's `1 + x + x² + x³ + ...` which is `1/(1-x)` (the geometric series), not `exp(x)`. Different function.

**Option B.2: pre-multiply to avoid division.** `exp(x) ≈ (1 + x/n)^n` for large n. Or work in log space. Both add complexity.

**Option B.3: re-frame the prototype.** Don't claim to compute exp; instead claim to compute "an exp-shaped expression family." Use `(1+x)^n` (via repeated multiplication) as the "exp-like" function. This IS expressible from six primitives. It has the right qualitative shape (monotone increasing, super-linear growth for x > 0, decay-toward-zero for x < 0). Different from true exp but tests the same vision-claim-#1 question.

Going with **Option B.3.** The prototype's "exp-like" function is `(1+x)^n` for various n. Specifically: `power(1+x, 3)`, `power(1+x, 4)`, `power(1+x, 5)` where `power(base, n)` is `base * base * ... * base` (n copies). All built from add and mul.

This shifts the gates' interpretation:
- G1: `power(1+x, 3)`, `power(1+x, 4)`, `power(1+x, 5)` should NOT merge (they're different functions). Adjust to: do these three produce DIFFERENT classes (proving signatures discriminate)? Or do they all merge into "monotone-increasing" (proving saturation)?
- G2: `power(1+x+y, 3)` and `power(1+x, 3) * power(1+y, 3)` should NOT merge (mathematically different unless x=0 or y=0). The natural-equivalence test changes shape: maybe `power(2x, 3)` ≡ `8 * power(x, 3)` should hold by sign? Yes — both are sign(x)³ = sign(x) at every input.
- G3: `power(1+x, 3)` and `power(1-x, 3)` should produce different sign profiles (positive for 1+x>0 vs 1-x>0).

This isn't quite the Taylor-exp test we wanted, but it's still a vision-claim-#1 test: can complex functions like `(1+x)^n` be expressed AND DISCRIMINATED by the routing system?

Actually — this is reframing the prototype away from exp/log specifically. That changes what's being tested. Vision claim #1 names exp/log explicitly. `power(base, n)` isn't exp/log.

Honest acknowledgment: **without division, Path B cannot express true exp/log in integer arithmetic.** The "Taylor truncation" framing was wrong. Path B needs to be either:
- Accept that exp/log require Path A (substrate primitive with built-in scaling/division)
- Or work with whatever transcendental-shaped functions ARE expressible (powers, polynomials, monotone shapes)

This is a real finding from the SYNTHESIZE process. The cheap-test plan needs revision before code is written.

## Revised SYNTHESIZE conclusion

**The original Path B prototype design has a hidden problem: integer arithmetic without division can't express true Taylor exp/log.** The substrate offers no `/` operation. Range reduction (`exp(x/2)^2`) requires division too.

Three honest options now:

**Option Y: Path B prototype with `(1+x)^n` instead of exp.** Tests vision-claim-#1 expressibility for transcendental-SHAPED functions, but doesn't actually implement exp/log. ~3 days.

**Option Z: Path A from the start.** Add substrate primitives `m4t_mtfp_exp` and `m4t_mtfp_log` with documented precision contract and integer arithmetic shortcuts (e.g., lookup tables for small ranges, range reduction inside the kernel via shifts that are substrate-internal not consumer-exposed). 2-4 weeks.

**Option Ω: Defer the exp/log work; address division as a separate question first.** Vision claim #1's "all required compute math from six primitives" probably implies division IS one of those six (or derivable). Currently we have add, sub, mul, neg, max, min — division is not in the substrate either. If division is needed for exp/log, then division is the first primitive to add, not exp/log directly. ~design cycle worth.

The synthesize's earlier framing assumed division was a non-issue. SYNTHESIZE's pre-commit-and-discover discipline just surfaced that this assumption was wrong. **Proper LMM discipline says: loop back to RAW with this finding rather than ship a broken plan.**

## Loop-back trigger fired

Per the LMM rules from `LMM.md`:

> **Back to RAW:** State the real problem in one sentence, then continue.

The real problem in one sentence: **the substrate has no division operation, so true exp/log are not buildable from the existing six computational atoms — vision claim #1's expressibility hinges on whether division is in the six or not.**

Per discipline, the next cycle is a re-RAW that grapples with: is division a missing primitive? Is it derivable from the existing six? Or does vision claim #1's "six" actually mean "six families including division and transcendentals as primitives"?

**This SYNTHESIZE doesn't ship a plan; it ships a deferral.** The honest conclusion of the cycle is "we discovered the problem isn't where we thought it was; loop back to RAW with a better-framed question."

## What ships from this cycle

Not code. The four LMM artifacts (raw, nodes, reflect, synthesize) document a problem-reframing: P1-1 isn't about exp/log first, it's about division (and possibly other unstated primitives) first. This is the right kind of LMM output when the cycle reveals the question was wrong.

## Recommended next move

A new cycle: `journal/primitives_floor_division_*.md` that asks the prior question:

> What does "six frozen primitives" actually mean? Is division one of them? Is it derivable? If derivable, from what? If not derivable and not in the current six, the substrate is short of vision claim #1 by one primitive and that gap should be closed before exp/log work begins.

Owner gets to decide: re-frame to division question, or push forward to Path A (with kernel-internal division as substrate-private detail), or defer P1-1 entirely and pivot to a different track.

The honest end of this cycle is: **the question we thought we were answering wasn't the real question.** The cheap-test discipline surfaced that within the synthesize, before code shipped.
