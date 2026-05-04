# Nodes of Interest: Expression Routing

## Node 1: Signature rule depends entirely on the test-input set
The behavior-based signature is "evaluate at fixed inputs, ternarize." The choice of inputs is the choice of the lens. Bad inputs → false equivalences (collapse) or false distinctions (brittleness). I designed the inputs in minutes; the design IS the heart of the rule.
**Why it matters:** the success or failure of P0-4 hinges on this design choice, not on any other engineering. Underbudgeting design here propagates.

## Node 2: Bank shape mismatch between data and expressions
Data bank: tile signatures + class labels, used by forward to VOTE among top-k. Expression bank: would need expression IDs as labels; vote is meaningless for expressions. The existing bank type doesn't fit the expression case naively.
**Why it matters:** P0-3 as written assumes drop-in compatibility. Either invent a new bank type or repurpose the existing one with different label semantics.

## Node 3: No expression analog for class-mean
Data signatures derive from averaging samples per class. An expression has one deterministic evaluation, not a distribution. There's no "average expression" that maps onto class-mean.
**Why it matters:** the symmetry I assumed between data-signatures and expression-signatures doesn't hold. The two are differently-shaped objects living in (possibly) the same trit space.

## Node 4: Equivalence vs. discrimination tension (T1)
Larger sig_dim → more discriminative power but also more brittleness (two equivalent expressions must agree on more positions to share an address). Smaller sig_dim → more collapse (different expressions get the same address).
**Why it matters:** there is no universal right answer for sig_dim; the right value depends on what "equivalence" means in the system. Without an explicit equivalence rule, sig_dim is arbitrary.

## Node 5: My probe is trivial-by-construction
I proposed "does `x*x` route to `x²`." But in the expression representation `x²` doesn't exist as a distinct op — it would BE `EXPR_MUL(x, x)`, which equals `x*x` literally. The probe tests nothing.
**Why it matters:** I designed a test that doesn't test the claim. The actual question is "does a SYNTACTICALLY DIFFERENT but SEMANTICALLY EQUIVALENT expression route to the same address." Different probe shape needed.

## Node 6: Arity creates incomparable signature spaces
1-var expressions evaluated at 16 inputs → 16-trit signature. 2-var expressions evaluated at 16 input pairs → also 16 trits, but the inputs are differently shaped. A 1-var sig and a 2-var sig live in formally the same trit space but are not semantically comparable.
**Why it matters:** cross-arity routing — a key part of vision claim #2 — silently breaks if not addressed. P0 should scope to single arity; P1 handles unification.

## Node 7: Compose-equivalence vs. cost-blindness (T4)
If `exp(x)` (a P1-1 primitive or compose-tree) and its 5-term Taylor truncation evaluate to the same MTFP value at every test input, they get the same signature. Routing returns one address for both. The system can no longer distinguish "exact" from "approximate" by address alone.
**Why it matters:** this is either elegance (equivalence-by-behavior is what we want) or a hidden cost (we can't track precision tradeoffs in routing). Probably elegance, but worth flagging.

## Node 8: The verdict gate was unjustified
I picked ≥7/10 as PASS with no derivation. The project's other PASS gates carry actual statistical reasoning (paired t-CI with t* matching df, multi-seed). My gate was placeholder.
**Why it matters:** weak gate → weak conclusion, regardless of probe outcome. Need a defensible gate.

## Node 9: 5-day budget is suspect
P0-1 design alone could fill days. I estimated like an engineer who already knew the design. Real work probably 2-3× longer.
**Why it matters:** budget tells the user what to expect. Wrong budget → wrong expectations → cycle pressure.

## Node 10: Constants degenerate the signature
A constant expression evaluates to the same value at every input → all-+1 or all-(-1) signature. Useless for routing.
**Why it matters:** either constants are excluded from the bank (loses generality), or they need a different signature derivation (special case in the rule), or signature dim is augmented with non-evaluation features (complexity creep). All three are real options; none is obvious.

---

## Tension Summary

- **T1 (Node 4):** equivalence vs. discrimination — solved by an explicit equivalence rule, not by sig_dim tuning.
- **T2 (Nodes 2, 3):** bank/vote shape inherited from data doesn't apply to expressions — need a new lookup primitive, OR reframe what "label" means.
- **T3 (Node 6):** cross-arity routing breaks naive shared-signature-space — P0 must scope to single arity.
- **T4 (Node 7):** compose-equivalence is feature OR hidden cost — depends on whether system needs cost-awareness in routing.
- **T5 (Node 10):** constants are degenerate under the proposed rule — need special handling or principled exclusion.

## Dependencies

- **D1:** Probe design (Node 5) depends on signature-rule design (Node 1).
- **D2:** Bank type (Node 2) depends on what equivalence means (Node 4).
- **D3:** P1-2 (unified space) silently depends on P0 making consistent arity choices (Node 6).
- **D4:** Verdict gate (Node 8) depends on probe size and equivalence-class structure (Nodes 4, 5).
