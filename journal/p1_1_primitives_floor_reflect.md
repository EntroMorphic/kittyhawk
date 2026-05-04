# Reflections: P1-1 Primitives Floor

## Core Insight

The same pattern that resolved R1 applies here: when uncertain between paths, run the cheapest test that distinguishes them. **Build a Path B prototype as the falsification test.** Don't pre-commit to Path A or Path B as the answer.

Path B prototype tests three things at once:

1. **Vision claim #1 falsification:** can exp/log actually be expressed from the existing six computational atoms (add, sub, mul, neg, max, min)? If yes, Reading 1 of the foundation is operational and Path A becomes optional. If no, Path A is necessary.

2. **Consumer saturation under transcendentals:** does the expression-routing consumer recognize different Taylor truncations as equivalent, or do they all collapse to "monotone-positive" signatures (the same saturation that arity-1 hit)? If recognized, vision claim #2's scope expands meaningfully. If saturated, the consumer's limits extend to transcendentals.

3. **Natural mathematical equivalences:** do `exp(x)*exp(y)` and `exp(x+y)` route to the same class? This is a richer test of vision claim #2 than the polynomial bank could provide.

A 1-week prototype that answers these three questions is the right scope. Path A becomes a follow-on cycle only if Path B reveals it's necessary.

## Resolved Tensions

**T1 (foundation interpretation) — RESOLVED by deferring to data.** Reading 1 vs Reading 2 is a question about the foundation's intent. Owner gets to decide eventually, but the data from a Path B prototype tells us whether Reading 1 is even FEASIBLE. If Path B works, Reading 1 is operational; the choice between readings is then down to owner preference. If Path B doesn't work, Reading 1 is impossible and Reading 2 is the only viable interpretation.

**T2 (discipline vs directive) — RESOLVED by scoping.** Owner directive (foundation names exp/log) justifies starting work. Substrate-discipline rule (no primitive without demand) gates the scope: build the cheapest thing that responds to the directive. Path B prototype is the cheapest. If it works, no substrate primitive is added — the directive is satisfied at the consumer layer.

**T3 (saturation might apply to transcendentals) — UNRESOLVED but FRAMED.** The Path B prototype is exactly the test that surfaces this. Three outcomes:
- Taylor variants merge cleanly → vision claim #2 extends to transcendentals → big win.
- Taylor variants merge but `exp(x)*exp(y) ≠ exp(x+y)` → equivalence-recognition fails for the natural identity → consumer needs richer signatures or different bank shape, not Path A specifically.
- Taylor variants don't merge → signatures are too rich, or test inputs are wrong, or transcendentals need a different signature rule entirely → revisit at that point.

**T4 (motivation framing) — RESOLVED.** Motivate P1-1 by vision claim #1 (the foundation directly names these primitives). Drop the vision claim #3 motivation that the fork closeout speculated. Vision claim #3's status is unchanged by P1-1 — it remains an open question independent of this work.

## Challenged Assumptions

**A1: "P1-1 must address vision claim #3 too."** False. The fork closeout speculated this; it's not actually required. P1-1 stands on vision claim #1 alone. If transcendentals happen to use third-state semantics (Path A's "domain undefined" idea), that's bonus, not necessary.

**A2: "Path B is necessarily slow."** Partially false. Slow per-evaluation, yes. But for routing purposes, evaluation happens once per signature build (setup time) and the routing itself uses the resulting signature. The slow-evaluation cost is paid in bank construction, not query time. Acceptable.

**A3: "Range reduction in integer arithmetic without division is intractable."** Partially false. We don't need general division — we need `exp(x)` for x in our test-input range. If we restrict the bank's test inputs to a range where Taylor converges directly (e.g., [-3, 3]), no range reduction needed. The test can be local; broader range is a follow-on.

**A4: "Defer P1-1 to do R2 first."** False. The fork experiment partially answered R2's central question (saturation is real, mechanism scales but equivalence classes plateau). The remaining R2 question (does vocabulary expansion produce richer structure) is exactly what the Path B prototype tests. Same cycle, different framing.

## What I Now Understand

**The minimum viable P1-1 is a Path B prototype, scoped tight.** Build:
- A small `exp_taylor(x, depth)` function that returns an expression tree for the depth-N Taylor series of exp.
- A bank that includes Taylor truncations at depth 3, depth 4, depth 5.
- A bank that also includes `exp_taylor(x) * exp_taylor(y)` and `exp_taylor(x+y)` (as separate trees that should merge).
- Run the routing. Measure equivalence-class structure.

If the prototype's measurements suggest Path A is needed (saturation, precision regimes that signatures can't capture), that becomes a separate cycle with its own design. If not, P1-1 is closed via Path B.

**The work belongs to vision claim #1, not vision claim #3.** Drop the third-state motivation. exp/log enter the system because the foundation names them, and Path B is the cheapest implementation that satisfies the naming.

**The substrate-discipline rule is honored by the prototype's scope.** No new substrate kernels until measured demand exists. Path B prototype generates that demand if it fails; otherwise the directive is satisfied at the consumer layer.

## Remaining Questions

- What test-input range maximizes the prototype's signal? Probably small (e.g., [-3, 3]) where Taylor converges without range reduction. But the project's existing test inputs are [-30, 30]; using two different ranges in different banks creates a per-arity-style fragmentation.
- How deep should the Taylor truncations go? Depth 3 (terms 1, x, x²/2, x³/6) is minimal. Depth 5 adds x⁴/24 and x⁵/120. The factorial denominators are integer-divisible only at small depths — beyond depth 5, integer division loses precision. Cap at 5.
- Should `log` be in the prototype too, or just `exp`? Newton iteration for log is more complex than Taylor for exp. Start with exp only; add log if Path B viable.
- What's the bank's test-input shape? If we use [-3, 3] with 7 inputs, sig_dim=7 — quite small. Could use 16 inputs in [-3, 3] (denser sampling). Decision for SYNTHESIZE.

## A note on the LMM cycle pattern

This is the second time in two cycles that the right answer was "build the cheap test that distinguishes paths." First the R1 fork experiment; now the P1-1 Path B prototype. The pattern:

- Don't pre-commit to a path when the data doesn't yet distinguish them.
- Build the cheapest version that produces falsifying data.
- Let the data pick the path.

This pattern should probably be lifted to project methodology: when a major design choice is contested between paths, the next cycle is "build the falsification test," not "design the chosen path."
