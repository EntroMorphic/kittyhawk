# Nodes of Interest: P1-1 Primitives Floor

## Node 1: The "six" can be read two ways
Vision claim #1 says "all required compute math derives from ~6 frozen primitives (add, sub, exp, log, ...)." This is ambiguous:
- **Reading 1:** Six COMPUTATIONAL ATOMS (add, sub, mul, neg, max, min). exp/log are derivations of these. Path B is the operationalization.
- **Reading 2:** Six NAMED PRIMITIVES including transcendentals (add, sub, exp, log, plus two more). Substrate must offer them natively. Path A is the operationalization.

The user's "etc" leaves both readings possible. Path B has the substrate-discipline-respecting interpretation; Path A has the literal naming.
**Why it matters:** the choice between Paths is downstream of which reading we adopt. If Reading 1, Path B IS vision claim #1 made operational. If Reading 2, only Path A satisfies the claim.

## Node 2: Substrate-discipline rule was overridden once
`docs/REMEDIATION_PLAN.md` had a consumer-discovery cycle gating Tier 3. The owner overrode it. The override was later flagged (substrate-discipline cleanup found 9 sites of hand-rolled equivalents because the "consumer" never actually used the kernels). Repeating the override pattern here — building exp/log without measured consumer demand — would repeat the mistake.
**Why it matters:** there's project history of "build kernel, then discover no consumer needs it." P1-1 risks the same shape.

## Node 3: Path B is FALSIFIABLE cheaply
Build Taylor exp(x) as an expression tree (3-5 terms + range reduction). Add to a small bank. Test:
- Do different-depth Taylor truncations merge into one class? (the "expression-routing recognizes equivalence" test)
- Do `exp(x)*exp(y)` and `exp(x+y)` (both as Taylor trees) merge? (the natural mathematical equivalence test)
~1 week, very cheap relative to Path A's open-numerical-methods scope.
**Why it matters:** the cheap test distinguishes Path B's viability without committing to either path. Same fork-experiment pattern that just resolved R1.

## Node 4: Path A is OPEN (numerical methods)
Integer-only base-3 transcendentals don't have a known tie-free formulation. The substrate's odd-divisor lemma worked for cross-exponent accumulator because powers of 3 are odd; exp/log don't share that property. New substrate spec section, new precision contract, new tests. Weeks-to-months of real numerical-methods work.
**Why it matters:** Path A is genuinely hard. Worth doing only if Path B fails or if there's measured downstream demand.

## Node 5: The fork experiment partially answered R2 already
Random-bank class count for arity-1 plateaued at 27 (sign-only) and 38 (dual) regardless of sig_dim. The mechanism scales (banks build, signatures compute) but the equivalence-class count stops growing. Adding more candidates would push higher but probably still saturate.

The remaining R2 question isn't "does the mechanism scale?" — it's "does the equivalence relation reveal interesting structure at vocabulary-rich scale?" Path B's exp trees are a vocabulary expansion that tests this directly.
**Why it matters:** R2 isn't a separate cycle. It's a question that's partly answered and partly addressable through P1-1 Path B. Saves a redundant cycle.

## Node 6: Vision claim #3 should not motivate P1-1
The fork closeout speculated P1-1 might naturally use third-state semantics (e.g., "domain undefined" for log). But that's reaching — exp/log can be implemented (Path A or B) without ANY use of substrate-distinctive features. The "third state for domain undefined" is a possible design choice in Path A, not a requirement of either path.
**Why it matters:** motivating P1-1 by vision claim #3 puts P1-1 on the same rocky ground as R-track. Motivate by vision claim #1 directly: the foundation names exp/log; the substrate must offer them somehow.

## Node 7: Owner directive vs measured consumer demand
The user explicitly named exp/log in the foundation. That's an owner directive. The substrate-discipline rule normally requires measured consumer demand before adding a primitive. Owner directive can override discipline (precedent: Tier 3 override, with caveats).

The cleanest move: treat owner directive as the demand for STARTING the work, but use the substrate-discipline rule to gate the SCOPE of the work. Build the cheapest thing that satisfies the directive (Path B prototype). Only escalate to Path A if Path B doesn't suffice.
**Why it matters:** preserves discipline while honoring directive.

## Node 8: Deferring P1-1 isn't well-motivated either
The two reasons to defer:
- Wait for measured consumer demand → owner directive supersedes (Node 7)
- Do vision claim #2 scale experiment first → already partly done by fork (Node 5)

Neither holds up. P1-1 is a forward-moving cycle as long as we scope it cheaply.
**Why it matters:** "step back further" sounded right in RAW but doesn't actually have a target.

## Node 9: Path B's deep trees might saturate signatures
Taylor exp(x) at depth 5 with range reduction is a tree of ~30 nodes. Sign-only signature collapses all monotone-increasing expressions. Different Taylor truncations might all hash to the same signature ("monotone-positive on [0, 30]"). If they do, the bank can't tell them apart — Path B's premise breaks.

This is the fork-experiment lesson applied: maybe the consumer is also saturated for transcendental-shape expressions. The cheap Path B test would surface this immediately.
**Why it matters:** if Path B's signatures saturate, it tells us either (a) sign-only is fundamentally too coarse for transcendentals, demanding Path A; or (b) the test inputs are wrong for transcendentals (need denser sampling near zero where Taylor truncations differ); or (c) the bank shape needs revisiting for this kind of expression. All three are real outcomes worth knowing.

## Node 10: Range reduction is the design problem in Path B
Taylor for exp converges only near x=0. For x in [-30, 30], you need range reduction: e.g., exp(x) = exp(x/2)^2 = exp(x/4)^4. Each reduction doubles tree depth. To handle x=30 with reasonable Taylor convergence you might need 4-5 reductions → tree depth ~20 → very deep expression tree.

Range reduction in integer arithmetic without division is itself a design challenge. exp(x/2) requires computing x/2 — but our primitives don't have division. Maybe we need exp(x) = (exp(1))^x for integer x? That's a different decomposition.

Or: restrict the test inputs to small range where Taylor converges directly (e.g., [-3, 3]). Then exp tree is ~5 terms deep. Much cleaner test.
**Why it matters:** the Path B prototype's design isn't trivial. Several real choices — range, decomposition strategy, Taylor depth. Best handled in a focused design pass before implementation.

---

## Tension Summary

- **T1 (foundation interpretation):** Reading 1 (six computational atoms) vs Reading 2 (six named primitives including transcendentals). Picks Path B vs Path A.
- **T2 (discipline vs directive):** substrate-discipline rule says no primitive without consumer demand; owner directive names exp/log as foundation. Cleanest resolution: directive justifies starting; discipline gates scope (cheap Path B first).
- **T3 (saturation might apply to transcendentals too):** if Path B's deep trees saturate signatures the same way arity-1 did, the cheap test surfaces it and forces Path A or vocabulary redesign.
- **T4 (motivation framing):** vision claim #3 motivation for P1-1 is reaching; vision claim #1 motivation is direct. Pick claim #1.

## Dependencies

- **D1:** Path B prototype is the cheapest test that distinguishes Path B viability from Path A necessity.
- **D2:** Range reduction strategy in Path B needs a design choice; small-range Taylor is the simplest.
- **D3:** Owner reads of vision claim #1 (Reading 1 vs Reading 2) determines whether Path A is ever needed regardless of Path B's outcome.
