---
status: P0/P1 — owner directive 2026-05-03 (post-LMM revision)
authority: owner directive, derived from project vision claim #2
scope: close the gap between data-derived signatures and expression-derived signatures
supersedes: nothing — adds a new track parallel to existing P0 cycles
lmm_cycle: journal/expression_routing_{raw,nodes,reflect,synthesize}.md
---

# Plan — Expression Routing (closing vision claim #2)

## What this is

Vision claim #2: *all mathematics can be classified and expressed as signatures via routing over the frozen primitives.* Today the system can route a data query (an image, a synthetic sample) to a data-derived address. It cannot yet route a math expression to an expression-derived address. Every signature in the codebase comes from data, not from algebraic structure.

This plan adds the missing bridge — and adds it in the right shape, which is **equivalence-class lookup**, not a drop-in expression bank.

The bridge has six pieces. Four are **P0** — small, fast, falsifiable; they answer the central question (does expression-as-signature-via-routing work *at all*?) in roughly two weeks of focused work. Two are **P1** — only attempted if P0 passes, because they're hard and only earn their cost if the P0 probe survives contact with reality.

## Vision claim, recap

> Given that all required compute math derives from ~6 frozen primitives (add, sub, exp, log, …), all mathematics can be classified and expressed as signatures via routing over those primitives.

The operational consequence: training and lookup are the same shape regardless of whether you're classifying a digit or evaluating `exp(x+y)`. Both reduce to "compute the query's address; return what's nearest."

## The conceptual fix the LMM cycle produced

Pre-LMM framing: "build an expression bank, same shape as the data bank, sourced from expressions instead of samples."

Post-LMM framing: data signatures and expression signatures are different shapes. Data signatures are *learned representations of distributions over samples* (class-mean is an aggregate). Expression signatures are *defined representations of single deterministic evaluations* (one expression → one signature, no aggregation).

The right primitive isn't a bank substitute. It's **equivalence-class lookup**: given a query expression, return the simplest known expression that behaves identically on a fixed test-input set. The bank holds equivalence-class representatives. The label is the representative's identity. The forward path is 1-NN nearest-tile.

Routing-as-equivalence-recognition. Not "find the closest expression in the bank." But: "this query is a re-derivation of class C; here is the canonical representative."

## P0 — the smallest probe (the falsifiable test)

### P0-1: Signature derivation rule + test-input set

**The gap.** No code today computes a signature from an expression.

**What "fixed" looks like.** A function `expr_to_signature(out, expr, test_inputs, sig_dim)` that turns any expression into a packed-trit signature, with the property: two expressions that *behave identically on the test inputs* produce byte-equal signatures.

**Rule:** evaluate the expression at each test input, ternarize each output by sign at tau=0 (using the existing `m4t_route_threshold_extract`). Concatenate into a packed-trit signature. No structure-hash, no name-hash. Equivalence-on-test-inputs is the operational definition of expression equality.

**Test-input design.** Substrate-native MTFP values, symmetric around zero, including endpoints, mid-range, and points where simple expressions change sign. First pass: 16 single-variable test inputs `{-30, -15, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 15, 20, 30}` and 16 two-variable input pairs (a 4×4 grid over `{-10, -3, 3, 10}`).

**Sig_dim.** 16 for both arities (one trit per test input by construction). Sweep deferred — if discrimination is poor at 16, verdict gate surfaces it and we iterate.

**Substrate-discipline.** Ternarization through `m4t_route_threshold_extract`. No open-coded sign step.

### P0-2: Expression representation

**The gap.** No expression representation exists in libgesh.

**What "fixed" looks like.** A small expression tree type over the existing substrate primitives:

```c
typedef enum {
    EXPR_VAR, EXPR_CONST,
    EXPR_NEG,
    EXPR_ADD, EXPR_SUB, EXPR_MUL,
    EXPR_MAX, EXPR_MIN,
} expr_op_t;
```

Plus an evaluator that returns int64 (room for nested products on bounded inputs without overflow concerns).

**Anti-pattern.** Building a parser, pretty-printer, or simplifier. P0 doesn't need any of that. Hand-built C constructors are sufficient.

### P0-3: Equivalence-class bank constructor

**The gap.** No expression-derived bank exists.

**What "fixed" looks like.** A constructor that:
1. Takes a list of named candidate expressions.
2. Computes each candidate's signature via P0-1's rule.
3. Detects equivalence classes (signatures byte-equal).
4. Picks one representative per class (first-in-order = simplest by construction order).
5. Builds the bank tiles from representatives only.
6. Returns a candidate→class map for use by the probe.

The bank type carries arity as metadata. P0 builds two banks: arity-1 and arity-2.

**Starting candidate sets.** ~12 deliberately-chosen single-variable candidates and ~14 two-variable candidates, with intentional redundancy (e.g., `(x+y)*(x-y)` should equivalence-class to `x²-y²`; `min(x,y)+max(x,y)` to `x+y`). The bank constructor's job is to detect and merge these — proving the equivalence machinery works by construction.

**Substrate-discipline.** Packing through `m4t_pack_trits_1d`. No open-coded sign step.

### P0-4: Equivalence-recognition probe

**The gap.** No measurement exists that tests whether routing-as-equivalence-recognition works.

**What "fixed" looks like.** A benchmark binary `gesh_expr_routing_probe` that:

1. Builds the arity-1 and arity-2 banks via P0-3.
2. Constructs ~30 probe expressions per bank — syntactically distinct equivalents to bank candidates. Each probe has a hand-written "expected candidate."
3. For each probe: compute its signature, find nearest bank tile, compare to expected class.
4. Tally per-class hit rates and overall.
5. Apply the verdict gate.

**Pre-committed verdict gate.**

| Outcome | Verdict |
|---|---|
| ≥85% overall (≥51/60) AND every class with ≥1 probe has ≥1/3 of its probes correct | **PASS** — equivalence-as-routing works at toy scale. Proceed to P1. |
| 60–84% overall (36–50/60) | **WEAK** — signature rule needs iteration. Likely the test-input set or sig_dim. Don't proceed to P1. |
| ≤59% overall (≤35/60) | **FAIL** — behavior-based equivalence does not capture the equivalence the test set defines. Stop and rethink. |

The per-class floor in PASS prevents passing on easy classes while hard classes silently fail.

**Anti-pattern.** Tuning the bank, the test-input set, or the probes against probe results until the gate PASSes. The probes are the test; tweaking the test until you pass it is the failure mode the project's discipline rules were designed to prevent. WEAK iteration is allowed only against probes constructed *after* the iteration.

### P0 budget

~2 weeks of focused work:
- P0-1: ~1 week (signature rule design, test-input set, sig_dim sweep if needed)
- P0-2: 1–2 days
- P0-3: 2 days
- P0-4: 3 days (probe construction + analysis)

(The original 5-day estimate was overconfident; halve the optimism.)

---

## P1 — only if P0 passes

These are gated on P0-4 returning PASS. If P0 returns WEAK or FAIL, P1 is not the right next move; iterate on P0 instead.

### P1-1: Closing the primitives floor (exp, log)

**The gap.** Vision claim #1 names exp and log among the six frozen primitives. The substrate has add, sub, mul, neg, max, min, eq — none of exp, log, or anything that produces them.

**Two paths.**

**Path A: exp and log as substrate primitives.** Add `m4t_mtfp_exp` and `m4t_mtfp_log` kernels. Both operate on MTFP cells, both base-3 internally. Real numerical-methods work; integer-only base-3 transcendentals aren't a solved problem. Likely involves a precomputed base-3 lookup table for small-magnitude inputs, range reduction via the natural MTFP form (`m × 3^k`), and a new substrate spec section formalizing the precision contract.

**Path B: exp and log as compositions.** Show that `exp(x)` can be computed as a finite composition of the existing six (e.g., a Taylor-series-like expansion the way old hardware computed `exp` before transcendental units existed). Same for `log` via Newton iteration. The "primitive" exp/log become specific expression trees, not new substrate kernels. Keeps the floor literally six. Costs: every exp/log evaluation is many primitive ops (slow); precision bounded by term count.

**Verification.** The expression `exp(x+y)` enters the bank, and a probe expression like `exp(x)*exp(y)` routes to it. (Mathematically identical; the bank had better see them as the same address.)

**Risk.** May force a substrate spec amendment.

### P1-2: Unified address space (data tasks AND expression tasks)

**The gap.** Today data signatures and (post-P0) expression signatures live in trit space, but no claim or test that they live in the *same* trit space. Per-arity expression banks (P0 scope) and data banks at sig_dim=64+ are not even comparable.

**What "fixed" looks like.** Both encoders (data → signature, expression → signature) produce signatures in a shared trit space, with the property that a query of either kind can route into a bank that contains tiles of either kind.

**This is the deep claim.** If it works, the system is doing something no other ML system does: a single address space where data identity and computational identity coexist. If it doesn't work, the project is "small ML system + interesting numerical kernels" — still valuable, but not the original vision.

**Verification.** A unified-space probe (final shape designed during the cycle): hand the system an image of a digit AND the expression `digit_value(image)` separately; verify they share an address. Or: hand it `2+3` and an image of "5" separately; verify they share an address.

**Risk.** May not survive contact with reality. Acceptable failure mode: P0-4 PASS + P1-1 lands + P1-2 fails means routing-as-equivalence works in expression-only world; unified-space ambition was a step too far. Document and move on.

---

## What kills this plan

- **P0-4 returns FAIL** (≤35/60 probes correct). Behavior-based signatures don't carry meaning at this scale. Either the test-input set is wrong, sig_dim is too small, or the equivalence rule itself is wrong. Triage in that order; if all three fail to recover, vision claim #2 needs revision before more substrate work is justified.
- **P1-1 cannot produce a tie-free precision contract for exp/log.** If the substrate's odd-divisor rounding discipline doesn't extend to transcendentals, the substrate has a real numerical limitation that the rest of the plan would have to work around. Worth surfacing if it happens.
- **P1-2 architecture cycle reveals data and expression signatures can't share a space without one of them losing essential structure.** The vision then has two incompatible domains, not one unified one. Not fatal; just smaller than the original claim.

## What this plan deliberately does NOT do

- Does not build a parser, pretty-printer, or expression simplifier. Hand-built C constructors are sufficient for the probe.
- Does not include constants in P0 banks. Constants give degenerate all-+1 or all-(-1) signatures and need a different signature derivation. Punt to P1 if needed.
- Does not attempt cross-arity routing in P0. Per-arity banks; cross-arity is P1-2.
- Does not attempt unified data-and-expression routing in P0. Saved for P1-2.
- Does not extend the substrate without measured demand. P1-1 only happens because P0-4 passing creates the demand.
- Does not retire any existing P0 cycle (P0-1 through P0-4 of the prior remediation plan). This is a parallel track.

## Notes on substrate-discipline

- All new code routes through libm4t kernels for ternarization, packing, distance. No hand-rolled MAC, no hand-rolled sign-threshold (the rule from the substrate-discipline cleanup applies to expression-evaluator code too).
- All new code under `-Werror` with the project's standard warning flags.
- All new tests under ctest.
- The CHANGELOG entry for each P0 piece lands with the work, not later.
