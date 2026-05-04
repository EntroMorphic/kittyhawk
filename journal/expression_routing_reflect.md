# Reflections: Expression Routing

## Core Insight

The original plan treated expression-signatures as a drop-in substitute for data-signatures. They aren't substitutes — they're different shapes living (possibly) in the same trit space.

**Data signatures are LEARNED REPRESENTATIONS of distributions over samples.** A class-mean tile is the system's best guess at "what does this class look like in this trit space." It's an aggregate.

**Expression signatures are DEFINED REPRESENTATIONS of single deterministic evaluations.** An expression's tile is computed exactly from the expression's behavior on the test inputs. There's no aggregation, no learning, no distribution.

The original plan tried to bend the first into the shape of the second. The right move is to build the right primitive for the second case.

The right primitive is **equivalence-class lookup**: given a query expression, return the simplest known expression that behaves identically on the test-input set. This single reframing resolves most of the tensions surfaced in NODES.

The bank holds equivalence-class REPRESENTATIVES, not arbitrary expressions. Each tile is one equivalence class; the label is the representative's identity. The forward path returns "which equivalence class does this query belong to," which IS what the routing thesis claims it can do.

This is what vision claim #2 actually means in operational form: **routing-as-equivalence-recognition**. Not "find the closest expression in the bank." Not "vote among similar expressions." But: "this query is a re-derivation of class C; here is the canonical representative of C."

## Resolved Tensions

**T1 (equivalence vs. discrimination, Node 4) — RESOLVED.** The equivalence-class framing dissolves the tension. The signature only needs to discriminate equivalence classes from each other. WITHIN a class, the signature SHOULD collapse — that's the whole point. Smaller sig_dim is fine if equivalence classes are well-separated by the test-input set. Sig_dim is a tunable for class-separation, not for expression-uniqueness.

**T2 (bank/vote shape, Nodes 2-3) — RESOLVED.** Reframe labels. Each bank tile = one equivalence class; each label = the representative's identity. The "vote" of forward becomes a 1-NN lookup (top_k=1), which is just nearest-tile. The existing `gesh_bank_t` works with this interpretation; only the constructor differs and the label semantics change. No new primitive needed — just the right consumption.

**T3 (cross-arity, Node 6) — RESOLVED by scoping.** Per-arity banks in P0. 1-var expressions in one bank, 2-var in another. Each bank stores arity as metadata. P1-2 (unified space) addresses cross-arity by either projecting into a shared low-dim core OR by meta-routing on arity first. Don't pretend P0 solves this; explicitly scope it out.

**T4 (compose-equivalence, Node 7) — REFRAMED as feature, not bug.** Equivalence-by-behavior IS the operational definition of "function equality" in this system. Routing `exp(x)` and its Taylor truncation to the same address is the system correctly recognizing they implement the same function on the test set. Cost-awareness in routing is a separate concern, properly scoped to P1 if it's needed at all. (Probably not needed; cost can be carried as metadata on the representative.)

**T5 (constants degenerate, Node 10) — RESOLVED by separation.** Constants live in a separate arity-0 bank with their own signature rule (just the constant's MTFP value packed). They don't pollute the variable-expression banks. Three banks at minimum: arity-0 (constants), arity-1 (single-variable), arity-2 (two-variable). Cross-bank routing is P1.

## Challenged Assumptions

**A1 (original plan: bank shape matches data bank).** False. Label-as-class-id semantics are wrong; should be label-as-equivalence-class-representative. Small fix in framing, large fix in what the probe measures.

**A2 (original probe: "does X route to Y").** Wrong question. The probe should test "do syntactically distinct but semantically equivalent expressions converge to the same equivalence-class address." Probe expressions must be syntactically DIVERSE equivalents to bank representatives, not just "things near bank entries."

**A3 (original signature dim of 16 was arbitrary).** Justification now possible: sig_dim must be large enough that equivalence classes (not expressions) are linearly separable in trit-Hamming space. The right number is empirical, derivable from a sweep on the equivalence-class set in the bank. Probably falls out of P0-1's design pass as a SWEEP target, not a fixed choice.

**A4 (5-day budget).** Was overconfident. Realistic estimate now: ~1 week for P0-1 (signature rule + test-input set design + equivalence-class theory + sig_dim sweep) + 2 days for P0-2 + 2 days for P0-3 + 3 days for P0-4 (with proper held-out probe construction) = roughly 2 weeks of focused work. Halve the optimism.

**A5 (treat 1-var and 2-var as a minor variation).** False. They're separate banks with separate signature spaces. Treating them as a minor variation would have buried T3 silently.

## What I Now Understand

The bridge from data-routing to expression-routing isn't about reusing the existing bank with different inputs. It's about defining an equivalence relation on expressions, computing a canonical representative per class, and using the existing routing primitives (`m4t_popcount_dist`, top-k=1) to do equivalence-class lookup.

The bank type doesn't change. The constructor does. The label semantics do. The probe does. The verdict gate does.

Once that framing is in place, the four-piece P0 plan from the original document is structurally right — but the FRAMING and the PROBE must be rewritten. P1's unified-space question gets sharper too: it becomes "can data equivalence classes and expression equivalence classes share a signature space" — which is a much more precise version of "do data and expression queries route into the same bank."

The Laundry Method (from LMM) applies cleanly here: partition by arity first, then equivalence-class within arity, then nearest-representative within class. Don't try to search the whole expression space at once.

## Remaining Questions

- What test-input set best separates the canonical equivalence classes you actually care about? Probably needs to be derived from the bank's representative expressions (so it's class-discriminating by construction), not designed in the abstract.
- Is "equivalence on a finite test-input set" the right equivalence relation, or do you need a richer one (e.g., equivalence on all MTFP-representable inputs)? The finite version is computable; the universal version probably isn't, in any practical sense. The finite version has known false positives — flag them, don't pretend they don't exist.
- Cost-aware routing (return the cheapest equivalent expression) — defer to P1 or carry cost as representative metadata in P0-3? Probably the latter; it's almost free in P0-3 and saves a P1 sub-cycle.
- Do constants need their own bank, or can they be excluded from P0 entirely? Probably exclude from P0; they're a P1 question if they ever come back.
