# Synthesis: Expression Routing (refined plan)

## Architecture

Build an **equivalence-class lookup** over expression behavior on a fixed test-input set. The bank holds one tile per equivalence class; each tile's label is the canonical representative's ID. Routing is 1-NN over packed-trit signatures using the existing `m4t_popcount_dist`. The four-piece P0 structure from the pre-LMM draft is preserved; the framing of each piece is sharpened by the cycle.

## Key Decisions

**D1: Signature rule = behavior on test-input set, ternarized.** [from REFLECT core insight]
Per-position output of `m4t_route_threshold_extract(eval_expression(input_i))` for `i` in `test_inputs`. No structure-hash, no name-hash. Equivalence-on-test-inputs is the operational definition of expression equality.

**D2: Bank holds equivalence-class representatives, not arbitrary expressions.** [from REFLECT, T2 resolution]
Each tile = one equivalence class. Each label = the canonical (simplest, lowest-cost) representative of that class. The forward path becomes equivalence-class lookup. No new bank type required; existing `gesh_bank_t` works with reframed `labels` semantics.

**D3: Per-arity banks in P0; cross-arity in P1.** [from REFLECT, T3 resolution]
1-variable expressions live in one bank; 2-variable in another. Each bank's metadata records its arity. Constants (arity 0) excluded from P0. P1-2's unified-space work addresses how arity is unified or routed-to.

**D4: Probe tests equivalence recognition, not "X routes to Y."** [from REFLECT, A2 challenge]
Probe expressions are syntactically distinct but semantically equivalent to bank representatives. PASS = the probe routes to the equivalence-class containing its semantic equivalent.

**D5: Compose-equivalence is a feature.** [from REFLECT, T4 resolution]
Routing `exp(x)` and its Taylor truncation to the same address is the system correctly recognizing functional equality. Cost-awareness, if needed, is metadata on the representative — not part of the address.

**D6: Sig_dim is derived, not chosen.** [from REFLECT, A3 challenge]
Run a sweep during P0-1 over candidate sig_dim values, picking the smallest dim that mutually distinguishes all equivalence classes in the starting bank with margin. Document the sweep result.

## Implementation Spec (revised P0)

### P0-1: Signature derivation rule + test-input set design + sig_dim sweep

Same primitive shape as the pre-LMM plan, but the design is now equivalence-driven:

- Choose test inputs by what separates equivalence classes in the starting bank.
- Validate by checking: every pair of representatives in the bank produces distinguishable signatures (distinct trit-Hamming distance ≥ a margin we'll set during the design pass).
- Sweep sig_dim ∈ {8, 16, 32, 64}; pick the smallest that mutually distinguishes all classes with margin.
- Iterate: if two genuinely distinct classes collide on signatures at every dim, expand the input set or revise the equivalence rule.

Output: function `expr_to_signature(out, expr, test_inputs, sig_dim)` + a fixed test-input set + a validation report showing all bank classes are mutually distinguishable.

Substrate-discipline: ternarization through `m4t_route_threshold_extract`. No open-coded sign step.

**Budget:** ~1 week. The signature rule is the foundation; do not undercommit. (Original 1-day estimate was wrong by 5×.)

### P0-2: Expression representation

Tree of nodes over the existing primitives (add, sub, mul, neg, max, min). exp/log explicitly absent (P1-1).

```c
typedef enum {
    EXPR_VAR,      // a variable: x, y
    EXPR_CONST,    // a constant MTFP value (used in expression bodies, not as bank entries in P0)
    EXPR_ADD, EXPR_SUB, EXPR_MUL, EXPR_NEG,
    EXPR_MAX, EXPR_MIN,
} expr_op_t;
```

Plus an evaluator that walks the tree and returns the MTFP value at a given input vector.

**Anti-pattern:** building a parser, pretty-printer, or simplifier. P0 doesn't need any of that. Hand-built C constructors are sufficient.

**Budget:** 1-2 days.

### P0-3: Equivalence-class bank constructor

```c
void gesh_bank_build_from_equivalence_classes(
    gesh_bank_t* bank,
    const expr_t* const* representatives,    // [n_classes]
    int n_classes,
    const m4t_mtfp_t* test_inputs,           // from P0-1
    int test_inputs_per_var, int n_vars,
    int sig_dim                              // from P0-1 sweep
);
```

Each bank tile = one representative's signature (computed via P0-1's rule). Label = representative's ID (NOT a class-id in the data sense). Bank carries arity as metadata.

Two starting banks for P0:

- **Arity-1 bank (16 classes):** `x`, `-x`, `|x|`, `x²`, `x³`, `x+1`, `x-1`, `2x`, `3x`, `x²+1`, `x²-1`, `x*(x-1)`, `x*(x+1)`, `(-x)²`, `max(x, 0)`, `min(x, 0)`.
- **Arity-2 bank (16 classes):** `x+y`, `x-y`, `y-x`, `x*y`, `min(x,y)`, `max(x,y)`, `|x-y|`, `x²+y²`, `(x+y)²`, `x*(x+y)`, `min(x,y)+max(x,y)`, `max(x,y)-min(x,y)`, `x+|y|`, `x*y+1`, `x²-y²`, `(x+y)*(x-y)`.

The arity-2 bank deliberately includes equivalence-redundancy that the probe will exploit (e.g., `min(x,y)+max(x,y)` should equivalence-class to `x+y`; `x²-y²` should equivalence-class to `(x+y)*(x-y)`). The bank constructor's job is to detect and merge these redundancies before publishing — final bank may have <16 classes if mergers occur. Document any mergers as part of the bank's metadata.

**Substrate-discipline:** packing through `m4t_pack_trits_1d`, ternarization through `m4t_route_threshold_extract`. No open-coded sign step.

**Budget:** 2 days.

### P0-4: Equivalence-recognition probe

For each post-merger bank class, construct 2-3 syntactically-distinct equivalents that should route to that class. For example:

- For `x²` (arity 1): `x*x`, `(x+0)*x`, `((x+x)-x)*x`, `-(-x*x)`
- For `min(x,y)` (arity 2): `min(min(x,y), x)`, `-max(-x,-y)`, `(x+y-|x-y|) / 2` is not expressible without `/2`, so skip it; use other constructions
- For `x+y` (arity 2): `(x+y)+0`, `min(x,y)+max(x,y)`, `y+x`, `(x+y)-0`

Probe: route each test expression; record which bank class it lands in; report match against expected class.

**Pre-committed verdict gate (revised, with derivation):**

- Total probes ≈ 30 arity-1 probes + 30 arity-2 probes = ~60 probe expressions across 2 banks.
- Random-routing baseline: 1/(bank size) per arity ≈ 6-7% correct → expected ~4/60 by chance.
- **PASS:** ≥85% of probes (≥51/60) AND no class with all its equivalents misrouted (per-class minimum ≥1/3 correct). The per-class floor prevents PASS being driven by easy classes while hard classes silently fail.
- **WEAK:** 60-84% (36-50/60); the rule from P0-1 needs iteration; do not proceed to P1 yet.
- **FAIL:** ≤59% (≤35/60); behavior-based equivalence does not capture the equivalence the test set defines; rethink before more work.

**Anti-pattern:** tuning the bank, the test-input set, or the probe construction against probe results until the gate PASSes. Probes are the test; tweaking the test until you pass is the failure mode the project's own discipline (multi-config rule, substrate-novelty audit) was designed to prevent. If P0-4 returns WEAK, iteration is allowed but only against probes constructed *after* the iteration.

**Budget:** 3 days (probe construction is non-trivial; ~60 syntactically-diverse equivalents need real thought, plus per-class verification).

### Revised P0 budget total

**~2 weeks of focused work.** Original 5-day estimate was overconfident (challenged in REFLECT, A4).

## Success Criteria (executable checklist)

- [ ] P0-1: Signature rule defined; test-input set designed; sig_dim sweep complete; all bank-class pairs mutually distinguishable in signature with documented margin.
- [ ] P0-2: Expression-tree representation lands in libgesh under -Werror; evaluator passes hand-built unit tests.
- [ ] P0-3: Equivalence-class bank constructor implemented for arity-1 and arity-2; both starting banks built, with any mergers documented.
- [ ] P0-4: ~60 probe expressions hand-built across both banks, with pre-committed expected routings.
- [ ] P0-4: Probe runs end-to-end; verdict gate reports PASS / WEAK / FAIL.
- [ ] CHANGELOG entry lands with the work, not after. Single-step review.

## Major Tensions Addressed

- **T1** (equivalence vs. discrimination): resolved by equivalence-class framing; sig_dim sized to separate classes, not expressions.
- **T2** (bank/vote shape): resolved by reframing labels as representative-IDs; no new bank type.
- **T3** (cross-arity): scoped out — per-arity banks in P0; cross-arity is P1.
- **T4** (compose-equivalence): reframed as feature; cost-awareness deferred to P1 or carried as metadata.
- **T5** (constants degenerate): scoped out — constants excluded from P0 banks; if needed in P1, get their own bank with a different rule.

## Loop-Back Triggers (per LMM)

- **Back to RAW** if signature-rule design reveals an entirely new equivalence question (e.g., "do we want behavioral equivalence or structural?").
- **Back to NODES** if probe construction reveals an unhandled tension. Likely candidate: how to handle expressions whose value depends on a numerical regime (e.g., `x²-1` is positive for |x|>1 and negative for |x|<1; the equivalence class might be input-set-dependent in unexpected ways).
- **Back to REFLECT** if PASS verdict is reached but the synthesis feels unjustified — sanity check the equivalence-class framing.
- **Run a full new cycle** if P1-1 (transcendentals) or P1-2 (unified space) reveals that P0's framing is wrong at a deeper level.

## What This Synthesis Does Differently from the Pre-LMM Draft

1. Replaces "expression bank" with "equivalence-class bank" — the conceptual fix that resolves T2.
2. Replaces trivial-by-construction probe with equivalence-recognition probe — fixes A2.
3. Scopes cross-arity out of P0 explicitly — fixes T3.
4. Replaces arbitrary 7/10 verdict gate with a derived ≥85%-with-per-class-floor gate — fixes A2-style hand-waving.
5. Doubles the budget honestly — fixes A4.
6. Frames sig_dim as derivable from a sweep on equivalence-class separability, not chosen by hand — sharpens P0-1's actual design question and resolves A3.
7. Excludes constants from P0 explicitly — fixes T5 silently.
8. Adds a bank-merger discipline (P0-3): if two "different" representatives equivalence-class to the same signature, merge them before publishing the bank. This is the equivalence-class machinery proving itself by construction.

The structural plan (4 P0 pieces, 2 P1 pieces) survives. The framing of each piece is what the LMM cycle changed.

## Handoff to the executable plan

`docs/PLAN_EXPRESSION_ROUTING.md` is the executable artifact. It should be revised to reflect this synthesis. Specifically:

- P0-1 picks up the equivalence-driven test-input design and the sig_dim sweep.
- P0-3 picks up the equivalence-class framing and bank-merger discipline.
- P0-4 picks up the equivalence-recognition probe shape and the revised verdict gate.
- P0 budget revised to ~2 weeks.
- Cross-arity and constants explicitly listed under "what this plan deliberately does NOT do" with a P1 forward-pointer.

The plan currently carries a status note pointing to this cycle. The next pass either rewrites it inline or supersedes it with a new plan doc that cites this synthesize as its source.
