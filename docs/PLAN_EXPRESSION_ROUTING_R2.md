---
status: P0 — owner directive 2026-05-03 (concerns-driven remediation, R2)
authority: owner directive — remediates 5 of 9 concerns surfaced after the original P0+remediation PASSed
scope: address shape-problems in the expression-routing work that the 100/100 remediation did NOT address
supersedes: nothing — extends docs/PLAN_EXPRESSION_ROUTING.md with a follow-on track
predecessor_cycle: journal/expression_routing_*.md (full LMM + red-team + 100/100 remediation, all PASS)
---

# Plan — Expression Routing R2 (concerns remediation)

## What this is

The original expression-routing P0 PASSed, and the 100/100 red-team remediation PASSed. Both verdicts stand. But after running, five concerns remain that the prior cycles did NOT address:

1. **Scope gap.** Vision claim #2 says "all mathematics." Our evidence is 19 hand-designed equivalence classes. The cumulative work doesn't visibly close the gap.
2. **The substrate's distinctive kernels are mostly unused.** libm4t has wildcard distance, dual-threshold extract, confidence-weighted routing, MTFP cross-exponent accumulator. The expression-routing work uses only `popcount_dist` and `threshold_extract`. The substrate is built to a higher spec than this consumer demands.
3. **Sign-only signatures use base-3 as a binary store.** The signature rule threshold-extracts at tau=0; the third state shows up only when an expression evaluates to *exactly* zero, which is rare. The system would behave nearly identically with binary signatures. Vision claim #3 (base-3 carries info base-2 collapses) is unexercised.
4. **Compose-equivalence may be a foundational bug.** Decision D5 declared `exp(x)` and Taylor truncation routing to the same address as a feature. Mathematically they're not the same function. Routing them together loses precision/cost information that downstream consumers may depend on.
5. **Arity-1 inter-class minimum distance is 3** (below the 4-trit headroom threshold). Two of 10 arity-1 classes are perilously close in signature; future bank growth at the same dim will start producing collisions.

These five concerns share two roots, not five roots:

- **Root A: the signature rule is too coarse.** Concerns 2, 3, 7 (compose-equivalence) all reduce to "the per-cell encoding throws away information that the substrate could carry." Concern 8 (headroom) is partly about per-cell information density too.
- **Root B: scale gap.** Concern 1 stands alone. Even a perfect signature rule wouldn't close it without engagement at scale.

This plan has three tracks: **R1** addresses root A (richer signature rule), **R2** addresses root B (scale experiment), **R3** addresses concern 8's headroom-specific question (principled sig_dim choice). R1 ships first because R2 and R3 build on it.

## Vision claim recap (the lens)

> Given that all required compute math derives from ~6 frozen primitives, all mathematics can be classified and expressed as signatures via routing over those primitives.

Plus the substrate's specific affordance:

> Ternary / base-3 fundamentally allows more complete modeling than base-2.

The R2 plan tests both at once: by using a richer signature rule that exercises base-3-distinctive kernels, the work now bears on both vision claim #2 and #3. Sign-only signatures only bore on #2 (and not even cleanly).

## The conceptual fix (what changed since R1)

Sign-only ternarization treats the third state as accidental ("exact zero on test input — rare"). The richer rules treat the third state as *load-bearing*: either as a magnitude band (weak vs strong sign) or as a wildcard (this dim doesn't discriminate here). Both paths use substrate kernels that currently sit unused.

The bank type and equivalence-class framing don't change. The signature derivation rule does, and with it the distance kernel paired against the new signatures.

---

## R1 — Richer signature rule

**The gap.** Sign-only at tau=0 collapses the third state into rarity. Base-3 storage becomes binary in practice.

**What "fixed" looks like.** A new signature derivation function that produces signatures using the substrate's third-state-distinctive kernels. Two paths under design; the LMM cycle picks one or combines them.

**Path A: Dual-threshold (5-state encoding).** Use `m4t_route_threshold_extract_dual` from libm4t (already shipped). For each test input, classify the expression's value into one of 5 states:

- strong-negative (`v < -tau_strong`)
- weak-negative (`-tau_strong ≤ v < -tau_weak`)
- zero (`|v| ≤ tau_weak`)
- weak-positive (`tau_weak < v ≤ tau_strong`)
- strong-positive (`v > tau_strong`)

Stored as (trit, confidence-bit) pairs. Distance via `m4t_route_confidence_weighted_dist`. Magnitude information is preserved at band granularity.

This addresses:
- Concern 2 (uses substrate kernels for the first time in the consumer)
- Concern 3 (third state — both zero AND the weak bands — now carries information)
- Concern 7 partially (`exp(x)` and Taylor truncation will differ at strong-band edges where Taylor diverges, if the bands are tight enough)

**Path B: Wildcard semantics.** When the expression's value is within `±tau_wild` of zero on a test input, mark that position as a wildcard (third state). Distance via `m4t_route_wildcard_dist`. The third state means "this dim doesn't discriminate here" — exact zero AND values too small to call directionally.

This addresses:
- Concern 2 (uses wildcard distance kernel)
- Concern 3 differently (third state means "uncertain," which is mathematically defensible)
- Concern 7 differently (cost-distinct expressions whose precision diverges produce wildcards in different positions)

**Path A vs Path B tradeoff.** Path A doubles signature storage (trit + conf bit per position). Path B keeps single-trit-per-position storage. Path A captures magnitude bands; Path B captures uncertainty regions. Both use substrate-distinctive kernels. The LMM cycle picks based on which captures more information for the target application class.

**Required design work.** One LMM cycle on the signature rule itself:
- RAW: dump tradeoffs of A, B, hybrid, other
- NODES: name what information each rule captures vs collapses
- REFLECT: which rule actually exercises the substrate's distinctive affordances most cleanly?
- SYNTHESIZE: pick one (or hybrid), commit to tau values and storage cost

**Required code work.**
- `gesh/src/expr_signature.c`: extend with a function `expr_to_signature_dual` (Path A) or `expr_to_signature_wildcard` (Path B). Routes through `m4t_route_threshold_extract_dual` or `m4t_route_threshold_extract` + wildcard-position derivation.
- `gesh/src/expr_bank.c`: extend the bank constructor to use the new signature function. Or add a new constructor variant.
- Tests / property tests for the new rule.

**Pre-committed verification gates.**

A. **Backward compatibility:** under the new rule, the original 30 subagent probes still match ≥ 70%. Below: the new rule has broken something the old rule did right; fix or revert.

B. **Information gain:** the new rule produces non-trivially-different signatures from sign-only on ≥ 30% of expressions in a random sample of 100 random expression trees. Below: the new rule is wasting cells on information that doesn't change the answer; rethink.

C. **Substrate-kernel-use audit:** the new rule's call path includes at least one substrate kernel that was unused before (`m4t_route_threshold_extract_dual` or `m4t_route_confidence_weighted_dist` or `m4t_route_wildcard_dist`). Verifiable by grep + code review.

**Anti-pattern.** Adding a "richer rule" that's secretly still sign-only with cosmetic third-state mentions. The audit must show the new rule produces a different equivalence partition than sign-only on representative inputs, not just different bytes.

**Budget.** 1 week.

---

## R2 — Scale experiment

**The gap.** 19 hand-designed equivalence classes is not "all mathematics." There is no current evidence that the mechanism extends beyond toy.

**What "fixed" looks like.** A scaling study that builds banks of {100, 500, 1000, 2000} random expression candidates under the (R1-chosen) richer signature rule and reports:

- Merger rate vs candidate count
- Inter-class distance distribution (min, mean, max)
- Probe-routing consistency on a generative (auto-generated) probe set

**Pre-committed verification gates.**

A. **Merger rate at scale:** at 1000 random candidates, merger rate ≤ 50% (i.e., ≥ 500 distinct equivalence classes). Below: the rule is too coarse for scale; the equivalence relation collapses too aggressively; report FAIL.

B. **Inter-class distance at scale:** at 1000 random candidates, minimum inter-class distance ≥ 4 (one quarter of sig_dim, assuming sig_dim=16). Below: discrimination headroom is exhausted; either expand sig_dim (per R3) or the rule needs more cells.

C. **Probe consistency at scale:** on an auto-generated probe set (programmatically constructed near-equivalents of bank representatives), routing match rate ≥ 70%. Below: as bank grows, near-equivalent probes start landing in adjacent classes; the rule's discrimination is brittle at scale.

**Honest framing.** This experiment will likely surface failure modes. That's the point. If it FAILs, we learn which root cause matters: rule coarseness, dim insufficiency, or fundamental mechanism limit. The FAIL is informative, not catastrophic.

**Required design work.** Half a day on the random-candidate generator (extending `expr_random.c`) and the auto-generative probe-set construction. The probe-generator is the non-trivial part: probes need to be near-equivalents that aren't byte-identical to bank tiles. Likely strategy: start from a bank candidate, apply small structural mutations (constant shift, operator swap, subtree replacement) that preserve approximate behavior.

**Required code work.**
- `gesh/src/expr_random.c`: extend with `expr_mutate_near` for generating near-equivalent probes from a source expression.
- `gesh/bench/expr_routing_scale.c`: new probe binary that runs the scaling study and reports the three gates above.

**Anti-pattern.** Tuning the random-generator's depth/op-distribution against the gates after seeing results. The generator's parameters are pre-committed in this plan; if the scaling experiment FAILs at the chosen parameters, that's the result.

**Random-generator parameters (pre-committed):** uniform op selection from {neg, add, sub, mul, max, min}, uniform leaf selection between var (70%) and const (30%) drawn from {-5,-3,-2,-1,0,1,2,3,5}, max depth 3. Same as `expr_random` already shipped.

**Budget.** 1 week.

---

## R3 — Discrimination headroom analysis

**The gap.** Concern 8 — arity-1 min inter-class distance is 3, below the 4-trit headroom threshold. We documented but didn't design a response.

**What "fixed" looks like.** A principled choice of sig_dim per arity, with documented relationship between sig_dim, bank size, and inter-class distance.

**Sweep design (pre-committed).** Run the curated arity-1 and arity-2 banks (12 and 14 candidates) under the R1-chosen signature rule, with sig_dim ∈ {16, 32, 64, 128}. For each (rule, dim), report:

- Number of equivalence classes after merging
- Minimum, mean, maximum inter-class distance
- Subagent-probe match rate

**Pre-committed verification gates.**

A. **Find a sig_dim that gives ≥ 6 minimum inter-class distance** for both arity banks. Below: even the largest tested dim is insufficient — the equivalence rule itself is collapsing classes that shouldn't merge, and R1 needs revisiting.

B. **The chosen sig_dim's probe match rate ≥ 90%** on subagent probes. Below: discrimination at the chosen dim is real but routing isn't robust to it; revisit the rule.

**Output.** A table that future cycles can reference: "for arity N with bank size K, use sig_dim D under signature rule R."

**Required code work.**
- Extend `expr_routing_remediation.c`'s diagnostics, or add a new probe binary `expr_routing_dim_sweep.c`.
- Test inputs: extending the curated set to ≥ 32 inputs requires designing additional inputs that preserve the symmetric / sign-flip-spanning property of the existing 16. Extending to ≥ 64 requires 4× test inputs for arity-1 (and 8 by 8 for arity-2). The construction is mechanical.

**Budget.** 3 days.

---

## Track sequencing and budget

R1 ships first. It changes the substrate of everything downstream — both R2 and R3 build on the new signature rule.

R3 ships next. With the new rule chosen, the dim sweep tells us what sig_dim the curated banks need.

R2 ships last. With a defensible rule and dim, the scaling experiment is the test of whether the mechanism extends beyond toy.

| Track | Addresses | Budget |
|-------|-----------|--------|
| R1 | concerns 2, 3, 7, partially 8 | ~1 week |
| R3 | concern 8 (specifically) | ~3 days |
| R2 | concern 1 | ~1 week |

Total: ~3 weeks focused work, gated on each track's PASS before next.

## Pre-committed gates summary

A R2 PASS requires all of:

1. **R1-A** (backward-compat): ≥70% subagent probe match under new rule
2. **R1-B** (information gain): ≥30% of random expressions get different signatures vs sign-only
3. **R1-C** (substrate-kernel use): new rule includes at least one previously-unused substrate kernel
4. **R3-A** (discrimination capacity): a sig_dim exists giving min inter-class distance ≥6 for both arities
5. **R3-B** (probe routing at chosen dim): ≥90% subagent probe match
6. **R2-A** (merger rate at scale): ≤50% at 1000 random candidates
7. **R2-B** (inter-class distance at scale): ≥4 minimum at 1000 candidates
8. **R2-C** (auto-probe consistency at scale): ≥70% match rate

A R2 WEAK is any combination not meeting all eight but not FAILing any.

A R2 FAIL is any of:
- R1-A below 50%
- R1-C absent (substrate kernels still not used)
- R2-A above 70% (mergers dominate)
- R2-C below 50%

## What this plan deliberately does NOT do

- **Does not change the bank semantics.** Still equivalence-class lookup; the change is in how signatures are derived, not how the bank is shaped or what it looks up.
- **Does not add cross-arity routing.** Still P1-2.
- **Does not add exp/log primitives.** Still P1-1.
- **Does not address the cooperative-vs-adversarial probe concern** flagged in `journal/expression_routing_remediation_closeout.md`. That's a separate cycle (adversarial probe construction) and it's downstream of having a richer rule.
- **Does not retire any existing P0 cycle.** All prior PASSes stand on their original (narrower) terms.
- **Does not address concern 9** (might-be-wrong). User flagged that as healthy; not actionable in a plan.

## Notes on substrate-discipline (carried forward)

- All new code routes through libm4t kernels for ternarization, packing, distance. The substrate-discipline rule extends here without exception.
- Every new substrate kernel use must be visible in code review (grep-able). The R1-C gate makes this explicit.
- All new code under `-Werror`. All new tests under ctest.
- Pre-commit ALL gates before any code (per H4 from prior remediation, now project-wide rule).
- CHANGELOG entry per track lands with the work.

## Risk register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| R1 design cycle picks the wrong rule (Path A vs B) | Medium | LMM cycle's REFLECT + SYNTHESIZE explicitly evaluates both; backward-compat gate (R1-A) prevents shipping a broken rule. |
| R1 ships but the new rule is more brittle (lower probe match rate than sign-only) | Medium | Backward-compat gate at ≥70% gives margin. If sign-only's 96.7% drops to 70-95%, that's an acceptable tradeoff for substrate-distinctness. Below 70% is FAIL. |
| R3 finds no sig_dim large enough | Low (but possible) | Means the rule itself is collapsing real distinctions. Forces R1 revisit. |
| R2 scaling reveals fundamental ceiling (e.g., random expressions all merge at scale) | Medium-high | The FAIL is informative; tells us the rule's equivalence relation is too coarse for "all math." Either revisit the rule or accept that the mechanism scales to (say) 100 classes but not 100,000. Both are real outcomes. |
| Substrate spec amendment needed for new rule | Low | Path A and B both use kernels already in the substrate. No new primitives required. |
