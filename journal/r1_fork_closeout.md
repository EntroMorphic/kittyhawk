# Closeout: R1 Fork Experiment — F3 wins (wrong layer)

Per `docs/PLAN_R1_FORK.md` and `journal/r1_path_forward_synthesize.md`.

## Verdict: F3 wins — pivot to P1-1; archive R1

```
F1 (wrong rule)  : lose  (dual at sig_dim=64 only +1 trit over sign-only; gate required +2)
F2 (wrong axis)  : lose  (sign-only at sig_dim=64 reaches min=3, gate required >=6)
F3 (wrong layer) : WIN   (both rules capped <6 at arity-1 sig_dim=64; dim doesn't help meaningfully)
```

## What the data showed

### Arity-1 sweep (the deciding arity)

| sig_dim | rule | curated min | random-bank classes (3-seed avg) |
|---------|------|-------------|----------------------------------|
| 16 | sign-only | 3 | 27 |
| 16 | dual      | 1 | 33 |
| 32 | sign-only | 3 | 27 |
| 32 | dual      | 1 | 36 |
| 64 | sign-only | 3 | 27 |
| 64 | dual      | 4 | 38 |

**Sign-only is fundamentally stuck.** Adding cells from 16 → 64 produced ZERO improvement in either curated min distance or random-bank class count.

**Dual recovers some at sig_dim=64** (min=4 vs sig_dim=16's min=1) but is still below the 6-trit headroom threshold and still doesn't add ≥2 trits over sign-only.

**Random-bank class count plateaus at ~27 for sign-only and ~38 for dual.** The expression set's intrinsic discriminability is the ceiling, regardless of signature dim.

### Arity-2 sweep

| sig_dim | rule | curated min | random-bank classes |
|---------|------|-------------|---------------------|
| 16 | sign-only | 8 | 42 |
| 16 | dual | 8 | 60 |
| 32 | sign-only | 16 | 45 |
| 32 | dual | 14 | 60 |
| 64 | sign-only | 32 | 48 |
| 64 | dual | 30 | 61 |

**Arity-2 is NOT saturated** — discrimination scales with dim (sign-only min: 8 → 16 → 32). Dual gives marginally more random-bank classes (60 → 61) but barely changes inter-class distance.

### Cross-rule partition change

Across all (arity, sig_dim) combinations, partition-change rate between sign-only and dual is **2-5%** — meaning that even where the rules differ, they place the same expressions in the same classes ~96% of the time.

## What this means

**The arity-1 expression set has ~10-11 sign-equivalence classes, period.** No amount of signature engineering reveals more structure than the expressions actually have.

**The expression-routing consumer is signature-saturated for the curated arity-1 bank.** Both rules work; neither delivers substrate-distinctness because there's no substrate-distinctness to deliver at this consumer's scale.

**Vision claim #3's "substrate-distinctness in the consumer" cannot be demonstrated by this consumer.** The substrate's affordances (third state, magnitude bands, wildcards) are real, but the expression-routing consumer doesn't have a problem they solve. The consumer's problem is "recognize sign-equivalence among ~12 expressions" — sign-only signatures handle that.

**Concerns 2, 3, 7, 8 from the original red-team are recategorized:**

- **Concern 2 (substrate kernels unused):** the kernels are real and tested at the substrate layer. The expression-routing consumer doesn't need them. They wait for a consumer that does.
- **Concern 3 (third state as binary):** the third state is binary in this consumer because the consumer's problem is binary-discrimination-shaped. Different consumer, different shape.
- **Concern 7 (compose-equivalence):** mostly N/A without exp/log in the vocabulary. Re-evaluate after P1-1.
- **Concern 8 (low headroom):** real, but caused by the expression set, not the signature rule. Adding more candidates would expand the bank's capacity until headroom returns.

## Implications for the project

### R-track: closed

The R-track (R1, R2, R3) attempted to address concerns 1, 2, 3, 7, 8 by enriching the consumer. The fork resolved that the consumer cannot address concerns 2, 3, 7, 8 — they're substrate-level. Concern 1 (scope gap) and concern 8's expression-set component remain open.

**Action:**
- **Revert R1's dual rule for arity-1** (sign-only is the simpler primary, no operational disadvantage at toy scale). Keep `expr_to_signature_dual` and `expr_bank_dual_t` in the codebase (ship-with-FAIL discipline; they're useful for documenting what was tried).
- **Arity-2 unchanged.** Either rule works fine; no need to revert.
- **Cancel the original R3 (sig_dim sweep) and R2 (scale experiment) cycles as planned.** They were designed for the success case where R1 worked. Replanning is needed before either is run.

### P1-1: prioritized

**Pivot the substrate-distinctness work to P1-1 (close primitives floor with exp/log).** Two reasons:

1. exp/log are independent of the R-track outcome and are foundational for vision claim #2 to scale beyond the current expression vocabulary. `exp(x)*exp(y) ≡ exp(x+y)` is a natural mathematical equivalence the system currently can't even express.

2. P1-1's design naturally uses substrate-distinctive features. The third state can carry "domain undefined" (e.g., `log` of zero or negative) or "approximation regime" (e.g., Taylor truncation valid range). These are operational uses of base-3-ness that the expression-routing consumer didn't need.

**P1-1 RAW phase begins now** (parallel track, owner-authorized in the previous turn).

### What stays open

- **Concern 1 (scope gap):** "all mathematics" remains aspirational. The arity-1 saturation suggests the toy bank shape doesn't extend trivially. A scaling experiment with a much larger expression vocabulary (post-P1-1) would test whether the routing mechanism extends to ~1000+ classes or hits a different ceiling.

- **R2 (scale experiment):** deferred to post-P1-1. Test scaling once the vocabulary includes exp/log; the test is fairer when the vocabulary isn't artificially small.

- **Vision claim #3 (base-3 carries info base-2 collapses):** must be demonstrated by P1-1's transcendental work, not by the expression-routing consumer.

## Substrate-discipline notes

- Ship-with-FAIL: R1 code stays in the codebase. The dual rule's binaries (`gesh_expr_routing_r1`, `gesh_expr_routing_r1_remediation`) remain green. Their FAIL verdict is documented at the journal level.
- All 14 ctest binaries still PASS (verified post-experiment).
- Fork experiment binary (`gesh_expr_routing_r1_fork`) builds clean under -Werror.
- Pre-committed framings honored: F3 won per its stated criteria; no post-hoc gate revision.

## Per-finding final disposition (concerns 2, 3, 7, 8 from R1 red-team era)

| Concern | Status |
|---------|--------|
| 2 (substrate kernels unused in consumer) | RECATEGORIZED. Kernels are real at substrate layer; consumer doesn't need them. Wait for a consumer that does. |
| 3 (third state as binary) | RECATEGORIZED. Consumer's problem is binary-shaped. Vision claim #3 manifestation moves to P1-1. |
| 7 (compose-equivalence) | DEFERRED to post-P1-1. Cannot be tested without exp/log in vocabulary. |
| 8 (low headroom) | EXPRESSION-SET-LIMITED, not signature-rule-limited. Add candidates to grow capacity; not a rule problem. |

## Methodology notes lifted from this cycle

**Fork experiments are cheap and high-value when the verdict is contested.** ~3 days of code distinguished three framings that would otherwise have been picked on intuition. Without the fork, we might have spent 2 weeks on R1 v2 (Option B) only to discover saturation.

**Per-arity outcomes are real evidence, not architecture.** Arity-1 saturated; arity-2 didn't. The data is the architecture; the choice (per-arity rules) is downstream of the data.

**The original R2 plan's three-track structure (R1 → R3 → R2) was sequenced for the success case.** A more robust planning pattern: include the fork experiment as the FIRST step when a major rule change is being tested, with branches for success and various failure modes. Lifted to project-wide methodology suggestion.
