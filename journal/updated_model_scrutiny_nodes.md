---
date: 2026-04-14
phase: NODES
topic: Scrutinizing the updated model
---

# Nodes

## N1. C-sub fails on its own test for sign_extract
Applied literally: collapse the zero state → `v ≥ 0 → +1, v < 0 → -1`. This is a well-defined binary sign-test. Sign_extract trivially reduces to a binary operation under state-collapse. C-sub FAILS. The prior synthesize claimed it passed — that was rationalization.

## N2. C-sub and C-con collapse into one criterion
Both ultimately ask: under the intended input distribution, is the primitive's three-way output actually utilized? C-sub's "structural three-wayness" is conditional on consumer inputs; C-con's "realizes all three" is the same question from the consumer side. The two-part framing was rhetorical, not structural. One criterion is correct.

## N3. A simpler criterion: emission coverage
**Every output state the primitive's API can emit must be emitted by a non-trivial set of realistic inputs under the intended deployment.** This is a behavioral test on the (primitive, input-distribution) pair. Mechanical: run the primitive on realistic inputs, observe the output distribution, check every code is hit.

## N4. The "keep sign_extract + annotate contract" decision was incremental, not rigorous
Under minimum-surface discipline (which the substrate has been applying), deletion is the correct choice. sign_extract is a degenerate case of threshold_extract (τ=0); having both is redundant. Renaming the contract is weaker than removing the primitive, because docstrings can be ignored but absences cannot.

## N5. signature_update's sign_extract call is trivially replaceable
Becomes `threshold_extract(..., 0)`. ~3 line change. The internal call's correctness is preserved (integer-difference inputs realize zero meaningfully; τ=0 threshold_extract is structurally the same operation).

## N6. Option 3 requires a new substrate primitive
My synthesize claimed Option 3 (MTFP4-native signatures) is zero-substrate-change. False. To use a 4-trit MTFP4 mantissa array as input to popcount-over-packed-trits, you need to expand each mantissa into its four constituent trits. That requires `m4t_mtfp4_unpack_to_trits` (new primitive). Or a consumer-side loop duplicating substrate logic.

## N7. Risks of keeping sign_extract
- (A) Consumers bypass the docstring. The primitive's existence invites use; the warning in the docstring is a speed-bump, not a gate.
- (B) Two near-identical primitives (sign_extract and threshold_extract) create ongoing classification confusion.
- (C) The criterion requires depth to apply. Casual review sees "three output codes" and assumes ternary.
These risks disappear under deletion.

## N8. The substrate should specify sanctioned deployments
If the criterion is per-(primitive, input-distribution) pair (N3), then the substrate's job is to name the sanctioned pairs. Each primitive's docstring should state the input class for which it passes the criterion. Example: "threshold_extract(τ): passes emission-coverage for any input distribution where |v|<τ occurs non-trivially (pre-normalization, τ tuned to data scale)."

## N9. The scope was underestimated
Prior synthesize: ~180 LOC. Actual (under deletion + Option 3 enablement): ~160 LOC for deletion path, ~240 LOC if we also enable Option 3 via mtfp4_unpack_to_trits. Both are bounded, but the synthesize number was wrong.

## N10. The criterion should appear in the substrate spec as process, not just definition
Articulating C-sub/C-con in §18 is necessary but not sufficient. The spec should also say *how* new primitives are reviewed against it. Example review gate: "New primitives MUST ship with (a) an enumerated output space, (b) a stated input distribution class for which emission coverage holds, and (c) a test verifying emission coverage on that class."

## N11. The cycle discipline was load-bearing
Without this meta-cycle, we'd have committed to:
- A criterion that was less robust than I claimed.
- A decision (keep sign_extract) made from convenience.
- A scope that missed Option 3's substrate requirement.
Three compounding errors avoided. The user's insistence on rigor was not pedantic; it was warranted.

## N12. There might still be issues this cycle misses
I have moderate-to-high confidence in the revised criterion and the deletion decision. I have lower confidence that I've surfaced every latent issue. A third meta-meta-cycle might find something. At some point commitment is required; the right time is when current analysis stops surfacing substantive changes.

---

## Tensions

### T1. Simplicity vs backward compatibility
Keeping sign_extract = compatibility with internal callers. Deleting = simpler surface. Pre-1.0 repo with one consumer: compatibility is a premature consideration. Deletion wins on NORTH_STAR discipline.

### T2. Speed of iteration vs depth of verification
The meta-cycle caught real errors. A second meta-meta-cycle might catch more. But infinite regress isn't practical. What's the stopping rule? Probably: stop when a cycle surfaces only refinements, not substantive reversals. This cycle surfaced substantive reversals (C-sub failure, keep→delete, missed primitive). Another cycle might find more — or might not.

### T3. Criterion precision vs criterion usability
"Emission coverage on realistic inputs" is more precise than the two-part criterion but requires specifying "realistic inputs" per deployment. That moves burden onto consumers. The alternative is a substrate-side criterion that applies absolutely — but we've established that's not available (primitives aren't absolutely three-state-capable or not).

### T4. Scope creep vs completeness
Option 3's inclusion adds ~80 LOC. Excluding it keeps the pass tight but leaves a consumer pattern without substrate support. The right call might be: ship the deletion + threshold_extract in this pass; defer Option 3 until a consumer drives it (sticking to the substrate discipline).

---

## Dependencies

- Criterion simplification (N2, N3) → substrate spec §18 rewrite
- Deletion decision (N4) → signature_update update, test suite update, consumer (mnist_routed_lattice) update
- Review gate (N10) → process document or spec addendum
- Option 3 inclusion vs deferral (T4) → substrate scope for this pass
