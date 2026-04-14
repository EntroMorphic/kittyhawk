---
date: 2026-04-14
phase: SYNTHESIZE
topic: Scrutinizing the updated model — corrected plan
---

# Synthesize

## What the meta-cycle changed

The prior cycle's updated model had three specific errors:
1. C-sub / C-con was a rhetorical split, not a structural one. The two parts collapse.
2. "Keep sign_extract + annotate contract" was the incremental choice, not the architectural one.
3. Option 3 (MTFP4-native signatures) was claimed zero-substrate-change. False; it needs `mtfp4_unpack_to_trits`.

All three errors had the same root: choosing the familiar-feeling path at a decision point where the discipline indicated the other direction.

## The corrected criterion

**Emission coverage.** A (primitive, input-distribution) pair is base-3 native iff every output state the primitive's API can emit is emitted non-trivially under that input distribution.

- Single-part, behavioral, mechanical to test.
- Per-(primitive, input-class), not per-primitive-absolute.
- The substrate documents sanctioned (primitive, input-class) pairs.

### Review gate for new primitives

Every primitive added to the substrate ships with:
(a) An enumerated output space (what codes / values the API can emit).
(b) A stated input-class contract (what distribution makes emission coverage hold).
(c) A test that verifies emission coverage on (b).

This is the process that prevents the sign_extract failure from recurring.

## The corrected plan

### Delete `m4t_route_sign_extract`
- Remove function + header declaration.
- Remove test_sign_extract in test_m4t_route.c.
- It's redundant under threshold_extract (τ=0 degenerate).

### Add `m4t_route_threshold_extract(dst_packed, values, tau, n)`
- `|v| < τ → 0, v ≥ τ → +1, v ≤ -τ → -1`.
- Passes emission coverage for any τ > 0 plus input distributions where |v| ranges across ±τ.
- For τ = 0, degenerates to sign-extraction; passes emission coverage only when input distribution realizes zero meaningfully (integer arithmetic with potential exact equality).
- Docstring states both sanctioned input classes explicitly.

### Update `m4t_route_signature_update`
- Internal call `sign_extract(...)` becomes `threshold_extract(..., 0)`.
- Behavior preserved; structural truth now visible at the call site.
- No functional change to the algorithm.

### Add §18 to `m4t/docs/M4T_SUBSTRATE.md`
- Name the emission-coverage criterion.
- State the review gate (enumerated output + sanctioned input class + coverage test).
- Reference this LMM cycle and the prior one as the history.

### Close the sign_extract finding in `docs/REMEDIATION_PLAN.md`
- Original diagnosis: "sign_extract is binary-shaped." Misdiagnosis — it's redundant with threshold_extract, not inherently binary.
- Corrected action: deletion + threshold_extract addition.

### Re-audit all routing primitives against emission coverage
Before claiming the architecture is correct:
- `m4t_route_distance_batch` — output is int32 distance; output "space" isn't three-state; the criterion applies to its inputs (packed trits). Under consumers that feed three-state inputs (which includes the updated signature_update), coverage holds.
- `m4t_route_topk_abs` — output decisions carry a trit sign (+1, -1, 0 sentinel). Sanctioned input class: scores that generate nonzero picks plus potentially zero sentinels. Emission coverage holds when k < T (sentinels possible) AND scores include both signs. Docstring update.
- `m4t_route_apply_signed` — decisions with sign in {+1, -1, 0}. Emission coverage at the decision level holds under typical topk_abs outputs. Docstring update.
- `m4t_route_signature_update` — internal algorithm reviewed; after the threshold_extract(τ=0) substitution, its contract is the contract of threshold_extract with integer-difference inputs. Emission coverage holds because col_sum-vs-mean integer equality occurs naturally.

All routing primitives pass emission coverage under their intended use after this change. No further deletions indicated.

### Defer Option 3
- No consumer currently drives MTFP4-native signatures.
- `m4t_mtfp4_unpack_to_trits` is not added in this pass.
- Pattern documented as a future consumer possibility in `docs/THESIS.md` or a separate pattern note.

## Scope

| Change | LOC | Notes |
|---|---|---|
| Remove sign_extract impl | -30 | function + declaration |
| Remove sign_extract test | -15 | test_sign_extract |
| Add threshold_extract impl | +40 | function + header |
| Add threshold_extract tests | +50 | covering τ=0 and τ>0 cases |
| Update signature_update | +3 | one-line call change |
| Update test_signature_update | 0 | behavior preserved |
| Add §18 to substrate spec | +60 | criterion + review gate |
| Docstring updates (topk_abs, apply_signed) | +20 | name sanctioned input classes |
| Close finding in REMEDIATION_PLAN | +15 | misdiagnosis note + corrected action |

**Net: ~140 LOC** (some deletions, mostly additions). Smaller than both prior estimates. No Option 3. No legacy surface.

## What this pass explicitly does NOT include

- `m4t_mtfp4_unpack_to_trits` (Option 3 enablement). Deferred.
- Re-running mnist_routed_lattice with threshold_extract. That's a separate experiment, after this substrate change lands.
- Benchmark-bed selection or hardware-utilization measurement. Other open tracks.
- A third meta-cycle. Stopping rule: this cycle surfaced substantive reversals; implement them; re-cycle only if implementation reality surfaces new issues.

## Pause point

Per the user's "accuracy matters, mistakes compound" instruction: no code yet. This cycle produced a corrected plan. The user validates:

1. **The corrected criterion** (emission coverage, per-deployment, single-part).
2. **The review gate** (enumerated output + sanctioned input class + coverage test for every substrate primitive).
3. **The deletion decision** (remove sign_extract, add threshold_extract as sole extractor).
4. **The deferral of Option 3** (no consumer demand yet).
5. **The scope** (~140 LOC, one focused commit).

Once validated, implementation is mechanical.

## What the two-cycle discipline cost and bought

Cost: four files × two cycles = eight documents; roughly 1800 lines of writing across the cycles.

Bought:
- Caught the C-sub/C-con rhetorical split before committing it to the spec.
- Caught the "keep sign_extract" incrementalism before landing it as code.
- Caught the missing `mtfp4_unpack_to_trits` before claiming Option 3 was free.
- Produced a review gate that prevents the failure class from recurring.

Verdict: worth it. At 100/100 accuracy standards, two cycles was the minimum.
