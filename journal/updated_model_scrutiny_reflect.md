---
date: 2026-04-14
phase: REFLECT
topic: Scrutinizing the updated model
---

# Reflect

## Core insight

**The prior cycle's updated model had three specific errors, each avoidable in hindsight, each introduced by taking the incremental path at a decision point where the architectural path was clearly indicated.**

- C-sub doesn't separate from C-con; the two-part criterion was a rhetorical decomposition, not a structural one. Single criterion (emission coverage) is correct.
- "Keep sign_extract + annotate contract" was the incremental choice. Under minimum-surface discipline and pre-1.0 repo state, deletion is architecturally correct.
- Option 3 was claimed zero-substrate-change. That claim was wrong; it needs `m4t_mtfp4_unpack_to_trits`.

Each error came from the same root cause: **at decision points, I chose the familiar-feeling path rather than the discipline-indicated path.** "Keep sign_extract" felt safer than deletion. "Two-part criterion" felt cleaner than admitting the criterion is per-deployment. "Option 3 is free" felt optimistic. All three were errors in the direction of convenience over rigor.

The meta-cycle's job is to catch exactly this. It did.

---

## Resolved tensions

### T1 (simplicity vs backward compatibility) → RESOLVED: simplicity wins
Pre-1.0 repo. One consumer. No published API. "Backward compatibility" here means "I don't want to rewrite three lines of signature_update." That's not a real cost. Delete sign_extract.

### T2 (speed vs depth) → RESOLVED: stop when cycles stop reversing substantively
This meta-cycle surfaced substantive reversals (criterion structure, kept-vs-deleted primitive, scope). A third cycle might find more. But the stopping rule is: commit when a cycle only refines rather than reverses. Apply the current synthesis; run another cycle only if post-implementation reality surfaces new challenges.

### T3 (precision vs usability) → RESOLVED: per-deployment criterion with named sanctioned pairs
The criterion is inherently per-(primitive, input-distribution). That's not a bug — it's how base-3 nativity actually works. The substrate documents sanctioned deployments; consumers cite the applicable one. This IS usable, just not in the "apply once, universal answer" sense.

### T4 (scope: include Option 3 or defer) → RESOLVED: defer Option 3
No consumer currently needs MTFP4-native signatures. Adding `mtfp4_unpack_to_trits` without a consumer violates the substrate discipline ("no primitive without named consumer demand"). Defer. If a consumer drives it, re-open.

---

## Hidden assumptions challenged

1. **"A two-part criterion is cleaner than one-part."** False here. The two parts collapse into the same test in practice. Cleanliness was aesthetic; one-part is more honest.
2. **"Keeping sign_extract is cheap because signature_update depends on it."** Partial truth. signature_update depends on *some primitive with sign-extraction semantics*, not specifically on sign_extract-by-name. That dependency is trivially redirected to threshold_extract(τ=0).
3. **"The substrate's primitives are fixed fixtures we reason about in isolation."** Wrong. Primitives are evaluated in context of deployments. Substrate documentation should list (primitive, sanctioned input-class) pairs.
4. **"A docstring communicates a contract."** Weaker than I assumed. Docstrings are ignorable; type signatures are not; the absence of a primitive cannot be ignored. Architecture communicates by what's missing as much as by what's present.
5. **"Option 3 is free because the substrate already has packing."** Wrong. Packing takes individual trits, not multi-trit mantissas. The unpack step is missing.

---

## What I now understand

The prior cycle landed on a carefully-articulated two-part model that, under scrutiny, dissolves into a single behavioral criterion. The cycle's utility wasn't to confirm that model — it was to expose that the model was one iteration short. This scrutiny cycle takes the next step.

The correct model:

1. **One criterion: emission coverage.** Under the intended input distribution, every output state the primitive's API can emit must be emitted non-trivially. No substrate-absolute "is it base-3 native." Only per-deployment.

2. **Substrate documents sanctioned (primitive, input-class) pairs.** Each primitive's contract names the input distributions for which emission coverage holds. New primitives ship with enumerated output space, sanctioned input class, and an emission-coverage test.

3. **Delete sign_extract.** Add threshold_extract as the sole extractor. signature_update updates to `threshold_extract(..., 0)`. One primitive, parameterized; the sign-only case is τ=0 at every call site, making the structural truth visible.

4. **Option 3 deferred.** No consumer demands MTFP4-native signatures yet. Substrate discipline says no primitive without named demand. Defer `mtfp4_unpack_to_trits`.

5. **Scope.** ~160 LOC under deletion path. ~60 LOC new threshold_extract + tests. ~40 LOC spec addition (§18 emission-coverage criterion + review gate). ~15 LOC signature_update update. ~45 LOC test suite update (replacing sign_extract tests with threshold_extract tests). 0 Option 3 work this pass.

---

## What remains uncertain

1. **Is emission coverage the right criterion, or does it need one more revision?** Confidence: moderate-to-high. The criterion passes its own test on known cases. A third cycle might find refinement.

2. **Does the review gate (enumerated output + sanctioned input class + coverage test) catch future similar failures?** Confidence: moderate. It catches type-system theater specifically (sign_extract's three codes, two-in-practice). It may not catch subtler failures.

3. **Is there a primitive in the live surface that fails emission coverage under its current use that I haven't audited?** Confidence: moderate. I audited sign_extract explicitly. Re-auditing others under the revised criterion is wise before committing.

4. **Is this meta-cycle itself free of the same "incremental bias" it caught in the prior cycle?** Honest answer: I don't know. I've tried to apply the discipline; I might still be missing something. The test is: run the implementation, look at the result, see if new issues surface.

---

## What the cycle surfaced

- The two-part criterion collapses into one. Not an improvement; just a rhetorical split.
- "Keep sign_extract" was convenience, not architecture.
- Option 3 isn't free. Defer or commit.
- The substrate's contract is per-deployment, not per-primitive-absolute.
- The review gate needs to be explicit in the spec, not implicit.
- The meta-cycle's discipline caught three compounding errors. The user's insistence on rigor was correct.

---

## What to do next

1. Synthesize the corrected plan.
2. Present to user for validation.
3. If validated: implement the deletion path, ~160 LOC, as a single focused commit.
4. Defer Option 3.
5. Re-audit all other routing primitives against emission coverage before declaring the architecture correct.
