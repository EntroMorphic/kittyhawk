# Raw Thoughts: Path Forward After R1 Remediation FAIL

## Stream of Consciousness

R1's PASS was a mirage. The remediation gates that bit hard returned FAIL: 4.2% partition change (gate ≥30%), arity-1 inter-class min=1 (gate ≥4). The dual rule does what it claims (kernel calls happen, conf bits get set, rule-difference probes route correctly) but it doesn't deliver what the original concerns required (better discrimination, substrate-distinctness in operational terms).

The closeout proposed three options: A (revert arity-1, keep arity-2 dual), B (redesign R1), C (proceed to R3/R2 anyway). The user is asking me to think harder before picking.

What do I actually believe right now?

**The dual rule isn't broken — the BENEFIT it claimed to provide doesn't exist at this scale.** Substrate kernels run, conf bits encode magnitude bands, the partition does change in specific predictable cases. But: the partition mostly doesn't change, and where the rule discriminates more, it discriminates worse. The gain is cosmetic; the cost is real.

This means concerns 2, 3, 7, 8 — the things R1 was supposed to address — are still mostly open. The dual rule addresses them in a narrow technical sense (kernels exist in the call path) but not substantively (kernels don't enable better discrimination).

**Maybe the signature rule was the wrong place to address these concerns.** The original red-team's framing said "the per-cell encoding throws away information the substrate could carry." But the actual constraint may be different: maybe with 16 test inputs and a 11-class bank, there isn't much information to carry — the signature is information-saturated regardless of per-cell encoding. Adding more states per cell can't help when the bank's intrinsic discriminability is what's bounded.

**Or maybe the test inputs aren't dense enough.** If each cell has to carry more discriminating information, you can do that by widening the cells (dual rule) OR by adding more cells (sig_dim sweep). R3 was supposed to test the latter. Maybe sig_dim=64 with sign-only would beat sig_dim=16 with dual.

**Or maybe we're solving the wrong problem.** The concerns were about the SUBSTRATE's distinctive value proposition. The expression-routing CONSUMER doesn't have to be the place those concerns get resolved. Maybe vision claim #3 ("base-3 carries info base-2 collapses") manifests at a different layer entirely — perhaps in P1-1 (transcendentals) where the substrate's exp/log primitive could use the third state for "domain undefined" or "approximation regime."

**Or maybe vision claim #2 doesn't actually NEED vision claim #3.** The two are listed as parallel claims. Expression-routing might work fine with sign-only signatures forever; the substrate-distinctness work belongs to a different consumer (one that does numerical computation, not equivalence routing).

What are options I haven't surfaced yet?

**Option D: Run R3 with both rules.** Sweep sig_dim ∈ {16, 32, 64, 128} for both sign-only AND dual. If sign-only at sig_dim=64 wins on discrimination, then R1's premise was wrong — more cells beats richer cells.

**Option E: Adversarial probes.** Hand someone "design probes specifically meant to break the routing." Not yet done. Could surface either confidence (rule survives adversarial scrutiny) or further FAILs. Useful before committing to scale.

**Option F: Skip R2/R3; pivot to P1-1.** The R1 FAIL tells us vision claim #3 isn't easily addressed via signature richness. Maybe vision claim #1 (close primitives floor with exp/log) is both more tractable and more central to vision claim #2 (because exp/log expand what expressions can be ROUTED, not just what signatures can ENCODE).

**Option G: Step back. Reconsider whether the R2 plan was the right scope.** The R2 plan addressed 5 of 9 concerns. Maybe the 5 weren't the right ones to bundle. Maybe concern 1 (scope gap) is the only one actually worth a cycle, and the others are derived problems that don't matter until concern 1 is solved.

**Option H: Per-arity rules.** What the closeout's Option A proposed. Sign-only for arity-1; dual for arity-2. Pragmatic. Feels ad hoc but might just be honest acknowledgment that arity matters.

What scares me about each option:

- A (revert arity-1): we end up with two rules in the codebase. Per-arity dispatch fragments the consumer. The "right rule" depends on arity in a non-principled way.
- B (redesign): more cycles, more code, no guarantee the next rule works. We've already invested ~3 weeks in this track. How many more cycles before we admit the signature-richness path is wrong?
- C (proceed anyway): builds on a known-broken foundation. The R3 sig_dim sweep would surface the dual rule's weaknesses at scale; R2 scaling would compound them. False confidence.
- D (sweep both rules): more work but resolves the question "is the right move more cells or richer cells?" empirically. Might find sign-only at higher dim is the right answer all along.
- E (adversarial probes): cheap but doesn't change the verdict — just sharpens it. Doesn't move the project forward.
- F (pivot to P1-1): ambitious. Closes the primitives floor (vision claim #1) which is foundational. Might unblock concerns the consumer can't address alone.
- G (rescope R2): meta-pause. Honest but feels like running away from the work.
- H (per-arity rules): pragmatic but ad hoc.

What's probably wrong with my first instinct? My first instinct was Option A from the closeout — revert and move on. That instinct was driven by "let's not waste more time on a failing track." But Option A leaves the underlying questions unanswered. The honest move might be D (run the sweep, see if richness or breadth wins) before committing to revert.

What scares me most? That the project's path might require a redesign of vision claim #3's operational manifestation. The original framing (signatures use third state load-bearingly) might be wrong for the expression-routing consumer. That's a vision-level concern, not a cycle-level one.

## Questions Arising

- Does sig_dim matter more than per-cell encoding for our use case?
- Is vision claim #3 a substrate-level claim or a consumer-level claim?
- Did R1 fail because the rule is wrong, OR because the bank is too small, OR because the test inputs are too sparse?
- Is the R2 plan's three-track structure (R1, R3, R2) still the right shape after R1 FAILed?
- Should P1 work proceed independently of R-track outcomes?
- What's the cheapest test that would distinguish "rule wrong" from "bank too small" from "vision-claim mismatch"?

## First Instincts (suspect; to be challenged)

- Option A is cheapest and gets us moving.
- Option D (sweep both rules) is the most epistemically honest.
- Option F (pivot to P1-1) is the boldest but might be the right one.
- Vision claim #3's operational manifestation in the expression-routing consumer might be a category error.

## Risks I Already See

- Sunk-cost reasoning pulling us toward "fix R1 v2" when the deeper question is whether signature-richness was ever the right approach.
- Per-arity rules fragmenting the consumer with no clean theoretical justification.
- Pivoting to P1-1 before establishing whether R2 (scale experiment) would have worked, leaving concern 1 (scope gap) permanently open.
- Running more cycles without clear stopping criteria — the project's discipline produces good evidence per cycle but doesn't decide WHEN to stop.
