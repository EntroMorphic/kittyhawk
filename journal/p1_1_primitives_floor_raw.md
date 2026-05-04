# Raw Thoughts: P1-1 — Close the Primitives Floor (exp, log)

## Stream of Consciousness

Vision claim #1 names six frozen primitives: add, sub, exp, log, ... (etc). The substrate today has add, sub, mul, neg, max, min, eq — six trit operations plus MTFP arithmetic. exp and log are NOT in the substrate. They're explicitly in the user's named six but explicitly absent from libm4t.

The R-track FAILed at vision claim #3 (substrate-distinctness in the consumer). The fork experiment showed the expression-routing consumer is signature-saturated — it can't deliver substrate-distinctness via the signature rule. The closeout argued P1-1 might naturally use third-state semantics (e.g., "domain undefined" for log of zero). I want to interrogate that.

What do I actually believe about P1-1?

**Two paths, both real:**

- **Path A (substrate primitives):** add `m4t_mtfp_exp` and `m4t_mtfp_log` kernels to libm4t. Hardware-shape native, fast. Uses base-3 internally — possibly with third state for "domain undefined." Real numerical-methods work; integer-only base-3 transcendentals are an open problem.

- **Path B (compositions):** show exp/log can be built as compositions of existing primitives (Taylor series for exp, Newton iteration for log). Keeps the floor literally six (add, sub, mul, neg, max, min). Fits the substrate-discipline rule "no primitive without consumer demand." Slow per evaluation; precision bounded by series length.

The R2 plan's P1-1 section listed both paths and didn't pick. This RAW phase should pick.

What scares me about Path A:
- Integer-only base-3 transcendentals have no obvious tie-free formulation. The cross-exponent accumulator's odd-divisor lemma worked because powers of 3 are odd; transcendental functions don't have that structure.
- The substrate spec would need amendment. New section, new precision contract, new tests.
- The third-state-for-domain-undefined claim is speculative. Maybe log of negative just returns a saturation flag like every other lossy operation in the substrate.
- Building a transcendental kernel is weeks-to-months of real numerical work. We've spent the last few weeks on R-track stuff that didn't pan out. Another speculative cycle is expensive.

What scares me about Path B:
- Tree depth grows fast for accurate exp/log. Taylor for exp converges only near x=0; range reduction (e.g., `exp(x) = exp(x/2)^2`) fixes that but doubles tree depth.
- The expression-routing consumer would see exp(x) as a deep tree. Sign-only signatures of monotone-increasing trees might collapse together (everything with similar shape gets the same signature) — the same signature-saturation we just observed.
- Path B's "exp" is actually "Taylor truncation" with some precision bound. Calling that "exp" is honest only if we document the precision regime.

What's probably wrong with my first instinct?

My first instinct is "Path B because it respects the substrate-discipline rule." But the rule says "no primitive without measured CONSUMER demand." We don't have a consumer asking for exp/log right now. Maybe the right move is neither A nor B yet — the right move is to ASK what consumer would benefit from exp/log and what that consumer needs.

That's a discovery cycle, not a build cycle. Same shape as the original Tier 3 consumer-discovery cycle from `docs/REMEDIATION_PLAN.md`. The owner-override on that cycle (skip discovery, build directly) was later flagged as a discipline violation.

Repeating the discipline violation here — building exp/log without a named consumer — repeats the mistake.

But there's an alternative reading: vision claim #1 itself IS the consumer demand. The user has stated as foundation "all compute math derives from these six primitives" — meaning the six are the floor of what the substrate must offer. Owner directive supersedes consumer-discovery in this case because the directive IS the demand.

Both readings are defensible. The owner gets to decide.

What do I think the answer should be?

Honestly, I think the answer might be: **do Path B as a proof of expressibility, then test whether the expression-routing consumer recognizes Taylor truncation as ≡ exp.** If it does, the consumer-layer exp is "good enough" and Path A isn't yet needed. If it doesn't (because of saturation, precision loss, or signature collapse), THEN we have a measured demand for Path A.

That's a cheap test. Build a Taylor expansion of exp(x) as a tree (3-5 terms, hand-rolled). Add it to the bank. Build another candidate that's an algebraically-different equivalent (e.g., a different truncation depth, or a Newton iteration). See if they merge under sign-only signatures. If yes → consumer recognizes exp-equivalence → Path B suffices. If no → Path A is justified.

But ALSO: the fork experiment just showed sign-only is saturated for the existing 12-candidate bank. Adding exp candidates might just saturate further. The test would need to use a fresh bank that's primarily exp-shaped, with multiple exp variants.

Hmm. Maybe the right answer is even more radical: **don't do P1-1 yet.** Step back. Ask what the system as a whole needs to demonstrate vision claim #2 at non-toy scale. Maybe it's not exp/log at all — maybe it's a real-data benchmark that exercises the existing primitives at scale.

Vision claim #2 says "all mathematics expressible as routing." We've shown this for ~20 hand-designed expressions. Scaling to 1000+ random expressions (the original R2 plan) would test the claim more honestly than adding exp/log to the vocabulary.

What scares me about deferring P1-1:
- The user named exp/log specifically in the foundation. Deferring might feel like ignoring the directive.
- The substrate's claim to be a "primitives floor" is hollow without exp/log.

What scares me about doing P1-1 immediately:
- Building speculatively without measured demand violates substrate-discipline.
- Tying up another 2-4 weeks on an open numerical-methods problem when R-track work just FAILed makes me worry about momentum-loss.

OK genuine first-instinct ranking:

1. **Discovery cycle:** half a day asking "what consumer would benefit from exp/log being primitive vs compositional?" If we identify a real consumer, build for it. If we don't, defer.
2. **Path B proof-of-concept:** 2 days. Build hand-rolled Taylor exp(x) as expression tree. Test whether the existing routing consumer recognizes Taylor variants as equivalent. If yes → consumer-layer exp suffices. If no → measured demand for Path A.
3. **Vision claim #2 scale experiment without exp/log:** the original R2 (scale experiment) without waiting on P1-1. Tests whether the routing mechanism extends to large random-expression banks. This is concern 1 (scope gap) standing alone.
4. **Path A substrate primitives:** only after either #1 or #2 reveals demand.

What's the deepest question?

The deepest question is whether exp/log are CRITICAL to the project's vision being demonstrable, or just NAMED in the foundation. If critical → Path A or B is necessary. If just named → defer until critical.

The fork experiment's verdict (F3 wins; consumer can't deliver substrate-distinctness) makes me think exp/log might be more critical than I'd initially thought. Because: if the substrate's distinctive value is supposed to manifest somewhere, and it doesn't manifest in the expression-routing consumer, then it has to manifest somewhere else. P1-1 (transcendentals) is a natural place because transcendentals naturally have domain restrictions and precision regimes — places where third-state semantics make sense.

But I'm also wary of the "substrate-distinctness must manifest somewhere" framing. Maybe vision claim #3 is wrong, or wrong in scope. Maybe base-3 doesn't carry information base-2 collapses in the way the project's been framing it. The R-track FAIL is partial evidence for that.

## Questions Arising

- Is Path B's "exp = Taylor truncation as expression tree" actually testable in the current consumer, given the saturation finding from the fork experiment?
- Does the substrate-discipline rule "no primitive without consumer demand" apply when the foundation explicitly names a primitive?
- What consumer would benefit MOST from exp/log being primitive vs compositional?
- Is there a real-data benchmark for vision claim #2 that doesn't require exp/log?
- Could vision claim #3 be wrong, or wrong in scope, or right but in a different layer than expected?
- If we defer P1-1, what's the next forward-moving cycle?

## First Instincts (suspect; to be challenged)

- A discovery cycle is the most discipline-respecting move.
- Path B proof-of-concept is the cheapest way to find out if Path A is needed.
- The R-track FAIL increases the urgency of P1-1 only if vision claim #3 is correct in framing.
- Building exp/log speculatively repeats the Tier 3 consumer-discovery override mistake.

## Risks I Already See

- Picking Path A or Path B without first asking "what consumer needs this?" — repeats prior discipline-violation mistakes.
- Deferring P1-1 in a way that's actually ducking the question of whether vision claim #3 holds.
- Doing P1-1 work that produces a beautiful kernel with no demonstrable user.
- Letting the R-track FAIL push us into a panic-pivot to P1-1 when the right move might be a step further back.
