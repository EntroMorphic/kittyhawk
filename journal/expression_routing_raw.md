# Raw Thoughts: Expression Routing (closing vision claim #2)

## Stream of Consciousness

Vision claim #2 says routing over the six primitives is how all math gets classified and expressed. Today the system routes data queries to data-derived addresses. The bridge to expression-derived addresses doesn't exist. I just wrote a plan that says "do these six pieces, four are P0 in roughly a week." Now I'm being asked to actually run the method on the plan I wrote. The honest read: I jumped straight to synthesis without doing the cognitive work. The original plan is a "first instinct" dressed up as a build sheet.

What do I actually believe about this problem?

Behavior-based signatures sound right because they ground identity in what an expression *does*, not how it's written. But "does" depends entirely on the test inputs you evaluate at. Pick the wrong inputs and equivalent expressions get different signatures (false negative — they look different but aren't) or different expressions get the same signature (false positive — they look same but aren't). The whole rule's correctness rides on the test-input set, and I designed that set in three minutes with no real justification beyond "MTFP4-native, symmetric around zero."

I'm also worried about a more fundamental thing: I treated "expression bank" as a drop-in for "data bank" — same shape, just different source. But the existing bank's `labels` are class indices, used by the forward pass to vote among top-k. Expression labels would be... what? Expression IDs? The forward pass's vote is meaningless for expressions. The whole thing might be the wrong shape, not a small refactor.

And: data signatures are class-MEANS. An average over many samples. A class is defined by its distribution. Expressions don't have distributions — each expression evaluates deterministically. So the analog of "class-mean over samples" for expressions doesn't exist. Either expressions and data live in genuinely different routing structures, or there's a deeper unification that I haven't named.

The probe I specified ("does `x*x` route to `x²`") is actually trivial — in the proposed expression representation, `x*x` and `x²` are literally the same expression (`EXPR_MUL(x, x)`). The test passes by construction, not by routing. I designed a probe that doesn't probe.

What scares me about this problem? The vision is genuinely interesting and I might botch the design phase by jumping to engineering. The audit I just did showed the project has a real pattern of skipping RAW and going straight to synth, then having to remediate. I'm doing exactly that here.

What would the naive approach be? Hash the expression text. Already rejected, but the reason it's wrong is illuminating: it gives identity by name, not by behavior. Behavior-based is the right alternative — but only if the behavior-test is sharp enough to capture equivalence without being so loose it collapses everything.

What's probably wrong with my first instinct? The bank-shape assumption. The forward-vote inheritance. The arbitrary 16-input choice. The 5-day budget. The verdict gate I pulled out of thin air.

## Questions Arising

- What does it mean for two expressions to be "the same" in this system?
- What is the expression analog of "class-mean over samples"?
- Is the existing bank/forward vote shape even the right primitive for expression lookup?
- How does signature dimension relate to discriminability vs. equivalence?
- Single-variable and multi-variable expressions can't share a signature space naively — does that break the unified-routing claim or just refine it?
- What happens to constants (their signatures are all-+1 or all-(-1) — useless)?
- If `exp(x)` and its Taylor truncation route to the same address, is that elegant or alarming?
- What's the prior that says behavior-based equivalence captures mathematical equivalence?
- Why ≥7/10 for the verdict gate? Pulled it from nowhere.
- Is the 5-day budget realistic, or am I estimating like an engineer who already knew the design?

## First Instincts (suspect; to be challenged)

- Behavior-based signatures over a small set of MTFP4 inputs.
- 16-trit signature dim.
- Drop-in bank shape from data side.
- 5-day budget for P0.
- "Does X route to Y" probe shape.
- Hand-pick the verdict-gate threshold.

## Risks I Already See

- Designing a probe that passes by construction (the `x*x → x²` problem above).
- Choosing inputs that make all expressions in some family look identical (collapse).
- Choosing inputs that make functionally equivalent expressions look different (brittleness).
- Treating arity (1-var, 2-var, n-var) as a non-issue when it's actually a hard partition of the signature space.
- Inheriting bank/vote semantics that don't apply to expressions.
- Setting an aspirational budget that the work won't fit in, and then either rushing or reporting failure on a discipline issue rather than a substance issue.
