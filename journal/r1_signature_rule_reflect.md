# Reflections: R1 Signature Rule Design

## Core Insight

The Path A vs Path B framing was a distraction. The real choice — the one that determines whether the new rule actually carries information beyond sign — is **per-expression tau scaling vs fixed tau**.

Per-expression tau makes the signature reflect the expression's *internal* magnitude structure, not its absolute scale. That achieves the kind of scale-invariance vision claim #2 wants: same internal shape merges regardless of absolute scale; different shapes don't.

Once per-expression tau is the foundation, **Path A wins on substance**: it uses the third state to carry monotone-magnitude information that Path B's wildcard semantics actively discards. Path B's "uncertainty at near-zero" eats discriminating signal at exactly the positions where it's most informative (sign-flip points).

The choice isn't "which substrate kernel do we want to exercise" — both paths use shipped kernels. The choice is "which third-state semantic carries information for THIS use case." For mathematical expression equivalence, magnitude bands carry more useful information than uncertainty regions.

## Resolved Tensions

**T1 (Path B eats information at sign-flip points) — RESOLVED.** Path A doesn't have this problem because near-zero values land in the "zero" band by design, contributing 0-trit positions. Sign-flip points (`x²-1` at x=0 = -1) are tiny but nonzero; per-expression tau (with max_abs > 1) puts them in weak-negative, not zero. They retain their discriminating signal.

**T2 (Path A storage is 1.5x Path B) — RESOLVED by negligible absolute cost.** At sig_dim=16, Path A is 6 bytes/tile vs Path B's 4. At 1000 tiles, 6KB vs 4KB. Storage isn't the constraint. Discrimination is.

**T3 (Path A's distance kernel is slower) — RESOLVED with a flag.** Real concern at scale, irrelevant at toy. R2's scaling experiment will surface whether this matters in practice. If the confidence-weighted distance becomes the bottleneck, a NEON variant follows; the substrate's spec contract permits it.

**T4 (per-expression tau adds setup-time evaluator pass) — RESOLVED by acceptable cost.** Setup runs once per bank; queries are single-pass either way. The double-pass is paid in bank construction and in signature generation per query; both are cheap.

## Challenged Assumptions

**A1 (fixed tau is simpler, hence preferable).** False. Simpler at the implementation level but wrong-shaped for the problem. Expressions span 1.5+ orders of magnitude in our test inputs; no fixed tau matches. Per-expression tau is the only choice that makes the rule meaningful across a heterogeneous bank.

**A2 (Path B is cheaper because single-trit signatures).** Technically true, substantively wrong. The single-trit savings cost us the magnitude-band information that's the whole reason for the new rule. Cheaper-but-broken is worse than slightly-more-expensive-but-correct.

**A3 (R1-B gate is easy to clear).** Possibly false. If tau_weak and tau_strong are too close together, all positions land in the same band and Path A degenerates to sign-only. The choice tau_weak = max/4, tau_strong = max/2 leaves the bands at meaningful separation — but the gate is real, and the rule must be designed to clear it intentionally, not by accident.

**A4 (the equivalence-class machinery needs updating).** False. The equivalence-class detection is "memcmp on signatures." Concatenating trit_sig + conf_bits and memcmp-ing the whole still works. The machinery doesn't change; only the input format does.

## What I Now Understand

The right rule for R1 is:

**"Per-expression-tau dual-threshold over MTFP test-input evaluations, ternarized via `m4t_route_threshold_extract_dual` with tau_weak = max_abs/4 and tau_strong = max_abs/2, paired with `m4t_route_confidence_weighted_dist` for routing."**

This:

- Uses two substrate kernels currently unused in the consumer (R1-C cleared by construction).
- Uses the third state load-bearingly (zero band is "values within max/4 of zero, relative to this expression's scale" — meaningful, not accidental).
- Captures monotone-magnitude information (concern 7 partially addressed: `exp(x)` and Taylor truncation will produce different band distributions where they diverge).
- Doubles signature cells, helping the headroom problem (concern 8 partially addressed: more bits per tile).
- Preserves scale-invariance at the right granularity (`x` ≡ `2*x` still merges; `x` and `x²` still differ).

What's still uncertain:

- Whether the chosen tau ratios (max/4, max/2) are optimal. They're a starting point; might need iteration if R1-B gate result is borderline.
- Whether expressions whose max_abs is achieved at a single test input get skewed signatures. Probably fine but worth eyeballing.
- Whether the new rule changes the original 30 subagent probes' routing in mathematically-defensible ways (R1-A gate). Some probes might now route to *different* classes that are equivalent under sign-only but distinct under bands. That's not a regression — it's the new rule doing its job. The gate threshold (≥70%) gives room for this.

## Remaining Questions

- For arity-2 with input pairs, max_abs is computed across all 16 pairs. That's the right scale. Confirmed.
- The 5-state encoding (per-position trit + conf) is what `m4t_route_threshold_extract_dual` produces. Confirmed by reading m4t_route.h.
- Confidence-weighted distance kernel cost table: opposite-sign mismatches at conf-0/conf-1 weights of 2/3/4. Reasonable for our use; the relative ordering of similar expressions doesn't depend on absolute cost magnitudes.
