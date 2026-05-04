# Raw Thoughts: R1 Signature Rule Design

## Stream of Consciousness

The rule today is sign-extract at tau=0. That throws away every magnitude distinction. The R2 plan listed two replacement paths:

- **Path A: dual-threshold 5-state** (strong-neg, weak-neg, zero, weak-pos, strong-pos), uses `m4t_route_threshold_extract_dual` + `m4t_route_confidence_weighted_dist`.
- **Path B: wildcard semantics** (near-zero → wildcard), uses `m4t_route_wildcard_dist`.

Both use shipped-but-currently-unused substrate kernels, so both clear gate R1-C.

What I'm worried about most is **tau choice**. Both paths need a tau value. Our test inputs span {-30, 30} for arity-1 and {-10, 10} for arity-2. But expression values span much more:

- linear (`x`, `x+5`): peaks at ~30
- quadratic (`x*x`, `x²-1`): peaks at ~900
- mins/maxes: bounded by inputs (~30)
- products of vars: up to 100s

So expression values span 1.5+ orders of magnitude depending on shape. **A single fixed tau cannot fit all of them.** If tau_strong = 100, then `x` is always weak and `x*x` is always strong — discrimination collapses *within each expression's own range*.

Two ways out:
- **Fixed tau, accept the mismatch.** Simple, deterministic, but coarse.
- **Per-expression tau scaled to that expression's max output.** Each expression gets tau values matched to its own magnitude range. More compute (extra evaluator pass at setup), but the signature reflects the expression's *internal* magnitude structure, not its absolute scale.

Per-expression tau is interesting because it's actually the right kind of scale-invariance: `x` and `2*x` should still merge (same internal structure, different absolute scale), while `x` and `x²` should differ (different internal structures even at similar scales). With fixed tau, `x` and `2*x` might be in the same magnitude band depending on absolute values; with per-expression tau, they're guaranteed to be.

What concerns me about per-expression tau:
- Setup cost doubles (eval once for max, eval again for signature). Acceptable for bank construction; potentially expensive if queries also need it. But query is single-pass anyway, so not a problem.
- If `max_abs == 0` (e.g., `x - x` always evaluates to 0), tau values are 0 and the rule degenerates to sign-only. Edge case but well-defined.
- If max occurs at only one test input (rare extreme), the bands get skewed by that single input. Probably fine in practice.

What concerns me about Path B specifically:
- Wildcard at near-zero marks small values as "uncertain." But small values are often the most discriminating (sign-flip points like x²-1 at x=0). Marking them wildcard would *lose* the only distinguishing signal.
- The wildcard distance kernel rewards (query=±1, tile=0) at cost 0 — i.e., wildcard tile matches anything. That's the wrong direction for our use: we want the third state to *carry* information, not eat it.

What concerns me about Path A specifically:
- Doubles signature storage (trit + conf bit per position). At sig_dim=16 that's still tiny but the cost compounds at scale.
- `m4t_route_confidence_weighted_dist` has a per-position scan with a few branches — slower than `m4t_popcount_dist`'s pure SIMD popcount. Might matter at scale but not at toy.

Could we hybrid? Path A bands + Path B wildcard for genuine zero (when max_abs == 0 region locally)? Probably overkill for P0; substrate has 5-state dual-threshold but not 7-state.

What's probably wrong with my first instinct? I'm gravitating toward Path A with per-expression tau. The instinct is right but the gate-bite needs care: R1-B requires the new rule to produce different signatures than sign-only on ≥30% of random expressions. Per-expression tau will do that easily for expressions that span large magnitude ranges (quadratics, products). For purely sign-monotone expressions (`x`, `-x`, `x+5`), the dual-threshold signature collapses to weak-pos/weak-neg distinctions that may or may not differ from sign-only depending on tau.

Test that R1-B gate seriously before committing.

## Questions Arising

- Does per-expression tau actually help `exp(x)` vs Taylor-truncation distinguishability (concern 7)? Both have similar magnitude profiles in the safe region; they diverge at extremes. The Taylor's max would be similar but the *band distribution* across positions would differ.
- Should tau_weak/tau_strong be max/4 and max/2, or something else? The choice determines how many positions land in each band. Equal thirds would put roughly 1/3 of cells in each of strong, weak, zero (assuming uniform value distribution, which is wrong for most expressions).
- Path A's storage doubling: does the bank constructor need to rewrite, or can the existing `expr_bank_t` carry an extra conf_bits buffer per tile?
- Can the existing equivalence-class detection (memcmp on signatures) extend to (trit_sig, conf_bits) pairs trivially? Yes — just memcmp the concatenation.
- For arity-2 with input pairs, max_abs is computed across all 16 pairs. Same shape.

## First Instincts (suspect; to be challenged)

- Path A wins because it uses the more substrate-distinctive kernels (dual-threshold + confidence-weighted distance).
- Per-expression tau scaled to the expression's own max.
- tau_weak = max/4, tau_strong = max/2.
- Storage: extend `expr_bank_t` with a parallel `conf_bits` array per tile.
- Path B is the wrong shape for our use because wildcard semantics eat information at the most informative positions.
