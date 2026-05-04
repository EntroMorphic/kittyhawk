# Reflections: Elemental Floor Audit

## Core Insight

The substrate's API surface mixes **elementals** (ops that can't be built from other ops) with **composites** (ops that CAN be built from elementals but are kept as kernels for performance). This mixture is fine engineering — composites get fast kernels, elementals get cleanly-defined primitives — but the audit reveals two specific gaps:

1. **shift3 isn't exposed as a runtime op** even though it's the natural base-3 positional primitive and is implicit everywhere in MTFP arithmetic.
2. **select isn't exposed as a clean trit-controlled mux** at the cell level, even though it's the conditional primitive needed by every composite that branches.

Closing these two gaps (plus auditing the docs of existing composite kernels) is what "closing the elemental floor" actually means.

The cell-level elemental floor is **5 ops + 3 constants**:

```
ops:       add, neg, shift3, sign, select
constants: -1, 0, +1
```

(Or 4 ops if you derive neg from select + constants. Engineering choice; keep neg as a primitive because it's fast and natural.)

Everything else — mul, sub, div, max, min, eq, abs, exp, log, sin, cos, sqrt — is **composite**. Some get fast kernels in the substrate (mul, max, min, eq, sub); others get built at the consumer level (exp, log, etc.).

## Resolved Tensions

**T1 (level of abstraction) — RESOLVED.** Cell level. Trit-level analysis is interesting but not operational (trit-add doesn't return a trit; trit-level ops are really TBL lookups, not standalone primitives).

**T2 (is neg in the floor?) — RESOLVED engineering-side.** Mathematically derivable from select + constants. Pragmatically: keep neg as a substrate primitive because (a) it's fast (bit-swap in the trit pack encoding — already 5 instructions per 64 trits), (b) it's used everywhere in cell arithmetic, (c) deriving it at the consumer level adds dispatch overhead for no benefit.

**T3 (is sign elemental at cell level?) — RESOLVED pragmatically.** Mathematically composite (find-leading-nonzero needs iteration). Substrate-API-level: atomic, exposed as `m4t_route_threshold_extract` (with tau=0 = sign extraction). Keep as primitive at the API.

**T4 (composite kernels stay) — RESOLVED.** Yes, stay. The substrate provides fast kernels for performance-critical composites (mul, sub, max, min, eq, ternary_matmul, SDOT). They get documented as composite, not elemental, but the kernels remain.

## Challenged Assumptions

**A1 ("six frozen primitives" implies six SUBSTRATE ops).** Possibly false. The foundation says "all required compute math derives from ~6 frozen primitives." That can be read two ways:
- **Reading I:** the substrate offers 6 primitive ops, from which all compute math is built.
- **Reading II:** there exists a set of ~6 elemental ops (a mathematical fact) that the substrate must contain.

Both are consistent with the elemental floor being 4–5 ops. The "~6" is approximate; "5 ops + 3 constants" or "4 ops + 3 constants + neg-as-fast-derived" are both close to 6 in spirit.

**A2 ("exp and log are foundational").** The original P1-1 cycle assumed yes. The audit says no — exp and log are CONSUMER-LEVEL constructions built by iterating elementals (Taylor for exp, Newton for log). They might still be added to the substrate as performance kernels (Path A from prior cycle), but that's an engineering choice for speed, not a foundational requirement.

**A3 ("the substrate is short by exp/log").** False per A2. The substrate is short by **shift3** (as exposed runtime op) and **select** (as cell-level cleanly-named primitive). Those are the actual missing elementals.

**A4 ("division must be a substrate primitive").** Possibly false. Division is composite (long division = iterated shift + sub + select). The substrate doesn't need a `div` kernel any more than it needs a `mul` kernel mathematically — both are composites. Mul gets a kernel because matmul is the substrate's hot path; div would get a kernel only if some hot path needed it.

## What I Now Understand

**The audit's product is NOT new primitives in the substrate — it's clarity about what IS and ISN'T elemental.** Two new primitives (shift3, select) close the floor. Documentation cleanup names existing composite kernels as composite. Vision claim #1 is then satisfied with a defensible, audited floor.

**This work is small.** The elemental floor analysis was the hard part (and the conversation just resolved it). The actual code changes are:

1. Add `m4t_mtfp_shift3(cell, k)` — multiply by 3^k. Implementation: increment per-block exponent by k, with clamp/round on negative k. Maybe 30–50 lines of code, plus tests.
2. Add `m4t_route_select(c, a, b, d)` or similar — trit-controlled cell-level mux. Implementation: a few lines of conditional select, NEON-vectorizable. Plus tests.
3. Documentation pass: name composite kernels as composite in their headers. Mostly comment edits.

**Total scope: ~1 week.** No open numerical-methods problem. No substrate spec amendment beyond defining the two new primitives. No consumer-demand question (foundation directly justifies).

**This is the cleanest forward step the project has had in two weeks.** The R-track failed, R1-fork resolved F3, P1-1 looped back, and now the audit gives a small concrete deliverable that closes vision claim #1 substantively.

## A note on the cycle's evolution

Three observations about the meta-pattern:

1. **The owner's "Mul, if made of two conditions, is composite" was the inflection point.** Without that pressure, this cycle would have written a synthesis that treated mul as elemental and added exp/log as substrate primitives. Owner pressure on the foundational analysis is the right shape for foundational research.

2. **The "no consumer demand" framing being eliminated unblocked the analysis.** Without that bias, the question becomes "what's mathematically elemental?" instead of "what does some hypothetical consumer need?" Cleaner question, cleaner answer.

3. **The previous cycle's loop-back trigger (P1-1) was not failure — it was the discipline working.** The cycle correctly refused to ship a plan that was built on a false assumption. The audit then arrived at the right answer.

## Remaining Questions

- shift3 by negative k requires rounding. Round-to-nearest-even (per substrate's existing rounding discipline)? Or truncate? The substrate's odd-divisor lemma should extend cleanly here since 3^|k| is always odd.
- select API: width-uniform (all inputs same width), or polymorphic? Probably uniform for simplicity.
- Should `m4t_route_threshold_extract` be renamed to `m4t_sign` to match the elemental nomenclature, or kept under its current name with documentation that it IS sign-at-tau-0? Keeping current name is least-disruptive; new docs explain the role.
- Should existing composite kernels (`m4t_trit_mul`, `m4t_trit_sub`, `m4t_trit_max`, `m4t_trit_min`, `m4t_trit_eq`) get an `_atomic` or `_composite` suffix, or just header comments? Comments are less disruptive.
