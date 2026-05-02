---
cycle: xexpo_design
phase: NODES
date: 2026-05-01
scope: extract discrete points and tensions from RAW
---

# Nodes — xexpo_design

## Node 1 — Path A (align up) was the only defensible choice
The decision between max-exponent (Path A) and min-exponent (Path B) alignment was forced. Path B causes the larger operand to saturate when Δ is non-trivial, destroying the dominant magnitude — catastrophic. Path A matches IEEE-754 alignment semantics: smaller operand truncates, larger preserved.
**Why it matters:** Path A is *not* a base-2 ergonomic; it's the geometry any positional alignment forces, in any base. The discipline ("rage against the trodden") doesn't apply here — the choice is forced by math, not convention.

## Node 2 — "Smaller operand vanishes" is consumer-dependent
Path A's defining property: when Δ is large, the smaller operand truncates to zero post-rescale. The design framed this as "matches floating-point semantics" — true, but ducks the question of whether consumers want this behavior.
**Tension with Node 1:** Path A is mathematically forced; whether the consumer is happy with the loss is empirical.

## Node 3 — The pairwise-add vs running-accumulator gap
The cited consumers (multi-table SUM, multi-tile routed accumulation) accumulate distances/outputs into a running sum across many calls. The design specifies pairwise `vec_add_aligning(dst, a, e_a, b, e_b)`, not running `vec_accum_aligning(running, &e_running, new, e_new)`. The design might solve a non-consumer problem.
**Why it matters:** This is the largest gap in the original design. If consumers want accumulation, the API and saturation semantics both change.

## Node 4 — Per-tensor exponent granularity is consumer-justified, not substrate-justified
Per-block was the spec's stated intent (§7); per-tensor was chosen because the named consumers emit one logical scale per tensor. The justification is consumer-shape, not substrate-philosophy.
**Tension with substrate spec:** §7's per-block intent may need amending if per-tensor is the working answer. Or the design is provisional and the spec stands.

## Node 5 — `result_block_exp` out-parameter removal traded one risk for another
Removing the out-param eliminates "stale exponent" bugs but adds "caller forgets to call the helper" bugs. Both are real failure modes. The choice was for the simpler API; correctness depends on consumer discipline.
**Tension with Node 3:** if the API becomes `vec_accum_aligning`, the running exponent IS necessarily an in-out parameter — the out-param removal becomes irrelevant.

## Node 6 — Saturation contract bound (`3^(e_d − 1)`) is half-derived, half-chosen
Half-trit precision at the result exponent is the natural bound from integer truncation. But "half-trit" was reasoning by analogy to floating-point ULP/2; not strictly derived from the integer-truncation error model.
**Why it matters:** The property test pass criterion depends on this bound. If the bound is wrong (too tight or too loose), the test either rejects correct kernels or accepts wrong ones.

## Node 7 — `|Δ| ≤ 19` precondition is convenient, not obviously natural
Beyond Δ=19, the smaller operand mantissa zeroes by truncation — `int32 / 3^19 ≈ int32 / 1.16e9 = 0` for any input fitting MAX_VAL. The precondition formalizes "the operation is meaningful." But it's also a hard error if violated; should it instead silently zero?
**Tension:** assertion vs graceful degradation. The current design picks assertion.

## Node 8 — `sat_flags` design is speculation about observability
One uint8_t per cell, exactly when the consumer cares about saturation. But the consumer's saturation rate is unknown until the cycle measures it. If saturation is rare (<0.1%), the per-cell layout is wasteful and a counter would suffice.
**Tension with Node 3:** if the kernel is `vec_accum_aligning`, saturation tracking across a sequence of accumulations is a very different shape — likely cumulative, not per-call.

## Node 9 — NEON deferral is honest but conceals real cost
ARM has no integer divide; vectorizing requires reciprocal multiplication (libdivide-style). The design defers this with "tier 3 correctness first" — defensible but conceals that performance work is non-trivial. If the kernel ever needs to run in a hot path, the design's "scalar MVP" is misleading.
**Why it matters:** The cycle should ask whether the kernel is ever in a hot path. If yes, NEON cost belongs in the design.

## Node 10 — The design is hypothetical until the cycle runs
The whole document is conditional on consumers exhibiting heterogeneous block_exp values, on Δ being meaningfully > 0, on saturation being non-trivial. None of those have been measured. The design is a vetted hypothesis, not a vetted spec.
**Tension with discipline:** producing a vetted design before measurement is ALLOWED ("design exploration ahead of cycle") but borders on the violation it claims to avoid. Question: is the threshold "did we ship code?" or "did we make a built thing harder to walk back?"

## Node 11 — IEEE-754 framing is mostly safe but worth pressure-testing
Path A's "matches IEEE-754" justification is correct as math but uses a base-2 idiom for a base-3 substrate. The geometry transfers; the idiom maybe doesn't. NORTH_STAR's "rage against the trodden" applies if the idiom is doing work the geometry isn't.
**Resolution candidate:** state the geometry directly (alignment to larger preserves dominant magnitude), drop the IEEE-754 reference except as historical anchor.

## Node 12 — Spec amendment plan is one-directional
The design plans to amend `M4T_SUBSTRATE.md` §14.2 if the kernel ships, but doesn't plan for §14.2's existing prose to *correct* the design. The substrate spec might know something the design missed; the design treats §14.2 as a sketch to be replaced rather than a constraint to honor.
**Why it matters:** I haven't actually re-read §14.2. The cycle's design memo phase should.

## Tensions summary

- **Path A's "vanish" property** vs **consumer's preference for precision retention** (Nodes 1, 2)
- **Pairwise add primitive** vs **running accumulator primitive** (Node 3) — *largest gap*
- **Per-tensor design** vs **per-block spec intent** (Node 4)
- **Out-param removal simplicity** vs **caller-recompute risk** (Node 5)
- **Half-trit tolerance bound by analogy** vs **bound derived from error model** (Node 6)
- **Hard assertion at Δ=19** vs **silent degradation past Δ=19** (Node 7)
- **Per-cell sat_flags** vs **lighter observability if saturation is rare** (Node 8)
- **Scalar MVP "correctness first"** vs **NEON cost concealed** (Node 9)
- **Design exploration ahead of cycle** vs **discipline violation by built-in momentum** (Node 10)
- **IEEE-754 idiom** vs **base-3-native framing** (Node 11)
- **Design as spec replacement** vs **spec as constraint on design** (Node 12)
