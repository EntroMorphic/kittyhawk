---
cycle: xexpo_design
phase: RAW
date: 2026-05-01
scope: LMM applied to docs/DESIGN_X-EXPO.md and the intent behind it
---

# Raw — xexpo_design

## Stream of consciousness

I wrote the design with confidence I'm not sure I earned. The remediation plan says no kernel work without a measured consumer, and yet I produced a full design including saturation contract, property tests, and an API. I'm framing it as "exploration ahead of the cycle" — true, but is that a fig leaf? Designing a thing makes it more likely to get built. Momentum bias.

The Path A vs Path B decision felt sharp once I worked through the example (a=100 at e=0, b=50 at e=5). Path A matches IEEE-754. But the prior project's whole thesis is that base-2 framings are the wrong ergonomics — and IEEE-754 IS a base-2 framing. Did I just default to the comfortable answer? I don't think so — the math actually works the same way in any base, alignment-to-larger is just the geometry. But I should check.

The "smaller operand vanishes" property of Path A is genuinely concerning. I waved at it as "matches floating-point semantics" — true — but is it what consumers actually want? The multi-table SUM resolver sums distances. If one table has a small distance and another has a large distance, summing them, do I really want the small distance to vanish at the cell level when the exponents differ? That's exactly the precision the resolver depends on. Path A might be a substrate-correct kernel that produces consumer-wrong results.

I removed the `result_block_exp` out-parameter because it's deterministic. But the caller has to compute it via the helper. Did I just shift cognitive load? In some idioms the out-parameter would be the safer choice — fewer chances to forget.

The `|Δ| ≤ 19` precondition is suspicious. I justified it by "values beyond ±19 are out of MTFP19's representable range anyway." That's true for the *result*. But the inputs can have any int8 exponent, and the kernel asserts on Δ rather than the absolute exponent. If I get a=at e=20 and b at e=39, Δ=19 is OK by the precondition, but the resulting e_d=39 is fine for the result. So the precondition is consistent. Just felt convenient.

The biggest gap I noticed only after writing: the consumers I cited (multi-table SUM, multi-tile accumulation) don't do *one* add — they do a SEQUENCE of adds, accumulating into a running sum. `vec_add_aligning` is pairwise. Is the right primitive `vec_accum_aligning(running, new, e_running, e_new)`? That's a different signature. The current design might solve a different problem than the consumers actually have.

NEON deferred — I justified it by "tier 3 is correctness first, no profile evidence." That's defensible but also convenient. ARM lacks integer divide; making the kernel performant is genuinely harder than the design suggests. I sidestepped it.

I designed `sat_flags` as one byte per cell. That's wasteful. I justified it by "MVP, simplest to test." But if the consumer's saturation rate is <0.1%, sat_flags is dead infrastructure regardless of layout. The cycle will tell me if it's needed at all.

## Questions arising

1. Is `vec_add_aligning` the right primitive shape, or is `vec_accum_aligning` what the consumers actually need?
2. Did I default to IEEE-754 alignment because it's familiar, or because it's right? (I think the latter, but should pressure-test.)
3. The "smaller operand vanishes" property — is this what the multi-table SUM resolver wants? Counter-argument: when distances differ in scale, the smaller one IS less informative for the SUM ranking; vanishing is the right behavior. But this needs measurement, not vibes.
4. Is per-tensor exponent storage actually right, or does some consumer want per-tile within a per-block layout?
5. The kernel might be mathematically correct AND mechanically useful AND still earn nothing — if consumer block_exps turn out to be uniform in practice. Have I designed for a problem that doesn't exist?

## First instincts (now suspicious)

- Path A is right. (Probably correct, but only because alignment-to-larger is the only choice that doesn't catastrophically saturate the larger operand.)
- Per-tensor MVP. (Probably right for the named consumers but I haven't verified that the named consumers will be the consumers that drive this — which is exactly what the cycle is for.)
- API without out-param. (Convenient; the caller-recomputes-it pattern works for one rule but breaks if the rule ever changes.)
- Scalar implementation MVP. (True for tier 3 correctness focus, but conceals real engineering cost.)
- Property tests as I sketched them. (10000 samples, half-trit tolerance — the tolerance is the part I'm least sure about. Half-trit is the natural-looking number, not a derived one.)

## What scares me

I designed a kernel for a problem I haven't measured. The discipline says no primitive without consumer demand; I obeyed the *letter* (didn't ship code) but maybe not the *spirit* (produced a vetted design that's hard to walk back). If the cycle reveals the consumers are uniform — or that they need accumulation, not pairwise add — this design is not just unbuilt but *wrongly shaped*.

The other thing that scares me: the design is too clean. The README's "if your synthesis is surprisingly clean, you've done it right" — but it might also mean I haven't found the real complexity yet. The complexity I missed is probably in the consumer's actual call pattern, which I didn't measure.
