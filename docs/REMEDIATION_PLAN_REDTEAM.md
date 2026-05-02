---
title: Red-team review — Substrate Remediation Plan (Tiers 2 & 3)
date: 2026-05-01
companions: docs/REMEDIATION_PLAN.md
status: COMPLETE — findings folded into REMEDIATION_PLAN.md revision
---

# Red-team — Remediation Plan

Adversarial review of [`REMEDIATION_PLAN.md`](REMEDIATION_PLAN.md). Twelve findings; two are substantive (T2, T10) and reshape tier 3.

## Substantive findings

### T2 — Tier 3's consumer candidates are speculative, not measured

**Finding.** The plan's "discipline check" gate ("no kernel without a named consumer") is correct, but the three candidates listed (multi-tile routed accumulation, multi-table SUM resolver, routed-autodiff gradient accumulation) are *hypothetical* problems. None of them was observed to actually pay a measurable cost in the prior cycle. The prior `mnist_routed_bucket_multi` SUM resolver hit 97.24% with single-block-exponent arithmetic; no measurement says heterogeneity hurt it. The prior libtrain MVP collapsed to one `block_exp`; no measurement says gradient precision suffered.

**Implication.** The gate as written would let tier 3 start because a consumer is "named." But the plan's deeper discipline ("no primitive without *demand*") requires the demand to be measurable, not just narratable. As drafted, tier 3 risks building infrastructure on speculation — exactly the failure mode the discipline exists to prevent.

**Resolution.** Tier 3 must begin with a consumer-discovery cycle, not a design memo. The cycle's deliverable: for each candidate consumer, a measurement that establishes whether the same-block-exponent assumption costs anything on real data. Only candidates that pay a measurable cost qualify as named consumers under the discipline.

### T10 — Tier 3 inherits unverified spec ambiguity

**Finding.** The plan defers the kernel's exact semantics to `M4T_SUBSTRATE.md` §14.2. But the audit's framing ("named-but-unbuilt") implies §14.2 itself is a sketch — written when the spec author anticipated a kernel that has never been forced to ship. Tier 2's fallback ("if a contract is genuinely under-specified, open a journal cycle") applies even more strongly to §14.2: the spec hasn't been pressure-tested against an implementation.

**Implication.** Tier 3's "design (sketch)" section is doing double duty — it's a sketch *and* a tacit spec amendment. Either §14.2 stays the authority and the sketch must conform to it (in which case the sketch should be vetted against §14.2 explicitly), or the journal cycle amends §14.2 and the sketch is provisional. The plan picks neither.

**Resolution.** The consumer-discovery cycle (per T2) also drives a §14.2 review: which assumptions in §14.2 hold under the candidate consumer's pressure, and which need amendment? Spec amendments land before kernel implementation, not after.

## Cosmetic / minor findings

### T1 — "Tier 2" is housekeeping, not remediation

**Finding.** Tier 2 is documentation + asserts + one helper. It changes no algorithm and no measured behavior. Calling it a "tier" alongside tier 3 (a real kernel build) overweights it.

**Resolution.** Keep the work; reframe it as "tier-2 hygiene pass" in the revised plan. Treat it as a precondition for any future kernel work, not as parallel-tier work.

### T3 — Tier 1's verbatim lift was assumed green when the plan was written

**Finding.** The plan was drafted before tier 1 was rebuilt and tested in the new tree.

**Resolution.** Tier 1 is now green (3/3 tests, clean build). Note in the revised plan.

### T4 — `apply_signed` `T` parameter is a breaking API change

**Finding.** The plan adds `T` to `apply_signed`'s signature for assert purposes but doesn't decide between (a) breaking the existing signature or (b) adding a `_checked` variant. The archived consumers used the old signature; any lift from `01MAY26_archived/` will need to know.

**Resolution.** Pick (a) — break the signature. With the archive ignored and the only future callers being post-rebuild code, there is no in-tree consumer to break. Document the break in CHANGELOG.

### T5 — MTFP9 is absent everywhere

**Finding.** The plan covers MTFP4, MTFP19, and MTFP39 mentions. MTFP9 (16-bit, 9 trits) sits between them and was never used.

**Resolution.** State explicitly that MTFP9 is dropped from the active substrate until a consumer asks for it. Type stays in `m4t_types.h` for forward compatibility; no kernels until demanded.

### T6 — Tier 2 wall-clock estimate has no escape

**Finding.** "Half a day" assumes no surprises. If a contract is genuinely under-specified, cost balloons.

**Resolution.** Add a 1-day cap; re-evaluate at the cap rather than grinding.

### T7 — Tier 3 cycle estimate is optimistic

**Finding.** "1 day for design memo" is optimistic for a journal cycle that has to identify a consumer, measure the consumer's actual cost, sketch the API, and produce decision endpoints.

**Resolution.** Revise to "2-3 days" for the consumer-discovery cycle (raw → nodes → reflect → synthesize). Implementation cost (1-2 days) only kicks in if the cycle surfaces a qualifying consumer.

### T8 — "Stays at synthetic-only" outcome is under-specified

**Finding.** The plan says "this is a real result, not a failure" but doesn't say what the substrate's documentation looks like in that state.

**Resolution.** Specify: if tier 3 ends without a kernel, `m4t/README.md`'s status section explicitly states "MTFP-capable substrate, fixed-point-in-practice — no consumer has yet driven the cross-exponent kernel."

### T9 — No mention of CI verification

**Finding.** `build.yml` was copied verbatim from the archive. The plan doesn't verify it still passes against the new structure.

**Resolution.** After tier 1 lands, push and watch CI. If it fails, fix before tier 2.

### T11 — Per-block vs per-tensor block_exp is presented as a binary choice

**Finding.** Tied to T2: the right granularity is consumer-driven. The plan picks "per-tensor for the MVP" without consumer evidence.

**Resolution.** The consumer-discovery cycle decides granularity; the spec sketch is provisional until then.

### T12 — No observability story

**Finding.** Asserts and saturation flags exist; nothing aggregates them. A researcher cannot easily ask "how often did saturation fire on this run?"

**Resolution.** Note as future cycle. Not blocking.

## Revised execution order

Tier 1 (lift) — DONE (3/3 tests green).

**Tier 2 (hygiene pass).** Execute now. ~half-day, capped at one day.

**Tier 3 — three sub-steps, each gated on the previous.**
1. **Consumer-discovery cycle.** Journal cycle (raw → nodes → reflect → synthesize). For each of three candidate consumers (multi-tile accumulation, multi-table SUM, routed autodiff), produce a measurement establishing whether the same-block-exponent assumption costs anything on real data. 2-3 days.
2. **Conditional on (1) producing a qualifying consumer:** §14.2 review + design memo. 1 day.
3. **Conditional on (2) landing a vetted design:** kernel implementation + property-based tests + consumer integration. 2-3 days.

If the consumer-discovery cycle produces no qualifying consumer, the substrate stays at "MTFP-capable, fixed-point-in-practice" and the documentation reflects that. This is a measured outcome, not a failure.

## Two practical consequences for "then execute"

1. **Tier 2 can execute immediately.** The hygiene pass is fully scoped and does not depend on the consumer-discovery cycle.
2. **Tier 3 cannot execute the kernel implementation in this session.** The consumer-discovery cycle is the next-step gate; it requires real-data measurements that take time to set up. The right move is to land tier 2, document the consumer-discovery cycle's intent in a journal RAW file, and stop.
