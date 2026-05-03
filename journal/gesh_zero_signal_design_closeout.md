---
cycle: gesh_zero_signal_design (P0-1)
phase: CLOSEOUT
date: 2026-05-02
scope: post-synthesize observations on the design cycle; record loop-back triggers; commit to next-step work
companions: gesh_zero_signal_design_{raw,nodes,reflect,synthesize}.md
status: COMPLETE — design cycle committed; awaits substrate-spec amendments and kernel implementation
---

# Closeout — gesh_zero_signal_design

The cycle ran cleanly to a SYNTHESIZE commitment. No mid-cycle reframings. Closeout is thin.

## What the LMM cycle revealed that direct design wouldn't have

A direct "build a wildcard kernel" approach would have produced `m4t_route_wildcard_dist` in isolation. The cycle's REFLECT surfaced the structural coupling between P1 (kernel) and P6 (bank) — they must ship together because their semantics are joint. A standalone P1 against an emergent-zeros bank would over-promote ambiguous matches. A standalone P6 with current Hamming would re-cost wildcards as half-mismatches, defeating their purpose.

This coupling wasn't obvious from RAW or NODES; REFLECT's pressure-test on "what does zero in a tile actually mean across constructors" surfaced it.

The cycle's other substantive contributions:
- **REFLECT downgraded the P3 speedup expectation** from ~2.5× to ~1.3–1.5× at MNIST scale, after honest cost analysis. Saved us from over-promising.
- **REFLECT identified that MNIST is the wrong primary benchmark** for substrate-novelty (Q5). The verification benchmark needs explicit don't-care structure; SYNTHESIZE committed to building synth_wildcard for this.
- **REFLECT promoted the substrate-novelty audit from "criterion" to "gate"** (Gate 3). Without it, we'd have been measuring "does this work" not "is this substrate-distinct."

## What did NOT surface

- No reframing of the substrate-claim itself. The "zero is the substrate's free third state" framing held.
- No conflict between P0-1 work and prior cycles. The class-mean bank, the lattice update, the SDOT kernels — all stay valid; P0-1 adds primitives alongside.
- No reason to revise the four-P0 ordering (zero → exponent → geometric → multi-stage). The ordering held.

## Commitments restated

P0-1 deliverables:
1. Substrate-spec amendments to `m4t/docs/M4T_SUBSTRATE.md` (wildcard semantics, sanctioned input classes).
2. `m4t_route_wildcard_dist` kernel + property tests.
3. `gesh_bank_build_class_wildcard` constructor + tests.
4. `gesh_forward_classify_wildcard` consumer integration + tests.
5. `synth_wildcard.c` benchmark.
6. Gates 1–4 measurements + verdict.

Gates pre-committed in SYNTHESIZE; reproducible verdicts.

## Loop-back triggers

- **Back to RAW** if Gate 3 (substrate-novelty audit) FAILS — the wildcard primitives don't demonstrate base-3-only capability vs base-2 with masks. Then the substrate-claim framing for zero is wrong.
- **Back to NODES** if Gates 1+4 jointly FAIL — wildcards regress on substrate-friendly synth AND on MNIST regression-guard. The primitive set isn't doing what we expected.
- **Back to REFLECT** if Gate 1 INCONCLUSIVE persists across multi-seed validation — the wildcard advantage is real but smaller than the design predicted, and the framing should be more cautious.
- **No loop-back** if Gates 1, 2, 3, 4 produce a clean PASS-or-INCONCLUSIVE-acceptable set. Then P0-1 closes; substrate-spec is amended; P0-2 starts.

## Methodology check

- **Substrate-novelty audit ran throughout the cycle.** RAW enumerated where zero is currently treated as default; NODES tagged each candidate primitive with its base-3-only capability vs base-2 cost; REFLECT pressure-tested each against base-2 alternatives with mask bits; SYNTHESIZE committed to Gate 3 as a hard verification.
- **Multi-seed expectation was pre-committed** for the verification cells (Gate 1's INCONCLUSIVE band requires multi-seed escalation).
- **No code in the cycle.** Per the P0 protocol. Code starts after this CLOSEOUT lands and the substrate-spec amendments are written.

## What this cycle does not finish

This is a **design cycle**, not an implementation cycle. The hard work — kernel implementation, consumer integration, benchmark construction, gate measurement — is downstream. This cycle commits to *what* gets built and *how* it'll be evaluated; *building it* is the next phase of P0-1.

## Next concrete step

Write the substrate-spec amendments (`m4t/docs/M4T_SUBSTRATE.md` §X.Y and §X.Z) per principle 7. **Spec amendment is the gate before kernel implementation.**

After spec amendments land:
- Implement `m4t_route_wildcard_dist` kernel + tests.
- Implement `gesh_bank_build_class_wildcard` + tests.
- Implement `gesh_forward_classify_wildcard` + tests.
- Implement `synth_wildcard.c`.
- Run Gates 1–4.
- Update CLOSEOUT with verdicts.
- Red-team pass.
- Then P0-2.

Each step is a discrete commit with the journal trail it produces.

## Methodology pattern worth carrying forward

This cycle was the first one to **explicitly use the substrate-novelty audit as a gate** (Gate 3 in SYNTHESIZE). Prior cycles applied multi-seed and multi-config gates; this is the first to apply substrate-novelty. The pattern:

> **Every P0 cycle's SYNTHESIZE must include a Gate that explicitly tests "would base-2 with appropriate substitutes match this work?" If yes, the work is correctness-shaped, not substrate-claim-shaped.**

This is the substrate-novelty audit operationalized as a measurement, not just a checklist item. Worth adopting for P0-2, P0-3, P0-4 cycles.

## End of design cycle

Awaiting:
- Substrate-spec amendments.
- Kernel implementation.
- Consumer integration.
- Benchmark.
- Gate measurements.
- Verdict.

Per the P0 protocol, P0-2 (exponent signal) does not begin until P0-1's verification gate produces a verdict.
