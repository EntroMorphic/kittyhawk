---
title: Thesis brief
companions: NORTH_STAR.md · m4t/docs/M4T_SUBSTRATE.md · docs/FINDINGS.md
status: substrate complete + BitNet inference at ~92% strict pass + first direct Part-B evidence on N4 sparse attention (2026-05-10)
---

# Thesis

The substrate claim, in one sentence:

> Routing-first base-3 (MTFP) computation on commodity ARM is not just achievable; it is the natural shape of the hardware, and it surfaces structural primitives that base-2 framings hide.

Two parts. Both are falsifiable.

## Part A — base-3 is the natural shape

**Claim.** SDOT, TBL, masked-VCNT, and `vmull_s32` are ternary primitives wearing base-2 documentation. A substrate that reads them as ternary will run at hardware-native cost; a substrate that pretends they are general-purpose will pay a translation tax.

**What would falsify it.** Microbenchmarks showing that the ternary-native kernel is *no faster* than a well-tuned dense kernel running on the same hardware, when both are operating at the same effective accuracy on the same task. If the hardware-shape advantage is real, ternary-native must measurably win at one of: throughput, energy, or instruction count, on at least one realistic workload.

## Part B — routing is essential, not optional

**Claim.** In a base-3 environment, 1/3 of cells carry zero by construction. Routing-first architectures exploit this; dense architectures pay for it. As task complexity rises, the gap should widen, not close.

**What would falsify it.** A real benchmark on which:
- A routing-native consumer matches a dense baseline at *equal* compute, AND
- The routing advantage does not widen as task structure becomes richer (more classes, more modalities, more compositional structure).

The previous cycle measured a 60pp gap between random-U + sign-only routing and learned-U routing on a 10-class toy. That measurement is consistent with Part B but does not establish it; it establishes that *some* routing structure beats *no* routing structure on that toy.

**As of 2026-05-10: first direct Part-B evidence on a real workload.**
Cycle 2 of the LMM-derived mode-shift plan executed N4 (post-hoc sparse
attention via the substrate's `m4t_route_threshold_extract` +
`m4t_route_distance_batch` primitives) on BitNet b1.58-2B-4T inference,
24-prompt × 4-arm × 6-k battery. All three pre-committed EVIDENCE gates
passed:
- At k=64, routed within 10pp of dense (0pp gap)
- At k=16, routed beats random by +16.7pp (75% vs 58.3%)
- The gap (routed − random) widens monotonically with sparsity:
  k=16 +4 prompts, k=8 +5, k=4 +6

Routed at k=4 (22/24) outperforms dense (19/24) on three loop-prone
prompts. Mechanism hypothesis (not confirmed): aggressive substrate-routed
attention selects diverse-by-signature K positions, breaking attention-
loop dynamics that dense locks into.

This is one workload (BitNet inference) at one model scale (2B params)
under one decoding strategy (greedy). Not a closure on Part B — but the
first time it has been TESTED with pre-commit gates and PASSED. Per
`journal/cycle2_full_battery_findings.md`.

## Closed questions (substrate-side)

1. ~~Does cross-block-exponent MTFP arithmetic earn its complexity?~~ **CLOSED 2026-05-01.** Built ahead of measured consumer demand under owner authorization. The kernel ships in `m4t_mtfp_vec_accum_aligning` with 14 property tests; the substrate is now floating-point in base 3 at per-tensor exponent granularity. Whether any future consumer's call pattern actually exercises the cross-exp path is a usage-study question, not a substrate question. See `journal/xexpo_design_*` and `journal/xexpo_kernel_redteam.md`.

## Open questions for the consumer-layer rebuild

1. **What benchmark is the substrate's right arbiter?** MNIST and Fashion-MNIST are base-2-framed. CIFAR-10 hits a representation tax that base-3 alone does not close. The consumer-layer rebuild should pick its arbiter deliberately rather than defaulting to image canon. **As of 2026-05-10**, BitNet b1.58-2B-4T inference (`gesh/bitnet/`) runs end-to-end on the substrate with ~92% strict pass on a 24-prompt battery (factual / definitional / narrative / math / code / structured-output / long-context / edge categories). Beyond Part-A, the same substrate is now the platform for the first Part-B evidence (above) — N4 sparse attention demonstrated routing-essentiality with pre-committed gates passing. BitNet inference is the substrate's first real-LLM consumer AND the first workload on which both halves of the thesis have been tested.

2. **Is the SDOT-native MTFP4 path the load-bearing primitive?** The substrate's strongest "MTFP-as-hardware" claim sits at MTFP4 × ternary → MTFP19 (Case W per §8.4). The kernel ships and is property-tested at K up to 1M. If a routing consumer can drive this kernel into a benchmark win, that is the substrate-claim's cleanest demonstration. Untested at the benchmark level until consumers come back online.

3. **Does the LMM cycle methodology generalize beyond this codebase?** Auditing the prior implementation's journal showed 37 cycles, including self-revising closeouts and explicit substrate-discipline gates. The substrate rebuild added three more: `xexpo_design`, `xexpo_kernel_redteam`, `xexpo_spec_amend`, plus `m4t_matmul_redteam`. Each found issues a single pass would have missed. The pattern of same-author adversarial review-after-build appears robust within this project; cross-project generalizability is open.

## Re-read

When a measurement makes the claim look obvious, check Part A and Part B separately. The advantage on a single benchmark is consistent with the thesis but does not establish it. The thesis stands or falls on whether the *structure* — routing-first, base-3-native, hardware-shape-aware — beats well-tuned base-2 alternatives across a range of tasks.
