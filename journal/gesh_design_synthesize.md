---
cycle: gesh_design
phase: SYNTHESIZE
date: 2026-05-01
scope: actionable plan; reframes Gesh from production-infra to research-probe; commits to a benchmark; converts six provisionals to pre-committed gates
companions: ../Documents/GESH/GESH_DESIGN.md · gesh_design_{raw,nodes,reflect}.md
---

# Synthesize — gesh_design

## What this synthesis is

A reframing of the Gesh proposal from "production-infra routing layer with three mechanisms" to "research probe testing the architectural-unification claim on a deliberately-chosen benchmark." Concrete benchmark proposed. Six §6 provisionals converted to pre-committed gates. Build sequencing flipped from all-three-from-start to incremental-with-measurement.

## Cycle conclusion

The Gesh proposal as written is shaped by attention's surface area; the test of Gesh is whether routing-first base-3 beats base-2 attention on a deliberately-chosen task. Without picking the task, every other design decision is a guess.

The architectural-unification claim — "attention-plus-FFN collapsed into routed retrieval from a frozen ternary store" — is the substrate's THESIS Part B demonstration. It's testable. The three Gs are mechanism-hypotheses about what the demonstration needs.

The minimum viable cycle: pick the benchmark, build the simplest Gesh that could plausibly work on it, measure where it fails, add the G that addresses the specific failure. This is what §7 of the design proposed; the build plan should commit to it.

## Key decisions

### D1 — Benchmark commitment: induction-head (or equivalent associative recall)

**Choice:** an induction-head-style synthetic associative-recall task (e.g., "given a sequence with a marker, retrieve the token that followed the previous occurrence of the marker").

**Why this benchmark:**
- **Tests the unification claim directly.** Induction-head is the canonical attention-success-story; it's *the* mechanism transformer-mechanistic-interpretability identifies as where attention earns its keep.
- **Has hierarchical structure that exercises stage 1.** Different parts of the sequence carry different roles; the global signature can route to "the part of memory that holds completions of marker-X."
- **Has ground-truth retrieval.** Routing decisions are inspectable: did Gesh route to the right region? Different from final-accuracy-only metrics.
- **Cheap to iterate.** Sequences of length 64–256, vocab of 32–256, generation is closed-form. Whole training run in minutes on CPU.
- **Substrate-claim relevant.** If routing-first base-3 can do induction-head, attention's claimed exclusive territory has a substrate-shape competitor.

**Why not MNIST:** no hierarchical structure for stage 1 to exercise; "Gesh works on MNIST" wouldn't test the unification claim.

**Why not Go:** the substrate-distance refinement cycle's Go signal was cycle-specific; without re-deriving on the new substrate, it's not a fresh prior. Could be the next-cycle benchmark; not the first.

**Decision endpoint:** if induction-head doesn't suit (e.g., turns out to be easier than expected and stage 1 doesn't exercise), pivot to a small character-level sequence-modeling task on a tiny corpus.

### D2 — Build sequence: stage 2 alone first, ablation-driven escalation

The simplest Gesh that could plausibly do induction-head:

**Phase A (Stage 2 only):**
- Single bank of `T` tiles (no regions, no global stage).
- Local signature via PCA-initialized **ternary** projection (no MTFP4).
- Hard top-k tile selection (no Gumbel-softmax).
- STE backward through ternary quantization.
- `apply_signed` accumulator for tile combination (same-block-exp; cross-exp not exercised yet).
- Pre-committed gate: if Phase A reaches >X% on induction-head matching a small attention baseline, the unification claim has its first evidence and the three-Gs hypothesis simplifies. Specifically: **if Phase A hits within 5pp of a 1-layer attention baseline**, declare a positive substrate-claim measurement.

**Phase B (add Global stage IF needed):** triggered if Phase A fails because the bank is too large for stage 2's top-k to find the right tiles efficiently. Adds the Global signature + region partition. Build only if measurement says the failure is bank-size-driven, not projection-quality-driven.

**Phase C (escalate projections IF needed):** triggered if Phase A or B fails because ternary projections lose too much information. Path 1: PCA-on-bank (different geometry, same kernel). Path 2: MTFP4 projections (more capacity). Try Path 1 first; cheaper.

**Phase D (add training-only mechanisms IF needed):** triggered if STE fails to train. Add Gumbel-softmax. Add periodic refresh. Each escalation gated on a specific failure mode.

### D3 — Pre-committed gates for the §6 provisionals

| §6 question | Provisional choice | Cycle gate |
|---|---|---|
| 6.1 Region assignment | Clustered partition | **Stage 1 deferred until Phase B is reached.** No region assignment until then. |
| 6.2 One/two thresholds | Two thresholds | Test single-threshold first (existing kernel); two-threshold only if symmetric is insufficient. |
| 6.3 Per-region/shared local proj | Per-region | Stage 2 only in Phase A → no regions → shared by definition. Per-region question deferred to Phase B. |
| 6.4 Refresh frequency | Every 1000 steps | Phase A: no refresh (single-bank, frozen signatures). Refresh question deferred to Phase B/D. |
| 6.5 dim_g, dim_l, k_g, k_l, R | Provisional values | Phase A: dim_l only, swept against attention-baseline accuracy. Other dims deferred. |
| 6.6 Backward pass | Build it from start | Build STE from start (Phase A includes); Gumbel-softmax in Phase D iff STE fails. |

Each gate has a falsifiable measurement on induction-head.

### D4 — Substrate vs libglyph kernel placement

**Default to libglyph.** Build Gesh entirely in libglyph using existing substrate primitives:
- `m4t_route_threshold_extract` (single-threshold; existing) for ternary signature extraction.
- `m4t_popcount_dist` (existing) + sort for top-k retrieval. (`m4t_route_topk_abs` exists for sign-magnitude; for Hamming-distance top-k, libglyph does the sort.)
- `m4t_route_apply_signed` (existing) for tile combination.

**Promote to substrate only when profile shows it's earned:**
- `threshold_extract2` (asymmetric two-threshold) — promote only if Phase A measures that symmetric thresholds are insufficient.
- `signature_match` — keep as libglyph composition; promote only if profile shows the composition is hot.

**Result:** Phase A may need zero new substrate primitives. The Gesh proposal's "two new hot-path kernels" is provisional.

### D5 — Frozen-bank framing

Treat the frozen-bank constraint as the design's central architectural commitment, not an implementation detail. The cycle measures Gesh against a small joint-trained-attention baseline; if Gesh on a frozen bank matches or exceeds the baseline, the frozen-bank choice is validated. If not, we know the cost of substrate-purity at this task and decide whether to pay it.

### D6 — Three-Gs framing reframed

The design's "three Gs" framing stays as a hypothesis structure, with explicit clarification:

> The three Gs are mechanism-hypotheses about what the routing-first substitute for attention needs. The cycle tests them by ablation: build the simplest Gesh that could work, measure where it fails, add only the G that addresses the failure. The architectural-unification claim is the load-bearing test, not the three Gs.

## Action plan

### Action 1 — Pick the benchmark concretely (today, no code)

Specify the induction-head task at the level of:
- Sequence length (proposed: 128).
- Vocab size (proposed: 64).
- Marker frequency (proposed: 5–10% of tokens are marker candidates).
- Dataset size (proposed: 100k sequences for training, 10k for eval).
- Attention baseline shape (proposed: 1-layer single-head attention with `d_model=64`, `d_head=64`).
- Substrate-claim accuracy bar: **Gesh within 5pp of attention baseline** = positive substrate claim.

Output: a half-page benchmark spec in `journal/induction_head_benchmark.md` (or similar).

### Action 2 — Sketch Phase A end-to-end (next session, code)

Write the Phase A consumer in libglyph code (no new substrate primitives). The sketch surfaces:
- What libglyph functions need to exist (likely a small `glyph_gesh.{h,c}` module).
- What data shapes the bank takes.
- What the training loop looks like with STE + hard top-k.
- Concrete measurement of where existing substrate composes vs where it strains.

This sketch may or may not run; it's the "writing the code reveals the missing primitives" step.

### Action 3 — Run Phase A on induction-head (within ~1 week)

Train Phase A. Compare to the attention baseline. The gate-A measurement decides:
- **Within 5pp of baseline:** unification claim has positive evidence. Cycle pivots from "does Gesh work?" to "what's the minimum Gesh to claim the win?"
- **5–15pp gap:** specific failure mode determines next phase (B for bank-size, C for projection capacity, D for training stability).
- **>15pp gap:** Phase A is insufficient; question what failure mode dominates and address it before scaling Gesh.

### Action 4 — Document each phase as its own LMM cycle

Phase A cycle records the Gesh-base measurement.
Phase B/C/D cycles, conditional on phase-specific gates, each get their own RAW → SYNTHESIZE.

This keeps the cycle structure incremental and matches the §7 framing.

## Success criteria

- [ ] Induction-head benchmark specified concretely (sequence length, vocab, accuracy bars).
- [ ] Phase A Gesh sketched in libglyph code.
- [ ] Phase A trained and measured against attention baseline.
- [ ] Each subsequent phase gated on a measured failure mode of the previous phase.
- [ ] Substrate primitives added only when profile shows they're earned (default: zero new substrate primitives).

## What this synthesis deliberately does not decide

- The exact accuracy bars for Phase B/C/D gates. These are set when Phase A's measurement comes in.
- Whether the cycle generalizes beyond induction-head. The first benchmark is a probe; if Phase A passes, the next cycle picks a richer benchmark. If Phase A fails, the next cycle diagnoses the failure mode.
- The MTFP4 vs ternary projection question in absolute terms. Tied to Phase C's gate; ternary-first by default.
- Whether the architectural-unification claim is true. Phase A is one data point; the claim survives or falls based on accumulated evidence across multiple benchmarks. First-cycle scope is "does Gesh exhibit the unification on induction-head?" — not "is the unification universally true?"

## Loop-back triggers

- **Back to RAW** if induction-head turns out to not exercise stage 1 either (e.g., the global structure isn't actually engaged by the task as designed). Re-pick the benchmark.
- **Back to NODES** if Phase A's failure mode doesn't match any of the three Gs cleanly (i.e., adding one G doesn't address the failure). The three-Gs framing might be incomplete.
- **Back to REFLECT** if the unification claim itself is challenged by Phase A's measurement (e.g., the gap is so large the architectural framing is wrong).
- **No loop-back** if Phase A passes the gate. That's the wood cutting itself; proceed to Phase B/C/D as needed.

## Methodology note

This is the third LMM cycle on this codebase under the new substrate. The pattern is now consistent:

1. **Cycle proposes (or reviews) a design.**
2. **REFLECT phase finds that the design is shaped against a wrong reference frame** (substrate spec, attention surface, prior-cycle priors).
3. **SYNTHESIZE reframes** against the correct reference frame and produces a more incremental, measurement-driven build plan.

For Gesh: the wrong reference frame was attention's surface area. The correct reference frame is the task's demand. The cycle's job was to surface the difference.
