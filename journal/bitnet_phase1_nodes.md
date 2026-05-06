---
cycle: bitnet_phase1
phase: NODES
date: 2026-05-06
scope: critical decisions, tensions, and constraints extracted from the RAW phase. Surfaces what must be resolved before the SYNTHESIZE plan can be load-bearing.
companions: bitnet_phase1_raw.md
---

# Nodes — bitnet_phase1

## Decisions that must be made

### N1 — Architecture verification before planning
The RAW exposed that I'm planning against an architecture I haven't actually read. Phase D's scope (RMSNorm, RoPE, LUTs) is asserted, not verified. **Decision before SYNTHESIZE:** read BitNet b1.58-2B-4T's `config.json` + the paper, list the actual primitives the model uses, confirm or revise the gap list. This is not optional — it's the source of the most dangerous asymmetry between plan and reality.

### N2 — Numerical fidelity success criterion
"Match HF's reference" is incoherent as a target if MTFP19 and bf16 round differently across 30 layers. **Decision before SYNTHESIZE:** define the precise gate. Candidates:
- (a) Bit-exact at single-layer level only; tolerated divergence end-to-end.
- (b) Tolerance band on per-layer outputs (relative error ≤ ε), measured end-to-end.
- (c) Task-quality match: HF and m4t agree within X% on a held-out benchmark, regardless of per-layer divergence.
- (d) "Better than HF on at least one axis" — looser, more honest about what we're demonstrating.

These are not equivalent. (a) is cheapest to verify but the weakest claim. (c) is strongest but most expensive. (d) is strategic but blurs the validation message.

### N3 — Quantization protocol fidelity
BitNet's training-time quantization rule (absmean) is specified in the paper. The substrate has `ternarize_absmean` in `audit/tristate_l4_strong.c` (introduced for TD-4) but not in libm4t. Inference doesn't re-quantize weights (they're already ternary at load time), but it DOES quantize activations layer-by-layer. **Decision before SYNTHESIZE:** is the activation quantization rule load-bearing? Does BitNet's exact rule need to live in libm4t?

### N4 — Substrate gap closure path: kernel-level vs research-level
RMSNorm needs a `rsqrt(mean(x²) + ε)`. The substrate has no sqrt. Options:
- (a) LUT-based rsqrt (small input range; bounded approximation).
- (b) Newton-Raphson iteration with a magic-number initial guess.
- (c) Borrow rsqrt from `<math.h>` (FP path — violates project rule).
- (d) Reformulate RMSNorm to avoid the sqrt (research cycle).

(a) and (b) are kernel-level. (d) is a research cycle. (c) is non-starter. **Decision before SYNTHESIZE:** which option commits us, and what's the fallback plan if it fails to converge to acceptable quality?

### N5 — RoPE design choice
RoPE applies rotation `(cos θ, sin θ)` per pair of dimensions, where θ depends on position and dimension index. Options:
- (a) Pre-compute `(cos, sin)` LUT sized to context_length × hidden_dim/2. For BitNet's likely context (4096 or 8192) and hidden (~1536 or 2048), table size is ~16-32 MB. Storage cost.
- (b) On-the-fly computation. Needs trig primitives or polynomial approximation.
- (c) Hybrid: small LUT of fundamental angles + cumulative multiplication.

**Decision before SYNTHESIZE:** which?

### N6 — Phase D vs Phase B sequencing
RAW noted: shape mismatch surfaces in Phase B (when we actually call kernels with BitNet's shapes). But Phase D presupposes the kernels we build will be useful. **Decision:** do Phase D first (commit to substrate gap closure under speculation), or do a thin Phase B harness FIRST to surface what's actually needed (smaller speculation, more re-work)?

### N7 — Phase 1's relationship to the broader arc
RAW concluded: Phase 1 is engineering check, not research validation. The deep test is Phase 2 (gradients) and Phase 3 (training). **Decision before SYNTHESIZE:** how does Phase 1's success criterion relate to readiness for Phase 2? What does Phase 1 specifically need to prove for Phase 2 to be feasible?

## Tensions

### T1 — Speed-up phase D vs Reduce risk by phase D
Doing Phase D first (substrate gap closure) feels orderly. But it commits substrate work that Phase B might invalidate. The tension: speculative substrate investment vs. delaying substrate work until consumer demand surfaces it. **The project rule** says "demand-gated wiring" — implying B-before-D. **The convenient ordering** says D-before-B because gaps block composition.

### T2 — Honest fidelity gate vs Achievable fidelity gate
Numerical fidelity (N2): the cleanest target ("bit-exact match HF") is unachievable. The most useful target ("task quality") is expensive. The most honest target ("we differ from HF, here's the magnitude, here's why") is hard to communicate as a binary go/no-go. **Tension:** what's the gate that's both useful and achievable, without becoming a moving goalpost?

### T3 — Substrate-extension vs Architecture-conformance
RMSNorm needs sqrt. We could (a) extend the substrate (add sqrt primitive) or (b) reformulate RMSNorm (research cycle). Extension is conservative (substrate grows to match BitNet's needs). Reformulation is base-3-native (don't carry base-2 architecture choices into base-3). **Tension:** which way are we biasing the substrate's identity? More extensions = "we run BitNet." More reformulations = "we redesign ML for base-3."

### T4 — Recognition-borrowing vs Independent contribution
Phase 1 produces "we ran Microsoft's model on our substrate." That's load-bearing for credibility but isn't novel ML. **Tension:** how much of Phase 1's framing leans on Microsoft's recognition, vs how much frames Phase 1 as a stepping stone to something we contribute?

### T5 — Phase 1 ambition vs gate for Phase 2
If Phase 1 sets the bar low (single forward pass, single test input, "doesn't crash"), Phase 2 inherits an under-tested substrate. If Phase 1 sets the bar high (full quality match across multiple benchmarks), Phase 1 swallows Phase 2's budget. **Tension:** where do we draw the line that lets Phase 1 conclude AND lets Phase 2 start with confidence?

### T6 — Skill gap vs schedule honesty
RAW acknowledged: I haven't done LLM inference engineering before. KV cache, generation loop, sampling, tokenizer wiring — learning-while-building. **Tension:** the schedule estimate either has to absorb this (soft estimate, possibly long), or has to acknowledge we're operating without prior reps in this domain (no estimate; pace it work-unit by work-unit).

## Constraints

### C1 — Project rule: NEON-only production
RMSNorm and RoPE primitives must be NEON paths. No FP fallbacks. No `<math.h>` calls in production. `_scalar_ref` test oracles allowed. This constrains N4 and N5's options.

### C2 — Project rule: no random weights
We're loading real BitNet weights (not random). Constraint already aligned with the plan; named for clarity.

### C3 — Bit-exact verification gates per kernel
Every new kernel (RMSNorm, RoPE, sqrt or its substitute, LUT-backed silu/softmax) must have NEON-vs-scalar_ref bit-exact tests. This is non-negotiable per the no-scalar audit's discipline lift. Non-trivial work per primitive.

### C4 — Substrate can't widen its license-relevant footprint
Substrate is MIT. Pulling in BitNet weights (Microsoft's release; check license) for tests is fine. Pulling in a tokenizer (likely HF's; check license) for tests is fine. Distribution / packaging concerns are Phase 4, not Phase 1.

### C5 — Single-threaded, NEON-only, aarch64-only
Substrate's invariants: single-threaded at the opcode level, NEON+aarch64 required. Phase 1 inference inherits these. We don't run on x86. We don't multi-thread. (Multi-threading the inference is a Phase 4 concern, possibly never if the substrate's identity is "single-thread base-3 reference implementation.")

### C6 — Memory footprint must be reasonable for development
Loading a 2B-parameter model: at int8 ternary, ~2 GB; at 5-in-8 packed, ~400 MB. KV cache for context 4096 at hidden 2048 at MTFP19 (4 bytes/cell) per layer × 30 layers ≈ 1 GB. Total: ~2 GB working memory. Apple Silicon dev machines typically have 16-64 GB; this fits.

## Open questions that surfaced

The RAW questions were:
1. Architecture details (resolved by N1's pre-SYNTHESIZE step).
2. Quantization protocol (resolved by N3 decision).
3. MTFP19 vs bf16 range (subsumed in N2).
4. LUT applicability (subsumed in N4 / N5).
5. RMSNorm sqrt (N4).
6. RoPE rotation primitive (N5).
7. End-to-end success criterion (N2).
8. Architecture identity commitment (T3).
9. Phase 1 ↔ broader arc (N7).
10. Failure budget (T6).

Subset of these can be resolved at SYNTHESIZE; subset are open questions that the cycle won't close (N7 in particular is a strategic question for the user).
