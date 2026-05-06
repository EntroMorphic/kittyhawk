---
cycle: bitnet_phase1
phase: REFLECT
date: 2026-05-06
scope: analyze the nodes against the now-verified BitNet architecture; resolve tensions where possible; surface what stays open for SYNTHESIZE.
companions: bitnet_phase1_raw.md · bitnet_phase1_nodes.md
---

# Reflect — bitnet_phase1

## What the architecture verification (N1) actually surfaced

I planned against an imagined transformer. The verified spec is meaningfully different from my assumptions:

| What I assumed | What BitNet b1.58-2B-4T actually uses |
|---|---|
| RMSNorm | `subln` (Sub-Layer Normalization — RMSNorm placement variant; same primitive) |
| SwiGLU FFN | **ReLU² FFN** (much simpler — no smooth nonlinearity in FFN) |
| Smooth nonlinearity LUTs needed | LUTs only needed for **attention softmax**, not FFN |
| Standard MHA | **GQA 4:1** (20 Q heads, 5 KV heads) |
| MTFP19-or-bf16 activations | **A8** (int8 absmax-quantized per-token) |
| Hidden dim ~1536 | **2560** |
| FFN dim ~3000 | **6912** |
| 30 layers (guess) | 30 layers (correct) |
| RoPE (correct) | RoPE with `theta = 500000.0` |
| `norm_eps = 1e-6` (guess) | `1e-5` |
| 4096 context (guess) | 4096 (correct) |

**Net effect of the corrections:**

1. **Phase D shrinks.** ReLU² is trivial (`max(0,x)²` — direct compute, no LUT). The smooth-nonlinearity LUT generator (TD-14) is needed *only* for softmax. That's one LUT, not several. TD-14's restoration scope shrinks correspondingly.

2. **Phase D gains a hard problem.** subln *is* RMSNorm-shape: `x * rsqrt(mean(x²) + ε)`. The substrate has no rsqrt. This was the gap I was most worried about, and the verification confirms it's real. Newton-Raphson with a magic-number initial guess is the standard technique; would need to be bit-exact-verified against a scalar reference.

3. **Phase D gains a new node.** The A8 activation format is *not* a substrate primitive shape. A8 is `int8 × per-token FP scale` — base-2 framing. The substrate has int8 cells (m4t_mtfp4 — but base-3 framing) and int32 cells (m4t_mtfp_t — base-3 framing). A8 is structurally different.

4. **GQA 4:1 doesn't change kernel surface** but changes the compute pattern. K/V tensors are 1/4 the size of Q. Memory layout matters; matmul calls don't.

5. **No-bias-anywhere is a simplification.** Inference doesn't compute `Wx + b` ever — just `Wx`. No bias-add primitive needed.

The single most important correction: **A8 activations are a fork**, not a kernel-shape choice. Working through this is the central reflection of this cycle.

## The A8 fork (resolution of N3 + new tension)

BitNet's spec quantizes activations to int8 absmax per token at every layer boundary. The math is:
```
scale = absmax(x) / 127
x_int8 = round(x / scale)        # stored
x_dequant = x_int8 * scale       # used
```

Two paths for the substrate:

**Path α — Match the spec.** Add an `m4t_a8_t` primitive: int8 cell + FP scale per token. Quantize at layer boundaries; de-quantize to MTFP19 for matmul; store back to A8. Output approximates HF reference closely; numerical fidelity gate can target tight bounds.

**Path β — Run at MTFP19 throughout.** Skip activation quantization entirely. Substrate carries activations in MTFP19 between layers (more precision, more memory). Output diverges from HF reference but possibly with *higher* fidelity (less rounding loss). Numerical fidelity gate becomes "match HF on task quality, not on per-cell value."

Path α is reference-conforming and engineering-flat. Path β is base-3-native and embodies the project thesis ("don't pretend ternary is base-2; use the substrate's actual numeric system"). Both are defensible. Path α is more conservative for Phase 1 (validation step); path β is more aligned with the broader arc (base-3 ML).

**My read:** Path β is the right end-state but Path α may be the right Phase 1 step. Phase 1's job is "kernels compose correctly" — adding a base-2-framed primitive (A8) in service of matching the reference is acceptable for validation. Phase 2+ can revisit and ask "do we still need A8 once we control the whole pipeline?" This is a soft argument; the user might disagree.

The user's prior framing — "BitNet's end-to-end inference is conflated with and based on base-2; we built a substrate to serve and honor ternary ML" — suggests Path β alignment. Path α leans toward the conflation we're trying to expose. I'm leaning Path α for Phase 1 pragmatism, but flagging this as a strategic question for the user.

## Resolution of T1 (D-vs-B sequencing)

The project rule says demand-gated wiring. Phase D first (substrate gap closure) presupposes demand we haven't measured.

The right ordering is **thin B → D → full B**:

- **Thin B**: a minimal harness that loads BitNet weights and tries to run *one transformer block forward pass*, using whatever the substrate has, with stubs for missing primitives. Discovers what shapes the kernels are actually called with. Surfaces composition issues that planning can't predict.
- **D**: substrate gap closure, scoped by what thin B revealed.
- **Full B**: real per-layer comparison vs HF reference.

This ordering pushes substrate work *behind* a demand signal. Costs ~3-5 days to set up thin B before we know what D needs to fill in. Saves us from speculating wrong.

## Resolution of T2 (fidelity gate)

The four candidates from N2:
- (a) bit-exact at single-layer (weakest)
- (b) tolerance band on per-layer outputs
- (c) task-quality match (strongest, expensive)
- (d) "better on at least one axis" (rhetorical, blurry)

(a) is too weak for a gate to Phase 2. (d) is rhetorical. The choice is between (b) and (c).

(b) is more honest as a Phase 1 deliverable: "we match HF within ε on every per-layer output, on a fixed test input." It's quantifiable, achievable, and forces us to characterize the substrate's numerical behavior. It does NOT prove task quality — that's a Phase 2/3 concern when we have a real workload.

(c) is the right Phase 3 gate. For Phase 1 it's expensive in calendar time (running benchmarks like HellaSwag isn't cheap) and may be misleading: a substrate with per-layer drift can still produce coherent text on simple benchmarks just by having approximately-correct attention.

**Phase 1 fidelity gate: (b) per-layer tolerance band.** Specifically: per-layer L2-relative-error ≤ ε for some ε to be determined empirically (likely 1e-2 to 1e-3 given MTFP19 vs bf16 rounding). Gate is "ε is bounded and we understand why." Not "ε ≤ X" with X picked in advance — we don't know enough to pick X.

## Resolution of T3 (extension vs reformulation)

For Phase 1: **extend the substrate to match BitNet's needs.** Add rsqrt, add LUT-backed softmax, add RoPE primitive, add (probably) A8 if we go Path α.

For Phase 3 (training from scratch): **revisit and ask which extensions are actually needed**. Maybe RMSNorm reformulates without sqrt in a base-3-native ML model. Maybe RoPE is wrong-shape and a base-3 positional encoding works better. Phase 3 has the freedom to redesign the architecture; Phase 1 doesn't.

This biases the substrate's identity *toward* "we run base-2-shaped models" in Phase 1, then *toward* "we redesign for base-3" in Phase 3. The risk is Phase 3 never happens and we're left with a substrate that's just "ARM ternary inference engine." I think that's an acceptable risk for now.

## Resolution of T5 (Phase 1 ambition vs Phase 2 readiness)

Phase 1 must prove *enough* for Phase 2 to start with confidence:

1. **Forward pass composes.** All kernels chain without accumulating substrate-level errors.
2. **Numerical behavior is characterized.** We know the per-layer drift magnitude. ε is bounded.
3. **Substrate gaps are closed.** The new primitives (rsqrt, RoPE, softmax LUT, possibly A8) are in libm4t with bit-exact tests.
4. **Real model loads.** BitNet weights from HF format are converted to substrate-native packed-trit storage.
5. **Generation works.** End-to-end token generation produces coherent (not necessarily HF-identical) output.

What Phase 1 explicitly does *not* need to prove:
- Bit-exact output match to HF.
- Performance parity with bitnet.cpp.
- Quality match on benchmark X.
- Multi-threaded inference.

Phase 2's prerequisites: items 1–4 above. Generation (item 5) is nice-to-have for confidence but not strictly required.

## Resolution of T6 (skill gap vs schedule honesty)

I haven't done LLM inference engineering before. The schedule estimate is therefore unreliable. The honest move: **drop the calendar-time estimate from SYNTHESIZE.** Replace with work-units (tasks + ordering + dependencies). Pace work-unit by work-unit; revise as data comes in.

## Resolution of T4 (recognition-borrowing vs independent contribution)

Phase 1 is explicitly a port. The independent contribution is Phase 2+ (fine-tuning + training infrastructure on the substrate). Phase 1's framing should be: *we built the substrate to serve ternary ML; BitNet validates that the substrate can serve at least one recognized ternary model end-to-end.* The story is "the substrate is real," not "we built BitNet."

This is honest. It's also strategically sound: the contribution we lean on is the *substrate*, not the *port*. Microsoft's recognition of ternary at scale is the gating evidence that base-3 ML is viable; our claim is that base-3 ML is *also* viable in its native numeric system.

## What stays open for SYNTHESIZE (and beyond)

**Open to SYNTHESIZE:**
- Path α (A8) vs Path β (MTFP19 throughout). Strategic question for the user. SYNTHESIZE will propose Path α with rationale and explicit revisit-in-Phase-2.
- ε tolerance band: empirically determined, not pre-picked.
- Tail-of-Phase-D scope: actual per-primitive design for rsqrt, softmax LUT, RoPE, optionally A8.

**Open to user (cycle won't resolve):**
- Whether Path β is the strategic preference even at Phase 1 cost.
- Whether Phase 4 (productization) is load-bearing for the project's mission or aspirational.
- Whether the Phase 1 calendar-time uncertainty is acceptable, or whether we need to bound the cycle (e.g., "if Phase 1 hasn't produced a working forward pass in 6 weeks, we re-plan").

**Open to future cycles (not Phase 1):**
- Phase 2's training-side substrate work (TD-19/TD-20/TD-21/TD-22/TD-23 sketches).
- Phase 3's research-side training cycle.
- Phase 4's productization shape.

## What I learned in this cycle

1. **Verifying the architecture before planning is non-negotiable.** I was about to commit to RMSNorm, SwiGLU, smooth-LUTs-everywhere — three of those four were wrong. The cost of the verification step (one WebFetch call + a config.json fetch) is trivial compared to the cost of building against the wrong target.

2. **The Path α / Path β fork is bigger than the kernel-design questions.** Whether activations carry A8 or MTFP19 commits the substrate's identity for Phase 1 in a way that affects Phase 2's starting point. This is the strategic question that SYNTHESIZE should call out and the user should decide.

3. **"Ready for a consumer" was true but the consumer reshapes the substrate.** Even with most kernels in place, BitNet pulls in 3-4 new primitives. The substrate isn't done; it just has the foundations to add specifically what consumers want.

4. **My calendar-time estimate was theater.** Confirmed by RAW; reinforced by REFLECT. SYNTHESIZE will work in work-units, not weeks.
