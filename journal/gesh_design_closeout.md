---
cycle: gesh_design
phase: CLOSEOUT
date: 2026-05-01
scope: owner observation surfaces during/after synthesize phase: STE is a base-2 fix for a problem that doesn't exist in the lattice
companions: gesh_design_{raw,nodes,reflect,synthesize}.md · ../Documents/GESH/GESH_DESIGN.md
status: COMPLETE — synthesize-phase Phase A revised; new questions surfaced for the next cycle
---

# Closeout — gesh_design

## What happened

After the synthesize phase committed Phase A to "ternary projections + STE backward + hard top-k," owner observation:

> GESH doesn't need STE. Our data is already shaped. It's geometric in nature. The lattice is the geometry.

This drops one of the three Gs (Gradient, in its STE formulation) before Phase A even starts. The synthesize-phase build plan needs revision.

## The insight, traced

STE (straight-through estimator) is a workaround for the discontinuity between continuous parameters and quantized outputs in base-2 networks. The continuous parameters live in float space; the forward pass quantizes; the backward pass pretends the quantization didn't happen so gradients can flow.

In a base-3-native system where:
- The substrate is bit-exact integer arithmetic.
- The projections are ternary from the start (not float-then-quantize).
- The data lives on the trit lattice (not in float space passing through).
- The loss is computable bit-exactly per the verified kernels.

...there is no discontinuity to estimate through. STE is solving a problem that doesn't exist. The "gradient sensitivity" the design called for becomes **lattice-update sensitivity** — a different mechanism with the same conceptual role (end-to-end learnable routing).

This isn't a small refinement. It changes the entire training story.

## What changes in Phase A

### Was (synthesize-phase Phase A)
- Ternary PCA-init projection.
- Forward: project, quantize, route.
- Backward: STE through ternary quantization, AdamW on continuous shadow parameters.
- Hard top-k tile selection.
- Periodic refresh as shadow parameters shift.

### Now (closeout-revised Phase A)
- Ternary PCA-init projection (unchanged).
- Forward: project (bit-exact integer), route (unchanged).
- **No backward pass through quantization.** No STE. No shadow parameters.
- **Lattice update rule:** for each candidate trit-flip in the projection, compute the bit-exact loss delta on a batch. Apply flips that reduce loss. Specific algorithm TBD; candidates include:
  - Coordinate descent (one trit at a time, exhaustive over the projection).
  - Random-subset sampled flips (cheaper per step; stochastic).
  - Greedy batched flips (compute deltas in parallel, apply non-conflicting ones together).
- **No periodic refresh** of tile signatures — they don't drift, because there are no continuous shadow parameters to track.
- Hard top-k unchanged.

The revised Phase A is **simpler** than the original, not more complex. Fewer mechanisms (no Gumbel, no STE, no shadow params, no refresh schedule); fewer training-only kernels (no STE backward, no Gumbel-softmax); honors substrate purity completely (no float anywhere in training).

## New questions surfaced

### Q1 — Top-k discontinuity in the loss surface
Hard top-k creates a discontinuous loss surface: one trit-flip in the projection can change which tiles are selected, which can change the loss by a lot. STE was partly smoothing this. Without STE, the optimization is over a genuinely discontinuous discrete surface.

**Hypothesis:** discrete optimization can handle this if it's smart about move-acceptance (e.g., consider moves that don't flip top-k first, occasionally allow moves that do). The lattice-update rule is more like local search than gradient descent; the algorithm needs to be designed accordingly.

**Cycle gate:** Phase A first attempts coordinate descent with simple "accept if loss decreases" rule. If convergence stalls, escalate to simulated-annealing-style move acceptance, then to more sophisticated discrete-search algorithms.

### Q2 — Computational cost of lattice updates
Per-trit flip costs forward-pass-on-batch. For an MxN projection, exhaustive coordinate descent costs M·N forward passes per epoch. With NEON-accelerated batch processing this is plausible at small scale but might not scale.

**Hypothesis:** random subset sampling of trit-flips per step keeps the cost bounded. Empirical: how many flips per epoch need to be evaluated to track the optimum closely enough?

**Cycle gate:** Phase A measures "trit-flip evaluations per epoch needed to converge" alongside the accuracy measurement. If the cost is prohibitive, escalate to smarter sampling.

### Q3 — Does the lattice update rule subsume what the design called "Geometric sensitivity"?
The design's "geometric" was about projections aligning to the data manifold via PCA + gradient refinement. Without STE, the gradient refinement becomes lattice updates. PCA initialization still helps (gives a starting point near a useful region of the lattice), but the manifold-alignment work is done by the lattice update rule, not by a separate "geometric loss."

**Hypothesis:** Geometric and Gradient (originally separate Gs) collapse into one mechanism — lattice update with manifold-aware initialization. The three Gs become two: **Global** (multiscale routing, separate concern) and **Lattice-Geometric** (the unified mechanism for projection learning).

**Cycle gate:** if Phase A's ablation can't separate "PCA init" from "lattice update" contributions cleanly (i.e., they're entangled by construction), the two-G framing is empirically supported.

### Q4 — What does "Gradient sensitivity" mean post-STE?
The original design used "Gradient" to mean "differentiable end-to-end via STE." Without STE, the term needs a new meaning or to be dropped.

**Provisional new meaning:** "Lattice sensitivity" — the routing structure can be updated in response to loss signal, where the update rule operates over the discrete lattice rather than continuous parameters. This preserves the "end-to-end learnable" property without committing to a specific gradient mechanism.

**Naming consequence:** "G³SH" or "Gesh" doesn't quite map to the new structure. Three Gs → two mechanisms (Global + Lattice-Geometric) with discrete lattice update as the training rule. Naming is downstream of the mechanism; can be revisited.

## Updated three-Gs framing

| Original | Revised |
|---|---|
| Global (multiscale routing) | **Global** — unchanged |
| Geometric (manifold-aware projections via PCA + gradient refinement) | **Lattice-Geometric** — PCA init + lattice update |
| Gradient (STE-based end-to-end training) | (Subsumed into Lattice-Geometric. The "differentiability" the original wanted is now "lattice-position addressability.") |

Two mechanisms, one possibly absent (Global, deferred to Phase B). Phase A tests Lattice-Geometric only.

## What survives from the synthesize phase

- **Benchmark commitment:** induction-head as the substrate-claim test. Unchanged.
- **Build sequencing:** stage 2 alone first, ablation-driven escalation. Unchanged.
- **Substrate placement:** default to libglyph; expect zero new substrate primitives in Phase A. **Strengthened** — without STE backward, even the training-only kernels are simpler (no Gumbel-softmax, no shadow params).
- **Pre-committed gates:** the §6 provisionals → measurement-driven gates. Unchanged in structure.
- **Frozen-bank framing:** the central architectural commitment. Unchanged.

## What changes from the synthesize phase

- **Phase A's training mechanism:** STE → lattice update rule.
- **Three-Gs framing:** three → two (with Global possibly deferred to Phase B).
- **Training-only kernel surface:** smaller. No STE backward, no Gumbel-softmax. Maybe a `lattice_flip_evaluate` helper for batched trit-flip loss deltas; even that's libglyph-level.
- **Phase D (training-only mechanisms):** mostly drops out. No Gumbel-softmax, no shadow-parameter refresh. The remaining "Phase D" is "smarter lattice-update strategies if simple coordinate descent stalls."

## Methodology note

Three cycles in this codebase have shown the same shape: design → REFLECT finds a wrong reference frame → SYNTHESIZE reframes. This is the *fourth* now: the synthesize phase committed to a Phase A that inherited STE from the original design's reference-frame (attention-shaped); owner observation surfaced that STE is a base-2 ergonomic in a base-3-native system.

The pattern: **even after reframing against the right reference frame, residual base-2 assumptions can hide in mechanism choices.** STE survived three reframings because it's so canonical in the ML literature it doesn't read as a base-2 commitment — it reads as "the standard way to train through quantization." It IS the standard way. It's also the wrong way for a lattice-native substrate.

Worth recording as a discipline rule: when a mechanism is "the standard way" in the surrounding literature, that's a flag, not a justification. The substrate-claim is that base-3 native compute exposes structures base-2 framings hide. STE is what hides "the lattice IS the geometry."

## Loop-back triggers from here

- **Back to NODES** if Phase A's lattice update rule fails to train at all on induction-head, even with smart move-acceptance. The "lattice update is sufficient" hypothesis would be falsified; STE or some other mechanism returns.
- **Back to RAW** if the two-G reframing turns out to undercount mechanisms (e.g., something else is needed that wasn't called out).
- **No loop-back** if Phase A trains via lattice update and the unification claim's gate is met. That's the wood cutting itself; Gesh becomes Lattice-Native Gesh, and the substrate's THESIS Part B has its first measurement.
