---
date: 2026-04-14
phase: RAW
topic: How ternary routing helps M4T kernels and the training of models
---

# Raw thoughts

Unfiltered. Two questions tangled together: (1) how does ternary routing help the KERNELS we expose in M4T, and (2) how does it help TRAINING of models on this substrate.

---

## Kernels — what I think I know

The headline: in base-3, 1/3 of cells are zero by construction. A routing-aware kernel can skip those provably; a dense kernel pays for them and discovers the zeros at output time. SDOT preserves this geometrically — zero inputs contribute exactly zero, no rounding, no propagation of base-2 near-zero noise. So routing "helps" in the sense that it aligns compute expenditure with where the signal actually lives.

But that's the PROJECTION step. The DECISION step is where I just ran aground. In the last experiment, `sign_extract` was binary in practice; popcount over sign bits is binary Hamming; the router picked over binary-shaped data and got binary results (58% vs. 81% for L1). The "routing" that's supposed to help may itself be mis-shaped.

So the honest first question: what IS ternary routing, as opposed to bit-vector LSH dressed in ternary vocabulary?

Candidate answers:
- Ternary routing = three-state dispatch. Each input triggers one of three paths (+1 arm, 0 arm, -1 arm). Not two-way branching.
- Ternary routing = native sparsity exploitation. The 0 state is "don't compute at all" — not "compute and multiply by zero".
- Ternary routing = lattice-geometric tile selection. Distances are measured in trit-Hamming, which weights trit mismatches by lattice distance (adjacent: 1, opposite: 2).

All three are plausible. They might be the same thing seen from different angles.

## Kernels — what I don't know

- Does "ternary routing help" mean routing IS the kernel, or routing DECORATES the kernel?
- If a kernel is already sparse-aware (SDOT), does routing on top add anything?
- Can routing overhead eat the savings? On small workloads, the topk_abs dispatch cost exceeds the skipped compute. Threshold?
- How do multi-layer networks compose routing? Does each layer route independently, or is there a hierarchical router that decides globally?

## Training — what I think I know

Training traditionally = gradient descent over continuous weights. In a ternary world:
- Gradients are float-ish by nature. Integer gradients exist but carry different semantics.
- Straight-through estimators (STE) put float back in the training loop. We rejected that explicitly.
- Alternative paradigms exist: evolutionary search, reinforcement learning, prototype accumulation, zero-order optimization.

In `m4t_route_signature_update`, "training" is already something different: offline column-sum → mean-subtract → sign-extract. That's prototype accumulation, not gradient descent. It happens once at model load and never again. Is this the ternary native "training"?

If yes, it has implications:
- Training is a ONE-SHOT operation, not iterative.
- The data volume needed to build good signatures might be smaller than what gradient descent wants.
- The "model" is a set of signatures, not a set of weights.
- Multi-layer training becomes: train layer 1, freeze it, train layer 2 on layer 1's outputs, etc. — greedy stacking.

## Training — what I don't know

- Can prototype-style ternary training match gradient-descent accuracy at scale?
- What does "backprop through a router" mean if routing is non-differentiable?
- Is there a ternary-native gradient? (Balanced-base-3 derivative? Finite differences on trits?)
- Do multi-layer routing networks have a notion of "depth" that helps, or are they flat?
- How do you sharpen a signature over time without re-accumulating from scratch?

## What scares me

The honest answer to "how does ternary routing help training" might be: we don't know yet because the training paradigm itself needs to be reinvented. Most of what gets called "ML training" assumes continuous weights and differentiable loss. Remove both and what's left?

That's exciting because it's real research. It's also a disclaimer on the thesis — we might be claiming "routing outperforms dense" before we have a way to train a routed model to convergence.

## What bothers me about my own framing

I keep collapsing "routing" into "MoE" (Mixture-of-Experts). MoE is one specific architecture where routing is bolted onto a dense transformer. Ternary routing in the NORTH_STAR sense is something more primitive — it's the natural shape of base-3 computation, of which MoE is a special case. I should watch for this narrowing.

Also: the sign_extract finding from the prior turn changes the substrate-level answer. If our routing primitive is base-2-shaped, then claims about "how routing helps" are inherited from base-2 routing research. The TRUE ternary routing help might look very different — and we haven't built the primitive yet.

## Questions arising

1. Is there a small kernel where ternary routing clearly dominates dense-with-sparsity on measurable grounds?
2. What's the minimal training task that exercises a genuine ternary training loop?
3. Does the sign_extract rebuild enable anything that's currently blocked?
4. Is "ternary training" actually "signature accumulation plus online refinement"?
5. If yes, what's the substrate-level primitive for refinement? (We don't have one.)
6. Does the 2/3-vs-1/3 compute ratio actually show up empirically, or is it swamped by overhead?

## First instincts to watch for

- "Routing = MoE": too narrow. Watch for it.
- "Training = prototype accumulation": probably right direction but maybe too simple.
- "Kernels benefit from zero-skip": true but not the whole story — trit-distance geometry is the deeper shape.
- "We just need bigger experiments": the risk that we substitute compute for thinking. The LMM protocol exists to prevent exactly this.
