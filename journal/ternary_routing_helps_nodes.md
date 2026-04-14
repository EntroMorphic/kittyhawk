---
date: 2026-04-14
phase: NODES
topic: How ternary routing helps M4T kernels and the training of models
---

# Nodes

## N1. Routing is the expression of structural sparsity
Dense compute over base-3 data pays for 1/3 of operations that mathematically contribute nothing. Routing turns that statistical fact into an architectural primitive: "don't compute here, the trit is zero." The 1/3 saved isn't a rounding optimization; it's a geometric invariant.

## N2. SDOT already expresses partial ternary routing at the hardware level
When SDOT sees a zero trit on either side, the multiply-accumulate lane produces exact zero — no float rounding, no near-miss cancellation. The hardware is doing structural zero-skip at the primitive level. What routing adds *on top* of SDOT is tile-level skip: "don't dispatch this whole matmul at all."

## N3. Ternary routing is three-way dispatch, not binary selection
Two-way: `if score > 0, go left else go right`. Three-way: `if score ≫ 0 take active-positive path, if score ≪ 0 take active-negative path, if score ≈ 0 take inactive path`. The inactive path is the base-3-native zero — not "do nothing", but "this computation is structurally absent here". MoE's on/off gating is the two-way degenerate case.

## N4. The substrate's current sign_extract is binary-shaped
Named `sign`, produces effectively two-state output on real inputs, feeds binary Hamming distance. Inherited from trix-z. Blocks any true ternary routing experiment until replaced.

## N5. Trit-Hamming distance IS lattice-geometric when data is three-state
Popcount over XOR of packed trits weights mismatches by bit-count: 0-vs-sign = 1 bit, opposite-signs = 2 bits. That's lattice distance on the trit manifold. Already ternary-aware — it just needs ternary-shape INPUTS to express its shape.

## N6. Training ≠ gradient descent in this substrate
The `m4t_route_signature_update` primitive is already a form of training: one-shot, prototype-style, no gradients. If that's the native training paradigm, the whole apparatus of backprop + SGD + Adam doesn't port. What replaces it isn't obvious.

## N7. Prototype accumulation has known limits
k-NN scales linearly in training data. Prototype methods (like 1-nearest-centroid) compress but have accuracy ceilings. Transformer-level capability has never been demonstrated with prototype training alone.

## N8. Signatures need to sharpen over time, not just accumulate
A one-shot signature is a static prototype. Real learning needs to REFINE: "these training examples got misclassified; adjust the signature to separate them." The substrate doesn't have this primitive. What would it look like? Online sign-flip on specific trit positions? Active-learning re-accumulation?

## N9. Routing composes across layers
A single routing decision picks from T tiles. A sequence of routing decisions picks from T^d combinations over d layers — exponentially expressive. This is where multi-layer routing networks *could* match or exceed dense nets, if the training paradigm can actually find good tile sequences.

## N10. The router is itself a kernel
The five m4t_route primitives are themselves the kernel. Routing doesn't "help the kernel" from outside — it IS the kernel shape. Asking "how does routing help kernels" is confused if we imagine the kernel as dense-matmul and routing as a modifier. In ternary substrate, routing-shape IS the kernel-shape.

## N11. Hardware utilization claim is still undischarged
We claim SDOT + TBL + VCNT are native. Never measured. Until we know utilization, we don't know whether the "routing helps" story is architectural or just the name we give to what the hardware does well.

## N12. MNIST is not the bed
Already established. Any claim about routing helping training must be measured on a task where sign/magnitude/trit structure is load-bearing. MNIST isn't that task.

---

## Tensions

### T1. Routing-overhead vs routing-savings
Small workloads: topk_abs dispatch cost ≫ skipped compute. Large workloads: routing savings dominate. Threshold unknown. Can't claim "routing helps" without naming the regime.

### T2. Signature accumulation vs signature refinement
N6/N7 vs N8. One-shot prototype training is natively ternary but hits a capacity ceiling. Refinement-over-time is how gradient methods get past that ceiling — but refinement in a discrete lattice is exactly what gradient methods can't do cleanly.

### T3. Ternary primitive purity vs existing binary surface
N4. Our routing primitive is binary-shaped. Fixing it is a substrate change. Not fixing it means every "ternary routing experiment" tests binary routing underneath. Can't straddle.

### T4. Kernel-level thinking vs architecture-level thinking
N10. Asking "how does routing help kernels" already assumes kernels are separable from routing. In a ternary substrate the question might be malformed — routing IS the kernel in the relevant compositions.

### T5. Thesis-level aspiration vs empirical state
The thesis claims routing will naturally outperform dense in base-3 environments. Empirically we have: one consumer tested, loses 23 points because its "routing" is binary-shape; no training paradigm specified; no hardware utilization measured; no benchmark bed chosen.

---

## Dependencies

- Ternary-native extraction primitive (replacement for sign_extract) → any genuine routing experiment
- Benchmark bed (THESIS §4) → any training-help measurement
- Hardware utilization measurement → discharge of "native" claims
- Refinement primitive → non-toy training
