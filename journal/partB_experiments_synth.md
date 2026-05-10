# SYNTH: Part-B experiment candidates — scored

Per `cycle1_plan_v2.md`. Output: scored list, top 1-2 scoped for next cycle,
"needs more research" bucket, honest caveats.

## Scoring rubric (re-scorable by future readers)

Each candidate scored 1–5 on each of 7 axes, totaled out of 35.

| Axis | What it measures |
|---|---|
| **Tractability** | 5=1 week, 4=1 month, 3=1 quarter, 2=>1 quarter, 1=requires major capability not yet built |
| **Inform+** | Quality of evidence if positive result (5=novel direct Part-B evidence; 1=marginal) |
| **Inform−** | Quality of evidence if negative result (5=clear Part-B falsification on this workload; 1=ambiguous) |
| **Substrate-dist** | Does the experiment USE the substrate's distinct capabilities (route primitives, packed trits) vs just live ON it? (5=route primitives are the central act; 1=routing is incidental) |
| **Op'ability** | Are the pre-commit gates concrete enough to declare success/failure? (5=yes; 1=fuzzy) |
| **Trajectory** | Can a single complexity axis be varied to test "gap widens" claim? (5=natively trajectory-shaped; 1=single-point only) |
| **Mechanism** | Can controls distinguish substrate routing from other effects? (5=clean controls available; 1=can't isolate) |

## Scored confidently

| ID | Candidate | Tract | Inform+ | Inform− | Subst | Op | Traj | Mech | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| **N4** | **Post-hoc sparse attention via route_topk_abs** | 4 | 5 | 5 | 5 | 4 | 5 | 4 | **32** |
| N3 | Hybrid layer-gating (training) | 1 | 5 | 5 | 5 | 3 | 5 | 5 | 29 |
| N1 | Routing-native attention (training) | 1 | 5 | 5 | 5 | 4 | 4 | 4 | 28 |
| N7 | Compression measurement (training) | 2 | 4 | 4 | 5 | 4 | 4 | 4 | 27 |
| N2 | Routing-native FFN (training) | 1 | 4 | 4 | 5 | 4 | 4 | 4 | 26 |
| N6 | Substrate-routed retrieval | 4 | 3 | 2 | 3 | 4 | 4 | 2 | 22 |
| N5 | Lattice classification on 3-state task | 4 | 3 | 2 | 3 | 4 | 2 | 2 | 20 |
| N8 | Lossy weight compression (partial) | 3 | 3 | 3 | 4 | 3 | 3 | 2 | 21 |
| N15 | Image classification (CIFAR/MNIST) | 4 | 1 | 2 | 2 | 4 | 2 | 2 | 17 |
| N16 | LLM perplexity on standard benchmarks | 4 | 1 | 1 | 2 | 4 | 2 | 1 | 15 |

## Needs more research before scoring

| ID | Candidate | What's needed before scoring |
|---|---|---|
| N9 | RL policy via routing | Project doesn't have RL infra; would need to scope what "small MDP" means and whether an RL substrate is in-scope at all |
| N10 | SCAN compositional generalization | Need to understand current SCAN benchmark conventions; held-out-composition methodology details |
| N11 | Symbolic reasoning (bAbI) | Routing-interpretability is mentioned but not yet a substrate-exposed feature; would need to design what "inspecting routing decisions" means concretely |
| N12 | Coding-theoretic ECC framing | Novel framing for me; would need to read coding-theory literature on ternary codes before scoring tractability |
| N13 | Routing as discrete VAE latent | Discrete VAE training is a well-developed area I'd need more background on |
| N14 | Signal processing routing-as-bandpass | Probably weak substrate-distinctiveness; not worth scoring further unless someone with audio-ML background sees something I don't |

## Top candidate for Cycle 2 — N4 (post-hoc sparse attention)

### Why N4 wins

The only candidate that combines:
- **Tractability** (1-month, no training prereq)
- **Substrate-distinctiveness** (route_topk_abs is the central act)
- **Compute-parity definability** (FLOPs scale linearly with k)
- **Trajectory-testable** (vary k from head_dim down to small values)
- **Mechanism-testable with controls** (random and oracle baselines available)
- **Not R1-vulnerable** (selects positions, not signatures)

It's also the only candidate that lets us test Part B WITHOUT first
building training capability. That's exactly what the synthesis's
mode-shift framing requires.

### Cycle 2 design sketch

**Workload:** BitNet b1.58-2B-4T inference on the existing 24-prompt
battery + the math_div / reason_word reasoning subset.

**Experimental arms (4 conditions):**
1. **Dense baseline** — current substrate at gate1+fudge2 default
2. **Substrate-routed top-k** — at each attention step, use route_topk_abs
   on Q·K signatures (computed via existing primitives) to pick the top-k
   K positions per Q; dot-product only on those
3. **Random top-k** — same k, random K positions per Q (sanity baseline)
4. **Oracle top-k** — same k, K positions selected post-hoc by computing
   full attention and picking the top-k weights (upper bound)

**Trajectory axis:** k ∈ {head_dim, head_dim/2, head_dim/4, head_dim/8,
head_dim/16, head_dim/32}. For BitNet's head_dim=128: k ∈ {128, 64, 32,
16, 8, 4}.

**Metrics per (arm, k):**
- Strict pass rate on 24-prompt battery (manual classification)
- Token agreement % vs dense baseline
- Per-layer ε vs dense baseline
- Wall-clock per token (sanity check that the routing IS skipping work)

**Pre-commit gates:**

PART-B EVIDENCE if all of:
- Substrate-routed top-k matches dense baseline strict pass rate within
  10pp at k=64 (half of head_dim)
- Substrate-routed beats random top-k by >10pp at k=16
- Gap between substrate-routed and random WIDENS as k decreases
  (trajectory)

PART-B FALSIFICATION if any of:
- Substrate-routed indistinguishable from random across the trajectory
  (substrate routing isn't earning its place)
- Substrate-routed degrades faster than random as k decreases (substrate
  routing actively hurts)
- No k value gives substrate-routed within striking distance of oracle

INCONCLUSIVE if:
- Quality varies wildly per prompt and the trajectory is noisy
- Wall-clock results don't show the expected FLOP savings
  (implementation issue rather than thesis result)

**Capability prerequisites:**
- New harness path: substrate-routed sparse attention. Probably 1-2 weeks
  of kernel composition (route_topk_abs + a sparse attention dot product).
  No new kernels needed beyond what exists.
- Random top-k path: trivial (uniform sampling).
- Oracle top-k path: requires computing full attention then sorting.
  Slow but straightforward.
- Battery infrastructure: already in place from prior cycles.

**Estimated total Cycle 2 effort:** 2-4 weeks. The synthesis's "2 weeks"
estimate was lower bound; if the routing implementation has surprises,
upper bound is a month.

### Honest risks for Cycle 2

- **R-N4-1**: Substrate-routed top-k might match dense at LARGE k but
  collapse at small k for non-substrate reasons (e.g., BitNet's specific
  attention pattern needs the smaller weights, not just the largest k).
  Mitigation: oracle baseline establishes the ceiling; if oracle also
  collapses, the issue is sparse-attention-on-this-model not substrate.

- **R-N4-2**: The compute parity might be confounded by implementation
  efficiency. Substrate-routed top-k saves dot-product FLOPs but adds
  signature-distance FLOPs. Need to count BOTH carefully.

- **R-N4-3**: We're testing on greedy decoding only. Sampling might
  shift the picture. Out of scope for Cycle 2; if substrate-routed wins
  on greedy, sampling is Cycle 3.

## Cycle 3 design (parallel scoping, not execution)

**Top training-required candidate: N1 (routing-native attention).**

If Cycle 2 produces Part-B evidence on N4, Cycle 3 extends to architectural
Part-B: design routing into BitNet's attention from scratch (not post-hoc),
fine-tune, measure at matched FLOPs.

**Cycle 3 prerequisites:**
- Gradient kernels for the substrate primitives used in attention:
  bitlinear_scale_bx_backward, rmsnorm_bx_backward, attn_v_combine_backward,
  route_topk_abs_backward (the gradient through the topk selection is
  the hard part — typically straight-through or REINFORCE).
- An optimizer state representation in MTFP (likely a lightweight
  adaptation of an existing optimizer; SGD-momentum or Adam with MTFP
  moments).
- A training loop that integrates with the existing inference harness.

**Cycle 3 design work that can happen in parallel with Cycle 2 execution:**
- Enumerate the gradient kernels needed; sketch their forward-vs-backward
  symmetry.
- Sketch the optimizer state representation.
- Identify which kernels are reusable from Cycle 2.

This makes the synthesis's mode-shift more honest: Cycle 2 = inference-only
Part-B test (N4); Cycle 3 = training-enabled Part-B test (N1). Sequenced,
not competing.

## What we can't test yet, and why

- **Routing-native architectures from scratch** (N1, N2, N3): require
  training. Cycle 3+ work.
- **Multi-modal routing claims**: substrate doesn't support multi-modal
  workloads. Out of current scope.
- **Compute-parity at the energy or wall-clock level on diverse hardware**:
  the substrate is single-architecture (Apple Silicon NEON). Energy
  measurements would require additional tooling.
- **R1-style direct signature tests at higher arity**: R1 was falsified
  at one specific operationalization. A different operationalization
  might pass, but the burden of proof is now higher and the experiment
  design is non-trivial.
- **Long-context Part-B claims**: existing battery is short-prompt.
  Long-context behavior of substrate routing is unmeasured.

## Honest assessment of the synthesis's mode-shift framing

**The framing survives.** N4 is a strong, tractable, inference-only Part-B
candidate. The synthesis predicted that an inference-only candidate would
exist; it does. The synthesis predicted that the substrate-novelty audit
would not disqualify everything; it doesn't.

**But narrowly.** Only ONE strong inference-only candidate emerged. If
N4 turns out to be unworkable in practice (Cycle 2 surprise), the
mode-shift framing collapses and we'd back into "training-first sequencing."

**The Cycle 3 plan partially defends against this**: even if Cycle 2
fails on N4, the design work for N1 (Cycle 3 candidate) makes the
training-first transition concrete rather than a fresh start.

## Methodology lift recorded

Cycle 1 produced more than the candidate list. It also produced:

1. **An operationalization of Part B** (matched FLOPs at inference;
   trajectory along a single complexity axis; existence + trajectory +
   mechanism as joint criteria). This is reusable for any future Part-B
   work.

2. **A scoring rubric** that can be re-applied as new candidates emerge.
   The rubric makes Cycle 1 re-runnable rather than one-off.

3. **A categorization** (R1-vulnerable vs not; tautological vs honest;
   inference-only vs training-required) that helps future candidates
   self-classify.

These artifacts may end up more durable than any single experiment's result.

## Cycle 2 launch criteria

Cycle 2 begins when:
- [ ] N4's experimental arms are designed in detail (separate design
      doc; this synth is sufficient for the high-level shape)
- [ ] The harness changes for substrate-routed sparse attention are
      planned (which existing kernels compose, what new kernel surface
      is needed)
- [ ] Pre-commit gates from this synth are formally entered as a
      Cycle 2 deliverable

If any of these blockers can't be cleared with current substrate
capability, that's information that loops back here.
