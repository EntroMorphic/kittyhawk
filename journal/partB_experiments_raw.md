# RAW: Part-B experiment candidates

Per `cycle1_plan_v2.md`. Writing without re-reading prior step_change notes.
Forcing 12+ candidates and a "less familiar areas" category.

## Sub-question first: what would constitute Part-B evidence?

Before listing candidates, let me wrestle with this. Part B (from THESIS.md
as I recall) says routing is essential not optional, and the routing
advantage widens as task richness increases.

The falsification: routing-native matches dense at equal compute AND the
gap doesn't widen with task structure.

So Part-B evidence = at least one (workload, routing-native impl, dense
baseline) triple where:
- routing-native > dense at fixed compute (existence)
- the gap grows when we increase {classes, modalities, compositional
  structure, sequence length, ...} (trajectory)

I'd weight existence and trajectory together. A single-point existence
result (routing wins on one task) is necessary but doesn't prove the
"essential" framing — could be coincidence, regularization effect,
hardware quirk. Trajectory is what makes the claim load-bearing.

Compute-parity is the killer constraint. Most "routing wins" claims I've
seen elsewhere were "routing wins at lower compute on dense weights" —
which is base-2-substrate flexibility, not base-3-substrate essentiality.
The honest version is "at FIXED compute (FLOPs, ms, energy), routing
beats dense."

## Candidate stream — let me dump 15+

### Cat A: Architectural (the obvious bucket)

1. **A1. Routing-native attention.** Replace BitNet's attention with a
   routing variant: compute Q·K signatures (sign-extract), use top-k
   routing to pick which K positions each Q attends to (sparse
   attention via the route_topk_abs primitive), then dot-product only
   on selected positions. Compare on coherent generation quality at
   matched FLOPs/token vs dense BitNet attention.

2. **A2. Routing-native FFN.** Replace BitNet's FFN with mixture-of-experts
   where the routing decision uses substrate route primitives
   (threshold_extract on a small projection of the input, route to k experts).
   Compare on perplexity / generation quality vs dense FFN at matched
   FLOPs.

3. **A3. Hybrid.** Keep BitNet's transformer blocks but inject a routing
   layer between every N layers (a learned signature that gates which
   downstream computation runs). Trajectory: vary N, see if routing
   advantage grows as we increase the gating frequency.

### Cat B: Post-hoc / inference-only (the underexplored bucket)

4. **B1. Post-hoc sparse attention via routing.** Take BitNet inference as
   shipped. At each attention step, use the substrate's route_topk_abs
   primitive to pick top-k K positions per Q based on a quick signature
   distance, dot-product only on those. Compare quality at varying k.
   No retraining. Trajectory: as k decreases (more aggressive routing),
   does base-3 routing degrade more gracefully than a base-2 top-k
   alternative?

5. **B2. Lattice classification on a real dataset.** Use the existing
   trit-lattice machinery from gesh phase A.1/A.2 to classify a real
   dataset (not synthetic). MNIST is the obvious base-2-friendly case;
   instead try a ternary-friendly task: e.g., a dataset where labels
   are inherently 3-state (positive/negative/neutral sentiment, or
   trinary classification). Compare lattice-routing vs dense-cosine at
   fixed memory.

6. **B3. Substrate-routed retrieval.** Build an embedding-retrieval
   benchmark where queries and docs are embedded as ternary signatures.
   Use the substrate's distance_batch + topk_abs for retrieval. Compare
   vs binary (sign-only) signatures and dense cosine. Trajectory: vary
   document collection size and embedding dimension.

### Cat C: Compression / information-theoretic

7. **C1. Routing as compression measurement.** For a given task,
   measure the information density per cell achievable with: dense
   bf16, ternary routed, ternary dense. The "routing wins" claim
   would be "ternary routed achieves the same task accuracy at lower
   bits-per-cell than ternary dense." Trajectory: vary task complexity.

8. **C2. Lossy compression of weights via routing.** Take a trained
   dense (bf16) model. Build two compressors: (a) ternary-quantize
   weights, (b) ternary-quantize-AND-route (use signatures to skip
   weights with low contribution). Measure quality recovery vs
   compression ratio. If (b) > (a), it's routing earning bits.

### Cat D: Sequential decisions / RL

9. **D1. Routing for action selection in a small MDP.** Use trit-routing
   as a policy head: state → ternary signature → top-k action via
   route primitives. Compare vs dense softmax policy at matched
   parameter count. Hard to do without RL infra; tractability concern.

### Cat E: Compositional / structured

10. **E1. Compositional generalization on a synthetic structured task.**
    SCAN-style compositional benchmark (commands → actions): does
    routing-based composition generalize better to held-out compositions
    than dense composition? Substrate-distinct because routing is
    explicitly compositional in the trit-lattice algebra.

11. **E2. Symbolic reasoning via routing.** Pose a symbolic-rule task
    (e.g., bAbI-style) where the routing decisions can be inspected.
    Substrate advantage if any: routing decisions are interpretable
    (can audit which trit triggered which path); dense alternative is
    opaque. The Part-B-relevant question: does routed reasoning
    generalize to held-out rules better than dense?

### Cat F: Less familiar areas (forced inclusion per R5)

12. **F1. Coding-theoretic experiment.** Treat ternary signatures as
    error-correcting codes. Build a noisy-channel transmission task
    (some bits flip during processing); does routing-based decoding
    degrade more gracefully than dense at high noise? I'm not sure
    this maps to Part B cleanly — it might be an "advantage of
    redundancy" result not specifically a routing one.

13. **F2. Neural compression / variational inference angle.** Use
    routing as a discrete latent variable in a VAE; does the routed
    latent produce better reconstructions at fixed bits than a
    dense-quantized latent? Adjacent to C1 but framed in VAE terms.
    Probably requires more ML background than I currently have to
    scope properly.

14. **F3. Signal processing.** Routing as an analog of bandpass
    filtering — each "expert" handles a frequency band. Could compare
    to dense convolutional alternatives on audio classification.
    Probably not substrate-distinct enough to count.

### Cat G: Things I'm suspicious of (won't work, but worth listing)

15. **G1. Image classification via lattice routing.** MNIST or CIFAR.
    NORTH_STAR.md disclaims image canon as base-2-framed. Past attempts
    showed CIFAR has a "representation tax" base-3 doesn't close. Probably
    a Part-B falsification on this workload.

16. **G2. LLM perplexity on standard benchmarks.** WikiText etc. The
    BitNet bf16 reference exists; substrate already reaches ~92% strict
    pass on a 24-prompt qualitative battery. Pure perplexity comparison
    doesn't isolate routing vs density — they'd be matched at the model
    level.

## Doubts and questions

- **Compute-parity is hard.** Most candidates I'm proposing don't have a
  clean "matched FLOPs" comparison because the dense baseline isn't trivially
  defined. For attention sparsification, the matched-FLOPs comparison is
  straightforward. For others, it's apples-to-oranges.

- **The substrate's routing primitives are designed for a specific shape**
  (sign-zero-sign trichotomy, packed-trit signatures, popcount distance).
  Workloads that fit this shape will look favorable for substrate-routing
  somewhat tautologically. The honest test is a workload that's NOT
  obviously trit-shaped.

- **R1 was already a Part-B-flavored test and got falsified.** R1 tested
  whether dual-threshold (sign+confidence) signatures discriminate better
  than sign-only. The result was "no." Some of my candidates above are
  R1-shaped (B2, B3 specifically use signatures). What's different that
  would let them pass when R1 failed? I should think about this in REFLECT.

- **Most candidates need TRAINING to be informative.** A1, A2, A3, C1, C2
  (the b case), D1, E1, E2 all want trained models. Inference-only
  candidates: B1, B2 (depending on classifier), B3, G1. So if the
  synthesis's mode-shift framing requires inference-only candidates,
  the bucket is small. That's important information.

- **I'm not sure which candidates are PART-B specifically.** Some are
  "demonstrate that the substrate works on this task" (Part-A or generality).
  Some are "demonstrate that routing structurally beats dense" (Part-B).
  The line is fuzzy. REFLECT needs to clarify.

## What I'm NOT including

- Multi-modal experiments. Out of scope for current substrate.
- Image-based experiments beyond G1. Disclaimed.
- Anything requiring distributed training. Substrate is single-threaded
  by design.

## Open questions to take into NODES

1. Is "compute parity" really the right constraint, or should it be
   "memory parity" or "energy parity"?
2. R1's falsification — what's the lesson? Are some of these candidates
   doomed for the same reason?
3. Inference-only candidate bucket is small (B1, B2, B3, G1). Is that
   evidence against the synthesis's mode-shift framing?
4. The substrate-distinctiveness audit — does it apply per-experiment or
   per-result?
