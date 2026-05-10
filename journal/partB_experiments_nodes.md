# NODES: Part-B experiment candidates

Extracted from `partB_experiments_raw.md`. Numbered for cross-reference.
Tensions marked explicitly. Groupings preserved from RAW (A–G).

## Nodes (one per candidate)

### A — Architectural / training-required

- **N1 (A1).** Routing-native attention. Sparse attention via route_topk_abs
  on Q·K signatures. Compares to dense BitNet attention at matched FLOPs.
  Requires: training (or fine-tune from BitNet weights). Substrate-distinct: YES
  (uses route primitives natively).
- **N2 (A2).** Routing-native FFN (MoE-style). Routes input to k experts via
  threshold_extract on a small projection. Requires: training. Substrate-distinct:
  YES.
- **N3 (A3).** Hybrid layer-gating. Inject routing layer between blocks; vary
  gating frequency to measure trajectory. Requires: training. Substrate-distinct:
  YES (gates use substrate signatures).

### B — Post-hoc / inference-only

- **N4 (B1).** Post-hoc sparse attention. Take BitNet inference; replace dense
  attention with substrate-routed top-k at varying k. NO retraining. Substrate-
  distinct: YES (route_topk_abs is the central primitive). Tractability: 1-week
  to 1-month, depending on whether the existing kernel surface suffices.
- **N5 (B2).** Lattice classification on a 3-class real dataset. Use existing
  trit-lattice machinery. Compare lattice-routing vs dense-cosine. Requires:
  no training (use precomputed embeddings or simple feature extraction).
  Substrate-distinct: YES (the lattice IS the substrate).
- **N6 (B3).** Substrate-routed retrieval. Ternary signatures + distance_batch
  + topk_abs retrieval pipeline; compare vs binary signatures and dense cosine.
  Trajectory axis: collection size + embedding dim. NO training. Substrate-
  distinct: YES (the route primitives are the retrieval engine).

### C — Compression / information-theoretic

- **N7 (C1).** Routing as compression measurement. Bits-per-cell at fixed task
  accuracy: dense bf16 vs ternary dense vs ternary routed. Requires: training
  for the routed variant (weight learning). Substrate-distinct: YES (asks a
  bits-per-cell question that's natural to ternary).
- **N8 (C2).** Lossy weight compression via routing. Take a trained dense
  model; build (a) ternary-quantize and (b) ternary-quantize-AND-route
  compressors; measure quality vs compression ratio. Mixed: ternary-quantize
  is inference-only on existing weights; ternary-quantize-AND-route is more
  involved. Substrate-distinct: PARTIAL (uses substrate primitives but the
  question is mostly about quantization).

### D — Sequential decisions / RL

- **N9 (D1).** Trit-routing as policy head in a small MDP. Compare vs dense
  softmax policy at matched param count. Requires: RL infra (we don't have
  any). Substrate-distinct: YES if the routing is the policy.

### E — Compositional / structured

- **N10 (E1).** SCAN-style compositional benchmark. Routing-based composition
  vs dense composition; held-out compositions test generalization. Requires:
  training. Substrate-distinct: YES if the trit-lattice composition is doing
  the work.
- **N11 (E2).** Symbolic reasoning via routing (bAbI-style). Routing decisions
  inspectable; compare to dense alternative. Requires: training. Substrate-
  distinct: YES (interpretability is unique to substrate's exposed routing).

### F — Less familiar areas

- **N12 (F1).** Coding-theoretic experiment. Ternary signatures as ECC; noisy-
  channel transmission; routing-based decoding vs dense at high noise. Tractability
  unclear (novel). Substrate-distinct: PARTIAL (about redundancy more than
  routing).
- **N13 (F2).** Routing as discrete VAE latent. Routed vs dense-quantized
  latent at fixed bits. Requires: VAE training infra. I lack the ML background
  to scope tightly.
- **N14 (F3).** Signal processing — routing as bandpass filtering analog.
  Audio classification. Substrate-distinct: WEAK (routing-as-filtering doesn't
  uniquely require trit primitives).

### G — Suspicious (probably won't pass Part-B)

- **N15 (G1).** Image classification (MNIST/CIFAR) via lattice routing.
  NORTH_STAR.md disclaims; CIFAR has known representation tax. Likely
  Part-B falsification on this workload — itself informative.
- **N16 (G2).** LLM perplexity on standard benchmarks (WikiText etc.).
  Substrate already runs BitNet at ~92% qualitative pass. Perplexity
  comparison doesn't isolate routing vs density.

## Tensions

### T1 — Tractability vs informativeness
- HIGH-tractability candidates (B-bucket: N4, N5, N6) are inference-only
  but constrained in what they can claim. They test ROUTING-AS-COMPONENT,
  not ROUTING-AS-ARCHITECTURE.
- HIGH-informativeness candidates (A-bucket: N1, N2, N3) are architectural
  but training-required. They test ROUTING-AS-ARCHITECTURE but at high cost.
- The synthesis's mode-shift framing privileged inference-only candidates;
  this tension is whether that framing was right.

### T2 — Training-required vs inference-only (the synthesis-frame bet)
- Inference-only: N4, N5, N6, N15, partial N8, partial N16. Bucket size: 4-6.
- Training-required: N1, N2, N3, N7, N8b, N9, N10, N11, N12 (probably),
  N13, N14. Bucket size: 9-11.
- The training bucket is roughly 2× the inference-only bucket. If we
  insist on inference-only per the synthesis's framing, we're working
  in a smaller candidate space. Whether that's adequate depends on
  whether the inference-only candidates are STRONG, not just NUMEROUS.

### T3 — Existence vs trajectory vs mechanism
- Existence-test only: N5 (single-task), N15 (single-task).
- Trajectory-test possible: N3 (vary gating frequency), N4 (vary k), N6
  (vary collection size + embedding dim), N7 (vary task complexity), N8
  (vary compression ratio), N10 (vary compositional held-out), N11 (vary
  rule complexity).
- Mechanism-test possible: N11 (interpretable routing), N7 (information
  density), maybe N12 (channel noise as mechanism).
- A strong Part-B candidate should support at least trajectory testing.
  Existence-only candidates can be informative but their result is
  weaker (one workload could be coincidence).

### T4 — Substrate-distinctiveness vs workload accessibility
- Workloads that "fit" the substrate's trit-shape (lattice classification,
  ternary signatures) make routing look natural and might be tautological
  wins. Workloads that DON'T fit (image, audio, dense LLM perplexity)
  often show base-3's representation tax.
- Best candidates land in the middle: workload has natural ternary structure
  AND the routing-vs-dense question is a real question on it.
- N5, N6 are workloads that fit substrate shape — risk of tautology.
- N15 doesn't fit — known representation tax.
- N1-N3 fit (LLM forward pass already runs on substrate) — promising.
- N4 is interesting because the workload is BitNet (which doesn't NATIVELY
  use substrate routing) but the experiment ADDS substrate routing.

### T5 — Compute-parity definability
- N4 (sparse attention at matched k) — clean. FLOPs scale with k.
- N1, N2, N3 (architectural) — ill-defined. Different architectures have
  different FLOP profiles; matching is hard.
- N5, N6 (lattice / retrieval) — clean if same #ops counted.
- N7 (compression) — bits-per-cell IS the parity metric, well-defined.
- N15 (image) — clean (FLOPs match).

### T6 — R1's failure mode
R1 tested per-expression-tau dual-threshold signatures vs sign-only
signatures. Falsified across 4 axes. The mechanism: the third state
(zero) didn't add discriminative power on the test inputs.

Candidates that share R1's failure mode (test "is the third state load-
bearing in this signature use") would risk repeating the falsification:
- N5, N6 directly use signatures. If their inputs don't realize all three
  trit states meaningfully, they'd repeat R1.

Candidates that are immune (the third state is exercised by construction
or by the workload):
- N1-N4 use trits as architectural primitives, not signatures.
- N7-N11 use trits as quantization or routing for non-signature purposes.

This is a partition of the candidate space. R1's falsification narrows
the signature-based bucket; the architectural and quantization buckets
remain open.

## Tensions summary

- T1: tractability vs informativeness — implies preferring inference-only
  candidates that are STRONG (not weak).
- T2: training-vs-inference bucket sizes — inference-only is smaller, so
  the synthesis's mode-shift may be vulnerable if inference-only candidates
  are all weak.
- T3: existence-vs-trajectory — strong candidates support trajectory.
- T4: substrate-distinctiveness vs accessibility — best candidates land
  in middle.
- T5: compute-parity definability — N4 and N7 are unusually clean.
- T6: R1 failure mode — narrows signature-based candidates (N5, N6).

## Emerging shape

If I prioritize:
- Inference-only (per synthesis framing, R7 from red-team)
- Strong (supports trajectory)
- Compute-parity-definable
- Not R1-vulnerable

The candidates that survive: **N4 (post-hoc sparse attention)** clearly,
**N7 (compression measurement)** if we can do inference-only weight
quantization (the (a) variant of N8), and possibly **N6 (substrate-routed
retrieval)** if we can argue past the R1-vulnerability.

If we relax inference-only: N1 (routing-native attention) and N2 (routing-
native FFN) become contenders, both training-required.

Going to REFLECT to find the structural insight beneath this.
