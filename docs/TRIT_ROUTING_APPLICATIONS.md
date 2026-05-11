---
title: Trit-routing primitive — application surface
companions: docs/THESIS.md · docs/FINDINGS.md · docs/TECHNICAL_DEBT.md · journal/cycle2_full_battery_findings.md · journal/td27_mechanism_2026-05-10.md
status: 1 application empirically validated (sparse attention); 12 applications scoped, awaiting tests (2026-05-10)
---

# Trit-routing primitive — application surface

## What we have

A substrate-distinct primitive: **direction-aware top-k via packed-trit
signatures + popcount distance**. Built from existing route primitives
(`m4t_route_threshold_extract` + `m4t_route_distance_batch`); validated
on one application (post-hoc sparse attention in BitNet b1.58-2B-4T
inference).

What "validated" means concretely (per `journal/cycle2_full_battery_findings.md`
and `journal/td27_mechanism_2026-05-10.md`):
- 24-prompt × 4-arm × 6-k battery passed all three pre-committed
  EVIDENCE gates for thesis Part B (substrate routing's advantage
  over alternatives widens with sparsity)
- Mechanism localized: trit signatures encode direction (sign + zero)
  natively, so signature-distance selection is direction-aware by
  construction. Direction-aware sparse selection beats direction-blind
  selection (oracle's `|Q·K|`); that's the bulk of the substrate's
  observed advantage

The primitive's substrate-distinct property is **direction-awareness as
a native representation feature**. Any "select top-k from a relevance
signal" decision in a ML pipeline is a candidate site where this
primitive could replace cosine similarity / LSH / score-magnitude top-k
/ small learned gates.

## Status of this document

This catalogues the application surface as of 2026-05-10. Most items are
**scoped opportunities, not measured findings**. The honest state is:
one application demonstrated, twelve plausible. The discipline that
validated the first should be applied to test the rest before claiming
they work.

Items are grouped:
- **A. Improvements to the validated application** (#1-#4): make the
  current sparse-attention implementation better
- **B. Research-level open questions** (#5-#7): what we don't know about
  the validated application
- **C. Applications beyond sparse attention** (#8-#13): where else the
  primitive could apply

Each item lists: what it is, how it works, cost estimate, substrate-
distinctiveness, prerequisites, what testing it would prove or refute,
and open questions specific to it.

---

## A. Improvements to the validated application

### #1 — K-signature caching

**What it is.** Cache K signatures alongside K and V in the KV cache.
Compute the signature once at K-write time; reuse on every subsequent
attention step.

**How it works.** Currently `bitnet_pick_routed_indices` recomputes K
signatures every attention step (O(seq_k × head_dim) per step). The
signatures depend only on K and tau. If tau is fixed (see #4 — it
currently isn't), K signatures are cacheable. Add a `cache->k_sig`
buffer alongside `cache->k`; populate at write; read at lookup.

**Cost.** ~1 day of engineering. Storage: 32 bytes (M4T_TRIT_PACKED_BYTES(128))
per cached K position per kv_head. For BitNet's max context 4096 ×
5 kv_heads × 32 bytes = 640 KB per layer, 19 MB across 30 layers —
acceptable.

**Substrate-distinctiveness.** Doesn't change the primitive; just
amortizes its cost.

**Prerequisites.**
- **Resolve the tau question first.** Currently `tau =
  bitnet_routed_pick_tau(Q)` is per-Q-step. To cache K signatures, tau
  must be deterministic from K alone (or a fixed constant). See #4.
- If a per-step tau IS load-bearing (sparser/denser based on Q's
  magnitude profile), the caching becomes more complex (cache K's RAW
  values + recompute signature at lookup, OR cache multiple K
  signatures per K).

**What it would test/prove.** Validates that the primitive's cost can
be amortized in a real KV-cached pipeline. Speedup at long context.

**Open questions.**
- Does fixed tau give the same quality as per-Q tau? (See #4 — must
  answer before this becomes a clean engineering win.)

### #2 — NEON-accelerate the sparse path

**What it is.** Production-quality NEON implementation of
`bitnet_sparse_attn_v_combine` (currently scalar — explicitly marked
experimental).

**How it works.** Same shape as `m4t_mtfp_attn_v_combine`'s NEON path
(`vmlal_s32` outer-product accumulate), but with an indirection through
the indices array. The indirection adds one gather per loop iteration;
substrate has `vld1q_lane_s32` for this.

**Cost.** ~2-3 days. Pattern is well-established in the substrate's
existing kernels.

**Substrate-distinctiveness.** Engineering. Doesn't change the primitive.

**Prerequisites.** None.

**What it would test/prove.** That the routed sparse path can be
production-quality (not just research code). Required before any
"production routed inference" claim.

**Open questions.** Whether the gather indirection's cost is dwarfed by
the FLOP savings at small k (probably yes, k=4 vs full 128 head_dim —
30× fewer dots per attention step).

### #3 — Hybrid two-stage routing

**What it is.** Use signature distance as a cheap first filter (top-k₁
candidates), then compute true Q·K on those candidates and pick top-k₂
< k₁ by signed score. Combines direction-awareness (via trit signatures)
with magnitude-awareness (via signed score).

**How it works.** Two sort passes:
1. Compute `seq_k` signature distances (cheap — popcount). Pick top-k₁
   smallest (e.g., k₁ = 16 of 128).
2. Compute true Q·K on those k₁ positions. Pick top-k₂ by signed score
   (e.g., k₂ = 4).

The first pass is `O(seq_k × head_dim_packed)`; the second is
`O(k₁ × head_dim)`. For seq_k=4096, k₁=16, k₂=4: total work is
4096 popcounts + 16 dots vs current routed's 4096 popcounts + 4 dots.
Modest cost increase; potentially better selection.

**Cost.** ~3-4 days. Adds a new arm to the harness.

**Substrate-distinctiveness.** HIGH. Combines two substrate-distinct
mechanisms (direction-aware filter + magnitude-aware refinement).

**Prerequisites.** None beyond current sparse-attention infrastructure.

**What it would test/prove.** Whether combining the two mechanisms beats
either alone. Could close (or re-open) the routed > posracle 1-prompt
gap. Could reveal interesting trade-offs (k₁/k₂ pareto curve).

**Open questions.** What's the right (k₁, k₂) ratio? Does the hybrid
approach generalize to other applications (#8-#13)?

### #4 — Per-layer / per-head / fixed tau (PARTIALLY CLOSED 2026-05-10)

**Verdict on the fixed-tau test:** per-Q tau is NOT load-bearing for
aggregate quality. A single fixed τ=5000 matches per-Q at 8/10 on the
focused subset (n=10). Lower fixed taus (500-2000) are at 7/10. Per-Q
adaptiveness costs an O(n log n) sort per Q-step and gives no
measurable quality benefit. **This unblocks #1.** Per
`journal/td27_4_fixed_tau_2026-05-10.md`.

Per-layer / per-head / learned tau still open as Phase 2 follow-ups —
the per-prompt variation pattern (per-Q and τ=5000 fail on different
prompts) suggests per-context calibration might outperform either
single-strategy approach.



**What it is.** Currently tau = 1/3-quantile of |Q| per Q-head per step.
Alternatives: fixed tau (calibrated once), per-layer tau (each layer
has its own constant), per-head tau (each head has its own constant),
learned tau (small parameter addition).

**How it works.** Replace `bitnet_routed_pick_tau(Q, head_dim)` with
one of:
- A constant (e.g., compile-time #define)
- A per-layer table (one tau per of 30 layers)
- A per-head table (one tau per of 20 attention heads)
- A learned scalar per layer/head (would require training — see #7)

**Cost.** ~1 day for fixed/per-layer/per-head; weeks for learned.

**Substrate-distinctiveness.** The current per-Q tau is one design
choice; alternatives are all valid.

**Prerequisites.** None for fixed/per-layer/per-head; #7 for learned.

**What it would test/prove.**
- If fixed tau gives the same quality as per-Q tau → routing decision
  doesn't need adaptive tau; K-signature caching (#1) becomes trivial.
- If per-Q tau is meaningfully better → the per-Q adaptiveness is itself
  a substrate-distinct contribution that adds to direction-awareness.

**Open questions.** What's the right tau? Is the answer the same across
prompts, layers, heads?

---

## B. Research-level open questions

### #5 — Why does routed slightly outperform posracle?

**What it is.** Cycle 2's full battery: routed_k=4 = 22/24, posracle_k=4
not yet measured at full battery scale (focused subset showed posracle
7/10 vs routed 8/10 — 1-prompt gap). Need larger eval to distinguish
real from noise.

**Hypotheses for the gap.**
- **H1 (representation robustness):** Trit quantization filters MTFP19
  noise that signed-score selection sees as signal.
- **H3 (head coordination):** Tau-based selection is more layer-coherent
  across heads than per-head argmax.

**Cost.** Run the full 24-prompt × 4-arm + posracle × 6-k battery (=
adds ~144 runs to the existing 456). ~1.5 hours wall-clock.

**Substrate-distinctiveness.** If H1 holds, that's a real substrate-
specific advantage (continuous-score-based posracle can't get the
benefit). If H3 holds, that's a research finding about attention
head coordination.

**Prerequisites.** None — `posracle` already implemented.

**What it would test/prove.** Whether the substrate's primitive has
ANY advantage over the simpler "direction-aware via signed-score filter"
explanation. If routed = posracle on the full battery, the substrate's
specific contribution is direction-awareness in a coarser representation
— still useful for compression/storage, less interesting as a quality
contribution.

**Open questions.** What workloads might amplify or shrink this gap?

### #6 — Effect with model scale

**What it is.** Tested on BitNet b1.58-2B-4T (2B parameters). Untested
on larger or smaller models.

**How it works.** Port the harness to another ternary model (BitNet
larger sizes if/when available, or a different ternary architecture).
Run the same 24-prompt × 4-arm × 6-k battery.

**Cost.** Depends on the model. If a same-architecture larger BitNet:
~1 week (weight conversion + harness verification). Different
architecture: months.

**Substrate-distinctiveness.** N/A — tests generalization.

**Prerequisites.** Another ternary model + weights.

**What it would test/prove.** Whether the routed > random / routed >
oracle pattern generalizes beyond a single model size. If the gap
WIDENS at larger scale, that's a strong scaling signal. If it shrinks,
the primitive may be an inefficiency-mitigation that bigger models
don't need.

**Open questions.** Does attention-loop frequency change with model
scale? Does sparsity tolerance change with scale?

### #7 — Cycle 3: routing-native attention with training

**What it is.** Architectural Part-B test. Design attention to use
substrate routing natively (not post-hoc); train; compare to dense
BitNet attention at matched FLOPs.

**How it works.** Three changes:
1. The Q/K projections produce trit-friendly representations directly
   (could be a trained MTFP4-output BitLinear instead of MTFP19).
2. The attention sparse-select is part of the forward pass (same as
   our current routed arm), differentiable via straight-through estimator
   or REINFORCE through `route_topk_abs`.
3. Training loop optimizes the model end-to-end.

**Cost.** Months of foundational kernel work. Prerequisites:
- `bitlinear_scale_bx_backward` and other gradient kernels for the
  attention path
- MTFP-native optimizer state (likely a lightweight Adam variant)
- Training loop integration

**Substrate-distinctiveness.** HIGH. Tests whether the substrate's
primitive amplifies advantage when the model is TRAINED to use it
rather than retrofitted.

**Prerequisites.** The full set of gradient kernels (TD-26).

**What it would test/prove.** The architectural Part-B claim. If
routing-native trained attention beats dense BitNet attention at
matched FLOPs, we have empirical evidence that routing-essentiality
isn't just a post-hoc mitigation.

**Open questions.** What's the right gradient estimator for the
discrete top-k selection? Does the training stability hold?

---

## C. Applications beyond sparse attention

### #8 — MoE gating via signatures

**What it is.** Mixture-of-experts gating that routes each token to k
experts based on signature similarity, instead of via a learned
gate-network MLP.

**How it works.** Per token:
1. Compute token's trit signature via `m4t_route_threshold_extract` (with
   tau from the token's own |feature| distribution or a calibrated
   constant).
2. Each expert has a "characteristic signature" (could be the average
   signature of the activations that historically routed to it,
   computed offline OR maintained as a running statistic).
3. Compute popcount distance from token's signature to each expert's
   signature.
4. Route token to top-k closest experts.

**Cost.** ~2 weeks: integrate with a MoE harness (none currently exists
in this project), implement the routing primitive call, validate quality
matches/beats a learned gate baseline.

**Substrate-distinctiveness.** HIGH. Substrate routing replacing a
learned neural network as the gating mechanism. If quality matches with
a fraction of the parameter count (no gate MLP needed), that's a clean
architectural advantage.

**Prerequisites.**
- A MoE harness (could be built on top of BitNet by replacing FFN with
  a small MoE block — ~1 week)
- A baseline learned gate to compare against

**What it would test/prove.** Whether the substrate's signature primitive
can replace learned routing. Direct test of "routing without learning the
router." If it works, it's a parameter-count reduction AND a Part-B
generalization test on a different attention-adjacent decision.

**Open questions.**
- How are expert "characteristic signatures" computed and maintained?
- Does the static (no-learning) routing give acceptable quality, or does
  it need calibration?
- Load balancing: substrate routing might cluster all tokens to a few
  experts; learned gates have explicit balancing losses.

**Why this is particularly interesting.** Learned MoE gating is famously
brittle (load-balancing issues, dead experts). Direction-awareness might
help by giving structurally different tokens structurally different
routing decisions.

### #9 — Sparse FFN activation prediction

**What it is.** Predict which `relu²(gate)` activations will be ZERO,
skip those cells' downstream computation.

**How it works.** BitNet's FFN: `gate_act = relu²(gate_proj(x)) × up_proj(x)`.
The relu² produces true zeros for negative inputs.

The proposal: a small "predictor" decides which gate cells will be
positive (and thus produce non-zero relu² output). Only compute
`up_proj(x)[j]` and the elementwise multiply for those cells.

The predictor: signature similarity. Maintain a small "characteristic
positive-gate signature" (positions where gate tends to be positive,
computed offline). At inference, compute token's signature and compare
to the characteristic — high similarity → likely positive gate cells.

**Cost.** ~3-4 weeks: requires more careful design than #8 because the
predictor's accuracy directly bounds quality (a cell predicted-zero
that's actually positive becomes a hard zero in the output).

**Substrate-distinctiveness.** HIGH. Direction-awareness is naturally
suited to predicting "is this dimension going to be positive."

**Prerequisites.** A characterization of the FFN's actual sparsity
pattern (what fraction of relu² outputs are zero on typical inputs).

**What it would test/prove.** Whether direction-aware prediction can
amortize FFN compute. Substrate-aligned because ternary networks
naturally produce sparse activations; predicting sparsity is a natural
use of the substrate's direction-awareness.

**Open questions.**
- What's the typical sparsity rate? If relu² output is mostly zeros
  already, this could give large speedups; if mostly non-zero, less.
- How to handle prediction errors (false-zero predictions)?

### #10 — KV cache eviction via signature distance

**What it is.** When KV cache fills up at long context, evict positions
whose K signatures are most-distant from recent Q signatures. Direction-
awareness is the right metric: opposite-direction positions are the
safest to drop.

**How it works.** Same routing pipeline, opposite decision. Maintain a
running "recent Q signature average." Compute distance from each cached
K's signature to this average. Evict the m positions with highest
distance when cache overflows.

**Cost.** ~1 week. Builds on #1 (K-signature caching). Adds an eviction
strategy module to the KV cache.

**Substrate-distinctiveness.** Direction-awareness as the eviction
heuristic. Other approaches (FIFO, LRU, attention-weight-based) are
simpler but direction-blind.

**Prerequisites.** #1 (K-signature caching). Long-context test workload
with cache pressure.

**What it would test/prove.** Whether direction-aware eviction outperforms
FIFO/LRU on long-context coherence.

**Open questions.** What's the right "recent Q signature" representation?
Single average, exponential moving average, attention-weighted?

### #11 — Retrieval / nearest-neighbor (extends Gesh phase A.1)

**What it is.** General-purpose top-k retrieval from a large embedding
collection. Gesh phase A.1 already used trit lattice signatures for
classification; extend to retrieval, recommendation, anomaly detection.

**How it works.** Standard retrieval pipeline:
1. Embed each item in the collection as a trit signature (one-time).
2. Embed query as a trit signature.
3. Compute popcount distance to each item.
4. Return top-k closest.

The substrate primitives (`threshold_extract` + `distance_batch` +
`topk_abs`) are the entire pipeline. Compared to:
- LSH: substrate signature is more direction-aware (LSH is binary and
  random-projection based)
- Cosine on embeddings: more compute, requires float
- Inverted-file approaches: substrate works without any training

**Cost.** Depends on application. For a classification task: ~1 week
(extend Gesh phase A's machinery to a real dataset). For retrieval:
~2-3 weeks (need a benchmark and baseline).

**Substrate-distinctiveness.** Trit signatures + popcount distance is
genuinely different from LSH or cosine. Direction-awareness might be
particularly valuable for recommendation (where "opposite direction"
items are the safe-to-skip negatives).

**Prerequisites.** A real benchmark (the prior cycle's CIFAR-10 hit a
"representation tax"; pick something more substrate-friendly).

**What it would test/prove.** Whether the substrate primitive competes
with established retrieval methods on a recognized benchmark. Would
broaden the substrate's claim from "ternary LLM inference works" to
"substrate primitive is competitive on retrieval."

**Open questions.** Which benchmark? Sentiment (3-class natural fit),
multi-label classification, dense retrieval?

### #12 — Speculative decoding via signature similarity

**What it is.** Detect when current generation state is similar to a
past state; reuse the past state's downstream computation as a
"speculation" that's accepted if validation matches.

**How it works.** During generation:
1. After each token, compute the residual stream's signature at some
   layer.
2. Maintain a small cache of (signature, downstream computation, token
   sequence).
3. For new tokens, check signature distance to cache entries.
4. If sufficiently close, speculate the cached downstream — verify
   correctness on a small subset, accept if correct.

**Cost.** ~4-6 weeks. Speculative decoding has known machinery; the
novelty here is the speculation rule (signature similarity vs. learned
draft model).

**Substrate-distinctiveness.** Substrate routing as the speculation
oracle. Different from standard speculative decoding (which uses a
smaller draft model to predict the larger model's output) because
there's no draft model — the substrate primitive IS the predictor.

**Prerequisites.** Significant — speculative decoding requires careful
state management; signature-based prediction is research-grade.

**What it would test/prove.** Whether direction-aware similarity is
load-bearing enough to support speculation. Aggressive but cool if it
works.

**Open questions.** What's the right signature layer? How big is the
cache? What's the verification rule?

### #13 — Cross-layer state caching

**What it is.** Within a forward pass, detect when an intermediate
representation (e.g., post-input-layernorm output) is similar to a
previously-seen state; reuse downstream computation.

**How it works.** Similar to #12 but within a single forward pass:
1. Compute signature of layer L's output.
2. Compare to a small cache of recent layer-L signatures + their
   layer-(L+1..L+n) outputs.
3. If similar enough, reuse the cached downstream outputs.

**Cost.** ~6-8 weeks. Fundamentally changes the forward pass control
flow. High risk; high reward if it works.

**Substrate-distinctiveness.** Substrate routing as the equivalence
oracle. Direction-awareness might detect "approximately the same state"
robustly across MTFP19 quantization.

**Prerequisites.** Significant. Requires modifying the forward pass to
support state caching + reuse decisions.

**What it would test/prove.** Whether the substrate's representation
makes "approximately equivalent state" cheaply detectable. If yes, it's
a substrate-aligned form of activation caching that base-2 substrates
can't easily replicate.

**Open questions.** How often does this actually trigger in real
inference? Is the per-token overhead of signature comparison less than
the saved compute?

---

## Suggested sequencing

If forced to prioritize, the following ordering balances:
- Cost (cheaper first)
- Information gain per unit cost
- Dependency chains

**Phase 1 — Validate primitive composition (~2 weeks):**
- #4 (fixed tau test) — answers prerequisite for #1
- #1 (K-signature caching, conditional on #4)
- #5 (full battery for posracle) — closes the routed > posracle gap

**Phase 2 — Test primitive generalization (~1-3 months):**
- #8 (MoE gating) — clean Part-B test on a different decision
- #11 (retrieval extension) — broadens the substrate's claim

**Phase 3 — Architectural / training (~3-6 months):**
- #7 (Cycle 3) — the architectural Part-B test
- #2 (NEON-accelerate sparse path) — production-quality

**Phase 4 — Speculative / aggressive (~6+ months):**
- #9 (sparse FFN activation prediction)
- #10 (KV cache eviction) — once #1 lands
- #3 (hybrid two-stage routing) — refinement

**Out of scope for now:**
- #6 (model scale) — depends on external model availability
- #12 (speculative decoding) — research-grade, big lift
- #13 (cross-layer state caching) — research-grade, biggest lift

## What this document IS

A scoped opportunity surface for the substrate's validated trit-routing
primitive. Each item sketches a research / engineering investigation,
not a proven result.

## What this document IS NOT

- A claim that any of #2-#13 will work
- A commitment to execute any specific item
- A roadmap for the project (the project's roadmap is driven by the LMM
  cycles; this is one input)

## Maintenance

When an item is investigated, link the relevant journal cycle from the
item's section. When closed (positive, negative, or methodologically
inconclusive), update the item with the verdict. Items that turn out to
be wrong-shaped or duplicate should be marked and explained.
