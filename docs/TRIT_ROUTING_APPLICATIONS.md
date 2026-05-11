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

### Scope tiers

To prevent flattening "1 day engineering" against "6-8 weeks speculative"
into the same rank, items are tagged:

- **TIER-S (small)**: hours to days. Engineering or focused experiment.
  Bounded scope; outcome predictable in shape if not in detail.
- **TIER-M (medium)**: 1-4 weeks. Research with a clear question; needs
  dedicated cycle but not foundational work.
- **TIER-L (large)**: months. Foundational additions OR speculative
  research-grade investigations.

### Substrate-distinctiveness rubric

To make HIGH / MEDIUM / LOW assessments calibrated:

- **HIGH**: Uses substrate primitives that have no base-2 analog at the
  implementation level. A base-2 implementation would require building
  trit-packing infrastructure first.
- **MEDIUM**: Uses substrate primitives but a base-2 implementation could
  approximate the same behavior with different primitives (e.g.,
  signed-score selection on a base-2 substrate ≈ direction-aware
  selection without trit signatures).
- **LOW**: No substrate primitives needed; the substrate just happens to
  be the implementation environment.

---

## A. Improvements to the validated application

### #1 — K-signature caching [TIER-S; CLOSED]

**Verdict:** implemented + bit-exact + small (~2%) speedup at short
context. Cache is a uint8_t buffer in `bitnet_kv_cache_t` (~19MB max);
populated at K-write when fixed tau active; read at routing lookup;
no-op when env unset (production unchanged). 4/4 prompts produce
identical output to uncached fixed-tau path. Speedup expected to scale
with context length (K-sig recompute is O(seq_k × head_dim) per step).
`BITNET_ATTN_NO_CACHE=1` env flag preserved as debug tool. Per
`journal/td27_1_k_sig_caching_2026-05-10.md`.



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

### #2 — NEON-accelerate the sparse path [TIER-S]

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

### #3 — Hybrid two-stage routing [TIER-M; CLOSED 2026-05-11]

**Implementation:** `gesh/bitnet/bitnet_harness.c` `BITNET_ATTN_MODE=hybrid`,
`BITNET_ATTN_K1` env (default 4×k_eff). Stage 1 calls
`bitnet_pick_routed_indices` → top-k₁ candidates; Stage 2 computes
true Q·K on those k₁ positions and picks top-k₂ by signed score via
`bitnet_pick_posracle_topk`. Sanity verified: at k₁ ≥ seq_k, hybrid
is bit-exact equal to posracle (Stage 1 collapses to a no-op).

**Result (focused subset, n=10, k=4, k₁ ∈ {8,16,32}):** at
first-30-token agreement with the dense reference, hybrid_k₁=16
aggregates 106/300 vs posracle 101/300 vs routed_fixed_τ_5000 28/300.
**hybrid_k₁=16 wins by 5% aggregate**, with the load-bearing
differences on `edge_single` (+3) and `long_summary` (+8); loss on
`long_history` (−6); ties elsewhere. **hybrid_k₁=8** is the right
choice on tight integer-token tasks (math_add: 9 vs 3 for all other
arms). Coherence (loop heuristic): all hybrid configs pass on all 10
prompts; routed_fixed_τ_5000 loops on `code_loop` (recorded as a
caveat to #4's "fixed τ acceptable at quality").

**Substrate-distinctiveness — qualified (LOAD-BEARING WITHIN SUBSTRATE).**
Originally rated HIGH; downgraded to LOAD-BEARING-WITHIN. Stage 1
filter uses the trit substrate's signature/distance primitive
natively (signatures are produced as a byproduct of routing, not
extra compute). A scalar substrate could implement the same two-
stage pattern using LSH or partial-dot for Stage 1, but would pay an
extra projection cost. Two-stage routing is a substrate-compatible
*operating point*, not a substrate-uniquely-superior algorithm.

**Cost story.** At seq_k=4096 with k₁=16: hybrid does ~16 dots/step
vs posracle's ~4096 dots/step — ~256× fewer dots, plus a cheap
popcount Stage 1. On THIS battery (seq_k ≤ 71), the gap is ~4×.
The scaling claim is extrapolation; a long-context measurement is
recorded as the next test.

**Open follow-ups (recorded as TDs, not blockers):**
- Long-context quality + cost measurement at seq_k ≥ 1024.
- Spot-check the divergent outputs (e.g. long_summary k₁=16 vs
  posracle) to confirm "better English" vs "coincidentally-tracks".
- k₁ default sweep at scale (current default 4×k_eff is best
  aggregate here but the response is noisy).

**Journal:** `journal/td27_3_hybrid_2026-05-11.md`.

### #4 — Per-layer / per-head / fixed tau (PARTIALLY CLOSED 2026-05-10)

**Verdict on the fixed-tau test (n=10, focused subset):** per-Q tau
appears NOT load-bearing for aggregate quality. A single fixed τ=5000
matches per-Q at 8/10 on the focused subset. Lower fixed taus (500-2000)
are at 7/10. **This PROVISIONALLY unblocks #1** — but the n=10 evidence
is thin, and per-Q vs τ=5000 had different per-prompt blind spots
(suggesting per-context tau might be the actual right answer). #1's
implementation should treat fixed-tau choice as a calibration step
informed by a wider set of prompts. Per
`journal/td27_4_fixed_tau_2026-05-10.md`.

Per-layer / per-head / learned tau still open as Phase 2 follow-ups.



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

### #5 — Why does routed slightly outperform posracle? [TIER-S; CLOSED 2026-05-10]

**Verdict:** routed ≈ posracle at most k values. The TD-27 focused-subset
finding (8/10 vs 7/10) doesn't replicate at full battery; loop heuristic
suggests posracle ≥ routed at k=8/16/32 but spot-checking reveals
heuristic FPs (heuristic systematically penalizes routed's noun-repetition
in coherent prose). At k=4, routed has +2 prompts of n=24 — may be real
or noise. Per `journal/td27_5_posracle_full_2026-05-10.md`. The
substrate-claim story tightens: substrate routing is **a competitive
implementation** of direction-aware sparse attention, not a uniquely
superior one in this workload at most k.



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

### #6 — Effect with model scale [TIER-L; UNDERSCOPED]

**Underscoping note (red-team correction):** "depends on model" cost
estimate isn't actionable. To make this concrete: would need either
(a) a larger BitNet variant if Microsoft releases one, OR (b) port
another small ternary model (~1-2 month engineering effort to replicate
the BitNet harness for a different architecture). Not pursuing without
a specific scope reduction.

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

### #7 — Cycle 3: routing-native attention with training [TIER-L]

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

### #8 — MoE gating via signatures [TIER-M; FALSIFICATION-PROBE DONE 2026-05-11; FULL DEFERRED]

**Cheap probe ran 2026-05-11 (prerequisite check):** partitioned
BitNet's existing FFN intermediate (6912 → 4 × 1728 slices), masked
k of N=4 slices per token via either oracle (sum |gate_act[slice]|)
or random scoring. Result (n=4 prompts):
- **25% slice mask (k=3/N=4, oracle):** robust on all 4 prompts.
- **50% slice mask (k=2/N=4, oracle):** task-sensitive — works on
  factual/conversational, breaks on math+code.
- **Oracle > random consistently at both ratios** — load-bearing
  result for #8's premise. Smart slice selection MATTERS; substrate
  routing has room to add value.

**What's still deferred (full #8):** substrate-routed gating against
a learned-gate baseline on a trained MoE model. Requires:
- A MoE harness (~1 week to build; doesn't exist).
- A trained learned-gate baseline (~1 week + training compute).
- Substrate-routing implementation (~1 week).
- Quality + load-balancing measurement (~1 week).
- Honest scope total: **3-5 weeks**, future cycle.

**Cost reality check (red-team correction):** the project doesn't have
a MoE harness. The "~2 weeks" estimate below assumed one existed; total
honest estimate including MoE harness construction is **~3-5 weeks**
(building a small MoE block is itself a research project on the
substrate; integrating it with BitNet's FFN replacement is harder still).

**Journal:** `journal/td27_8_ffn_probe_2026-05-11.md`.

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

### #9 — Sparse FFN activation prediction [TIER-M]

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

### #10 — KV cache eviction via signature distance [TIER-M; NEGATIVE RESULT 2026-05-11]

**Implementation:** `bitnet_harness.c` adds `BITNET_KV_EVICT_MODE`
∈ {none, fifo, random, sigdist} and `BITNET_KV_WINDOW`. `bitnet_kv_cache_t`
gains an `evicted` bitmask. sigdist policy evicts positions with MAX
summed-popcount K-sig distance to the **current position's K-sig**
(direction proxy). Dense attention masks evicted scores to
`-M4T_MTFP_MAX_VAL` before softmax. Sanity verified bit-exact dense
when window > seq_k.

**Result (4-prompt focused subset; max seq_k ~60-70; windows 16 and 32):**

| policy | first-30-agreement w=16 | first-30-agreement w=32 |
|---|---|---|
| fifo | 20/120 | 70/120 |
| random | **28/120** | **78/120** |
| sigdist | 26/120 | 60/120 |

**sigdist does not beat random eviction** on this battery. At w=16,
nar_storm sigdist triggers the loop heuristic: "The rain was the only
sound." × 4. **Self-reinforcing diversity collapse failure mode** —
when the model is in a repeating state, current K-sig clusters with
recent K-sigs, sigdist evicts the *diverse* older positions, the model
loses semantic anchors, the loop reinforces.

**Substrate-distinctiveness — NOT VALIDATED.** The direction-aware
eviction premise is plausible, but the simple "current K-sig as
direction proxy" implementation fails. Refinements (running-mean K-sig,
Q-sig direction, exclude-N-most-recent) are recorded as follow-ups but
NOT pursued in this cycle.

**What this validates:** the eviction infrastructure works (bit-exact
sanity, composes with #1's K-sig cache). The FFN/attention tolerates
50% eviction in moderate-seq_k regime with non-substrate policies.

**What this does NOT validate:** sigdist as a substrate-distinctive win;
long-context eviction quality (seq_k > 1024 unmeasured).

**Journal:** `journal/td27_10_kv_evict_2026-05-11.md`.

**This is the substrate's second clean negative result (after P0-4),
honestly recorded.**

### #11 — Retrieval / nearest-neighbor (extends Gesh phase A.1) [TIER-M]

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

### #12 — Speculative decoding via signature similarity [TIER-L; SPECULATIVE]

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

### #13 — Cross-layer state caching [TIER-L; SPECULATIVE]

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
