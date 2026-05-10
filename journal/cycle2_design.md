# Cycle 2 Design — N4 sparse attention experiment

Per `partB_experiments_synth.md` Cycle 2 launch criteria. Implementation-shape
detail for the four experimental arms.

## Experimental arms — what each does

The current attention path (`gesh/bitnet/bitnet_harness.c` lines 350–436):

```
for each query head h:
    for each cached position t in [0, position]:
        scores_i64[t] = vec_dot_i64(Q[h], K_cache[t][kv_head])
    score_shift = adaptive_shift_for_softmax(scores_i64)
    scores_int = scores_i64 >> score_shift  (clamp to MTFP19)
    weights = softmax(scores_int)
    out[h] = attn_v_combine(weights, V_cache, ..., seq_k)
```

The four arms differ in how they construct `weights` over `seq_k` positions.

### Arm A — Dense (current behavior)

Weights computed over all `seq_k` positions. Baseline.

### Arm B — Random top-k

Pick `k` positions uniformly at random from `[0, seq_k)` (without
replacement). Compute scores and weights only on those. Mask the rest.

Implementation: generate index array, dot only on selected positions,
softmax on the k scores, attn_v_combine on the k positions.

### Arm C — Substrate-routed top-k (the experimental arm)

Use the substrate's route primitives to pick the `k` positions with
highest predicted Q·K affinity, before computing the actual dot products.

Implementation:
1. Compute Q signature: `m4t_route_threshold_extract(qh, tau)` →
   packed-trit signature
2. Compute K signatures: same primitive on each `K_cache[t][kv_head]`
   (cache-able, but for now compute per-step)
3. `m4t_route_distance_batch(q_sig, k_sigs[0..seq_k])` → seq_k popcount
   distances (smaller distance = closer signatures = predicted higher affinity)
4. `m4t_route_topk_abs` on negated distances (so top-k by closeness) →
   indices of top-k positions
5. Compute true Q·K dot only on those k positions
6. Softmax + attn_v_combine on the k positions

The `tau` parameter for threshold_extract needs choosing. For Q and K
which are int32 mantissas at activation scale, a reasonable default is
the median of |element| in the vector, ensuring all three trit states
are realized (per §18 input-class contract).

### Arm D — Oracle top-k (upper bound)

Compute the FULL dense scores first (same as Arm A). Then pick the k
positions with highest |score|. Recompute weights as softmax on JUST
those k scores. attn_v_combine on those k positions.

Implementation: dense scores first, sort by abs value, keep top-k, recompute.
This is slower than dense (extra sort + second softmax) — that's fine,
it's a research baseline.

## Mode selection

Add an env variable `BITNET_ATTN_MODE` and `BITNET_ATTN_K`:

- `BITNET_ATTN_MODE` ∈ {`dense`, `random`, `routed`, `oracle`} (default: `dense`)
- `BITNET_ATTN_K` ∈ positive int (default: head_dim = 128, i.e. equivalent
  to dense for `random`/`routed`/`oracle` modes)

Read once at harness startup; passed into `bitnet_forward_block` via the
state struct or a global static.

When mode is `dense` and the env is unset, behavior is bit-exact identical
to the current production path. The four-arm code lives in `bitnet_harness.c`
behind a runtime conditional but doesn't change the default.

## What we measure per (mode, k)

For each `(mode, k)` config:

1. **Strict pass rate** on the 24-prompt battery (manual classification +
   loop heuristic with manual review of borderline cases).
2. **Token agreement %** vs dense baseline: same prompts, count token
   matches at each position; aggregate.
3. **Per-layer ε** vs dense baseline at one or two probe positions.
4. **Wall-clock per token** (sanity: routing should save FLOPs at small k).

## Pre-commit gates (formalized from the SYNTH)

| Outcome | Conditions (ALL must hold) |
|---|---|
| **PART-B EVIDENCE** | At k=64 (head_dim/2), routed strict-pass within 10pp of dense AND at k=16 routed beats random by >10pp AND gap routed-vs-random WIDENS as k decreases |
| **PART-B FALSIFICATION** | Routed indistinguishable from random across the trajectory (within ±5pp at every k) OR routed degrades faster than random as k decreases OR no k value gets routed within 20pp of oracle |
| **INCONCLUSIVE** | Quality varies wildly per prompt and trajectory is noisy OR wall-clock results contradict expected FLOP savings (implementation issue) |

## Trajectory

`k ∈ {128, 64, 32, 16, 8, 4}`. With BitNet's `head_dim = 128`, k=128 is
equivalent to dense (sanity check that the sparse paths reduce to dense).

For `random`/`routed`/`oracle`, k=128 should produce identical output to
dense (modulo stable-sort tiebreaks).

## Implementation phasing

To de-risk, implement in increasing complexity:

**Phase 2.1** — Add mode selection infrastructure (env var, struct field,
pass-through to forward_block). Default = dense. Verify zero diff vs
current behavior when env is unset.

**Phase 2.2** — Implement Arm B (random top-k). Easiest. Validates the
sparse-attention infrastructure end-to-end. Runs on 1-2 prompts to
sanity-check.

**Phase 2.3** — Implement Arm D (oracle top-k). Tests the post-hoc
selection logic. Slow but well-defined.

**Phase 2.4** — Implement Arm C (substrate-routed top-k). The experimental
arm. The hard one because of signature computation + cache management.

**Phase 2.5** — Run the full trajectory on 1 probe prompt across all 4
arms × 6 k values = 24 runs. Validate methodology before scaling up.

**Phase 2.6** — Full battery: 4 arms × 6 k values × 24 prompts = 576
prompt-runs. Probably 4-8 hours wall-clock. Run in background.

**Phase 2.7** — Analyze, decide pre-commit gate verdict, write up.

## Implementation notes

- Existing route primitives are in `m4t/src/m4t_route.{h,c}`. The required
  ones (threshold_extract, distance_batch, topk_abs) all exist and are
  NEON-routed.
- Q/K signatures need to be computed in packed-trit format. The route
  primitives expect this format directly — no additional conversion needed.
- For Arm C, K signatures could in principle be CACHED alongside V/K in
  the KV cache (compute once on K-store, reuse per Q step). For Cycle 2
  initial implementation, recompute per step — simpler. Optimize later
  if performance is a concern.
- The `route_topk_abs` primitive has a hardcoded max T=64. For our use
  with `seq_k` up to 4096 (BitNet max position), we'd exceed this.
  Mitigation: chunk K signatures into 64-position blocks, run topk_abs
  per block, then merge the per-block top-ks via a final scalar pass.
  OR: add a wider topk_abs variant. For Cycle 2, chunking is the
  simpler choice.

## What this design does NOT cover

- **Caching of K signatures** in the KV cache. Performance optimization
  for later.
- **Adaptive k selection** (vary k per step based on attention entropy).
  Out of scope; we want the trajectory measurement first.
- **Combined routing modes** (e.g., always include positions 0 and -1
  for sink-attention style). Out of scope.
- **Sampling decoding** (vs greedy). Cycle 2 is greedy-only; sampling
  is a follow-up.

## Loop-back triggers

If during implementation any of these surface, loop back:

1. **The route primitives don't compose cleanly** for this use case →
   may need to design a wrapper function. Loop back to Cycle 2 design.
2. **The chunked topk_abs approach is too lossy** (chunked top-ks merge
   poorly compared to a global top-k) → may need to widen the route_topk_abs
   max T or implement a custom ranking. Loop back.
3. **Arm B (random) shows wildly different quality from dense even at
   k=128** → implementation bug; halt and debug before proceeding.
4. **The wall-clock for a single probe prompt is multiple minutes** →
   the full battery becomes intractable; need to scale down (fewer
   prompts or fewer k values).

## Estimated effort

- Phase 2.1: 1-2 hours (plumbing)
- Phase 2.2 (random): 2-4 hours
- Phase 2.3 (oracle): 2-4 hours
- Phase 2.4 (substrate-routed): 4-8 hours (signature computation + topk_abs chunking)
- Phase 2.5 (probe): 1-2 hours
- Phase 2.6 (full battery): 4-8 hours wall-clock (mostly inference)
- Phase 2.7 (analysis): 2-4 hours

Total: 16-32 hours of focused work. The Cycle 1 estimate was 2-4 weeks
calendar time; this matches at ~10 hours/week pace.

## What I commit to in this session

This session will likely accomplish Phase 2.1 + Phase 2.2 (random top-k
working as a sanity-checkable baseline), and possibly Phase 2.3 (oracle).
Phase 2.4 (the substrate-routed arm — the actual experimental arm) is
significant enough to warrant its own session and possibly its own LMM
sub-cycle.

The honest framing: this session's deliverable is "Cycle 2 infrastructure
in place + 1-2 baselines working," not "Cycle 2 result."
