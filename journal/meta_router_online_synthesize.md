---
date: 2026-04-15
scope: LMM cycle — online, inline meta-router
phase: SYNTHESIZE
status: CLOSED 2026-04-15 — superseded by journal/fused_filter_fix.md
---

> **CLOSING NOTE (added 2026-04-15):** The experiment plan below was executed in part. P1 passed cleanly (tools/mnist_local_vs_global.c, 891 rescues / 331 damages, 2.7:1 ratio). The P1 atomic decomposition (tools/mnist_lvg_atomics.c, journal/lvg_atomics_decomposition.md) predicted ~88% ceiling for any observable-signal meta-router. Before building P2, a follow-up rerun with a structural fix at the local level (tools/mnist_local_v2.c, journal/fused_filter_fix.md) recovered 89% of the L→Gq gap WITHOUT any meta-routing: L200_H12 (fused H1+H2 filter) lifted local from 83.86% to 88.87%, within 0.59 points of pure global Gq. The meta-router was a proposal to route around a deficient filter; the correct answer turned out to be to deepen the filter. **P2 is not worth building** — the oracle ceiling on top of L200_H12 is 91.75%, leaving only ~2.88 points of theoretical headroom even for a perfect router, and rescues/damages remain observationally indistinguishable. The LMM cycle still produced its value: the P1 gate forced the atomic measurement, the atomic measurement exposed the filter-ranking structure of the gap, and the fused-filter fix became visible. See `journal/fused_filter_fix.md` for the result that replaces this experiment plan.

# SYNTHESIZE: the meta-router experiment

## Thesis

**Build a meta-router as k-NN over routing-context signatures, whose bank grows inline from cascade execution traces, and test whether it recovers meaningful ground between local fusion (83.86%) and the global-equivalent ceiling (~91.55% at N_PROJ=16×4).** The experiment runs in two gated phases: (P1) a prerequisite measurement that answers whether global beats local on local-failures, and (P2) — only if P1 passes — a minimal meta-router prototype.

## Architecture (restated from REFLECT)

```
query
  ↓
H1 primary hash over all 60K prototypes      [GLOBAL, 960K trit-ops]
  ↓
top-50 candidate pool
  ↓
H2+H3+H4 fusion over top-50                   [LOCAL, 4K trit-ops]
  ↓
emit cascade state: (H1_min_dist, tied_count,
  H1-H2_disagreement, H2-H3_disagreement,
  quadruple_margin, query_sig_bucket)
  ↓
pack state into routing-context signature     [TRIT PACK, 4 bytes]
  ↓
meta-router k-NN over bank[N_bank]             [OBSERVER, N_bank × 4 bytes]
  ↓
dominant action among M nearest neighbors
  ↓
  ├─ accept-local: return quadruple-fusion top-1
  └─ escalate-global: run H1+H2+H3+H4 over all 60K
                      and return top-1 of summed distance
  ↓
APPEND (context_sig, action_taken, final_signal) TO BANK
  ↓
return predicted label
```

**Key architectural facts:**

- H1, H2+H3+H4 local, and the context signature computation all happen *every* query. This is the always-on local cost.
- Global escalation runs conditionally. If the bank says this query's context looks like past queries that benefited from global, we pay the cost; otherwise we don't.
- The bank update (append-on-execution) happens *every* query, not just escalated ones. The bank accumulates behaviors uniformly across the query stream.

## Routing-context signature design

Pack six signals into 16 trits (32 bits of 2-bit codes, 4 bytes — one NEON word for Hamming):

| signal | bits | trit interpretation |
|---|---|---|
| H1 min distance bucket | 3 bits → 2 trits | 0..3=very close, 4..8=medium, 9..16=loose, 17+=far |
| tied-min count bucket | 3 bits → 2 trits | 1, 2-4, 5-15, 16+ |
| H1-H2 disagreement on top-1 | 1 bit → 1 trit | +1 same, −1 differ |
| H2-H3 disagreement on top-1 | 1 bit → 1 trit | +1 same, −1 differ |
| quadruple-fusion margin bucket | 3 bits → 2 trits | quantized rank-1 vs rank-2 gap |
| query signature checksum (8 trits) | 8 trits | XOR-folded H1 query sig — spatial locality |

Total 16 trits per context entry. Bank entry = 4 bytes.

## Bank structure

- Fixed-size ring buffer: 4096 entries × 4 bytes = 16 KB.
- Each entry holds (context_sig, action_taken, self_supervised_outcome_bit).
- Action alphabet: binary (0 = accept-local, 1 = escalate-global).
- Self-supervised outcome bit: 1 if the final-stage top-1 had "high confidence" (margin above a threshold after the chosen path), 0 otherwise. This is the signal, not ground truth — it marks *apparent* success.
- Insertion policy: append to ring; overwrite oldest entry on wrap.
- Optional reservoir-sample mode for stationary distributions (skip overwrites with probability 1/(n_seen / N_bank)).

## Meta-router lookup

- Compute query's context signature.
- Compute Hamming distance to all N_bank entries. For 4096 entries × 4 bytes this is ~16K trit-ops — about 1.7% of the H1 primary cost. Amortizable.
- Take top-M neighbors (M=5). Count action_taken weighted by self_supervised_outcome_bit. Action with higher weighted count wins.
- Ties broken toward accept-local (cheaper default).

## Cold-start strategy

- **Phase seed (offline, pre-run):** run local quadruple + global quadruple over first 2000 training queries. For each, compute the context signature, record (context, action_that_agreed_with_ground_truth_label, outcome_bit). Preload the ring buffer with these 2000 entries.
- **Phase explore (live, first 500 queries):** regardless of meta-router recommendation, randomly escalate 30% of queries to gather diverse behavior data. Bank grows.
- **Phase exploit (live, after 500 queries):** meta-router recommendation is authoritative. Random exploration drops to 5% to keep the bank calibrated under slow drift.

## Experiment protocol

### P1 — Prerequisite measurement (gate)

Tool: extend `tools/mnist_cascade_atomics.c` or add `tools/mnist_local_vs_global.c`. Over all 10K MNIST test queries, compute both:

- Local quadruple cascade prediction (current best: 83.86%)
- Global quadruple prediction: `(topd[j] + dB[j] + dC[j] + dD[j])` computed over all 60K prototypes (rather than over top-50 only)

Report the 2×2 contingency table:

|  | local correct | local wrong |
|---|---|---|
| global correct | a | **b** |
| global wrong | c | **d** |

**Gate criterion:** if `b > d` (global rescues more than global damages), meta-router is architecturally sound — proceed to P2. If `b ≈ d`, both architectures are hitting the same ceiling and meta-routing can't help; cycle ends with an honest negative.

**Predicted outcome of P1:** `b` is probably in the range 400-800 queries (4-8% of test set), `d` probably <100. The ratio should heavily favor escalation. But this is a prediction, not a certainty.

### P2 — Meta-router prototype (conditional)

Tool: new `tools/mnist_meta_router.c`. Implements the architecture above end to end. Reports:

1. Aggregate accuracy.
2. Escalation rate (fraction of queries that took the global path).
3. Conditional accuracy of accept-local vs escalate-global decisions.
4. Bank churn (how many entries were overwritten).
5. Cold-start vs warm-state comparison (accuracy over the first 500 queries vs after).
6. Comparison against three baselines:
   - Pure local quadruple (83.86%)
   - Pure global quadruple (predicted ~91.55%)
   - Random-escalation baseline at the same escalation rate as the meta-router

Success criteria:

- Aggregate accuracy must beat pure local by ≥ 2% (margin significant).
- Escalation rate < 50% (otherwise meta-router is just global fusion with overhead).
- Meta-router must beat random-escalation-at-same-rate by ≥ 1% (otherwise it isn't actually learning anything).
- Cold-start accuracy (first 500) should be no worse than pure local.

### Size estimates

- P1 tool: ~150 lines C (copy from resolver_sweep, add global scan).
- P2 tool: ~500 lines C (signature packing, ring buffer, meta-router lookup, 6 reporting sections).
- Total: ~650 lines, one afternoon.

## Implementation details

### Self-supervised outcome bit

After the chosen path (local or global) produces a top-1 label, compute the *confidence margin*: the difference between the top-1 summed distance and the top-2 summed distance, normalized by max_dist. Mark outcome_bit = 1 if this margin exceeds the median margin seen so far; 0 otherwise.

This is label-free. It captures "did this query end in an unambiguous answer" regardless of whether the answer was correct. Noisy but usable: meta-router learns to associate "this context ends unambiguously under action X" with action X, which is a reasonable heuristic.

### Exploration schedule

```
if query_idx < 500: exploration_prob = 0.30
elif query_idx < 2000: exploration_prob = 0.10
else: exploration_prob = 0.05
```

### Bank eviction

Simple FIFO ring on the first implementation. Optional reservoir sampling as a knob.

## Follow-ups (post-P2)

1. **Expand action vocabulary.** Binary → 4-way (accept-local / widen-K / escalate-global / committee-vote). Requires wider bank schema but same k-NN lookup.
2. **Per-class meta-routers.** Separate bank per candidate class. Measures whether the meta-router's context signatures carry class-conditional information.
3. **Drift detection.** Compare rolling-window bank entries against the stable mass. Trigger full-explore phase on detected drift.
4. **Context signature as learned primitive.** Replace the hand-designed 16-trit context with a learned signature produced by a small routing head. Still label-free — learned from cascade state, not from correctness.
5. **Integration with existing adaptive-voting idea.** The N_PROJ=16 atomic probe proposed adaptive voting based on tied_count. That's a 1-bit meta-router. This cycle generalizes it to k-NN lookup.

## What this experiment *isn't*

- Not a learned classifier. No gradients, no backprop, no differentiable anything. Everything is routing primitives.
- Not a proof of "learning" in the strong sense. The bank accumulates observed cascade states; it doesn't build a causal model. Whether this qualifies as "learning" is a terminology question.
- Not a production-ready architecture. It's a minimum-viable mechanism test.
- Not dependent on labels at inference time. Labels appear only in the cold-start seed (offline training data) — if needed at all. Warm-state operation is self-supervised.

## Honest risks

1. **P1 might fail.** Global may not beat local on local-failures. In that case we've falsified the premise cheaply — that's the point of prerequisite gating.
2. **The context signature may not carry enough information.** 16 trits is tight. If the meta-router can't separate easy from hard queries, the accuracy gain vanishes.
3. **The bank may saturate early and stop helping.** If context signature space is small (few distinguishable contexts), the bank fills and updates become noise.
4. **Self-supervised outcome bit may correlate too weakly with actual correctness.** The whole update rule collapses to random noise if the outcome proxy is bad.

All four risks are measurable in P2. A failed experiment is a good experiment if it tells us which risk fired.

## Next action

Build `tools/mnist_local_vs_global.c` for P1. Run it. Read the contingency table. Decide whether to proceed with P2.
