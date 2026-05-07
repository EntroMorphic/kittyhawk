---
cycle: routed16 activation sparsity (compressed LMM)
phase: measurement + verdict
date: 2026-05-07
scope: SYNTHESIZE step #1 from journal/routed16_synthesis.md —
       measure post-ReLU² and other BitLinear input sparsity in a
       real BitNet forward pass to determine whether routed16 has
       a current win condition.
companions: journal/routed16_synthesis.md (the question this answers),
            commit 8737959 (routed16 red-team remediation),
            scripts/measure_activation_sparsity.py (measurement tool).
---

# routed16 activation sparsity — verdict

## Question

routed16's K=6912 down_proj crossover sits at 92-94% sparsity (and
96-97% for K=N=2560). BitNet's ternary weights are 38-50% sparse,
well below all crossovers. The remaining open question (routed16
SYNTHESIZE #1): does any BitLinear's *activation input* exceed 92%
sparsity in real inference? If yes, routed16 has a place on that
specific BitLinear. If no, routed16 has no current win condition.

## Method

8 diverse prompts (capital_france, largest_planet, def_fib,
haiku_ocean, first_president, year_2024, reading_mind,
quick_brown_fox), tokenized via the BitNet HF tokenizer to 4-9
token IDs each. Ran the existing harness (--prompt-tokens + --dump)
to capture per-(layer, position) activation snapshots in the
ACTV2 format. Computed zero-cell fraction at every site that feeds
a BitLinear:

  - x_norm — feeds q_proj, k_proj, v_proj, gate_proj, up_proj
  - attn_sub_norm — feeds o_proj
  - ffn_sub_norm — feeds down_proj

Total samples: 8 prompts × 56 positions × 30 layers = 1680 per
site (some prompts shorter, captured per-position).

## Result

  site             min %    median %   max %    pairs ≥ 92%
  x_norm           0.00     0.00       0.23     0 / 1680
  attn_sub_norm    0.00     0.04       24.14    0 / 1680
  ffn_sub_norm     13.06    58.26      87.49    0 / 1680

Zero out of 1680 samples reach ANY crossover for any shape.

The closest case is ffn_sub_norm (down_proj input) which hits 87.49%
in the highest-sparsity (layer, position) pair across the entire
sweep. Extrapolating from the v2 bench curve at K=6912: 85% → 0.86×,
90% → 0.93×. So even at peak sparsity, routed16 loses to dense by
~14%.

ReLU² is the source of essentially all FFN sparsity. Gate (post-relu²)
and ffn_sub_norm show near-identical sparsity distributions, meaning
the up-proj multiply and RMSNorm step preserve the zero pattern
without amplifying it.

x_norm (the input to 5 of 7 BitLinears) is wholly dense — RMSNorm
on a sum of attended outputs basically never produces exact zeros.

## Verdict

**routed16 has no current win condition in BitNet inference.**

All 7 BitLinears read inputs whose maximum observed sparsity falls
below the relevant crossover. The closest call is down_proj at
~87% sparsity vs 92% crossover.

This is a complete, end-of-investigation negative result. The
investigation question ("can routing-as-speed pay off in BitNet?")
is closed: it cannot, with the existing weights, the existing
activation distribution, and the existing kernel representation.

## Disposition

routed16 remains in libm4t as **infrastructure**, not as a
production-active kernel:

  - It is correct (bit-exact across 33 test cases, ASAN+UBSAN clean).
  - It documents the empirical crossover by shape.
  - It is callable today by any future operation whose sparsity
    exceeds the relevant crossover.

Operations that *might* exceed the crossover, none of which BitNet's
training recipe produces:

  - Structured top-k attention (only top-k scores nonzero per query)
  - MoE routing (active expert mask creates >90% structural sparsity
    in expert selection)
  - Retrieval-sparse queries (lookup against a much larger key space
    where most keys score zero)
  - Pruned-and-distilled BitNet variants with explicit activation
    sparsification training

None of these are currently in scope. If any becomes scope, routed16
is ready.

## Why this matters more than the kernel

The "math as signatures via routing" foundation (project memory,
project_vision.md) is a *representational* claim — the substrate
represents and computes ternary projections via routing. That
remains validated.

The *speed* version of the routing claim — "routing-as-skip
outperforms dense" — is now closed for BitNet: at the operation,
shape, and sparsity BitNet actually produces, dense wins. This is
not a refutation of the foundation; it is a tight bound on where
the foundation buys speed.

What this teaches: the substrate's value is correctness and
representational fidelity, not (yet) compute reduction. If we want
the substrate to also win on speed, we either need an operation
with naturally extreme sparsity, or a different sparse
representation that beats SDOT throughput at moderate sparsity.
Both are open research questions; neither is a near-term path
through BitNet.

## What I would do next (not done now)

1. **Close the routed16 work-stream.** It is complete and correct;
   leave it as infrastructure. No further optimization investment
   until a use case appears.
2. **Document the speed-vs-correctness distinction in the project
   vision.** The memory entry feedback_routing_correctness_vs_speed.md
   captures this; lifting it into project_vision.md may be
   warranted on the next vision-level edit.
3. **Investigate whether activation sparsity can be *induced*** —
   e.g., add a per-token L1 penalty to the FFN intermediate during
   a quick fine-tune. This is a research direction, not engineering.
4. **Move attention from sparsity to a different speed lever** for
   the substrate: A8 → MTFP4 transition, KV cache compression,
   batched prefill. Each is potentially load-bearing without
   needing extreme sparsity.

The honest end of this branch is: routed16 is correct, costs nothing
to keep, and waits for a use case. Most of the value here was the
empirical floor we mapped — what wins, what doesn't, and why.
