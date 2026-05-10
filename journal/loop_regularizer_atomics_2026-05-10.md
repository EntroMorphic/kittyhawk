# Loop-regularizer atomics — mechanism investigation

Per `journal/cycle2_full_battery_findings.md`. The Cycle 2 surprise
finding (substrate-routed at k=4 outperforms dense at k=128 on
loop-prone prompts) had a hypothesis attached: "routing-as-attention-loop-
regularizer." This investigation tests the hypothesis with the same
atomics methodology used for RMSNorm and math_div.

## The canary

`edge_repetitive` (input: 15 "yes" tokens). Dense continues the loop
forever ("yes yes yes...×60"). Routed_k=4 breaks out at gen position 2:

```
gen pos 0 ✓  dense → " yes"  routed → " yes"
gen pos 1 ✓  dense → " yes"  routed → " yes"
gen pos 2 ✗  dense → " yes" (loop) routed → "\n" (break)
```

Both arms see the SAME 18-token shared context at gen position 2.

## Phase 1 — where divergence enters

Per-layer comparison at position 17 (the next-token-prediction position):

| L0 site | dense vs routed_k=4 |
|---|---|
| x_norm_input | **bit-exact identical** |
| q_pre_rope | bit-exact identical |
| k_pre_rope | bit-exact identical |
| v | bit-exact identical |
| q_post_rope | bit-exact identical |
| k_post_rope | bit-exact identical |
| **attn_sub_norm** | **ε = 0.31, 73% of cells differ** ← injection point |
| x_norm | ε = 0.12 |
| ... (downstream) | ... |

Confirmed: divergence enters EXACTLY at the attention computation. All
pre-attention sites are bit-exact identical (same input → same
embeddings → same projections → same RoPE).

## Phase 2 — divergence cascade

`block_output` ε per layer (the carrier of divergence through the
residual stream):

```
L 0  ε=0.108  ████
L 1  ε=0.176  ███████
L 2  ε=0.234  █████████
L 3  ε=0.255  ██████████
L 4-13 PLATEAU around 0.25-0.27
L14-29 SHRINKING (0.26 → 0.12)
```

**Surprise: divergence rises early, plateaus, then SHRINKS through
late layers.** The residual stream + RMSNorm performs error correction:
each layer's normalization pulls magnitudes back to a common scale,
and the residual's accumulated state dominates per-layer attention
contribution as we go deeper. Late layers' attention contributions
are smaller relative to the residual.

By L29, ε is back down to 0.12 — but logits at int64 magnitude ~1e10
mean small ε on residuals = enough delta on logits to flip an argmax
between two close candidates (10035 " yes" vs 198 "\n").

## Phase 3 — what positions did substrate routing select?

Replicated the routing decision in Python (using captured `q_post_rope`
and reconstructed K cache from per-position L0 dumps):

| head | routed top-4 indices | dense top-4 by score | overlap |
|---|---|---|---|
| 0 | [0, 15, 16, 17] | [0, 15, 16, 17] | 4/4 |
| 1 | [0, 15, 16, 17] | [0, 15, 16, 17] | 4/4 |
| 2 | [0, 15, 16, 17] | [0, 15, 16, 17] | 4/4 |
| 3 | [0, 15, 16, 17] | [0, 15, 16, 17] | 4/4 |
| 4 | [7, 8, 9, 10] | [0, 8, 9, 10] | 3/4 |
| 5 | [2, 3, 13, 14] | [2, 12, 13, 14] | 3/4 |
| 6 | [13, 14, 15, 16] | [13, 14, 15, 16] | 4/4 |
| 7 | [12, 13, 14, 15] | [0, 12, 13, 14] | 3/4 |
| ... | ... | ... | ... |

**Average overlap: ~3.3 of 4 positions.** Substrate routing largely
picks the SAME positions dense's softmax weights most heavily. The
selection is dominated by BOS (position 0) and the recent tokens
(positions 14-17), with some heads picking a middle window.

The substrate ISN'T picking radically different positions from what
dense would emphasize. The substrate is picking SIMILAR positions and
forcing the softmax to renormalize to JUST those.

## Phase 4 — testing the hypothesis with controls

The original hypothesis: "substrate routing acts as a loop regularizer
by selecting diverse-by-signature K positions, breaking attention-loop
dynamics."

But the full-battery data (which we already have) tells us:
- **oracle_k=4 ALSO broke the edge_repetitive loop** (produced
  "and it's a good for the other..." — non-loop, just different
  non-loop content)
- random_k=4 produced "yes, yes yes, and the same difference between
  them yes yes no no..." — also non-strict-loop, but incoherent
- ONLY dense_k=128 produced the strict loop "yes yes yes yes...×60"

**Sparsity itself breaks the loop, regardless of selection rule.**
The substrate routing IS NOT the loop-regularizer mechanism. The
mechanism is sparsity.

## Revised mechanism story

| arm | sparsity? | relevance-aware? | result on loop-prompt |
|---|---|---|---|
| dense_k=128 | NO | N/A | **LOOP** (mid-weight tokens cumulatively reinforce) |
| random_k=4 | YES | NO | INCOHERENT (positions aren't meaningful) |
| oracle_k=4 | YES | by raw \|Q·K\| | COHERENT (different content) |
| routed_k=4 | YES | by signature distance | COHERENT (different content) |

**What's happening:** dense's softmax distributes some weight across
ALL 18 positions. The mid-weight positions (the middle "yes" tokens)
each contribute small mass but **cumulatively reinforce the loop
pattern**. Aggressive sparsification cuts this cumulative reinforcement
off. Sparsity is the loop-breaking mechanism; the choice of routing
rule determines whether the resulting output is coherent vs incoherent.

## What this means for Cycle 2's Part-B claim

**Refines, doesn't undermine.** The Part-B EVIDENCE finding from
Cycle 2 was: routed beats random by widening margins as k decreases.
That's still true. The atomics now tells us WHAT routed is doing
better than random:

- Both routed and random sparsify (so both break loops)
- Random's chosen positions aren't relevance-aware → output is
  incoherent garbage
- Routed's signature-distance selection picks positions that ARE
  relevance-aware → output is coherent

The substrate-distinct contribution is **coherent sparse attention via
packed-trit signature similarity**. Not "loop regularization" — that's
generic to any sparsity. The substrate's specific contribution is
making sparse attention MEANINGFUL via the signature pipeline.

## The routed > oracle finding (full battery: 22/24 vs 15/24 at k=4)

This is the genuinely substrate-distinct signal that the atomics
reveals. Both routed and oracle pick relevance-aware sparse positions,
but they pick BY DIFFERENT METRICS:

- Oracle: top-k by raw |Q·K| score (mantissa magnitude)
- Routed: top-k by signature distance (packed-trit popcount)

Routed's metric is COARSER (3 trit states per cell × 128 cells vs
continuous Q·K) but might be more ROBUST in the small-bx regime BitNet
operates in. Or routed's metric might pick positions that are USEFUL
under softmax-redistribution rather than positions with high raw scores
(which oracle picks).

Mechanism for "routed > oracle" is undertheorized. The atomics here
doesn't isolate it; would need a separate experiment varying selection
metric while holding sparsity constant. Recorded as TD candidate.

## Updated honest framing for the surprise finding

Original framing (from cycle2_full_battery_findings.md):
> "aggressive substrate-routed attention selects diverse-by-signature
> K positions, breaking attention-loop dynamics that dense locks into"

Refined framing:
> "Aggressive sparsification — by ANY rule (random, oracle, or substrate-
> routed) — breaks attention-loop dynamics in greedy decoding by cutting
> off cumulative mid-weight reinforcement. Substrate-routed sparsification
> is distinguished from random by producing COHERENT output (because
> signature-distance selection is relevance-aware), and from oracle by
> a measurable but mechanistically-undertheorized advantage on the
> 24-prompt battery."

## Methodology lifts

1. **Atomics investigation as hypothesis refiner.** The atomics didn't
   confirm the original hypothesis; it REFINED it. The "routing-as-loop-
   regularizer" framing was overclaiming — sparsity IS the regularizer,
   substrate routing is what makes the regularization coherent.

2. **Replicating the substrate decision in Python on captured dumps.**
   The K cache could be reconstructed from per-position L0 dumps; the
   routing decision (threshold_extract + popcount distance) is simple
   enough to replicate without needing harness instrumentation.

3. **Existing battery data answered competing-explanation questions.**
   The "did sparsity alone break the loop" question was answered by
   looking at oracle_k=4's existing result on edge_repetitive — no new
   experiment needed.

4. **Negative-result-as-finding (again).** The atomics did NOT find
   a substrate-distinct loop-regularizer mechanism. That's a finding
   about what the substrate does NOT specially do, which is informative.

## Open follow-ups

- **Why does routed > oracle on the full battery?** Not answered here.
  Would need a controlled experiment varying selection metric (signature
  distance vs raw |Q·K| vs other) while holding sparsity constant.
  Possibly relevant: the discrete-vs-continuous selection rule,
  robustness to noise, softmax-redistribution interaction.

- **Does the "sparsity breaks loops" effect generalize beyond BitNet
  greedy decoding?** Untested. Could test on another small LLM or under
  sampling decoding.

- **What if dense had additional damping?** E.g., dense + temperature-
  adjusted softmax that suppresses mid-weight positions explicitly.
  Would test whether the "extra damping" can recover dense's quality
  without sparsification.

## Verdict

The atomics RESOLVES the mechanism question:
- LOOP-BREAKING is from sparsity, not from substrate routing specifically
- COHERENT sparse attention is the substrate-distinct contribution
- The original "routing-as-loop-regularizer" phrasing was a hypothesis
  that the atomics REFINED rather than confirmed

The Cycle 2 Part-B EVIDENCE finding stands. The mechanism story is now
more honest.
