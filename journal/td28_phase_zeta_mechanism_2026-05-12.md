# Phase ζ mechanism — why substrate eviction doesn't beat random

User directive after Phase ζ negative result: "Let's dig into the
atomics and find out why we're not beating random."

Phase ζ showed that production `BITNET_KV_EVICT_MODE=sigdist` does
not beat random eviction on generated tokens. Phase ε had shown that
L1 substrate selection reduces per-Q-head attention-output L2 error
by 38-62% relative — a real, large, robust effect. Two atomic
investigations now explain the gap completely.

## Headline mechanism

**Production `sigdist` does not implement Phase ε's measured operation.**

- **Phase ε oracle (`experiments/phase_epsilon/eviction_full.py`)**:
  rank cached K-signatures by L1 distance to the **current Q's
  signature**. Keep the top-k closest.
- **Production sigdist (`gesh/bitnet/bitnet_harness.c:579`)**:
  rank cached K-signatures by L1 distance to the **current
  position's K signature** (with M=1, the default). Evict the
  farthest.

Spearman correlation between the two rankings, measured across
166,800 trials (5 prompts × 64 positions × 30 layers × 5 kv_heads ×
4 q_heads, cache sizes 8-63):

| statistic                                | value     |
|------------------------------------------|-----------|
| mean Spearman ρ (K-K vs Q-K)             | **+0.055**|
| median Spearman ρ                        | +0.074    |
| std of Spearman ρ                        | 0.301     |
| 10th / 50th / 90th percentile            | -0.36 / +0.07 / +0.43 |
| fraction with ρ > 0.5                    | 5.8%      |
| top-1 eviction-target match (K-K vs Q-K) | **5.2%**  |

The top-1 match rate of 5.2% is consistent with uniform random
matching for cache sizes around 20-30. **Production sigdist's K-K
criterion is approximately uncorrelated with Phase ε's Q-K oracle.**

This means the entire Phase ε arc measured a property of an
operation that production sigdist **does not perform**. The
38-62% L2-error advantage is real for the Q-K oracle, but production
sigdist is not the Q-K oracle.

## Secondary mechanism — argmax robustness

Per-step instrumentation added to the harness
(`BITNET_LOG_PERSTEP=1` prints top-1 token, top-1 accumulator,
top-2 accumulator, and margin) ran on medium_11 and bos_only at
window=16. Findings:

**Argmax-flip counts (24 generation steps, vs no-eviction baseline):**

| prompt    | fifo  | random | sigdist |
|-----------|-------|--------|---------|
| medium_11 | 2/24  | 3/24   | **5/24**|
| bos_only  | 4/24  | 0/24   | **5/24**|

**Sigdist flips the argmax more often than random** on both
probed prompts. Not just "no advantage" — actively worse.

**Margin distribution at no-eviction baseline (medium_11, w=16):**

Margins range from ~8 billion (gen=4) to ~893 billion (gen=5). Most
generation steps have margins on the order of 10^11. The
cross-mode top-1 accumulator perturbations are on the same order
(~10^11). Argmax flips concentrate at low-margin steps:

- gen=8 (margin=119B): fifo flipped 46770→9099 with only 14B
  accumulator shift — the margin was not the safety it appeared.
- gen=14 (margin=20B): random flipped 2728→279 with 165B shift.
- gen=20 (margin=28B in bos_only): fifo flipped with 210B shift.

The chosen-token accumulator perturbations are not systematically
smaller under sigdist than under random. They are **direction-
random** because the K-K proxy is uncorrelated with Q-K — exactly
what the primary finding predicts.

## Why the K-K proxy fails on real data

The production design comment says (line 521-522):

> M=1: signature of current_position's K (original probe behavior;
> uses pre-cached k_sig directly).

The justification is that K_t (just written) approximates the
direction the model will attend toward at the next step (Q_{t+1}'s
direction). This relies on Q-K alignment after RoPE and through
the W_Q / W_K projections.

In the data: ρ(L1(K_t, K_i), L1(Q_{t+1}, K_i)) ≈ 0. The Q and K
projections produce signatures that, on these prompts at τ=5000,
rank the cached K's nearly independently. The K-K proxy is not
predictive of Q-K relevance.

This is why production sigdist behaves like random at the harness
level. It is not "L1 selection didn't translate"; it is "production
isn't performing L1 selection toward Q at all."

## Two falsified intuitions, one preserved

**Falsified A (cheap to test, ruled out immediately):**
"Phase ε tested the wrong baseline (Hamming vs L1) and random was
never measured." — FALSE. Phase ε did measure random; at k_keep=32
random's attn-output L2 error is 0.584 vs L1's 0.016, a 35×
advantage to L1 in the Q-K oracle. The Q-K oracle advantage is
real; the issue is that production doesn't implement Q-K oracle.

**Falsified B (per-step probe ruled out):**
"Sigdist's per-step logits are closer to no-eviction than random's;
the argmax is just robust to the difference." — FALSE on the
direction. Sigdist's argmax flips MORE often than random's, and
the chosen-token accumulator perturbations are not systematically
smaller. Argmax robustness exists at high-margin steps, but
sigdist isn't on the right side of the margin even when flips
matter.

**Preserved (the actual mechanism):**
The K-K direction proxy used by production sigdist is approximately
uncorrelated with the Q-K direction that would actually serve
attention. This is the load-bearing fact.

## What this means for the substrate-claim arc

- **L1 path-graph distance** as a metric: still well-defined, still
  correctly implemented in `m4t_popcount_dist`, still substrate-
  distinctive (collapses to Hamming on binary).
- **L1(Q-sig, K-sig) as an eviction oracle**: still a strong
  measured property (Phase ε numbers stand).
- **L1(K-sig[current], K-sig[i]) as a production eviction
  proxy**: **does not approximate the oracle**. On these data it's
  approximately uncorrelated with Q-K rankings. This is what
  production sigdist does today.

The arc's surviving claim now reads:

> The L1 path-graph metric on ternary signatures is well-defined
> and correctly implemented. **In oracle form (Q-K L1)** it is a
> strong KV-eviction selector — recall@k 5-17pp above Hamming,
> attention-output L2 error 38-62% lower. **In production sigdist
> form (K-K L1 with current K as direction proxy)** it is
> approximately equivalent to random eviction, because the K
> direction proxy is uncorrelated with the Q direction it's meant
> to approximate. The substrate distinctive property exists; its
> current production use does not exercise it.

## Implications for what would be testable next

Three obvious directions, not pursued in this investigation:

1. **Use Q directly.** Eviction currently happens at K-write time,
   before Q is computed. If eviction were deferred to the next-
   position attention step (where Q is available), the production
   path could use L1(Q-sig, K-sig) — Phase ε's oracle. This
   requires restructuring the eviction call site. Whether the
   restructuring is worth the speed cost is unmeasured.

2. **Improve the K direction proxy.** The harness supports
   `BITNET_KV_EVICT_M > 1`, which averages over the last M K's.
   Whether M=4 or M=8 (running attention-direction estimate)
   yields a better Q-correlated proxy is unmeasured. If
   ρ(K-mean-of-last-M, Q_{t+1}) > ρ(K-just-written, Q_{t+1})
   it could rescue the production criterion. Worth a quick sweep.

3. **Train an explicit direction predictor.** Use the prior token's
   embedding (or last hidden state) plus learned weights to
   predict next-Q-sig directly. Adds parameters; out of scope.

## Files

- `experiments/phase_zeta/perstep_probe.py` — per-step margin
  telemetry driver.
- `experiments/phase_zeta/qk_vs_kk_correlation.py` — the smoking-
  gun correlation analysis.
- `gesh/bitnet/bitnet_harness.c` — `bitnet_top2_full_vocab` and
  per-step logging gated by `BITNET_LOG_PERSTEP`.
- `experiments/phase_zeta/results/perstep/` — per-step logs.

## Discipline log

Third reversal-by-deeper-measurement in one day:

1. td28: Phase β/γ/δ/ε compared Python-Hamming-strawman vs
   Python-L1-target, but production C kernel already computed L1.
   Reframe.
2. Phase ζ (territory): Phase ε's per-Q-head L2 proxy did not
   predict end-to-end harness generation quality. Reframe.
3. Phase ζ mechanism (this entry): Phase ε measured Q-K L1
   selection; production sigdist does K-K L1 selection. Different
   operation. The "operationally meaningful" framing fails one
   more layer down.

The arc has now produced a clean, complete, mechanistic explanation
of its own primary reversal. The substrate-claim direction is real
for the oracle; the production criterion is approximately blind to
the oracle's structure. **The next testable substrate claim is
whether a Q-aware production eviction implementation would recover
Phase ε's advantage** — that's an implementation question, not a
measurement question.

This finding goes back to `feedback_verify_production_semantics`:
verify production semantics before designing the measurement. The
production semantic for sigdist is K-K L1 with M=1 K-mean proxy,
not Q-K L1. Phase ε's design didn't model this. The verification
should have happened before Phase β.
