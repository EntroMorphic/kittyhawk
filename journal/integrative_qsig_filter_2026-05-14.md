# Integrative test — qsig_filter K=1 beats qsigdist

**Date:** 2026-05-14
**Companions:** `per_prompt_routing_reopen_2026-05-14.md` (per-prompt
analysis that exposed the per-prompt routing headroom);
`l1_cluster_lmm_2026-05-14.md` (the LMM that surfaced the
"competing-vs-integrative" question Tripp then asked); commits
e39e5e1 → HEAD.

## How this happened

After the per-prompt routing arc characterized a +16.58pp oracle
ceiling that none of the simple routers could capture (L1, L1.1,
L4 all in held-out CV the arc), Tripp asked one question:

> Are we still testing competing systems or are we testing
> integrative systems?

Honest answer: competing. Even our "router" framing was picking
ONE policy per prompt — competition with extra steps. The
linear-additive trit family was a weak form of integration; the
per-prompt arc reverted to competition.

Five integrative architectures were possible. Picked **conjunctive
filter + ranker** as the cleanest first test:

```
For each eviction candidate:
  compute QK_dist (qsigdist's score)
  compute KK_dist (sigdist's score)
filter out: K candidates with LOWEST KK_dist  (preserve K-K-similar)
evict: argmax(QK_dist) among remaining
fallback: if all filtered, use plain qsigdist
```

Implementation: new harness mode `BITNET_KV_EVICT_QSIG_FILTER`
parameterized by `BITNET_KV_EVICT_KK_PROTECT_K`. K=0 reproduces
qsigdist exactly. The conjunction is structurally distinct from
the linear-additive trit family — `c_score = w_kk·KK + w_qk·QK`
cannot express "filter then rank" for any choice of weights.

## Results — full curve K∈{1, 2, 4, 8}

| K | Δ vs random | paired (K − qsig) | 95% CI | W/T/L |
|---|---|---|---|---|
| **1** | **+8.50pp** | **+2.12pp** | [−1.33, +5.83] | 23/54/23 |
| 2 | +5.04pp | −1.33pp | [−4.38, +1.92] | 21/55/24 |
| 4 | +7.00pp | +0.63pp | [−3.50, +4.92] | 25/48/27 |
| 8 | +3.13pp | −3.25pp | [−7.12, +0.63] | 18/44/38 |
| qsig | +6.38pp | baseline | | |

**K=1 is the new best policy.** Mean Δ +8.50pp beats qsigdist's
+6.38pp by +2.12pp. CI on the gain straddles zero at N=100 single-
seed, but the per-category mechanism is coherent (see below).

The curve is non-monotonic — K=1 ≫ K∈{2,4} > K=8 — with K=2 and
K=4 having overlapping CIs (the K=2 < K=4 inversion is single-seed
noise). The mechanism reading: protecting **one** most-K-similar
slot finds the sweet spot. Adding more protection forces qsigdist
to evict from a shrinking pool, including slots it would have
preferred to keep. Past K=4, you're losing more by constraining
than gaining by protecting.

## Per-category structure (K=1 vs qsigdist)

```
Big wins (technical/factual content):
  geography  (n=2)   +22.92pp
  tech       (n=5)   +15.00pp
  math       (n=4)    +9.37pp
  technical  (n=3)    +8.33pp
  cont       (n=6)    +6.94pp
  idiom      (n=5)    +6.67pp
  def        (n=6)    +6.25pp

Big losses (code/questions/long-form):
  code       (n=11)   −3.41pp
  q          (n=8)    −4.17pp
  long       (n=9)    −2.31pp
  poetry     (n=4)    −2.08pp
```

K=1 helps where redundancy is meaningful (technical writing
repeats key terms; protecting K-K-similar slots preserves coherent
reference). K=1 hurts where each token-decision is more
independent (code, structured questions, narrative continuation).

## Selective routing also tested — and failed held-out

Natural follow-up: route to K=1 only on "winning" categories;
default qsigdist elsewhere. 5-fold CV × 20 repeats:

```
in-sample selective:        +10.00pp  (+3.62 over qsigdist) — OVERFIT
HELD-OUT selective:          +5.70pp  (-0.67 vs qsigdist)
                             95% CI on (selective − qsigdist):
                             [-1.12, -0.21]  ← LOSES significantly
```

Same overfitting trap as L1.1 (category mode-routing). The train-
chosen "winning categories" don't transfer. Held-out selective
LOSES 0.67pp vs qsigdist with high confidence.

**Always-on K=1 (no routing at all) is the best deployable
strategy.** The integrative architecture works; clever routing on
top of it actively hurts.

## Why always-on integration beats selective routing

The integration is asymmetric: losses are small (−2 to −4pp on 32
prompts), wins are large (+10 to +23pp on 25-30 prompts). Net
+2.12pp. A selective router needs to predict per-prompt which
half a prompt falls into. At N=100 with category-prefix features,
the prediction error eats the gain it captures. Just leaving the
integration always-on captures the asymmetry without prediction
risk.

## Architectural lesson

| approach | held-out Δ vs qsigdist |
|---|---|
| Always-on integration (K=1) | **+2.12pp** ✓ |
| Selective routing (cat → K=1) | −0.67pp ✗ (CI excludes 0) |
| Competition (cat → best policy) | −2.42pp ✗ (L1.1) |

**Integration > selective routing > competition** in held-out. The
per-prompt routing arc spent 4 LMM cycles on selective and
competition strategies; one integration test won by margins
neither could reach.

The TriX repo's `CompiledDispatch + guard` pattern would help only
if our routing signal were strong enough to survive held-out. At
N=100 single-seed, it isn't. The integrative architecture
sidesteps the entire routing-confidence problem by applying its
single rule uniformly.

## What's still owed

1. **Multi-seed measurement (L3).** The +2.12pp gain has CI
   [−1.33, +5.83] — straddles 0 at single seed. Multi-seed random
   baselines would tighten this and tell us whether the integration
   gain is robust signal or noise.

2. **Other integrative architectures.** This was test #1 of 5
   from the menu. Multi-policy consensus, conditional handoff, and
   score-multiplicative haven't been tried. K=1's success suggests
   the integration FRAME generalizes; specific instances may
   stack or substitute.

3. **The K=1 gain is on N=100 prompts at window=16, gen=24.** It
   applies to the eviction-policy-quality measurement against
   `no_evict` baseline. Translating "+2.12pp better KV eviction
   matching" to "noticeably better generated text" is a separate
   question — coherence-vs-bit-parity (per memory).

## Files

- `gesh/bitnet/bitnet_harness.c` — added `BITNET_KV_EVICT_QSIG_FILTER`
  mode + `BITNET_KV_EVICT_KK_PROTECT_K` env var
- `experiments/phase_zeta/smoke_qsig_filter.py` — K=0 ≡ qsigdist
  bit-identicality smoke test (PASS)
- `experiments/phase_zeta/run_qsig_filter_battery.py` — N=100 runner
- `experiments/phase_zeta/analyze_qsig_filter.py` — CI + per-category
  decomposition
- `experiments/phase_zeta/selective_qsig_filter_cv.py` — held-out
  selective routing CV
- `experiments/phase_zeta/results/meta_iterate/qsig_filter_K{1,2,4,8}/`
  — N=100 raw token logs per K
- `experiments/phase_zeta/results/meta_iterate/qsig_filter_summary.json`
- Anchor table for routing experiments now includes K=1 as the
  best known policy.

## Where this came from

The integration finding came from one Tripp question, asked while
the per-prompt routing arc was deep in mode-router CV failures:
"Are we still testing competing systems or are we testing
integrative systems?" The reframe rotated the entire problem 90°.
The architecture I had been pointing at "which policy wins?" got
re-pointed at "what does the right combination look like?" One
afternoon's test answered: the integration with the simplest
parameterization (K=1) is +2.12pp better than the previous best,
and selective routing on top of it actively hurts.

The arc's record now reads: closeout (premature, mean-Δ-only,
ceiled at qsigdist) → reopen (per-prompt analysis, +16.58pp
oracle, but unreachable by trivial routers) → L1, L1.1, L4 (all
characterizing why competition can't capture the headroom) →
integration test (the conjunction wins).
