# Plan B — Q-aware eviction (qsigdist) NEGATIVE in the harness

User directive: "Red-team and remediate A. Then proceed with B."

Plan A established that production sigdist's K-K direction proxy
fails to approximate Phase ε's Q-K oracle (ρ ≈ 0.055 across all M).
Red-team A then showed K-K eviction is 2× WORSE than random in the
single-shot oracle (L2 1.18 vs 0.59 at k_keep=32).

Plan B implements the Q-K oracle directly as a new production
eviction mode (`qsigdist`). At eviction time, Q is already in the
state buffer (`s->q`), so the change is architectural-but-bounded:
extend `bitnet_kv_evict_pick_victim` to accept a Q pointer and
compute eviction cost as Σ over all 20 Q-heads of L1(Q-sig_qh,
K-sig at p for qh's KV-head). Evict argmax.

## Single-shot oracle (sanity)

Python simulation of the production qsigdist decision (layer-global
Σ across 5 KV-heads × 4 Q-heads/KV) on the Phase ε 5-prompt corpus.
Mean attention-output L2 error per q-head trial:

| policy                              | k_keep=8 | k_keep=16 | k_keep=32 |
|-------------------------------------|---------:|----------:|----------:|
| oracle per-q-head (Phase ε's L1)    | 0.1504   | 0.0660    | **0.0164**|
| **qsigdist (layer-consensus)**      | 0.2815   | 0.1583    | **0.0602**|
| K-K M=1 (production sigdist)        | 1.5165   | 1.4318    | 1.3186    |
| random                              | 1.3502   | 1.0361    | 0.5790    |

qsigdist achieves L2 = 0.060 — **10× lower than random** (0.579) and
**22× lower than K-K M=1** (1.319). The layer-consensus penalty vs
per-q-head oracle is 3.68× (0.0602 / 0.0164). Strong expectation
from this number alone: qsigdist should beat random in the harness.

## Harness territory (the territory)

Full battery rerun with qsigdist added (5 prompts × 24 gen × 3 windows):

| window | fifo | random | sigdist | **qsigdist** |
|-------:|-----:|-------:|--------:|-------------:|
| 8      | 22.5%| 26.7%  | 26.7%   | **26.7%**    |
| 16     | 81.7%| 84.2%  | 78.3%   | **73.3%**    |
| 32     | 99.2%|100.0%  |100.0%   |100.0%        |

(mean match-rate vs no-eviction baseline; window=32 has no eviction
pressure on these prompts.)

**At window=16, qsigdist is ~11pp WORSE than random — the worst
non-no_evict policy on aggregate.** Per-prompt:

| prompt          | fifo  | random | sigdist | qsigdist |
|-----------------|------:|-------:|--------:|---------:|
| bos_only        | 83.3% | 100.0% | 79.2%   | 100.0%   |
| capital_france  |100.0% | 100.0% | 100.0%  | **75.0%**|
| medium_11       | 91.7% |  87.5% | 79.2%   | **37.5%**|
| short_a         | 58.3% |  58.3% | 58.3%   | **79.2%**|
| short_b         | 75.0% |  75.0% | 75.0%   | 75.0%    |

qsigdist is **bimodal**: matches random on bos_only, beats random by
21pp on short_a, but loses by 50pp on medium_11 and 25pp on
capital_france. The mean is dragged down by the two catastrophic
losses.

## The arc's reversal pattern

| layer of measurement                  | result                              |
|---------------------------------------|-------------------------------------|
| Phase α/β/γ (synthetic)               | Q-K L1 is a good selector           |
| Phase δ/ε (real K-cache, per-q-head)  | Q-K L1 reduces L2 by 38-62% vs Hamming, 35× vs random |
| Phase ζ harness (production sigdist)  | K-K proxy ≈ random; slightly worse on flips |
| Plan A (M sweep)                      | No M recovers Q-K from K-K          |
| Plan A red-team (single-shot L2)      | K-K proxy is 2× WORSE than random   |
| **Plan B (qsigdist) single-shot**     | **Q-K oracle: 10× BETTER than random**|
| **Plan B harness territory**          | **Q-K oracle: ~11pp WORSE than random** |

**The arc has reversed at every layer past per-q-head L2.** The
substrate's Q-K oracle property is real, repeatedly measured, and
correctly implemented in qsigdist. It still doesn't survive the
harness territory.

## Working hypothesis for the new reversal

Single-shot vs sequential-decision dynamics. The single-shot oracle
asks: "Given this Q and a frozen cache, which K's serve this one
attention call best?" The harness asks: "Given a stream of Q's,
each making one irreversible eviction decision, how does the cache
trajectory hold up over 24 generation steps?"

Random eviction preserves cache **diversity** by uniform sampling.
qsigdist preserves the K's most similar to the **current** Q's
direction at each step. Over many steps, the "current Q direction"
rotates as generation continues; qsigdist greedily evicts K's far
from each step's Q, but those K's may have been needed by a future
Q. The decisions compound destructively, and the cache is
progressively biased toward "K's similar to recently-generated
content" — a recency-by-semantic-similarity that fails when
generation needs older context.

This is consistent with the bimodality: short_a (where qsigdist
wins) likely has Q-directions that don't rotate much (single
content focus, qsigdist's greedy choice happens to keep useful
K's). medium_11 (where it catastrophically loses) likely has
content shifts that need diverse historical K's; random preserves
them, qsigdist throws them away.

This is the same shape as `feedback_proxy_to_territory_pattern`,
one layer deeper: even the per-q-head L2 oracle, which IS the
"operational" metric I previously promoted, fails to predict the
sequential-decision harness behavior. The oracle is a property of
single decisions; the territory is a property of trajectories.

## Honest verdict

The substrate-claim arc on KV-eviction is now **complete with a
final NEGATIVE result on the territory**:

- L1 path-graph distance: well-defined, correctly implemented in
  `m4t_popcount_dist`, substrate-distinctive. **Unchanged.**
- Q-K L1 oracle (Phase ε): real, measurable, 38-62% better than
  Hamming and 35× better than random in per-q-head single-shot
  oracle. **Unchanged.**
- Production sigdist (K-K proxy): does not approximate the oracle;
  is 2× worse than random in single-shot terms. **Anti-correlated
  with attention relevance.**
- qsigdist (Q-K oracle, correctly implemented in production):
  matches single-shot oracle predictions (10× better than random)
  but **loses to random by ~11pp at window=16 in harness
  generation, with catastrophic per-prompt variance.**

**No substrate-based KV-eviction policy beats random on this
harness benchmark.** The Phase α→ε arc earned the per-q-head
oracle finding fairly, but that finding does not predict
sequential-eviction harness behavior even when the oracle is
implemented correctly.

## What the arc accomplished, honestly

- Established that the production substrate kernel computes L1 path-
  graph distance (td28 finding).
- Showed Q-K L1 is a strong per-q-head selector against Hamming and
  random (Phase α/β/γ/δ/ε).
- Caught that production sigdist uses K-K, not Q-K, making the
  Phase ε result inapplicable to production as-shipped (Phase ζ
  mechanism investigation).
- Implemented Q-K eviction correctly (qsigdist) and showed it
  still doesn't win the territory test.
- Demonstrated multiple disciplines: territory-vs-map, oracle-vs-
  trajectory, proxy-vs-end-to-end.

## What to do with the code

Plan C (proceed):

1. Mark `qsigdist` and `sigdist` as **research modes**, not
   production-recommended. The production default (no eviction) is
   unchanged; both opt-in modes are demonstrably no better than
   random on this benchmark, and sigdist is actively worse.
2. Document this finding in `gesh/bitnet/README.md` so future
   readers don't see the modes and assume they're production-ready
   improvements.
3. Leave the code intact: it's measurement infrastructure for the
   substrate-claim arc, useful for future studies.

The remaining honest research question — "is there ANY substrate-
based KV-eviction policy that beats random in autoregressive
generation?" — is open. Plan B answered the *cleanest* version
("use the Q-K oracle") with no. Further variations (M-step Q
averaging, attention-weighted eviction, history-aware policies)
might fare better, but each would need its own territory test, and
this arc has spent its budget.

## Files

- `gesh/bitnet/bitnet_harness.c` — adds `BITNET_KV_EVICT_QSIGDIST`
  mode (lines ~157-167, ~191, ~210-216, ~488-668, ~1106-1128).
- `experiments/phase_zeta/qsigdist_oracle_sanity.py` — single-shot
  oracle reference.
- `experiments/phase_zeta/eviction_battery.py` — updated to include
  qsigdist in all aggregations.
- `experiments/phase_zeta/results/battery_results.json` — territory
  battery with qsigdist column.

## Discipline log

The substrate-claim arc has produced one clean methodological
contribution per phase. Plan B's contribution is: **the per-Q-head
oracle metric, which I called "operationally meaningful" in Phase ε,
fails to predict sequential-decision harness behavior even when
correctly implemented.** This is one step deeper than
`feedback_proxy_to_territory_pattern`: not just "proxy doesn't
predict territory" but "the right oracle doesn't predict the right
territory when the territory is sequential-decision."

Adding to that memory: oracles measure single-decision quality;
territories measure trajectory quality. When the territory involves
many sequential decisions, no single-decision metric — not even the
optimal one — necessarily predicts trajectory quality. The
diversity-preservation property of random sampling is genuinely
load-bearing for sequential decisions and is not captured by any
single-decision oracle metric.

This is the deepest substrate-claim-arc finding, and it's negative.
The honest conclusion is the substrate is not a useful eviction-
selection criterion in this harness regardless of which oracle
implements it. Future work that wants substrate eviction to win
needs to attack at the trajectory level (e.g., diversity-aware
selection), not the single-decision level.
