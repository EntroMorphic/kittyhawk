# Plan B red-team — retracts the harness loss claim

User directive: "Red-team B".

Plan B (`td28_phase_zeta_planB_2026-05-12.md`) claimed qsigdist
loses to random by ~11pp at window=16 in the harness, on 5 prompts.
The single-shot oracle and trajectory simulation had qsigdist beating
random by 5-10× in L2 — a stark contradiction the journal attributed
to "trajectory dynamics."

The red-team uncovered three layered findings, each upgrading the
previous understanding. The final finding **retracts plan B's
negative verdict.**

## R-B1: Q-sig sparsity at τ=5000

At τ=5000, K-sigs are 70% nonzero (matches Phase γ measurement).
**Q-sigs at the same τ are 40% nonzero.** Real asymmetry. Q-sigs
are not degenerate (zero or all-nonzero), so signal exists, but
the Q-K L1 metric operates over differently-sparse signatures than
the K-K metric. Phase ε's τ=5000 was tuned for K, not Q.

Not the primary cause of the harness result; worth recording.

## R-B2: Drop-one single-shot — kills my trajectory hypothesis

Plan B's working hypothesis was "single-shot k_keep=32 advantage
doesn't apply to the harness's per-step drop-one granularity." Test:
single-shot L2 of dropping ONE position from cache size N ∈ {16,32,48}:

| policy   | N=16 | N=32 | N=48 |
|----------|-----:|-----:|-----:|
| qsigdist | **0.020**| **0.007**| **0.004**|
| kk_M1    | 0.076| 0.068| 0.087|
| random   | 0.074| 0.037| 0.025|
| fifo     | 0.072| 0.065| 0.100|

At N=32 drop-one, qsigdist's L2 is **5.1× lower than random**.
qsigdist still beats random at the harness's eviction granularity.
The "trajectory dynamics" hypothesis is wrong: the advantage IS
present at single-decision-granularity matching the harness.

## R-B3: Trajectory simulation — kills the correlated-drift hypothesis

If qsigdist's per-decision drift compounds correlatedly (always
biasing in one direction) while random's drift cancels by chance,
qsigdist would lose over many steps. Python trajectory simulation
walks position-by-position from window=16, applies one policy
eviction per step, measures cumulative L2 vs no_evict full cache:

| policy | cumulative L2 (mean over 750 trials) | last-step L2 |
|--------|--------------------------------------:|-------------:|
| qsigdist | **7.53** | **0.22** |
| kk_M1   | 63.51 | 1.43 |
| random  | 51.05 | 1.37 |
| fifo    | 64.79 | 1.41 |

qsigdist's cumulative drift is **6.8× SMALLER than random's** over a
48-step trajectory; last-step L2 is 6.2× better. **The trajectory
simulation predicts qsigdist should dominate random in the harness,
not lose to it.**

The simulation has one unavoidable assumption: Q at each step comes
from the no_evict dump (real harness Q depends on the policy's
evicted cache → diverges secondarily). So the simulation is an
upper bound on qsigdist's quality; the real harness loss could come
from secondary Q-drift. But the gap is so large (6.8×) that even
substantial secondary drift wouldn't explain plan B's reported 11pp
harness loss.

## R-B4: The bombshell — gibberish prompts

Decode the original eviction_battery prompts under the actual model
tokenizer (`microsoft/bitnet-b1.58-2B-4T-bf16`):

| label            | token IDs                          | decoded                                |
|------------------|------------------------------------|----------------------------------------|
| capital_france   | 1,1841,8085,341,9099,1735          | `'" car<p {\n minorobject'`            |
| short_a          | 1,464,2944,18                      | `'";\r\n reason3'`                     |
| short_b          | 1,791,28036,9100                   | `'"The bedsipes'`                      |
| medium_11        | 1,...,2728                         | `'" car<p {\n minorobject is the house at given'` |

**The entire 5-prompt Phase ζ harness battery was running on
out-of-distribution gibberish.** Token ID 1 (used as BOS) isn't even
this tokenizer's BOS (128000). The "capital_france" label was
misleading — the actual input was nonsense.

This nullifies:
- Phase ζ's "sigdist ≈ random or slightly worse" verdict (`td28_phase_zeta_eviction_territory_2026-05-12.md`).
- Phase ζ mechanism's argmax-flip analysis (5/24 sigdist vs 3/24 random) — flips were measured on gibberish-prompt generation.
- Plan B's "qsigdist loses by 11pp" verdict.
- The per-prompt bimodality framing (was noise on N=5 OOD inputs).

## R-B5: Rerun on real natural-language prompts (N=20)

Tokenize 20 diverse natural-language prompts with the correct
tokenizer (questions, definitions, continuations, math, dialog,
idioms, long descriptive). Rerun the harness battery at window=16:

| mode     | match% | 95% CI       | std    | wins/ties/losses vs random |
|----------|-------:|--------------|-------:|---------------------------|
| no_evict | 100%   | —            | —      | —                          |
| fifo     | 52.5%  | [41.5, 63.5] | 25.6pp | 9/4/7  (Δ=+1.5pp)          |
| random   | 51.0%  | [40.8, 59.8] | 22.7pp | —                          |
| sigdist  | 50.8%  | [41.9, 59.6] | 20.3pp | 8/7/5  (Δ=-0.2pp)          |
|**qsigdist**| **57.1%** | **[47.1, 67.1]** | **22.9pp** | 7/7/6  (**Δ=+6.0pp**) |

**Headline: qsigdist mean Δ vs random is +6.0pp, 95% CI [-5.6, +18.1].**

- Mean direction is **POSITIVE** for qsigdist (not negative as plan B
  claimed). The point estimate is qsigdist beats random by ~6pp.
- 95% CI straddles zero, so not statistically significant at α=0.05
  with N=20.
- Wins-ties-losses pattern (7/7/6) is balanced — qsigdist is
  competitive with random, not clearly winning or losing.
- Sigdist is essentially tied with random (-0.2pp), not 2× worse
  as the single-shot oracle implied. The single-shot oracle's harsh
  K-K result was an artifact of single-shot k_keep=32 dropping many
  positions at once; sequential drop-one over many steps does not
  produce the same per-decision compounding.
- Even fifo is competitive (+1.5pp). On natural-language prompts
  where attention has strong recency bias, all eviction policies
  cluster around 50-57% match-rate.

## Retractions

The following claims from prior journals must be retracted or
weakened:

1. **`td28_phase_zeta_eviction_territory_2026-05-12.md`**: "Substrate
   eviction does NOT beat random eviction on generation quality."
   Status: **measured on gibberish prompts; rerun on natural language
   shows sigdist tied with random (-0.2pp), qsigdist trending
   positive (+6pp, CI straddles zero).** Direction-of-effect is now
   different.

2. **`td28_phase_zeta_mechanism_2026-05-12.md`**: "Sigdist flips
   the argmax more often than random... 5/24 vs 3/24." Status: **on
   gibberish prompts; needs rerun on natural language to be
   load-bearing.**

3. **`td28_phase_zeta_planA_redteam_2026-05-12.md`**: "K-K's L2
   error is 2.02× random's" (single-shot oracle k_keep=32). Status:
   **the single-shot oracle finding is on Phase ε's c_dump_v3
   activations (which may be from gibberish or real prompts — origin
   unverified) and uses k_keep=32 drop-many granularity, not the
   harness's drop-one granularity. The 2× claim does NOT transfer
   to the harness on natural language**, where sigdist's match-rate
   is statistically identical to random's.

4. **`td28_phase_zeta_planB_2026-05-12.md`**: "qsigdist loses to
   random by ~11pp at window=16 in the harness territory." Status:
   **RETRACTED.** On 20 natural-language prompts, qsigdist's mean Δ
   vs random is +6.0pp. The "no substrate-based KV-eviction policy
   beats random on this harness benchmark" final verdict is
   **withdrawn**.

## What the arc actually says now

After all reversals and red-teaming:

- **L1 path-graph distance is well-defined and correctly implemented**
  in `m4t_popcount_dist`. Unchanged.
- **L1(Q-sig, K-sig) oracle (Phase ε)** beats Hamming by 38-62% in
  per-q-head single-shot attention L2 error. Unchanged.
- **Production sigdist (K-K M=1, current_K direction proxy)**: on
  natural-language harness territory, statistically tied with random
  (Δ=-0.2pp, N=20). The earlier "2× worse than random" was a
  single-shot-oracle artifact that does not translate to per-step
  harness behavior.
- **Production qsigdist (Q-K L1 with layer-consensus across all 20
  Q-heads)**: on natural-language harness territory, mean Δ vs random
  = **+6.0pp**, 95% CI [-5.6, +18.1] (statistically inconclusive at
  N=20 but direction is positive). The substrate's Q-K oracle
  property DOES partially propagate to the harness in this implementation.

**The substrate-claim arc has a possibly-positive harness territory
result on natural language**, contrary to my earlier "negative"
synthesis. The result needs more prompts to reach statistical
significance, but the direction is now opposite to what plan B
claimed.

## Discipline log (the big one)

Three layered red-team findings, each upgrading or overturning the
previous:

1. Plan A red-team upgraded plan A's "K-K is uncorrelated with Q-K"
   to "K-K is 2× worse than random in single-shot L2."
2. Plan B's "trajectory dynamics" hypothesis was upgraded by drop-one
   single-shot (still wins) and trajectory simulation (still wins by
   6.8×) to "must be Q-drift secondary or N=5 noise."
3. **Plan B red-team R-B4 discovered the harness was running on
   gibberish prompts.** All prior territory-layer findings need
   re-verification on natural language. The 20-prompt rerun shows
   qsigdist trending POSITIVE vs random, retracting plan B.

**Lesson:** when a measurement contradicts multiple independent
oracle predictions (per-Q-head L2, drop-one single-shot, trajectory
simulation all said qsigdist >> random), the FIRST suspicion should
be that the measurement is on the wrong inputs. I spent multiple
journals constructing increasingly-elaborate mechanism hypotheses
(trajectory dynamics, correlated drift, sequential-decision
asymmetry) to explain a contradiction that was actually caused by
running the test on gibberish.

This pattern is severe enough to deserve its own memory entry:
**when measurement contradicts multiple oracles, validate the input
before constructing mechanism.** Saving as
`feedback_validate_input_before_mechanism`.

## Files

- `experiments/phase_zeta/redteam_b.py` — R-B1 (sparsity) + R-B2
  (drop-one single-shot).
- `experiments/phase_zeta/redteam_b_trajectory.py` — R-B3 (cumulative
  trajectory simulation).
- `experiments/phase_zeta/tokenize_prompts.py` — generates 20
  natural-language prompts under correct tokenizer.
- `experiments/phase_zeta/redteam_b_harness.py` — 20-prompt natural-
  language battery at window=16.
- `experiments/phase_zeta/results/redteam_b_harness/` — per-trial
  logs + battery_results.json.

## Sign-off

The substrate-claim arc's territory verdict is no longer cleanly
negative. With proper natural-language prompts at N=20, qsigdist
trends positive against random (+6pp) but is not statistically
significant. The next testable claim is "does qsigdist's positive
trend reach significance at N=50-100?" — which would convert the
arc's territory-layer outcome from "inconclusive" to a measured
positive.

Plan B's retraction is the headline. Phase ζ's territory verdict is
now: **inconclusive with positive trend on natural language**, not
"clearly negative on gibberish."
