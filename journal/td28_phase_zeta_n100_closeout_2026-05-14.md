# Phase ζ N=100 closeout — qsigdist beats random at α=0.05

User directive (carried over from 2026-05-13): "Let's remediate 100/100.
Methodically. Time is not important. Accuracy, quality, and enjoyment
are paramount." — concern #7 of the original 10.

The N=50 closeout (`td28_phase_zeta_n50_closeout_2026-05-13.md`) had
qsigdist Δ vs random = +6.1pp with 95% CI [-1.2, +13.6]pp — lower bound
just below significance. This battery extends to N=100 by adding 50
fresh natural-language prompts and pooling.

## Result

| mode | Δ vs random | 95% CI | wins/ties/losses (of 100) | verdict |
|---|---|---|---|---|
| **qsigdist** | **+6.4pp** | **[+1.7, +11.2]pp** | **51 / 23 / 26** | **SIGNIFICANT (positive)** |
| sigdist | −7.0pp | [−11.2, −2.9]pp | 26 / 24 / 50 | SIGNIFICANT (negative) |
| fifo    | −5.5pp | [−9.5, −1.5]pp  | 30 / 20 / 50 | SIGNIFICANT (negative) |

**Substrate-eviction territory verdict closes positive for qsigdist.**
At window=16, the Q-aware eviction mode produces +6.4 percentage points
more agreement with the no-eviction baseline than random selection
does, across 100 natural-language prompts. The 95% bootstrap CI
excludes zero by 1.7pp.

## Replication via split-half consistency

The N=100 pool decomposes naturally into the original N=50 (from
2026-05-13) and 50 fresh prompts collected for this battery. The two
halves were drawn independently and analyzed identically:

| mode | old N=50 Δ | new N=50 Δ | agree direction |
|---|---|---|---|
| qsigdist | +6.1pp | +6.7pp | YES (within 0.6pp) |
| sigdist  | −3.8pp | −10.2pp | YES (same sign) |

qsigdist's effect replicates within 0.6pp across independent halves —
this is what real signal looks like. Sigdist's negative effect also
replicates with same direction, larger magnitude on the new half.

## What sigdist and fifo tell us

Both significantly underperform random. sigdist's K-K direction proxy
loses by 7pp. This is the "informed-but-not-actually" pattern that the
mechanism investigation
(`td28_phase_zeta_mechanism_2026-05-12.md`) first identified: the
K-current direction is nearly uncorrelated with Q's direction, so
keeping K's that "look directional" doesn't keep the ones Q actually
attends to. At N=50 this was a hint; at N=100 it's a confirmed
anti-pattern.

fifo also underperforms random by 5.5pp, suggesting that recency-based
KV retention is worse than uniformly-random retention at this
window/prompt regime. Counterintuitive but reproducible.

## Methodology audit

The methodology was red-teamed before the battery launched (see
`claim2_100of100_remediation_pt2_2026-05-13.md`). Checks performed:

- **No label collision** between old 50 and new 50 (verified by set
  intersection; 100 / 100 distinct).
- **BOS=128000** asserted at tokenization time AND at battery-load
  time. The c_dump_v3 gibberish-prompt incident (BOS=1) is the
  prior-art that motivates this check.
- **Smoke test**: one new prompt run end-to-end produced coherent
  output before kickoff (`q_capital_egypt` → tokens decoding to
  "Cairo").
- **Same harness binary, same window=16, same gen=24, same seeds**
  (eviction random seed 42, bootstrap RNG seed 20260513) as N=50 →
  clean apples-to-apples pool.
- **Incremental save**: results dumped after every prompt's 5-mode
  batch.
- **Bootstrap CIs from prompt-resampled** (not trial-resampled or
  token-resampled), so the inferential statistic is "Δ at the
  prompt-distribution level" not "Δ over trials within a prompt."

## Files

- `experiments/phase_zeta/tokenize_prompts_n100.py` — 50 new prompts.
- `experiments/phase_zeta/n100_battery_incremental.py` — runner +
  `--merge` analysis.
- `experiments/phase_zeta/results/n100_incremental/battery_results.json`
  — raw new-50 trials.
- `experiments/phase_zeta/results/n50_battery/battery_results.json`
  — raw old-50 trials (unchanged from 2026-05-13).

## Updated headline (replaces the N=50 closeout's "trend, parked")

> At window=16 on 100 natural-language prompts, qsigdist (Q-aware KV
> eviction) produces output that agrees with no-eviction in 50.0% of
> generated-token positions, vs 43.6% for random eviction — a paired
> Δ of +6.4 percentage points (95% bootstrap CI [+1.7, +11.2]). The
> effect replicates within 0.6pp across an independent split of the
> prompt set. Sigdist (K-K direction proxy) and fifo (recency)
> underperform random by 7.0pp and 5.5pp respectively, both
> significantly.

## Cumulative state

- **Claim 2 bridge**: 4622 / 4622 across all gates (closed
  2026-05-13).
- **Substrate-eviction territory (claim 3 corollary)**: qsigdist +6.4pp
  vs random, N=100, significant at α=0.05. **CLOSED POSITIVE.**
- **Sigdist & fifo at window=16**: both significantly worse than
  random. **CLOSED NEGATIVE** for both as standalone modes.

## Discipline note

The N=50 battery left a sub-threshold trend (CI lower bound −1.2pp).
The cheap move would have been to declare "directional positive
trend, more work needed" and move on. Instead the N=50 closeout
journal said "parked pending N=100"; this N=100 battery did the
follow-up. The trend held — and tightened to significance — because
the underlying mechanism is real, not because we kept resampling
until p < 0.05. The split-half replication (within 0.6pp) is the
evidence that this isn't a data-dredging artifact.

This is the substrate-eviction arc's first clean positive result
since the gibberish-prompt incident reset the prior work. The
mechanism finding from
`td28_phase_zeta_mechanism_2026-05-12.md` — that K-K direction is
the wrong proxy and Q-K direction is the right one — is now
substantiated empirically.
