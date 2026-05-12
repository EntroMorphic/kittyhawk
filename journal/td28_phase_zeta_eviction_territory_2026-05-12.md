# Phase ζ — substrate eviction in the territory: NEGATIVE RESULT

User directive: "We do NOT hide behind documentation. We do the work.
We want the territory, not the map. Remediate 100/100."

The cold-eye audit (§3) noted that the substrate-claim arc had been
measuring properties of an **opt-in** mode that the production default
never touches. Phase ε's headline ("L1-substrate eviction reduces
attention-output L2 error by 38-62% vs Hamming-substrate eviction")
was per-Q-head per-position oracle math, not end-to-end generation.

Phase ζ runs the territory test: enable `BITNET_KV_EVICT_MODE=sigdist`
in the production harness, generate tokens, compare against no-eviction
and against random-eviction (the null baseline). 5 prompts × 4 modes ×
3 window sizes × 24 generated tokens = 60 harness runs.

## Headline result

**Substrate-distance eviction does NOT beat random eviction on
generation quality at any tested window size.** At moderate pressure
(window=16) it is **strictly worse** on aggregate match-rate
(78.3% vs 82.5% — see CI note below) and **strictly worse** on two of
five prompts (medium_11, bos_only).

The Phase ε per-Q-head L2 advantage did not translate to end-to-end
generation quality. The arc's strongest surviving claim shrinks.

## Configuration

- Harness: `build/gesh/bitnet_harness` against `data/bitnet_b158_2b4t.bin`.
- 5 prompts (lengths 0–11 tokens), 24 generated tokens each, greedy.
- 4 modes: `no_evict` (production default, baseline), `fifo`
  (drop-oldest null), `random` (random-drop null with fixed seed=42),
  `sigdist` (substrate L1-distance eviction, the claim under test).
- 3 windows: 8, 16, 32. Total context ≤ 35 tokens, so window=32 has
  near-zero eviction pressure (sanity check).
- `BITNET_ATTN_FIXED_TAU=5000` for sigdist (matches Phase ε's regime).

## Results (mean across 5 prompts, vs no-eviction baseline)

### Window=8 (aggressive eviction)

| mode    | mean first-divergence | mean match-rate | mean distinct |
|---------|----------------------:|----------------:|--------------:|
| fifo    |  5.2                  | 22.5%           | 13.0          |
| random  |  6.0                  | **26.7%**       | **18.2**      |
| sigdist |  5.2                  | 26.7%           | 13.8          |

Base distinct = 17.4. Random ties sigdist on match-rate, **beats**
sigdist on distinct-count by 32% relative.

### Window=16 (moderate eviction)

| mode    | mean first-divergence | mean match-rate | mean distinct |
|---------|----------------------:|----------------:|--------------:|
| fifo    | 16.8                  | 81.7%           | 16.8          |
| random  | **18.8**              | **84.2%**       | **17.8**      |
| sigdist | 18.4                  | 78.3%           | 17.0          |

Random **beats sigdist on every metric**. fifo also beats sigdist on
match-rate.

### Window=32 (negligible eviction; total context ≤ 35)

| mode    | mean first-divergence | mean match-rate | mean distinct |
|---------|----------------------:|----------------:|--------------:|
| fifo    | 23.8                  |  99.2%          | 17.6          |
| random  | 24.0                  | 100.0%          | 17.4          |
| sigdist | 24.0                  | 100.0%          | 17.4          |

Sanity check: with the window above the total context length, all
policies are equivalent to no-eviction.

## Per-prompt result at window=16 (where the negative result is sharpest)

| prompt          | fifo div / match% | random div / match% | sigdist div / match% |
|-----------------|------------------:|--------------------:|---------------------:|
| capital_france  | 24 / 100.0%       | 24 / 100.0%         | 24 / 100.0%          |
| short_a         | 14 /  58.3%       | 14 /  58.3%         | 14 /  58.3%          |
| short_b         | 18 /  75.0%       | 18 /  75.0%         | 18 /  75.0%          |
| medium_11       |  8 /  91.7%       | 14 /  87.5%         | 18 /  **79.2%**      |
| bos_only        | 20 /  83.3%       | 24 / **100.0%**     | 18 /  **79.2%**      |

Sigdist is the **worst** mode on `medium_11` and `bos_only` (tied for
worst on bos_only with itself only — random achieves perfect match
where sigdist drops to 79.2%).

## What this contradicts and what it preserves

**Contradicts (Phase ε's load-bearing framing):**

- Phase ε concluded: "L1-substrate eviction reduces attention-output
  L2 error by 38-62% relative... this is the strongest proxy for
  generation quality without running the harness end-to-end."
- Phase ζ runs the harness end-to-end and finds: **substrate eviction
  is no better than random**, and worse on the most-stressed prompts.
- The per-Q-head L2 advantage is real (Phase ε's math is not in
  dispute), but it **does not propagate** through layer composition
  and autoregressive decoding to better generated tokens. Either
  the local advantage is too small relative to downstream noise, or
  random eviction happens to drop the "right" tokens often enough on
  these prompts that the supposed advantage doesn't show up.

**Preserves:**

- The L1 metric itself is correctly implemented in production
  (`m4t_popcount_dist`; td28 finding).
- The Phase ε oracle measurement (attention-output L2 with vs without
  L1 selection) is mathematically valid; it just does not predict
  end-to-end generation outcomes on this benchmark.
- The substrate's intra-substrate utilization claim (L1 carries
  information that Hamming would lose on ternary) is unchanged. What
  fails is the **comparative-advantage claim against the simplest
  alternative eviction policy** (random).

## Honest caveats

1. **5 prompts is small.** This is a small enough N that one prompt
   swings the aggregate by ~5 percentage points. The direction is
   consistent (random ≥ sigdist on 5/5 prompts at window=16 for
   match-rate) but the confidence interval on the aggregate gap is
   wide.

2. **Greedy decoding.** Sampling-based decoding might surface
   different sensitivities. Match-rate against a no-eviction baseline
   becomes ill-defined under sampling, but quality measures like
   perplexity or downstream-task accuracy could be probed.

3. **No perplexity / NLL measurement.** Match-rate against the
   no-evict reference is a strict metric; a token differs as a "miss"
   even if it's near-equally-probable. A softer metric (KL between
   logit distributions, per-position NLL) might show sigdist closer
   to no-evict than this match-rate suggests. The harness currently
   prints `logits[0..3]` but not full per-position logprobs.

4. **Window sizes 8/16 vs production caches.** Real production may
   use much larger caches (1000+ K's). Whether the eviction-policy
   choice matters at all in that regime is unmeasured.

5. **τ regime in sigdist mode.** Phase ζ uses fixed τ=5000 (matches
   sigdist mode's actual production behavior). The cold-eye §2 gap
   (production default has per-Q-adaptive τ) is **not** in play here
   because the production default doesn't use sigdist eviction at all.
   Phase ζ measures sigdist as it would actually be invoked.

## What still isn't tested (and why this isn't fatal to the larger arc)

- **Long-context regimes** (cache > 64). Plausible that at high
  eviction rates over long contexts, substrate-distance selection
  starts to matter. Unmeasured.
- **Sampling decode quality.** Could differ qualitatively.
- **Routed attention** (`BITNET_ATTN_MODE=routed`) — a different
  opt-in mode where substrate signatures gate attention rather than
  evict KV. Phase ζ doesn't touch routed mode.

These are open. But for the eviction territory specifically, the
result is clear and negative.

## What this changes for the arc

**The strongest substrate-claim that survived adversarial scrutiny
through Phase ε no longer survives Phase ζ.** The arc's history:

- Phase α (M1 estimator) — methodology pivot, no claim.
- Phase β (L1 application) — measurement plumbing, no claim.
- Phase γ (robustness battery) — null-control variations, mixed.
- Phase δ (eviction quality, oracle math) — direction holds,
  effect size small.
- Phase ε (eviction quality, six variations) — direction holds
  robustly, effect size 38-62% relative on attn-output L2.
- **Phase ζ (eviction quality, end-to-end harness) — direction
  does not hold; random ≥ sigdist.**

The arc's surviving claim now reads:

> The L1 path-graph metric on ternary signatures is well-defined,
> correctly implemented in production (`m4t_popcount_dist`), and
> measurably exploits sign+magnitude information that Hamming
> collapses. **However**, when used as a KV-eviction selection
> criterion in the BitNet harness, it does not produce better
> generated tokens than random eviction on a 5-prompt battery at
> windows 8 and 16. The Phase ε per-Q-head L2 advantage does not
> propagate through to end-to-end generation in this benchmark.

This is the territory result. The map said L1>Hamming. The territory
says L1≈random on this benchmark, and sometimes worse.

## Discipline log

This is the second time on this arc that a strongly-framed positive
result has reversed when measured one layer closer to production:

1. td28 (2026-05-12 earlier): "Phase β/γ/δ/ε measured L1 vs Hamming
   in Python; production was already L1" → the comparison was
   never against what production does. Reframe.
2. Phase ζ (now): "L1 oracle math beats Hamming by 38-62% on attn-L2;
   should translate to generation quality" → it doesn't. Reframe.

The pattern is consistent with `feedback_verify_production_semantics`
(written 2026-05-12 from td28): **load-bearing claims need to be
measured at the level they're claimed at.** Phase ε's "operationally
relevant" proxy (attention-output L2) was a proxy. The actual
operational test is generated tokens. Don't claim "operationally
meaningful" of a proxy until the actual operational measurement is
done.

Saving this pattern as `feedback_proxy_to_territory_pattern` for
future arcs.

## Files

- `experiments/phase_zeta/eviction_battery.py` — battery driver.
- `experiments/phase_zeta/results/battery_results.json` — raw trials.
- `experiments/phase_zeta/results/*.log` — per-trial harness output.
- This journal.

## Sign-off

The substrate-claim arc has its first end-to-end NEGATIVE RESULT on
the eviction claim. The arc remains valuable as a measurement
discipline exercise — every reversal taught a methodology lesson.
The substrate is not vindicated as a comparative-advantage eviction
metric in the harness territory. **The Phase ε result should be
re-headlined as a property of the oracle, not a prediction about
the territory.**
