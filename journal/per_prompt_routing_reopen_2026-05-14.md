# Per-prompt routing — the arc isn't closed

**Date:** 2026-05-14
**Companions:** `meta_routing_arc_closeout_2026-05-14.md` (the
closeout I just shipped that this entry partially overturns);
`meta_routing_arc_lmm_2026-05-14.md`; `experiments/phase_zeta/
per_prompt_routing.py`.

## How this reopened

Right after the closeout was committed and pushed, Tripp asked:

> I understand that qsigdist is the best so far. But, would any of
> the other ideas we have explored enhance it vs compete with it?

And:

> entropy is structure

That reframe exposed a problem with the closeout's empirical
verdict. The arc tested "find a layer-2 program with higher *mean
Δ* than qsigdist" and refuted it. It never tested "is qsigdist the
best policy on every prompt, or only on average?"

The answer to the second question, on existing data with no new
harness runs, is **decisively no**.

## What the data says

Per-prompt match-rate analysis on the 100-prompt battery, across
all 4 fixed modes + 9 meta-mode anchors (12 policies total,
excluding `random` as the baseline):

**Oracle Δ (pick the per-prompt winner): +22.96pp**
qsigdist Δ: +6.38pp
**Headroom: +16.58pp, 95% CI [+12.83, +20.42]** (bootstrap n=10000)

The headroom is bigger than qsigdist's entire gain over random.

## Win distribution

```
qsigdist          25%  (25 prompts)
fifo              20%  (20)
meta(1, 1, 1)     10%
meta(0, 1, 1)      8%
meta(1, 0, 1)      7%
sigdist            7%
meta(-1, 1, 1)     7%
meta(-1, 0, 1)     6%
meta(1, -1, 1)     4%
meta(0, -1, 0)     4%   ← death-zone policy wins on 4 prompts
meta(1, -1, 0)     2%   ← death-zone policy wins on 2 prompts
```

Entropy of win distribution: **3.122 bits / 3.459 bits max =
0.902 normalized.** Near-uniform spread across 11 policies. The
death-zone policies, which had catastrophic mean Δ (−13.7 and
−13.9pp), are not bad in general — they win on 4–6% of prompts.
Their mean collapsed because they fail on most prompts and succeed
on a few specialized ones.

The closeout journal said "no untested cell predicted above
qsigdist's +6.4pp ceiling, empirical claim refuted." That sentence
is true at the *mean-Δ* level. It is false at the *per-prompt*
level — the 27-cell family DOES contain a champion, but the
champion is a *router over the family*, not any single member.

## Where the headroom lives

Top-10 prompts by oracle gap over qsigdist:

```
tech_neural          +66.7pp   meta(1, 0, 1)
tech_quantum         +66.7pp   meta(0, 1, 1)
tech_protein         +62.5pp   meta(-1, 0, 1)
code_sql             +58.3pp   sigdist
logic_causal2        +58.3pp   fifo
long_storm           +58.3pp   meta(1, -1, 1)
dialog_greet         +58.3pp   meta(1, 1, 1)
idiom_horse          +54.2pp   meta(1, 0, 1)
long_market          +54.2pp   meta(0, 1, 1)
def_metaphor         +50.0pp   meta(1, 1, 1)
```

These are 14/24 to 16/24 tokens of match-rate difference between
qsigdist and the per-prompt winner. Not noise. There is an apparent
pattern (technical prompts favor different cells from dialog or
long-form), but I haven't characterized it formally.

## What this changes about the architecture verdict

The kernel + structural-wall predictor was tested under the wrong
target. It was predicting *mean Δ* per cell. The right target is
*per-prompt winner* — which cell maximizes match-rate for a given
prompt. The Layer 3 architecture itself is unchanged; the loss
signal it should be predicting is different.

This doesn't invalidate the closeout's architectural finding
(structural separation works as a predictor-honesty pattern). It
does invalidate the closeout's empirical finding (qsigdist is
the family ceiling). The right empirical claim is:

- qsigdist is the best *single* policy in the family at +6.4pp.
- A per-prompt router over the family has +22.96pp headroom.
- The realizable router's Δ depends on how predictable the
  per-prompt winner is from observable features — an open
  question.

## What we don't know yet

1. **Is the winner predictable?** The oracle cheats. A real router
   needs prompt features → winning policy. Three things to check
   first, all read-only on existing data:
   - Cluster prompts by winning policy; look for coherent groups.
   - Test simplest possible routers (e.g., "code → sigdist, else
     qsigdist") and measure achieved Δ vs oracle ceiling.
   - Estimate the predictability ceiling — what fraction of the
     +16.58pp can a "first-N-token" feature-based router capture?

2. **Is the +22.96pp oracle robust?** Each per-prompt Δ has
   measurement granularity 1/24 ≈ 4.2pp per token. The oracle
   picks max over 11 policies, which is selection-on-max — it
   inflates the apparent ceiling because it picks the lucky
   policy. Bootstrap CI is on the *mean* of per-prompt picks, so
   it's not entirely fake, but a held-out cross-validation
   (oracle trained on 50, tested on 50) would be tighter. Worth
   running.

3. **Does the death-zone policies' specialization survive
   re-measurement?** With per-prompt N=1 (one generation per
   prompt per policy), the policy that won a given prompt may
   have done so by 1-token noise. Multi-seed measurement of the
   top-headroom prompts would distinguish "this policy genuinely
   specialized" from "this policy happened to match by luck."

## What this changes about the arc closeout

The closeout journal and LMM cycle were shipped before this
analysis. The closeout's section "Verdicts" claims:

> B (empirical — Layer 3 finds a policy beating qsigdist):
> ✗ refuted

That verdict needs a footnote:

> Refuted at the mean-Δ level (no single policy in the 27-cell
> linear-score family beats qsigdist). NOT refuted at the
> per-prompt level (an oracle picking the per-prompt winner gains
> +16.58pp over qsigdist, 95% CI [+12.83, +20.42]). The arc tested
> the wrong target; the right empirical question is whether a
> learnable router can capture a fraction of the per-prompt
> headroom.

The architectural lessons (structural separation, in-sample MAE
overfit trap, death-zone observation) stand. The empirical
conclusion stands with the qualifier above.

## Files

- `experiments/phase_zeta/per_prompt_routing.py` — analysis
- `experiments/phase_zeta/results/meta_iterate/per_prompt_routing.json`
  — full rates matrix + per-prompt winners + bootstrap CI

## What was Tripp's read-of-the-arc that I missed

I had been treating mean Δ as if it were the only ground truth. He
asked one question — "enhance vs compete" — that turned the entire
analysis. The reframe wasn't a different model or a different
metric; it was a different *question*. The architecture I built can
answer it; I just hadn't pointed it at the right target.

"Entropy is structure" was the operationalization — if all 11
policies have non-trivial win shares, the variance across policies
IS the signal that routing matters. The entropy of the win
distribution at 0.902 of maximum is the most concise statement of
"there is real per-prompt structure here, go find it."
