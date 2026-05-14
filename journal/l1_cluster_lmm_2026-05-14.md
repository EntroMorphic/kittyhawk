---
cycle: L1 cluster-by-winner LMM
phase: ALL (raw + nodes + reflect + synthesize)
date: 2026-05-14
scope: synthesize the L1 cluster-by-winner finding (category MI=0.44
       of winner; trivial mode-router achieves +0.42pp over qsigdist
       with CI spanning 0; big specialized wins offset by big
       high-variance losses) into a coherent picture and a principled
       next step. The per-prompt routing reopen found +16.58pp oracle
       headroom; L1 just measured how much of it the cheapest router
       captures. The answer is 2.5%. What does this MEAN, and what
       should we do?
companions: l1_cluster_by_winner_2026-05-14.md (factual record);
            per_prompt_routing_reopen_2026-05-14.md (the reframe that
            opened L1); per_prompt_routing_lmm_2026-05-14.md (LMM on
            the reframe).
---

# L1 cluster-by-winner — LMM cycle

## RAW

What I actually think, unfiltered:

The 44% mutual information is a genuinely positive finding. The 2.5%
headroom-captured is a genuinely negative finding. Both are true.
The trap I'm watching out for in real-time is using one to dismiss
the other — i.e., declaring "the routing arc is hopeless because
the trivial router doesn't capture much" OR declaring "the routing
arc is succeeding because the MI is high." Neither alone is
honest. The honest read is: signal exists at category granularity
but mode-routing is the wrong extraction tool.

What surprised me: the specialized wins are LARGER than I expected.
tech → meta(1,0,1) at +20.9pp net over qsigdist is a real,
substantive specialization on a 5-prompt category. biology at
+20.9pp is n=1 so I can't trust it but it's directionally
consistent. conditional at +45.8pp is n=1 but it's huge. If even a
few of these survive held-out CV, the routing arc has a concrete
forward path.

What concerns me: the losses are also large. long → fifo at −16.7pp
on a 9-prompt category is the kind of damage that wipes out 4
specialized wins. The router's net +0.42pp is the SUM of large
opposing signed contributions, not a small uniform improvement.
This is a fragile equilibrium — slightly different category-mode
choices (or a slightly different threshold for "use mode vs
default") would change the net delta by several pp.

What I missed in the previous LMM: the L1–L4 plan assumed L1's
result would be informative either way. It is — but I had been
imagining "L1 succeeds → build router on it; L1 fails → try L2/L4."
The actual result is "L1 reveals signal exists, but the simplest
extraction loses it." That's a third possibility — and it implies
L2 should be more principled (selective routing with confidence
gates) rather than just trying more features at the same
granularity.

What I'm avoiding looking at:

- The CI on (router − qsigdist) is [-3.25, +4.13]. That's a 7.4pp
  width on a 100-prompt sample. A reasonable cross-validation oracle
  would have similar uncertainty. Even if the realizable router
  captures 30-40% of the +16.58pp headroom, our CI on the
  measurement is wide enough that small effect sizes will be hard to
  distinguish from noise.

- The N=1 categories. 8 of 29 categories have only one prompt. For
  each, the "winner" IS the only choice; the mode is meaningless.
  Including them in the trivial router inflated apparent win-count
  for niche policies (negation → meta(-1,0,1), conditional →
  meta(-1,1,1), etc.). Some of these may evaporate under held-out
  CV; others may not. We don't know which.

- The categories with H(winner|cat) ≥ 2 bits (code, long, def, tech,
  q): five categories totaling 39 prompts (39% of the battery).
  These are where the router COULD have the most leverage if a
  finer-grained feature distinguishes within them, AND where the
  trivial router has the most damage potential. Both effects compound
  on the same population.

- The N=100 measurement floor. Each per-prompt match-rate is over 24
  gen tokens, so granularity is ~4.2pp per token. A "win" by 1 token
  is inside measurement noise. The L1 oracle and router calculations
  treat 1-token wins as wins, which inflates apparent variance and
  may make the +16.58pp oracle headroom an overestimate. Multi-seed
  measurement (L3) would tighten this.

## NODES

Extracted tensions and constraints:

**N1 — MI is real; extraction fails.** The information exists at
category granularity (44% of winner uncertainty resolved). The
trivial extraction strategy (mode-routing) captures almost none of
the headroom. These are different facts. Conflating them either
direction is wrong.

**N2 — Wins and losses approximately cancel.** Net +0.42pp router
gain is the sum of large opposing signed contributions:
  - Wins from tech/biology/conditional/negation ≈ +30pp aggregate.
  - Losses from long/def/technical ≈ -30pp aggregate.
This is a knife-edge equilibrium, not a small improvement.

**N3 — N=1 categories distort the analysis.** 8 of 29 categories
have only one prompt. Their "wins" are real on the seen prompt but
not reproducible. They contribute to both the win-count distribution
and the achieved-router-Δ, and we can't tell which side they're on.

**N4 — High within-category variance is the dominant failure mode.**
Categories with H(winner|cat) ≥ 2 bits cover 39% of prompts and are
where mode-routing loses worst. The signal in these categories is
at finer granularity than category prefix.

**N5 — Specialization signal is large where it exists.** When mode-
routing wins, it wins by 20-45pp on a category. That's not noise.
The architectural ARM that ENABLES specialized signal extraction
(once we figure out how to extract it) is genuinely valuable.

**N6 — Measurement noise floor is ~4pp per prompt.** N=24 gen
tokens means single-token differences are 4.2pp of match-rate. Many
"wins" may be 1-token noise. L3 (multi-seed) would distinguish.

**N7 — Confidence gates matter.** A selective router (route only
when confident) probably nets positive where the trivial uniform-
mode-router nets ~0. The data tells us where to gate: low
H(winner|cat) AND meaningful margin over qsigdist.

**N8 — Held-out CV is still owed.** The L1 analysis trained and
tested on the same 100 prompts. Realized router Δ on held-out
prompts will be lower. We don't have the realistic ceiling.

## REFLECT

Structure, assumptions to challenge, leverage points:

**Structure.** Three distinct strata of finding:

1. **Information-theoretic** (does category contain info about
   winner?): Yes, 44% of winner uncertainty resolved.

2. **Realizable-extraction** (does the simplest strategy capture
   that info?): No, 2.5% of headroom captured.

3. **Diagnostic** (what's the obstruction?): Within-category
   variance in high-H categories (39% of prompts); n=1 categories
   inflating apparent win-counts.

Strata 1 and 3 are positive directions for L2; stratum 2 is the
negative result we just learned.

**Assumptions to challenge:**

1. **"Mode-routing is the right baseline for L1."** Maybe not.
   Mode-routing is the SIMPLEST 1-feature strategy. A more
   informed baseline would be: route to mode WHERE
   H(winner|category) ≤ threshold, default to qsigdist
   elsewhere. That's a confidence-gated 1-feature router. Likely
   nets positive where uniform-mode is ~0.

2. **"Category prefix is the right feature."** Maybe. But token-
   level features (presence of digits, code markers, question
   marks) might be both cheaper-to-compute AND finer-grained.
   The MI for "has digit" or "has paren" might be different from
   "category prefix" MI and possibly higher.

3. **"The oracle ceiling +22.96pp is the realistic upper bound."**
   It isn't. The oracle selects on max over 11 policies per prompt,
   so it inflates by selection bias. A held-out oracle is ~60-80%
   of this, say +14-18pp over random or +8-12pp over qsigdist.

4. **"N=1 category wins should be excluded as noise."** Maybe.
   Some n=1 wins are noise; others are real specialization that
   would replicate. Without held-out validation we can't tell.
   The safer interim move is to weight category contributions by
   their sample size in the router decision.

5. **"L2 should add features."** Maybe — but L2 should ALSO try a
   confidence-gated version of L1 first. The same single feature
   used selectively might extract significantly more than used
   uniformly.

**Leverage points:**

L1.1 (NEW). **Confidence-gated category router.** Same data, same
feature, different strategy: route only when (a)
H(winner|cat) ≤ τ and (b) mode-Δ_cat − qsig-Δ_cat ≥ δ. Sweep τ
and δ. Should net positive where the uniform mode-router nets ~0.
Cost: ~10 min (same script, add gating).

L1.2 (NEW). **Held-out cross-validation of L1's wins.** Split 100
into 50/50 sets. Train (compute category modes) on first 50, test
on second 50. Repeat several times. Realized router Δ on held-out
sets is the realistic estimate; if it stays at ~0pp, the apparent
+0.42pp was lucky; if it stays at ~+0.4pp on a clean split, it's
modest but real. Cost: ~15 min.

L2.1. **Token-level features instead of category prefix.** Run
the same router-strategy analysis with "has digit", "has paren",
"prompt length ≥ X tokens", "has uppercase", "ends with ?", etc.
Compare MI to category MI. The features with higher MI are better
candidates for a real router. Cost: ~30 min.

L3 (deferred). **Multi-seed re-measurement.** Run top-10
oracle-headroom prompts with 3 RNG seeds × all 11 policies.
Distinguish genuine specialization from 1-token-noise wins. Costs
1-2 hours of harness; worth doing only after L1.1 and L1.2.

L4 (still owed). **Held-out oracle CV.** Same as L1.2 but
estimating the realistic oracle ceiling rather than the realistic
router Δ. Cost: ~15 min.

## SYNTHESIZE

Concrete actionable output:

### What L1 actually delivered

1. **Information signal confirmed**: I(winner; category) = 1.37 bits
   = 44% of winner-uncertainty resolved by category alone.

2. **Trivial extraction strategy fails**: mode-on-category-prefix
   captures only 2.5% of the +16.58pp oracle headroom; CI on
   (router − qsigdist) spans 0.

3. **Diagnostic: failure modes identified**:
   - 39% of prompts in high-H(winner|cat) categories where mode is
     barely the mode.
   - 8/29 categories with n=1 inflate apparent win-counts.
   - Specialization is real where it exists (tech +20.9pp, n=5,
     margin 2/5) but coexists with damage where within-cat
     variance is high (long −16.7pp, def −15.9pp).

### Forward plan (priority order)

1. **L1.1 — confidence-gated category router.** Same feature,
   smarter strategy: route only when within-category confidence is
   high AND mode-policy differs from qsigdist by a meaningful
   margin. Sweep the two thresholds. ~10 min.

2. **L1.2 — held-out CV of category routing.** Split 50/50; train
   mode on first 50, evaluate on second 50; repeat. Gives a
   realistic estimate of category-routing Δ. ~15 min.

3. **L2.1 — token-level feature MI comparison.** Compute MI for
   "has digit", "has paren", "prompt length quartile", "has
   uppercase", "ends with ?" with the winner. Identify which
   features (alone or paired) carry more info than category prefix.
   ~30 min.

4. **L4 — held-out oracle CV.** ~15 min. Should run alongside
   L1.2 (same split mechanism).

5. **L3 (deferred)** — multi-seed re-measurement. Only after L1.1
   + L1.2 + L4 give a realistic picture. Cost: 1-2 hours of
   harness.

### What NOT to do

- Conclude from L1's failure that the routing arc is hopeless. The
  44% MI is real signal; we just need a better extractor.
- Add features (L2) before testing the smarter strategy on the
  existing feature (L1.1). Single feature + confidence gate is
  cheaper and may suffice.
- Train a complicated classifier before establishing what simple
  thresholded routing can achieve.
- Trust the +22.96pp oracle as the realistic ceiling. It's a
  selection-on-max upper bound. The realistic ceiling is what L4
  measures via held-out CV.
- Skip multi-seed (L3). Some apparent specializations are 1-token
  noise; we need the discipline before quoting per-prompt wins.

### Honest framing for forward communication

L1 found that prompt category contains 44% of the information
needed to predict the per-prompt winning policy — non-trivial signal.
But the simplest extraction strategy (route to category-mode winner)
captures only 2.5% of the +16.58pp oracle headroom; CI on (router −
qsigdist) spans 0. The failure mode is well-characterized: high
within-category winner-variance in 39% of prompts wipes out the
specialization gains in low-variance categories.

The next test is whether a confidence-gated version of the same
1-feature strategy nets meaningfully positive (L1.1), and whether
the apparent gains survive held-out CV (L1.2). If L1.1 captures
say 20-30% of the headroom on held-out data, the routing arc has
a concrete forward path. If it stays near 0pp, the signal exists
at finer-than-category granularity and L2 (token-level features)
becomes the right move.

## Status

LMM cycle complete. L1's mixed verdict — strong information-theoretic
signal, weak realizable extraction — has been parsed without
overclaiming the positive direction or underclaiming the negative.
Forward plan distinguishes between (a) smarter strategy on existing
feature, (b) held-out validation, (c) more features only if needed,
(d) multi-seed measurement only after the basic picture clarifies.
