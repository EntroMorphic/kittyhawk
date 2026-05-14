# L1 — clustering prompts by winning policy

**Date:** 2026-05-14
**Companions:** `per_prompt_routing_reopen_2026-05-14.md` (the
reframe that opened L1–L5); `experiments/phase_zeta/cluster_prompts_
by_winner.py`; `experiments/phase_zeta/results/meta_iterate/cluster_
by_winner.json`.

## What L1 tested

Hypothesis: if the per-prompt winning policy clusters meaningfully
by prompt type (`code` prompts favor policy X, `tech` favor Y, etc.),
then a *trivial* one-feature router — category-prefix → most-common
winner in that category — should capture a non-trivial fraction of
the +16.58pp oracle headroom over qsigdist.

If the trivial router fails, we learn: either (a) the signal isn't
at category granularity, or (b) mode-routing throws away too much
within-category variance.

## What the data says

**Information-theoretic structure:**
```
H(winner) marginal           = 3.122 bits
H(winner | category)         = 1.752 bits
Mutual info I(winner; cat)   = 1.370 bits
Normalized                   = 0.439
```

Category prefix is informative. Knowing whether a prompt is `tech`,
`code`, `dialog`, etc., resolves 43.9% of the entropy in the winning
policy. That's real signal.

**Trivial router Δ:**
```
Router (route to category-mode winner)   +6.79pp
qsigdist Δ                                +6.38pp
Oracle Δ                                 +22.96pp
Headroom captured                          2.5%
Router − qsigdist: +0.42pp  CI [-3.25, +4.13]  (not significant)
```

Mode-on-category-prefix barely beats qsigdist. 95% CI spans zero.
The +16.58pp oracle headroom is reachable in principle but not with
the simplest tool.

## Where the router wins and loses

**Wins where mode-routing helps (specialization survives at category
level):**
```
tech (n=5)        meta(1, 0, 1)    +19.2pp   vs qsig -1.7pp   net  +20.9pp
biology (n=1)     meta(1, 1, 1)    +41.7pp   vs qsig +20.8pp  net  +20.9pp
conditional (n=1) meta(-1, 1, 1)   +45.8pp   vs qsig  +0.0pp  net  +45.8pp
negation (n=1)    meta(-1, 0, 1)   +20.8pp   vs qsig  -8.3pp  net  +29.1pp
hypothesis (n=1)  sigdist          +29.2pp   vs qsig +25.0pp  net   +4.2pp
```

**Losses where mode-routing hurts (within-category variance too
high):**
```
def (n=6)         fifo              -6.9pp   vs qsig +9.0pp   net -15.9pp
long (n=9)        fifo              -7.9pp   vs qsig +8.8pp   net -16.7pp
technical (n=3)   fifo             -12.5pp   vs qsig -6.9pp   net  -5.6pp
poetry (n=4)      qsigdist         -10.4pp   vs qsig -10.4pp  net   0.0pp
```

Wins and losses approximately cancel — the +0.42pp net effect is
not significant.

## What this means

The signal is real (44% MI) but the strategy is too coarse to
extract it. Two distinct failure modes:

**Failure mode 1: small categories (n=1 to n=3).** Categories with
only 1 prompt make the mode equal to whatever won that single
prompt. The wins are real on the seen prompt but generalization
beyond n=1 is unverified. Some of the +20pp gains may not survive
held-out validation.

**Failure mode 2: high within-category variance (`code` H=2.66,
`long` H=2.50, `def` H=2.25).** In these categories, the mode is
barely the mode — `code` has 11 prompts split across 8 different
winners, and `qsigdist` only wins 3/11. Routing all 11 to qsigdist
captures fewer wins than the oracle but still loses to the policies
that won the other 8. Routing all 11 to fifo would lose even more.

The trivial router maximizes the mean-mode hit rate but doesn't
capture the per-prompt structure because the most common winner in
high-variance categories is barely a plurality.

## Per-category conditional entropy distribution

```
H(winner|cat) ≤ 1.0 bit       (decided): reasoning, geography, history
                                          poetry, quantifier, color
H(winner|cat) ≈ 1.5-2.0 bits  (moderate): cont, dialog, error, idiom,
                                          instr, instruct, logic, math,
                                          temporal, hypothesis
H(winner|cat) ≥ 2.0 bits      (contested): code, def, long, q, tech
```

The "decided" categories ARE good candidates for selective routing.
Reasoning at H=0.92 with mode qsigdist on 2/3 prompts is a
reasonable safe bet. Poetry at H=1.0 with mode qsigdist on 2/4 is
borderline. The "contested" categories should DEFER to qsigdist (the
overall best single policy) and accept the per-prompt losses.

A "selective router" — route to category-mode only when both (a)
the within-category margin is meaningful, AND (b) the routed policy
significantly differs from qsigdist on that category — might
capture more headroom than the uniform mode-router.

## What we learned

1. **The +16.58pp oracle ceiling is real but the trivial router
   doesn't reach it.** Cheap routing on a single coarse feature
   captures 2.5% of available headroom.

2. **Categories carry 44% of winner information.** Adding finer
   features (token presence, prompt length, etc.) could push this
   higher — but each feature has its own granularity vs sample-size
   tradeoff.

3. **Within-category variance is the dominant cost.** Categories
   with H(winner|cat) ≥ 2.0 bits cost the router as much as the
   "decided" categories save it. A selective strategy that defers
   to qsigdist for contested categories should net positive.

4. **Some category-wins are real specialization** (tech → meta(1,0,1)
   +20.9pp, n=5 with margin 2/5). Others are n=1 lucky picks. Need
   held-out CV to distinguish.

## What L2 should test

A "selective router" with these rules:
1. If category H(winner|cat) ≤ threshold (e.g., 1.5 bits) AND
   mode-policy mean Δ on category > qsigdist mean Δ + margin,
   route to mode-policy.
2. Otherwise, route to qsigdist.

This is still 1-feature (category) but adds a confidence gate.
Plus: try other features — prompt length quartile, digit presence,
parenthesis presence (code marker) — as alternative or supplemental
discriminators. Each should be evaluated against the qsigdist
baseline AND the oracle ceiling.

## Files

- `experiments/phase_zeta/cluster_prompts_by_winner.py` — L1 analysis
- `experiments/phase_zeta/results/meta_iterate/cluster_by_winner.json` —
  full per-category breakdown
