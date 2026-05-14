"""L1: cluster prompts by winning policy.

Read-only on existing data. Per-prompt analysis already computed
which of 11 policies wins each of 100 prompts (per_prompt_routing.json).
Question: does the win structure cluster meaningfully by prompt type
(code, tech, dialog, etc.), or are wins essentially random?

If clusters are coherent → a simple feature-based router can capture
much of the +16.58pp headroom over qsigdist.
If wins are scattered uniformly across prompt types → the routing
signal exists at finer-than-prompt-type granularity (or is noise).

Method:
  1. Group the 100 prompt labels into coarse categories by label-prefix.
  2. For each (category, winning-policy) pair, count.
  3. Compute conditional entropy H(winner | category) per category.
  4. Compute total H(winner) marginal — if H(winner|category) is much
     lower than H(winner), categories ARE informative.
  5. Display the (category → most-common-winner) routing rule and
     report its achieved Δ vs the oracle ceiling.

The simplest router this analysis suggests: route on category-prefix
to the within-category mode-winner. Achievable Δ = mean of
per-prompt (winner-within-its-category) match-rate minus random
match-rate.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np

THIS = os.path.dirname(__file__)
ROUTING = os.path.join(THIS, "results/meta_iterate/per_prompt_routing.json")


def category_of(label: str) -> str:
    """Coarse category from label prefix. Many labels are 'kind_subkind'
    or 'kind_id'; first underscore-separated token is the category."""
    # Some labels are bare (e.g., 'comparison', 'negation'); use as-is.
    return label.split("_")[0] if "_" in label else label


def main():
    with open(ROUTING) as f:
        d = json.load(f)

    policy_names = d["policy_names"]
    labels       = d["labels"]
    rates        = np.array(d["rates"])
    winners      = d["winner_per_prompt"]
    oracle_delta = d["oracle_delta_pp"]
    qsig_delta   = d["qsigdist_delta_pp"]

    random_idx = policy_names.index("random")

    # ----- Step 1: group labels by category -----
    by_cat = defaultdict(list)
    for label in labels:
        by_cat[category_of(label)].append(label)
    print(f"{len(by_cat)} categories from {len(labels)} prompts:\n")
    for cat, ls in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:<12} ({len(ls):>2})  {ls[:5]}{'...' if len(ls) > 5 else ''}")

    # ----- Step 2: (category, winner) cross-tabulation -----
    print(f"\nWinning policy by category (winner counts):")
    cat_winners = defaultdict(Counter)
    label_to_winner = dict(zip(labels, winners))
    for cat, ls in by_cat.items():
        for label in ls:
            w = label_to_winner.get(label)
            if w is not None:
                cat_winners[cat][w] += 1

    # ----- Step 3: per-category entropy of winners -----
    print(f"\n{'category':<12} {'n':>3}  {'H(winner|cat)':>14}  {'mode winner (margin)':<40}")
    cat_entropies = {}
    cat_modes = {}
    for cat in sorted(cat_winners.keys()):
        wc = cat_winners[cat]
        n = sum(wc.values())
        probs = np.array(list(wc.values())) / n
        H = -np.sum(probs * np.log2(probs + 1e-12))
        cat_entropies[cat] = H
        sorted_wins = wc.most_common(2)
        mode_w, mode_n = sorted_wins[0]
        margin = mode_n - (sorted_wins[1][1] if len(sorted_wins) > 1 else 0)
        cat_modes[cat] = mode_w
        print(f"  {cat:<12} {n:>3}  {H:>9.2f} bits  {mode_w} ({mode_n}/{n}, margin {margin})")

    # ----- Step 4: marginal entropy comparison -----
    all_winners_flat = [w for w in winners if w is not None]
    marg_counts = Counter(all_winners_flat)
    n_total = sum(marg_counts.values())
    probs_marg = np.array(list(marg_counts.values())) / n_total
    H_marg = -np.sum(probs_marg * np.log2(probs_marg))
    H_cond = sum(len(by_cat[cat]) / len(labels) * cat_entropies[cat]
                 for cat in cat_winners) if cat_winners else 0
    info_gain = H_marg - H_cond
    print(f"\nH(winner)        marginal  = {H_marg:.3f} bits")
    print(f"H(winner | cat)  conditional = {H_cond:.3f} bits")
    print(f"Mutual info I(winner; cat)   = {info_gain:.3f} bits")
    print(f"  Normalized: {info_gain/H_marg:.3f}  "
          f"(0 = cat tells you nothing about winner; 1 = cat determines winner)")

    # ----- Step 5: realized Δ of the category-mode-winner router -----
    # Strategy: route prompt to its category's most-common winner.
    achieved_rates = []
    qsig_rates = []
    rand_rates = rates[random_idx, :]
    for j, label in enumerate(labels):
        cat = category_of(label)
        routed_policy = cat_modes.get(cat)
        if routed_policy is None:
            continue
        p_idx = policy_names.index(routed_policy)
        achieved_rates.append(rates[p_idx, j])
        qsig_rates.append(rates[policy_names.index("qsigdist"), j])

    achieved = np.array(achieved_rates) - rand_rates[:len(achieved_rates)]
    achieved_delta = np.nanmean(achieved) * 100
    print(f"\nRouter strategy: route each prompt to its category's most-common winner.")
    print(f"  Achieved Δ:  {achieved_delta:+.2f}pp")
    print(f"  qsigdist Δ:  {qsig_delta:+.2f}pp")
    print(f"  Oracle Δ:    {oracle_delta:+.2f}pp")
    headroom_captured = (achieved_delta - qsig_delta) / (oracle_delta - qsig_delta)
    print(f"  Headroom captured: {headroom_captured*100:.1f}% of the way from "
          f"qsigdist to oracle")

    # ----- Bootstrap CI on (router Δ − qsigdist Δ) -----
    qsig_idx = policy_names.index("qsigdist")
    qsig_rates_all = rates[qsig_idx, :]
    routed_rates_full = np.full(len(labels), np.nan)
    for j, label in enumerate(labels):
        cat = category_of(label)
        rp = cat_modes.get(cat)
        if rp is None:
            continue
        routed_rates_full[j] = rates[policy_names.index(rp), j]
    diff_per_prompt = (routed_rates_full - qsig_rates_all) * 100
    rng = np.random.default_rng(20260514)
    boots = []
    valid = diff_per_prompt[~np.isnan(diff_per_prompt)]
    for _ in range(10000):
        idx = rng.integers(0, len(valid), size=len(valid))
        boots.append(valid[idx].mean())
    ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])
    print(f"  Router − qsigdist Δ: {valid.mean():+.2f}pp  "
          f"95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]pp")

    # ----- Step 6: per-category routing detail -----
    print(f"\nPer-category routing detail (category → routed policy → achieved Δ):")
    print(f"  {'category':<12} {'n':>3}  {'routed → ':<24} "
          f"{'cat Δ':>8} {'qsig Δ':>8} {'oracle Δ':>10}")
    for cat in sorted(cat_modes.keys()):
        rp = cat_modes[cat]
        prompts_in_cat = by_cat[cat]
        rated = []
        for label in prompts_in_cat:
            j = labels.index(label)
            rated.append((rates[policy_names.index(rp), j] - rand_rates[j],
                          rates[qsig_idx, j] - rand_rates[j],
                          np.nanmax(rates[:, j]) - rand_rates[j]))
        rated = np.array(rated) * 100
        n = len(prompts_in_cat)
        print(f"  {cat:<12} {n:>3}  {rp:<24} "
              f"{rated[:,0].mean():>+7.1f} {rated[:,1].mean():>+7.1f} "
              f"{rated[:,2].mean():>+9.1f}")

    # ----- Save downstream -----
    out = {
        "categories":           dict(by_cat),
        "cat_winners":          {c: dict(wc) for c, wc in cat_winners.items()},
        "cat_mode_winner":      cat_modes,
        "cat_entropy_bits":     cat_entropies,
        "marginal_entropy_bits": H_marg,
        "conditional_entropy_bits": H_cond,
        "mutual_information_bits": info_gain,
        "router_delta_pp":      achieved_delta,
        "router_minus_qsig_ci_pp": [float(ci_lo), float(ci_hi)],
        "oracle_delta_pp":      oracle_delta,
        "qsigdist_delta_pp":    qsig_delta,
        "headroom_captured_pct": headroom_captured * 100,
    }
    out_path = os.path.join(THIS, "results/meta_iterate/cluster_by_winner.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
