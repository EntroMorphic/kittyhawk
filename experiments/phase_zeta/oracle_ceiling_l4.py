"""L4: held-out oracle CV — establish the realistic routing ceiling.

The full-data oracle reports Δ = +22.96pp (per-prompt max over 11
policies). That's biased upward by selection-on-max with single-
observation per cell. The realistic oracle (what a perfect router
COULD achieve, knowing the true per-prompt winner) is somewhere
between two bounds we can compute from the existing data:

  Upper bound (what we measured): full oracle.
  Lower bound (most pessimistic): 2nd-best policy per prompt
    — treats per-prompt winners as if they were noise artifacts and
    falls back to the runner-up.

Additional checks:
  - Bootstrap CI on full-oracle Δ (cross-prompt variance).
  - Leave-best-policy-out (LOO over policies): for each prompt, drop
    the observed winner, recompute oracle over the remaining 10
    policies. If LBO oracle is close to full oracle, the winner is
    NOT the only policy that excels on that prompt (specialization
    is broad); if LBO oracle is much lower, the winner is uniquely
    specialized (selection-bias risk higher).
  - Split-half reproducibility: split prompts 50/50 randomly,
    compute oracle on each half. Many splits → distribution of
    half-sample oracles. Tells us how stable the +22.96pp is at
    smaller sample sizes.
  - "Common-knowledge oracle": for each prompt, is the per-prompt
    winner the SAME as the best-mean policy on the other 99
    prompts? If high overlap → oracle is "qsigdist most of the
    time" and the +22.96pp is largely about per-prompt random
    matches, not specialization. If low overlap → specialization is
    real and per-prompt-winner identity is informative.

Each estimate is computed and reported. Together they bracket the
realistic oracle ceiling.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np

THIS = os.path.dirname(__file__)
ROUTING_JSON = os.path.join(THIS, "results/meta_iterate/per_prompt_routing.json")


def main():
    with open(ROUTING_JSON) as f:
        d = json.load(f)
    policy_names = d["policy_names"]
    labels = d["labels"]
    rates = np.array(d["rates"])  # rows=policies, cols=prompts
    random_idx = policy_names.index("random")
    qsig_idx   = policy_names.index("qsigdist")
    rand_rates = rates[random_idx, :]
    qsig_rates = rates[qsig_idx, :]

    # exclude random from competition
    non_rand_idx = [i for i in range(len(policy_names)) if i != random_idx]
    rates_nr = rates[non_rand_idx, :]
    names_nr = [policy_names[i] for i in non_rand_idx]
    qsig_idx_nr = names_nr.index("qsigdist")

    n_prompts = len(labels)
    n_pol = len(names_nr)
    print(f"Loaded {n_prompts} prompts × {n_pol} policies (excluding random)\n")

    # ---- Baseline ----
    qsig_delta = float(np.nanmean(qsig_rates - rand_rates)) * 100
    print(f"qsigdist Δ vs random: {qsig_delta:+.2f}pp")

    # ---- Full oracle ----
    full_oracle_rates = np.nanmax(rates_nr, axis=0)
    full_oracle = float(np.nanmean(full_oracle_rates - rand_rates)) * 100
    print(f"\nFull oracle Δ (max over {n_pol} policies per prompt): "
          f"{full_oracle:+.2f}pp")

    # ---- Bootstrap CI on full oracle ----
    rng = np.random.default_rng(20260514)
    bs = []
    for _ in range(10000):
        idx = rng.integers(0, n_prompts, size=n_prompts)
        bs_oracle = float(np.nanmean(
            np.nanmax(rates_nr[:, idx], axis=0) - rand_rates[idx])) * 100
        bs.append(bs_oracle)
    ci_lo, ci_hi = np.quantile(bs, [0.025, 0.975])
    print(f"  bootstrap 95% CI (cross-prompt variance): "
          f"[{ci_lo:+.2f}, {ci_hi:+.2f}]pp")

    # ---- Leave-best-policy-out oracle ----
    lbo_rates = np.empty(n_prompts)
    for j in range(n_prompts):
        col = rates_nr[:, j]
        if np.all(np.isnan(col)):
            lbo_rates[j] = np.nan
            continue
        best_i = int(np.nanargmax(col))
        rest = np.delete(col, best_i)
        lbo_rates[j] = np.nanmax(rest)
    lbo_oracle = float(np.nanmean(lbo_rates - rand_rates)) * 100
    print(f"\nLeave-best-policy-out oracle Δ:  {lbo_oracle:+.2f}pp")
    print(f"  (per-prompt: drop winner, take 2nd-best over {n_pol-1} remaining)")
    print(f"  Full − LBO = {full_oracle - lbo_oracle:+.2f}pp "
          f"(the part of the gain that depends on a single winning policy)")

    # ---- 2nd-best-policy-per-prompt (pessimistic) ----
    second_best_rates = np.empty(n_prompts)
    for j in range(n_prompts):
        col = rates_nr[:, j]
        if np.all(np.isnan(col)):
            second_best_rates[j] = np.nan
            continue
        sorted_col = np.sort(col[~np.isnan(col)])
        # second-best = second from top
        second_best_rates[j] = sorted_col[-2] if len(sorted_col) >= 2 else sorted_col[-1]
    second_oracle = float(np.nanmean(second_best_rates - rand_rates)) * 100
    print(f"\n2nd-best-per-prompt oracle Δ:    {second_oracle:+.2f}pp")
    print(f"  Pessimistic: every prompt routed to its runner-up. Lower bound.")

    # ---- Median-policy-per-prompt (centre of distribution) ----
    median_rates = np.nanmedian(rates_nr, axis=0)
    median_oracle = float(np.nanmean(median_rates - rand_rates)) * 100
    print(f"\nMedian-policy-per-prompt Δ:      {median_oracle:+.2f}pp")
    print(f"  Cross-policy median per prompt. Not really an oracle, but")
    print(f"  measures the centre of the per-policy distribution.")

    # ---- Common-knowledge oracle ----
    # For each prompt p: the "global best" excluding p is the policy with
    # max mean delta on the OTHER 99 prompts. Use that policy's rate on p.
    ck_rates = np.empty(n_prompts)
    common_winner_count = Counter()
    pp_winners = np.empty(n_prompts, dtype=object)
    for j in range(n_prompts):
        idxs = [i for i in range(n_prompts) if i != j]
        mean_d = np.nanmean(rates_nr[:, idxs] - rand_rates[idxs], axis=1)
        if np.all(np.isnan(mean_d)):
            ck_rates[j] = np.nan
            continue
        ck_pol = int(np.nanargmax(mean_d))
        ck_rates[j] = rates_nr[ck_pol, j]
        common_winner_count[names_nr[ck_pol]] += 1
        col_j = rates_nr[:, j]
        if not np.all(np.isnan(col_j)):
            pp_winners[j] = names_nr[int(np.nanargmax(col_j))]
    ck_oracle = float(np.nanmean(ck_rates - rand_rates)) * 100
    print(f"\nCommon-knowledge oracle Δ:        {ck_oracle:+.2f}pp")
    print(f"  Per prompt, use the policy that's best on the OTHER 99 prompts.")
    print(f"  Always-best policies (excluding j): {dict(common_winner_count)}")

    # Per-prompt winner identity vs common-knowledge identity
    ck_predicted = np.empty(n_prompts, dtype=object)
    for j in range(n_prompts):
        idxs = [i for i in range(n_prompts) if i != j]
        mean_d = np.nanmean(rates_nr[:, idxs] - rand_rates[idxs], axis=1)
        if not np.all(np.isnan(mean_d)):
            ck_predicted[j] = names_nr[int(np.nanargmax(mean_d))]
    # Overlap: how often is the per-prompt winner the same as CK
    same = sum(1 for j in range(n_prompts)
               if pp_winners[j] is not None and ck_predicted[j] is not None
               and pp_winners[j] == ck_predicted[j])
    print(f"  Per-prompt winner ≡ CK winner on {same}/{n_prompts} prompts "
          f"({same/n_prompts*100:.0f}%)")
    print(f"  → On {n_prompts-same} prompts ({(n_prompts-same)/n_prompts*100:.0f}%) "
          f"the per-prompt winner is NOT predictable from cross-prompt data.")
    print(f"  This is the prompt-specific-specialization fraction — the part")
    print(f"  of the oracle headroom that requires per-prompt features to capture.")

    # ---- Split-half reproducibility ----
    rng2 = np.random.default_rng(20260517)
    n_repeats = 1000
    split_oracles_A = []
    split_oracles_B = []
    for _ in range(n_repeats):
        perm = rng2.permutation(n_prompts)
        A = perm[:n_prompts//2]
        B = perm[n_prompts//2:]
        rA = float(np.nanmean(
            np.nanmax(rates_nr[:, A], axis=0) - rand_rates[A])) * 100
        rB = float(np.nanmean(
            np.nanmax(rates_nr[:, B], axis=0) - rand_rates[B])) * 100
        split_oracles_A.append(rA)
        split_oracles_B.append(rB)
    split_oracles_A = np.array(split_oracles_A)
    split_oracles_B = np.array(split_oracles_B)
    print(f"\nSplit-half oracle reproducibility ({n_repeats} random splits):")
    print(f"  Mean half-sample oracle:  {split_oracles_A.mean():+.2f}pp")
    print(f"  Std:                       {split_oracles_A.std():.2f}pp")
    print(f"  P25 / P50 / P75:           {np.quantile(split_oracles_A, 0.25):+.2f} / "
          f"{np.median(split_oracles_A):+.2f} / "
          f"{np.quantile(split_oracles_A, 0.75):+.2f}pp")
    diff_AB = split_oracles_A - split_oracles_B
    print(f"  Mean |A - B|:              {np.mean(np.abs(diff_AB)):.2f}pp")
    print(f"  → How much the oracle estimate moves when prompt sample changes.")

    # ---- Summary ----
    print(f"\n{'='*60}\nSummary — realistic oracle ceiling estimates:")
    print(f"  Pessimistic floor (2nd-best per prompt):  {second_oracle:+.2f}pp")
    print(f"  Common-knowledge oracle:                  {ck_oracle:+.2f}pp")
    print(f"  Leave-best-policy-out:                    {lbo_oracle:+.2f}pp")
    print(f"  Full oracle (in-sample):                  {full_oracle:+.2f}pp")
    print(f"  Half-sample oracle mean:                  {split_oracles_A.mean():+.2f}pp")
    print(f"\n  qsigdist baseline:                        {qsig_delta:+.2f}pp")
    print(f"\nHeadroom over qsigdist:")
    print(f"  Pessimistic floor:   {second_oracle - qsig_delta:+.2f}pp")
    print(f"  Common-knowledge:    {ck_oracle - qsig_delta:+.2f}pp")
    print(f"  LBO:                 {lbo_oracle - qsig_delta:+.2f}pp")
    print(f"  Full (overstated):   {full_oracle - qsig_delta:+.2f}pp")

    out = {
        "qsigdist_delta_pp": qsig_delta,
        "full_oracle_pp": full_oracle,
        "full_oracle_ci_pp": [float(ci_lo), float(ci_hi)],
        "leave_best_policy_out_pp": lbo_oracle,
        "second_best_pp": second_oracle,
        "median_policy_pp": median_oracle,
        "common_knowledge_oracle_pp": ck_oracle,
        "pp_winner_eq_ck_winner_count": same,
        "specialization_fraction_pct": (n_prompts - same) / n_prompts * 100,
        "split_half_oracle_mean_pp": float(split_oracles_A.mean()),
        "split_half_oracle_std_pp": float(split_oracles_A.std()),
        "split_half_oracle_p25_p50_p75_pp": [
            float(np.quantile(split_oracles_A, 0.25)),
            float(np.median(split_oracles_A)),
            float(np.quantile(split_oracles_A, 0.75)),
        ],
        "split_half_mean_abs_diff_pp": float(np.mean(np.abs(diff_AB))),
    }
    out_path = os.path.join(THIS, "results/meta_iterate/oracle_ceiling_l4.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
