"""Per-prompt routing analysis on the existing anchor data.

Reframes the meta-routing arc from 'beat qsigdist with a single policy'
to 'enhance qsigdist by routing prompts to the policy that wins on them.'

Entropy is structure: if every prompt's match-rate is dominated by one
policy (low entropy across policies per prompt), there's a per-prompt
winner — and the entropy of WHICH policy wins across prompts tells us
whether routing has headroom (high entropy) or qsigdist is universally
best (low entropy).

Method:
  1. Load no_evict tokens (oracle baseline) and random tokens (Δ
     baseline) per prompt.
  2. For each anchored policy, load its 100 prompt token outputs.
  3. Compute per-prompt match-rate vs no_evict (quality) and per-prompt
     Δ vs random (gain over baseline).
  4. Identify the per-prompt WINNER (max match-rate policy).
  5. Compute the ORACLE Δ — what we'd see if we could pick the winning
     policy per prompt.
  6. Compute the entropy of the win distribution H(winner | prompt).
     If H ≈ 0: one policy wins everywhere; routing dead.
     If H ≈ log2(N_policies): wins evenly distributed; full routing
     headroom.
  7. For each prompt, compute the per-prompt entropy of match-rates
     across policies (how "decided" the prompt is). Compare contested
     vs decided prompts.
  8. Margin analysis: how much would the oracle gain over qsigdist?

The analysis is read-only on existing data; no harness runs needed.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter

import numpy as np

THIS = os.path.dirname(__file__)
N50  = os.path.join(THIS, "results/n50_battery/battery_results.json")
N100 = os.path.join(THIS, "results/n100_incremental/battery_results.json")
META = os.path.join(THIS, "results/meta_iterate")


def match_rate(a, b):
    if not a or not b:
        return None
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]) / n if n else None


def load_fixed_modes():
    """Load the 4 fixed-mode policies (random, fifo, sigdist, qsigdist,
    no_evict) from the N=100 closeout files."""
    with open(N50) as f:
        old = json.load(f)["trials"]
    with open(N100) as f:
        new = json.load(f)["trials"]
    trials = old + new
    by_lm = {}
    for t in trials:
        by_lm[(t["label"], t["mode"])] = t["tokens"]
    return by_lm


def load_meta_iter(w):
    """Load tokens for a meta-mode anchor (looking in iter_* or oneshot_*
    directories). Returns dict label -> tokens."""
    candidates = [
        os.path.join(META, f"iter_{w[0]}_{w[1]}_{w[2]}"),
        os.path.join(META, f"oneshot_{w[0]}_{w[1]}_{w[2]}"),
    ]
    outdir = next((d for d in candidates if os.path.isdir(d)), None)
    if outdir is None:
        return None
    tokens = {}
    import re
    for fname in os.listdir(outdir):
        if not fname.endswith(".log"):
            continue
        label = fname[:-4]
        with open(os.path.join(outdir, fname)) as f:
            content = f.read()
        m = re.search(r"generated tokens\s*=\s*([\d\s\-]+)", content)
        if m:
            tokens[label] = [int(t) for t in m.group(1).strip().split()]
    return tokens


def main():
    fixed = load_fixed_modes()
    # Reconstruct per-policy token dicts: policy_name -> {label: tokens}
    policies = {}
    for name in ("random", "fifo", "sigdist", "qsigdist"):
        policies[name] = {
            label: toks for (label, mode), toks in fixed.items() if mode == name
        }
    no_evict = {label: toks for (label, mode), toks in fixed.items() if mode == "no_evict"}

    # Meta-mode anchored cells (from anchors.json)
    meta_cells = [
        (0, -1, 1), (0, 1, 1), (1, 1, 1), (0, -1, 0), (-1, 1, 1),
        (1, 0, 1), (-1, 0, 1), (1, -1, 1), (1, -1, 0),
    ]
    for w in meta_cells:
        toks = load_meta_iter(w)
        if toks is None:
            print(f"WARN: meta{w} dir not found; skipping", file=sys.stderr)
            continue
        policies[f"meta{w}"] = toks

    # Common labels = intersect across all policies + no_evict + random
    labels = set(no_evict.keys())
    for p in policies.values():
        labels &= set(p.keys())
    labels = sorted(labels)
    print(f"Loaded {len(policies)} policies and {len(labels)} common-label prompts\n")

    # ----- Per-prompt match-rate matrix -----
    policy_names = list(policies.keys())
    rates = np.zeros((len(policy_names), len(labels)))  # rows=policies, cols=prompts
    for j, label in enumerate(labels):
        ne = no_evict[label]
        for i, name in enumerate(policy_names):
            toks = policies[name].get(label)
            r = match_rate(ne, toks) if toks else None
            rates[i, j] = r if r is not None else np.nan

    # ----- Per-policy mean Δ vs random -----
    random_idx = policy_names.index("random")
    delta_vs_random = rates - rates[random_idx, :]
    mean_delta = np.nanmean(delta_vs_random, axis=1) * 100
    print("Mean Δ vs random (pp):")
    order = np.argsort(-mean_delta)
    for i in order:
        print(f"  {policy_names[i]:<18}  {mean_delta[i]:+6.2f}pp")

    # ----- Per-prompt winner (excluding random) -----
    non_rand_idx = [i for i, n in enumerate(policy_names) if n != "random"]
    rates_nr = rates[non_rand_idx, :]
    names_nr = [policy_names[i] for i in non_rand_idx]
    winner_per_prompt = []
    margin_per_prompt = []  # winner_rate - second_rate
    for j in range(len(labels)):
        col = rates_nr[:, j]
        if np.all(np.isnan(col)):
            winner_per_prompt.append(None)
            margin_per_prompt.append(0)
            continue
        winner_i = int(np.nanargmax(col))
        sorted_col = np.sort(col[~np.isnan(col)])
        margin = sorted_col[-1] - sorted_col[-2] if len(sorted_col) >= 2 else 0
        winner_per_prompt.append(names_nr[winner_i])
        margin_per_prompt.append(margin)

    # ----- Win distribution -----
    win_counts = Counter(w for w in winner_per_prompt if w is not None)
    n_decided = sum(win_counts.values())
    print(f"\nPer-prompt winners (out of {n_decided} prompts; ties broken by argmax order):")
    for name, cnt in sorted(win_counts.items(), key=lambda x: -x[1]):
        share = cnt / n_decided
        print(f"  {name:<18}  {cnt:>3}  ({share*100:>4.1f}%)")

    # ----- Entropy of win distribution -----
    probs = np.array([cnt / n_decided for cnt in win_counts.values()])
    H_winners = -np.sum(probs * np.log2(probs)) if len(probs) > 1 else 0
    max_H = math.log2(len(non_rand_idx))
    print(f"\nEntropy of win distribution: H = {H_winners:.3f} bits")
    print(f"  Maximum possible (uniform over {len(non_rand_idx)} policies): {max_H:.3f} bits")
    print(f"  Normalized: {H_winners/max_H:.3f} (0 = one policy wins everywhere; 1 = uniform)")

    # ----- Oracle Δ — pick the per-prompt winner -----
    oracle_rates = np.array([
        rates_nr[non_rand_idx.index(policy_names.index(w)) if False else names_nr.index(w), j]
        if w is not None else np.nan
        for j, w in enumerate(winner_per_prompt)
    ])
    # Re-do simply
    oracle_rates = np.full(len(labels), np.nan)
    for j in range(len(labels)):
        col = rates_nr[:, j]
        if not np.all(np.isnan(col)):
            oracle_rates[j] = np.nanmax(col)
    oracle_delta = np.nanmean(oracle_rates - rates[random_idx, :]) * 100
    qsig_delta = mean_delta[policy_names.index("qsigdist")]
    print(f"\nOracle Δ (pick per-prompt winner): {oracle_delta:+.2f}pp")
    print(f"qsigdist Δ (best fixed):            {qsig_delta:+.2f}pp")
    print(f"Headroom over qsigdist:             {oracle_delta - qsig_delta:+.2f}pp")

    # Confidence: bootstrap CI on the oracle - qsig difference
    diff_per_prompt = (oracle_rates - rates[policy_names.index("qsigdist"), :]) * 100
    rng = np.random.default_rng(20260514)
    boots = []
    valid = diff_per_prompt[~np.isnan(diff_per_prompt)]
    for _ in range(10000):
        idx = rng.integers(0, len(valid), size=len(valid))
        boots.append(valid[idx].mean())
    ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])
    print(f"Bootstrap 95% CI on (oracle − qsigdist): [{ci_lo:+.2f}, {ci_hi:+.2f}]pp")
    if ci_lo > 0:
        print(f"  ROUTING HEADROOM IS SIGNIFICANT (lower CI > 0)")
    elif ci_hi < 0:
        print(f"  qsigdist beats oracle?? (sanity-check failure)")
    else:
        print(f"  Headroom not statistically distinguishable from 0")

    # ----- Per-prompt entropy of policy match rates -----
    # For each prompt, normalize the policy match-rates to a probability
    # distribution and compute entropy. High entropy = no clear winner;
    # low entropy = one policy dominates. Then compare margin distribution
    # for high-vs-low entropy prompts.
    per_prompt_H = np.full(len(labels), np.nan)
    for j in range(len(labels)):
        col = rates_nr[:, j]
        col = col[~np.isnan(col)]
        # Softmax-like normalization (avoid 0 mass when all are equal)
        if col.size == 0 or col.sum() == 0:
            continue
        p = col / col.sum()
        p = p[p > 0]
        per_prompt_H[j] = -np.sum(p * np.log2(p))
    print(f"\nPer-prompt entropy of policy-match-rates "
          f"(higher = more contested):")
    print(f"  Mean: {np.nanmean(per_prompt_H):.3f} bits")
    print(f"  Median: {np.nanmedian(per_prompt_H):.3f}")
    print(f"  90th pctile: {np.nanquantile(per_prompt_H, 0.9):.3f}")
    print(f"  10th pctile: {np.nanquantile(per_prompt_H, 0.1):.3f}")

    # ----- Which prompts have the biggest oracle > qsigdist gap? -----
    print(f"\nTop 10 prompts by oracle headroom over qsigdist:")
    diff_per_prompt = (oracle_rates - rates[policy_names.index("qsigdist"), :]) * 100
    pairs = sorted(zip(labels, diff_per_prompt, winner_per_prompt),
                   key=lambda p: -p[1] if not math.isnan(p[1]) else 0)
    for label, d, w in pairs[:10]:
        print(f"  {label:<28} +{d:>5.1f}pp   winner: {w}")
    print(f"\nTop 10 prompts where qsigdist STRICTLY loses:")
    losers = [(l, d, w) for l, d, w in zip(labels, diff_per_prompt, winner_per_prompt)
              if d > 0 and w != "qsigdist"]
    losers.sort(key=lambda p: -p[1])
    for label, d, w in losers[:10]:
        print(f"  {label:<28} +{d:>5.1f}pp   winner: {w}")

    # ----- Save full result for downstream use -----
    out_path = os.path.join(META, "per_prompt_routing.json")
    with open(out_path, "w") as f:
        json.dump({
            "policy_names": policy_names,
            "labels": labels,
            "rates": rates.tolist(),
            "mean_delta_vs_random_pp": mean_delta.tolist(),
            "winner_per_prompt": winner_per_prompt,
            "oracle_delta_pp": oracle_delta,
            "qsigdist_delta_pp": float(qsig_delta),
            "headroom_pp": oracle_delta - float(qsig_delta),
            "headroom_ci_pp": [float(ci_lo), float(ci_hi)],
            "win_distribution_entropy_bits": float(H_winners),
            "win_distribution_max_entropy_bits": max_H,
            "per_prompt_entropy_mean_bits": float(np.nanmean(per_prompt_H)),
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
