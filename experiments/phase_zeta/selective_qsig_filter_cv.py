"""Held-out CV: selective router between qsigdist (default) and
qsig_filter K=1 (for "filter-helps" categories).

The qsig_filter K=1 battery showed:
  - Mean Δ: +8.50pp (vs qsigdist +6.38pp; +2.12pp gain, CI [-1.33, +5.83])
  - Per-category: clear winners (tech +20.9, math +9.4, geography +22.9, etc.)
                  clear losers (code -3.4, q -4.2, long -2.3, poetry -2.1)

Selective router: route to K=1 only on categories where K=1 helps;
default to qsigdist elsewhere. This tests whether the categorical
specialization transfers to held-out prompts.

Method (5-fold CV × 20 repeats):
  1. Split prompts 80/20 train/test.
  2. On TRAIN: compute (K=1 - qsigdist) per category. Identify
     categories with mean gain > δ AND n_train ≥ 2.
  3. Apply selective rule to TEST: routed_policy = K=1 if test
     prompt's category is in the winning set, else qsigdist.
  4. Compute realized Δ on TEST.

Compare to:
  - Always-qsigdist (baseline).
  - Always-K=1 (the +8.50pp full-data Δ).
  - In-sample selective (overfit ceiling).

This is structurally the L1.2 method but on a 2-policy space with
one being true integration.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

THIS = os.path.dirname(__file__)
META = os.path.join(THIS, "results/meta_iterate")
N50  = os.path.join(THIS, "results/n50_battery/battery_results.json")
N100 = os.path.join(THIS, "results/n100_incremental/battery_results.json")
K1_DIR = os.path.join(META, "qsig_filter_K1")


def category_of(label):
    return label.split("_")[0] if "_" in label else label


def match_rate(a, b):
    if not a or not b: return None
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]) / n if n else None


def load_baselines():
    with open(N50) as f:
        old = json.load(f)["trials"]
    with open(N100) as f:
        new = json.load(f)["trials"]
    base = {(t["label"], t["mode"]): t["tokens"] for t in old + new}
    return base


def load_k1_tokens():
    out = {}
    for fn in os.listdir(K1_DIR):
        if not fn.endswith(".log"): continue
        with open(os.path.join(K1_DIR, fn)) as f:
            content = f.read()
        m = re.search(r"generated tokens\s*=\s*([\d\s\-]+)", content)
        if m:
            out[fn[:-4]] = [int(t) for t in m.group(1).strip().split()]
    return out


def main():
    base = load_baselines()
    k1   = load_k1_tokens()
    labels = sorted(set(k1.keys()) & {l for (l, m) in base if m == "qsigdist"}
                                   & {l for (l, m) in base if m == "no_evict"}
                                   & {l for (l, m) in base if m == "random"})
    print(f"Common labels: {len(labels)}")

    # Per-prompt rates
    qsig_r = np.array([match_rate(base[(l,"no_evict")], base[(l,"qsigdist")]) for l in labels])
    rand_r = np.array([match_rate(base[(l,"no_evict")], base[(l,"random")]) for l in labels])
    k1_r   = np.array([match_rate(base[(l,"no_evict")], k1[l]) for l in labels])

    qsig_delta_full = np.nanmean(qsig_r - rand_r) * 100
    k1_delta_full   = np.nanmean(k1_r - rand_r) * 100
    print(f"Always-qsigdist Δ:  {qsig_delta_full:+.2f}pp")
    print(f"Always-K=1 Δ:        {k1_delta_full:+.2f}pp")
    print(f"  K=1 − qsigdist on full data: {(k1_r - qsig_r).mean()*100:+.2f}pp")

    # In-sample selective (oracle ceiling)
    cat_gain = defaultdict(list)
    for l, gain in zip(labels, (k1_r - qsig_r) * 100):
        if not np.isnan(gain):
            cat_gain[category_of(l)].append(gain)
    in_sample_winning_cats = {c for c, g in cat_gain.items() if np.mean(g) > 0}
    selective_in_rates = np.array([
        k1_r[i] if category_of(labels[i]) in in_sample_winning_cats else qsig_r[i]
        for i in range(len(labels))])
    selective_in_delta = np.nanmean(selective_in_rates - rand_r) * 100
    print(f"\nIn-sample selective (route K=1 if cat-mean > 0): "
          f"{selective_in_delta:+.2f}pp")
    print(f"  Selective − qsigdist (in-sample): "
          f"{(selective_in_rates - qsig_r).mean()*100:+.2f}pp")
    print(f"  Categories routed to K=1 (in-sample): "
          f"{sorted(in_sample_winning_cats)}")

    # Held-out CV
    print("\n" + "=" * 70)
    print("HELD-OUT CV (5-fold × 20 repeats)")
    print("=" * 70)
    K_FOLDS = 5
    REPEATS = 20
    rng = np.random.default_rng(20260514)
    n = len(labels)

    test_diffs_vs_qsig = []
    test_diffs_vs_k1   = []
    test_routed_rates  = []
    test_qsig_rates    = []
    test_rand_rates    = []
    fold_chosen_cats   = []
    for rep in range(REPEATS):
        perm = rng.permutation(n)
        folds = np.array_split(perm, K_FOLDS)
        for k in range(K_FOLDS):
            test = list(folds[k])
            train = [i for f in (folds[:k] + folds[k+1:]) for i in f]
            # Sweep δ on train; pick best
            train_cat_gain = defaultdict(list)
            for i in train:
                g = (k1_r[i] - qsig_r[i]) * 100
                if not np.isnan(g):
                    train_cat_gain[category_of(labels[i])].append(g)
            # Try a few δ thresholds; pick best by train selective-Δ
            best_delta = -np.inf; best_thr = 0; best_cats = set()
            for thr in [-2, 0, 2, 5, 8, 12]:
                cats = {c for c, gs in train_cat_gain.items() if np.mean(gs) > thr and len(gs) >= 2}
                # Apply to TRAIN
                tr_routed = np.array([
                    k1_r[i] if category_of(labels[i]) in cats else qsig_r[i]
                    for i in train])
                tr_d = (tr_routed - qsig_r[train]).mean() * 100
                if tr_d > best_delta:
                    best_delta = tr_d; best_thr = thr; best_cats = cats
            # Apply best (cats set) to TEST
            te_routed = np.array([
                k1_r[i] if category_of(labels[i]) in best_cats else qsig_r[i]
                for i in test])
            te_diff_q = (te_routed - qsig_r[test]) * 100
            te_diff_k = (te_routed - k1_r[test]) * 100
            test_diffs_vs_qsig.extend(te_diff_q.tolist())
            test_diffs_vs_k1.extend(te_diff_k.tolist())
            test_routed_rates.extend(te_routed.tolist())
            test_qsig_rates.extend(qsig_r[test].tolist())
            test_rand_rates.extend(rand_r[test].tolist())
            fold_chosen_cats.append(tuple(sorted(best_cats)))

    test_diffs_vs_qsig = np.array(test_diffs_vs_qsig)
    test_routed_rates  = np.array(test_routed_rates)
    test_rand_rates    = np.array(test_rand_rates)

    selective_test_delta = np.nanmean(test_routed_rates - test_rand_rates) * 100
    print(f"  Held-out selective Δ:           {selective_test_delta:+.2f}pp")
    print(f"  Held-out selective − qsigdist:  {test_diffs_vs_qsig.mean():+.2f}pp")
    # Bootstrap CI
    rng2 = np.random.default_rng(20260516)
    bs = []
    for _ in range(10000):
        idx = rng2.integers(0, len(test_diffs_vs_qsig), size=len(test_diffs_vs_qsig))
        bs.append(test_diffs_vs_qsig[idx].mean())
    ci_lo, ci_hi = np.quantile(bs, [0.025, 0.975])
    print(f"  95% CI:                         [{ci_lo:+.2f}, {ci_hi:+.2f}]pp")
    if ci_lo > 0:
        print(f"  HELD-OUT SELECTIVE BEATS QSIGDIST (CI excludes 0)")
    elif ci_hi < 0:
        print(f"  HELD-OUT SELECTIVE LOSES TO QSIGDIST")
    else:
        print(f"  HELD-OUT SELECTIVE not distinguishable from qsigdist")

    # Compare to always-K=1 in held-out
    held_out_k1_delta = np.nanmean(k1_r - rand_r) * 100  # same as full since no train involved for k1
    test_diffs_vs_k1 = np.array(test_diffs_vs_k1)
    print(f"\n  Always-K=1 Δ (full-data):       {held_out_k1_delta:+.2f}pp")
    print(f"  Selective − Always-K=1 (held):  {test_diffs_vs_k1.mean():+.2f}pp")

    # Summary
    print(f"\n{'='*70}\nSummary:\n")
    print(f"  baseline qsigdist:                 {qsig_delta_full:+.2f}pp")
    print(f"  always-K=1 (full data):            {k1_delta_full:+.2f}pp  (+{k1_delta_full-qsig_delta_full:.2f})")
    print(f"  in-sample selective:               {selective_in_delta:+.2f}pp  (+{selective_in_delta-qsig_delta_full:.2f})")
    print(f"  HELD-OUT selective:                {selective_test_delta:+.2f}pp  (+{selective_test_delta-qsig_delta_full:.2f})")

    # Most chosen categories
    from collections import Counter
    cat_freq = Counter()
    for cats in fold_chosen_cats:
        for c in cats:
            cat_freq[c] += 1
    print(f"\n  Top categories chosen across folds (out of {K_FOLDS*REPEATS} folds):")
    for c, count in cat_freq.most_common(12):
        pct = count / (K_FOLDS * REPEATS) * 100
        print(f"    {c:<14}  {count:>3} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
