"""Analyze qsig_filter results: paired CI vs qsigdist + per-prompt
complementarity.

For each completed K (looks for results/meta_iterate/qsig_filter_K*),
compute:
  - mean Δ vs random
  - paired (qsig_filter - qsigdist) per prompt
  - bootstrap CI on the paired difference
  - per-prompt: how often does filter beat qsigdist? by how much?
  - which categories does filter help most?
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

THIS = os.path.dirname(__file__)
META_DIR = os.path.join(THIS, "results/meta_iterate")
N50  = os.path.join(THIS, "results/n50_battery/battery_results.json")
N100 = os.path.join(THIS, "results/n100_incremental/battery_results.json")


def category_of(label):
    return label.split("_")[0] if "_" in label else label


def match_rate(a, b):
    if not a or not b:
        return None
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]) / n if n else None


def load_filter_tokens(K):
    outdir = os.path.join(META_DIR, f"qsig_filter_K{K}")
    if not os.path.isdir(outdir):
        return None
    tokens = {}
    for fn in os.listdir(outdir):
        if not fn.endswith(".log"):
            continue
        label = fn[:-4]
        with open(os.path.join(outdir, fn)) as f:
            content = f.read()
        m = re.search(r"generated tokens\s*=\s*([\d\s\-]+)", content)
        if m:
            tokens[label] = [int(t) for t in m.group(1).strip().split()]
    return tokens


def load_baselines():
    with open(N50) as f:
        old = json.load(f)["trials"]
    with open(N100) as f:
        new = json.load(f)["trials"]
    base = {(t["label"], t["mode"]): t["tokens"] for t in old + new}
    return base


def main():
    base = load_baselines()
    Ks = []
    for d in glob.glob(os.path.join(META_DIR, "qsig_filter_K*")):
        K = int(os.path.basename(d).replace("qsig_filter_K", ""))
        Ks.append(K)
    Ks.sort()
    print(f"Found Ks: {Ks}\n")

    # Per-K analysis: use each K's OWN completed labels (don't intersect)
    rng = np.random.default_rng(20260514)
    print(f"{'K':>3}  {'n':>3}  {'mean Δ':>8}  {'(K - qsig)':>12}  {'95% CI on (K-qsig)':>22}  "
          f"{'wins/ties/losses':>20}")
    for K in Ks:
        toks = load_filter_tokens(K)
        if toks is None: continue
        # K's labels with valid baselines
        valid = [l for l in toks
                 if (l, "no_evict") in base and (l, "qsigdist") in base
                 and (l, "random") in base]
        if not valid: continue
        qsig_rates = np.array([
            match_rate(base[(l, "no_evict")], base[(l, "qsigdist")]) for l in valid])
        rand_rates = np.array([
            match_rate(base[(l, "no_evict")], base[(l, "random")]) for l in valid])
        filt_rates = np.array([
            match_rate(base[(l, "no_evict")], toks[l]) for l in valid])
        delta_vs_rand = (filt_rates - rand_rates) * 100
        delta_vs_qsig = (filt_rates - qsig_rates) * 100
        valid_diffs = delta_vs_qsig[~np.isnan(delta_vs_qsig)]
        boots = []
        for _ in range(10000):
            idx = rng.integers(0, len(valid_diffs), size=len(valid_diffs))
            boots.append(valid_diffs[idx].mean())
        ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])
        wins = int(np.sum(valid_diffs > 0))
        ties = int(np.sum(valid_diffs == 0))
        losses = int(np.sum(valid_diffs < 0))
        sig_marker = " ***" if ci_lo > 0 else " (worse)" if ci_hi < 0 else ""
        print(f"  {K:>3}  {len(valid):>3}  {np.nanmean(delta_vs_rand):>+7.2f}pp  "
              f"{valid_diffs.mean():>+11.2f}pp  "
              f"[{ci_lo:>+5.2f}, {ci_hi:>+5.2f}]pp  "
              f"{wins:>3}/{ties:>3}/{losses:>3}{sig_marker}")

    # For best (only-fully-complete) K, decompose by category
    best_K = None
    best_delta = -np.inf
    for K in Ks:
        toks = load_filter_tokens(K)
        if toks is None or len(toks) < 100: continue
        valid_K = [l for l in toks
                   if (l, "no_evict") in base and (l, "qsigdist") in base
                   and (l, "random") in base]
        qr = np.array([match_rate(base[(l, "no_evict")], base[(l, "qsigdist")]) for l in valid_K])
        rr = np.array([match_rate(base[(l, "no_evict")], base[(l, "random")]) for l in valid_K])
        fr = np.array([match_rate(base[(l, "no_evict")], toks[l]) for l in valid_K])
        d = np.nanmean(fr - rr) * 100
        if d > best_delta:
            best_delta = d; best_K = K
    print(f"\nBest fully-complete K = {best_K} at Δ = {best_delta:+.2f}pp")
    if best_K is not None:
        toks = load_filter_tokens(best_K)
        valid_K = [l for l in toks
                   if (l, "no_evict") in base and (l, "qsigdist") in base
                   and (l, "random") in base]
        qr = np.array([match_rate(base[(l, "no_evict")], base[(l, "qsigdist")]) for l in valid_K])
        fr = np.array([match_rate(base[(l, "no_evict")], toks[l]) for l in valid_K])
        delta_vs_qsig = (fr - qr) * 100
        cat_gain = defaultdict(list)
        for l, d in zip(valid_K, delta_vs_qsig):
            if not np.isnan(d):
                cat_gain[category_of(l)].append(d)
        print(f"\nPer-category K={best_K} − qsigdist (only categories n≥2 shown):")
        cat_means = sorted(((c, np.mean(g), len(g)) for c, g in cat_gain.items() if len(g) >= 2),
                            key=lambda x: -x[1])
        for c, m, n in cat_means:
            mark = " ***" if m > 5 else " (hurt)" if m < -5 else ""
            print(f"  {c:<14} n={n:>2}  Δ = {m:>+6.2f}pp{mark}")

    # Save
    out = {"qsigdist_delta_pp": float(qsig_delta), "results": {}}
    for K in Ks:
        toks = load_filter_tokens(K)
        if toks is None: continue
        filt_rates = np.array([
            match_rate(base[(l, "no_evict")], toks.get(l)) if toks.get(l) else np.nan
            for l in valid])
        out["results"][K] = {
            "mean_delta_pp": float(np.nanmean(filt_rates - rand_rates) * 100),
            "mean_delta_vs_qsig_pp": float(np.nanmean((filt_rates - qsig_rates) * 100)),
        }
    out_path = os.path.join(META_DIR, "qsig_filter_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
