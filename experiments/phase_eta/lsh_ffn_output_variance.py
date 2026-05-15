"""B1.6: how much of FFN OUTPUT variance is explained by INPUT bucket?

This bounds the quality of the simplest possible LSH FFN tile:
  predict_output(input) = bucket_mean_output[bucket(input)]

If output variance explained ≥ 0.5: bucket-mean prediction recovers
most of the dense FFN. Even constant per-bucket tiles work.

If output variance explained ≪ input variance explained: same
bucket maps to varied outputs → tiles need to do real computation
(matrix per bucket, or subset of dense FFN, etc.).

Also computes:
  - Held-out CV: split prompts 80/20, train means on 80, evaluate on 20.
    Reports cosine similarity between predicted and true outputs per
    held-out token + L2 distance.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict

import numpy as np

THIS = os.path.dirname(__file__)
DUMP_DIR = os.path.join(THIS, "results/ffn_dump")
HIDDEN = 2560


def parse_filename(fn):
    m = re.match(r"^(.+)_p(\d+)_l(\d+)(_out)?\.bin$", fn)
    if not m: return None
    is_out = m.group(4) is not None
    return m.group(1), int(m.group(2)), int(m.group(3)), is_out


def load_layer_pairs(layer):
    by_key = defaultdict(dict)
    for fn in os.listdir(DUMP_DIR):
        p = parse_filename(fn)
        if not p: continue
        label, pos, l, is_out = p
        if l != layer: continue
        key = (label, pos)
        which = "out" if is_out else "in"
        a = np.fromfile(os.path.join(DUMP_DIR, fn), dtype=np.int32)
        if a.shape[0] != HIDDEN: continue
        by_key[key][which] = a
    # Keep only pairs where both in and out are present
    labels, positions, ins, outs = [], [], [], []
    for (label, pos), d in by_key.items():
        if "in" not in d or "out" not in d: continue
        labels.append(label); positions.append(pos)
        ins.append(d["in"]); outs.append(d["out"])
    return labels, np.array(positions), \
           np.stack(ins, axis=0).astype(np.float64), \
           np.stack(outs, axis=0).astype(np.float64)


def threshold_extract(acts, tau):
    if tau == "adaptive":
        med = np.median(np.abs(acts), axis=1, keepdims=True)
        tau_arr = med
    else:
        tau_arr = np.full((acts.shape[0], 1), tau)
    sig = np.zeros_like(acts, dtype=np.int8)
    sig[acts > tau_arr] = 1
    sig[acts < -tau_arr] = -1
    return sig


def hash_buckets(sig, k):
    sub = sig[:, :k]
    digits = (sub + 1).astype(np.int64)
    powers = 3 ** np.arange(k, dtype=np.int64)
    return (digits * powers).sum(axis=1)


def variance_explained(acts, labels):
    mu_total = acts.mean(axis=0)
    ss_total = ((acts - mu_total) ** 2).sum()
    if ss_total <= 0: return 0.0
    ss_between = 0.0
    for c in np.unique(labels):
        mask = labels == c
        n_c = int(mask.sum())
        if n_c == 0: continue
        mu_c = acts[mask].mean(axis=0)
        ss_between += n_c * ((mu_c - mu_total) ** 2).sum()
    return float(ss_between / ss_total)


def held_out_cv(ins, outs, buckets, prompt_labels, n_splits=5, seed=20260514):
    """For each split: train bucket-mean output on TRAIN, predict on TEST,
    measure cosine similarity vs true output."""
    rng = np.random.default_rng(seed)
    unique_labels = sorted(set(prompt_labels))
    all_cos = []
    all_l2 = []
    for split in range(n_splits):
        # Random per-prompt split
        perm = list(unique_labels)
        rng.shuffle(perm)
        n_test = max(1, len(perm) // 5)  # 80/20
        test_prompts = set(perm[:n_test])
        train_idx = [i for i, l in enumerate(prompt_labels) if l not in test_prompts]
        test_idx  = [i for i, l in enumerate(prompt_labels) if l in test_prompts]
        # Bucket means on TRAIN
        bucket_mean = {}
        for i in train_idx:
            b = int(buckets[i])
            if b not in bucket_mean: bucket_mean[b] = []
            bucket_mean[b].append(outs[i])
        bucket_mean = {b: np.mean(np.stack(v, axis=0), axis=0) for b, v in bucket_mean.items()}
        # Train mean overall (fallback for cold buckets)
        train_overall_mean = outs[train_idx].mean(axis=0)
        # Predict on TEST
        for i in test_idx:
            b = int(buckets[i])
            pred = bucket_mean.get(b, train_overall_mean)
            true = outs[i]
            cos = float(np.dot(pred, true) / (np.linalg.norm(pred) * np.linalg.norm(true) + 1e-12))
            l2  = float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))
            all_cos.append(cos)
            all_l2.append(l2)
    return float(np.mean(all_cos)), float(np.median(all_cos)), float(np.mean(all_l2))


def main():
    print("B1.6 — FFN output variance explained by INPUT bucket\n")
    for layer in (2, 15, 27):
        labels, positions, ins, outs = load_layer_pairs(layer)
        n = ins.shape[0]
        print(f"\n{'='*70}")
        print(f"Layer {layer}: {n} (input, output) pairs")
        print(f"  input  std={ins.std():.0f}  abs_p50={np.median(np.abs(ins)):.0f}")
        print(f"  output std={outs.std():.0f}  abs_p50={np.median(np.abs(outs)):.0f}")
        print(f"{'='*70}")

        # Total output variance
        total_out_var = ((outs - outs.mean(axis=0)) ** 2).sum()
        print(f"  Total output variance (Frobenius): {total_out_var:.2e}")

        print(f"\n  {'tau':>10}  {'k':>3}  {'n_buckets':>10}  "
              f"{'IN var_expl':>13}  {'OUT var_expl':>13}  "
              f"{'CV cos (μ/median)':>18}  {'CV L2-rel':>10}")
        for tau in (2500, "adaptive"):
            for k in (4, 5, 6, 8, 10):
                sig = threshold_extract(ins, tau)
                buckets = hash_buckets(sig, k)
                n_used = len(set(buckets.tolist()))
                in_ve  = variance_explained(ins, buckets)
                out_ve = variance_explained(outs, buckets)
                cos_mean, cos_med, l2_rel = held_out_cv(ins, outs, buckets, labels)
                print(f"  {str(tau):>10}  {k:>3}  {n_used:>10}  "
                      f"{in_ve:>13.4f}  {out_ve:>13.4f}  "
                      f"{cos_mean:>+8.3f} / {cos_med:>+6.3f}  "
                      f"{l2_rel:>9.3f}")


if __name__ == "__main__":
    main()
