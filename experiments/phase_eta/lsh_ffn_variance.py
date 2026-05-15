"""B1.5: bucket-conditioned variance reduction.

For each (layer, k), measure how much of the FFN-input variance is
explained by trit-lattice bucket assignment alone. Conceptually:
  total_var(X) = within_bucket_var + between_bucket_var
  variance_explained = between_bucket_var / total_var

Higher = bucket assignment captures more of the input structure;
within-bucket inputs are mutually similar → bucket-conditioned
compute can plausibly specialize. Lower = inputs within a bucket
are nearly as variable as inputs across buckets → bucketing isn't
informative for the FFN.

Compares against k-means baseline (same K_clusters) as the upper
bound on what unsupervised partition can achieve at the given
cluster count.

Decision criterion for proceeding to B2 (LSH FFN drop-in):
  - LSH variance_explained ≥ 0.5 × k-means variance_explained at
    matched cluster count → LSH partition is "competitive" with
    a learned partition.
  - LSH variance_explained ≥ 0.3 in absolute terms → bucket
    assignment captures meaningful structure.
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
    m = re.match(r"^(.+)_p(\d+)_l(\d+)\.bin$", fn)
    if not m: return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def load_layer(layer):
    recs = []
    for fn in os.listdir(DUMP_DIR):
        p = parse_filename(fn)
        if not p: continue
        label, pos, l = p
        if l != layer: continue
        a = np.fromfile(os.path.join(DUMP_DIR, fn), dtype=np.int32)
        if a.shape[0] != HIDDEN: continue
        recs.append((label, pos, a))
    labels = [r[0] for r in recs]
    acts = np.stack([r[2] for r in recs], axis=0).astype(np.float64)
    return labels, acts


def threshold_extract(acts, tau):
    """Float64 → trit signature."""
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


def kmeans_labels(acts, K, seed=20260514):
    rng = np.random.default_rng(seed)
    n, d = acts.shape
    centers = np.empty((K, d))
    centers[0] = acts[rng.integers(0, n)]
    for c in range(1, K):
        d2 = np.min(((acts[:, None, :] - centers[None, :c, :]) ** 2).sum(axis=2), axis=1)
        probs = d2 / (d2.sum() + 1e-12)
        centers[c] = acts[rng.choice(n, p=probs)]
    labels = np.zeros(n, dtype=np.int32)
    for it in range(50):
        d2 = ((acts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1)
        if it > 0 and (new_labels == labels).all(): break
        labels = new_labels
        for c in range(K):
            mask = labels == c
            if mask.any():
                centers[c] = acts[mask].mean(axis=0)
    return labels


def variance_explained(acts, labels):
    """Between-bucket variance / total variance.

    For each cluster c: compute mean μ_c of cluster members, count n_c.
    SS_between = sum_c n_c * ||μ_c - μ_total||²
    SS_total   = sum_i ||x_i - μ_total||²
    """
    mu_total = acts.mean(axis=0)
    ss_total = ((acts - mu_total) ** 2).sum()
    ss_between = 0.0
    unique = np.unique(labels)
    for c in unique:
        mask = labels == c
        n_c = int(mask.sum())
        if n_c == 0: continue
        mu_c = acts[mask].mean(axis=0)
        ss_between += n_c * ((mu_c - mu_total) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def main():
    print("B1.5 — bucket-conditioned variance reduction\n")
    for layer in (2, 15, 27):
        labels, acts = load_layer(layer)
        n = acts.shape[0]
        print(f"\n{'='*70}")
        print(f"Layer {layer}: n={n} samples, dim {acts.shape[1]}")
        print(f"{'='*70}")
        print(f"{'k':>2}  {'n_buckets':>10}  {'tau':>10}  {'LSH var_expl':>14}  "
              f"{'k-means @ same K':>18}  {'ratio LSH/km':>14}")
        for k in (4, 5, 6, 8, 10):
            for tau in (2500, "adaptive"):
                sig = threshold_extract(acts, tau)
                buckets = hash_buckets(sig, k)
                n_used = len(set(buckets.tolist()))
                lsh_ve = variance_explained(acts, buckets)
                # k-means at MATCHED cluster count (n_used). Cap at sample count.
                K_cmp = min(n_used, n - 1, 50)  # k-means with too many clusters is slow + degenerate
                km_lab = kmeans_labels(acts, K_cmp)
                km_ve = variance_explained(acts, km_lab)
                ratio = lsh_ve / km_ve if km_ve > 0 else 0
                marker = ""
                if lsh_ve >= 0.3 and ratio >= 0.5: marker = " ✓"
                elif lsh_ve >= 0.3 or ratio >= 0.5: marker = " ~"
                print(f"  {k:>2}  {n_used:>10}  {str(tau):>10}  "
                      f"{lsh_ve:>13.4f}  "
                      f"{km_ve:>17.4f} (K={K_cmp})  "
                      f"{ratio:>13.3f}{marker}")


if __name__ == "__main__":
    main()
