"""Trit Lattice LSH FFN — synthetic-validation prototype.

Validates the addressing scheme before any harness work. Variant (a)
from the proposal: hard hash, append-only.

Protocol:
  1. Generate M cluster centers as ternary vectors of dim d.
  2. Per-cluster: produce N samples by perturbing the center
     (flip a fraction of trits to {-1, 0, +1} uniformly).
  3. Hash each sample by taking the first k trits of its full
     signature → bucket key in [0, 3^k).
  4. Measure:
     - In-cluster collision rate: pairs from same cluster sharing bucket
     - Cross-cluster collision rate: pairs from different clusters sharing bucket
     - Bucket cluster purity: for each populated bucket, what fraction
       of its samples come from the dominant cluster
     - Bucket utilization Gini (skewness of bucket occupancy)
  5. Sweep k ∈ {4, 6, 8, 10, 12, 14}; sweep noise ∈ {0.05, 0.10, 0.20}.

Success criteria for the addressing scheme:
  - In-cluster >> cross-cluster collision (good LSH property)
  - Average bucket purity > 0.8 at moderate k (clusters auto-discovered)
  - Append-only learning: each new sample either hits an existing
    bucket (use existing tile) or instantiates a fresh one — never
    overwrites.
"""
from __future__ import annotations

import argparse
import itertools
import os
from collections import Counter, defaultdict

import numpy as np


def make_clusters(M: int, d: int, rng: np.random.Generator):
    """M cluster centers as random ternary vectors in {-1, 0, +1}^d."""
    return rng.choice([-1, 0, 1], size=(M, d))


def sample_around(center: np.ndarray, n: int, noise: float, rng: np.random.Generator):
    """Per-trit, with probability noise replace with a random trit
    from {-1, 0, +1}; otherwise keep the center's trit."""
    d = center.shape[0]
    out = np.tile(center, (n, 1)).copy()
    flip_mask = rng.random((n, d)) < noise
    new_trits = rng.choice([-1, 0, 1], size=(n, d))
    out[flip_mask] = new_trits[flip_mask]
    return out


def hash_to_bucket(samples: np.ndarray, k: int) -> np.ndarray:
    """Take first k trits as base-3 integer key. Trits {-1, 0, +1}
    encoded as {0, 1, 2}. Returns array of bucket ids."""
    sig = samples[:, :k]  # (n, k) ∈ {-1, 0, +1}
    digits = (sig + 1).astype(np.int64)  # → {0, 1, 2}
    powers = 3 ** np.arange(k, dtype=np.int64)
    return (digits * powers).sum(axis=1)


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of a non-negative count distribution."""
    if counts.size == 0:
        return 0.0
    sorted_c = np.sort(counts)
    n = sorted_c.size
    cumulative = np.cumsum(sorted_c)
    return float((2 * (np.arange(1, n + 1) * sorted_c).sum()
                  - (n + 1) * cumulative[-1]) / (n * cumulative[-1])) if cumulative[-1] > 0 else 0.0


def measure(M: int, N: int, d: int, k: int, noise: float, seed: int):
    """Run the protocol and return measurements."""
    rng = np.random.default_rng(seed)
    centers = make_clusters(M, d, rng)
    all_samples = []
    all_labels = []
    for c in range(M):
        samples = sample_around(centers[c], N, noise, rng)
        all_samples.append(samples)
        all_labels.append(np.full(N, c))
    samples = np.concatenate(all_samples, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    n_total = M * N

    bucket_ids = hash_to_bucket(samples, k)

    # Collision rates: in-cluster vs cross-cluster
    bucket_to_samples = defaultdict(list)
    for i, b in enumerate(bucket_ids):
        bucket_to_samples[int(b)].append(i)

    in_collisions = 0
    in_pairs = 0
    cross_collisions = 0
    cross_pairs = 0
    for b, idxs in bucket_to_samples.items():
        if len(idxs) < 2:
            continue
        for i, j in itertools.combinations(idxs, 2):
            if labels[i] == labels[j]:
                in_collisions += 1
            else:
                cross_collisions += 1
    # All possible pairs in the dataset
    for c in range(M):
        per_cluster = N
        in_pairs += per_cluster * (per_cluster - 1) // 2
    cross_pairs = n_total * (n_total - 1) // 2 - in_pairs

    in_rate = in_collisions / in_pairs if in_pairs else 0.0
    cross_rate = cross_collisions / cross_pairs if cross_pairs else 0.0

    # Per-bucket dominant-cluster purity (only buckets with ≥1 sample)
    purities = []
    bucket_sizes = []
    for b, idxs in bucket_to_samples.items():
        if not idxs:
            continue
        cluster_counts = Counter(int(labels[i]) for i in idxs)
        purity = max(cluster_counts.values()) / len(idxs)
        purities.append(purity)
        bucket_sizes.append(len(idxs))
    mean_purity = float(np.mean(purities)) if purities else 0.0
    weighted_purity = float(
        np.sum(np.array(purities) * np.array(bucket_sizes)) / np.sum(bucket_sizes)
    ) if bucket_sizes else 0.0

    # Utilization
    n_buckets_used = len(bucket_to_samples)
    n_buckets_possible = 3 ** k
    util_frac = n_buckets_used / n_buckets_possible if n_buckets_possible else 0.0

    # Gini of bucket occupancy
    counts = np.array(bucket_sizes, dtype=float)
    gini_v = gini(counts)

    return {
        "k": k, "M": M, "N": N, "d": d, "noise": noise,
        "n_total": n_total,
        "n_buckets_used": n_buckets_used,
        "n_buckets_possible": n_buckets_possible,
        "utilization_frac": util_frac,
        "gini_of_occupancy": gini_v,
        "in_cluster_collision_rate": in_rate,
        "cross_cluster_collision_rate": cross_rate,
        "in_over_cross_ratio": (in_rate / cross_rate) if cross_rate > 0 else float("inf"),
        "mean_bucket_purity": mean_purity,
        "weighted_bucket_purity": weighted_purity,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=10, help="number of clusters")
    ap.add_argument("--N", type=int, default=100, help="samples per cluster")
    ap.add_argument("--d", type=int, default=128, help="signature dim")
    ap.add_argument("--seed", type=int, default=20260514)
    args = ap.parse_args()

    print(f"M={args.M} clusters × N={args.N} samples = {args.M*args.N} ternary vectors of dim d={args.d}\n")
    print(f"{'k':>3} {'noise':>6}  {'used':>7}/{'total':>7}  {'util%':>6}  "
          f"{'gini':>5}  {'in-coll':>8}  {'x-coll':>8}  {'in/x':>7}  "
          f"{'purity_w':>8}")
    rows = []
    for k in [4, 6, 8, 10, 12, 14]:
        for noise in [0.05, 0.10, 0.20]:
            r = measure(args.M, args.N, args.d, k, noise, args.seed)
            rows.append(r)
            ratio = r["in_over_cross_ratio"]
            ratio_s = f"{ratio:>7.1f}" if ratio != float("inf") else "    inf"
            print(f"{k:>3} {noise:>6.2f}  "
                  f"{r['n_buckets_used']:>7}/{r['n_buckets_possible']:>7}  "
                  f"{r['utilization_frac']*100:>5.2f}%  "
                  f"{r['gini_of_occupancy']:>5.2f}  "
                  f"{r['in_cluster_collision_rate']:>7.4f}  "
                  f"{r['cross_cluster_collision_rate']:>7.4f}  "
                  f"{ratio_s}  "
                  f"{r['weighted_bucket_purity']:>7.3f}")

    # Save
    import json
    out_path = os.path.join(os.path.dirname(__file__),
                             "results/lsh_synth_M{0}_N{1}_d{2}.json".format(args.M, args.N, args.d))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Quick summary
    print(f"\nSummary by k (averaging across noise):")
    for k in [4, 6, 8, 10, 12, 14]:
        ks = [r for r in rows if r["k"] == k]
        avg_purity = np.mean([r["weighted_bucket_purity"] for r in ks])
        avg_ratio  = np.mean([r["in_over_cross_ratio"] for r in ks
                               if r["in_over_cross_ratio"] != float("inf")])
        avg_util   = np.mean([r["utilization_frac"] for r in ks])
        print(f"  k={k:>2}  weighted_purity={avg_purity:.3f}  "
              f"in/cross={avg_ratio:>5.1f}x  util_frac={avg_util*100:.2f}%")


if __name__ == "__main__":
    main()
