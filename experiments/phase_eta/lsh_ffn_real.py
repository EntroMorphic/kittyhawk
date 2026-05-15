"""B1 step 2: real-activation clustering analysis for Trit Lattice LSH FFN.

Loads dumped FFN-input activations, threshold-extracts to trit
signatures (sweep tau), buckets via first-k-trits hash, and measures
clustering quality against:
  (1) k-means pseudo-labels (cluster discovery: do LSH buckets align
      with what k-means finds?)
  (2) prompt-category labels (semantic clustering: do same-category
      prompts share buckets?)

Per-layer separately (early=L2, mid=L15, late=L27) because activation
distributions differ across depth.

Red-team angles addressed:
  - Tau choice: sweep {1000, 2500, 5000, 10000} + adaptive (per-token median absmax)
  - Activation-distribution sanity: print histogram-summary per layer
  - Multiple label schemes (k-means + category) — independent validations
  - Empty-bucket handling: skip in purity calc; report utilization fraction
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

THIS = os.path.dirname(__file__)
DUMP_DIR = os.path.join(THIS, "results/ffn_dump")
OUT_DIR  = os.path.join(THIS, "results")

HIDDEN = 2560  # BITNET_HIDDEN_SIZE


def parse_filename(fn):
    # {label}_p{pos:04d}_l{layer:02d}.bin
    m = re.match(r"^(.+)_p(\d+)_l(\d+)\.bin$", fn)
    if not m: return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def category_of(label):
    return label.split("_")[0] if "_" in label else label


def load_all():
    """Returns dict layer → (labels:list, positions:list, cats:list, acts:np.ndarray)."""
    by_layer = defaultdict(list)
    for fn in os.listdir(DUMP_DIR):
        parsed = parse_filename(fn)
        if not parsed: continue
        label, pos, layer = parsed
        path = os.path.join(DUMP_DIR, fn)
        a = np.fromfile(path, dtype=np.int32)
        if a.shape[0] != HIDDEN:
            print(f"WARN: {fn} has {a.shape[0]} ints (expected {HIDDEN}); skipping")
            continue
        by_layer[layer].append((label, pos, category_of(label), a))
    out = {}
    for layer, recs in by_layer.items():
        labels    = [r[0] for r in recs]
        positions = np.array([r[1] for r in recs])
        cats      = [r[2] for r in recs]
        acts      = np.stack([r[3] for r in recs], axis=0)
        out[layer] = (labels, positions, cats, acts)
    return out


def threshold_extract(acts: np.ndarray, tau) -> np.ndarray:
    """Convert (n, d) int32 activations to (n, d) trit signatures
    {-1, 0, +1}. tau can be int (fixed) or 'adaptive' (per-row median |x|).
    """
    if tau == "adaptive":
        med = np.median(np.abs(acts), axis=1, keepdims=True)
        tau_arr = med
    else:
        tau_arr = np.full((acts.shape[0], 1), tau, dtype=np.int64)
    sig = np.zeros_like(acts, dtype=np.int8)
    sig[acts > tau_arr] = 1
    sig[acts < -tau_arr] = -1
    return sig


def hash_buckets(sig: np.ndarray, k: int) -> np.ndarray:
    """First k trits → base-3 bucket id."""
    sub = sig[:, :k]   # (n, k) ∈ {-1, 0, +1}
    digits = (sub + 1).astype(np.int64)  # → {0, 1, 2}
    powers = 3 ** np.arange(k, dtype=np.int64)
    return (digits * powers).sum(axis=1)


def kmeans_labels(acts: np.ndarray, n_clusters: int, seed: int = 20260514):
    """Simple k-means via NumPy (avoid sklearn dep). Returns labels."""
    rng = np.random.default_rng(seed)
    n, d = acts.shape
    # Init: k-means++ light. Pick first center random, then iteratively
    # pick next as far from existing as possible.
    centers = np.empty((n_clusters, d), dtype=np.float64)
    idx0 = rng.integers(0, n)
    centers[0] = acts[idx0].astype(np.float64)
    for c in range(1, n_clusters):
        # Distance from each point to nearest existing center
        dists = np.min(np.linalg.norm(
            acts[:, None, :].astype(np.float64) - centers[None, :c, :], axis=2),
                       axis=1)
        # Probabilistic far-point pick
        probs = dists ** 2
        probs /= probs.sum() + 1e-12
        idx = rng.choice(n, p=probs)
        centers[c] = acts[idx].astype(np.float64)
    # Lloyd iterations
    labels = np.zeros(n, dtype=np.int32)
    for it in range(50):
        # Assign
        d2 = ((acts.astype(np.float64)[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1)
        if it > 0 and (new_labels == labels).all():
            break
        labels = new_labels
        # Update
        for c in range(n_clusters):
            mask = labels == c
            if mask.any():
                centers[c] = acts[mask].astype(np.float64).mean(axis=0)
    return labels


def measure_purity(buckets: np.ndarray, labels) -> tuple[float, float]:
    """For each populated bucket, dominant-label fraction. Returns
    (mean_purity, weighted_purity)."""
    by_b = defaultdict(list)
    for b, l in zip(buckets, labels):
        by_b[int(b)].append(l)
    purs = []
    sizes = []
    for b, ls in by_b.items():
        c = Counter(ls)
        purs.append(max(c.values()) / len(ls))
        sizes.append(len(ls))
    purs = np.array(purs); sizes = np.array(sizes)
    return float(purs.mean()), float((purs * sizes).sum() / sizes.sum())


def collision_rates(buckets, labels):
    """Pair-level in-class vs cross-class collision rates."""
    n = len(buckets)
    if n < 2: return 0.0, 0.0, 0.0
    # All pairs vs same-bucket pairs vs same-class pairs
    by_b = defaultdict(list)
    by_l = defaultdict(list)
    for i in range(n):
        by_b[int(buckets[i])].append(i)
        by_l[labels[i]].append(i)
    in_pairs = 0; in_collisions = 0
    for ls in by_l.values():
        in_pairs += len(ls) * (len(ls) - 1) // 2
        for i in range(len(ls)):
            for j in range(i + 1, len(ls)):
                if buckets[ls[i]] == buckets[ls[j]]:
                    in_collisions += 1
    cross_pairs = n * (n - 1) // 2 - in_pairs
    total_collisions = sum(len(idxs) * (len(idxs) - 1) // 2 for idxs in by_b.values())
    cross_collisions = total_collisions - in_collisions
    in_rate = in_collisions / in_pairs if in_pairs else 0.0
    cross_rate = cross_collisions / cross_pairs if cross_pairs else 0.0
    return in_rate, cross_rate, (in_rate / cross_rate) if cross_rate > 0 else float("inf")


def gini(counts):
    if len(counts) == 0: return 0.0
    s = np.sort(np.asarray(counts, dtype=float))
    n = s.size
    if s.sum() == 0: return 0.0
    return float((2 * (np.arange(1, n + 1) * s).sum() - (n + 1) * s.sum()) / (n * s.sum()))


def main():
    print("Loading dumps...")
    by_layer = load_all()
    print(f"Layers loaded: {sorted(by_layer.keys())}")
    for layer, (labels, positions, cats, acts) in by_layer.items():
        print(f"  L{layer}: {acts.shape[0]} samples, dim {acts.shape[1]}")
        print(f"    activation stats: mean={acts.mean():.0f} std={acts.std():.0f} "
              f"abs_p50={np.median(np.abs(acts)):.0f} abs_p90={np.quantile(np.abs(acts), 0.9):.0f}")

    summary = {}
    for layer in sorted(by_layer.keys()):
        labels, positions, cats, acts = by_layer[layer]
        n = acts.shape[0]
        print(f"\n{'='*70}\nLayer {layer} — n={n}, {len(set(cats))} categories")
        print(f"{'='*70}")

        # K-means pseudo-labels (run once per layer at moderate K')
        for K_means in (10, 20):
            print(f"\nk-means with K'={K_means} (cluster discovery)")
            km_labels = kmeans_labels(acts, K_means)

            # Sweep tau and k for LSH
            print(f"  {'tau':>10}  {'k':>3}  {'used':>6}  {'gini':>5}  "
                  f"{'in/x (km)':>11}  {'pur_w(km)':>10}  "
                  f"{'in/x (cat)':>11}  {'pur_w(cat)':>10}")
            for tau in (1000, 2500, 5000, 10000, "adaptive"):
                sig = threshold_extract(acts, tau)
                trit_dist = (sig != 0).mean()  # fraction non-zero
                for k in (4, 5, 6, 8, 10):
                    buckets = hash_buckets(sig, k)
                    n_used = len(set(buckets.tolist()))
                    sizes = list(Counter(buckets.tolist()).values())
                    g = gini(sizes)
                    in_km, x_km, ratio_km = collision_rates(buckets, km_labels)
                    in_ct, x_ct, ratio_ct = collision_rates(buckets, cats)
                    _, pur_km = measure_purity(buckets, km_labels)
                    _, pur_ct = measure_purity(buckets, cats)
                    rk = f"{ratio_km:>8.1f}" if ratio_km != float("inf") else "    inf"
                    rc = f"{ratio_ct:>8.1f}" if ratio_ct != float("inf") else "    inf"
                    print(f"  {str(tau):>10}  {k:>3}  {n_used:>6}  {g:>5.2f}  "
                          f"{rk}  {pur_km:>10.3f}  {rc}  {pur_ct:>10.3f}")
            summary[(layer, K_means)] = {"n": n, "n_categories": len(set(cats))}

    out_path = os.path.join(OUT_DIR, "lsh_ffn_real_summary.json")
    with open(out_path, "w") as f:
        json.dump({"layers": list(by_layer.keys()), "results_summary": str(summary)}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
