#!/usr/bin/env python3
"""
Analyze sparsity STRUCTURE in real BitNet weights.

The routed16 atomics analysis revealed that the structural floor is
the per-output-column X-load: dense register-tiles 4 columns sharing
one X-load; routed16's per-column tile sequences can't share. The
question this script answers: do real BitNet weights have correlated
sparsity that a group-wise routed kernel could exploit, or are they
effectively random (uncorrelated 40-50% per-cell zero density)?

Measurements per BitLinear:

  1. Per-column nnz distribution         (uniform vs. clustered cols)
  2. Per-row (K-position) nnz dist        (some positions inherently sparser?)
  3. Pair-wise jaccard similarity         (sampled column pairs)
  4. Conditional P(zero | other-zero)     (correlated vs. independent)
  5. 32-trit window all-zero rate         (single col, block-skip viable?)
  6. 4-col-group window all-zero rate     (group block-skip viable?)
  7. Position-coverage histogram          (k positions w/ many vs. few cols)

Output: per-BitLinear table with random-baseline comparisons. If real
data matches random within noise, group-wise routing has no structure
to exploit. If significantly more structured, the prototype is viable.
"""

import argparse
import os
import struct
import sys

import numpy as np


HIDDEN = 2560
INTERMEDIATE = 6912
KV_PROJ = 640


# Layer-0 BitLinear tensor slots in the substrate blob.
# (matches /tmp/bench_routed_real.c's slot layout)
LAYER0_BITLINEARS = [
    ("q_proj",    0, HIDDEN,       HIDDEN),
    ("k_proj",    1, HIDDEN,       KV_PROJ),
    ("v_proj",    2, HIDDEN,       KV_PROJ),
    ("o_proj",    3, HIDDEN,       HIDDEN),
    ("gate_proj", 4, HIDDEN,       INTERMEDIATE),
    ("up_proj",   5, HIDDEN,       INTERMEDIATE),
    ("down_proj", 6, INTERMEDIATE, HIDDEN),
]


POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)


def decode_5in8(W_packed, K, N):
    """Decode 5-in-8 packed ternary weights into a (N, K) int8 matrix."""
    Kp = (K + 4) // 5
    W = np.frombuffer(W_packed[: N * Kp], dtype=np.uint8).reshape(N, Kp)
    out = np.zeros((N, K), dtype=np.int8)
    for d in range(5):
        u = (W // POW3[d]) % 3  # shape (N, Kp)
        # Place into positions k = b*5 + d
        cols = d + 5 * np.arange(Kp)
        valid = cols < K
        cols = cols[valid]
        u = u[:, : cols.size]
        # u: 0→0, 1→+1, 2→-1
        signed = np.where(u == 1, 1, np.where(u == 2, -1, 0)).astype(np.int8)
        out[:, cols] = signed
    return out


def load_blob(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"M4TB":
        # Best-effort: just trust offsets/sizes
        pass
    # Header: 4 magic + 4 version + 4 lm_head_tied + 4 n_tensors
    version, lm_head_tied, n_tensors = struct.unpack("<iii", data[4:16])
    off = 16
    block_exps = struct.unpack(f"<{n_tensors}i", data[off : off + 4 * n_tensors])
    off += 4 * n_tensors
    offsets = struct.unpack(f"<{n_tensors}Q", data[off : off + 8 * n_tensors])
    off += 8 * n_tensors
    sizes = struct.unpack(f"<{n_tensors}Q", data[off : off + 8 * n_tensors])
    return data, list(zip(block_exps, offsets, sizes))


def get_weight_matrix(data, descriptors, slot, K, N):
    # Layer-0 BitLinear weights start at index 1 (index 0 = embedding).
    idx = 1 + slot
    _, off, size = descriptors[idx]
    W_packed = data[off : off + size]
    return decode_5in8(W_packed, K, N)


def percentile(arr, p):
    return float(np.percentile(arr, p))


def analyze_bitlinear(W, name, K, N, n_pairs=2000, n_groups=2000, group_size=4):
    print(f"\n══ {name}  K={K}  N={N} ══")

    nz_mask = (W != 0)
    sparsity = 1.0 - nz_mask.mean()
    print(f"  overall sparsity:    {100*sparsity:.2f}%")

    # 1. Per-column nnz distribution (output columns)
    nnz_per_col = nz_mask.sum(axis=1)  # shape (N,)
    print(
        f"  nnz per column:      min={nnz_per_col.min()}  "
        f"p10={percentile(nnz_per_col, 10):.0f}  "
        f"p50={percentile(nnz_per_col, 50):.0f}  "
        f"p90={percentile(nnz_per_col, 90):.0f}  "
        f"max={nnz_per_col.max()}  (of {K})"
    )

    # 2. Per-row (per-K-position) nnz distribution
    nnz_per_row = nz_mask.sum(axis=0)  # shape (K,)
    print(
        f"  nnz per K-position:  min={nnz_per_row.min()}  "
        f"p10={percentile(nnz_per_row, 10):.0f}  "
        f"p50={percentile(nnz_per_row, 50):.0f}  "
        f"p90={percentile(nnz_per_row, 90):.0f}  "
        f"max={nnz_per_row.max()}  (of {N})"
    )
    # If random, every K-position would have ~N * (1-sparsity) nonzeros.
    expected_nz_per_row = N * (1 - sparsity)
    sd_nz_per_row = float(nnz_per_row.std())
    expected_sd_random = (N * (1 - sparsity) * sparsity) ** 0.5
    print(
        f"    (expected random:  mean≈{expected_nz_per_row:.0f}  "
        f"sd≈{expected_sd_random:.1f};  observed sd={sd_nz_per_row:.1f})"
    )

    # 3. Pair-wise jaccard similarity (sample n_pairs random pairs of columns)
    rng = np.random.default_rng(seed=42)
    pair_jaccards = []
    pair_intersects = []  # |intersect| / K (= P(both nonzero at same k))
    for _ in range(n_pairs):
        j1, j2 = rng.integers(0, N, size=2)
        if j1 == j2:
            continue
        nz1 = nz_mask[j1]
        nz2 = nz_mask[j2]
        inter = int(np.logical_and(nz1, nz2).sum())
        union = int(np.logical_or(nz1, nz2).sum())
        if union > 0:
            pair_jaccards.append(inter / union)
        pair_intersects.append(inter / K)
    pair_jaccards = np.array(pair_jaccards)
    pair_intersects = np.array(pair_intersects)
    # Random expectation: jaccard = (1-s)² / (1 - s²) where s = sparsity
    s = sparsity
    rand_jaccard = (1 - s) ** 2 / (1 - s ** 2) if s < 1.0 else 0.0
    rand_intersect = (1 - s) ** 2  # P(both nonzero at random k)
    print(f"  pairwise jaccard:    mean={pair_jaccards.mean():.4f}  "
          f"sd={pair_jaccards.std():.4f}  "
          f"(random baseline ≈ {rand_jaccard:.4f})")
    print(f"  pairwise P(both nz): mean={pair_intersects.mean():.4f}  "
          f"sd={pair_intersects.std():.4f}  "
          f"(random baseline ≈ {rand_intersect:.4f})")

    # 4. 32-trit window all-zero rate (per column)
    window = 32
    n_windows = K // window
    if n_windows > 0:
        # Reshape to (N, n_windows, window)
        W_blocks = nz_mask[:, : n_windows * window].reshape(N, n_windows, window)
        block_all_zero = ~W_blocks.any(axis=2)  # shape (N, n_windows)
        single_col_skip_rate = block_all_zero.mean()
        # Random baseline
        rand_block_skip = s ** window
        print(
            f"  32-trit window all-zero (single col): "
            f"{100*single_col_skip_rate:.4f}%  "
            f"(random baseline ≈ {100*rand_block_skip:.6f}%)"
        )

        # 5. Group-of-4 block-skip rate
        # For each random group of group_size cols, what fraction of
        # 32-trit windows are all-zero across the entire group?
        group_skip_rates = []
        for _ in range(n_groups):
            cols = rng.integers(0, N, size=group_size)
            group_blocks = block_all_zero[cols]  # shape (group_size, n_windows)
            all_skip = group_blocks.all(axis=0)  # shape (n_windows,)
            group_skip_rates.append(all_skip.mean())
        group_skip_rates = np.array(group_skip_rates)
        rand_group_skip = rand_block_skip ** group_size
        print(
            f"  32-trit window all-zero ({group_size}-col group): "
            f"mean={100*group_skip_rates.mean():.6f}%  "
            f"max={100*group_skip_rates.max():.6f}%  "
            f"(random baseline ≈ {100*rand_group_skip:.6e}%)"
        )

        # 6. For 4-col group: distribution of "lanes covered per window"
        # = # of K positions in window where AT LEAST one col has a nonzero.
        # If all 4 cols have density ~s, then per K position:
        #   P(at least one of 4 has nonzero) = 1 - s^4
        # Per 32-trit window, expected covered lanes = 32 * (1 - s^4).
        # If covered lanes per window is high, group-wise sharing is hard
        # (most positions need processing); if low, sharing helps.
        cov_per_window = []
        for _ in range(n_groups):
            cols = rng.integers(0, N, size=group_size)
            group_W = nz_mask[cols]  # shape (group_size, K)
            any_nz = group_W.any(axis=0)  # shape (K,) — True where ≥1 col has nz
            blocks = any_nz[: n_windows * window].reshape(n_windows, window)
            covered = blocks.sum(axis=1)  # shape (n_windows,) — lanes covered
            cov_per_window.append(covered.mean())
        cov_per_window = np.array(cov_per_window)
        rand_covered_mean = 32 * (1 - s ** group_size)
        print(
            f"  {group_size}-col group lanes covered per window: "
            f"mean={cov_per_window.mean():.2f}  "
            f"(random baseline ≈ {rand_covered_mean:.2f})"
        )

        # 7. Per-window total nnz across 4 cols (sum of nz across all 4 cols
        # in this window). If high, the group has a lot of work; the X-load
        # sharing benefit is large in absolute terms even if the position
        # set is random.
        total_nnz_per_window = []
        for _ in range(n_groups):
            cols = rng.integers(0, N, size=group_size)
            group_W = nz_mask[cols]  # shape (group_size, K)
            blocks = group_W[:, : n_windows * window].reshape(group_size, n_windows, window)
            total = blocks.sum(axis=(0, 2))  # shape (n_windows,)
            total_nnz_per_window.append(total.mean())
        total_nnz_per_window = np.array(total_nnz_per_window)
        rand_total_nnz = group_size * window * (1 - s)
        print(
            f"  {group_size}-col group total nnz per window: "
            f"mean={total_nnz_per_window.mean():.2f}  "
            f"(random baseline ≈ {rand_total_nnz:.2f})"
        )

    # 8. Verdict heuristic.
    # Structure ratios:
    #   jaccard / random_jaccard
    #   group_skip / random_group_skip
    #   row_sd / expected_random_sd
    j_ratio = pair_jaccards.mean() / rand_jaccard if rand_jaccard > 0 else 1.0
    sd_ratio = sd_nz_per_row / expected_sd_random if expected_sd_random > 0 else 1.0
    print(f"  → structure ratios:  jaccard/random={j_ratio:.3f}  "
          f"row-sd/random={sd_ratio:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blob", help="Path to BitNet substrate weights blob")
    ap.add_argument("--n-pairs", type=int, default=2000)
    ap.add_argument("--n-groups", type=int, default=2000)
    ap.add_argument("--group-size", type=int, default=4)
    args = ap.parse_args()

    if not os.path.exists(args.blob):
        print(f"missing: {args.blob}", file=sys.stderr)
        return 1

    data, descriptors = load_blob(args.blob)
    print(f"loaded {len(descriptors)} tensor descriptors from {args.blob}")
    print(f"sample size: {args.n_pairs} pairs, {args.n_groups} groups of {args.group_size}")

    for name, slot, K, N in LAYER0_BITLINEARS:
        try:
            W = get_weight_matrix(data, descriptors, slot, K, N)
        except Exception as e:
            print(f"\n{name}: load failed — {e}", file=sys.stderr)
            continue
        analyze_bitlinear(W, name, K, N,
                          n_pairs=args.n_pairs,
                          n_groups=args.n_groups,
                          group_size=args.group_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
