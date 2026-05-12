"""Plan A: sweep BITNET_KV_EVICT_M and measure ρ(K-mean proxy, Q-K oracle).

The Phase ζ mechanism finding showed production sigdist (M=1) has
Spearman ρ ≈ +0.055 vs the Q-K oracle Phase ε measured. The harness
already supports M>1, which averages the last M K vectors before
threshold-extracting a direction signature.

Question: does ρ rise with M? If yes, the K-K proxy is fixable in
production via env var (cheap). If not, the proxy is fundamentally
inadequate for this model and Q-aware eviction would be required.

For each (prompt, position, layer, kv_head):
  - Cache = K-sigs at positions [0..p-1]
  - For each M: M-mean K = mean of K at positions [p, p-1, ..., p-M+1],
                threshold-extract → direction sig.
  - L1 distance from M-mean direction sig to each cached K-sig.
  - Spearman ρ vs L1(Q-sig, K-sig) for each q_head.

Reports mean ρ, median ρ, fraction strong (>0.5), top-1 match.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.stats import spearmanr

THIS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(THIS, "..", "phase_alpha"))
sys.path.insert(0, os.path.join(THIS, "..", "phase_epsilon"))

from load_k_signatures import HEAD_DIM, NUM_KV_HEADS, THRESHOLD_TAU, threshold_extract
from eviction_full import load_qkv, substrate_sig, l1_dist, Q_HEADS_PER_KV


M_VALUES = [1, 2, 4, 8, 16]


def mmean_sig(K_seq: np.ndarray, tau: int = THRESHOLD_TAU) -> np.ndarray:
    """Mean K over the M rows (integer division as in production line 560),
    then threshold-extract. Matches bitnet_harness.c lines 546-566."""
    if K_seq.shape[0] == 1:
        return threshold_extract(K_seq[0].astype(np.int32), tau=tau)
    k_mean = K_seq.astype(np.int64).sum(axis=0) // K_seq.shape[0]
    return threshold_extract(k_mean.astype(np.int32), tau=tau)


def main():
    qkv = load_qkv(["data/c_dump_v3"])
    print(f"Loaded prompts: {sorted(qkv.keys())}")

    # corrs[M] = list of per-trial ρ values
    corrs = {m: [] for m in M_VALUES}
    top1_match = {m: [] for m in M_VALUES}
    topk_jaccard = {m: {k: [] for k in (8, 16, 32)} for m in M_VALUES}

    for prompt_id, by_pos in qkv.items():
        positions = sorted(by_pos.keys())
        for p_idx, p in enumerate(positions):
            if p_idx < 8:
                continue
            cache_positions = positions[:p_idx]
            # Walk-backward order from p: [p, p-1, p-2, ...] using available
            # positions in our dump (we have a contiguous range, so the last
            # M positions starting at p are positions[p_idx], positions[p_idx-1], ...
            backward = [p] + list(reversed(cache_positions))

            for layer in by_pos[p]:
                for kv in range(NUM_KV_HEADS):
                    # Build K-cache
                    K_cache_list = [by_pos[cp][layer][kv]["K"]
                                    for cp in cache_positions
                                    if layer in by_pos.get(cp, {})]
                    if len(K_cache_list) < 8:
                        continue
                    K_cache = np.stack(K_cache_list)
                    K_cache_sigs = substrate_sig(K_cache)

                    # For each M, build the M-mean K direction signature
                    dir_sigs = {}
                    for m_val in M_VALUES:
                        m_positions = [bp for bp in backward[:m_val]
                                       if layer in by_pos.get(bp, {})]
                        if not m_positions:
                            continue
                        K_seq = np.stack(
                            [by_pos[bp][layer][kv]["K"] for bp in m_positions]
                        )
                        dir_sigs[m_val] = mmean_sig(K_seq)

                    # K-K distances per M (from direction sig to each cached K-sig)
                    kk_dists = {m_val: l1_dist(dir_sigs[m_val], K_cache_sigs)
                                for m_val in dir_sigs}

                    # Q-K ground truth per q_head
                    Q_heads = by_pos[p][layer][kv]["Q"]
                    for qh in range(Q_HEADS_PER_KV):
                        Q = Q_heads[qh]
                        Q_sig = substrate_sig(Q.reshape(1, -1))[0]
                        qk_dist = l1_dist(Q_sig, K_cache_sigs)

                        for m_val, kk_dist in kk_dists.items():
                            rho, _ = spearmanr(kk_dist, qk_dist)
                            if rho is not None and not np.isnan(rho):
                                corrs[m_val].append(rho)
                            top1_match[m_val].append(
                                int(np.argmax(kk_dist) == np.argmax(qk_dist))
                            )
                            for k_keep in (8, 16, 32):
                                if k_keep > len(qk_dist):
                                    continue
                                kk_keep = set(np.argsort(kk_dist)[:k_keep].tolist())
                                qk_keep = set(np.argsort(qk_dist)[:k_keep].tolist())
                                jacc = len(kk_keep & qk_keep) / len(kk_keep | qk_keep)
                                topk_jaccard[m_val][k_keep].append(jacc)

    print(f"\n{'='*78}")
    print(f"Spearman ρ between K-mean proxy (varying M) and Q-K oracle")
    print(f"{'='*78}")
    print(f"\n{'M':>4} {'N':>8} {'mean ρ':>9} {'median':>9} {'p10':>8} {'p90':>8}"
          f" {'frac>0.5':>10} {'top1':>8}")
    print("-" * 78)
    for m_val in M_VALUES:
        rs = np.array(corrs[m_val])
        n = len(rs)
        if n == 0:
            continue
        print(f"{m_val:>4d} {n:>8d} {rs.mean():>+9.4f} {np.median(rs):>+9.4f} "
              f"{np.percentile(rs, 10):>+8.3f} {np.percentile(rs, 90):>+8.3f} "
              f"{float(np.mean(rs > 0.5)):>10.3f} "
              f"{float(np.mean(top1_match[m_val])):>8.4f}")

    print(f"\n{'='*78}")
    print(f"Top-k KEPT Jaccard (K-mean vs Q-K, lower-dist = kept)")
    print(f"{'='*78}")
    print(f"\n{'M':>4} {'k_keep=8':>10} {'k_keep=16':>11} {'k_keep=32':>11}")
    print("-" * 50)
    for m_val in M_VALUES:
        row = f"{m_val:>4d}"
        for k_keep in (8, 16, 32):
            vals = topk_jaccard[m_val][k_keep]
            if vals:
                row += f" {np.mean(vals):>10.4f}"
            else:
                row += f" {'-':>10}"
        print(row)


if __name__ == "__main__":
    main()
