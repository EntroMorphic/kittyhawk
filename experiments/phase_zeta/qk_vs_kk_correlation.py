"""Atomic test: does the production K-K eviction criterion approximate
Phase ε's Q-K oracle criterion?

Phase ε ranked candidate-K's by L1(Q-sig, K-sig).
Production sigdist ranks them by L1(K-sig[current], K-sig[i]).

If the two rankings disagree, the per-Q-head L2 advantage Phase ε
demonstrated for Q-K-L1 selection does NOT speak to what production
sigdist actually does. The Phase ζ proxy-to-territory gap then
includes a "different operation entirely" component, not just an
end-to-end-noise component.

Spearman correlation of the two rankings, averaged across positions,
layers, kv_heads, and prompts. If correlation is ~0 or ~1, we know.
"""
from __future__ import annotations

import os
import sys
import glob
import numpy as np
from scipy.stats import spearmanr

THIS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(THIS, "..", "phase_alpha"))
sys.path.insert(0, os.path.join(THIS, "..", "phase_epsilon"))

from load_k_signatures import (
    HEAD_DIM, NUM_KV_HEADS, NUM_HIDDEN, THRESHOLD_TAU,
    read_actv2, parse_filename, threshold_extract,
)
from eviction_full import load_qkv, substrate_sig, l1_dist, Q_HEADS_PER_KV


def main():
    qkv = load_qkv(["data/c_dump_v3"])
    print(f"Loaded prompts: {sorted(qkv.keys())}")

    all_corrs = []          # rank-correlation per (prompt, position, layer, kv, q_head)
    all_top1_overlap = []   # did the K-K eviction target also rank top in Q-K?
    all_topk_jaccard = {}   # for k_keep in {8, 16, 32}: Jaccard of kept sets

    for k_keep in (8, 16, 32):
        all_topk_jaccard[k_keep] = []

    for prompt_id, by_pos in qkv.items():
        positions = sorted(by_pos.keys())
        for p_idx, p in enumerate(positions):
            if p_idx < 8:  # need a non-trivial cache
                continue
            cache_positions = positions[:p_idx]
            for layer in by_pos[p]:
                for kv in range(NUM_KV_HEADS):
                    # K-sigs of all cached positions
                    K_cache_list = [by_pos[cp][layer][kv]["K"]
                                    for cp in cache_positions
                                    if layer in by_pos.get(cp, {})]
                    if len(K_cache_list) < 8:
                        continue
                    K_cache = np.stack(K_cache_list)
                    K_cache_sigs = substrate_sig(K_cache)  # (P, HEAD_DIM)

                    # Current-position K-sig (the production direction proxy)
                    K_current = by_pos[p][layer][kv]["K"]
                    K_current_sig = substrate_sig(K_current.reshape(1, -1))[0]

                    # Production K-K distances (lower = MORE similar to current K;
                    # production EVICTS those with HIGHEST L1, equivalently KEEPS
                    # those with LOWEST L1 to current K).
                    kk_dist = l1_dist(K_current_sig, K_cache_sigs)

                    # Phase-ε Q-K distance per q_head
                    Q_heads = by_pos[p][layer][kv]["Q"]
                    for qh in range(Q_HEADS_PER_KV):
                        Q = Q_heads[qh]
                        Q_sig = substrate_sig(Q.reshape(1, -1))[0]
                        qk_dist = l1_dist(Q_sig, K_cache_sigs)

                        # Spearman: do K-K and Q-K rank the cache the same way?
                        # rho near +1 means production sigdist ≈ Phase-ε oracle.
                        # rho near 0 means they rank ~independently.
                        rho, _ = spearmanr(kk_dist, qk_dist)
                        if rho is not None and not np.isnan(rho):
                            all_corrs.append(rho)

                        # Top-1 OVERLAP at the "evict" end:
                        #   production evicts argmax(kk_dist)
                        #   Phase-ε would evict argmax(qk_dist)
                        all_top1_overlap.append(
                            int(np.argmax(kk_dist) == np.argmax(qk_dist))
                        )

                        # Top-k KEPT set Jaccard:
                        #   production keeps argsort(kk_dist)[:k_keep] (lowest dist)
                        #   Phase-ε keeps argsort(qk_dist)[:k_keep]
                        for k_keep in (8, 16, 32):
                            if k_keep > len(qk_dist):
                                continue
                            kk_keep = set(np.argsort(kk_dist)[:k_keep].tolist())
                            qk_keep = set(np.argsort(qk_dist)[:k_keep].tolist())
                            jacc = len(kk_keep & qk_keep) / len(kk_keep | qk_keep)
                            all_topk_jaccard[k_keep].append(jacc)

    print(f"\nTotal trials: {len(all_corrs)}")
    print(f"\nSpearman ρ between K-K and Q-K rankings:")
    print(f"  mean = {np.mean(all_corrs):+.4f}")
    print(f"  median = {np.median(all_corrs):+.4f}")
    print(f"  std = {np.std(all_corrs):.4f}")
    print(f"  10th/50th/90th pct = "
          f"{np.percentile(all_corrs, 10):+.3f} / "
          f"{np.percentile(all_corrs, 50):+.3f} / "
          f"{np.percentile(all_corrs, 90):+.3f}")
    pct_strong = float(np.mean(np.array(all_corrs) > 0.5))
    pct_near0 = float(np.mean(np.abs(np.array(all_corrs)) < 0.1))
    print(f"  fraction ρ > 0.5: {pct_strong:.3f}")
    print(f"  fraction |ρ| < 0.1: {pct_near0:.3f}")

    print(f"\nTop-1 eviction-target match (K-K argmax == Q-K argmax):")
    print(f"  rate = {np.mean(all_top1_overlap):.4f} "
          f"(uniform-random expectation ≈ 1/cache_size)")

    print(f"\nTop-k KEPT set Jaccard (K-K vs Q-K with same k_keep):")
    for k_keep in (8, 16, 32):
        vals = all_topk_jaccard[k_keep]
        if vals:
            print(f"  k_keep={k_keep}: mean Jaccard = {np.mean(vals):.4f}  "
                  f"(N={len(vals)})")


if __name__ == "__main__":
    main()
