"""Red-team plan A.

Three augmentations of qk_vs_kk_correlation.py:

  (R1) Baseline rankings for ρ — fifo (rank by position index) and
       random (rank by fixed-seed permutation) — to calibrate "ρ ≈ 0".
       Sigdist's ρ ≈ 0.055 only matters if fifo/random are also ~0.

  (R2) Attention-output L2 error CONSEQUENCE of K-K eviction. Phase ε
       measured L2 error for Hamming, L1 (Q-K), and random selection.
       Here we add K-K (production sigdist) and measure. If K-K's L2
       ≈ random's L2 (≫ Q-K's), the ranking ρ ≈ 0 finding has a
       direct attention-error consequence.

  (R3) Per-layer breakdown of ρ to check for hidden layer-specific
       structure that aggregation hides.
"""
from __future__ import annotations

import os
import sys
import numpy as np
from scipy.stats import spearmanr

THIS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(THIS, "..", "phase_alpha"))
sys.path.insert(0, os.path.join(THIS, "..", "phase_epsilon"))

from load_k_signatures import HEAD_DIM, NUM_KV_HEADS, THRESHOLD_TAU
from eviction_full import (
    load_qkv, substrate_sig, l1_dist, hamming_dist, softmax_stable,
    Q_HEADS_PER_KV,
)


RNG_SEED = 20260512


def evaluate(scores: np.ndarray, V_f: np.ndarray, kept_idx: np.ndarray,
             out_full: np.ndarray) -> float:
    """L2-error of attention output when only `kept_idx` are kept."""
    w_kept = softmax_stable(scores[kept_idx])
    out_kept = V_f[kept_idx].T @ w_kept
    denom = np.linalg.norm(out_full)
    if denom <= 0:
        return float("nan")
    return float(np.linalg.norm(out_kept - out_full) / denom)


def main():
    qkv = load_qkv(["data/c_dump_v3"])
    print(f"Loaded prompts: {sorted(qkv.keys())}")

    rng = np.random.default_rng(RNG_SEED)
    K_KEEPS = (8, 16, 32)

    # R1+R3: ρ vs Q-K oracle for each candidate ranking, per layer
    rhos = {                       # rhos[policy][layer] = list of ρ
        "kk_M1":  {}, "fifo": {}, "random": {},
    }
    # R2: L2 error per policy per k_keep
    l2errs = {p: {k: [] for k in K_KEEPS}
              for p in ("oracle_qk", "kk_M1", "hamming", "random", "fifo")}

    for prompt_id, by_pos in qkv.items():
        positions = sorted(by_pos.keys())
        for p_idx, p in enumerate(positions):
            if p_idx < 8:
                continue
            cache_positions = positions[:p_idx]
            for layer in by_pos[p]:
                for kv in range(NUM_KV_HEADS):
                    K_list = [by_pos[cp][layer][kv]["K"]
                              for cp in cache_positions
                              if layer in by_pos.get(cp, {})]
                    V_list = [by_pos[cp][layer][kv]["V"]
                              for cp in cache_positions
                              if layer in by_pos.get(cp, {})]
                    if len(K_list) < 8:
                        continue
                    K_cache = np.stack(K_list)
                    V_cache = np.stack(V_list)
                    K_sigs  = substrate_sig(K_cache)

                    # Production K-K M=1 (current K-sig is the direction)
                    K_current_sig = substrate_sig(
                        by_pos[p][layer][kv]["K"].reshape(1, -1))[0]
                    kk_dist = l1_dist(K_current_sig, K_sigs)

                    # FIFO: rank by position index (older = higher eviction
                    # score = lower keep score). For "lower-is-keep" we want
                    # distance-like. Position index ascending = oldest first;
                    # to align with "keep low values" we use age (descending
                    # position index) so most-recent has lowest "distance."
                    P = K_cache.shape[0]
                    fifo_score = np.arange(P, 0, -1)  # P, P-1, ..., 1; recent = low

                    # RANDOM: per-trial permutation
                    rand_perm = rng.permutation(P)

                    # Compute dense scores once
                    scale = 1.0 / np.sqrt(HEAD_DIM)
                    Q_heads = by_pos[p][layer][kv]["Q"]
                    for qh in range(Q_HEADS_PER_KV):
                        Q = Q_heads[qh]
                        Q_sig = substrate_sig(Q.reshape(1, -1))[0]
                        qk_dist = l1_dist(Q_sig, K_sigs)

                        # R1+R3: ρ per candidate ranking
                        for name, score in (("kk_M1", kk_dist),
                                             ("fifo", fifo_score),
                                             ("random", rand_perm)):
                            rho, _ = spearmanr(score, qk_dist)
                            if rho is not None and not np.isnan(rho):
                                rhos[name].setdefault(int(layer), []).append(rho)

                        # R2: L2-error consequence
                        Q_f = Q.astype(np.float64)
                        K_f = K_cache.astype(np.float64)
                        V_f = V_cache.astype(np.float64)
                        scores = (K_f @ Q_f) * scale
                        weights = softmax_stable(scores)
                        out_full = V_f.T @ weights

                        for k_keep in K_KEEPS:
                            if k_keep >= P:
                                continue
                            oracle_qk = np.argsort(qk_dist)[:k_keep]
                            kk_M1     = np.argsort(kk_dist)[:k_keep]
                            ham_sig   = hamming_dist(Q_sig, K_sigs)
                            ham_kept  = np.argsort(ham_sig)[:k_keep]
                            rand_idx  = rng.choice(P, size=k_keep, replace=False)
                            fifo_kept = np.argsort(fifo_score)[:k_keep]

                            l2errs["oracle_qk"][k_keep].append(
                                evaluate(scores, V_f, oracle_qk, out_full))
                            l2errs["kk_M1"][k_keep].append(
                                evaluate(scores, V_f, kk_M1, out_full))
                            l2errs["hamming"][k_keep].append(
                                evaluate(scores, V_f, ham_kept, out_full))
                            l2errs["random"][k_keep].append(
                                evaluate(scores, V_f, rand_idx, out_full))
                            l2errs["fifo"][k_keep].append(
                                evaluate(scores, V_f, fifo_kept, out_full))

    # R1: baselines for ρ
    print(f"\n{'='*78}")
    print(f"R1: ρ vs Q-K oracle by ranking, ALL layers pooled")
    print(f"{'='*78}")
    print(f"{'ranking':<10} {'N':>8} {'mean ρ':>9} {'median':>9}"
          f" {'p10':>8} {'p90':>8}")
    print("-" * 60)
    for name in ("kk_M1", "fifo", "random"):
        rs = np.concatenate([np.array(v) for v in rhos[name].values()])
        if len(rs) == 0: continue
        print(f"{name:<10} {len(rs):>8d} {rs.mean():>+9.4f} "
              f"{np.median(rs):>+9.4f} {np.percentile(rs, 10):>+8.3f} "
              f"{np.percentile(rs, 90):>+8.3f}")

    # R3: per-layer
    print(f"\n{'='*78}")
    print(f"R3: ρ vs Q-K oracle per layer (mean across trials)")
    print(f"{'='*78}")
    layers = sorted(set().union(*(set(rhos[p].keys()) for p in rhos)))
    print(f"{'layer':>5}  {'kk_M1':>9}  {'fifo':>9}  {'random':>9}")
    print("-" * 50)
    for layer in layers:
        row = f"{layer:>5d} "
        for name in ("kk_M1", "fifo", "random"):
            vals = rhos[name].get(layer, [])
            if vals:
                row += f"  {np.mean(vals):>+9.4f}"
            else:
                row += f"  {'-':>9}"
        print(row)

    # R2: L2-error consequence
    print(f"\n{'='*78}")
    print(f"R2: Attention-output L2 error (mean across trials)")
    print(f"{'='*78}")
    print(f"  Lower = closer to no-eviction. Phase ε reported "
          f"oracle Q-K = 0.016, random = 0.584 at k_keep=32.")
    print(f"\n{'policy':<12} {'k_keep=8':>10} {'k_keep=16':>11} {'k_keep=32':>11}")
    print("-" * 50)
    for policy in ("oracle_qk", "hamming", "kk_M1", "fifo", "random"):
        row = f"{policy:<12}"
        for k_keep in K_KEEPS:
            vals = l2errs[policy][k_keep]
            if vals:
                row += f" {np.mean(vals):>10.4f}"
            else:
                row += f" {'-':>10}"
        print(row)

    # Headline numbers
    kk_l2 = np.mean(l2errs["kk_M1"][32]) if l2errs["kk_M1"][32] else float("nan")
    rnd_l2 = np.mean(l2errs["random"][32]) if l2errs["random"][32] else float("nan")
    oracle_l2 = np.mean(l2errs["oracle_qk"][32]) if l2errs["oracle_qk"][32] else float("nan")
    print(f"\nHeadline (k_keep=32):")
    print(f"  oracle Q-K   L2 = {oracle_l2:.4f}")
    print(f"  K-K (M=1)    L2 = {kk_l2:.4f}")
    print(f"  random       L2 = {rnd_l2:.4f}")
    if not np.isnan(kk_l2 - rnd_l2):
        ratio = kk_l2 / rnd_l2 if rnd_l2 > 0 else float("nan")
        print(f"  K-K / random = {ratio:.3f}  (= 1.0 means K-K ≈ random)")


if __name__ == "__main__":
    main()
