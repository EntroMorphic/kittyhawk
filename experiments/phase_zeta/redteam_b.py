"""Red-team plan B.

Two augmentations to validate (or kill) the plan B single-shot result:

  (B1) Q-sig sparsity at τ=5000. Phase ε uses τ=5000 chosen for K-sig
       distribution (~62% nonzero). If Q-sigs at the same τ are
       degenerate (all-zero or all-nonzero), the L1(Q-sig, K-sig)
       metric loses signal and qsigdist's "informed" decision would
       be effectively random. Compare distributions.

  (B2) Drop-one-at-a-time single-shot L2. The harness drops ONE
       position per layer per step. My earlier sanity used k_keep ∈
       {8,16,32} (drop many). Redo at k_drop=1 from cache states
       near the harness's window=16 operating point. If qsigdist's
       10× win shrinks to ~0 at k_drop=1, the gap is at the
       single-decision level (and my "trajectory diversity"
       hypothesis is wrong / incomplete). If qsigdist still wins by
       large factor at k_drop=1, the trajectory hypothesis stands.
"""
from __future__ import annotations

import os
import sys
import numpy as np

THIS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(THIS, "..", "phase_alpha"))
sys.path.insert(0, os.path.join(THIS, "..", "phase_epsilon"))

from load_k_signatures import HEAD_DIM, NUM_KV_HEADS, THRESHOLD_TAU
from eviction_full import load_qkv, substrate_sig, l1_dist, softmax_stable, Q_HEADS_PER_KV


def main():
    qkv = load_qkv(["data/c_dump_v3"])
    print(f"Loaded prompts: {sorted(qkv.keys())}")

    # ===== B1: Q-sig vs K-sig sparsity =====
    print(f"\n{'='*70}")
    print(f"B1: Q-sig vs K-sig sparsity at τ={THRESHOLD_TAU}")
    print(f"{'='*70}")
    q_frac_nz, k_frac_nz = [], []
    q_frac_minus, q_frac_zero, q_frac_plus = [], [], []
    k_frac_minus, k_frac_zero, k_frac_plus = [], [], []
    for prompt_id, by_pos in qkv.items():
        for p in by_pos:
            for layer in by_pos[p]:
                for kv in range(NUM_KV_HEADS):
                    K = by_pos[p][layer][kv]["K"]
                    K_sig = substrate_sig(K.reshape(1, -1))[0]
                    k_frac_nz.append(float(np.mean(K_sig != 0)))
                    k_frac_minus.append(float(np.mean(K_sig == -1)))
                    k_frac_zero.append(float(np.mean(K_sig == 0)))
                    k_frac_plus.append(float(np.mean(K_sig == +1)))
                    for qh in range(Q_HEADS_PER_KV):
                        Q = by_pos[p][layer][kv]["Q"][qh]
                        Q_sig = substrate_sig(Q.reshape(1, -1))[0]
                        q_frac_nz.append(float(np.mean(Q_sig != 0)))
                        q_frac_minus.append(float(np.mean(Q_sig == -1)))
                        q_frac_zero.append(float(np.mean(Q_sig == 0)))
                        q_frac_plus.append(float(np.mean(Q_sig == +1)))
    print(f"\nK-sig (cached during attention):")
    print(f"  -1: {np.mean(k_frac_minus):.3f}   "
          f"0: {np.mean(k_frac_zero):.3f}   "
          f"+1: {np.mean(k_frac_plus):.3f}   nonzero: {np.mean(k_frac_nz):.3f}")
    print(f"  (Phase γ measured ~62% nonzero for K-sigs at τ=5000)")
    print(f"\nQ-sig (would be computed by qsigdist):")
    print(f"  -1: {np.mean(q_frac_minus):.3f}   "
          f"0: {np.mean(q_frac_zero):.3f}   "
          f"+1: {np.mean(q_frac_plus):.3f}   nonzero: {np.mean(q_frac_nz):.3f}")
    qk_ratio = np.mean(q_frac_nz) / np.mean(k_frac_nz) if np.mean(k_frac_nz) > 0 else float('nan')
    print(f"\n  Q/K nonzero ratio = {qk_ratio:.3f}  "
          f"(1.0 = same regime; ≪1 = Q degenerate-zero; ≫1 = K degenerate-zero)")
    print(f"  Q-sig distribution p10/p50/p90 nonzero: "
          f"{np.percentile(q_frac_nz, 10):.3f} / "
          f"{np.percentile(q_frac_nz, 50):.3f} / "
          f"{np.percentile(q_frac_nz, 90):.3f}")

    # ===== B2: Drop-one-at-a-time L2 =====
    print(f"\n{'='*70}")
    print(f"B2: Drop-ONE-at-a-time single-shot L2")
    print(f"{'='*70}")
    print(f"  Mimics harness eviction granularity: pick ONE position to evict")
    print(f"  from a cache of size N, measure resulting per-q-head L2.")
    print(f"  Use N ∈ {{16, 32, 48}} matching harness window=16 operating")
    print(f"  states (cache transiently 17,18,...,window+1 before evict).")

    rng = np.random.default_rng(20260512)
    N_VALUES = [16, 32, 48]
    l2errs = {p: {n: [] for n in N_VALUES}
              for p in ("qsigdist", "kk_M1", "random", "fifo_oldest")}

    for prompt_id, by_pos in qkv.items():
        positions = sorted(by_pos.keys())
        for p_idx, p in enumerate(positions):
            if p_idx < 16:
                continue  # need at least 16 cache positions
            cache_positions = positions[:p_idx]
            for layer in by_pos[p]:
                # Build all KV heads' caches once per layer
                K_caches, V_caches, K_sigs_by_kv, Q_heads_by_kv, K_curr_sigs = \
                    {}, {}, {}, {}, {}
                for kv in range(NUM_KV_HEADS):
                    K_list = [by_pos[cp][layer][kv]["K"]
                              for cp in cache_positions
                              if layer in by_pos.get(cp, {})]
                    V_list = [by_pos[cp][layer][kv]["V"]
                              for cp in cache_positions
                              if layer in by_pos.get(cp, {})]
                    if len(K_list) < 16: continue
                    K_caches[kv] = np.stack(K_list)
                    V_caches[kv] = np.stack(V_list)
                    K_sigs_by_kv[kv] = substrate_sig(K_caches[kv])
                    K_curr_sigs[kv] = substrate_sig(
                        by_pos[p][layer][kv]["K"].reshape(1, -1))[0]
                    Q_heads_by_kv[kv] = by_pos[p][layer][kv]["Q"]
                if len(K_caches) != NUM_KV_HEADS: continue

                P = K_caches[0].shape[0]
                if P < 16: continue

                # For each N: take the LAST N positions as the cache state
                # at that operating point. Evict ONE from the N candidates.
                for N in N_VALUES:
                    if P < N: continue
                    # Use the most recent N positions (mimic harness)
                    keep_window = np.arange(P - N, P)

                    # === qsigdist: layer-global Q-aware cost ===
                    qsig_cost = np.zeros(N, dtype=np.int64)
                    for kv in range(NUM_KV_HEADS):
                        K_sigs_window = K_sigs_by_kv[kv][keep_window]
                        for qh in range(Q_HEADS_PER_KV):
                            Q = Q_heads_by_kv[kv][qh]
                            Q_sig = substrate_sig(Q.reshape(1, -1))[0]
                            qsig_cost += l1_dist(Q_sig, K_sigs_window)
                    # Evict argmax in WINDOW indexing → drop that position
                    qsig_drop = int(np.argmax(qsig_cost))

                    # === K-K M=1 cost ===
                    kk_cost = np.zeros(N, dtype=np.int64)
                    for kv in range(NUM_KV_HEADS):
                        K_sigs_window = K_sigs_by_kv[kv][keep_window]
                        kk_cost += l1_dist(K_curr_sigs[kv], K_sigs_window)
                    kk_drop = int(np.argmax(kk_cost))

                    # === Random ===
                    rand_drop = int(rng.integers(0, N))

                    # === FIFO-oldest ===
                    fifo_drop = 0  # within the keep_window, index 0 = oldest

                    for kv in range(NUM_KV_HEADS):
                        K_cache_full = K_caches[kv]
                        V_cache_full = V_caches[kv]
                        # restrict to the operating-point window
                        K_win = K_cache_full[keep_window]
                        V_win = V_cache_full[keep_window]
                        scale = 1.0 / np.sqrt(HEAD_DIM)
                        for qh in range(Q_HEADS_PER_KV):
                            Q = Q_heads_by_kv[kv][qh]
                            Q_f = Q.astype(np.float64)
                            K_f = K_win.astype(np.float64)
                            V_f = V_win.astype(np.float64)
                            scores = (K_f @ Q_f) * scale
                            weights = softmax_stable(scores)
                            out_full = V_f.T @ weights
                            denom = np.linalg.norm(out_full)
                            if denom <= 0: continue
                            def err_with_drop(drop_idx):
                                keep = np.array([i for i in range(N) if i != drop_idx])
                                w_kept = softmax_stable(scores[keep])
                                out_kept = V_f[keep].T @ w_kept
                                return float(np.linalg.norm(out_kept - out_full) / denom)
                            l2errs["qsigdist"][N].append(err_with_drop(qsig_drop))
                            l2errs["kk_M1"][N].append(err_with_drop(kk_drop))
                            l2errs["random"][N].append(err_with_drop(rand_drop))
                            l2errs["fifo_oldest"][N].append(err_with_drop(fifo_drop))

    print(f"\n  Mean per-q-head L2 error from dropping ONE position:")
    print(f"  {'policy':<14} {'N=16':>10} {'N=32':>10} {'N=48':>10}")
    print("-" * 50)
    for policy in ("qsigdist", "kk_M1", "random", "fifo_oldest"):
        row = f"  {policy:<14}"
        for N in N_VALUES:
            vals = l2errs[policy][N]
            if vals:
                row += f" {np.mean(vals):>10.5f}"
            else:
                row += f" {'-':>10}"
        print(row)

    # Headline
    qsig_n32 = np.mean(l2errs["qsigdist"][32]) if l2errs["qsigdist"][32] else float("nan")
    rand_n32 = np.mean(l2errs["random"][32]) if l2errs["random"][32] else float("nan")
    if not np.isnan(qsig_n32) and rand_n32 > 0:
        print(f"\n  Headline (N=32, drop 1):")
        print(f"    qsigdist L2 / random L2 = {qsig_n32 / rand_n32:.3f}× "
              f"({'BEATS random' if qsig_n32 < rand_n32 else 'LOSES to random'})")
        print(f"    (Earlier drop-MANY result at k_keep=32 from cache≤63: "
              f"qsigdist/random = 0.104×)")


if __name__ == "__main__":
    main()
