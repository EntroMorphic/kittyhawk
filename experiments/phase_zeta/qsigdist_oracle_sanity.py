"""Sanity check: Python reference for the production QSIGDIST eviction.

Production QSIGDIST (bitnet_harness.c lines 605-660):
  For each candidate position p in cache (excluding current_position):
    cost[p] = Σ over all 20 Q-heads of L1(Q-sig_qh, K-sig at p for qh's kv-head)
  Evict argmax(cost[p]).

This Python simulation reproduces the same decision and measures the
attention-output L2 error of the resulting eviction selection
(k_keep = P - 1, drop one position at a time, single shot for
each (prompt, position, layer) decision).

Compare against:
  - oracle Q-K per-q-head (Phase ε's L1 policy, optimal per Q-head)
  - random
  - K-K M=1 (production sigdist)

This confirms whether the consensus-across-Q-heads loses to per-Q-head
selection, and whether QSIGDIST should still beat random and K-K in
the single-shot oracle.
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

    rng = np.random.default_rng(20260512)

    # We compute per-LAYER eviction decisions (mimicking production):
    # at each (prompt, position, layer), all 5 KV-heads share one eviction
    # victim. The cost is summed across all 20 Q-heads (across all KV-heads).
    # We then measure the average per-q-head attention-output L2 error.

    K_KEEPS = (8, 16, 32)
    l2errs = {p: {k: [] for k in K_KEEPS}
              for p in ("oracle_qk_perhead", "qsigdist", "kk_M1", "random")}

    for prompt_id, by_pos in qkv.items():
        positions = sorted(by_pos.keys())
        for p_idx, p in enumerate(positions):
            if p_idx < 8:
                continue
            cache_positions = positions[:p_idx]
            for layer in by_pos[p]:
                # Build per-KV cache + Q + K-sigs
                K_caches = {}
                K_sigs_by_kv = {}
                V_caches = {}
                Q_heads_by_kv = {}
                K_current_sigs = {}
                for kv in range(NUM_KV_HEADS):
                    K_list = [by_pos[cp][layer][kv]["K"]
                              for cp in cache_positions
                              if layer in by_pos.get(cp, {})]
                    V_list = [by_pos[cp][layer][kv]["V"]
                              for cp in cache_positions
                              if layer in by_pos.get(cp, {})]
                    if len(K_list) < 8:
                        continue
                    K_caches[kv] = np.stack(K_list)
                    V_caches[kv] = np.stack(V_list)
                    K_sigs_by_kv[kv] = substrate_sig(K_caches[kv])
                    K_current_sigs[kv] = substrate_sig(
                        by_pos[p][layer][kv]["K"].reshape(1, -1))[0]
                    Q_heads_by_kv[kv] = by_pos[p][layer][kv]["Q"]
                if len(K_caches) != NUM_KV_HEADS:
                    continue

                P = K_caches[0].shape[0]
                if P < 8: continue

                # === QSIGDIST: layer-global Q-aware cost ===
                # cost[p] = Σ_kv Σ_qh L1(Q-sig_qh, K-sig_p^kv)
                qsigdist_cost = np.zeros(P, dtype=np.int64)
                for kv in range(NUM_KV_HEADS):
                    for qh in range(Q_HEADS_PER_KV):
                        Q = Q_heads_by_kv[kv][qh]
                        Q_sig = substrate_sig(Q.reshape(1, -1))[0]
                        qsigdist_cost += l1_dist(Q_sig, K_sigs_by_kv[kv])

                # === KK_M1 (production sigdist): layer-global K-K cost ===
                kk_cost = np.zeros(P, dtype=np.int64)
                for kv in range(NUM_KV_HEADS):
                    kk_cost += l1_dist(K_current_sigs[kv], K_sigs_by_kv[kv])

                for k_keep in K_KEEPS:
                    if k_keep >= P: continue
                    # Layer-shared eviction selections
                    qsig_keep = np.argsort(qsigdist_cost)[:k_keep]
                    kk_keep   = np.argsort(kk_cost)[:k_keep]
                    rand_keep = rng.choice(P, size=k_keep, replace=False)

                    # Measure attention-output L2 per (kv, qh) and average
                    for kv in range(NUM_KV_HEADS):
                        K_cache = K_caches[kv]
                        V_cache = V_caches[kv]
                        K_sigs  = K_sigs_by_kv[kv]
                        scale = 1.0 / np.sqrt(HEAD_DIM)
                        for qh in range(Q_HEADS_PER_KV):
                            Q = Q_heads_by_kv[kv][qh]
                            Q_sig = substrate_sig(Q.reshape(1, -1))[0]
                            Q_f = Q.astype(np.float64)
                            K_f = K_cache.astype(np.float64)
                            V_f = V_cache.astype(np.float64)
                            scores = (K_f @ Q_f) * scale
                            weights = softmax_stable(scores)
                            out_full = V_f.T @ weights

                            def err(kept):
                                w_kept = softmax_stable(scores[kept])
                                out_kept = V_f[kept].T @ w_kept
                                denom = np.linalg.norm(out_full)
                                if denom <= 0: return float("nan")
                                return float(np.linalg.norm(out_kept - out_full) / denom)

                            # Oracle per-q-head (best achievable for this q_head)
                            qk = l1_dist(Q_sig, K_sigs)
                            oracle_keep = np.argsort(qk)[:k_keep]
                            l2errs["oracle_qk_perhead"][k_keep].append(err(oracle_keep))
                            l2errs["qsigdist"][k_keep].append(err(qsig_keep))
                            l2errs["kk_M1"][k_keep].append(err(kk_keep))
                            l2errs["random"][k_keep].append(err(rand_keep))

    print(f"\n{'='*78}")
    print(f"Single-shot attention-output L2 error (mean per q-head trial)")
    print(f"{'='*78}")
    print(f"\n{'policy':<22} {'k_keep=8':>10} {'k_keep=16':>11} {'k_keep=32':>11}")
    print("-" * 60)
    for policy in ("oracle_qk_perhead", "qsigdist", "kk_M1", "random"):
        row = f"{policy:<22}"
        for k_keep in K_KEEPS:
            vals = l2errs[policy][k_keep]
            if vals:
                row += f" {np.mean(vals):>10.4f}"
            else:
                row += f" {'-':>10}"
        print(row)

    qsig_l2 = np.mean(l2errs["qsigdist"][32])
    oracle_l2 = np.mean(l2errs["oracle_qk_perhead"][32])
    kk_l2 = np.mean(l2errs["kk_M1"][32])
    rand_l2 = np.mean(l2errs["random"][32])
    print(f"\nHeadline (k_keep=32):")
    print(f"  oracle per-q-head L2 = {oracle_l2:.4f}  (Phase ε's per-q-head optimum)")
    print(f"  QSIGDIST (layer-consensus) L2 = {qsig_l2:.4f}")
    print(f"  K-K (production sigdist) L2 = {kk_l2:.4f}")
    print(f"  random L2 = {rand_l2:.4f}")
    print(f"\n  QSIGDIST / oracle = {qsig_l2/oracle_l2:.2f}× (consensus penalty)")
    print(f"  QSIGDIST / random = {qsig_l2/rand_l2:.3f}× "
          f"({'beats random' if qsig_l2 < rand_l2 else 'loses to random'})")


if __name__ == "__main__":
    main()
