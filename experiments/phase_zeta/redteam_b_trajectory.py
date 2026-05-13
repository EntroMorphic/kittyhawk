"""Red-team B follow-up: trajectory simulation.

B2 killed the "single-shot vs trajectory" hypothesis — qsigdist wins
by 5-7× at the harness's drop-one granularity. So the harness loss
must come from inter-decision consequences: qsigdist's choice at
step T modifies the cache state for step T+1's decision, and the
compounded effect over many steps may differ from random.

This script simulates the trajectory:
  - Start with cache = K[0..W-1] (W = window size, e.g. 16).
  - For each subsequent position p ∈ [W, N-1]:
    - Add K[p] to cache.
    - Apply policy to drop ONE position.
    - Q[p] is read from the dump (Phase ε's Q at that position).
  - At each step, measure per-q-head L2 error of the policy's cache's
    attention output vs the FULL (no-eviction) cache's attention
    output at that step.

Compare cumulative L2 growth across qsigdist vs random vs fifo.

If qsigdist's cumulative L2 grows linearly while random's grows
sublinearly (cancellation), the "correlated drift" hypothesis is
confirmed. If they grow at similar rates, the mechanism is
something else.

Key subtlety: Q[p] at each step is read from the no_evict dump.
That's the Q that the model WOULD compute if the cache were the
no_evict cache. In the actual harness, qsigdist's diverged cache
would produce a DIFFERENT Q at step p+1 (because attention at step
p uses the evicted cache → different x_{p+1} → different Q). This
Python simulation does NOT model that secondary drift — it assumes
Q stays the same as no_evict's Q. This is an UPPER BOUND on
qsigdist's quality; the real harness loss could be even larger
due to Q-drift.
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


WINDOW = 16


def attn_l2(Q: np.ndarray, K_kept: np.ndarray, V_kept: np.ndarray,
            K_full: np.ndarray, V_full: np.ndarray) -> float:
    """L2 of attn output with kept cache vs full cache."""
    scale = 1.0 / np.sqrt(HEAD_DIM)
    Q_f = Q.astype(np.float64)
    K_full_f = K_full.astype(np.float64)
    V_full_f = V_full.astype(np.float64)
    scores_full = (K_full_f @ Q_f) * scale
    weights_full = softmax_stable(scores_full)
    out_full = V_full_f.T @ weights_full
    K_k = K_kept.astype(np.float64)
    V_k = V_kept.astype(np.float64)
    scores_k = (K_k @ Q_f) * scale
    weights_k = softmax_stable(scores_k)
    out_k = V_k.T @ weights_k
    denom = np.linalg.norm(out_full)
    if denom <= 0: return float("nan")
    return float(np.linalg.norm(out_k - out_full) / denom)


def simulate_trajectory(policy_pick, K_seq, V_seq, K_sig_seq, Q_seq, window):
    """policy_pick(cache_idx_in_seq, k_sigs_window, q_all_heads) → window-relative idx to drop.

    Returns: list of per-step L2 errors (averaged across q-heads).
    """
    N = K_seq.shape[0]
    # Start: cache = first `window` positions
    cache_idx = list(range(window))
    step_l2s = []

    for p in range(window, N):
        # Add K[p] to cache
        cache_idx.append(p)
        # Pick eviction within this expanded cache (size window+1)
        drop_rel = policy_pick(cache_idx, K_sig_seq, Q_seq[p])
        # Drop that position
        cache_idx.pop(drop_rel)
        # Now cache has `window` positions. Measure attn L2 against full
        # cache (positions 0..p inclusive).
        K_kept = K_seq[cache_idx]
        V_kept = V_seq[cache_idx]
        K_full = K_seq[:p + 1]
        V_full = V_seq[:p + 1]
        # Avg L2 across 4 q-heads (Q_seq[p] is shape (4, HEAD_DIM))
        per_qh = [attn_l2(Q_seq[p][qh], K_kept, V_kept, K_full, V_full)
                  for qh in range(Q_HEADS_PER_KV)]
        step_l2s.append(float(np.mean(per_qh)))
    return step_l2s


def policy_qsigdist(cache_idx, K_sig_seq, Q_heads_all):
    """Q-aware: sum L1(Q_qh, K_p) over 4 q-heads, evict argmax."""
    K_sigs_window = K_sig_seq[cache_idx]
    cost = np.zeros(len(cache_idx), dtype=np.int64)
    for qh in range(Q_HEADS_PER_KV):
        Q_sig = substrate_sig(Q_heads_all[qh].reshape(1, -1))[0]
        cost += l1_dist(Q_sig, K_sigs_window)
    return int(np.argmax(cost))


def policy_kk(cache_idx, K_sig_seq, Q_heads_all):
    """Production sigdist: evict K most distant from current (newest) K-sig."""
    K_sigs_window = K_sig_seq[cache_idx]
    K_curr_sig = K_sig_seq[cache_idx[-1]]  # newest K-sig
    cost = l1_dist(K_curr_sig, K_sigs_window)
    # don't evict current_position (last in cache_idx); set its cost to -1
    cost[-1] = -1
    return int(np.argmax(cost))


def policy_random(cache_idx, K_sig_seq, Q_heads_all, rng=np.random.default_rng(42)):
    """Random within the window, excluding the just-added position."""
    n = len(cache_idx)
    return int(rng.integers(0, n - 1))  # exclude last (current_position)


def policy_fifo(cache_idx, K_sig_seq, Q_heads_all):
    """Drop oldest (index 0 in the window)."""
    return 0


def main():
    qkv = load_qkv(["data/c_dump_v3"])
    print(f"Loaded prompts: {sorted(qkv.keys())}")
    print(f"Window = {WINDOW}\n")

    cumulative_by_policy = {p: [] for p in ("qsigdist", "kk_M1", "random", "fifo")}
    last_step_l2_by_policy = {p: [] for p in ("qsigdist", "kk_M1", "random", "fifo")}
    per_step_means = {p: None for p in cumulative_by_policy}  # mean L2 trajectory

    rng = np.random.default_rng(20260512)

    trial_count = 0
    for prompt_id, by_pos in qkv.items():
        positions = sorted(by_pos.keys())
        N = len(positions)
        if N <= WINDOW: continue
        for layer in by_pos[positions[0]]:
            for kv in range(NUM_KV_HEADS):
                # Build per-layer per-kv sequences
                try:
                    K_seq = np.stack([by_pos[p][layer][kv]["K"] for p in positions])
                    V_seq = np.stack([by_pos[p][layer][kv]["V"] for p in positions])
                    Q_seq = np.stack([by_pos[p][layer][kv]["Q"] for p in positions])
                except KeyError:
                    continue
                K_sig_seq = substrate_sig(K_seq)

                # Per-policy trajectory
                trial_count += 1
                # For deterministic random within this trial:
                rand_rng = np.random.default_rng(20260512 + trial_count)
                def policy_random_local(cache_idx, K_sig_seq, Q_heads_all):
                    n = len(cache_idx)
                    return int(rand_rng.integers(0, n - 1))
                policies = {
                    "qsigdist": policy_qsigdist,
                    "kk_M1":    policy_kk,
                    "random":   policy_random_local,
                    "fifo":     policy_fifo,
                }
                for name, fn in policies.items():
                    step_l2s = simulate_trajectory(fn, K_seq, V_seq, K_sig_seq,
                                                    Q_seq, WINDOW)
                    cumulative_by_policy[name].append(np.sum(step_l2s))
                    last_step_l2_by_policy[name].append(step_l2s[-1])
                    arr = np.array(step_l2s)
                    if per_step_means[name] is None:
                        per_step_means[name] = arr
                    else:
                        # Pad to same length (shouldn't differ across trials at fixed W and N)
                        m = min(len(per_step_means[name]), len(arr))
                        per_step_means[name] = (per_step_means[name][:m] + arr[:m]) / 2
                        # (rough running mean; for first 10000+ trials this stabilizes)

    print(f"Trials simulated: {trial_count}")
    print(f"\n{'='*70}")
    print(f"Cumulative L2 over trajectory (sum of per-step L2 vs no_evict)")
    print(f"{'='*70}")
    print(f"\n{'policy':<12} {'mean cumL2':>12} {'median':>10}"
          f" {'p10':>10} {'p90':>10}  {'last-step L2':>14}")
    print("-" * 80)
    for name in ("qsigdist", "kk_M1", "random", "fifo"):
        c = np.array(cumulative_by_policy[name])
        last = np.array(last_step_l2_by_policy[name])
        print(f"{name:<12} {c.mean():>12.4f} {np.median(c):>10.4f} "
              f"{np.percentile(c, 10):>10.4f} {np.percentile(c, 90):>10.4f}"
              f"  {last.mean():>14.4f}")

    qsig = np.mean(cumulative_by_policy["qsigdist"])
    rand = np.mean(cumulative_by_policy["random"])
    print(f"\nHeadline: qsigdist cumulative L2 / random cumulative L2 = "
          f"{qsig / rand:.3f}×")
    if qsig < rand:
        print(f"  qsigdist still BEATS random over the trajectory.")
        print(f"  → the harness loss is NOT explained by cumulative L2 drift")
        print(f"  → other mechanisms (Q-drift from diverged cache, argmax noise,")
        print(f"     N=5 prompt variance) must be implicated.")
    else:
        print(f"  qsigdist LOSES to random over the trajectory.")
        print(f"  → CONFIRMS correlated-drift hypothesis: per-step optimization")
        print(f"     compounds destructively over many sequential decisions.")

    # Per-step trajectory shape: does qsigdist's per-step L2 grow over steps?
    print(f"\nPer-step L2 trajectory (mean across trials, first 10 steps + last):")
    n_steps = min(len(per_step_means[p]) for p in per_step_means)
    print(f"{'step':>6}", end="")
    for name in ("qsigdist", "random"):
        print(f" {name:>10}", end="")
    print()
    for i in list(range(min(10, n_steps))) + [n_steps - 1]:
        print(f"{i:>6}", end="")
        for name in ("qsigdist", "random"):
            v = per_step_means[name][i] if name in per_step_means else float('nan')
            print(f" {v:>10.4f}", end="")
        print()


if __name__ == "__main__":
    main()
