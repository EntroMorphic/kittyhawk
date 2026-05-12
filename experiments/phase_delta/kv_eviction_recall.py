"""Phase δ — application-level KV-eviction recall@k benchmark.

The Phase γ robustness matrix found that substrate-L1 has a 47pp
d̂/D_amb gap over PCA-binary IN THE CLOSE REGIME (within layer/kv_head/
site). If that geometric finding is load-bearing, it should cash out
when used as a KV-cache eviction policy: substrate-L1 distance from a
query should rank the K-vectors with high dense-attention scores
HIGHER than substrate-Hamming distance does.

This script measures that directly on real K-cache from our dumps.

Setup per (layer, kv_head):
  - Cache K = K_post_rope for positions [0, 1, ..., P-1]
  - Query Q = Q_post_rope for the LAST position
  - Oracle attention scores: s_i = Q · K_i (fp32 dot product)
  - Oracle top-k:  argsort(s)[-k:]
  - Substrate Q sig = threshold_extract(Q, τ=5000)
  - Substrate K sigs = threshold_extract(K_i, τ=5000)
  - Hamming policy: keep k K-vectors minimizing Hamming(Q_sig, K_sig)
  - L1 policy:      keep k K-vectors minimizing L1(Q_sig, K_sig)
  - Random policy:  keep k uniformly at random (null)

Recall@k = |oracle_top_k ∩ policy_top_k| / k

Compare Hamming-recall vs L1-recall across (layer, kv_head, query
position). If L1 beats Hamming, the Phase γ close-regime advantage
translates to eviction quality — the substrate-distinctive geometric
property has an application consequence.

Reported: per-(layer, kv_head) and pooled, with bootstrap CIs by
resampling the query positions.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np

THIS = os.path.dirname(__file__)
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(THIS, "..", "phase_alpha"))
sys.path.insert(0, os.path.join(THIS, "..", "phase_beta"))

from load_k_signatures import (
    HEAD_DIM, NUM_KV_HEADS, KV_PROJ, NUM_HIDDEN,
    THRESHOLD_TAU, CAPTURE_ORDER_V2,
    read_actv2, parse_filename, threshold_extract,
    pairwise_hamming_int8,
)
from m1_l1_estimator import pairwise_L1_int8
import glob


DUMP_DIRS = ["data/c_dump", "data/c_dump_v2"]
RNG_SEED = 20260512


# ============================================================================
# Load K + Q per (prompt, position, layer, kv_head)
# ============================================================================

def load_qk_by_prompt(dump_dirs):
    """Return dict: prompt_id → {position: {layer: {(kv_head, 'K'|'Q'): vec}}}.

    Each Q/K is a HEAD_DIM=128 int32 array.
    """
    paths = []
    for d in dump_dirs:
        paths.extend(glob.glob(os.path.join(d, "*.layer*.bin")))
    paths = sorted(paths)
    out: dict = {}
    for path in paths:
        prompt_id, position, layer = parse_filename(path)
        try:
            data = read_actv2(path)
        except Exception:
            continue
        layer = data["_layer"]
        q_arr = data["attn.q_post_rope"]   # (NUM_HIDDEN,)
        k_arr = data["attn.k_post_rope"]   # (KV_PROJ,) = (NUM_KV_HEADS * HEAD_DIM,)
        q_arr = q_arr.reshape(-1, HEAD_DIM)        # (num_q_heads, HEAD_DIM)
        k_arr = k_arr.reshape(NUM_KV_HEADS, HEAD_DIM)
        # For attention, Q from a Q-head attends to K from the corresponding
        # kv_head via GQA. BitNet 2B has num_q_heads = 20, num_kv_heads = 5,
        # so 4 Q-heads share each kv_head. We sum or pick a representative
        # Q per kv_head. Use the MEAN of the 4 sharing Q-heads as the
        # "effective query" per kv_head — this matches how attention
        # logits aggregate per group.
        n_q_per_kv = q_arr.shape[0] // NUM_KV_HEADS
        out.setdefault(prompt_id, {})
        out[prompt_id].setdefault(position, {})
        out[prompt_id][position][layer] = {}
        for kv in range(NUM_KV_HEADS):
            q_start, q_end = kv * n_q_per_kv, (kv + 1) * n_q_per_kv
            q_grp_mean = q_arr[q_start:q_end].astype(np.float64).mean(axis=0)
            out[prompt_id][position][layer][(kv, "Q")] = q_grp_mean
            out[prompt_id][position][layer][(kv, "K")] = k_arr[kv].astype(np.int32)
    return out


# ============================================================================
# Per-(prompt, layer, kv_head) eviction trial
# ============================================================================

def eviction_trial(K_cache: np.ndarray, Q: np.ndarray, k_keep: int,
                    rng: np.random.Generator):
    """Compute recall@k for Hamming-substrate, L1-substrate, random.

    K_cache: (P, HEAD_DIM) int32 — past K-vectors.
    Q:       (HEAD_DIM,) float64 — current query.
    k_keep:  how many K-vectors to keep.

    Returns dict of recall@k by policy.
    """
    P = K_cache.shape[0]
    if k_keep >= P:
        return {"hamming": 1.0, "l1": 1.0, "random": 1.0, "P": P, "k": k_keep}
    # Oracle (dense attention top-k)
    K_f = K_cache.astype(np.float64)
    scores = K_f @ Q  # (P,)
    oracle = set(np.argsort(scores)[-k_keep:].tolist())

    # Substrate signatures
    K_sub = threshold_extract(K_cache, tau=THRESHOLD_TAU)
    Q_sub = threshold_extract(Q.reshape(1, -1).astype(np.int32), tau=THRESHOLD_TAU)[0]

    # Hamming distance: count cells where K_sub[i] != Q_sub
    ham_dist = np.sum(K_sub != Q_sub[None, :], axis=1).astype(np.int32)
    hamming_topk = set(np.argsort(ham_dist)[:k_keep].tolist())  # smallest dist

    # L1 distance: sum |K_sub - Q_sub|
    l1_dist = np.abs(K_sub.astype(np.int32) - Q_sub.astype(np.int32)).sum(axis=1)
    l1_topk = set(np.argsort(l1_dist)[:k_keep].tolist())

    # Random baseline
    rand_topk = set(rng.choice(P, size=k_keep, replace=False).tolist())

    return {
        "hamming": len(oracle & hamming_topk) / k_keep,
        "l1":      len(oracle & l1_topk) / k_keep,
        "random":  len(oracle & rand_topk) / k_keep,
        "P": P, "k": k_keep,
    }


# ============================================================================
# Driver
# ============================================================================

def run_eviction_benchmark(qk_by_prompt: dict,
                            k_fractions=(0.25, 0.50, 0.75),
                            min_cache_size: int = 4):
    """For each (prompt, query position, layer, kv_head), run eviction
    trials at multiple k-keep fractions.
    """
    rng = np.random.default_rng(RNG_SEED)
    trials = []  # list of dicts
    for prompt_id, by_pos in qk_by_prompt.items():
        positions = sorted(by_pos.keys())
        if len(positions) < 2:
            continue
        # For each position p > 0, use prior positions [0..p-1] as K-cache,
        # current position p as query.
        for p_idx, p in enumerate(positions):
            if p_idx == 0:
                continue  # no prior K-cache
            cache_positions = positions[:p_idx]
            for layer in by_pos[p]:
                for kv in range(NUM_KV_HEADS):
                    K_list = []
                    for cp in cache_positions:
                        if layer in by_pos.get(cp, {}):
                            K_list.append(by_pos[cp][layer][(kv, "K")])
                    if len(K_list) < min_cache_size:
                        continue
                    K_cache = np.stack(K_list)
                    Q = by_pos[p][layer][(kv, "Q")]
                    for k_frac in k_fractions:
                        k_keep = max(1, int(len(K_list) * k_frac))
                        r = eviction_trial(K_cache, Q, k_keep, rng)
                        r.update({"prompt": prompt_id, "position": p,
                                   "layer": layer, "kv_head": kv,
                                   "k_frac": k_frac})
                        trials.append(r)
    return trials


def summarize(trials):
    """Per-k_frac pooled and per-layer recall stats with bootstrap CIs."""
    if not trials:
        return {}
    by_kfrac: dict = {}
    for t in trials:
        by_kfrac.setdefault(t["k_frac"], []).append(t)
    out = {}
    rng = np.random.default_rng(RNG_SEED)
    for kf, ts in by_kfrac.items():
        ham = np.array([t["hamming"] for t in ts])
        l1  = np.array([t["l1"] for t in ts])
        rnd = np.array([t["random"] for t in ts])

        def boot_ci(arr, B=500):
            means = [np.mean(rng.choice(arr, size=len(arr), replace=True))
                     for _ in range(B)]
            return float(np.mean(arr)), float(np.quantile(means, 0.025)), \
                   float(np.quantile(means, 0.975))

        ham_m, ham_lo, ham_hi = boot_ci(ham)
        l1_m,  l1_lo,  l1_hi  = boot_ci(l1)
        rnd_m, rnd_lo, rnd_hi = boot_ci(rnd)
        # Paired diff (L1 - Hamming) per trial — direct comparison
        diff = l1 - ham
        diff_m, diff_lo, diff_hi = boot_ci(diff)
        out[f"k_frac_{kf}"] = {
            "n_trials": len(ts),
            "hamming_mean": ham_m, "hamming_ci": [ham_lo, ham_hi],
            "l1_mean": l1_m,       "l1_ci":      [l1_lo, l1_hi],
            "random_mean": rnd_m,  "random_ci":  [rnd_lo, rnd_hi],
            "paired_diff_l1_minus_hamming_mean": diff_m,
            "paired_diff_ci": [diff_lo, diff_hi],
            "l1_beats_hamming_frac": float(np.mean(diff > 0)),
            "l1_ties_hamming_frac": float(np.mean(diff == 0)),
        }

    # Also per-layer summary at k_frac=0.5 (the middle)
    per_layer = {}
    for t in trials:
        if t["k_frac"] != 0.5:
            continue
        per_layer.setdefault(t["layer"], {"ham": [], "l1": []})
        per_layer[t["layer"]]["ham"].append(t["hamming"])
        per_layer[t["layer"]]["l1"].append(t["l1"])
    out["per_layer_kfrac_0.5"] = {
        int(L): {
            "n": len(d["ham"]),
            "hamming_mean": float(np.mean(d["ham"])),
            "l1_mean": float(np.mean(d["l1"])),
            "l1_minus_ham": float(np.mean(d["l1"]) - np.mean(d["ham"])),
        }
        for L, d in sorted(per_layer.items())
    }
    return out


def main(out_dir: str = "experiments/phase_delta/results"):
    os.makedirs(out_dir, exist_ok=True)
    print("=== Phase δ — KV-eviction recall@k benchmark ===\n")
    print(f"Loading Q + K from {DUMP_DIRS} ...")
    t0 = time.time()
    qk = load_qk_by_prompt(DUMP_DIRS)
    n_prompts = len(qk)
    n_total = sum(len(positions) for positions in qk.values())
    print(f"  {n_prompts} prompts, {n_total} (prompt, position) combos  ({time.time()-t0:.1f}s)\n")

    print("Running eviction trials ...")
    t0 = time.time()
    trials = run_eviction_benchmark(qk)
    print(f"  {len(trials)} trials  ({time.time()-t0:.1f}s)\n")

    summary = summarize(trials)
    print("=== Pooled recall@k results ===\n")
    print(f"{'k_frac':>8} {'n':>6} {'Hamming':>14} {'L1':>14} {'random':>14} "
          f"{'L1-Ham':>16} {'L1 wins%':>10}")
    for kf_key in sorted(k for k in summary if k.startswith("k_frac")):
        s = summary[kf_key]
        kf = kf_key.replace("k_frac_", "")
        print(f"{kf:>8} {s['n_trials']:>6d} "
              f"{s['hamming_mean']:>6.3f} [{s['hamming_ci'][0]:.3f},{s['hamming_ci'][1]:.3f}] "
              f"{s['l1_mean']:>6.3f} [{s['l1_ci'][0]:.3f},{s['l1_ci'][1]:.3f}] "
              f"{s['random_mean']:>6.3f} [{s['random_ci'][0]:.3f},{s['random_ci'][1]:.3f}] "
              f"{s['paired_diff_l1_minus_hamming_mean']:>+7.4f} [{s['paired_diff_ci'][0]:+.4f},{s['paired_diff_ci'][1]:+.4f}] "
              f"{s['l1_beats_hamming_frac']:>9.1%}")

    print("\n=== Per-layer (k_frac=0.5) ===\n")
    print(f"{'layer':>5} {'n':>5} {'Ham':>7} {'L1':>7} {'L1-Ham':>10}")
    pl = summary.get("per_layer_kfrac_0.5", {})
    for L in sorted(pl):
        d = pl[L]
        print(f"{L:>5d} {d['n']:>5d} {d['hamming_mean']:>7.3f} "
              f"{d['l1_mean']:>7.3f} {d['l1_minus_ham']:>+10.4f}")

    # Headline verdict
    print("\n=== Verdict ===")
    s_50 = summary.get("k_frac_0.5", {})
    diff = s_50.get("paired_diff_l1_minus_hamming_mean", 0)
    diff_ci = s_50.get("paired_diff_ci", [0, 0])
    if diff_ci[0] > 0:
        print(f"L1-substrate eviction beats Hamming-substrate eviction at k_frac=0.5 by "
              f"{diff:+.4f} (95% CI [{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}]) — significant.")
    elif diff_ci[1] < 0:
        print(f"L1-substrate eviction LOSES to Hamming-substrate at k_frac=0.5 by "
              f"{diff:+.4f} — significant negative.")
    else:
        print(f"L1 vs Hamming difference at k_frac=0.5: {diff:+.4f} "
              f"95% CI [{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}] — CI crosses zero.")

    # Persist
    def serial(x):
        if isinstance(x, dict): return {str(k): serial(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [serial(v) for v in x]
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (bool, np.bool_)): return bool(x)
        return x

    out_path = os.path.join(out_dir, "kv_eviction_results.json")
    with open(out_path, "w") as f:
        json.dump({"trials": serial(trials), "summary": serial(summary)}, f, indent=2)
    print(f"\nResults: {out_path}")
    return summary, trials


if __name__ == "__main__":
    main()
