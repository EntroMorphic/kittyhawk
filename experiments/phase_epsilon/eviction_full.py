"""Phase ε — full-stack KV-eviction quality measurement.

Addresses all five untested concerns from Phase δ red-team
(td28_phase_delta_redteam_2026-05-12.md):

  ε-1: Longer-prompt dumps (64-token corpus → cache up to 63 vectors).
  ε-2: Per-Q-head oracle (each Q-head attends to its kv_head's K
       independently, per BitNet GQA). δ-1's mean-of-Q-heads was wrong.
  ε-3: Softmax-mass preservation alongside recall@k. Operationally
       relevant: attention output is softmax-weighted, not rank-ordered.
  ε-4: Attention-output L2 error — the actual preservation of
       softmax(Q·K)·V with vs. without eviction. The strongest proxy
       for generation quality without running the full harness.
  ε-5: Shuffled-K null control. If L1 > Hamming persists on shuffled
       K-cache (where learned correlations are destroyed), the
       advantage is metric-on-marginals not substrate-specific.

Three eviction policies compared:
  - hamming: keep K-vectors with smallest substrate Hamming(Q_sig, K_sig)
  - l1:      keep K-vectors with smallest substrate L1(Q_sig, K_sig)
  - random:  keep K-vectors uniformly at random (null)

Six trial dimensions:
  (prompt, query_position, layer, q_head, kv_head, cache_subset_size)
"""
from __future__ import annotations

import json
import os
import sys
import time
import glob
import numpy as np

THIS = os.path.dirname(__file__)
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(THIS, "..", "phase_alpha"))
sys.path.insert(0, os.path.join(THIS, "..", "phase_beta"))

from load_k_signatures import (
    HEAD_DIM, NUM_KV_HEADS, KV_PROJ, NUM_HIDDEN,
    THRESHOLD_TAU, CAPTURE_ORDER_V2,
    read_actv2, parse_filename, threshold_extract,
)


RNG_SEED = 20260512
DUMP_DIRS = ["data/c_dump_v3"]  # 64-token long prompt(s)
NUM_Q_HEADS = NUM_HIDDEN // HEAD_DIM  # 2560 / 128 = 20
Q_HEADS_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS  # 20 / 5 = 4


# ============================================================================
# Loader: per-Q-head, per-(layer, kv_head, position) Q / K / V
# ============================================================================

def load_qkv(dump_dirs):
    """Returns nested dict:
       prompt → position → layer → kv_head → {'K': (HEAD_DIM,) int32,
                                                'V': (HEAD_DIM,) int32,
                                                'Q': (Q_HEADS_PER_KV, HEAD_DIM) int32 — per q-head}
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
        q_arr = data["attn.q_post_rope"].reshape(NUM_Q_HEADS, HEAD_DIM)
        k_arr = data["attn.k_post_rope"].reshape(NUM_KV_HEADS, HEAD_DIM)
        v_arr = data["attn.v"].reshape(NUM_KV_HEADS, HEAD_DIM)
        out.setdefault(prompt_id, {}).setdefault(position, {}).setdefault(layer, {})
        for kv in range(NUM_KV_HEADS):
            q_start = kv * Q_HEADS_PER_KV
            q_end = (kv + 1) * Q_HEADS_PER_KV
            out[prompt_id][position][layer][kv] = {
                "K": k_arr[kv].astype(np.int32),
                "V": v_arr[kv].astype(np.int32),
                "Q": q_arr[q_start:q_end].astype(np.int32),
            }
    return out


# ============================================================================
# Substrate signatures
# ============================================================================

def substrate_sig(K: np.ndarray, tau: int = THRESHOLD_TAU) -> np.ndarray:
    """Threshold-extract to {-1, 0, +1}."""
    return threshold_extract(K.astype(np.int32), tau=tau)


def hamming_dist(q_sig: np.ndarray, K_sigs: np.ndarray) -> np.ndarray:
    """Per-K Hamming distance from a single Q sig to N K sigs."""
    return np.sum(K_sigs != q_sig[None, :], axis=1).astype(np.int32)


def l1_dist(q_sig: np.ndarray, K_sigs: np.ndarray) -> np.ndarray:
    """Per-K L1 distance from a single Q sig to N K sigs."""
    return np.abs(K_sigs.astype(np.int32) - q_sig.astype(np.int32)[None, :]).sum(axis=1)


# ============================================================================
# Trial: eviction quality for one (layer, kv_head, q_head, query_position)
# ============================================================================

def softmax_stable(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def trial_eviction(K_cache: np.ndarray, V_cache: np.ndarray,
                    Q: np.ndarray, k_keep: int, rng: np.random.Generator,
                    shuffle_cache: bool = False):
    """For one Q-head at one position with prior K-cache + V-cache.

    Args:
        K_cache: (P, HEAD_DIM) int32
        V_cache: (P, HEAD_DIM) int32
        Q:       (HEAD_DIM,) int32
        k_keep:  cache positions to keep
        rng:     for random baseline
        shuffle_cache: if True, shuffle cells within each K-vector
            (ε-5 null control; destroys cross-cell correlations, preserves
            marginals)

    Returns dict with recall@k, softmax-mass, attention-output L2 error
    per policy (hamming, l1, random).
    """
    P = K_cache.shape[0]
    if k_keep >= P:
        return None

    K_used = K_cache.copy()
    if shuffle_cache:
        # Shuffle each row independently (destroys learned cross-cell corr,
        # preserves per-row marginals). Uses the caller's rng so different
        # trials get different shuffle patterns; the previous fixed-seed
        # variant applied the same row-permutations across all trials,
        # which was functionally OK for the ε-5 control but a subtle
        # quirk caught in journal/td28_cold_eye_audit_2026-05-12.md.
        for i in range(P):
            K_used[i] = K_used[i, rng.permutation(HEAD_DIM)]

    # Substrate sigs
    K_sigs = substrate_sig(K_used)
    Q_sig  = substrate_sig(Q.reshape(1, -1))[0]

    # Dense attention scores (oracle) and softmax weights
    # Scale by sqrt(HEAD_DIM) per standard attention scaling
    scale = 1.0 / np.sqrt(HEAD_DIM)
    Q_f = Q.astype(np.float64)
    K_f = K_used.astype(np.float64)
    V_f = V_cache.astype(np.float64)
    scores = (K_f @ Q_f) * scale          # (P,)
    weights = softmax_stable(scores)       # (P,)
    out_full = V_f.T @ weights              # (HEAD_DIM,)

    # Oracle top-k by scores (recall metric reference)
    oracle = set(np.argsort(scores)[-k_keep:].tolist())
    oracle_mass = float(weights[list(oracle)].sum())

    # Policies
    def evaluate(kept_idx: list[int]):
        kept = set(kept_idx)
        recall = len(oracle & kept) / k_keep
        mass = float(weights[list(kept)].sum())
        # Attention output with only kept K's (renormalize)
        kept_arr = np.array(list(kept))
        w_kept = softmax_stable(scores[kept_arr])
        out_kept = V_f[kept_arr].T @ w_kept
        denom = np.linalg.norm(out_full)
        l2_err = float(np.linalg.norm(out_kept - out_full) / denom) if denom > 0 else float("nan")
        return {"recall": recall, "mass": mass, "l2_err": l2_err}

    # Hamming policy
    ham_d = hamming_dist(Q_sig, K_sigs)
    ham_kept = np.argsort(ham_d)[:k_keep].tolist()
    ham_eval = evaluate(ham_kept)

    # L1 policy
    l1_d = l1_dist(Q_sig, K_sigs)
    l1_kept = np.argsort(l1_d)[:k_keep].tolist()
    l1_eval = evaluate(l1_kept)

    # Random policy
    rand_kept = rng.choice(P, size=k_keep, replace=False).tolist()
    rand_eval = evaluate(rand_kept)

    return {
        "P": P, "k_keep": k_keep,
        "oracle_mass": oracle_mass,
        "hamming": ham_eval,
        "l1": l1_eval,
        "random": rand_eval,
    }


# ============================================================================
# Driver
# ============================================================================

def run_benchmark(qkv: dict, k_keeps: list[int], shuffle_cache: bool = False):
    rng = np.random.default_rng(RNG_SEED + (1 if shuffle_cache else 0))
    trials = []
    for prompt_id, by_pos in qkv.items():
        positions = sorted(by_pos.keys())
        for p_idx, p in enumerate(positions):
            if p_idx == 0:
                continue
            cache_positions = positions[:p_idx]
            for layer in by_pos[p]:
                for kv in range(NUM_KV_HEADS):
                    # Build K-cache and V-cache from prior positions
                    K_list = [by_pos[cp][layer][kv]["K"] for cp in cache_positions
                              if layer in by_pos.get(cp, {})]
                    V_list = [by_pos[cp][layer][kv]["V"] for cp in cache_positions
                              if layer in by_pos.get(cp, {})]
                    if len(K_list) < 8:
                        continue
                    K_cache = np.stack(K_list)
                    V_cache = np.stack(V_list)
                    # Per Q-head
                    Q_heads = by_pos[p][layer][kv]["Q"]
                    for qh in range(Q_HEADS_PER_KV):
                        Q = Q_heads[qh]
                        for k_keep in k_keeps:
                            if k_keep >= K_cache.shape[0]:
                                continue
                            r = trial_eviction(K_cache, V_cache, Q, k_keep,
                                                rng, shuffle_cache=shuffle_cache)
                            if r is None:
                                continue
                            r.update({"prompt": prompt_id, "position": p,
                                       "layer": layer, "kv_head": kv,
                                       "q_head_local": qh,
                                       "shuffled": shuffle_cache})
                            trials.append(r)
    return trials


def summarize_paired(trials: list[dict], metric: str):
    """For a given metric ∈ {recall, mass, l2_err}, compute paired
    (L1 - Hamming) statistics with prompt-clustered bootstrap CI.

    Vectorized: precompute per-prompt means of gap, bootstrap the
    prompt-level means (not the trial-level data).
    """
    if not trials:
        return None
    by_k = {}
    for t in trials:
        by_k.setdefault(t["k_keep"], []).append(t)
    out = {}
    for k_keep, ts in by_k.items():
        ham = np.array([t["hamming"][metric] for t in ts])
        l1  = np.array([t["l1"][metric]      for t in ts])
        rnd = np.array([t["random"][metric]  for t in ts])
        if metric == "l2_err":
            gap = ham - l1
        else:
            gap = l1 - ham
        prompts_arr = np.array([t["prompt"] for t in ts])
        prompts = sorted(set(prompts_arr.tolist()))
        # Per-prompt gap stats
        per_prompt_means = np.array([gap[prompts_arr == p].mean()
                                      for p in prompts])
        per_prompt_n = np.array([(prompts_arr == p).sum() for p in prompts])
        # Bootstrap by resampling prompts (with replacement)
        rng = np.random.default_rng(RNG_SEED)
        n_prompts = len(prompts)
        # Weighted by trial counts (each resample uses per-prompt mean
        # with per-prompt n as weight) — but since we want CI on the
        # OVERALL paired gap, use weighted mean with weights = per_prompt_n
        boot_means = []
        for _ in range(2000):
            idx = rng.integers(0, n_prompts, size=n_prompts)
            w = per_prompt_n[idx]
            m = (per_prompt_means[idx] * w).sum() / w.sum()
            boot_means.append(m)
        ci_lo, ci_hi = np.quantile(boot_means, [0.025, 0.975])
        # Leave-one-prompt-out range
        loo_means = []
        for i in range(n_prompts):
            mask = np.arange(n_prompts) != i
            w = per_prompt_n[mask]
            m = (per_prompt_means[mask] * w).sum() / w.sum()
            loo_means.append(m)
        out[k_keep] = {
            "n_trials": len(ts),
            "n_prompts": n_prompts,
            "hamming_mean": float(np.mean(ham)),
            "l1_mean": float(np.mean(l1)),
            "random_mean": float(np.mean(rnd)),
            "paired_gap_mean": float(np.mean(gap)),
            "paired_gap_ci": [float(ci_lo), float(ci_hi)],
            "loo_range": [float(min(loo_means)), float(max(loo_means))],
            "per_prompt_gaps": [float(x) for x in per_prompt_means],
            "l1_strictly_better_frac": float(np.mean(gap > 0)),
        }
    return out


def main(out_dir: str = "experiments/phase_epsilon/results"):
    os.makedirs(out_dir, exist_ok=True)
    print("=== Phase ε — full-stack KV-eviction quality ===\n")
    print(f"Loading from {DUMP_DIRS} ...")
    t0 = time.time()
    qkv = load_qkv(DUMP_DIRS)
    print(f"  prompts: {sorted(qkv.keys())}")
    sample_prompt = next(iter(qkv.keys()))
    print(f"  positions in {sample_prompt}: {len(qkv[sample_prompt])}")
    print(f"  load time: {time.time()-t0:.1f}s\n")

    k_keeps = [8, 16, 32]

    print("=" * 70)
    print("[Real K-cache]")
    print("=" * 70)
    trials_real = run_benchmark(qkv, k_keeps, shuffle_cache=False)
    print(f"  {len(trials_real)} trials\n")
    sum_real = {
        "recall":  summarize_paired(trials_real, "recall"),
        "mass":    summarize_paired(trials_real, "mass"),
        "l2_err":  summarize_paired(trials_real, "l2_err"),
    }
    for metric in ("recall", "mass", "l2_err"):
        print(f"\n  {metric}:")
        print(f"    {'k_keep':>7} {'Ham':>8} {'L1':>8} {'rand':>8}"
              f" {'paired_gap':>11} {'95% CI':>22}")
        for k_keep, s in sum_real[metric].items():
            print(f"    {k_keep:>7d} {s['hamming_mean']:>8.4f} "
                  f"{s['l1_mean']:>8.4f} {s['random_mean']:>8.4f} "
                  f"{s['paired_gap_mean']:>+11.4f}"
                  f"  [{s['paired_gap_ci'][0]:+.4f}, {s['paired_gap_ci'][1]:+.4f}]")

    print("\n" + "=" * 70)
    print("[Shuffled K-cache (ε-5 null control)]")
    print("=" * 70)
    trials_shuf = run_benchmark(qkv, k_keeps, shuffle_cache=True)
    print(f"  {len(trials_shuf)} trials\n")
    sum_shuf = {
        "recall":  summarize_paired(trials_shuf, "recall"),
        "mass":    summarize_paired(trials_shuf, "mass"),
        "l2_err":  summarize_paired(trials_shuf, "l2_err"),
    }
    for metric in ("recall", "mass", "l2_err"):
        print(f"\n  {metric}:")
        print(f"    {'k_keep':>7} {'Ham':>8} {'L1':>8} {'rand':>8}"
              f" {'paired_gap':>11} {'95% CI':>22}")
        for k_keep, s in sum_shuf[metric].items():
            print(f"    {k_keep:>7d} {s['hamming_mean']:>8.4f} "
                  f"{s['l1_mean']:>8.4f} {s['random_mean']:>8.4f} "
                  f"{s['paired_gap_mean']:>+11.4f}"
                  f"  [{s['paired_gap_ci'][0]:+.4f}, {s['paired_gap_ci'][1]:+.4f}]")

    # Final verdict
    print("\n" + "=" * 70)
    print("[Final verdict]")
    print("=" * 70)
    real_l2_32 = sum_real["l2_err"].get(32)
    shuf_l2_32 = sum_shuf["l2_err"].get(32)
    if real_l2_32 and shuf_l2_32:
        print(f"  At k_keep=32, attention-output L2 error (lower is better):")
        print(f"    Real K:    Ham={real_l2_32['hamming_mean']:.4f}  "
              f"L1={real_l2_32['l1_mean']:.4f}  "
              f"gap=Ham-L1={real_l2_32['paired_gap_mean']:+.4f} "
              f"[{real_l2_32['paired_gap_ci'][0]:+.4f}, {real_l2_32['paired_gap_ci'][1]:+.4f}]")
        print(f"    Shuf K:    Ham={shuf_l2_32['hamming_mean']:.4f}  "
              f"L1={shuf_l2_32['l1_mean']:.4f}  "
              f"gap=Ham-L1={shuf_l2_32['paired_gap_mean']:+.4f} "
              f"[{shuf_l2_32['paired_gap_ci'][0]:+.4f}, {shuf_l2_32['paired_gap_ci'][1]:+.4f}]")

    # Persist
    def serial(x):
        if isinstance(x, dict): return {str(k): serial(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [serial(v) for v in x]
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (bool, np.bool_)): return bool(x)
        return x

    out_path = os.path.join(out_dir, "phase_epsilon_results.json")
    with open(out_path, "w") as f:
        json.dump({"real": serial(sum_real), "shuffled": serial(sum_shuf),
                    "n_trials_real": len(trials_real),
                    "n_trials_shuffled": len(trials_shuf)}, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
