"""Phase ζ — harness-level generation-quality benchmark for substrate eviction.

The Phase ε attention-output L2 result said L1-substrate eviction
reduces attention error by 38-62% vs Hamming-substrate eviction.
But the production DEFAULT is no-eviction (cold-eye audit, §3).

This script measures THE TERRITORY: does enabling substrate eviction
(BITNET_KV_EVICT_MODE=sigdist) on real inference produce
coherent / different / degraded generation vs no-eviction baseline?

For each prompt × eviction-mode combination:
  - Run the harness end-to-end with --gen N (N tokens generated greedily).
  - Capture: generated token sequence; post-final-norm x[0..3]; logits[0..3].
  - Compare modes pairwise: where does sigdist diverge from no-evict?

Modes:
  no_evict — production default. Full K-cache retained.
  fifo     — drop-oldest. Null baseline (cheapest eviction).
  sigdist  — substrate L1 distance eviction. The claim under test.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from typing import Optional

HARNESS = "build/gesh/bitnet_harness"
DATA    = "data/bitnet_b158_2b4t.bin"
OUTDIR  = "experiments/phase_zeta/results"
WINDOWS = [8, 16, 32]   # sweep eviction pressure
GEN_N   = 24

PROMPTS = {
    "capital_france": "1,1841,8085,341,9099,1735",
    "short_a":        "1,464,2944,18",
    "short_b":        "1,791,28036,9100",
    "medium_11":      "1,1841,8085,341,9099,1735,374,279,3838,520,2728",
    "bos_only":       "",
}

def mode_env(mode: str, window: int) -> dict:
    if mode == "no_evict":
        return {"BITNET_KV_EVICT_MODE": "none"}
    if mode == "fifo":
        return {"BITNET_KV_EVICT_MODE": "fifo", "BITNET_KV_WINDOW": str(window)}
    if mode == "sigdist":
        return {"BITNET_KV_EVICT_MODE": "sigdist", "BITNET_KV_WINDOW": str(window),
                "BITNET_ATTN_FIXED_TAU": "5000"}
    if mode == "qsigdist":
        return {"BITNET_KV_EVICT_MODE": "qsigdist", "BITNET_KV_WINDOW": str(window),
                "BITNET_ATTN_FIXED_TAU": "5000"}
    if mode == "random":
        return {"BITNET_KV_EVICT_MODE": "random", "BITNET_KV_WINDOW": str(window),
                "BITNET_KV_EVICT_SEED": "42"}
    raise ValueError(mode)


def run_one(label: str, prompt: str, mode: str, env_overrides: dict) -> dict:
    env = os.environ.copy()
    env.update(env_overrides)
    if prompt:
        cmd = [HARNESS, DATA, "--prompt-tokens", prompt, "--gen", str(GEN_N)]
    else:
        cmd = [HARNESS, DATA, "--token", "1", "--positions", "1", "--gen", str(GEN_N)]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0
    out = proc.stdout + proc.stderr
    # Save full log
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f"{label}_{mode}.log"), "w") as f:
        f.write(out)
    return {
        "label": label,
        "mode": mode,
        "exit_code": proc.returncode,
        "elapsed_s": elapsed,
        "stdout_tail": out.splitlines()[-15:] if out else [],
        "tokens": extract_tokens(out),
        "x_first4": extract_x(out),
        "logits_first4": extract_logits(out),
        "argmax": extract_argmax(out),
    }


def extract_tokens(text: str) -> Optional[list[int]]:
    m = re.search(r"generated tokens\s*=\s*([\d\s\-]+)", text)
    if not m:
        return None
    return [int(t) for t in m.group(1).strip().split()]


def extract_x(text: str) -> Optional[list[int]]:
    m = re.search(r"post-final-norm x\[0\.\.3\]\s*=\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)", text)
    if not m:
        return None
    return [int(x) for x in m.groups()]


def extract_logits(text: str) -> Optional[list[int]]:
    m = re.search(r"logits\[0\.\.3\]\s*=\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)", text)
    if not m:
        return None
    return [int(x) for x in m.groups()]


def extract_argmax(text: str) -> Optional[int]:
    m = re.search(r"argmax over full vocab\s*=\s*(-?\d+)", text)
    if not m:
        return None
    return int(m.group(1))


def divergence_point(a: list[int], b: list[int]) -> int:
    """First index where a, b differ. Returns len(a) if identical up to min len."""
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b))


def distinct_count(seq: list[int]) -> int:
    return len(set(seq))


def match_rate_vs_baseline(base: list[int], cand: list[int]) -> float:
    """Position-wise exact-match rate (longer-than-shorter excluded)."""
    n = min(len(base), len(cand))
    if n == 0: return float("nan")
    return sum(1 for i in range(n) if base[i] == cand[i]) / n


def main():
    modes = ["no_evict", "fifo", "random", "sigdist", "qsigdist"]
    all_results: list[dict] = []
    for window in WINDOWS:
        print(f"\n{'='*70}")
        print(f"=== window={window}, gen={GEN_N} ===")
        print(f"{'='*70}\n")
        print(f"{'label':<16} {'mode':<9} {'elapsed':>8}  {'distinct':>8}  tokens")
        print("-" * 110)
        win_results: list[dict] = []
        for label, prompt in PROMPTS.items():
            base_toks = None
            for mode in modes:
                env = mode_env(mode, window)
                r = run_one(label, prompt, mode, env)
                r["window"] = window
                win_results.append(r)
                all_results.append(r)
                toks = r["tokens"] or []
                distinct = distinct_count(toks)
                tok_str = " ".join(str(t) for t in toks[:14])
                print(f"{label:<16} {mode:<9} {r['elapsed_s']:>7.2f}s  {distinct:>8d}  {tok_str}")
                if mode == "no_evict":
                    base_toks = toks
            print()

        # Per-window divergence + match-rate summary
        print(f"\n=== window={window} analysis vs no_evict ===\n")
        print(f"{'label':<16} {'mode':<9} {'1st_div':>8} {'match%':>8} {'distinct':>8}")
        print("-" * 70)
        by_lm = {(r["label"], r["mode"]): r for r in win_results}
        for label in PROMPTS:
            base = by_lm[(label, "no_evict")]["tokens"]
            if base is None: continue
            base_distinct = distinct_count(base)
            print(f"{label:<16} {'(base)':<9} {'-':>8} {'-':>8} {base_distinct:>8d}")
            for mode in ("fifo", "random", "sigdist", "qsigdist"):
                toks = by_lm[(label, mode)]["tokens"]
                if toks is None: continue
                div = divergence_point(base, toks)
                mr = match_rate_vs_baseline(base, toks) * 100
                d = distinct_count(toks)
                print(f"{label:<16} {mode:<9} {div:>8d} {mr:>7.1f}% {d:>8d}")
            print()

        # Aggregate match-rate per mode for this window
        import numpy as np
        print(f"=== window={window} aggregate (mean across prompts) ===")
        for mode in ("fifo", "random", "sigdist", "qsigdist"):
            divs, mrs, dists = [], [], []
            for label in PROMPTS:
                base = by_lm[(label, "no_evict")]["tokens"]
                toks = by_lm[(label, mode)]["tokens"]
                if base is None or toks is None: continue
                divs.append(divergence_point(base, toks))
                mrs.append(match_rate_vs_baseline(base, toks))
                dists.append(distinct_count(toks))
            base_dist = np.mean([distinct_count(by_lm[(l, "no_evict")]["tokens"] or [])
                                  for l in PROMPTS])
            print(f"  {mode:<9} mean_div={np.mean(divs):>5.1f}  mean_match={np.mean(mrs)*100:>5.1f}%  "
                  f"mean_distinct={np.mean(dists):>4.1f} (base {base_dist:.1f})")

    out = {"config": {"windows": WINDOWS, "gen_n": GEN_N},
           "trials": all_results}
    with open(os.path.join(OUTDIR, "battery_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults: {OUTDIR}/battery_results.json")


if __name__ == "__main__":
    main()
