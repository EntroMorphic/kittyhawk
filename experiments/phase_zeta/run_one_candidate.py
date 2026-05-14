"""One-shot runner: run a specific meta(w_r, w_kk, w_qk) on the N=100
battery and print observed Δ vs random. Does NOT mutate anchors.json
(callers append manually after consolidating parallel runs).

Usage:
    python run_one_candidate.py <w_r> <w_kk> <w_qk>
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import numpy as np

THIS = os.path.dirname(__file__)
HARNESS = "build/gesh/bitnet_harness"
DATA    = "data/bitnet_b158_2b4t.bin"
WINDOW  = 16
GEN_N   = 24


def load_n100_prompts() -> dict[str, str]:
    sys.path.insert(0, THIS)
    try:
        from n50_battery import PROMPTS as OLD
        from n100_battery_incremental import NEW_PROMPTS as NEW
    finally:
        sys.path.pop(0)
    p = {}; p.update(OLD); p.update(NEW)
    assert len(p) == 100
    return p


def run(w):
    prompts = load_n100_prompts()
    iter_name = f"oneshot_{w[0]}_{w[1]}_{w[2]}"
    outdir = os.path.join(THIS, f"results/meta_iterate/{iter_name}")
    os.makedirs(outdir, exist_ok=True)

    tokens_by_label = {}
    print(f"[runner] candidate {w}; outdir = {outdir}\n", flush=True)
    for i, (label, prompt) in enumerate(prompts.items()):
        env = os.environ.copy()
        env.update({
            "BITNET_KV_EVICT_MODE":  "meta",
            "BITNET_KV_WINDOW":      str(WINDOW),
            "BITNET_KV_EVICT_W_R":   str(w[0]),
            "BITNET_KV_EVICT_W_KK":  str(w[1]),
            "BITNET_KV_EVICT_W_QK":  str(w[2]),
            "BITNET_ATTN_FIXED_TAU": "5000",
        })
        cmd = [HARNESS, DATA, "--prompt-tokens", prompt, "--gen", str(GEN_N)]
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        elapsed = time.time() - t0
        out = proc.stdout + proc.stderr
        with open(os.path.join(outdir, f"{label}.log"), "w") as f:
            f.write(out)
        m = re.search(r"generated tokens\s*=\s*([\d\s\-]+)", out)
        toks = [int(t) for t in m.group(1).strip().split()] if m else None
        tokens_by_label[label] = toks
        tok_str = " ".join(str(t) for t in (toks or [])[:8])
        print(f"  [{i+1:>3}/100] {label:<24} {elapsed:>5.1f}s  {tok_str}",
              flush=True)
    return tokens_by_label


def compute_delta_vs_random(tokens_by_label):
    with open(os.path.join(THIS, "results/n50_battery/battery_results.json")) as f:
        old = json.load(f)["trials"]
    with open(os.path.join(THIS, "results/n100_incremental/battery_results.json")) as f:
        new = json.load(f)["trials"]
    base = {(t["label"], t["mode"]): t["tokens"] for t in old + new}

    def match_rate(a, b):
        n = min(len(a), len(b))
        return sum(1 for i in range(n) if a[i] == b[i]) / n if n else 0.0

    diffs = []
    for label, cand_tokens in tokens_by_label.items():
        no_evict = base.get((label, "no_evict"))
        randc    = base.get((label, "random"))
        if not (no_evict and cand_tokens and randc):
            continue
        diffs.append(match_rate(no_evict, cand_tokens) - match_rate(no_evict, randc))
    return 100.0 * float(np.mean(diffs))


def main():
    if len(sys.argv) != 4:
        print("usage: python run_one_candidate.py <w_r> <w_kk> <w_qk>")
        sys.exit(2)
    w = tuple(int(x) for x in sys.argv[1:4])
    for v in w:
        if v not in (-1, 0, 1):
            print(f"weights must be in {{-1, 0, 1}}; got {w}")
            sys.exit(2)
    toks = run(w)
    delta = compute_delta_vs_random(toks)
    print(f"\n[runner] candidate {w}: Δ vs random = {delta:+.2f}pp\n")
    # Print a machine-readable line for downstream parsing
    print(f"RESULT w={list(w)} delta={delta:.4f}")


if __name__ == "__main__":
    main()
