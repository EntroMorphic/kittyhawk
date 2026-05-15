"""Step 1.5a: hybrid routed FFN quality test.

Compares dense baseline vs hybrid LSH (route well-populated buckets,
fall back to dense for sparse) at multiple n_min thresholds. Tests
the hypothesis: quality loss from Step 1 PoC is driven by sparse
buckets, not by tile architecture.

Configs (all at L15 only, the best Step 1 single-layer):
  dense (baseline)
  hybrid n_min=1   (all-routed, ≡ Step 1's lsh_L15)
  hybrid n_min=3   (skip singletons + doubles)
  hybrid n_min=5   (skip up to quartiles)
  hybrid n_min=10  (only most populous buckets)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

import numpy as np

THIS = os.path.dirname(__file__)
HARNESS = "build/gesh/bitnet_harness"
DATA = "data/bitnet_b158_2b4t.bin"
WINDOW = 16
GEN = 24

CONFIGS = [
    ("dense", None),
    ("hybrid_nmin1_L15",  os.path.join(THIS, "results/lsh_ffn_dict.bin")),
    ("hybrid_nmin3_L15",  os.path.join(THIS, "results/lsh_ffn_dict_nmin3.bin")),
    ("hybrid_nmin5_L15",  os.path.join(THIS, "results/lsh_ffn_dict_nmin5.bin")),
    ("hybrid_nmin10_L15", os.path.join(THIS, "results/lsh_ffn_dict_nmin10.bin")),
]

LABELS = [
    "tech_neural", "tech_quantum",
    "code_python_fn", "code_sql",
    "dialog_qa", "dialog_greet",
    "q_capital_egypt", "q_dna_full",
    "math_word_problem", "math_div",
    "logic_causal2", "logic_negation2",
    "long_storm", "long_meeting",
    "poetry_haiku", "poetry_blake",
    "idiom_horse", "idiom_back",
    "def_metaphor", "history_moon",
]


def load_prompts():
    sys.path.insert(0, "experiments/phase_zeta")
    try:
        from n50_battery import PROMPTS as OLD
        from n100_battery_incremental import NEW_PROMPTS as NEW
    finally:
        sys.path.pop(0)
    p = {}; p.update(OLD); p.update(NEW)
    return p


def run_one(prompt, dict_path):
    env = os.environ.copy()
    env.update({
        "BITNET_KV_EVICT_MODE": "qsigdist",
        "BITNET_KV_WINDOW": str(WINDOW),
        "BITNET_ATTN_FIXED_TAU": "5000",
    })
    if dict_path:
        env["BITNET_FFN_LSH_DICT"] = dict_path
        env["BITNET_FFN_LSH_LAYERS"] = "15"
    cmd = [HARNESS, DATA, "--prompt-tokens", prompt, "--gen", str(GEN)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    m = re.search(r"generated tokens\s*=\s*([\d\s\-]+)", out)
    return [int(t) for t in m.group(1).strip().split()] if m else None


def match_rate(a, b):
    if not a or not b: return 0.0
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]) / n if n else 0.0


def main():
    prompts = load_prompts()
    selected = [(l, prompts[l]) for l in LABELS if l in prompts]
    print(f"Running {len(CONFIGS)} configs × {len(selected)} prompts × gen={GEN}\n")

    t0 = time.time()
    results = {cfg: {} for cfg, _ in CONFIGS}
    for ci, (cfg, dict_path) in enumerate(CONFIGS):
        print(f"[{ci+1}/{len(CONFIGS)}] config={cfg}")
        for label, prompt in selected:
            ts = time.time()
            toks = run_one(prompt, dict_path)
            print(f"    {label:<24} {time.time()-ts:>5.1f}s", flush=True)
            results[cfg][label] = toks

    print(f"\n{'='*70}")
    print("Per-config match rate vs dense baseline (L15 routing only)")
    print(f"{'='*70}")
    dense = results["dense"]
    print(f"  {'config':<22}  {'mean match':>11}  {'>= 0.5':>7}  "
          f"{'>= 0.8':>7}  {'== 1.0':>7}")
    for cfg, _ in CONFIGS:
        if cfg == "dense": continue
        rates = [match_rate(dense.get(l), results[cfg].get(l)) for l, _ in selected]
        rates = np.array(rates)
        print(f"  {cfg:<22}  {rates.mean():>10.3f}  "
              f"{(rates >= 0.5).mean():>6.1%}  "
              f"{(rates >= 0.8).mean():>6.1%}  "
              f"{(rates == 1.0).mean():>6.1%}")

    print(f"\nElapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
