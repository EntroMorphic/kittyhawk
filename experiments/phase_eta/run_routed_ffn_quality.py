"""Step 1 PoC: end-to-end quality measurement of routed FFN at various
active-layer sets vs dense baseline.

For each config:
  - Run N=20 diverse prompts × 24 gen tokens
  - Compare token outputs to dense baseline (token-level match rate)
  - Report mean match-rate per config

Configs:
  dense (baseline, no LSH)
  LSH at L2 only
  LSH at L15 only
  LSH at L27 only
  LSH at all three (L2, L15, L27)
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
DICT = os.path.join(THIS, "results/lsh_ffn_dict.bin")

CONFIGS = [
    ("dense", ""),
    ("lsh_L2", "2"),
    ("lsh_L15", "15"),
    ("lsh_L27", "27"),
    ("lsh_L2_L15", "2,15"),
    ("lsh_L2_L15_L27", "2,15,27"),
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


def run_one(prompt, lsh_layers):
    env = os.environ.copy()
    env.update({
        "BITNET_KV_EVICT_MODE": "qsigdist",
        "BITNET_KV_WINDOW": str(WINDOW),
        "BITNET_ATTN_FIXED_TAU": "5000",
    })
    if lsh_layers:
        env["BITNET_FFN_LSH_DICT"] = DICT
        env["BITNET_FFN_LSH_LAYERS"] = lsh_layers
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
    for ci, (cfg, layers) in enumerate(CONFIGS):
        print(f"[{ci+1}/{len(CONFIGS)}] config={cfg} layers={layers!r}")
        for label, prompt in selected:
            ts = time.time()
            toks = run_one(prompt, layers)
            print(f"    {label:<24} {time.time()-ts:>5.1f}s", flush=True)
            results[cfg][label] = toks

    # Compare each LSH config to dense
    print(f"\n{'='*70}")
    print("Per-config match rate vs dense baseline")
    print(f"{'='*70}")
    dense = results["dense"]
    print(f"  {'config':<22}  {'mean match rate':>16}  {'>= 0.5':>7}  "
          f"{'>= 0.8':>7}  {'== 1.0':>7}")
    for cfg, _ in CONFIGS:
        if cfg == "dense": continue
        rates = []
        for label, _ in selected:
            r = match_rate(dense.get(label), results[cfg].get(label))
            rates.append(r)
        rates = np.array(rates)
        print(f"  {cfg:<22}  {rates.mean():>15.3f}  "
              f"{(rates >= 0.5).mean():>6.1%}  "
              f"{(rates >= 0.8).mean():>6.1%}  "
              f"{(rates == 1.0).mean():>6.1%}")

    # Per-prompt detail for the most interesting config
    print(f"\nPer-prompt match rate (vs dense):")
    print(f"  {'prompt':<24}  " + "  ".join(f"{c[:11]:>11}" for c, _ in CONFIGS[1:]))
    for label, _ in selected:
        rates = [match_rate(dense.get(label), results[cfg].get(label))
                 for cfg, _ in CONFIGS[1:]]
        print(f"  {label:<24}  " + "  ".join(f"{r:>11.3f}" for r in rates))

    print(f"\nElapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
