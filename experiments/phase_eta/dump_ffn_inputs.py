"""B1 step 1: dump FFN-input activations from N=20 diverse prompts.

Selects one prompt per category-prefix where possible (covers tech,
code, dialog, q, math, logic, etc.); samples 3 layers (early=L2,
middle=L15, late=L27); generates 8 tokens per prompt.

Output: experiments/phase_eta/results/ffn_dump/ — binary files per
(prompt, position, layer).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

THIS = os.path.dirname(__file__)
HARNESS = "build/gesh/bitnet_harness"
DATA = "data/bitnet_b158_2b4t.bin"
WINDOW = 16
GEN = 8
LAYERS = "2,15,27"
DUMP_DIR = os.path.join(THIS, "results/ffn_dump")

# Hand-picked prompts: one per category where category has >= 1 prompt
SELECTED = None  # None = use all available prompts from the N=100 battery


def load_prompts():
    sys.path.insert(0, "experiments/phase_zeta")
    try:
        from n50_battery import PROMPTS as OLD
        from n100_battery_incremental import NEW_PROMPTS as NEW
    finally:
        sys.path.pop(0)
    p = {}; p.update(OLD); p.update(NEW)
    return p


def main():
    os.makedirs(DUMP_DIR, exist_ok=True)
    prompts = load_prompts()
    if SELECTED is None:
        selected = sorted(prompts.items())
    else:
        missing = [l for l in SELECTED if l not in prompts]
        if missing:
            print(f"WARN: missing labels {missing}")
        selected = [(l, prompts[l]) for l in SELECTED if l in prompts]
    print(f"Dumping FFN inputs for {len(selected)} prompts "
          f"× layers={LAYERS} × gen={GEN}\n")
    t0 = time.time()
    for i, (label, prompt) in enumerate(selected):
        env = os.environ.copy()
        env.update({
            "BITNET_DUMP_FFN_INPUTS_DIR": DUMP_DIR,
            "BITNET_DUMP_FFN_INPUTS_LAYERS": LAYERS,
            "BITNET_DUMP_LABEL": label,
            "BITNET_KV_EVICT_MODE": "qsigdist",
            "BITNET_KV_WINDOW": str(WINDOW),
            "BITNET_ATTN_FIXED_TAU": "5000",
        })
        cmd = [HARNESS, DATA, "--prompt-tokens", prompt, "--gen", str(GEN)]
        ts = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - ts
        ok = proc.returncode == 0
        print(f"  [{i+1:>2}/{len(selected)}] {label:<24} {elapsed:>5.1f}s  "
              f"{'OK' if ok else 'FAIL'}", flush=True)
    print(f"\nTotal elapsed: {time.time() - t0:.0f}s")
    n_files = len(os.listdir(DUMP_DIR))
    print(f"Files in {DUMP_DIR}: {n_files}")


if __name__ == "__main__":
    main()
