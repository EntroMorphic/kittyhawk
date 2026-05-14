"""Red-team: verify meta-mode is deterministic.

Same env, same prompt, same weights → same tokens twice. If this
fails, the iter Δs are noisy in a way the architectural test never
accounted for.

Sample: 3 prompts × 2 candidates (one from existing anchors, one
hypothetical) × 2 repeats. PASS = all 6 (prompt × candidate) cells
produce identical tokens across both repeats.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

HARNESS = "build/gesh/bitnet_harness"
DATA    = "data/bitnet_b158_2b4t.bin"
WINDOW  = 16
GEN_N   = 24
OUTDIR  = "experiments/phase_zeta/results/redteam_meta_determinism"

SAMPLE_LABELS = ["long_desc_forest", "code_python_fn", "logic_conditional2"]
CANDIDATES = [(0, -1, 1), (1, 0, 1)]  # iter 1 and iter 6 from anchor store


def load_prompts() -> dict[str, str]:
    sys.path.insert(0, "experiments/phase_zeta")
    try:
        from n50_battery import PROMPTS as OLD
        from n100_battery_incremental import NEW_PROMPTS as NEW
    finally:
        sys.path.pop(0)
    p = {}; p.update(OLD); p.update(NEW)
    return p


def run(prompt: str, w) -> list[int] | None:
    env = os.environ.copy()
    env["BITNET_KV_EVICT_MODE"]  = "meta"
    env["BITNET_KV_WINDOW"]      = str(WINDOW)
    env["BITNET_KV_EVICT_W_R"]   = str(w[0])
    env["BITNET_KV_EVICT_W_KK"]  = str(w[1])
    env["BITNET_KV_EVICT_W_QK"]  = str(w[2])
    env["BITNET_ATTN_FIXED_TAU"] = "5000"
    cmd = [HARNESS, DATA, "--prompt-tokens", prompt, "--gen", str(GEN_N)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    m = re.search(r"generated tokens\s*=\s*([\d\s\-]+)", out)
    return [int(t) for t in m.group(1).strip().split()] if m else None


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    prompts = load_prompts()
    print(f"Red-team: meta-mode determinism")
    print(f"  {len(SAMPLE_LABELS)} prompts × {len(CANDIDATES)} candidates × 2 repeats\n")
    results = []
    fails = 0
    t0 = time.time()
    for label in SAMPLE_LABELS:
        for w in CANDIDATES:
            tok_a = run(prompts[label], w)
            tok_b = run(prompts[label], w)
            match = tok_a == tok_b and tok_a is not None
            if not match:
                fails += 1
            results.append({
                "label": label,
                "weights": list(w),
                "tokens_a": tok_a,
                "tokens_b": tok_b,
                "identical": match,
            })
            mark = "OK" if match else "FAIL"
            print(f"  {label:<24} meta{w} repeat-pair  {mark}", flush=True)
    elapsed = time.time() - t0
    total = len(results)
    print(f"\nElapsed: {elapsed:.0f}s   PASS: {total-fails}/{total}")
    with open(os.path.join(OUTDIR, "results.json"), "w") as f:
        json.dump({"results": results, "fails": fails, "total": total}, f, indent=2)
    if fails:
        for r in results:
            if not r["identical"]:
                print(f"  diff: {r['label']} meta{tuple(r['weights'])}")
                print(f"    a: {r['tokens_a']}")
                print(f"    b: {r['tokens_b']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
