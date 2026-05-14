"""Red-team: verify meta-mode anchor-point bit-identicality.

The meta-routing iteration arc rests on the assumption that the
parameterized harness mode `meta` reproduces the fixed-policy modes
(fifo, sigdist, qsigdist) bit-identically when its weights map to
those fixed points:

    fifo     ≡ meta(-1,  0,  0)
    sigdist  ≡ meta( 0, +1,  0)
    qsigdist ≡ meta( 0,  0, +1)

If any of these mappings produce different token outputs, the four
"original anchor" Δs (from fixed modes) live on a different response
surface from the iter Δs (from meta mode), and the kernel's
calibration result is contaminated.

Run a sample of N=10 prompts × 3 fixed/meta pairs. Diff token-by-token.

PASS criterion: 100% bit-identical across all 30 (prompt × pair) cells.
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
OUTDIR  = "experiments/phase_zeta/results/redteam_meta_fixedpoints"

# 10 prompts: a representative slice across long/short, code/prose,
# logic/QA. Labels come from the N=100 battery.
SAMPLE_LABELS = [
    "long_desc_forest",       # long prose
    "code_python_fn",         # code
    "poetry_haiku",           # short structured
    "dialog_qa",              # dialog
    "error_traceback",        # adversarial
    "instr_explain",          # instruction-following
    "math_word_problem",      # quantitative
    "q_dna_full",             # QA + technical
    "history_moon",           # short factual
    "logic_conditional2",     # logic
]

# (fixed_mode_name, meta_weights)
PAIRS = [
    ("fifo",     (-1, 0, 0)),
    ("sigdist",  (0, +1, 0)),
    ("qsigdist", (0, 0, +1)),
]


def load_prompts() -> dict[str, str]:
    sys.path.insert(0, "experiments/phase_zeta")
    try:
        from n50_battery import PROMPTS as OLD
        from n100_battery_incremental import NEW_PROMPTS as NEW
    finally:
        sys.path.pop(0)
    p = {}
    p.update(OLD); p.update(NEW)
    return p


def run(label: str, prompt: str, mode: str, w=None) -> list[int] | None:
    env = os.environ.copy()
    env["BITNET_KV_EVICT_MODE"]  = mode
    env["BITNET_KV_WINDOW"]      = str(WINDOW)
    env["BITNET_ATTN_FIXED_TAU"] = "5000"
    if w is not None:
        env["BITNET_KV_EVICT_W_R"]  = str(w[0])
        env["BITNET_KV_EVICT_W_KK"] = str(w[1])
        env["BITNET_KV_EVICT_W_QK"] = str(w[2])
    cmd = [HARNESS, DATA, "--prompt-tokens", prompt, "--gen", str(GEN_N)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    m = re.search(r"generated tokens\s*=\s*([\d\s\-]+)", out)
    return [int(t) for t in m.group(1).strip().split()] if m else None


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    prompts = load_prompts()
    missing = [l for l in SAMPLE_LABELS if l not in prompts]
    if missing:
        print(f"ERROR: missing labels {missing}")
        sys.exit(1)
    print(f"Red-team: meta-mode fixed-point bit-identicality")
    print(f"  {len(SAMPLE_LABELS)} prompts × {len(PAIRS)} fixed/meta pairs")
    print(f"  window={WINDOW}, gen={GEN_N}, tau=5000\n")

    results = []
    fails = 0
    t0 = time.time()
    for i, label in enumerate(SAMPLE_LABELS):
        prompt = prompts[label]
        for fixed_name, w in PAIRS:
            tok_fixed = run(label, prompt, fixed_name)
            tok_meta  = run(label, prompt, "meta", w)
            match = tok_fixed == tok_meta and tok_fixed is not None
            if not match:
                fails += 1
            results.append({
                "label": label,
                "fixed_mode": fixed_name,
                "meta_w": list(w),
                "fixed_tokens": tok_fixed,
                "meta_tokens": tok_meta,
                "identical": match,
            })
            mark = "OK" if match else "FAIL"
            print(f"  [{i+1:>2}/{len(SAMPLE_LABELS)}] {label:<24} {fixed_name:>9} vs meta{w}  {mark}",
                  flush=True)
    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.0f}s   PASS: {len(results)-fails}/{len(results)}   FAIL: {fails}")
    out_path = os.path.join(OUTDIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "fails": fails, "total": len(results)}, f, indent=2)
    print(f"Results: {out_path}")
    if fails > 0:
        print("\nFAIL details:")
        for r in results:
            if not r["identical"]:
                print(f"  {r['label']:<24} {r['fixed_mode']:>9} vs meta{tuple(r['meta_w'])}")
                print(f"    fixed: {r['fixed_tokens']}")
                print(f"    meta:  {r['meta_tokens']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
