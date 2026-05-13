"""Plan B red-team harness battery on REAL natural-language prompts.

Red-team B uncovered that the original eviction_battery.py prompts
("capital_france", "short_a", etc.) were gibberish token sequences
that decode to '" car<p {\\n minorobject' under the actual model
tokenizer. The Phase ζ harness territory test (including plan B's
qsigdist loss claim) was measured on out-of-distribution inputs.

This script reruns the territory test with 20 natural-language
prompts (tokenized via microsoft/bitnet-b1.58-2B-4T-bf16 tokenizer)
at window=16, the regime where plan B claimed qsigdist underperformed
random.

Aggregate match-rate per mode + 95% CI via prompt-resampled bootstrap.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time

import numpy as np

HARNESS = "build/gesh/bitnet_harness"
DATA    = "data/bitnet_b158_2b4t.bin"
OUTDIR  = "experiments/phase_zeta/results/redteam_b_harness"
WINDOW  = 16
GEN_N   = 24
RNG_SEED = 20260512

PROMPTS = {
    "q_capital_france":   ("128000,3923,374,279,6864,315,9822,30", "What is the capital of France?"),
    "q_capital_japan":    ("128000,3923,374,279,6864,315,6457,30", "What is the capital of Japan?"),
    "q_largest_planet":   ("128000,3923,374,279,7928,11841,304,1057,13238,1887,30", "What is the largest planet in our solar system?"),
    "q_who_hamlet":       ("128000,15546,6267,279,1514,9777,1169,30", "Who wrote the play Hamlet?"),
    "def_photosynth":     ("128000,32872,74767,374,279,1920,555,902", "Photosynthesis is the process by which"),
    "def_gravity":        ("128000,39509,374,264,16188,5457,430", "Gravity is a fundamental force that"),
    "cont_once":          ("128000,12805,5304,264,892,11,304,264,26135,3117,3201,11", "Once upon a time, in a kingdom far away,"),
    "cont_dark_stormy":   ("128000,2181,574,264,6453,323,13766,88,3814,994,279", "It was a dark and stormy night when the"),
    "math_add":           ("128000,717,5636,220,22,17239", "12 plus 7 equals"),
    "math_mul":           ("128000,38120,3115,8223,17239", "Five times eight equals"),
    "color_sky":          ("128000,791,1933,315,279,13180,389,264,2867,1938,374", "The color of the sky on a clear day is"),
    "reasoning_water":    ("128000,29353,90055,520,220,1041,12628,62447,520", "Water boils at 100 degrees Celsius at"),
    "instr_translate":    ("128000,28573,279,2768,311,8753,25,22691,11,1268,527,499,30", "Translate the following to French: Hello, how are you?"),
    "instr_summary":      ("128000,644,832,11914,11,63179,1148,7397,74767,3445,25", "In one sentence, summarize what photosynthesis means:"),
    "dialog_greet":       ("128000,32,25,22691,11,1268,527,499,3432,30,426,25", "A: Hello, how are you today? B:"),
    "idiom_break_ice":    ("128000,791,17571,364,9137,279,10054,6,3445", "The phrase 'break the ice' means"),
    "long_desc_forest":   ("128000,34564,2949,279,14154,13952,11,1405,40120,20025,8813,279,78343,84408,5015,11", "Deep within the ancient forest..."),
    "long_lab":           ("128000,791,28568,15884,24257,279,73757,11,38936,279,15332,369,904,1879,315", "The scientist carefully adjusted..."),
    "long_recipe":        ("128000,1271,1304,264,4832,297,2727,1169,11,1176,17944,1403,3544,19335,1139,264,19763,323,41759,1124,3871,449", "To make a perfect omelet..."),
    "long_argument":      ("128000,16179,23531,617,18784,430,279,4947,374,2288,34348,11,15879,10519,430", "Although critics have argued..."),
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


def run_one(label: str, prompt: str, mode: str, window: int) -> dict:
    env = os.environ.copy()
    env.update(mode_env(mode, window))
    cmd = [HARNESS, DATA, "--prompt-tokens", prompt, "--gen", str(GEN_N)]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0
    out = proc.stdout + proc.stderr
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f"{label}_{mode}.log"), "w") as f:
        f.write(out)
    m = re.search(r"generated tokens\s*=\s*([\d\s\-]+)", out)
    tokens = [int(t) for t in m.group(1).strip().split()] if m else None
    return {"label": label, "mode": mode, "tokens": tokens,
            "elapsed_s": elapsed, "exit": proc.returncode}


def match_rate(base, cand):
    n = min(len(base), len(cand))
    return sum(1 for i in range(n) if base[i] == cand[i]) / n if n else float('nan')


def divergence(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]: return i
    return min(len(a), len(b))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    modes = ["no_evict", "fifo", "random", "sigdist", "qsigdist"]
    all_results = []
    print(f"{'label':<22} {'mode':<10} {'elapsed':>8}  tokens")
    print("-" * 110)
    for label, (prompt, _text) in PROMPTS.items():
        for mode in modes:
            r = run_one(label, prompt, mode, WINDOW)
            all_results.append(r)
            toks = r["tokens"] or []
            tok_str = " ".join(str(t) for t in toks[:14])
            print(f"{label:<22} {mode:<10} {r['elapsed_s']:>7.2f}s  {tok_str}")

    # Aggregate
    by_lm = {(r["label"], r["mode"]): r for r in all_results}
    print(f"\n=== window={WINDOW} aggregate (N={len(PROMPTS)} natural-language prompts) ===\n")
    rng = np.random.default_rng(RNG_SEED)
    for mode in ("fifo", "random", "sigdist", "qsigdist"):
        per_prompt_match = []
        per_prompt_div = []
        for label in PROMPTS:
            base = by_lm.get((label, "no_evict"), {}).get("tokens")
            cand = by_lm.get((label, mode), {}).get("tokens")
            if not base or not cand: continue
            per_prompt_match.append(match_rate(base, cand))
            per_prompt_div.append(divergence(base, cand))
        arr = np.array(per_prompt_match)
        # Bootstrap CI over prompts
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(arr), size=len(arr))
            boots.append(arr[idx].mean())
        ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])
        print(f"  {mode:<9} mean_div={np.mean(per_prompt_div):>5.1f}  "
              f"match={arr.mean()*100:>5.1f}%  "
              f"95%CI=[{ci_lo*100:>5.1f}, {ci_hi*100:>5.1f}]%  "
              f"std={arr.std()*100:>5.1f}pp  N={len(arr)}")

    # Pairwise (vs random)
    print(f"\n=== Paired comparison vs random (Δ match per prompt, 95% CI) ===\n")
    rng = np.random.default_rng(RNG_SEED)
    rand_match = np.array([
        match_rate(by_lm[(l, "no_evict")]["tokens"], by_lm[(l, "random")]["tokens"])
        for l in PROMPTS
        if by_lm.get((l, "no_evict"), {}).get("tokens") and by_lm.get((l, "random"), {}).get("tokens")
    ])
    for mode in ("fifo", "sigdist", "qsigdist"):
        diffs = []
        for label in PROMPTS:
            base = by_lm.get((label, "no_evict"), {}).get("tokens")
            cand = by_lm.get((label, mode), {}).get("tokens")
            randc = by_lm.get((label, "random"), {}).get("tokens")
            if not (base and cand and randc): continue
            diffs.append(match_rate(base, cand) - match_rate(base, randc))
        d = np.array(diffs)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(d), size=len(d))
            boots.append(d[idx].mean())
        ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])
        wins = int(np.sum(d > 0))
        losses = int(np.sum(d < 0))
        ties = int(np.sum(d == 0))
        print(f"  {mode:<9} Δ vs random = {d.mean()*100:+5.1f}pp  "
              f"95%CI=[{ci_lo*100:+5.1f}, {ci_hi*100:+5.1f}]pp  "
              f"wins/ties/losses = {wins}/{ties}/{losses} of {len(d)}")

    out_path = os.path.join(OUTDIR, "battery_results.json")
    with open(out_path, "w") as f:
        json.dump({"config": {"window": WINDOW, "gen_n": GEN_N},
                   "trials": all_results}, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
