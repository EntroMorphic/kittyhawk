"""Track B: N=50 eviction battery (settle the qsigdist +6pp trend).

Per glyph_gaps_2026-05-13_synthesize.md Track B: extend the 20-prompt
red-team B harness battery to 50 natural-language prompts. If qsigdist's
Δ vs random reaches statistical significance (95% CI excludes zero),
the substrate-eviction territory verdict closes positive. Otherwise the
arc closes as "inconclusive with positive trend, parked."

Runs at window=16 only (the regime where qsigdist's trend was visible
in the N=20 battery). Skips window=8 (all policies fail) and window=32
(no eviction pressure).
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
OUTDIR  = "experiments/phase_zeta/results/n50_battery"
WINDOW  = 16
GEN_N   = 24
RNG_SEED = 20260513

PROMPTS = {
    "q_capital_france":     "128000,3923,374,279,6864,315,9822,30",
    "q_capital_japan":      "128000,3923,374,279,6864,315,6457,30",
    "q_largest_planet":     "128000,3923,374,279,7928,11841,304,1057,13238,1887,30",
    "q_who_hamlet":         "128000,15546,6267,279,1514,9777,1169,30",
    "def_photosynth":       "128000,32872,74767,374,279,1920,555,902",
    "def_gravity":          "128000,39509,374,264,16188,5457,430",
    "cont_once":            "128000,12805,5304,264,892,11,304,264,26135,3117,3201,11",
    "cont_dark_stormy":     "128000,2181,574,264,6453,323,13766,88,3814,994,279",
    "math_add":             "128000,717,5636,220,22,17239",
    "math_mul":             "128000,38120,3115,8223,17239",
    "color_sky":            "128000,791,1933,315,279,13180,389,264,2867,1938,374",
    "reasoning_water":      "128000,29353,90055,520,220,1041,12628,62447,520",
    "instr_translate":      "128000,28573,279,2768,311,8753,25,22691,11,1268,527,499,30",
    "instr_summary":        "128000,644,832,11914,11,63179,1148,7397,74767,3445,25",
    "dialog_greet":         "128000,32,25,22691,11,1268,527,499,3432,30,426,25",
    "idiom_break_ice":      "128000,791,17571,364,9137,279,10054,6,3445",
    "long_desc_forest":     "128000,34564,2949,279,14154,13952,11,1405,40120,20025,8813,279,78343,84408,5015,11",
    "long_lab":             "128000,791,28568,15884,24257,279,73757,11,38936,279,15332,369,904,1879,315",
    "long_recipe":          "128000,1271,1304,264,4832,297,2727,1169,11,1176,17944,1403,3544,19335,1139,264,19763,323,41759,1124,3871,449",
    "long_argument":        "128000,16179,23531,617,18784,430,279,4947,374,2288,34348,11,15879,10519,430",
    "code_python_fn":       "128000,755,76798,1471,997,262,422,308,2717,220,16,512,286,471,308,198,262,471",
    "code_loop":            "128000,2000,602,304,2134,7,605,997,262,1194,7",
    "code_class":           "128000,1058,21995,512,262,711,1328,2381,3889,726,11,836,997,286,659,2710,284",
    "code_import":          "128000,475,8760,439,2660,271,755,3152,11179,997,262,471",
    "code_sql":             "128000,4963,836,11,4325,4393,3932,5401,4325,871,220,972,15888,7866",
    "poetry_haiku":         "128000,1163,5515,50823,7085,4498,198,31631,398,389,279,11594,36670",
    "poetry_iambic":        "128000,2059,543,358,9616,40344,311,264,7474,596,1938,5380,1016,283,1989,810,17104,323,810",
    "dialog_qa":            "128000,48,25,2650,656,11012,636,4907,5380,32,25,50298,636,4907,1555",
    "dialog_multi":         "128000,62786,25,358,3077,1027,7422,922,430,2363,627,33488,25,16299,832,5380,62786,25",
    "technical_ml":         "128000,32,43678,374,264,30828,4009,18112,430,5829,659,12,54203,311",
    "technical_physics":    "128000,791,2132,2383,315,30945,80011,5415,430",
    "technical_chem":       "128000,50,47876,82882,14091,18685,304,3090,1606",
    "error_traceback":      "128000,6687,1445,320,3646,3293,1650,1566,997,220,2958,330,680,7345,498,1584,220,717,11,304,366,4450,397,262,1121,284,12849,7",
    "error_message":        "128000,1480,25,35755,1373,3424,364,609,6,315,5732",
    "instruct_step":        "128000,8468,220,16,25,5377,279,6462,627,8468,220,17,25,11016,279,29219,4632,627,8468,220,18,25",
    "instruct_recipe":      "128000,5451,11,864,20559,279,24276,311,220,8652,12628,13,5112,11",
    "hypothesis":           "128000,2746,279,2853,315,13238,21988,9731,311,4498,11,1243",
    "comparison":           "128000,44179,8776,5528,11,902,17631,389,11630,3477,11,420,5603",
    "negation":             "128000,2181,374,539,279,1162,430,682,20229,649,11722,26,369,3187,11",
    "quantifier":           "128000,11769,5575,304,279,538,14976,872,16720,3734,369",
    "temporal":             "128000,51377,358,4024,311,279,3637,11,3432,358,1097,3318,505,2162,11,323,16986,358,690",
    "conditional":          "128000,2746,433,62555,16986,11,584,690,1205,311",
    "causal":               "128000,18433,279,4817,72389,660,11,279,1841",
    "definition_term":      "128000,22333,6975,374,264,27084,315,21075,11478,430,18065",
    "history_fact":         "128000,10343,5111,8105,9670,304",
    "geography_river":      "128000,791,22807,15140,304,279,1917,374,279",
    "biology_cell":         "128000,791,55042,4298,315,264,2849,527,8647,369",
    "idiom_spill":          "128000,4599,1364,1071,364,2203,484,279,27994,2965,1364,8967",
    "idiom_back":           "128000,1548,1071,433,574,264,6710,315,19692,11,7438,430,279,3465,574",
    "longform_essay":       "128000,791,25563,22910,43593,24411,3823,8396,1555,2380,6156,24717,25,1176,11,279,7852,2065,315,5788,11618,26,2132,11",
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
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
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
    print(f"N={len(PROMPTS)} prompts × {len(modes)} modes × window={WINDOW} × gen={GEN_N}")
    print(f"{'label':<22} {'mode':<10} {'elapsed':>9}  tokens[:14]")
    print("-" * 110)
    for i, (label, prompt) in enumerate(PROMPTS.items()):
        for mode in modes:
            r = run_one(label, prompt, mode, WINDOW)
            all_results.append(r)
            toks = r["tokens"] or []
            tok_str = " ".join(str(t) for t in toks[:14])
            print(f"[{i+1:>2d}/{len(PROMPTS)}] {label:<14} {mode:<10} {r['elapsed_s']:>7.2f}s  {tok_str}",
                  flush=True)

    # Save raw trials immediately so the JSON is durable
    out_path = os.path.join(OUTDIR, "battery_results.json")
    with open(out_path, "w") as f:
        json.dump({"config": {"window": WINDOW, "gen_n": GEN_N, "n_prompts": len(PROMPTS)},
                   "trials": all_results}, f, indent=2)

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
        boots = []
        for _ in range(5000):
            idx = rng.integers(0, len(arr), size=len(arr))
            boots.append(arr[idx].mean())
        ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])
        print(f"  {mode:<9} mean_div={np.mean(per_prompt_div):>5.1f}  "
              f"match={arr.mean()*100:>5.1f}%  "
              f"95%CI=[{ci_lo*100:>5.1f}, {ci_hi*100:>5.1f}]%  "
              f"std={arr.std()*100:>5.1f}pp  N={len(arr)}")

    # Paired comparison vs random
    print(f"\n=== Paired Δ vs random (95% CI from prompt-resampled bootstrap) ===\n")
    rng = np.random.default_rng(RNG_SEED)
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
        for _ in range(5000):
            idx = rng.integers(0, len(d), size=len(d))
            boots.append(d[idx].mean())
        ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])
        wins = int(np.sum(d > 0))
        losses = int(np.sum(d < 0))
        ties = int(np.sum(d == 0))
        sig = "**SIGNIFICANT**" if (ci_lo > 0 or ci_hi < 0) else "(not significant)"
        print(f"  {mode:<9} Δ={d.mean()*100:+5.1f}pp  "
              f"95%CI=[{ci_lo*100:+5.1f}, {ci_hi*100:+5.1f}]pp  "
              f"wins/ties/losses = {wins}/{ties}/{losses} of {len(d)}   {sig}")

    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
