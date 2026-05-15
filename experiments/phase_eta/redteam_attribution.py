"""RT#5+#6: per-prompt attribution.

For each prompt at n_min=10 (the headline result), extract:
  - routing fraction (% of FFN calls that took routed path)
  - match-rate vs dense

Question: are the perfect-match prompts ones where routing % was LOW
(i.e., dense fallback dominated, so the result trivially matches dense)?
Or are they prompts where routing fired and produced near-dense output?

If the FORMER, the 0.596 mean is a confound: low-routing prompts
trivially match, high-routing prompts diverge, and the mean is just
"how often did routing fire."

If the LATTER, routing IS preserving generation quality on the
prompts where it fires.
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
DICT_NMIN10 = os.path.join(THIS, "results/lsh_ffn_dict_nmin10.bin")
DICT_NMIN1  = os.path.join(THIS, "results/lsh_ffn_dict.bin")

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


def run_one(prompt, dict_path, lsh_layers):
    env = os.environ.copy()
    env.update({
        "BITNET_KV_EVICT_MODE": "qsigdist",
        "BITNET_KV_WINDOW": str(WINDOW),
        "BITNET_ATTN_FIXED_TAU": "5000",
    })
    if dict_path:
        env["BITNET_FFN_LSH_DICT"] = dict_path
        env["BITNET_FFN_LSH_LAYERS"] = lsh_layers
    cmd = [HARNESS, DATA, "--prompt-tokens", prompt, "--gen", str(GEN)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    m = re.search(r"generated tokens\s*=\s*([\d\s\-]+)", out)
    toks = [int(t) for t in m.group(1).strip().split()] if m else None
    routed_pct = None
    routed_count = 0; fallback_count = 0
    rm = re.search(r"LSH FFN routed/fallback\s*=\s*(\d+)/(\d+)\s*\(([\d.]+)% routed\)", out)
    if rm:
        routed_count = int(rm.group(1))
        fallback_count = int(rm.group(2))
        routed_pct = float(rm.group(3))
    return toks, routed_count, fallback_count, routed_pct


def match_rate(a, b):
    if not a or not b: return 0.0
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]) / n if n else 0.0


def main():
    prompts = load_prompts()
    selected = [(l, prompts[l]) for l in LABELS if l in prompts]
    print(f"RT#5+#6 attribution: routing fraction vs match-rate per prompt")
    print(f"Routing layer = L15 only\n")

    t0 = time.time()
    print("Step 1: dense baseline")
    dense = {}
    for label, prompt in selected:
        toks, _, _, _ = run_one(prompt, None, "")
        dense[label] = toks
        print(f"  {label:<24} OK", flush=True)

    print("\nStep 2: nmin=10 (headline config)")
    nmin10 = {}
    for label, prompt in selected:
        toks, rc, fc, pct = run_one(prompt, DICT_NMIN10, "15")
        nmin10[label] = (toks, rc, fc, pct)
        mr = match_rate(dense[label], toks)
        print(f"  {label:<24} routing={pct:>5.1f}%  match={mr:.3f}", flush=True)

    print("\nStep 3: nmin=1 (all routed comparison)")
    nmin1 = {}
    for label, prompt in selected:
        toks, rc, fc, pct = run_one(prompt, DICT_NMIN1, "15")
        nmin1[label] = (toks, rc, fc, pct)
        mr = match_rate(dense[label], toks)
        print(f"  {label:<24} routing={pct:>5.1f}%  match={mr:.3f}", flush=True)

    print(f"\n{'='*70}")
    print("ATTRIBUTION TABLE — n_min=10")
    print(f"{'='*70}")
    print(f"  {'prompt':<24}  {'routing %':>9}  {'match':>6}  {'class':>20}")
    rows = []
    for label, _ in selected:
        toks, rc, fc, pct = nmin10[label]
        mr = match_rate(dense[label], toks)
        if pct is None: continue
        rows.append((label, pct, mr))
        cls = ""
        if pct < 5 and mr >= 0.95: cls = "low-route, high-match (FALLBACK WIN)"
        elif pct < 5 and mr <  0.5: cls = "low-route, low-match (NOISE)"
        elif pct >= 30 and mr >= 0.5: cls = "high-route, high-match (ROUTING WIN ✓)"
        elif pct >= 30 and mr <  0.5: cls = "high-route, low-match (ROUTING FAIL)"
        else: cls = "mixed"
        print(f"  {label:<24}  {pct:>8.1f}  {mr:>6.3f}  {cls}")

    pcts = np.array([r[1] for r in rows])
    mrs = np.array([r[2] for r in rows])
    print(f"\nCorrelation routing% vs match: r = {np.corrcoef(pcts, mrs)[0,1]:.3f}")
    print(f"Mean routing %: {pcts.mean():.1f}")
    print(f"Mean match: {mrs.mean():.3f}")
    print(f"\nStratified:")
    low_route = mrs[pcts < 10]
    high_route = mrs[pcts >= 30]
    if len(low_route):
        print(f"  prompts with routing < 10%:  n={len(low_route)}, mean match = {low_route.mean():.3f}")
    if len(high_route):
        print(f"  prompts with routing >= 30%: n={len(high_route)}, mean match = {high_route.mean():.3f}")

    print(f"\n{'='*70}")
    print("COMPARISON — nmin=1 (all-routed)")
    print(f"{'='*70}")
    print(f"  {'prompt':<24}  {'routing %':>9}  {'match':>6}")
    rows1 = []
    for label, _ in selected:
        toks, rc, fc, pct = nmin1[label]
        mr = match_rate(dense[label], toks)
        if pct is None: continue
        rows1.append((label, pct, mr))
        print(f"  {label:<24}  {pct:>8.1f}  {mr:>6.3f}")
    pcts1 = np.array([r[1] for r in rows1])
    mrs1 = np.array([r[2] for r in rows1])
    print(f"\nMean routing %: {pcts1.mean():.1f}")
    print(f"Mean match: {mrs1.mean():.3f}")
    print(f"Correlation routing% vs match: r = {np.corrcoef(pcts1, mrs1)[0,1]:.3f}")

    print(f"\nElapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
