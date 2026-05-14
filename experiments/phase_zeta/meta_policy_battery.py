"""Empirical validation of layer-3 meta-routing predicted candidates.

Runs the candidate policy meta(w_r, w_kk, w_qk) = (0, -1, +1) across
the same 100 prompts used in the N=100 closeout (50 old + 50 new),
then compares the observed Δ vs random to the layer-3 model's
prediction of +13.3pp.

Pass criterion (architectural): observed Δ is in [+5pp, +20pp]. Loose
because the additive model has 4 anchors × 4 params = zero degrees of
freedom; any second-order effect along the (w_kk = -1) axis would
push the observation away from the linear prediction.

Stronger claim — if observed Δ > qsigdist's +6.4pp (CI [+1.7, +11.2])
with non-overlapping CI, then meta-routing has found a strictly
better policy than the best hand-coded one. That's the headline.

Usage:
  python experiments/phase_zeta/meta_policy_battery.py
  # then analyze with --analyze:
  python experiments/phase_zeta/meta_policy_battery.py --analyze
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HARNESS = "build/gesh/bitnet_harness"
DATA    = "data/bitnet_b158_2b4t.bin"
OUTDIR  = "experiments/phase_zeta/results/meta_policy_battery"
N50_JSON  = "experiments/phase_zeta/results/n50_battery/battery_results.json"
N100_JSON = "experiments/phase_zeta/results/n100_incremental/battery_results.json"
WINDOW  = 16
GEN_N   = 24
RNG_SEED = 20260513

# The candidate from the layer-3 prediction. Trit weights in the Python
# convention from meta_routing.py. The harness translates these for its
# internal "evict argmax" convention.
CANDIDATE = (0, -1, 1)


def load_n100_prompts() -> dict[str, str]:
    """Reconstruct the 100-prompt dict by walking the existing N=50 + new-50
    result files (each label appears in only one of them)."""
    prompts = {}
    # Old N=50: in tokenize_prompts.py — but we can just import the dict
    # from n50_battery.py:
    sys.path.insert(0, "experiments/phase_zeta")
    try:
        from n50_battery import PROMPTS as OLD_PROMPTS
        from n100_battery_incremental import NEW_PROMPTS
    finally:
        sys.path.pop(0)
    prompts.update(OLD_PROMPTS)
    prompts.update(NEW_PROMPTS)
    assert len(prompts) == 100, f"expected 100 prompts, got {len(prompts)}"
    return prompts


def run_one(label: str, prompt: str) -> dict:
    env = os.environ.copy()
    env["BITNET_KV_EVICT_MODE"]  = "meta"
    env["BITNET_KV_WINDOW"]      = str(WINDOW)
    env["BITNET_KV_EVICT_W_R"]   = str(CANDIDATE[0])
    env["BITNET_KV_EVICT_W_KK"]  = str(CANDIDATE[1])
    env["BITNET_KV_EVICT_W_QK"]  = str(CANDIDATE[2])
    env["BITNET_ATTN_FIXED_TAU"] = "5000"
    cmd = [HARNESS, DATA, "--prompt-tokens", prompt, "--gen", str(GEN_N)]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0
    out = proc.stdout + proc.stderr
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f"{label}_meta_0_-1_+1.log"), "w") as f:
        f.write(out)
    m = re.search(r"generated tokens\s*=\s*([\d\s\-]+)", out)
    tokens = [int(t) for t in m.group(1).strip().split()] if m else None
    return {"label": label, "mode": f"meta_{CANDIDATE[0]}_{CANDIDATE[1]}_{CANDIDATE[2]}",
            "tokens": tokens, "elapsed_s": elapsed, "exit": proc.returncode}


def match_rate(a, b):
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]) / n if n else 0.0


def run_battery():
    os.makedirs(OUTDIR, exist_ok=True)
    prompts = load_n100_prompts()
    print(f"Candidate: meta{CANDIDATE} — layer-3 prediction +13.3pp vs random")
    print(f"Running N={len(prompts)} prompts × 1 mode × window={WINDOW} × gen={GEN_N}\n")
    results = []
    out_path = os.path.join(OUTDIR, "battery_results.json")
    for i, (label, prompt) in enumerate(prompts.items()):
        r = run_one(label, prompt)
        results.append(r)
        toks = r["tokens"] or []
        tok_str = " ".join(str(t) for t in toks[:10])
        print(f"[{i+1:>3}/{len(prompts)}] {label:<24} {r['elapsed_s']:>6.1f}s  {tok_str}",
              flush=True)
        # Save incrementally
        with open(out_path, "w") as f:
            json.dump({"config": {"window": WINDOW, "gen_n": GEN_N,
                                  "candidate": list(CANDIDATE),
                                  "n_prompts": len(prompts)},
                       "trials": results}, f, indent=2)
    print(f"\nResults: {out_path}")


def analyze():
    """Compute Δ vs random for the candidate and compare to the
    layer-3 prediction. Bootstrap CI for honesty."""
    candidate_path = os.path.join(OUTDIR, "battery_results.json")
    if not os.path.exists(candidate_path):
        print(f"ERROR: {candidate_path} not found. Run without --analyze first.")
        sys.exit(1)

    # Load the candidate trials
    with open(candidate_path) as f:
        cand_data = json.load(f)
    cand_trials = cand_data["trials"]
    cand_by_label = {t["label"]: t["tokens"] for t in cand_trials}

    # Load the pooled N=100 (random + no_evict baselines)
    with open(N50_JSON) as f:
        old = json.load(f)["trials"]
    with open(N100_JSON) as f:
        new = json.load(f)["trials"]
    pooled = old + new
    base_by_lm = {(t["label"], t["mode"]): t["tokens"] for t in pooled}

    # Compute per-prompt Δ vs random
    diffs = []
    qsig_diffs = []  # for comparison with qsigdist on the same prompt set
    for label, cand_tokens in cand_by_label.items():
        no_evict = base_by_lm.get((label, "no_evict"))
        randc    = base_by_lm.get((label, "random"))
        qsig     = base_by_lm.get((label, "qsigdist"))
        if not (no_evict and cand_tokens and randc and qsig):
            print(f"WARNING: missing data for {label}, skipping")
            continue
        diffs.append(match_rate(no_evict, cand_tokens) - match_rate(no_evict, randc))
        qsig_diffs.append(match_rate(no_evict, qsig) - match_rate(no_evict, randc))

    d = np.array(diffs)
    q = np.array(qsig_diffs)
    rng = np.random.default_rng(RNG_SEED)

    # Bootstrap CI for the candidate's Δ vs random
    boots_d = []
    boots_q = []
    boots_diff = []  # paired: candidate vs qsigdist
    for _ in range(5000):
        idx = rng.integers(0, len(d), size=len(d))
        boots_d.append(d[idx].mean())
        boots_q.append(q[idx].mean())
        boots_diff.append((d[idx] - q[idx]).mean())
    ci_d  = np.quantile(boots_d, [0.025, 0.975])
    ci_q  = np.quantile(boots_q, [0.025, 0.975])
    ci_dq = np.quantile(boots_diff, [0.025, 0.975])

    print("=" * 70)
    print(f"Meta-routing empirical validation: candidate {CANDIDATE}")
    print("=" * 70)
    print()
    print(f"Layer-3 prediction: +13.3pp")
    print()
    print(f"meta{CANDIDATE} Δ vs random:")
    print(f"    Δ = {d.mean()*100:+5.2f}pp   95% CI [{ci_d[0]*100:+5.2f}, {ci_d[1]*100:+5.2f}]pp")
    print(f"qsigdist Δ vs random (same prompts):")
    print(f"    Δ = {q.mean()*100:+5.2f}pp   95% CI [{ci_q[0]*100:+5.2f}, {ci_q[1]*100:+5.2f}]pp")
    print(f"Paired (meta − qsigdist):")
    print(f"    Δ = {(d-q).mean()*100:+5.2f}pp   95% CI [{ci_dq[0]*100:+5.2f}, {ci_dq[1]*100:+5.2f}]pp")
    print()
    print(f"Wins/ties/losses vs qsigdist: "
          f"{int(np.sum(d > q))} / {int(np.sum(d == q))} / {int(np.sum(d < q))} of {len(d)}")

    print()
    print("Verdicts:")
    # Architectural: observed within architectural tolerance of predicted
    obs = d.mean() * 100
    pred = 13.3
    pred_err = abs(obs - pred)
    print(f"  Architectural — observed {obs:+.1f}pp vs predicted {pred:+.1f}pp; "
          f"err = {pred_err:.1f}pp")
    if pred_err <= 5:
        print(f"    [SHARP HIT] additive linear model held to within ±5pp.")
    elif pred_err <= 10:
        print(f"    [ROUGH HIT] within ±10pp; additive direction correct, magnitude off.")
    elif (obs > 0 and pred > 0) or (obs < 0 and pred < 0):
        print(f"    [DIRECTION ONLY] same sign as predicted but magnitude wrong.")
    else:
        print(f"    [MISS] direction wrong; additive assumption needs revision.")

    # Stronger: candidate beats qsigdist?
    beats_qsig_strict = ci_dq[0] > 0
    beats_qsig_trend = (d - q).mean() > 0
    print(f"  Beats qsigdist:")
    if beats_qsig_strict:
        print(f"    [SIGNIFICANT] paired CI excludes 0; meta-routing found a "
              f"strictly better policy.")
    elif beats_qsig_trend:
        print(f"    [TRENDS YES] mean is higher but CI spans 0; suggestive, not conclusive.")
    else:
        print(f"    [NO] qsigdist remains the best policy known.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true",
                    help="Skip the run; analyze existing battery_results.json")
    args = ap.parse_args()
    if args.analyze:
        analyze()
    else:
        run_battery()


if __name__ == "__main__":
    main()
