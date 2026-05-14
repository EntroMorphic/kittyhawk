"""Coupled three-layer meta-routing — the iteration version.

The previous prototype (meta_routing.py) was feed-forward only: fit
once, predict once, test once, stop. That's not a three-layer
architecture — it's a static predictor.

A real "cogs in a watch" system has the layers continuously informing
each other. Each iteration:

  1. Layer 3 fits a model on ALL accumulated anchors.
  2. Model class auto-scales with anchor count:
       4 anchors → additive linear (3 main + intercept)
       5 anchors → +1 interaction term (the one implicated by the
                   most recent miss, if any)
       7+ anchors → all 3 pairwise interactions
  3. Layer 3 proposes the next candidate — the highest-predicted
     UNTESTED policy within trit distance ≤ 1 from any anchor
     (limit extrapolation; let the model earn its predictions).
  4. Layer 2 runs the candidate on N=100 prompts via the harness.
  5. The observation feeds back as a new anchor.
  6. Layer 3 refits with the augmented anchor set.

The architecture passes if, after K iterations, layer 3:
  - converges on a policy beating qsigdist with non-overlapping CI, OR
  - converges (top prediction stable, no new info from runs) on a
    policy ≤ qsigdist, indicating the response surface has been
    adequately characterized.

The architecture fails if:
  - Iterations don't converge AND predictions remain unreliable
    (each new anchor radically reshapes the model).

This script runs ONE iteration at a time so the user can stop / pivot
between runs. State is persisted in anchors.json.

Usage:
  python meta_iterate.py status       # show current state
  python meta_iterate.py propose      # show the next candidate
  python meta_iterate.py iterate      # propose → run → observe (~20min)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

THIS = os.path.dirname(__file__)
ANCHORS_JSON = os.path.join(THIS, "results/meta_iterate/anchors.json")
HARNESS = "build/gesh/bitnet_harness"
DATA    = "data/bitnet_b158_2b4t.bin"
WINDOW  = 16
GEN_N   = 24

# Bootstrap seed for paired-CI computation
RNG_SEED = 20260513

TRITS = (-1, 0, +1)


# ============================================================================
# Anchor store
# ============================================================================

# Bootstrap with the 5 known observations we already have.
INITIAL_ANCHORS = [
    {"w": [ 0,  0,  0], "delta": 0.0,    "source": "random (definitional baseline)"},
    {"w": [-1,  0,  0], "delta": -5.5,   "source": "fifo (N=100 closeout)"},
    {"w": [ 0,  1,  0], "delta": -7.0,   "source": "sigdist (N=100 closeout)"},
    {"w": [ 0,  0,  1], "delta": 6.4,    "source": "qsigdist (N=100 closeout)"},
    {"w": [ 0, -1,  1], "delta": -3.0,   "source": "meta(0,-1,+1) (2026-05-14 iteration 1)"},
]


def load_anchors() -> list[dict]:
    """Load anchors from JSON, initializing if file doesn't exist."""
    if os.path.exists(ANCHORS_JSON):
        with open(ANCHORS_JSON) as f:
            return json.load(f)["anchors"]
    os.makedirs(os.path.dirname(ANCHORS_JSON), exist_ok=True)
    return list(INITIAL_ANCHORS)


def save_anchors(anchors: list[dict]) -> None:
    with open(ANCHORS_JSON, "w") as f:
        json.dump({"anchors": anchors}, f, indent=2)


# ============================================================================
# Layer 3: model with anchor-count-dependent capacity
# ============================================================================

def features_linear(w):
    """4 features: intercept + 3 main effects."""
    return [1.0, w[0], w[1], w[2]]


def features_one_interaction(w):
    """5 features: above + w_kk * w_qk (the interaction implicated by
    the iteration-1 miss)."""
    return [1.0, w[0], w[1], w[2], w[1] * w[2]]


def features_all_pairwise(w):
    """7 features: intercept + 3 main + 3 pairwise."""
    return [1.0, w[0], w[1], w[2],
            w[0] * w[1], w[0] * w[2], w[1] * w[2]]


def select_model(n_anchors: int):
    """Auto-scale model class with anchor budget. Each model needs at
    least as many anchors as features for a non-degenerate fit."""
    if n_anchors >= 7:
        return features_all_pairwise, "linear + all 3 pairwise (7 params)"
    if n_anchors >= 5:
        return features_one_interaction, "linear + w_kk·w_qk interaction (5 params)"
    return features_linear, "additive linear (4 params)"


def fit_model(anchors: list[dict]):
    """Returns (predict_fn, model_label)."""
    feat, label = select_model(len(anchors))
    X = np.array([feat(tuple(a["w"])) for a in anchors])
    y = np.array([a["delta"] for a in anchors])
    if X.shape[0] == X.shape[1]:
        # Exactly determined
        coeffs = np.linalg.solve(X, y)
    else:
        # Over- or underdetermined: min-norm lstsq
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    def predict(w):
        return float(np.dot(feat(tuple(w)), coeffs))

    return predict, label, coeffs


# ============================================================================
# Candidate proposer
# ============================================================================

def trit_distance(a, b):
    return sum(abs(a[i] - b[i]) for i in range(3))


def propose_next(anchors: list[dict], max_extrapolation: int = 1) -> tuple:
    """Return (candidate, predicted_delta, model_label, reasoning).

    Strategy: pick the highest-predicted UNTESTED candidate within trit
    distance ≤ max_extrapolation from any anchor. This limits Layer 3
    to predictions it's earned, rather than running wild extrapolations.
    """
    predict, model_label, _coeffs = fit_model(anchors)
    tested = {tuple(a["w"]) for a in anchors}
    candidates = []
    for w in itertools.product(TRITS, repeat=3):
        if w in tested:
            continue
        min_dist = min(trit_distance(w, tuple(a["w"])) for a in anchors)
        if min_dist > max_extrapolation:
            continue
        candidates.append((w, predict(w), min_dist))
    candidates.sort(key=lambda c: -c[1])
    if not candidates:
        return None, None, model_label, "no candidates within max_extrapolation"
    w, pred, dist = candidates[0]
    reasoning = (f"highest-predicted untested within trit-distance "
                 f"{max_extrapolation}; distance from nearest anchor = {dist}")
    return w, pred, model_label, reasoning


# ============================================================================
# Layer 2 runner — call into the harness with the proposed candidate
# ============================================================================

def load_n100_prompts() -> dict[str, str]:
    """Reconstruct the 100-prompt dict."""
    sys.path.insert(0, THIS)
    try:
        from n50_battery import PROMPTS as OLD_PROMPTS
        from n100_battery_incremental import NEW_PROMPTS
    finally:
        sys.path.pop(0)
    out = {}
    out.update(OLD_PROMPTS)
    out.update(NEW_PROMPTS)
    assert len(out) == 100
    return out


def run_candidate(w: tuple[int, int, int]) -> float:
    """Run the harness for all 100 prompts at the given weights and
    return Δ vs random in percentage points."""
    prompts = load_n100_prompts()
    iter_name = f"iter_{w[0]}_{w[1]}_{w[2]}"
    outdir = os.path.join(THIS, f"results/meta_iterate/{iter_name}")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n[runner] candidate {w}; saving to {outdir}\n")

    tokens_by_label = {}
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

    # Compute Δ vs random using existing N=100 baselines
    delta = compute_delta_vs_random(tokens_by_label)
    print(f"\n[runner] candidate {w}: Δ vs random = {delta:+.2f}pp\n")
    return delta


def compute_delta_vs_random(tokens_by_label: dict) -> float:
    """Compare candidate tokens vs no_evict, vs random's tokens (baseline)."""
    with open(os.path.join(THIS, "results/n50_battery/battery_results.json")) as f:
        old = json.load(f)["trials"]
    with open(os.path.join(THIS, "results/n100_incremental/battery_results.json")) as f:
        new = json.load(f)["trials"]
    base_by_lm = {(t["label"], t["mode"]): t["tokens"] for t in old + new}

    def match_rate(a, b):
        n = min(len(a), len(b))
        return sum(1 for i in range(n) if a[i] == b[i]) / n if n else 0.0

    diffs = []
    for label, cand_tokens in tokens_by_label.items():
        no_evict = base_by_lm.get((label, "no_evict"))
        randc = base_by_lm.get((label, "random"))
        if not (no_evict and cand_tokens and randc):
            continue
        diffs.append(match_rate(no_evict, cand_tokens) - match_rate(no_evict, randc))
    return 100.0 * float(np.mean(diffs))


# ============================================================================
# CLI
# ============================================================================

def cmd_status():
    anchors = load_anchors()
    print(f"Accumulated anchors: {len(anchors)}\n")
    for a in anchors:
        w = tuple(a["w"])
        print(f"  {w}  Δ = {a['delta']:+5.1f}pp   {a['source']}")
    print()
    predict, model_label, coeffs = fit_model(anchors)
    print(f"Active model: {model_label}")
    print(f"Coefficients: {coeffs.round(3)}")
    # Show top 5 predictions across untested points
    tested = {tuple(a["w"]) for a in anchors}
    preds = []
    for w in itertools.product(TRITS, repeat=3):
        if w in tested:
            continue
        preds.append((w, predict(w),
                      min(trit_distance(w, tuple(a["w"])) for a in anchors)))
    preds.sort(key=lambda x: -x[1])
    print(f"\nTop 5 untested predictions:")
    for w, p, dist in preds[:5]:
        print(f"  {w}  predicted Δ = {p:+5.1f}pp  (trit-distance from anchor = {dist})")


def cmd_propose():
    anchors = load_anchors()
    w, pred, model_label, reasoning = propose_next(anchors)
    print(f"Active model:  {model_label}")
    print(f"Next candidate: {w}")
    print(f"Predicted Δ:    {pred:+.1f}pp")
    print(f"Reasoning:      {reasoning}")


def cmd_iterate():
    """One full iteration: propose, run, observe, refit."""
    anchors = load_anchors()
    n_before = len(anchors)
    w, pred, model_label, reasoning = propose_next(anchors)
    if w is None:
        print("No candidate available (search space exhausted within "
              "max_extrapolation). Try a wider radius.")
        return

    print(f"=== Iteration {n_before - 4 + 1} ===")
    print(f"Active model:   {model_label}")
    print(f"Candidate:      {w}")
    print(f"Predicted Δ:    {pred:+.1f}pp")
    print(f"Reasoning:      {reasoning}")

    observed_delta = run_candidate(w)

    # Append to anchors
    anchors.append({
        "w": list(w),
        "delta": observed_delta,
        "source": f"meta_iterate iteration {n_before - 4 + 1}; predicted {pred:+.1f}pp",
    })
    save_anchors(anchors)

    # Refit and show what the new prediction surface looks like
    print("=" * 50)
    print(f"After iteration {n_before - 4 + 1}:")
    print(f"  predicted Δ = {pred:+.1f}pp  observed Δ = {observed_delta:+.1f}pp")
    print(f"  error = {abs(pred - observed_delta):.1f}pp")
    new_predict, new_label, new_coeffs = fit_model(anchors)
    print(f"  refit model: {new_label}")
    print(f"  new coefficients: {new_coeffs.round(3)}")
    next_w, next_pred, _, _ = propose_next(anchors)
    if next_w is not None:
        print(f"  next proposal: {next_w} (predicted {next_pred:+.1f}pp)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "propose", "iterate"])
    args = ap.parse_args()
    if args.cmd == "status": cmd_status()
    elif args.cmd == "propose": cmd_propose()
    elif args.cmd == "iterate": cmd_iterate()


if __name__ == "__main__":
    main()
