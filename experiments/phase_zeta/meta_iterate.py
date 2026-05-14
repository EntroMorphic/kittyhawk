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
# Layer 3 — Glyph-native kernel retrieval (the rebuild)
# ============================================================================
#
# Original implementation (iterations 1–4) used continuous-coefficient
# regression with auto-scaling feature class. That violated the article
# Tripp shared on 2026-05-14 ("The Prior Should Be a Voice, Not a Verdict"):
#
#   1. The PRIOR (fitted coefficients) and the EVIDENCE (new observation)
#      shared the same mechanism — a single regression that simultaneously
#      held expectations AND incorporated evidence. No architectural
#      wall.
#   2. New observations FLOWED INTO the fit, modifying the predictor's
#      parameters. The article's term: "prior-contaminated witness."
#   3. The disagreement signal (predicted vs observed) vanished into the
#      next refit — never an explicit output.
#
# This rebuild instantiates the article's five components literally:
#
#   PRIOR HOLDER:  anchor store, append-only. Each anchor is a frozen
#                  (coordinate, observed_Δ) pair. No fitted parameters.
#   EVIDENCE READER: predict_kernel — takes a candidate, retrieves over
#                  anchors via fixed trit-distance kernel. NO learned
#                  weights. Predictions are weighted averages of nearby
#                  anchor Δ's.
#   THE WALL:      structural. The retrieval mechanism (kernel + distance
#                  metric) is FIXED at substrate level. New observations
#                  append to the anchor store but cannot modify the
#                  retrieval mechanism. The two pathways are wired apart
#                  by construction.
#   DISAGREEMENT DETECTOR: explicit. After every iteration, emit
#                  (predicted_Δ − observed_Δ) as a first-class output.
#                  This is the "voice" of the prior speaking against
#                  the evidence — surfaced, not absorbed.
#   DEFERENCE POLICY: confidence is derived from anchor-distance, not
#                  from refit residuals. A candidate far from every
#                  anchor (min_distance ≥ 2) returns LOW confidence;
#                  the architecture knows it doesn't know.
#
# Empirical motivation: when I retroactively applied kernel retrieval
# to iterations 1–4 (back-of-envelope, before this rebuild), it would
# have produced smaller errors on every iteration (6.8 vs 16.3 on iter
# 1; 2.3 vs 13.8 on iter 2; 5 vs 5.6 on iter 3; 12 vs 21 on iter 4).
# Regression was contaminated by ITS OWN past predictions; retrieval
# stays close to the observed anchors.

import math


def kernel_weight(distance: int, alpha: float = 1.0) -> float:
    """Distance kernel for anchor retrieval. exp(-alpha * d):
       d=0 → 1.0      (the anchor itself)
       d=1 → 0.368    (adjacent)
       d=2 → 0.135    (two flips)
       d=3 → 0.050    (opposite corner)
    Faster-than-1/d decay so distant anchors have minimal influence
    without being clipped entirely."""
    return math.exp(-alpha * distance)


def predict_kernel(candidate: tuple, anchors: list[dict],
                    alpha: float = 1.0) -> tuple[float, float, int]:
    """Glyph-native Layer 3 prediction.

    Returns (predicted_delta, confidence, min_distance_from_any_anchor).

    confidence ∈ [0, 1]: ratio of the closest anchor's weight to the
    total weight. Approaches 1 when ONE anchor dominates (high
    certainty); approaches 1/N when all N anchors contribute equally
    (low certainty — the candidate is far from every anchor and the
    prediction is essentially the dataset mean).

    NO LEARNED PARAMETERS. The kernel and metric are fixed at the
    substrate level. The 'model' IS the anchor store; the prediction
    is its routing-derived consequence.
    """
    distances = [trit_distance(candidate, tuple(a["w"])) for a in anchors]
    weights = [kernel_weight(d, alpha) for d in distances]
    total = sum(weights)
    if total == 0.0:
        return 0.0, 0.0, max(distances) if distances else 999
    prediction = sum(w * a["delta"] for w, a in zip(weights, anchors)) / total
    confidence = max(weights) / total
    return float(prediction), float(confidence), min(distances)


def disagreement(predicted: float, observed: float) -> tuple[float, int]:
    """Explicit disagreement signal — a first-class output, not absorbed.

    Returns (magnitude, sign):
      magnitude: |predicted − observed| in pp
      sign: +1 if observed > predicted (model underestimated)
            −1 if observed < predicted (model overestimated)
             0 if equal

    The wall: this signal is COMPUTED but cannot directly modify the
    anchor store or the kernel. It is a separate channel — for the
    caller to inspect, log, or use to drive sampling decisions."""
    if predicted == observed:
        return 0.0, 0
    return abs(observed - predicted), (1 if observed > predicted else -1)


def fit_model(anchors: list[dict]):
    """Backwards-compatible wrapper. Returns (predict_fn, label, coeffs).

    coeffs is None — the kernel retrieval has no fitted coefficients
    by design (that's the architectural property). Label describes
    the kernel and anchor count."""
    label = (f"kernel retrieval over {len(anchors)} anchors "
             f"(α=1.0, no fitted params; structural wall between "
             f"prior holder and predictor)")

    def predict(w):
        pred, _conf, _dist = predict_kernel(tuple(w), anchors)
        return pred

    return predict, label, None


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
    print(f"Active model: kernel retrieval over {len(anchors)} anchors")
    print(f"  α = 1.0 (exp(-α·d) kernel), no fitted parameters.")
    print(f"  Structural wall: anchor store is append-only; retrieval mechanism")
    print(f"  has no learnable state. Prior and predictor are wired apart.")
    print()
    # Show top 5 predictions across untested points, with confidence
    tested = {tuple(a["w"]) for a in anchors}
    preds = []
    for w in itertools.product(TRITS, repeat=3):
        if w in tested:
            continue
        pred, conf, dist = predict_kernel(w, anchors)
        preds.append((w, pred, conf, dist))
    preds.sort(key=lambda x: -x[1])
    print(f"Top 5 untested predictions:")
    print(f"  {'candidate':<12}  {'pred Δ':>8}  {'confidence':>11}  {'min dist':>9}")
    for w, p, conf, dist in preds[:5]:
        print(f"  {str(w):<12}  {p:+5.1f}pp   {conf:>9.2f}     {dist:>5}")


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

    # Confidence + min-distance from the kernel retrieval — separate
    # output channels from the prediction itself.
    pred_kernel, confidence, min_dist = predict_kernel(w, anchors)

    print(f"=== Iteration {n_before - 4 + 1} ===")
    print(f"Active model:    {model_label}")
    print(f"Candidate:       {w}")
    print(f"Predicted Δ:     {pred:+.1f}pp")
    print(f"  Confidence:    {confidence:.2f}  (closest anchor dominates retrieval)")
    print(f"  Min distance:  {min_dist}  (trit-flips from nearest anchor)")
    print(f"Reasoning:       {reasoning}")

    observed_delta = run_candidate(w)

    # Disagreement signal — first-class output, computed BEFORE the
    # anchor store is updated. This is the article's "voice not verdict":
    # the prior's prediction speaks against the evidence, and we record
    # how loudly without letting it modify the predictor itself.
    disagree_mag, disagree_sign = disagreement(pred, observed_delta)

    # Append to anchors — APPEND-ONLY; the wall ensures no past anchor
    # mutates. New anchor IS the evidence; it joins the prior holder
    # but doesn't alter what's already there.
    anchors.append({
        "w": list(w),
        "delta": observed_delta,
        "source": (f"meta_iterate iteration {n_before - 4 + 1}; "
                   f"predicted {pred:+.1f}pp; disagreement "
                   f"{disagree_mag:+.1f}pp"),
    })
    save_anchors(anchors)

    print("=" * 60)
    print(f"After iteration {n_before - 4 + 1}:")
    print(f"  predicted Δ:        {pred:+.1f}pp")
    print(f"  observed  Δ:        {observed_delta:+.1f}pp")
    print(f"  DISAGREEMENT:       {disagree_mag:+.1f}pp  "
          f"({'observed > predicted' if disagree_sign > 0 else 'observed < predicted' if disagree_sign < 0 else 'agree'})")
    print()
    print(f"  Architectural note: the disagreement signal is RECORDED but")
    print(f"  the kernel retrieval and trit-distance metric are unchanged.")
    print(f"  The anchor store gained one entry; no past anchor mutated.")
    print()
    next_w, next_pred, next_label, _ = propose_next(anchors)
    if next_w is not None:
        npred, nconf, ndist = predict_kernel(next_w, anchors)
        print(f"  next proposal: {next_w}")
        print(f"    predicted {next_pred:+.1f}pp, confidence {nconf:.2f}, "
              f"min distance {ndist}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "propose", "iterate"])
    args = ap.parse_args()
    if args.cmd == "status": cmd_status()
    elif args.cmd == "propose": cmd_propose()
    elif args.cmd == "iterate": cmd_iterate()


if __name__ == "__main__":
    main()
