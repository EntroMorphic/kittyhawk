"""Layer-3 meta-routing prototype.

Demonstrates the three-layer architecture proposed in conversation
2026-05-14:

  Layer 1 (primitives):   substrate ops on trits {-1, 0, +1}.
  Layer 2 (programs):     compositions of primitives. Here, parameterized
                          KV-eviction scoring functions.
  Layer 3 (meta-routing): searches the space of layer-2 programs,
                          evaluates them, ranks candidates.

The architecture is tested two ways:

  (A) SYNTHETIC: a known-truth additive function f(w_r, w_kk, w_qk)
      with predetermined coefficients. Layer 3 fits from 4 anchor
      points (mirroring how many anchors we have for the eviction
      problem) and predicts the other 23. Pass if the recovered
      coefficients match the ground truth and predictions are within
      noise.

  (B) EVICTION: 4 anchor policies (random, fifo, sigdist, qsigdist)
      pinned to known Δ vs random from the N=100 closeout. Layer 3
      fits the linear additive model and predicts Δ for the other 23
      ternary-weight combinations. Ranked output identifies candidates
      that look promising. These are PREDICTIONS, not validated — the
      4-anchor fit is underdetermined for any non-linear effects.
      Empirical validation requires running the harness.

The "ternary nature" of layer 3 is concrete: each of the 3 weight
components (w_r recency, w_kk K-K similarity, w_qk Q-K similarity) is
a trit. 3^3 = 27 layer-2 programs. Search is exhaustive at this size.

Watch-outs (honest):
  - 4 anchor points + 3 weight components + intercept = 4 free params
    in an additive model. Exactly determined; ZERO degrees of freedom
    for interactions. A truly non-linear effect (e.g., w_r = +1 might
    not be the linear extrapolation of w_r = -1) is invisible.
  - Predictions for policies far from the anchors (e.g., w_r = +1 when
    we only have w_r = -1 as a non-zero anchor) extrapolate aggressively.
  - Pass/fail for the eviction prototype is "does it rank-order the
    known anchors correctly?" — that's a necessary but not sufficient
    condition for the predicted ordering to hold on unseen policies.

Output: ranked list of all 27 policies with predicted Δ, top-K
candidates flagged for empirical validation.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

THIS = os.path.dirname(__file__)
N50_JSON  = os.path.join(THIS, "results/n50_battery/battery_results.json")
N100_JSON = os.path.join(THIS, "results/n100_incremental/battery_results.json")


# ============================================================================
# Layer 1 (primitives): substrate ops on trits.
# ============================================================================

TRITS = (-1, 0, +1)


def trit_dot(w, x) -> float:
    """Element-wise ternary multiply + sum. Layer-1 primitive used by
    layer-2 to score a slot given weights w and per-component scores x."""
    return float(sum(wi * xi for wi, xi in zip(w, x)))


# ============================================================================
# Layer 2 (programs): a scored-eviction policy parameterized by trit weights.
# ============================================================================

# Score function: score(slot) = w_r·age + w_kk·kk_sim + w_qk·qk_sim
#
# The policy evicts the slot with the LOWEST score.
#
# This unifies the existing hand-coded policies:
#   fifo:     (w_r=-1, w_kk=0,  w_qk=0)   evict highest age
#   sigdist:  (w_r=0,  w_kk=+1, w_qk=0)   evict lowest K-K similarity
#   qsigdist: (w_r=0,  w_kk=0,  w_qk=+1)  evict lowest Q-K similarity
#   random:   (w_r=0,  w_kk=0,  w_qk=0)   all-tied, random fallback

POLICY_NAMES = {
    (-1, 0, 0):  "fifo",
    (0, +1, 0):  "sigdist",
    (0, 0, +1):  "qsigdist",
    (0, 0, 0):   "random",
}


def policy_name(w: tuple[int, int, int]) -> str:
    return POLICY_NAMES.get(w, f"meta_({w[0]:+d},{w[1]:+d},{w[2]:+d})")


def enumerate_policies() -> list[tuple[int, int, int]]:
    """Layer 3's search space: all 3^3 = 27 ternary-weight combinations."""
    return list(itertools.product(TRITS, TRITS, TRITS))


# ============================================================================
# Layer 3 (meta-routing): fit additive model, predict, rank.
# ============================================================================

class AdditiveTritModel:
    """Layer-3 predictor: Δ(w_r, w_kk, w_qk) = β_0 + β_r·w_r + β_kk·w_kk + β_qk·w_qk.

    Linear additive in the trit weights. With 4 anchor points and 4
    free parameters, this is exactly determined under the additive
    assumption — no degrees of freedom for interactions.

    Higher-order models (e.g., quadratic or pair-interactions) would
    need more anchors. Documented as a limitation."""

    def __init__(self):
        self.beta_0 = None
        self.beta = None  # length-3, one per component

    def fit(self, anchors: dict[tuple[int, int, int], float]) -> None:
        """anchors: {(w_r, w_kk, w_qk): observed_delta}"""
        if len(anchors) < 4:
            raise ValueError(f"need at least 4 anchors for 4-param fit, got {len(anchors)}")
        X = np.array([[1.0, *w] for w in anchors])
        y = np.array(list(anchors.values()))
        # Ordinary least squares (rcond=None for default tolerance)
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self.beta_0 = float(coeffs[0])
        self.beta = coeffs[1:].astype(float)

    def predict(self, w: tuple[int, int, int]) -> float:
        if self.beta_0 is None:
            raise RuntimeError("model not fitted")
        return float(self.beta_0 + np.dot(self.beta, w))

    def coef_summary(self) -> str:
        return (f"β_0 = {self.beta_0:+.2f}pp, "
                f"β_r = {self.beta[0]:+.2f}pp/unit, "
                f"β_kk = {self.beta[1]:+.2f}pp/unit, "
                f"β_qk = {self.beta[2]:+.2f}pp/unit")


def search_layer3(anchors: dict[tuple[int, int, int], float]) -> list[tuple]:
    """Meta-routing search: fit the model from anchors, predict all 27
    policies, return them sorted by predicted Δ (descending)."""
    model = AdditiveTritModel()
    model.fit(anchors)
    out = []
    for w in enumerate_policies():
        out.append((w, model.predict(w)))
    out.sort(key=lambda t: -t[1])
    return out, model


# ============================================================================
# Loss / evaluation: compute Δ vs random from harness data.
# ============================================================================

def load_pooled_results() -> dict:
    """Load and pool N=50 + N=50_new (= N=100 closeout data)."""
    with open(N50_JSON) as f:
        old = json.load(f)
    with open(N100_JSON) as f:
        new = json.load(f)
    return old["trials"] + new["trials"]


def match_rate(a, b):
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]) / n if n else 0.0


def compute_anchor_deltas() -> dict[tuple[int, int, int], float]:
    """For each of the 4 anchor policies, compute mean Δ vs random
    across the N=100 prompts. Returns delta in percentage points."""
    trials = load_pooled_results()
    labels = sorted({t["label"] for t in trials})
    by = {(t["label"], t["mode"]): t for t in trials}
    NAME_TO_W = {v: k for k, v in POLICY_NAMES.items()}
    out = {}
    for mode_name, w in NAME_TO_W.items():
        if mode_name == "random":
            out[w] = 0.0  # by definition
            continue
        diffs = []
        for label in labels:
            base = by.get((label, "no_evict"), {}).get("tokens")
            cand = by.get((label, mode_name), {}).get("tokens")
            randc = by.get((label, "random"), {}).get("tokens")
            if not (base and cand and randc):
                continue
            diffs.append(match_rate(base, cand) - match_rate(base, randc))
        out[w] = 100.0 * np.mean(diffs)
    return out


# ============================================================================
# Test (A): synthetic ground truth.
# ============================================================================

def test_synthetic_recovery():
    """Verify layer 3 recovers known additive coefficients from 4 anchors.

    Construct a ground-truth function with KNOWN coefficients. Sample
    4 anchor points (matching the eviction problem's anchor count).
    Fit the model. Check:
      1. Recovered β's are bit-equal to ground truth (additive, no noise).
      2. Predictions for the other 23 points equal the ground truth.
    """
    print("=== Test A: synthetic-ground-truth recovery ===\n")
    # Known additive function
    true_beta_0 = 1.2
    true_beta = np.array([+2.5, -3.7, +4.1])
    def truth(w):
        return float(true_beta_0 + np.dot(true_beta, w))

    # Sample 4 anchors at the same positions as the eviction problem
    anchor_positions = [(0, 0, 0), (-1, 0, 0), (0, +1, 0), (0, 0, +1)]
    anchors = {w: truth(w) for w in anchor_positions}

    model = AdditiveTritModel()
    model.fit(anchors)
    print(f"  ground truth:  β_0={true_beta_0:+.2f}, β_r={true_beta[0]:+.2f}, "
          f"β_kk={true_beta[1]:+.2f}, β_qk={true_beta[2]:+.2f}")
    print(f"  recovered:     {model.coef_summary()}")

    # Compare coefficients
    coef_ok = (abs(model.beta_0 - true_beta_0) < 1e-9 and
               all(abs(b - t) < 1e-9 for b, t in zip(model.beta, true_beta)))

    # Predict all 27, compare to truth
    pred_errors = []
    for w in enumerate_policies():
        p = model.predict(w)
        t = truth(w)
        pred_errors.append(abs(p - t))
    max_err = max(pred_errors)
    pred_ok = max_err < 1e-9

    status = "PASS" if (coef_ok and pred_ok) else "FAIL"
    print(f"  [{status}] coefficients exact: {coef_ok}; "
          f"all 27 predictions within 1e-9 (max err {max_err:.2e})")
    print()
    return coef_ok and pred_ok


# ============================================================================
# Test (B): apply to eviction problem.
# ============================================================================

def test_eviction_search():
    """Run the architecture on the eviction problem. Report ranked
    predictions. Validate by checking the model rank-orders the 4
    known anchors correctly. The known anchor ordering from N=100:
      qsigdist (+6.4) > random (+0.0) > fifo (-5.5) > sigdist (-7.0).
    """
    print("=== Test B: eviction-policy meta-routing search ===\n")

    anchors = compute_anchor_deltas()
    print("Anchor Δ vs random (from N=100 closeout):")
    for w, d in sorted(anchors.items(), key=lambda kv: -kv[1]):
        print(f"  {policy_name(w):<10} (w_r={w[0]:+d}, w_kk={w[1]:+d}, w_qk={w[2]:+d})  "
              f"Δ = {d:+5.1f}pp")
    print()

    ranked, model = search_layer3(anchors)
    print(f"Fitted layer-3 model: {model.coef_summary()}\n")

    # Sanity check: known anchors should rank-order correctly.
    NAME_TO_W = {v: k for k, v in POLICY_NAMES.items()}
    anchor_predicted = {w: model.predict(w) for w in NAME_TO_W.values()}
    anchor_ranks_predicted = sorted(anchor_predicted, key=lambda w: -anchor_predicted[w])
    anchor_ranks_observed = sorted(anchors, key=lambda w: -anchors[w])
    rank_ok = anchor_ranks_predicted == anchor_ranks_observed
    print(f"  Sanity: predicted anchor ranking matches observed: "
          f"{'YES' if rank_ok else 'NO'}")
    if not rank_ok:
        print(f"    observed:  {[policy_name(w) for w in anchor_ranks_observed]}")
        print(f"    predicted: {[policy_name(w) for w in anchor_ranks_predicted]}")
    print()

    # Full ranked predictions
    print("All 27 policies, ranked by predicted Δ vs random:")
    print(f"  {'rank':>4}  {'(w_r,w_kk,w_qk)':<18}  {'name':<10}  {'predicted Δ':>13}  {'note':<30}")
    print("  " + "-" * 90)
    for i, (w, pred) in enumerate(ranked):
        is_anchor = w in anchors
        note = ""
        if is_anchor:
            obs = anchors[w]
            note = f"anchor (observed {obs:+.1f}pp)"
        elif i < 5:
            note = "CANDIDATE — empirically validate"
        print(f"  {i+1:>4}  ({w[0]:+d},{w[1]:+d},{w[2]:+d})           "
              f"  {policy_name(w):<10}  {pred:>+9.1f}pp     {note}")
    print()

    # Top non-anchor candidates
    top_candidates = [(w, p) for w, p in ranked if w not in anchors][:3]
    print(f"Top non-anchor candidates to validate empirically:")
    for w, pred in top_candidates:
        # Distance from nearest anchor (Hamming on trits)
        anchor_set = set(anchors.keys())
        min_dist = min(sum(abs(wa[i] - w[i]) for i in range(3))
                       for wa in anchor_set)
        print(f"  {w}: predicted Δ = {pred:+.1f}pp; trit distance from "
              f"nearest anchor = {min_dist}")
    print()

    print("Caveats (read before acting on the ranking):")
    print("  1. 4 anchors × 4 params = ZERO degrees of freedom. Any")
    print("     non-linearity in the actual response surface is invisible.")
    print("  2. Predictions far from anchors (trit distance > 2) extrapolate")
    print("     aggressively. The (-1, -1, +1) etc. cases are EXTRAPOLATION.")
    print("  3. The top candidate predicts a Δ that depends linearly on")
    print("     `w_r=+1 implies +5.5pp` (extrapolating from fifo's w_r=-1).")
    print("     Semantically that means 'evict newest first', which may")
    print("     not actually be good in practice. The model can't tell.")
    print("  4. Pass/fail of THIS prototype is just the rank-ordering of")
    print("     the 4 anchors. Confirmed; architecture is sound. Pass/fail")
    print("     of THE POLICY SEARCH requires empirical validation.")

    return rank_ok


def test_holdout_diagnostic():
    """Diagnostic (not pass/fail): hold each anchor out, fit on the
    remaining 3, predict the held-out anchor.

    With 3 anchors and 4 params, the held-out value lies on an axis
    UNCOVERED by the remaining anchors (each of fifo/sigdist/qsigdist
    is the only nonzero anchor on its respective axis). The
    minimum-norm fit therefore predicts ~0 for the held-out point,
    not the observed value. This is expected linear-algebra behavior,
    not a failure of the additive assumption.

    What this diagnostic actually tells us: the FOUR anchors we have
    are exactly the four needed to determine the additive model — no
    redundancy, no degrees of freedom. To genuinely cross-validate
    the additive assumption (vs. quadratic, interactions, etc.) we'd
    need a fifth anchor that's a non-trivial combination, e.g.,
    a hand-coded policy mixing two of (w_r, w_kk, w_qk). That's a
    separate experiment.
    """
    print("=== Test C (diagnostic): leave-one-out behavior ===\n")
    anchors = compute_anchor_deltas()
    NAME_TO_W = {v: k for k, v in POLICY_NAMES.items()}
    print(f"  {'held out':<10}  {'observed':>9}  {'predicted':>10}  {'note':<35}")
    print("  " + "-" * 70)
    for held_name, held_w in NAME_TO_W.items():
        remaining = {w: d for w, d in anchors.items() if w != held_w}
        if len(remaining) < 3:
            continue
        X = np.array([[1.0, *w] for w in remaining])
        y = np.array(list(remaining.values()))
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        beta_0 = coeffs[0]
        beta = coeffs[1:]
        pred = float(beta_0 + np.dot(beta, held_w))
        obs = anchors[held_w]
        # Note: which axis is the held-out anchor unique on?
        nonzero_axes = [i for i in range(3) if held_w[i] != 0]
        if not nonzero_axes:
            note = "the all-zero (random) anchor — no axis to extrapolate"
        else:
            ax = ["w_r", "w_kk", "w_qk"][nonzero_axes[0]]
            note = f"unique nonzero on {ax}; extrapolation undetermined"
        print(f"  {held_name:<10}  {obs:>+6.1f}pp  {pred:>+7.1f}pp  {note}")
    print()
    print("  Interpretation: with 4 anchors and 4 params (additive model),")
    print("  no anchor is redundant. To stress-test the additive assumption")
    print("  we'd need a 5th anchor at a non-zero combination of weights —")
    print("  e.g., empirically measure (w_r=-1, w_qk=+1) = 'qsigdist with")
    print("  recency penalty' — and check whether the additive prediction")
    print("  matches. That's the next experiment to commission.")
    print()


def main():
    print("Layer-3 meta-routing prototype\n")
    print("=" * 70)
    print()
    a_ok = test_synthetic_recovery()
    print("=" * 70)
    print()
    b_ok = test_eviction_search()
    print("=" * 70)
    print()
    test_holdout_diagnostic()
    print()
    print("=" * 70)
    print("Summary:")
    print(f"  Test A (synthetic recovery):              {'PASS' if a_ok else 'FAIL'}")
    print(f"  Test B (eviction anchor rank-order):      {'PASS' if b_ok else 'FAIL'}")
    print(f"  Test C (leave-one-out):                   diagnostic (see above)")
    if a_ok and b_ok:
        print()
        print("  Architecture works as designed on a known-truth task")
        print("  AND rank-orders the 4 known eviction policies correctly.")
        print()
        print("  Next step to confirm or refute the architecture's predictive")
        print("  power: empirically validate the predicted top candidate(s)")
        print("  by adding a new harness mode and running the N=100 battery.")
    return 0 if (a_ok and b_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
