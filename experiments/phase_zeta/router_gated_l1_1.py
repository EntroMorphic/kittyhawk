"""L1.1: confidence-gated category router + L1.2 held-out CV.

Same single feature (category prefix) as L1, smarter strategy: route
to the category's mode-winner ONLY when (a) within-category winner-
entropy ≤ τ AND (b) the mode-policy's category-mean Δ exceeds
qsigdist's category-mean Δ by ≥ δ. Otherwise default to qsigdist.

Two stages:

  Stage A (L1.1) — IN-SAMPLE sweep over (τ, δ). For each grid cell,
  compute the router's Δ and (router − qsigdist) on all 100 prompts.
  Find the best (τ, δ) by router-minus-qsigdist mean. WARNING: this
  selects τ, δ on the same data we evaluate on; the in-sample winner
  inflates the apparent gain.

  Stage B (L1.2) — HELD-OUT CV. Repeated k-fold (k=5) or
  random 50/50 splits. For each split: (i) compute category modes,
  category-mean Δs, and H(winner|cat) on TRAIN; (ii) sweep (τ, δ)
  on TRAIN; pick best by TRAIN router-minus-qsigdist; (iii) apply
  that gate to TEST. The held-out Δ is the realistic estimate.

Output:
  - Best in-sample (τ, δ) and its router Δ + headroom captured.
  - Held-out router Δ + CI on (router − qsigdist) over splits.
  - For interpretability: which categories got routed (passed the
    gate) in the best-train split.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

THIS = os.path.dirname(__file__)
ROUTING_JSON = os.path.join(THIS, "results/meta_iterate/per_prompt_routing.json")


def category_of(label: str) -> str:
    return label.split("_")[0] if "_" in label else label


def build_router_table(train_labels, train_idx, rates, policy_names,
                       random_idx, qsig_idx):
    """For each category present in train_labels, compute (mode_winner,
    cat_H, cat_mode_delta_pp, cat_qsig_delta_pp). Returns dict category
    -> (mode_winner_name, H_bits, mode_mean_d_pp, qsig_mean_d_pp)."""
    cat_to_prompts = defaultdict(list)
    for li in train_idx:
        cat_to_prompts[category_of(train_labels[li])].append(li)

    table = {}
    for cat, idxs in cat_to_prompts.items():
        # winners[j] is the winning policy for label j; we need to
        # recompute from rates (excluding random) — that's already
        # captured in 'winners' from the JSON but we redo here so the
        # function is self-contained.
        winners_in_cat = []
        for j in idxs:
            col = rates[:, j].copy()
            col[random_idx] = -np.inf  # exclude random from competition
            w_idx = int(np.argmax(col))
            winners_in_cat.append(policy_names[w_idx])
        wc = Counter(winners_in_cat)
        n = sum(wc.values())
        probs = np.array(list(wc.values())) / n
        H = -np.sum(probs * np.log2(probs + 1e-12))
        mode_w = wc.most_common(1)[0][0]
        mode_idx = policy_names.index(mode_w)
        rand_rates_train = rates[random_idx, idxs]
        mode_delta = float(np.mean(rates[mode_idx, idxs] - rand_rates_train)) * 100
        qsig_delta = float(np.mean(rates[qsig_idx, idxs] - rand_rates_train)) * 100
        table[cat] = (mode_w, float(H), mode_delta, qsig_delta)
    return table


def evaluate_router(eval_labels, eval_idx, rates, policy_names,
                    random_idx, qsig_idx, router_table, tau, delta_thr):
    """Apply gate (H ≤ τ AND mode_d - qsig_d ≥ δ) to each prompt in
    eval set. Route to mode if gate passes, else qsigdist. Return
    per-prompt Δ-vs-random, per-prompt Δ-vs-qsigdist."""
    deltas = []
    diffs  = []
    for li in eval_idx:
        label = eval_labels[li]
        cat = category_of(label)
        info = router_table.get(cat)
        if info is None:
            routed_idx = qsig_idx
        else:
            mode_w, H, mode_d, qsig_d = info
            if H <= tau and (mode_d - qsig_d) >= delta_thr:
                routed_idx = policy_names.index(mode_w)
            else:
                routed_idx = qsig_idx
        r = rates[routed_idx, li]
        rand = rates[random_idx, li]
        qsig = rates[qsig_idx, li]
        deltas.append(r - rand)
        diffs.append(r - qsig)
    return np.array(deltas), np.array(diffs)


def main():
    with open(ROUTING_JSON) as f:
        d = json.load(f)
    policy_names = d["policy_names"]
    labels = d["labels"]
    rates = np.array(d["rates"])
    qsig_delta_full = d["qsigdist_delta_pp"]
    oracle_delta_full = d["oracle_delta_pp"]
    random_idx = policy_names.index("random")
    qsig_idx = policy_names.index("qsigdist")

    all_idx = list(range(len(labels)))
    print(f"Loaded {len(labels)} prompts × {len(policy_names)} policies")
    print(f"Baselines:  qsigdist Δ = {qsig_delta_full:+.2f}pp;  "
          f"oracle Δ = {oracle_delta_full:+.2f}pp\n")

    # =========================================================
    # STAGE A — in-sample sweep over (τ, δ)
    # =========================================================
    table_full = build_router_table(labels, all_idx, rates, policy_names,
                                     random_idx, qsig_idx)
    taus   = [0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
    deltas = [0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0]
    print("=" * 70)
    print("STAGE A — in-sample (τ, δ) sweep")
    print("=" * 70)
    best_in = None
    grid = []
    for tau in taus:
        for dlt in deltas:
            d_vs_rand, d_vs_q = evaluate_router(
                labels, all_idx, rates, policy_names, random_idx, qsig_idx,
                table_full, tau, dlt)
            router_d = d_vs_rand.mean() * 100
            r_minus_q = d_vs_q.mean() * 100
            n_routed = sum(1 for li in all_idx
                           if (info := table_full.get(category_of(labels[li]))) is not None
                              and info[1] <= tau
                              and (info[2] - info[3]) >= dlt)
            grid.append((tau, dlt, router_d, r_minus_q, n_routed))
            if best_in is None or r_minus_q > best_in[3]:
                best_in = (tau, dlt, router_d, r_minus_q, n_routed)

    # Show a small top-10 by router-minus-qsig
    grid.sort(key=lambda g: -g[3])
    print(f"  {'τ':>5}  {'δ':>5}  {'router Δ':>9}  {'router-qsig':>12}  {'n routed':>9}")
    for tau, dlt, rd, rmq, nr in grid[:10]:
        print(f"  {tau:>5.2f}  {dlt:>5.1f}  {rd:>+8.2f}  {rmq:>+11.2f}  {nr:>9}")
    print(f"\nBest in-sample (τ, δ) = ({best_in[0]}, {best_in[1]})")
    print(f"  router Δ      = {best_in[2]:+.2f}pp")
    print(f"  router − qsig = {best_in[3]:+.2f}pp")
    print(f"  prompts routed (vs default qsig): {best_in[4]} / {len(labels)}")
    # Decompose: for the best (τ, δ), which categories pass the gate?
    tau_b, dlt_b = best_in[0], best_in[1]
    print(f"\n  categories routed under best (τ={tau_b}, δ={dlt_b}):")
    for cat, (mw, H, md, qd) in sorted(table_full.items()):
        if H <= tau_b and (md - qd) >= dlt_b:
            n_cat = sum(1 for li in all_idx if category_of(labels[li]) == cat)
            print(f"    {cat:<12} n={n_cat:>2}  H={H:.2f}  mode={mw:<18} "
                  f"mode_Δ-qsig_Δ = {md - qd:+.2f}pp")

    # In-sample CI
    _, in_diff = evaluate_router(labels, all_idx, rates, policy_names,
                                  random_idx, qsig_idx, table_full,
                                  tau_b, dlt_b)
    rng = np.random.default_rng(20260514)
    boots = []
    for _ in range(10000):
        idx = rng.integers(0, len(in_diff), size=len(in_diff))
        boots.append(in_diff[idx].mean() * 100)
    in_ci = np.quantile(boots, [0.025, 0.975])
    print(f"\n  In-sample CI: router−qsig = {best_in[3]:+.2f}pp  "
          f"95% CI [{in_ci[0]:+.2f}, {in_ci[1]:+.2f}]")

    # =========================================================
    # STAGE B — held-out CV
    # =========================================================
    print()
    print("=" * 70)
    print("STAGE B — held-out CV (5-fold × 20 random repeats)")
    print("=" * 70)
    n = len(labels)
    K = 5
    REPEATS = 20

    all_test_deltas    = []
    all_test_diffs     = []
    fold_best_params   = []
    rng2 = np.random.default_rng(20260515)
    for rep in range(REPEATS):
        perm = rng2.permutation(n)
        folds = np.array_split(perm, K)
        for k in range(K):
            test_idx  = list(folds[k])
            train_idx = [li for f in (folds[:k] + folds[k+1:]) for li in f]
            table_train = build_router_table(labels, train_idx, rates, policy_names,
                                             random_idx, qsig_idx)
            # Sweep (τ, δ) on TRAIN
            best_train = None
            for tau in taus:
                for dlt in deltas:
                    _, tr_diff = evaluate_router(
                        labels, train_idx, rates, policy_names, random_idx,
                        qsig_idx, table_train, tau, dlt)
                    score = tr_diff.mean() * 100
                    if best_train is None or score > best_train[2]:
                        best_train = (tau, dlt, score)
            tau_t, dlt_t, _ = best_train
            fold_best_params.append((tau_t, dlt_t))
            # Apply best (τ, δ) to TEST
            te_d, te_diff = evaluate_router(
                labels, test_idx, rates, policy_names, random_idx,
                qsig_idx, table_train, tau_t, dlt_t)
            all_test_deltas.extend((te_d * 100).tolist())
            all_test_diffs .extend((te_diff * 100).tolist())

    test_d = np.array(all_test_deltas)
    test_q = np.array(all_test_diffs)
    print(f"  Held-out router Δ:           {test_d.mean():+.2f}pp")
    print(f"  Held-out router − qsigdist:  {test_q.mean():+.2f}pp")
    # CI by bootstrap
    rng3 = np.random.default_rng(20260516)
    bs = []
    for _ in range(10000):
        idx = rng3.integers(0, len(test_q), size=len(test_q))
        bs.append(test_q[idx].mean())
    ci_lo, ci_hi = np.quantile(bs, [0.025, 0.975])
    print(f"  95% CI (router − qsig):      [{ci_lo:+.2f}, {ci_hi:+.2f}]pp")
    if ci_lo > 0:
        print(f"  HELD-OUT ROUTER BEATS QSIG (CI excludes 0)")
    elif ci_hi < 0:
        print(f"  HELD-OUT ROUTER LOSES TO QSIG (CI excludes 0 on the wrong side)")
    else:
        print(f"  HELD-OUT not distinguishable from qsigdist")
    # Best params chosen most often by folds
    pc = Counter(fold_best_params)
    print(f"\n  Most-chosen (τ, δ) across folds:")
    for params, cnt in pc.most_common(5):
        print(f"    (τ={params[0]}, δ={params[1]})  chosen {cnt}/{K*REPEATS} folds")

    # Headroom captured (held-out, realistic)
    held_router = test_d.mean()
    headroom_captured_pct = (held_router - qsig_delta_full) / (oracle_delta_full - qsig_delta_full) * 100
    print(f"\n  Headroom captured (held-out): {headroom_captured_pct:.1f}% "
          f"of the way from qsigdist to in-sample oracle")

    out = {
        "best_in_sample": {
            "tau": best_in[0], "delta": best_in[1],
            "router_delta_pp": best_in[2],
            "router_minus_qsig_pp": best_in[3],
            "router_minus_qsig_ci_pp": [float(in_ci[0]), float(in_ci[1])],
            "n_routed": best_in[4],
        },
        "held_out": {
            "router_delta_pp": float(test_d.mean()),
            "router_minus_qsig_pp": float(test_q.mean()),
            "router_minus_qsig_ci_pp": [float(ci_lo), float(ci_hi)],
            "headroom_captured_pct": float(headroom_captured_pct),
            "fold_best_params": [list(p) for p in fold_best_params],
            "most_chosen_params": [(list(p), c) for p, c in pc.most_common(5)],
        },
    }
    out_path = os.path.join(THIS, "results/meta_iterate/router_gated_l1_1.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
