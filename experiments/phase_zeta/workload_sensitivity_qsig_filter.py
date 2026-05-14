"""L4': workload-distribution sensitivity check.

The +2.12pp K=1 gain is on N=100 prompts spanning 29 categories
(balanced category coverage). Real deployment populations have
different mixes — code-heavy, dialog-heavy, technical-heavy, etc.
This script computes the K=1 gain (vs qsigdist) on category-skewed
subsets to estimate sensitivity.

Subsets:
  - "balanced" (full 100): the headline number
  - "code-heavy": all code/error/instr* prompts (~30 prompts)
  - "tech-heavy": all tech/technical/math/def* prompts (~25 prompts)
  - "dialog-heavy": dialog/cont/idiom/q prompts (~25 prompts)
  - "long-form": long_*, longform, poetry, hypothesis (~15 prompts)
  - "short factual": q_*, geography, history, biology (~12 prompts)

For each subset, report:
  - n
  - K=1 mean Δ vs random
  - qsigdist mean Δ vs random
  - paired (K=1 − qsigdist) and bootstrap CI
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import numpy as np

THIS = os.path.dirname(__file__)
META = os.path.join(THIS, "results/meta_iterate")
N50  = os.path.join(THIS, "results/n50_battery/battery_results.json")
N100 = os.path.join(THIS, "results/n100_incremental/battery_results.json")
K1_DIR = os.path.join(META, "qsig_filter_K1")


def category_of(label):
    return label.split("_")[0] if "_" in label else label


def match_rate(a, b):
    if not a or not b: return None
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]) / n if n else None


def load_baselines():
    with open(N50) as f:
        old = json.load(f)["trials"]
    with open(N100) as f:
        new = json.load(f)["trials"]
    return {(t["label"], t["mode"]): t["tokens"] for t in old + new}


def load_k1_tokens():
    out = {}
    for fn in os.listdir(K1_DIR):
        if not fn.endswith(".log"): continue
        with open(os.path.join(K1_DIR, fn)) as f:
            content = f.read()
        m = re.search(r"generated tokens\s*=\s*([\d\s\-]+)", content)
        if m:
            out[fn[:-4]] = [int(t) for t in m.group(1).strip().split()]
    return out


SUBSETS = {
    "balanced (full N=100)":  lambda l: True,
    "code-heavy":             lambda l: category_of(l) in {"code", "error", "instr", "instruct"},
    "tech-heavy":             lambda l: category_of(l) in {"tech", "technical", "math", "def", "definition"},
    "dialog-heavy":           lambda l: category_of(l) in {"dialog", "cont", "idiom", "q"},
    "long-form":              lambda l: category_of(l) in {"long", "longform", "poetry", "hypothesis"},
    "short factual":          lambda l: category_of(l) in {"q", "geography", "history", "biology", "color"},
    "logic/reasoning":        lambda l: category_of(l) in {"logic", "reasoning", "comparison", "negation",
                                                            "quantifier", "temporal", "conditional", "causal"},
}


def main():
    base = load_baselines()
    k1   = load_k1_tokens()
    labels = sorted(set(k1.keys()) & {l for (l, m) in base if m == "qsigdist"}
                                   & {l for (l, m) in base if m == "no_evict"}
                                   & {l for (l, m) in base if m == "random"})
    print(f"Common labels: {len(labels)}\n")

    rng = np.random.default_rng(20260514)
    print(f"{'subset':<30} {'n':>4}  {'K=1 Δ':>8}  {'qsig Δ':>8}  "
          f"{'(K1-qsig)':>10}  {'95% CI':>20}")
    for name, pred in SUBSETS.items():
        idxs = [i for i, l in enumerate(labels) if pred(l)]
        if not idxs:
            continue
        sub = [labels[i] for i in idxs]
        no_evict = [base[(l, "no_evict")] for l in sub]
        rand     = [base[(l, "random")] for l in sub]
        qsig     = [base[(l, "qsigdist")] for l in sub]
        k1_t     = [k1[l] for l in sub]
        ne_q  = np.array([match_rate(ne, q) for ne, q in zip(no_evict, qsig)])
        ne_r  = np.array([match_rate(ne, r) for ne, r in zip(no_evict, rand)])
        ne_k1 = np.array([match_rate(ne, k) for ne, k in zip(no_evict, k1_t)])
        d_k1   = (ne_k1 - ne_r) * 100
        d_qsig = (ne_q  - ne_r) * 100
        d_diff = (ne_k1 - ne_q) * 100
        # Bootstrap CI on the paired difference
        bs = []
        for _ in range(10000):
            idx = rng.integers(0, len(d_diff), size=len(d_diff))
            bs.append(d_diff[idx].mean())
        ci_lo, ci_hi = np.quantile(bs, [0.025, 0.975])
        sig = " ✓" if ci_lo > 0 else " ✗" if ci_hi < 0 else "  "
        print(f"{name:<30} {len(idxs):>4}  {d_k1.mean():>+7.2f}  {d_qsig.mean():>+7.2f}  "
              f"{d_diff.mean():>+9.2f}  [{ci_lo:>+5.2f}, {ci_hi:>+5.2f}]{sig}")


if __name__ == "__main__":
    main()
