"""B2 variant (i): fully-routed LSH FFN — LUT-style ternary tiles.

Each bucket stores:
  S_b ∈ {-1, 0, +1}^d   (ternary output signature)
  scale_b ∈ R           (per-bucket scalar)

Inference: output = scale_b × S_b.

No matmul. No learned weights beyond the (S, scale) lookup table.
Substrate-routed dispatch + substrate-stored content. The tile IS
the routing target, not a compute kernel.

Construction (per bucket b):
  mean_b = average output of all train samples landing in bucket b
  S_b    = sign(mean_b - tau_b)         (threshold-extract; tau_b = median |mean_b|)
  scale_b = (mean_b · S_b) / |S_b|²    (best L2 fit of scalar × signature)

Comparison points:
  (a) constant            : train overall mean (no bucketing)
  (b) bucket-mean (float)  : B1.6 baseline; per-bucket float mean
  (c) LUT-ternary (this)   : per-bucket scale_b × ternary signature

Held-out 5-fold CV; per-prompt split (test prompts unseen at train).

Stratified by bucket occupancy: separately analyze tokens whose
bucket has ≥ N_train_min samples in train, to isolate the
"data-per-bucket" effect from "ternarization" effect.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict

import numpy as np

THIS = os.path.dirname(__file__)
DUMP_DIR = os.path.join(THIS, "results/ffn_dump")
HIDDEN = 2560


def parse_filename(fn):
    m = re.match(r"^(.+)_p(\d+)_l(\d+)(_out)?\.bin$", fn)
    if not m: return None
    return m.group(1), int(m.group(2)), int(m.group(3)), m.group(4) is not None


def load_layer_pairs(layer):
    by_key = defaultdict(dict)
    for fn in os.listdir(DUMP_DIR):
        p = parse_filename(fn)
        if not p: continue
        label, pos, l, is_out = p
        if l != layer: continue
        a = np.fromfile(os.path.join(DUMP_DIR, fn), dtype=np.int32)
        if a.shape[0] != HIDDEN: continue
        by_key[(label, pos)]["out" if is_out else "in"] = a
    labels, positions, ins, outs = [], [], [], []
    for (label, pos), d in by_key.items():
        if "in" not in d or "out" not in d: continue
        labels.append(label); positions.append(pos)
        ins.append(d["in"]); outs.append(d["out"])
    return labels, np.array(positions), \
           np.stack(ins, axis=0).astype(np.float64), \
           np.stack(outs, axis=0).astype(np.float64)


def threshold_extract(acts, tau):
    if tau == "adaptive":
        med = np.median(np.abs(acts), axis=1, keepdims=True)
        tau_arr = med
    else:
        tau_arr = np.full((acts.shape[0], 1), tau)
    sig = np.zeros_like(acts, dtype=np.int8)
    sig[acts > tau_arr] = 1
    sig[acts < -tau_arr] = -1
    return sig


def hash_buckets(sig, k):
    sub = sig[:, :k]
    digits = (sub + 1).astype(np.int64)
    powers = 3 ** np.arange(k, dtype=np.int64)
    return (digits * powers).sum(axis=1)


def make_lut_ternary_tile(out_samples: np.ndarray):
    """Given the train OUTPUTS for one bucket, return (S, scale) such
    that scale × S best approximates each sample in L2.

    We compute the bucket's mean output, ternarize via per-vector
    median absolute, then best-fit scale.
    """
    mean = out_samples.mean(axis=0)
    # tau = median |mean|
    tau = float(np.median(np.abs(mean)))
    S = np.zeros_like(mean, dtype=np.int8)
    S[mean > tau] = 1
    S[mean < -tau] = -1
    nz = (S != 0).sum()
    if nz == 0:
        return S, 0.0
    scale = float((mean * S).sum() / nz)
    return S, scale


def predict_bucket_mean(out_samples: np.ndarray):
    """B1.6 baseline: per-bucket float mean."""
    return out_samples.mean(axis=0)


def cv_evaluate(ins, outs, buckets, prompt_labels, n_splits=5,
                bucket_min_size=1, seed=20260514):
    """Returns dict: predictor → list of per-token cos sim.
    Filters to test tokens whose bucket has ≥ bucket_min_size train samples
    if bucket_min_size > 1; else evaluate all (with overall-mean fallback)."""
    rng = np.random.default_rng(seed)
    unique_labels = sorted(set(prompt_labels))

    cos_const = []
    cos_mean  = []
    cos_lut   = []
    n_evaluated_with_full_data = []
    n_evaluated_total = []

    for split in range(n_splits):
        perm = list(unique_labels); rng.shuffle(perm)
        n_test = max(1, len(perm) // 5)
        test_prompts = set(perm[:n_test])
        train_idx = [i for i, l in enumerate(prompt_labels) if l not in test_prompts]
        test_idx  = [i for i, l in enumerate(prompt_labels) if l in test_prompts]

        # Train-overall mean (constant)
        const_pred = outs[train_idx].mean(axis=0)

        # Group train outputs by bucket
        by_b = defaultdict(list)
        for i in train_idx:
            by_b[int(buckets[i])].append(outs[i])

        # Build bucket-mean and LUT tiles for buckets with >= bucket_min_size train samples
        bucket_mean_table = {}
        lut_table = {}
        for b, samples in by_b.items():
            if len(samples) < bucket_min_size:
                continue
            arr = np.stack(samples, axis=0)
            bucket_mean_table[b] = predict_bucket_mean(arr)
            S, sc = make_lut_ternary_tile(arr)
            lut_table[b] = (S.astype(np.float64), sc)

        # Predict on TEST
        n_full = 0; n_total = 0
        for i in test_idx:
            n_total += 1
            true = outs[i]
            true_norm = np.linalg.norm(true) + 1e-12
            b = int(buckets[i])
            # Constant
            cos_const.append(float(np.dot(const_pred, true) /
                                    (np.linalg.norm(const_pred) * true_norm + 1e-12)))
            # Bucket-mean
            bm = bucket_mean_table.get(b, const_pred)
            cos_mean.append(float(np.dot(bm, true) / (np.linalg.norm(bm) * true_norm + 1e-12)))
            # LUT-ternary
            if b in lut_table:
                S, sc = lut_table[b]
                pred = sc * S
                cos_lut.append(float(np.dot(pred, true) / (np.linalg.norm(pred) * true_norm + 1e-12)))
                n_full += 1
            else:
                # Cold-bucket fallback: use constant
                cos_lut.append(float(np.dot(const_pred, true) /
                                      (np.linalg.norm(const_pred) * true_norm + 1e-12)))
        n_evaluated_with_full_data.append(n_full)
        n_evaluated_total.append(n_total)

    return {
        "const": np.array(cos_const),
        "bucket_mean": np.array(cos_mean),
        "lut_ternary": np.array(cos_lut),
        "n_full": int(np.mean(n_evaluated_with_full_data)),
        "n_total": int(np.mean(n_evaluated_total)),
    }


def main():
    print("B2-i — LUT-ternary tile prototype (fully routed)\n")
    for layer in (2, 15, 27):
        labels, positions, ins, outs = load_layer_pairs(layer)
        n = ins.shape[0]
        print(f"\n{'='*70}")
        print(f"Layer {layer}: {n} (input, output) pairs")
        print(f"{'='*70}")

        for k in (6, 8, 10):
            sig = threshold_extract(ins, 2500)
            buckets = hash_buckets(sig, k)
            n_buckets = len(set(buckets.tolist()))

            print(f"\n  k={k}, n_buckets={n_buckets} (tau=2500)")

            # All-bucket evaluation (cold buckets fall back to constant)
            res = cv_evaluate(ins, outs, buckets, labels, bucket_min_size=1)
            print(f"  {'predictor':<14}  {'mean cos':>9}  {'median':>7}  {'fraction > 0.3':>14}")
            for name in ("const", "bucket_mean", "lut_ternary"):
                a = res[name]
                print(f"  {name:<14}  {a.mean():>+8.3f}  {np.median(a):>+7.3f}  "
                      f"{(a > 0.3).mean():>13.2%}")

            # Stratified: only buckets with ≥3 train samples
            res3 = cv_evaluate(ins, outs, buckets, labels, bucket_min_size=3)
            print(f"  STRATIFIED (≥3 train samples per bucket): "
                  f"n_eval_with_data ≈ {res3['n_full']}/{res3['n_total']}")
            for name in ("bucket_mean", "lut_ternary"):
                a = res3[name]
                print(f"    {name:<12}  {a.mean():>+8.3f}  {np.median(a):>+7.3f}  "
                      f"{(a > 0.3).mean():>13.2%}")


if __name__ == "__main__":
    main()
