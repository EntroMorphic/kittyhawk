"""B2-ii per-token CV variant.

The B2-ii original used per-PROMPT held-out (test prompts entirely
unseen). This variant splits at TOKEN level — train and test have
tokens from the same prompts. Tests whether the architecture works
when the train and test distributions match (deployment-realistic
when dispatching among trained patterns).

Same predictors as B2-ii:
  (const)        train overall mean
  (bucket_mean)  per-bucket float mean
  (lut_ternary)  per-bucket scale × ternary signature
  (atom_comp)    per-bucket K-atom recipe over M shared atoms

Output reads: if per-token CV gives much HIGHER atom_comp than
per-prompt CV, the per-prompt split was the bottleneck. If results
are similar, the bottleneck is data scale (B2-ii original
diagnosis).
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


def build_atom_dictionary(train_outs, M):
    mu = train_outs.mean(axis=0, keepdims=True)
    centered = train_outs - mu
    n, d = centered.shape
    if n <= d:
        gram = centered @ centered.T
        eigvals, eigvecs = np.linalg.eigh(gram)
        idx = np.argsort(-eigvals)[:M]
        U = eigvecs[:, idx]
        sigma = np.sqrt(np.maximum(eigvals[idx], 1e-12))
        V = (centered.T @ U) / sigma[None, :]
        components = V.T
    else:
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(-eigvals)[:M]
        components = eigvecs[:, idx].T
    atoms = np.zeros_like(components, dtype=np.int8)
    for m in range(M):
        c = components[m]
        tau = np.median(np.abs(c))
        atoms[m, c > tau] = 1
        atoms[m, c < -tau] = -1
    return atoms.astype(np.float64), mu


def fit_recipe(target, atoms, K):
    residual = target.copy()
    indices = []
    scales = []
    norms_sq = (atoms * atoms).sum(axis=1)
    valid = norms_sq > 0
    for _ in range(K):
        proj = atoms @ residual
        proj_norm = np.where(valid, proj / np.maximum(norms_sq, 1e-12), 0.0)
        for i in indices:
            proj_norm[i] = 0.0
        best = int(np.argmax(np.abs(proj_norm)))
        if abs(proj_norm[best]) < 1e-9: break
        sel_indices = indices + [best]
        A_sel = atoms[sel_indices]
        gram = A_sel @ A_sel.T
        rhs  = A_sel @ target
        try:
            sol = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
        indices = sel_indices
        scales = sol.tolist()
        residual = target - (np.array(scales).reshape(-1, 1) * A_sel).sum(axis=0)
    return indices, scales


def cv_per_token(ins, outs, buckets, n_splits=5, seed=20260514, M_atoms=16, K_recipe=4):
    """Per-TOKEN held-out CV."""
    rng = np.random.default_rng(seed)
    n = len(ins)
    cos_const = []; cos_mean = []; cos_lut = []; cos_atom = []
    for split in range(n_splits):
        perm = rng.permutation(n)
        n_test = n // 5
        test_idx = perm[:n_test].tolist()
        train_idx = perm[n_test:].tolist()
        const_pred = outs[train_idx].mean(axis=0)
        # bucket predictors
        by_b = defaultdict(list)
        for i in train_idx:
            by_b[int(buckets[i])].append(outs[i])
        bm_table = {b: np.stack(v, axis=0).mean(axis=0) for b, v in by_b.items()}
        lut_table = {}
        for b, v in by_b.items():
            arr = np.stack(v, axis=0)
            mean = arr.mean(axis=0)
            tau = float(np.median(np.abs(mean)))
            S = np.zeros_like(mean, dtype=np.int8)
            S[mean > tau] = 1
            S[mean < -tau] = -1
            nz = (S != 0).sum()
            sc = float((mean * S).sum() / nz) if nz > 0 else 0.0
            lut_table[b] = (S.astype(np.float64), sc)
        # atom dict
        atoms, mu = build_atom_dictionary(outs[train_idx], M_atoms)
        recipes = {}
        for b, samples in by_b.items():
            arr = np.stack(samples, axis=0)
            target = arr.mean(axis=0) - mu.flatten()
            indices, scales = fit_recipe(target, atoms, K_recipe)
            recipes[b] = (indices, scales)
        # Evaluate
        for i in test_idx:
            true = outs[i]
            tn = np.linalg.norm(true) + 1e-12
            b = int(buckets[i])
            cos_const.append(float(np.dot(const_pred, true) /
                                    (np.linalg.norm(const_pred) * tn + 1e-12)))
            bm = bm_table.get(b, const_pred)
            cos_mean.append(float(np.dot(bm, true) / (np.linalg.norm(bm) * tn + 1e-12)))
            if b in lut_table:
                S, sc = lut_table[b]
                pred = sc * S
                cos_lut.append(float(np.dot(pred, true) / (np.linalg.norm(pred) * tn + 1e-12)))
            else:
                cos_lut.append(float(np.dot(const_pred, true) /
                                      (np.linalg.norm(const_pred) * tn + 1e-12)))
            if b in recipes:
                indices, scales = recipes[b]
                if indices:
                    A_sel = atoms[indices]
                    pred_atom = mu.flatten() + (np.array(scales).reshape(-1, 1) * A_sel).sum(axis=0)
                else:
                    pred_atom = mu.flatten()
            else:
                pred_atom = const_pred
            cos_atom.append(float(np.dot(pred_atom, true) /
                                   (np.linalg.norm(pred_atom) * tn + 1e-12)))
    return {
        "const": np.array(cos_const),
        "bucket_mean": np.array(cos_mean),
        "lut_ternary": np.array(cos_lut),
        "atom_comp": np.array(cos_atom),
    }


def main():
    print("B2-ii per-TOKEN CV (random token-level holdout, NOT per-prompt)\n")
    for layer in (2, 15, 27):
        labels, positions, ins, outs = load_layer_pairs(layer)
        n = ins.shape[0]
        print(f"\n{'='*70}")
        print(f"Layer {layer}: n={n} samples")
        print(f"{'='*70}")
        sig = threshold_extract(ins, 2500)
        buckets = hash_buckets(sig, 6)
        n_buckets = len(set(buckets.tolist()))
        print(f"k=6, n_buckets={n_buckets}, mean samples/bucket = {n/n_buckets:.1f}")

        for M_atoms, K_recipe in [(16, 4), (16, 8), (32, 4), (32, 8), (64, 8)]:
            res = cv_per_token(ins, outs, buckets, M_atoms=M_atoms, K_recipe=K_recipe)
            print(f"\n  M={M_atoms} K={K_recipe}:")
            for name in ("const", "bucket_mean", "lut_ternary", "atom_comp"):
                a = res[name]
                marker = ""
                if name == "atom_comp":
                    delta = a.mean() - res["const"].mean()
                    marker = f"  ({'+' if delta >= 0 else ''}{delta:.3f} vs const)"
                print(f"    {name:<14}  mean={a.mean():>+8.3f}  median={np.median(a):>+7.3f}  "
                      f">0.3:{(a > 0.3).mean():>6.1%}  >0.5:{(a > 0.5).mean():>6.1%}{marker}")


if __name__ == "__main__":
    main()
