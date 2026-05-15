"""B2 variant (ii): fully-routed LSH FFN with atom-composition tiles.

Solves variant (i)'s data-per-bucket bottleneck STRUCTURALLY by
sharing atoms across buckets. Each bucket stores only a SHORT
RECIPE — the atom dictionary is global.

Architecture:
  Global: M atoms A_1..A_M, each ∈ {-1, 0, +1}^d (ternary)
  Per-bucket b: recipe = K (atom_index, scale) pairs
  Output for input x in bucket b: sum_{j=1..K} scale_{b,j} × A_{idx_{b,j}}

Construction (training):
  1. Take ALL training outputs (across all buckets)
  2. Compute SVD; top-M left singular vectors = atom basis
  3. Ternarize each atom: sign(component)
  4. For each bucket b: compute mean output across bucket members,
     then fit K-sparse representation on atom dictionary via
     greedy orthogonal matching pursuit (OMP)

Why shared atoms solve data-per-bucket:
  - Atoms see ALL N samples (not just per-bucket subset)
  - Recipes are short (K small): need only a few effective samples
    per bucket to estimate which atoms apply

Comparison points:
  (const)        train overall mean
  (bucket_mean)  per-bucket float mean
  (lut_ternary)  per-bucket ternary signature × scale (B2-i)
  (atom_comp)    NEW: per-bucket K-atom recipe over M shared atoms

Cold-bucket policy: fall back to constant.
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


def build_atom_dictionary(train_outs: np.ndarray, M: int):
    """Top-M left singular vectors of train_outs, ternarized.

    train_outs: (n_train, d). Returns (M, d) atom matrix in {-1, 0, +1}.
    Threshold for ternarization: per-atom median |component|.
    """
    # Center
    mu = train_outs.mean(axis=0, keepdims=True)
    centered = train_outs - mu
    # SVD: U (n_train, n_train), S (k,), Vt (k, d) where k = min(n, d)
    # We want top-M directions in d-space → use Vt (right singular vectors)
    # which span the row-space of train_outs.
    # Truncated SVD via eigendecomp of A^T A (faster for n<d)
    n, d = centered.shape
    if n <= d:
        # Compute via covariance
        gram = centered @ centered.T  # (n, n)
        eigvals, eigvecs = np.linalg.eigh(gram)
        idx = np.argsort(-eigvals)[:M]
        U = eigvecs[:, idx]
        sigma = np.sqrt(np.maximum(eigvals[idx], 1e-12))
        # V = (centered^T @ U) / sigma
        V = (centered.T @ U) / sigma[None, :]  # (d, M)
        components = V.T  # (M, d)
    else:
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(-eigvals)[:M]
        components = eigvecs[:, idx].T  # (M, d)
    # Ternarize
    atoms = np.zeros_like(components, dtype=np.int8)
    for m in range(M):
        c = components[m]
        tau = np.median(np.abs(c))
        atoms[m, c > tau] = 1
        atoms[m, c < -tau] = -1
    return atoms.astype(np.float64), mu  # return atoms + center for reconstruction


def fit_recipe(target: np.ndarray, atoms: np.ndarray, K: int):
    """K-sparse representation of target ∈ R^d on atoms (M, d) via
    orthogonal matching pursuit. Returns (K-list of indices, K-list of scales).
    """
    residual = target.copy()
    indices = []
    scales = []
    norms_sq = (atoms * atoms).sum(axis=1)  # (M,)
    valid = norms_sq > 0
    for _ in range(K):
        # Project residual onto atoms; pick atom with largest |projection|
        proj = atoms @ residual  # (M,)
        proj_norm = np.where(valid, proj / np.maximum(norms_sq, 1e-12), 0.0)
        # Don't repeat already-selected atoms
        for i in indices:
            proj_norm[i] = 0.0
        best = int(np.argmax(np.abs(proj_norm)))
        if abs(proj_norm[best]) < 1e-9: break
        # Refit ALL selected scales jointly
        sel_indices = indices + [best]
        A_sel = atoms[sel_indices]  # (k, d)
        # Least-squares: A_sel.T @ scales = target → scales = (A_sel @ A_sel.T)^-1 @ A_sel @ target
        gram = A_sel @ A_sel.T  # (k, k)
        rhs  = A_sel @ target   # (k,)
        try:
            sol = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
        indices = sel_indices
        scales = sol.tolist()
        # Update residual
        residual = target - (np.array(scales).reshape(-1, 1) * A_sel).sum(axis=0)
    return indices, scales


def predict_atom_comp(indices, scales, atoms, mu):
    """Reconstruct prediction: sum scale_j * atom_idx_j + mu."""
    if not indices:
        return mu.flatten()
    A_sel = atoms[indices]
    return mu.flatten() + (np.array(scales).reshape(-1, 1) * A_sel).sum(axis=0)


def cv_evaluate_atom_comp(ins, outs, buckets, prompt_labels,
                           M_atoms, K_recipe, n_splits=5, seed=20260514):
    """Same CV protocol as B2-i, with atom-composition predictor."""
    rng = np.random.default_rng(seed)
    unique_labels = sorted(set(prompt_labels))
    cos_atom = []
    for split in range(n_splits):
        perm = list(unique_labels); rng.shuffle(perm)
        n_test = max(1, len(perm) // 5)
        test_prompts = set(perm[:n_test])
        train_idx = [i for i, l in enumerate(prompt_labels) if l not in test_prompts]
        test_idx  = [i for i, l in enumerate(prompt_labels) if l in test_prompts]
        train_outs = outs[train_idx]
        # Build atom dictionary on TRAIN
        atoms, mu = build_atom_dictionary(train_outs, M_atoms)
        # Per-bucket: fit recipe for each train bucket
        by_b = defaultdict(list)
        for i in train_idx:
            by_b[int(buckets[i])].append(outs[i])
        recipes = {}
        for b, samples in by_b.items():
            arr = np.stack(samples, axis=0)
            mean_out = arr.mean(axis=0)
            target = mean_out - mu.flatten()
            indices, scales = fit_recipe(target, atoms, K_recipe)
            recipes[b] = (indices, scales)
        # Predict on TEST
        const_pred = train_outs.mean(axis=0)
        for i in test_idx:
            true = outs[i]
            true_norm = np.linalg.norm(true) + 1e-12
            b = int(buckets[i])
            if b in recipes:
                indices, scales = recipes[b]
                pred = predict_atom_comp(indices, scales, atoms, mu)
            else:
                pred = const_pred
            cos = float(np.dot(pred, true) / (np.linalg.norm(pred) * true_norm + 1e-12))
            cos_atom.append(cos)
    return np.array(cos_atom)


def predict_bucket_mean_train(out_samples):
    return out_samples.mean(axis=0)


def make_lut_tile(out_samples):
    mean = out_samples.mean(axis=0)
    tau = float(np.median(np.abs(mean)))
    S = np.zeros_like(mean, dtype=np.int8)
    S[mean > tau] = 1
    S[mean < -tau] = -1
    nz = (S != 0).sum()
    if nz == 0: return S, 0.0
    scale = float((mean * S).sum() / nz)
    return S, scale


def cv_evaluate_baselines(ins, outs, buckets, prompt_labels, n_splits=5, seed=20260514):
    """Returns const, bucket_mean, lut_ternary cos sims."""
    rng = np.random.default_rng(seed)
    unique_labels = sorted(set(prompt_labels))
    cos_const = []; cos_mean = []; cos_lut = []
    for split in range(n_splits):
        perm = list(unique_labels); rng.shuffle(perm)
        n_test = max(1, len(perm) // 5)
        test_prompts = set(perm[:n_test])
        train_idx = [i for i, l in enumerate(prompt_labels) if l not in test_prompts]
        test_idx  = [i for i, l in enumerate(prompt_labels) if l in test_prompts]
        const_pred = outs[train_idx].mean(axis=0)
        by_b = defaultdict(list)
        for i in train_idx:
            by_b[int(buckets[i])].append(outs[i])
        bm_table = {b: predict_bucket_mean_train(np.stack(v, axis=0)) for b, v in by_b.items()}
        lut_table = {}
        for b, v in by_b.items():
            S, sc = make_lut_tile(np.stack(v, axis=0))
            lut_table[b] = (S.astype(np.float64), sc)
        for i in test_idx:
            true = outs[i]
            true_norm = np.linalg.norm(true) + 1e-12
            b = int(buckets[i])
            cos_const.append(float(np.dot(const_pred, true) /
                                    (np.linalg.norm(const_pred) * true_norm + 1e-12)))
            bm = bm_table.get(b, const_pred)
            cos_mean.append(float(np.dot(bm, true) / (np.linalg.norm(bm) * true_norm + 1e-12)))
            if b in lut_table:
                S, sc = lut_table[b]
                pred = sc * S
                cos_lut.append(float(np.dot(pred, true) / (np.linalg.norm(pred) * true_norm + 1e-12)))
            else:
                cos_lut.append(float(np.dot(const_pred, true) /
                                      (np.linalg.norm(const_pred) * true_norm + 1e-12)))
    return np.array(cos_const), np.array(cos_mean), np.array(cos_lut)


def main():
    print("B2-ii — atom-composition tile prototype (fully routed, shared atoms)\n")
    for layer in (2, 15, 27):
        labels, positions, ins, outs = load_layer_pairs(layer)
        n = ins.shape[0]
        print(f"\n{'='*70}")
        print(f"Layer {layer}: {n} (input, output) pairs")
        print(f"{'='*70}")

        # Use k=6 (the recommended sweet spot)
        for k in (6,):
            sig = threshold_extract(ins, 2500)
            buckets = hash_buckets(sig, k)
            n_buckets = len(set(buckets.tolist()))
            print(f"\n  k={k}, n_buckets={n_buckets}, tau=2500")

            # Baselines
            cos_const, cos_mean, cos_lut = cv_evaluate_baselines(ins, outs, buckets, labels)
            print(f"  {'predictor':<22}  {'mean cos':>9}  {'median':>7}  "
                  f"{'frac > 0.3':>10}  {'frac > 0.5':>10}")
            print(f"  {'(const)':<22}  {cos_const.mean():>+8.3f}  "
                  f"{np.median(cos_const):>+7.3f}  "
                  f"{(cos_const > 0.3).mean():>9.2%}  "
                  f"{(cos_const > 0.5).mean():>9.2%}")
            print(f"  {'(bucket_mean float)':<22}  {cos_mean.mean():>+8.3f}  "
                  f"{np.median(cos_mean):>+7.3f}  "
                  f"{(cos_mean > 0.3).mean():>9.2%}  "
                  f"{(cos_mean > 0.5).mean():>9.2%}")
            print(f"  {'(lut_ternary B2-i)':<22}  {cos_lut.mean():>+8.3f}  "
                  f"{np.median(cos_lut):>+7.3f}  "
                  f"{(cos_lut > 0.3).mean():>9.2%}  "
                  f"{(cos_lut > 0.5).mean():>9.2%}")

            # Atom-composition: sweep M (atoms) and K (recipe length)
            print(f"\n  atom-composition (variant ii) — share atoms across buckets:")
            print(f"  {'M atoms':>8} {'K recipe':>9}  {'mean cos':>9}  {'median':>7}  "
                  f"{'frac > 0.3':>10}  {'frac > 0.5':>10}")
            for M_atoms in (16, 32, 64):
                for K_recipe in (4, 8, 16):
                    if K_recipe > M_atoms: continue
                    cos_atom = cv_evaluate_atom_comp(ins, outs, buckets, labels,
                                                       M_atoms, K_recipe)
                    marker = ""
                    if cos_atom.mean() > cos_const.mean() + 0.02: marker = " ✓"
                    if cos_atom.mean() > cos_mean.mean() + 0.02: marker += " ✓✓"
                    print(f"  {M_atoms:>8} {K_recipe:>9}  {cos_atom.mean():>+8.3f}  "
                          f"{np.median(cos_atom):>+7.3f}  "
                          f"{(cos_atom > 0.3).mean():>9.2%}  "
                          f"{(cos_atom > 0.5).mean():>9.2%}{marker}")


if __name__ == "__main__":
    main()
