"""Step 1 PoC calibration: build LSH FFN dict file from N=100 dumps.

For each layer in the dump set:
  1. Load all (input, output) pairs.
  2. Build atom dictionary (M ternary atoms via SVD of outputs).
  3. Per-bucket: fit K-sparse recipe via OMP on (bucket_mean - mu).
  4. Serialize.

Output binary format (little-endian):

  Header (32 bytes):
    char[4] magic = 'GLFF'
    uint32  version = 1
    uint32  num_layers
    uint32  d_model
    uint32  k_lsh         (trit bits used for hash)
    uint32  m_atoms
    uint32  k_recipe_max
    int32   tau

  Per-layer chunk (variable):
    uint32  layer_idx
    uint32  num_buckets
    int32   mu[d_model]                       (FFN-output overall mean)
    int8    atoms[m_atoms × d_model]          (ternary, packed as int8 bytes)
    Per-bucket entry (variable):
      uint32  bucket_id
      uint32  recipe_len
      uint32  atom_idx[recipe_len]
      double  scale[recipe_len]

The harness loader walks this format and builds in-memory tables.
"""
from __future__ import annotations

import argparse
import os
import re
import struct
from collections import defaultdict

import numpy as np

THIS = os.path.dirname(__file__)
DUMP_DIR = os.path.join(THIS, "results/ffn_dump")
HIDDEN = 2560

K_LSH = 6
TAU = 2500
M_ATOMS = 16
K_RECIPE = 4


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
    ins, outs = [], []
    for (label, pos), d in by_key.items():
        if "in" not in d or "out" not in d: continue
        ins.append(d["in"]); outs.append(d["out"])
    return np.stack(ins, axis=0).astype(np.float64), \
           np.stack(outs, axis=0).astype(np.float64)


def threshold_extract_first_k(acts, k, tau):
    """Take first k coords, threshold-extract to {-1, 0, +1}."""
    sub = acts[:, :k]
    sig = np.zeros_like(sub, dtype=np.int8)
    sig[sub > tau] = 1
    sig[sub < -tau] = -1
    return sig  # (n, k)


def hash_buckets(sig):
    digits = (sig + 1).astype(np.int64)
    powers = 3 ** np.arange(sig.shape[1], dtype=np.int64)
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
    return atoms, mu.flatten()


def fit_recipe(target, atoms_f, K):
    """OMP. atoms_f is float (atoms cast to float for compute).
    Returns (indices, scales)."""
    residual = target.copy()
    indices = []
    scales = []
    norms_sq = (atoms_f * atoms_f).sum(axis=1)
    valid = norms_sq > 0
    for _ in range(K):
        proj = atoms_f @ residual
        proj_norm = np.where(valid, proj / np.maximum(norms_sq, 1e-12), 0.0)
        for i in indices:
            proj_norm[i] = 0.0
        best = int(np.argmax(np.abs(proj_norm)))
        if abs(proj_norm[best]) < 1e-9: break
        sel_indices = indices + [best]
        A_sel = atoms_f[sel_indices]
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


def calibrate_layer(layer, M_atoms, K_recipe, n_min=1):
    print(f"  Layer {layer}: loading dumps...", flush=True)
    ins, outs = load_layer_pairs(layer)
    n = ins.shape[0]
    print(f"  Layer {layer}: {n} samples")
    sig = threshold_extract_first_k(ins, K_LSH, TAU)
    buckets = hash_buckets(sig)
    n_unique = len(set(buckets.tolist()))
    print(f"  Layer {layer}: {n_unique} unique buckets")
    atoms, mu = build_atom_dictionary(outs, M_atoms)
    atoms_f = atoms.astype(np.float64)
    by_b = defaultdict(list)
    for i in range(n):
        by_b[int(buckets[i])].append(outs[i])
    recipes = {}
    n_skipped = 0
    for b, samples in by_b.items():
        if len(samples) < n_min:
            n_skipped += 1
            continue  # Don't build recipe; harness will fall back to dense
        arr = np.stack(samples, axis=0)
        target = arr.mean(axis=0) - mu
        indices, scales = fit_recipe(target, atoms_f, K_recipe)
        recipes[b] = (indices, scales)
    print(f"  Layer {layer}: {len(recipes)} bucket recipes built "
          f"({n_skipped} skipped due to n_min={n_min})")
    # Round mu to int32
    mu_i32 = np.clip(mu, -2**31 + 1, 2**31 - 1).astype(np.int32)
    return mu_i32, atoms, recipes


def serialize(out_path, layer_data, M_atoms, K_recipe_max):
    """layer_data: dict layer_idx → (mu_i32, atoms_int8, recipes)."""
    layers = sorted(layer_data.keys())
    with open(out_path, "wb") as f:
        # Header
        f.write(b"GLFF")
        f.write(struct.pack("<IIIIIIi",
                            1,           # version
                            len(layers), # num_layers
                            HIDDEN,      # d_model
                            K_LSH,       # k_lsh
                            M_atoms,     # m_atoms
                            K_recipe_max,# k_recipe_max
                            TAU))        # tau
        for layer in layers:
            mu_i32, atoms_i8, recipes = layer_data[layer]
            f.write(struct.pack("<II", layer, len(recipes)))
            f.write(mu_i32.tobytes())
            f.write(atoms_i8.tobytes())
            for b, (indices, scales) in recipes.items():
                f.write(struct.pack("<II", b, len(indices)))
                if indices:
                    f.write(np.array(indices, dtype=np.uint32).tobytes())
                    f.write(np.array(scales, dtype=np.float64).tobytes())
    print(f"\nWrote: {out_path}")
    print(f"  Size: {os.path.getsize(out_path)} bytes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="2,15,27", help="comma-separated layer ids")
    ap.add_argument("--m-atoms", type=int, default=M_ATOMS)
    ap.add_argument("--k-recipe", type=int, default=K_RECIPE)
    ap.add_argument("--output", default=os.path.join(THIS, "results/lsh_ffn_dict.bin"))
    ap.add_argument("--n-min", type=int, default=1,
                    help="Minimum bucket sample count to include recipe; "
                         "buckets below threshold get NO recipe and harness "
                         "should fall back to dense FFN.")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    print(f"Calibrating LSH FFN dict")
    print(f"  layers: {layers}")
    print(f"  M_atoms: {args.m_atoms}")
    print(f"  K_recipe: {args.k_recipe}")
    print(f"  k_lsh: {K_LSH}, tau: {TAU}\n")

    layer_data = {}
    for layer in layers:
        layer_data[layer] = calibrate_layer(layer, args.m_atoms, args.k_recipe, args.n_min)

    serialize(args.output, layer_data, args.m_atoms, args.k_recipe)


if __name__ == "__main__":
    main()
