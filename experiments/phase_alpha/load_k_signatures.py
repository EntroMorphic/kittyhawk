"""Load K-cache signatures from ACTV2 dumps and build substrate +
baseline representations.

Per FROZEN spec (td27_geometric_prereg_v2_2026-05-12.md):
- Capture sites: k_pre_rope, k_post_rope (2 K-types per dump)
- 30 layers × 8 (prompt, position) combos × 5 KV heads × 2 K-types
- HEAD_DIM = 128
- threshold_extract τ = 5000 → substrate signature in {-1, 0, +1}^128

Baselines:
- B1: raw K float32 (substrate input; tests if extract adds structure)
- B2: equal-bits sign-only — D=203 sign bits via fixed random orthogonal
  projection of raw K
- B3: same as B2 with a DIFFERENT random seed (tests projection-specific
  vs threshold-extract distinctive)
"""
from __future__ import annotations

import glob
import os
import struct
import re
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


HEAD_DIM = 128
NUM_KV_HEADS = 5
KV_PROJ = HEAD_DIM * NUM_KV_HEADS  # 640
NUM_HIDDEN = 2560
NUM_INTERMEDIATE = 6912

THRESHOLD_TAU = 5000  # from FROZEN spec
B2_BITS = 203  # equal-bits target (128 * log2(3) ≈ 202.86 → 203)

# ACTV2 capture order (from gesh/bitnet/scripts/compare_activations.py)
CAPTURE_ORDER_V2 = [
    ("input_layernorm.output",          "hidden"),
    ("attn.q_pre_rope",                  "hidden"),
    ("attn.k_pre_rope",                  "kv_proj"),
    ("attn.v",                           "kv_proj"),
    ("attn.q_post_rope",                 "hidden"),
    ("attn.k_post_rope",                 "kv_proj"),
    ("attn_sub_norm.output",             "hidden"),
    ("post_attention_layernorm.output",  "hidden"),
    ("ffn.gate_post_relu2",              "intermediate"),
    ("ffn.up_proj",                      "intermediate"),
    ("ffn_sub_norm.output",              "intermediate"),
    ("block_output",                     "hidden"),
]


@dataclass
class KSig:
    """A single K vector with provenance."""
    prompt_id: str            # "dump", "multitoken", "multitoken.pos3", etc.
    position: int             # token position (best-effort from filename)
    layer: int
    kv_head: int              # 0..4
    site: str                 # "k_pre_rope" or "k_post_rope"
    k_int32: np.ndarray       # raw int32 mantissa, shape (HEAD_DIM,)


def read_actv2(path: str) -> dict:
    """Parse one ACTV2 dump file. Returns dict of capture-site → int32 array."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:5] != b"ACTV2":
        raise ValueError(f"not ACTV2: {path}")
    layer_idx, hidden, intermediate, kv_proj = struct.unpack("<iiii", data[8:24])
    if (hidden, intermediate, kv_proj) != (NUM_HIDDEN, NUM_INTERMEDIATE, KV_PROJ):
        raise ValueError(
            f"unexpected sizes in {path}: {(hidden, intermediate, kv_proj)}"
        )
    sizes = {"hidden": hidden, "intermediate": intermediate, "kv_proj": kv_proj}
    out = {"_layer": layer_idx}
    offset = 24
    for name, size_key in CAPTURE_ORDER_V2:
        n = sizes[size_key]
        arr = np.frombuffer(data[offset:offset + n * 4], dtype=np.int32)
        out[name] = arr.copy()
        offset += n * 4
    return out


def parse_filename(fname: str) -> Tuple[str, int, int]:
    """Extract (prompt_id, position, layer) from a dump filename.

    Conventions:
      dump.layer<L>.bin                 → prompt_id="dump", position=0
      multitoken.layer<L>.bin           → prompt_id="multitoken", position=0
      multitoken.pos<P>.layer<L>.bin    → prompt_id="multitoken", position=P
      <prompt>.pos<P>.layer<L>.bin      → prompt_id=<prompt>, position=P
        (used by remediation corpus c_dump_v2/: p1..p5)
    """
    base = os.path.basename(fname)
    m = re.match(r"^(.+?)\.layer(\d+)\.bin$", base)
    if not m:
        raise ValueError(f"can't parse: {base}")
    prefix = m.group(1)
    layer = int(m.group(2))
    if prefix == "dump":
        return "dump", 0, layer
    pos_m = re.match(r"^(.+)\.pos(\d+)$", prefix)
    if pos_m:
        return pos_m.group(1), int(pos_m.group(2)), layer
    return prefix, 0, layer


def iter_k_signatures(dump_dirs) -> Iterator[KSig]:
    """Yield every K vector across all layers, prompts, positions, heads, sites.

    dump_dirs: str or list of str — directories to scan.
    """
    if isinstance(dump_dirs, str):
        dump_dirs = [dump_dirs]
    paths = []
    for d in dump_dirs:
        paths.extend(glob.glob(os.path.join(d, "*.layer*.bin")))
    paths = sorted(paths)
    for path in paths:
        prompt_id, position, layer = parse_filename(path)
        data = read_actv2(path)
        if data["_layer"] != layer:
            # filename / header disagreement — trust header
            layer = data["_layer"]
        for site in ("attn.k_pre_rope", "attn.k_post_rope"):
            arr = data[site].reshape(NUM_KV_HEADS, HEAD_DIM)
            for kv in range(NUM_KV_HEADS):
                yield KSig(
                    prompt_id=prompt_id,
                    position=position,
                    layer=layer,
                    kv_head=kv,
                    site=site.split(".", 1)[1],
                    k_int32=arr[kv].copy(),
                )


def collect_all(dump_dir):
    """Return arrays: K_raw (N, HEAD_DIM) int32, plus metadata columns.

    dump_dir: str or list of str.
    """
    sigs = list(iter_k_signatures(dump_dir))
    N = len(sigs)
    K = np.zeros((N, HEAD_DIM), dtype=np.int32)
    meta = {
        "prompt_id": [s.prompt_id for s in sigs],
        "position": np.array([s.position for s in sigs], dtype=np.int32),
        "layer": np.array([s.layer for s in sigs], dtype=np.int32),
        "kv_head": np.array([s.kv_head for s in sigs], dtype=np.int32),
        "site": [s.site for s in sigs],
    }
    for i, s in enumerate(sigs):
        K[i] = s.k_int32
    return K, meta


def threshold_extract(K_int32: np.ndarray, tau: int = THRESHOLD_TAU) -> np.ndarray:
    """Substrate signature: clamp |k_i| > tau → sign(k_i); else 0."""
    sig = np.zeros_like(K_int32, dtype=np.int8)
    sig[K_int32 > tau] = 1
    sig[K_int32 < -tau] = -1
    return sig


def b2_signature(K_int32: np.ndarray, seed: int, n_bits: int = B2_BITS) -> np.ndarray:
    """Sign of a random orthogonal projection from HEAD_DIM to n_bits.

    Returns int8 array of {-1, +1}^n_bits.
    """
    rng = np.random.default_rng(seed)
    # Random orthogonal: QR of a Gaussian (D, n_bits) matrix
    G = rng.standard_normal((HEAD_DIM, n_bits))
    # Use QR for orthonormal columns when n_bits <= D; otherwise just normalize.
    # n_bits = 203, D = 128 — overcomplete. Use a random Gaussian projection
    # (Johnson-Lindenstrauss style); columns are not orthogonal but cover the
    # full output space.
    K_f = K_int32.astype(np.float64)
    proj = K_f @ G
    sig = np.where(proj >= 0, 1, -1).astype(np.int8)
    return sig


def pairwise_hamming_int8(sigs: np.ndarray) -> np.ndarray:
    """Categorical Hamming via one-hot @ one-hot.T (BLAS-backed).

    For sigs ∈ alphabet of size A with D cells:
      agree(i, j) = #{cells where sigs[i] == sigs[j]} = onehot_i · onehot_j
      Hamming    = D - agree
    Memory: O(N · D · A) for the one-hot; speed: one float32 GEMM.
    """
    N, D = sigs.shape
    vals = np.unique(sigs)
    A = len(vals)
    onehot = np.zeros((N, D * A), dtype=np.float32)
    for k, v in enumerate(vals):
        onehot[:, k::A] = (sigs == v).astype(np.float32)
    agree = onehot @ onehot.T  # (N, N) float32
    return np.rint(D - agree).astype(np.int32)


def pairwise_l2(K_f: np.ndarray) -> np.ndarray:
    """Continuous L2 distance for B1 (raw K)."""
    G = K_f @ K_f.T
    sq = np.diag(G)
    d2 = sq[:, None] + sq[None, :] - 2 * G
    np.maximum(d2, 0, out=d2)
    return np.sqrt(d2)


if __name__ == "__main__":
    import sys
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else "data/c_dump"
    K, meta = collect_all(dump_dir)
    print(f"Loaded N = {K.shape[0]} K-signatures (D = {K.shape[1]})")
    print(f"  prompt_ids: {sorted(set(meta['prompt_id']))}")
    print(f"  positions:  {sorted(set(meta['position'].tolist()))}")
    print(f"  layers:     {len(set(meta['layer'].tolist()))} unique")
    print(f"  kv_heads:   {sorted(set(meta['kv_head'].tolist()))}")
    print(f"  sites:      {sorted(set(meta['site']))}")
    print(f"  K magnitude: median={np.median(np.abs(K)):.0f}, "
          f"99%={np.quantile(np.abs(K), 0.99):.0f}, max={np.max(np.abs(K)):.0f}")
    sig = threshold_extract(K)
    nonzero_frac = np.mean(sig != 0)
    print(f"  substrate sig (τ={THRESHOLD_TAU}): nonzero frac = {nonzero_frac:.3f}")
