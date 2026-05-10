#!/usr/bin/env python3
"""Replicate the substrate-routed top-k decision in Python.

For the edge_repetitive 18-token forward pass at the LAST position
(position 17), reconstruct what K positions each attention head's
substrate routing selected at L0. Compare to what dense's softmax
would weight most.

Substrate routing (per gesh/bitnet/bitnet_harness.c):
  1. tau = 1/3-quantile of |q_post_rope[h]|
  2. Q signature: trit value per cell:
        +1 if v >  tau
         0 if -tau <= v <= tau
        -1 if v < -tau
  3. Same for each K[t]_kv_head.
  4. Distance(Q_sig, K[t]_sig) = popcount on packed-trit XOR =
     sum over cells of {0 if same trit, 1 if one zero one sign,
                        2 if opposite signs}.
  5. Top-4 by smallest distance.
"""
import numpy as np

HIDDEN, INTERMEDIATE, KV = 2560, 6912, 640
HEADER = 5 + 3 + 4 * 4
HEAD_DIM = 128
NUM_HEADS = 20
NUM_KV_HEADS = 5
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS

def load_layer0(prefix, position):
    """Return q_post_rope (HIDDEN), k_post_rope (KV), v (KV) at L0."""
    raw = np.fromfile(f"{prefix}.pos{position}.layer0.bin", dtype=np.uint8)
    pos = HEADER
    sites = [("x_norm_input", HIDDEN), ("q_pre_rope", HIDDEN),
             ("k_pre_rope", KV), ("v", KV),
             ("q_post_rope", HIDDEN), ("k_post_rope", KV),
             ("attn_sub_norm", HIDDEN), ("x_norm", HIDDEN),
             ("gate", INTERMEDIATE), ("up", INTERMEDIATE),
             ("ffn_sub_norm", INTERMEDIATE), ("block_output", HIDDEN)]
    out = {}
    for n, sz in sites:
        out[n] = raw[pos:pos+sz*4].view(np.int32).copy()
        pos += sz*4
    return out

def trit_signature(v_arr, tau):
    """Convert int32 vector to per-cell trit ∈ {-1, 0, 1} using tau threshold."""
    sig = np.zeros_like(v_arr, dtype=np.int8)
    sig[v_arr >  tau] =  1
    sig[v_arr < -tau] = -1
    return sig

def packed_trit_distance(sig_a, sig_b):
    """Hamming distance on the substrate's trit packing scheme.
    +1 vs +1: 0    0 vs 0: 0    -1 vs -1: 0
    +1 vs 0: 1     0 vs -1: 1   (one is zero, one is sign)
    +1 vs -1: 2    -1 vs +1: 2  (opposite signs)
    """
    d = np.zeros_like(sig_a, dtype=np.int32)
    diff = sig_a != sig_b
    opposite = (sig_a == -sig_b) & (sig_a != 0)
    d[diff] = 1
    d[opposite] = 2
    return int(d.sum())

# Reconstruct K cache from per-position L0 dumps for routed run.
# K cache layout: [position][kv_head][head_dim].
SEQ_K = 18
k_cache = np.zeros((SEQ_K, NUM_KV_HEADS, HEAD_DIM), dtype=np.int32)
for t in range(SEQ_K):
    d = load_layer0("/tmp/loop_routed4", t)
    k_cache[t] = d["k_post_rope"].reshape(NUM_KV_HEADS, HEAD_DIM)

# Get Q at the LAST position (where the next-token decision happens)
q_last = load_layer0("/tmp/loop_routed4", SEQ_K - 1)["q_post_rope"].reshape(NUM_HEADS, HEAD_DIM)

print(f"Reconstructed K_cache shape: {k_cache.shape}")
print(f"Q at last position shape: {q_last.shape}")

# For each attention head, compute substrate routing's chosen indices
print(f"\n=== Substrate routing's chosen indices per head (top-4 of {SEQ_K} positions) ===")
print(f"{'head':>5s} {'kv_head':>7s} {'tau':>10s} {'top-4 indices':30s} {'all distances'}")
for h in range(NUM_HEADS):
    kv_head = h // Q_PER_KV
    q = q_last[h]
    abs_sorted = np.sort(np.abs(q))
    tau = max(int(abs_sorted[len(q) // 3]), 1)
    q_sig = trit_signature(q, tau)
    distances = []
    for t in range(SEQ_K):
        k = k_cache[t, kv_head]
        k_sig = trit_signature(k, tau)
        distances.append(packed_trit_distance(q_sig, k_sig))
    distances = np.array(distances)
    top4 = np.argsort(distances)[:4]
    print(f"  {h:>3d}    {kv_head:>3d}     {tau:>8d}  {sorted(top4.tolist())!s:30s} {distances.tolist()}")

# Show what dense's softmax would weight most. Use raw Q·K dot products.
print(f"\n=== Dense's Q·K argmax positions per head (for comparison) ===")
print(f"{'head':>5s}  {'argmax_t':>8s}  {'top-4 by score':30s}  {'distance from routed top-4'}")
for h in range(NUM_HEADS):
    kv_head = h // Q_PER_KV
    q = q_last[h]
    scores = np.array([int(np.dot(q.astype(np.int64), k_cache[t, kv_head].astype(np.int64))) for t in range(SEQ_K)])
    top4_dense = np.argsort(-scores)[:4]
    abs_sorted = np.sort(np.abs(q))
    tau = max(int(abs_sorted[len(q) // 3]), 1)
    q_sig = trit_signature(q, tau)
    distances = np.array([packed_trit_distance(q_sig, trit_signature(k_cache[t, kv_head], tau)) for t in range(SEQ_K)])
    routed_top4 = sorted(np.argsort(distances)[:4].tolist())
    dense_top4 = sorted(top4_dense.tolist())
    overlap = len(set(routed_top4) & set(dense_top4))
    argmax_t = int(np.argmax(scores))
    print(f"  {h:>3d}    {argmax_t:>5d}    {dense_top4!s:30s}  overlap={overlap}/4")
