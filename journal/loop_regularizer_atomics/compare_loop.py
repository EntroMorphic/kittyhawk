#!/usr/bin/env python3
"""Per-layer ε comparison: dense vs routed_k=4 on the edge_repetitive
loop-prompt. Both ran on the same 18-token shared context. Divergence
in activations starts at the attention layer (L0 attn_sub_norm) and
cascades. The question: does it amplify, stabilize, or shrink?"""
import numpy as np

HIDDEN, INTERMEDIATE, KV = 2560, 6912, 640
HEADER = 5 + 3 + 4 * 4
LAST_POS = 17

SITES = [
    ("x_norm_input",  HIDDEN),
    ("q_pre_rope",    HIDDEN),
    ("k_pre_rope",    KV),
    ("v",             KV),
    ("q_post_rope",   HIDDEN),
    ("k_post_rope",   KV),
    ("attn_sub_norm", HIDDEN),
    ("x_norm",        HIDDEN),
    ("gate",          INTERMEDIATE),
    ("up",            INTERMEDIATE),
    ("ffn_sub_norm",  INTERMEDIATE),
    ("block_output",  HIDDEN),
]

def load(prefix, layer):
    raw = np.fromfile(f"{prefix}.pos{LAST_POS}.layer{layer}.bin", dtype=np.uint8)
    out, pos = {}, HEADER
    for n, sz in SITES:
        out[n] = raw[pos:pos+sz*4].view(np.int32).copy()
        pos += sz*4
    return out

def fit_eps(a, b):
    """Best-scale ε between two int mantissa arrays (treat as same scale)."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    if (a*a).sum() == 0 and (b*b).sum() == 0: return 0.0, 1.0
    if (a*a).sum() == 0: return 1.0, 0.0
    s = (a*b).sum() / (a*a).sum()
    err = np.linalg.norm(a*s - b)
    ref = np.linalg.norm(b)
    return float(err / ref) if ref > 0 else float('nan'), float(s)

def raw_diff_pct(a, b):
    """Fraction of cells where dense != routed (exact bit-not-equal)."""
    return 100.0 * np.sum(a != b) / a.size

print(f"\n{'L':>3s}  {'site':14s}  {'eps':>8s}  {'diff%':>7s}  {'sub_max':>10s}  {'rou_max':>10s}")
print("-" * 80)
for L in range(30):
    d = load("/tmp/loop_dense", L)
    r = load("/tmp/loop_routed4", L)
    for site, _ in SITES:
        eps, _ = fit_eps(d[site], r[site])
        diff_pct = raw_diff_pct(d[site], r[site])
        if L < 3 or eps > 0.3 or diff_pct > 50 or site == "block_output":
            print(f"{L:>3d}  {site:14s}  {eps:>8.4f}  {diff_pct:>6.1f}%  "
                  f"{int(np.max(np.abs(d[site]))):>10d}  {int(np.max(np.abs(r[site]))):>10d}")

# Summary: per-layer block_output ε
print("\n--- block_output ε per layer (the carrier of divergence through residual stream) ---")
for L in range(30):
    d = load("/tmp/loop_dense", L)
    r = load("/tmp/loop_routed4", L)
    eps, _ = fit_eps(d["block_output"], r["block_output"])
    diff = raw_diff_pct(d["block_output"], r["block_output"])
    bar = "█" * min(40, int(eps * 40))
    print(f"  L{L:>2d}  ε={eps:.4f}  diff={diff:>5.1f}%  {bar}")
