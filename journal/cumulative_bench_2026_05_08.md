---
cycle: cumulative kernel bench (post all 2026-05-07 fixes)
phase: end-of-arc benchmark
date: 2026-05-08
scope: measure where BitNet inference performance lands after the
       compounding work from rowskip → K%80 dense fix → K%80
       audit remediation (xpacked, mtfp4, K=0 UB fixes).
companions: journal/k80_fix_closeout.md (K%80 dense fix),
            journal/k80_remediation.md (red-team remediation),
            journal/k80_audit_remediation.md (audit closeout).
---

# Cumulative kernel benchmark — where we landed

## Two measurements

### 1. Kernel-level (rowskip v2 across 210 BitLinear calls)

5 independent runs, deterministic seeds, n_iter=200 per kernel
inside each run, BitNet layer-0 weights × 30 layers × 7 BitLinears.

  Variant              Mean (ms)   σ      vs dense baseline
  --------------------+----------+------+------------------
  dense                  26.24    0.36     1.000×
  rs_no_skip             26.35    0.31    -0.4%  (tile-align neutralized ✓)
  rs (always-on)         26.04    0.30    +0.81%
  smart-dispatch (≥5%)   26.00    0.36    +0.92%

**rs_no_skip vs dense ≈ 0% confirms the K%80 fix captured the
tile-align bonus into dense itself**, where it now benefits all
callers (not just the rowskip path). Pre-K%80-fix this delta was
+4.9% (the side effect that inflated rowskip's headline).

Rowskip's residual benefit is +0.92% with smart dispatch — modest,
but real on the 4-5 BitLinears with substantial empty-row fraction.

### 2. End-to-end BitNet inference

bitnet_harness with --token 1 --layers 30 --positions N. User CPU
time, 5 runs at N=32, σ across runs = 0.008s.

  Positions   User time (s)
  -----------+--------------
  1           0.11   (setup-dominated)
  8           0.31
  16          0.56
  32 (5 runs) 1.074 ± 0.008

  Steady-state per-position user time: (1.074 - 0.11) / 31
                                     = 31.1 ms / token

  Of which:
    BitLinears (210 calls):  ≈ 26.0 ms (per kernel bench)
    Non-BitLinear:           ≈  5.1 ms (RMSNorm, RoPE, softmax,
                                        attention scoring, residuals,
                                        a8_quantize, vec_scale)

## Cumulative gain attribution

Pre-rowskip / pre-K%80-fix baseline (commit before f2eea9f):
  - BitLinear aggregate per token: ~27.98 ms
  - Total per-token user time: ~32 ms

Post-all-fixes (current head):
  - BitLinear aggregate per token: 26.04 ms (rowskip always-on)
                                   26.00 ms (smart dispatch)
  - Total per-token user time: 31.1 ms

Delta per token:
  - BitLinear share: -1.94 to -1.98 ms (~6.9%)
  - Total inference: -0.9 ms (~2.8% absolute)

The K%80 dense kernel fix accounts for nearly all of the gain
(~-1.74 ms/token from 27.98 → 26.24 ms). Rowskip smart-dispatch
adds ~-0.24 ms (~+0.9%). The xpacked K%80, mtfp4 K%16, and K=0 UB
fixes don't move BitNet's BitLinear path because:
  - xpacked is not in BitNet's call path (BitNet uses int8 X, not
    packed ternary X).
  - mtfp4 K%16 fix is no-op for K%16=0 (all BitNet K values).
  - K=0 UB never triggers in inference.

Their value is correctness/code-consistency, not BitNet speed.

## Honest limits

The remaining 5.1 ms/token of non-BitLinear work is now the
proportional bottleneck. It includes:
  - 30 × RMSNorm (input + post-attention + attn_sub_norm + ffn_sub_norm)
  - 30 × RoPE (Q and K rotations)
  - 30 × Softmax (attention scores)
  - 30 × A8 quantize (4 sites: q/k/v/o input, gate/up input, down input)
  - 30 × vec_scale (scale by alpha × absmax / 127, 7 BitLinears)
  - Residual sums (2 per layer)
  - Embedding lookup, LM head matmul
  - ReLU² + elementwise multiply (FFN intermediate)

If we wanted further speedup, that's where to look — but it's
spread across many small kernels, each individually a smaller
target than the ~50 µs/call savings we got from the K%80 fix on
down_proj.

## What this run did NOT measure

  - Attention compute (scores × V) — done via mtfp_ternary_matmul
    which is in this BitLinear count's tail. Per-token cost
    bounded by ~K × N × M_attn.
  - Memory bandwidth at warm vs cold cache. The 5-run σ (0.008s)
    suggests warm-cache steady-state.
  - First-position cold-cache penalty (single-position run was
    0.11s vs 30 ms steady-state — ~3× cold-cache penalty).
  - End-to-end including blob load (1.6 GB mmap dominates wall
    time on the first run after a clean boot).

## Disposition

The compute floor for BitNet inference on this substrate is now
~31 ms/token user CPU on Apple Silicon P-core, of which ~26 ms is
BitLinear matmuls (mostly down_proj at K=6912) and ~5 ms is
auxiliary kernels.

That's a real, measured number against a real model. The work
across rowskip + K%80 + audit remediation moved this from ~32
ms/token to 31 ms/token — modest but earned.

The next step-change would require either:
  - Algorithmic restructure (group-wise routed16 was tried;
    BitNet weight structure doesn't support it per the
    column-correlation analysis in journal/routed16_weight_structure.md)
  - A different operation in the path (post-ReLU² activation
    sparsity exists at ~58% median per token but never reaches
    the routed16 92% crossover per
    journal/routed16_activation_sparsity_finding.md)
  - Optimizing the non-BitLinear 5 ms (RMSNorm, RoPE, softmax,
    quant) — many small targets

This is the end of the local-optimization arc. Further compute
wins require structural decisions, not kernel patches.
