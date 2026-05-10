# Saturation audit (compute-clamp-rescale-down pattern) — 2026-05-09

After the RMSNorm `gamma_bx > target_bx` fix (`4d4c917`), swept the substrate
for other instances of the same bug shape, and empirically measured saturation
across BitNet's full forward pass. Negative result; recording per discipline.

## Bug shape

The RMSNorm bug had three ingredients:

1. **Compute at scale_A** that is HIGHER than the intended output scale_B.
2. **Clamp to MTFP19_MAX** at scale_A — silent saturation if the intermediate
   exceeds MAX_VAL even when the output at scale_B fits.
3. **Rescale-down to scale_B** AFTER the clamp — divides, hides the
   saturation, output looks plausible but is magnitude-collapsed.

The signature pairing is "clamp at intermediate scale → divide to lower
scale." A kernel that does "divide to lower scale → clamp at output scale"
(at sufficient intermediate precision, e.g. int64 or int128) does NOT have
this bug — the clamp can only fire when the genuine output overflows.

## Audit method

Grepped every `m4t_mtfp_clamp64` and `m4t_mtfp_rescale_bx` call site in
`m4t/src/`. For each clamp, checked: at what scale does the clamp happen, and
is there a subsequent rescale-down? For each rescale_bx, checked: was the
input clamped at a higher scale upstream?

## Findings

All bx-aware kernels except RMSNorm already do divide-before-clamp at int64+
precision. None has the bug shape.

| kernel | pattern | verdict |
|---|---|---|
| `m4t_mtfp_bitlinear_scale_bx` | int128 prod ÷ den, then clamp | clean |
| `m4t_mtfp_bitlinear_scale_no_a8_bx` | uint96 prod ÷ den, then clamp | clean |
| `m4t_mtfp_relu2_inplace_bx` | int64 sq ÷ 3^k, then clamp | clean |
| `m4t_mtfp_elementwise_mul_bx` | int64 prod ÷ 3^k, then clamp | clean |
| `m4t_mtfp_rmsnorm_bx` | (was buggy) — pre-rescale γ, then per-cell at target_bx | **fixed `4d4c917`** |
| `m4t_mtfp19_to_mtfp4` | int32 q = src ÷ 6561, then clamp to ±40 | clean |
| `m4t_mtfp_rope_apply` | int64 (a·c − b·s) >> 29, then clamp | clean (calibrated shift) |
| matmul kernels (ternary, 5in8, sdot) | int64 acc, clamp on store | clean (no rescale-down) |

Non-`_bx` legacy variants (`m4t_mtfp_rmsnorm`, `m4t_mtfp_relu2_inplace`,
`m4t_mtfp_elementwise_mul`) push bx tracking onto the caller and have no
internal rescale-down step, so they cannot host the same bug shape. They are
NOT called by the BitNet harness; whether to deprecate them is left for a
future cleanup cycle.

## Empirical saturation in BitNet inference

**First pass (under-scoped):** sampled 4 layers × 2 positions × 12 captured
sites and reported "5 cells, only at L0/p7 gate." Red-team caught the
under-scoping; the broader sweep below replaces it.

**Broader sweep (all 30 layers × all 8 positions × 12 captured sites)** for
the canary prompt `"What is the capital of France?"`:

```
Total saturations: 227 across the full sweep.

Hot spots (27 (layer, position, site) triples with sat > 0):
  L 0 p1-p7   gate            : 2-5 cells (post-relu² outlier clipping)
  L24-L29 p* block_output     : 1-9 cells (NEW finding)
```

Two distinct mechanisms:

**Hot spot 1 — early-layer `gate`** (relu² output): `relu2_inplace_bx` does
divide-before-clamp at int64; saturation only fires when `relu(gate)² ×
3^FFN_BX` genuinely exceeds MAX_VAL. For `BITNET_FFN_BX = 6`, that's
`gate_real > √(MAX_VAL/3^6) ≈ 893`. A handful of outlier cells reach that
magnitude per position. Downstream `ffn_sub_norm` normalizes magnitude;
this is the open-saturation case the spec allows (Case S per §8.5).

**Hot spot 2 — late-layer `block_output` (NEW, likely RMSNorm-fix-induced):**
1–9 cells per position at L24–L29, capping the residual stream at MAX_VAL.
The harness comment at `bitnet_harness.c:44` claims "zero saturation across
all 30 layers" at `BITNET_ACT_BX=8` — but that claim was made *before* the
RMSNorm `gamma_bx > target_bx` fix (`4d4c917`). Pre-fix, post_attn_norm
outputs were magnitude-collapsed by ~6.5× (the silent saturation bug), so
the residual stream stayed well within MTFP19 range. Post-fix, the correct
(larger) magnitudes propagate and saturate at the previously-safe ACT_BX=8
in late layers. **The fix exposed a latent scale-tightness that the bug
was masking.** Open follow-up below.

## Q·K dot accumulator overflow check

`m4t_mtfp_vec_dot_i64` is used by attention scoring (`gesh/bitnet/bitnet_harness.c`
line 334). Theoretical worst-case for head_dim=128 is
`128 × MAX_VAL² ≈ 2^65.2` — exceeds int64 max (2^63 − 1). Empirically: worst
|Q·K| over the full forward pass is 2.37e10, leaving 2^28.5 headroom vs int64
max. **Practically safe**, and as of audit item #4 the bound is now both
documented and assertion-guarded:

- Header docstring (`m4t_mtfp.h`) names the worst-case constant
  `M4T_VEC_DOT_I64_K_MAX_WORST_CASE = 27` (every cell at ±MAX_VAL) and
  cites the empirical BitNet headroom (2^28.5).
- Debug builds (`-UNDEBUG`, used by `m4t_test`) run an O(n) abs+max scan
  and assert `n × max|x| × max|y| < 2^62`. Production builds (`NDEBUG`)
  skip the scan entirely — zero hot-path overhead.
- Regression test `test_vec_dot_i64_bound_constant` pins the constant
  to its derivation (`floor((2^63-1) / MAX_VAL²)`) so MAX_VAL drift can't
  silently desync it, and verifies BitNet-shape inputs (n=128, ±10K
  cells) produce results well within the bound.

## Conclusion (revised)

The compute-clamp-rescale-down silent-saturation pattern: still no further
instances found beyond RMSNorm. Open clamping in `gate` is benign and
spec-sanctioned.

**One follow-on issue surfaced by the broader sweep: late-layer block_output
saturation, likely RMSNorm-fix-induced.** Not blocking — the end-to-end
battery (`journal/post_rmsnorm_fix_battery_2026-05-09/`) shows 8/8 coherent
output, so 0.35% saturation at L25–L26 isn't visibly hurting generation
quality. Possible remediation: lower `BITNET_ACT_BX` from 8 → 7 (3× more
real-space headroom) and re-run the battery to verify quality preserved.
Recorded as TD; not addressed in this audit.

## Doc fixes

I introduced a stale doc reference during the post-fix doc shore-up:
`m4t_mtfp_vec_mul_inplace` is not a defined function — the actual kernel is
`m4t_mtfp_elementwise_mul_bx`. Lived in two places:
- `gesh/bitnet/README.md` (fixed in the same commit as the original audit)
- `m4t/README.md` (red-team caught; fixed in the same commit as this revision)

Plus the activation-kernel names in both READMEs were the non-`_bx` legacy
names; corrected to the `_bx` variants the harness actually calls.
