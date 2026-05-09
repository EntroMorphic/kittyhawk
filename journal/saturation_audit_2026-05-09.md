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

Forward pass on the 8-token canary prompt (15 positions if generating; here
just the prompt forward). For each captured site, counted cells with
`|mantissa| ≥ MTFP19_MAX = 581_130_733`:

```
layer  site               n_cells   sat   sat%   max|mantissa|
L 0 p7 gate                 6912      5   0.1%      581130733  ⚠ saturated
L *  * (all other sites)               0   0.0%      < MAX_VAL
```

5 cells out of 6912 in `gate` (post-relu²-inplace) at layer 0, position 7.
Zero saturation across all 30 layers × all other capture sites × all 8
positions.

These 5 saturations are **genuine outlier clipping**, not silent corruption:
- `relu2_inplace_bx` does the divide BEFORE the clamp at int64 precision.
- The clamp fires only when `relu(gate)² × 3^FFN_BX` actually exceeds
  MAX_VAL — for `BITNET_FFN_BX = 6`, that's `gate_real > √(MAX_VAL/3^6) =
  √797K ≈ 893`. Five outlier cells reach that magnitude in this prompt.
- Downstream `ffn_sub_norm` normalizes magnitude; the loss of exact
  magnitude on these 5 cells doesn't propagate visibly (block_output and
  ffn_sub_norm at the same layer/position both stay well within range).
- End-to-end battery (`journal/post_rmsnorm_fix_battery_2026-05-09/`)
  shows 8/8 prompts coherent — outlier clipping is not affecting quality.

## Conclusion

No further fixes warranted. The substrate's other bx-aware kernels already
follow the correct divide-before-clamp pattern. The occasional outlier
clipping in relu² is acceptable: it's the open-saturation case the spec
allows (Case S per §8.5), not the silent-saturation bug pattern.

## Doc fix

Caught one stale doc reference I introduced in the prior `gesh/bitnet/README.md`
rewrite: listed `m4t_mtfp_relu2_inplace` and `m4t_mtfp_vec_mul_inplace` as
the activation kernels, but the harness uses `_bx` variants
(`m4t_mtfp_relu2_inplace_bx`, `m4t_mtfp_elementwise_mul_bx`). Fixed in the
same commit as this journal entry.
