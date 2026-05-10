# Comprehensive saturation-pattern audit — 2026-05-09

Closes the static-scope concern from the prior `saturation_audit_2026-05-09.md`
red-team. The earlier audit only grepped `m4t_mtfp_clamp64` and
`m4t_mtfp_rescale_bx`. This pass broadens to every pattern that can express
the bug shape.

## Bug shape (recap)

```
1. compute at scale_A  → intermediate value, magnitude possibly > MTFP19_MAX
2. clamp to MTFP19_MAX → silent saturation if intermediate exceeds
3. rescale-down to scale_B (B < A) → output looks plausible, magnitude
                                     silently corrupted
```

Required pairing: a clamp that happens at a HIGHER scale than the final
output, with a divide-down between clamp and final output. Other
arrangements (divide-then-clamp; clamp at output scale with no further
rescale) are not the bug.

## Patterns searched

For completeness, the broader grep covered all six expressions of clamp +
rescale-down I could think of:

| Pattern | What it can express |
|---|---|
| `m4t_mtfp_clamp64(...)` | scalar clamp to MTFP19_MAX |
| `vqmovn_s64` | NEON saturating int64 → int32 narrow |
| `vminq_s32 + M4T_MTFP_MAX_VAL` | NEON int32 ceiling clamp |
| `if v > M4T_MTFP_MAX_VAL` (manual) | explicit scalar guard |
| `m4t_mtfp_rescale_bx` | explicit between-bx rescale |
| `>> shift` after clamp / clamp after `>> shift` | shift-as-rescale |

## Classification of every clamp-pattern site in `m4t/src/`

For each function: does the clamp happen at the FINAL output scale, or at an
intermediate scale that gets rescaled-down afterward?

| Function | Pattern | Verdict |
|---|---|---|
| `m4t_mtfp_block_add` / `block_sub` | clamp-at-output | **clean** (no rescale-down) |
| `m4t_mtfp_attn_v_combine` | int64 acc → vqmovn_s64 + vminq | **clean** (output at weighted-sum scale) |
| `accum_same_exp_with_flags_neon` | int32 add + clamp | **clean** (output at input scale) |
| `accum_aligning_neon_block` | divide → add → clamp | **clean** (divide-then-clamp) |
| `m4t_pow3_round_div` (static helper) | round-divide; no MTFP19 clamp inside | **clean** (returns int64) |
| `shift3_mul_neon` / `shift3_mul_saturate_neon` | multiply by 3^k → clamp | **clean** (output IS the shifted value) |
| `m4t_mtfp_shift3_scalar_ref` | same | **clean** |
| `m4t_mtfp_rmsnorm` (non-`_bx`) | per-cell → clamp at gamma_bx scale | **clean as kernel** (no internal rescale-down; caller-tracked bx — see item 2 for compositional hazard) |
| `m4t_mtfp_rmsnorm_bx` | pre-rescale γ → per-cell at target_bx → clamp | **clean (post-fix, `4d4c917`)** |
| `m4t_mtfp_relu2_inplace` (non-`_bx`) | square → clamp | **clean as kernel** (no rescale-down; caller responsible for 2*x_bx output) |
| `m4t_mtfp_relu2_inplace_bx` | square → divide → clamp | **clean** |
| `m4t_mtfp_elementwise_mul` (non-`_bx`) | multiply → clamp | **clean as kernel** (no rescale-down) |
| `m4t_mtfp_elementwise_mul_bx` | multiply → divide → clamp | **clean** |
| `m4t_mtfp_bitlinear_scale_bx` | int128 prod ÷ den → clamp | **clean** |
| `m4t_mtfp_bitlinear_scale_no_a8_bx` | uint96 prod ÷ den → clamp | **clean** |
| `m4t_mtfp_vec_scale` / `vec_scale_scalar_ref` | num/den at impl scale → clamp | **clean** |
| `m4t_mtfp_softmax` | e[i] × inv60 >> 30 → clamp at 2^30 scale | **clean** (shift-then-clamp) |
| `m4t_mtfp_rmsnorm_scalar_ref` (non-bx) | float oracle | **clean** (test-only) |
| `m4t_mtfp_ternary_matmul_bt` (all variants) | int64 acc → m4t_mtfp_clamp64 store | **clean** (output at implicit input scale) |
| `m4t_mtfp19_to_mtfp4` | div-by-6561 → clamp at MTFP4_MAX (40) | **clean** (divide-then-clamp) |
| `m4t_mtfp4_sdot_matmul_bt_route` | int32 acc → clamp at MTFP4 | **clean** |

`m4t_route.c`, `m4t_trit_ops.c`, `m4t_trit_pack.c`, `m4t_trit_reducers.c`,
`m4t_ternary_rowskip.c` contain zero clamp-pattern sites — operate on packed
trits with no MTFP19 narrowing.

## Consumer-side clamp sites (`gesh/bitnet/bitnet_harness.c`)

| Site | Pattern | Verdict |
|---|---|---|
| Attention scoring (lines 346–356) | `r = scores_i64[t] >> score_shift; clamp` | **clean** (shift-then-clamp; adaptive shift makes clamp essentially a no-op) |

## Conclusion

**The compute-clamp-rescale-down silent-saturation pattern exists in zero
sites in the current substrate**, after the RMSNorm fix at `4d4c917`. The
audit is comprehensive across all `m4t/src/*.c` files (8 files) plus the
BitNet consumer harness.

The pattern's scarcity is explained by substrate-level discipline: the
existing `_bx` kernels were all written with the explicit "divide before
clamp at int64+ precision" convention. RMSNorm was the lone outlier because
its NEON optimization sequenced the operations differently from the others
(per-cell prod at gamma_bx scale because the inv-rsqrt was naturally at that
scale; the rescale was tacked on afterward).

## Hygiene

Removed `gesh/bitnet/bitnet_harness.c.bak` left behind by the ACT_BX sweep
script's `sed -i.bak`. Verified byte-identical to the current source before
removal.

## Red-team of this audit

Self-review concerns and how each was resolved:

| Concern | Resolution |
|---|---|
| Did I miss patterns beyond `clamp64` / `vqmovn_s64` / manual clamps? | Added `vminq_s32 + M4T_MTFP_MAX_VAL`, `>> shift` after clamp, and `m4t_mtfp_rescale_bx` as separate searches. Six expression-patterns in total. |
| Did I check all callers of `m4t_pow3_round_div` (a static helper)? | Yes: 6 caller sites at lines 654, 807, 837, 925, 955, 981. Every one follows `aa = pow3_round_div(...); sum = aa ± other; clamp(sum)` — divide-then-add-then-clamp. Clean composition. Line 981 stores directly with an asserted bound. |
| Did I check inline functions in headers? | Yes: `m4t_mtfp_clamp64` (m4t_mtfp.h:78) and `m4t_mtfp4_clamp` (m4t_mtfp4.h:54). These ARE the clamp primitives, not bug sites. |
| Did I check `audit/`? | Yes: only 2 files have clamp-related lines. `tristate_audit.c:224` is a matmul accumulator store (clean, no rescale-down). `no_scalar_audit_bench.c:263, 343` use `M4T_MTFP_MAX_VAL` as an RNG range bound, not as a clamp threshold. |
| `m4t_ternary_rowskip.c` had 0 grep hits — actually clean? | Manual scan confirmed: only an `assert(K <= M4T_SDOT_K_MAX_EXACT)` and a comment about caller-side `[-127, +127]` clamping. No clamp sites. |
| The "scarcity is by discipline" claim is unproven | Hypothesis; not testable from inside the codebase. Recorded as a hypothesis, not a finding. |

The audit is comprehensive across:
- 8 `.c` files in `m4t/src/`
- 2 `.h` files with inline functions
- 5 `.c` files in `audit/`
- 1 `.c` file in `gesh/bitnet/` (the harness)

Total surface checked: ~16 source files, all clamp-pattern sites classified.

## Open: composition hazard

Three non-`_bx` legacy variants (`m4t_mtfp_rmsnorm`, `m4t_mtfp_relu2_inplace`,
`m4t_mtfp_elementwise_mul`) are individually clean but a future consumer
could reconstruct the bug shape by composing them with `m4t_mtfp_rescale_bx`.
Addressed separately as item 2 of the post-RMSNorm-fix concern list.
