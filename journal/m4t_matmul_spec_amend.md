---
cycle: m4t_matmul_spec_amend
phase: SYNTHESIZE (lightweight)
date: 2026-05-01
scope: amendments to m4t/docs/M4T_SUBSTRATE.md §8.4, §8.5, §17 cross-reference table, prompted by tier-3b/3c kernel implementation
companions: m4t/docs/M4T_SUBSTRATE.md · m4t/src/m4t_mtfp4.{h,c} · m4t/src/m4t_ternary_matmul.{h,c} · journal/xexpo_spec_amend.md
---

# Spec amendment — §8.4 / §8.5 / §17 for tier-3b/3c implementations

## Context

The tier-3b/3c kernel work (SDOT MTFP4 matmul + cell-width conversions + MTFP19 ternary matmul) updated three sections of `M4T_SUBSTRATE.md`:

- **§8.5 Widen / saturate / round resolutions** — added pointers to the new tier-3 implementations under each Case.
- **§17 spec-to-code cross-reference table** — added entries for `m4t_mtfp4_sdot_matmul_bt` (§8.4), `m4t_mtfp_ternary_matmul_bt` (§8.5 Case S), and the cell-width conversions (§10).

Per principle 7 (`CONTRIBUTING.md`), spec amendments require a journal cycle. `journal/xexpo_spec_amend.md` covered the §14.2 + §14.4 amendments for the cross-exp accumulator. This cycle records the §8.x + §17 amendments that landed alongside the tier-3b/3c kernels.

The amendment is lightweight (synthesize-only) because it documents implementation pointers rather than revising substrate semantics.

## Amendments landed

### §8.4 (SDOT as ternary matmul)

The spec text described `MTFP4 × MTFP4 → MTFP19` as the canonical SDOT shape with case-W output widening; the implementation realizes the analogous `MTFP4 × ternary → MTFP19` shape (ternary is a specialization of MTFP4 for weights). The substrate's "exact by construction" theorem holds a fortiori — the tighter operand range (`|W| ≤ 1` vs `|W| ≤ 40`) gives even more headroom. The K bound for exactness:

- For the spec's MTFP4 × MTFP4 shape: K · 40 · 40 = 1600K must fit MTFP19. K ≤ 363,206 ≈ 363K.
- For the implemented MTFP4 × ternary shape: K · 40 · 1 = 40K must fit MTFP19. K ≤ 14,528,268 ≈ 14.5M.

The implemented kernel exposes `M4T_SDOT_K_MAX_EXACT = 14,528,268` as a public macro and asserts `K ≤ M4T_SDOT_K_MAX_EXACT` in the kernel.

**Spec status:** §8.4's contract holds; the implementation realizes a specialization with a tighter bound.

### §8.5 (Widen / saturate / round resolutions)

Each Case now has named implementation pointers:

| Case | Implementation |
|---|---|
| Case S — saturate | `m4t_mtfp_block_add` / `_sub`, `m4t_mtfp_clamp64`, `m4t_mtfp4_clamp`, `m4t_mtfp_ternary_matmul_bt` store |
| Case W — widen | `m4t_mtfp4_sdot_matmul_bt` (MTFP4 × ternary → MTFP19, exact) |
| Case R — round (named opt-in) | `m4t_mtfp_vec_accum_aligning` (cross-exponent, §14.2 IMPLEMENTED) |

§14.2's status flipped from DEFERRED to IMPLEMENTED (covered by `xexpo_spec_amend.md`). §8.5's Case S list grew to include the ternary matmul store; Case W gained the SDOT entry.

### §17 (Spec-to-code cross-reference)

Three rows updated:

- `8.2` (cross-block add): "Deferred" → "IMPLEMENTED — see §14.2 (round-to-nearest, named opt-in)."
- `8.4` (SDOT ternary matmul): added `m4t/src/m4t_mtfp4.c (m4t_mtfp4_sdot_matmul_bt); MTFP19 variant in m4t/src/m4t_ternary_matmul.c`.
- `8.5` (widen/saturate/round): updated to list each Case's primary implementations.

## Why a lightweight cycle

Same reasoning as `xexpo_spec_amend.md`: these amendments document existing implementations rather than revising substrate semantics. The decisions (Path A alignment, base-3 round-to-nearest-even, per-block flag layout) were made in the tier-3 design + red-team cycles. This cycle records the spec-side reflection.

A heavier cycle would be appropriate if any of these amendments *changed* substrate behavior. They don't.

## Loop-back triggers

- **Back to a full cycle** if the spec's MTFP4 × MTFP4 shape (§8.4 verbatim, with both inputs being MTFP4 mantissas in [-40, 40]) is ever needed — the K bound differs by ~40× from the implemented MTFP4 × ternary shape, so the kernel's macro and assert would change.
- **Back to a full cycle** if a consumer demands MTFP4-output saturation from the SDOT path (the archived "Case S to MTFP4" variant) — that would re-introduce a kernel the spec re-read decided against.
