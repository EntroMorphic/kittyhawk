# Glyph

A routing-first ternary compute stack for Apple Silicon. Built on the thesis that base-2 systems ignore one-third of the natural signal — the structural zero — and that base-3 silicon primitives (TBL, masked-VCNT, SDOT) are already ternary-shaped underneath the base-2 framings that pave them over.

**Start here:** [`NORTH_STAR.md`](NORTH_STAR.md) — the compass.

## Status (as of 2026-05-01)

Ground-zero rebuild **complete through Tier 3**. The substrate ships its full routing-first base-3 compute surface:

- **Tier 1 — pure base-3 layer.** Trit types, packing, element-wise ops (TBL), reductions (masked-VCNT). Zero MTFP entanglement.
- **Tier 2 — route primitives + MTFP19 mantissa arithmetic.** Five route primitives and same-block-exponent MTFP19 add/sub.
- **Tier 3a — cross-exponent accumulator** (`m4t_mtfp_vec_accum_aligning`). The kernel that distinguishes MTFP from fixed-point. Path A alignment, base-3 round-to-nearest-even, per-block status flags. Property-tested across 14 properties.
- **Tier 3b — SDOT MTFP4 matmul + cell-width conversions.** `m4t_mtfp4_sdot_matmul_bt` (Case W per §8.4: exact by construction up to `M4T_SDOT_K_MAX_EXACT = 14,528,268`), plus widen/narrow conversions.
- **Tier 3c — MTFP19 × packed-ternary matmul** (`m4t_mtfp_ternary_matmul_bt`). NEON-accelerated, int64 accumulator, Case S saturating store with optional flag tracking.

Each kernel layer was red-teamed adversarially after build; remediation cycles caught silent invariant violations, test coverage gaps, and spec deviations before they could land in a consumer. Full trail in `journal/`.

The prior implementation is preserved on disk in `01MAY26_archived/` (gitignored) as reference. The audit that motivated the reset is in `01MAY26_archived/REVIEWED.md`. The substrate's complete narrative is in [`CHANGELOG.md`](CHANGELOG.md).

What remains: consumer-side rebuild (libglyph, libtrain, tools). Out of scope for the substrate; planned separately when consumer demand drives it.

## Discipline

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full set of invariants. The headlines:

- **No binary floating point** in runtime kernels of `libm4t` or per-query/per-batch paths above it. MTFP (base-3) is the substrate-legal continuous geometry.
- **No random projections** in image classification — direct ternary quantization of pixels and gradients is the production representation.
- **No random weights** anywhere — every dimension must represent something specific.
- **No primitive without named consumer demand** — speculative infrastructure does not earn its place.
- **Substrate-level specs are upstream of kernel designs** — re-read the relevant spec section before any design memo.
- **DELETE = never.** Superseded code moves to an archive directory; it does not get removed.

## Numerical system

MTFP — Multi-Trit Floating Point, base 3. A value is `mantissa × 3^exponent`. Mantissa is an n-trit signed integer in one of four cell widths; exponent is sidecar metadata at the block level.

| Type | Container | Mantissa trits | Mantissa range | Cells per block |
|---|---|---|---|---|
| `m4t_mtfp4_t` | int8 | 4 | ±40 | 16 |
| `m4t_mtfp9_t` | int16 | 9 | ±9 841 | 8 |
| `m4t_mtfp_t` | int32 | 19 | ±581 130 733 | 4 |
| `m4t_mtfp_w_t` | int64 | 39 | ±1.72·10¹⁸ | 2 |

All blocks are 16 bytes — exactly one NEON vector. See [`m4t/docs/M4T_SUBSTRATE.md`](m4t/docs/M4T_SUBSTRATE.md) for the full contract.

## Build

```bash
# Requires aarch64 + NEON (Apple Silicon or compatible ARM).
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

`-Werror` is on by default. Eight ctest binaries, all green from a clean build.

## Documentation

| File | Purpose |
|---|---|
| [`NORTH_STAR.md`](NORTH_STAR.md) | The vision. Why base-3, why routing, what the end-game is not. Re-read when base-2 gravity pulls. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | The discipline — what every contribution must honor. |
| [`CHANGELOG.md`](CHANGELOG.md) | Complete narrative of the rebuild, commit-by-commit. |
| [`docs/THESIS.md`](docs/THESIS.md) | What would falsify the thesis. Open questions. |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | Running ledger of measurements. |
| [`docs/REMEDIATION_PLAN.md`](docs/REMEDIATION_PLAN.md) | Original kernel rebuild plan; preserved as historical artifact. |
| [`docs/REMEDIATION_PLAN_REDTEAM.md`](docs/REMEDIATION_PLAN_REDTEAM.md) | Adversarial review of the plan; 12 findings folded in. |
| [`docs/DESIGN_X-EXPO.md`](docs/DESIGN_X-EXPO.md) | Cross-exponent accumulator design (§14.2 named opt-in). |
| [`m4t/docs/M4T_SUBSTRATE.md`](m4t/docs/M4T_SUBSTRATE.md) | The substrate specification. Canonical reference for the numeric system. |
| [`m4t/README.md`](m4t/README.md) | Substrate-layer build, surface, and test inventory. |

### Journal — LMM cycles and red-team records

| File | Purpose |
|---|---|
| [`journal/xexpo_design_raw.md`](journal/xexpo_design_raw.md) → [`reflect`](journal/xexpo_design_reflect.md) → [`synthesize`](journal/xexpo_design_synthesize.md) → [`closeout`](journal/xexpo_design_closeout.md) | LMM cycle that scoped the cross-exponent kernel design before build. |
| [`journal/xexpo_kernel_redteam.md`](journal/xexpo_kernel_redteam.md) | Tier-3a kernel red-team (14 findings, all remediated). |
| [`journal/xexpo_spec_amend.md`](journal/xexpo_spec_amend.md) | Lightweight cycle: §14.2 + §14.4 spec amendments. |
| [`journal/m4t_matmul_redteam.md`](journal/m4t_matmul_redteam.md) | Tier-3b/3c kernel red-team (11 findings, all remediated). |

## License

[MIT](LICENSE).
