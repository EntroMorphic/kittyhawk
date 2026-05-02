# Glyph

A routing-first ternary compute stack for Apple Silicon. Built on the thesis that base-2 systems ignore one-third of the natural signal — the structural zero — and that base-3 silicon primitives (TBL, masked-VCNT, SDOT) are already ternary-shaped underneath the base-2 framings that pave them over.

**Start here:** [`NORTH_STAR.md`](NORTH_STAR.md) — the compass.

## Status

Ground-zero rebuild initiated **2026-05-01**. The prior implementation is preserved on disk in `01MAY26_archived/` (gitignored) as reference material. The rebuild starts from the kernels: a verified base-3 layer first, then MTFP arithmetic, then consumer infrastructure.

The audit that motivated the reset is in `01MAY26_archived/REVIEWED.md`. The kernel-level remediation plan is in [`docs/REMEDIATION_PLAN.md`](docs/REMEDIATION_PLAN.md).

This README will track real status as code lands.

## Discipline

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full set of invariants. The headlines:

- **No binary floating point** in runtime kernels of `libm4t` or per-query/per-batch paths above it. MTFP (base-3) is the substrate-legal continuous geometry.
- **No random projections** in image classification — direct ternary quantization of pixels and gradients is the production representation.
- **No random weights** anywhere — every dimension must represent something specific.
- **No primitive without named consumer demand** — speculative infrastructure does not earn its place.
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

`-Werror` is on by default.

## Documentation

| File | Purpose |
|---|---|
| [`NORTH_STAR.md`](NORTH_STAR.md) | The vision. Why base-3, why routing, what the end-game is not. Re-read when base-2 gravity pulls. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | The discipline — what every contribution must honor. |
| [`CHANGELOG.md`](CHANGELOG.md) | Notable changes. |
| [`docs/REMEDIATION_PLAN.md`](docs/REMEDIATION_PLAN.md) | Tier-2 and tier-3 kernel rebuild plan. |
| [`docs/THESIS.md`](docs/THESIS.md) | What would falsify the thesis. Open questions. |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | Running ledger of measurements. (Empty at ground zero.) |
| [`m4t/docs/M4T_SUBSTRATE.md`](m4t/docs/M4T_SUBSTRATE.md) | The substrate specification. Canonical reference for the numeric system. |
| [`m4t/README.md`](m4t/README.md) | Substrate-layer build and surface. |

## License

[MIT](LICENSE).
