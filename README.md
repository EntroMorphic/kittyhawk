# GLYPH

A routing-first ternary compute stack for Apple Silicon. Built on the thesis that base-2 systems ignore one-third of the natural signal — the structural zero — and that base-3 silicon primitives (TBL, masked-VCNT, SDOT) are already ternary-shaped underneath the base-2 framings that pave them over.

**Start here:** [`NORTH_STAR.md`](NORTH_STAR.md) — the compass.

---

## Status

Ground-zero rebuild in progress (initiated 2026-04-14). The prior implementation collapsed Multi-Trit Floating Point into a fixed-point reading with a shared global scale; the rebuild restores the F in MTFP (mantissa cells + per-block exponent metadata) and puts routing primitives first.

- **Substrate spec:** locked. See [`m4t/docs/M4T_SUBSTRATE.md`](m4t/docs/M4T_SUBSTRATE.md).
- **Numeric core:** block-native primitives for MTFP19, compile clean under `-Werror`, 5/5 test binaries pass.
- **Routing surface:** trit pack/unpack, TBL-based trit ops, masked-VCNT reducers, the five ternary routing primitives, ternary matmul (MTFP19 and SDOT-native MTFP4 paths).
- **Red-team:** two rounds complete. See [`docs/REMEDIATION_PLAN.md`](docs/REMEDIATION_PLAN.md).

## Architecture

```
m4t/         — the substrate. libm4t.a + block-native primitives.
  src/         numeric core, routing, ternary matmul, trit ops/pack/reducers
  tests/       5 test binaries, hand-derived integer golden values
  tools/       dev-only tools (trit_golden, lut_gen) — opt-in M4T_BUILD_TOOLS=ON
  docs/        the substrate specification
src/         — glyph wrapper headers over M4T. No glyph .c files yet.
tools/       — consumers built on the substrate.
               mnist_trit_lattice.c is the provisional primary consumer.
docs/        — thesis and remediation tracking.
journal/     — LMM-cycle research log (raw/nodes/reflect/synthesize per topic).
archive/     — superseded code and docs, retained for historical reference.
               See archive/README.md for what lives there and why.
```

## Numerical system

MTFP — Multi-Trit Floating Point, base 3. A value is `mantissa × 3^exponent`; the mantissa is an n-trit signed integer cell and the exponent is sidecar metadata at the block level (see spec §5–§7). Four cell widths at a fixed 16-byte block geometry:

| Cell | Container | Mantissa trits | Cells per block | Role |
|---|---|---|---|---|
| `m4t_mtfp4_t` | int8 | 4 | 16 | SDOT-native routing |
| `m4t_mtfp9_t` | int16 | 9 | 8 | narrow intermediates |
| `m4t_mtfp_t` | int32 | 19 | 4 | general activations (default) |
| `m4t_mtfp_w_t` | int64 | 39 | 2 | wide accumulation |

Binary floating point (IEEE-754 / float / double / float16 / bfloat16) is banned at runtime. The only sanctioned binary float lives in `m4t/tools/m4t_lut_gen.c`, a build-time LUT generator. See [`m4t/docs/M4T_SUBSTRATE.md`](m4t/docs/M4T_SUBSTRATE.md) for the full contract.

## Build

```bash
# Both targets require aarch64 + NEON (Apple Silicon or compatible ARM).
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

Optional dev tools (golden-value enumerator, offline LUT generator):

```bash
cmake -S m4t -B build-tools -DM4T_BUILD_TOOLS=ON
cmake --build build-tools -j
```

`-Werror` is on by default; warnings fail the build.

## Documentation map

| File | Purpose |
|---|---|
| [`NORTH_STAR.md`](NORTH_STAR.md) | The vision. Why base-3, why routing, what the end-game is not. Re-read when base-2 gravity pulls. |
| [`docs/THESIS.md`](docs/THESIS.md) | What would falsify the thesis. Current provisional consumer. Open benchmark bed. |
| [`m4t/docs/M4T_SUBSTRATE.md`](m4t/docs/M4T_SUBSTRATE.md) | The substrate specification. 16 numbered sections, traceable to conversation. |
| [`docs/REMEDIATION_PLAN.md`](docs/REMEDIATION_PLAN.md) | Red-team findings and remediation status. |
| [`CHANGELOG.md`](CHANGELOG.md) | Notable changes since the ground-zero rebuild. |
| [`m4t/README.md`](m4t/README.md) | Substrate-layer build and surface. |
| [`archive/README.md`](archive/README.md) | What's in the archive and why. |
| `journal/` | LMM-cycle research artifacts (raw → nodes → reflect → synthesize). |

## Results

Earlier-era measurements (MNIST 97.61% dense k-NN, 81.40% LSH routing) came from the pre-rebuild substrate. They are not claimed by the current state until a consumer runs on the rebuilt primitives and the numbers are re-measured. The thesis that "routing will naturally outperform dense in a base-3 environment" is open empirically — see [`docs/THESIS.md`](docs/THESIS.md) §2–§4.

## Origin

Forked from trix-z (ternary-routed transformer research). The original C kernels live in `archive/reference-code/` — quarantined because they contained float paths that do not belong in M4T.

## License

[MIT](LICENSE).
