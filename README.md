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

## Results (rebuilt substrate)

| Path | MNIST accuracy | Wall time | Notes |
|---|---|---|---|
| **Trit Lattice LSH k-NN (N_PROJ=2048, k=3, fully routed)** | **97.31%** | 7.0 s | Symmetric balanced-base-3 deployment; §18-passing. |
| MTFP19 L1 k-NN (N_PROJ=2048, k=3) | 97.05% | 75.9 s | Information-fidelity baseline; dense. |
| Trit Lattice LSH k-NN (N_PROJ=512, k=5, fully routed) | 96.81% | 1.8 s | Smaller projection space. |
| Trit Lattice LSH (L1 centroid, N_PROJ=2048) | 81.40% | ~10 s | Centroid simplification; not real LSH. |

Routed k-NN beats the dense MTFP L1 baseline on accuracy (0.26 points at k=3) AND speed (10.8× faster). This is the first empirical confirmation of NORTH_STAR's "routing will naturally outperform dense in a base-3 environment" on the rebuilt substrate. See [`journal/routed_knn_mnist.md`](journal/routed_knn_mnist.md) for the full writeup.

The thesis remains open for harder benchmarks; MNIST is one data point, not the end-game. See [`docs/THESIS.md`](docs/THESIS.md) §2–§4.

## Origin

Forked from trix-z (ternary-routed transformer research). The original C kernels live in `archive/reference-code/` — quarantined because they contained float paths that do not belong in M4T.

## License

[MIT](LICENSE).
