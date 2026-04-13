# GLYPH

Glass-box ternary routing architecture for interpretable AI on Apple Silicon. Built on Trit Lattice LSH — classification as geometric partitioning on the MTFP lattice.

## Architecture

Glyph is built on **M4T** (M4 Ternary Extensions), a generic ternary compute substrate that lives in `m4t/`. M4T provides the numerical core — MTFP arithmetic, trit packing, TBL-based trit operations, NEON-optimized matmul, LayerNorm. Glyph is the consumer that wires these primitives into a transformer with ternary routing.

```
m4t/          ← generic ternary substrate (libm4t.a)
  src/          types, MTFP arithmetic, trit ops, matmul, layernorm
  tests/        35 tests, hand-derived integer golden values
  tools/        bench, size check, LUT gen, golden-value enumerator
  docs/         contract, pipeline, red-team

src/            ← glyph application layer (pre-extraction, retained for now)
reference-code/ ← trix-z source (quarantined, not on include path)
journal/        ← LMM design journal for the ternary opcode architecture
docs/           ← glyph-era remediation plans and red-team trackers
```

## Numerical System

**Ternary / Multi-Trit / Multi-Trit Floating Point only.**

Four MTFP cell widths at clean power-of-3 boundaries:

| Cell | Storage | Trits | Primary use |
|---|---|---|---|
| MTFP4 | int8 | 4 | Routing (unlocks SDOT as native ternary matmul) |
| MTFP9 | int16 | 9 | Narrow activations, intermediate scores |
| MTFP19 | int32 | 19 | General-purpose FFN, layernorm, matmul (default) |
| MTFP39 | int64 | 39 | Wide accumulation, high-precision paths |

**Banned:** `float`, `double`, IEEE-754 types. `int8`/`int16` as binary quantization types (allowed as MTFP cell containers at trit boundaries).

## Build

```bash
# Build the M4T substrate
cd m4t && cmake -S . -B build && cmake --build build -j && ctest --test-dir build

# Build the glyph application layer (pre-extraction)
cd .. && cmake -S . -B build && cmake --build build -j && ctest --test-dir build
```

Requires aarch64 + NEON (Apple Silicon or compatible ARM).

## Results

| Path | MNIST Accuracy | Float in pipeline |
|---|---|---|
| Trit Lattice LSH (L1 centroid, 2048 proj) | 81.40% | **zero** |
| Trit Lattice k-NN (L2, 512 proj) | 96.79% | **zero** |
| **Trit Lattice k-NN (L2, deskewed pixels)** | **97.61%** | **zero** |
| Float-trained, M4T all-ternary inference | 97.46% | training only |
| trix-z reference (float + STE) | 97.41% | everywhere |

The Trit Lattice LSH path uses zero float anywhere — not in data loading, not in "training" (integer statistics on the lattice), not in inference. See `m4t/docs/TRIT_LATTICE_LSH.md`.

## Status

- **M4T substrate:** Four MTFP cell types (4/9/19/39 trits), 6 TBL trit ops, 3 masked-VCNT reducers, 5 routing primitives (including end-to-end k-of-T routing), SDOT-native ternary matmul (MTFP4), ternary matmul (MTFP19), dense matmul (MTFP19/39), layernorm. 68 tests across 6 binaries. 17.7 KB `.text` (71% of 24 KB budget).
- **Glyph application layer:** not yet wired to M4T. Opcode tables and wrapper layer next.
- **See:** `m4t/docs/M4T_PIPELINE.md` for the roadmap, `m4t/docs/M4T_CONTRACT.md` for the opcode contract, `m4t/docs/M4T_BEYOND.md` for post-pipeline future work.

## Origin

Forked from trix-z (ternary-routed transformer research). The trix-z C kernels are retained in `reference-code/` for reference; they contain float paths that do not belong in glyph or M4T.

## License

MIT.
