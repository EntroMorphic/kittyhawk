# GLYPH

A routing-first ternary compute stack for Apple Silicon. Built on the thesis that base-2 systems ignore one-third of the natural signal — the structural zero — and that base-3 silicon primitives (TBL, masked-VCNT, SDOT) are already ternary-shaped underneath the base-2 framings that pave them over.

**Start here:** [`NORTH_STAR.md`](NORTH_STAR.md) — the compass.

---

## Status

Ground-zero rebuild completed (initiated 2026-04-14). The prior implementation collapsed Multi-Trit Floating Point into a fixed-point reading with a shared global scale; the rebuild restored the F in MTFP (mantissa cells + per-block exponent metadata) and put routing primitives first. Since then the project has closed out six remediation rounds, converted every dense-resolver cascade tool to routing-native primitives, rebuilt the architecture with the signature-as-address reframe, and broken 97% accuracy on deskewed MNIST using a purely routed consumer at N_PROJ=16.

- **Substrate spec:** locked. See [`m4t/docs/M4T_SUBSTRATE.md`](m4t/docs/M4T_SUBSTRATE.md).
- **Numeric core:** block-native primitives for MTFP19, compile clean under repo-root `-Werror`, all tests pass.
- **Routing surface:** trit pack/unpack, TBL-based trit ops, masked-VCNT reducers, threshold-based signature extraction, ternary matmul (MTFP19 and SDOT-native MTFP4 paths).
- **Consumer library (`libglyph`):** higher-level routed k-NN infrastructure sitting on top of `libm4t` — MNIST loader, signature builder, bucket index, ternary multi-probe, resolver variants, CLI hyperparameter parser.
- **Production consumers:**
  - `mnist_routed_bucket` — single-table bucket-indexed LSH, signature-as-address (Axis 5)
  - `mnist_routed_bucket_multi` — multi-table bucket-indexed LSH with cross-table union-merge and summed-distance resolver (Axis 6); **breaks 97% at N_PROJ=16**
- **Architecture discipline:** every active routed consumer is zero-dense-scan at the application level; cascade tools are retained as research scaffolding.
- **Red-team:** six rounds plus a full libglyph refactor red-team complete. See [`docs/REMEDIATION_PLAN.md`](docs/REMEDIATION_PLAN.md) and recent `CHANGELOG.md` entries.
- **Tests:** 11/11 ctest binaries passing (`m4t_*` substrate tests, `glyph_wrapper`, `glyph_libglyph` unit tests, `routed_tool_smoke`, `multi_smoke`).

## Architecture

```
m4t/                  — the substrate (libm4t.a). Routing-first ternary kernels.
  src/                  numeric core, routing primitives, ternary matmul, trit ops/pack/reducers
  tests/                6 test binaries, hand-derived integer golden values
  tools/                dev-only tools (trit_golden, lut_gen) — opt-in M4T_BUILD_TOOLS=ON
  docs/                 substrate specification
src/                  — libglyph (libglyph.a). Consumer-side routed k-NN infrastructure.
  glyph_dataset.{h,c}   MNIST IDX loader + integer-moment deskew
  glyph_rng.{h,c}       xoshiro128+ RNG
  glyph_sig.{h,c}       random ternary projection + density-calibrated τ + signature encoder
  glyph_bucket.{h,c}    sorted bucket index keyed on packed-trit signatures
  glyph_multiprobe.{h,c} ternary Hamming neighbor enumeration (radius 0, 1, 2)
  glyph_resolver.{h,c}  6 resolver variants: VOTE, SUM, SUM-NEON4, PTM, voteweighted, radiusaware
  glyph_config.{h,c}    hyperparameter struct + CLI long-option parser
  glyph_*.h             thin wrapper headers that alias m4t_* into glyph_* namespace
tools/                — CLI consumer tools built on libglyph.
                         Production: mnist_routed_bucket, mnist_routed_bucket_multi
                         Diagnostic: fashion_atomics (resolver-gap atomics on Fashion-MNIST)
                         Research scaffolding: mnist_cascade_*, mnist_routed_knn, mnist_full_sweep,
                           mnist_resolver_sweep, mnist_local_*, mnist_lvg_*, mnist_probe_nproj16,
                           mnist_trit_lattice, mnist_routed_lattice, mnist_routed_weighted,
                           mnist_routed_trace, mnist_routed_amplified
tests/                — libm4t unit tests + glyph wrapper tests + libglyph unit tests
docs/                 — FINDINGS, THESIS, LIBGLYPH, HYPERPARAMETERS, REMEDIATION_PLAN
journal/              — LMM-cycle research log (raw → nodes → reflect → synthesize)
archive/              — superseded code and docs, retained for historical reference
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
# Requires aarch64 + NEON (Apple Silicon or compatible ARM).
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

## Running the production consumers

Every hyperparameter is a CLI flag. No source edits required to sweep N_PROJ, density, M, multi-probe radius, per-table candidate threshold, base seed, or dataset path. `--help` on either tool prints the full option list.

### Multi-table routed bucket (production best, Axis 6)

```bash
# Default: oracle pass over M ∈ {1,2,4,8,16,32,64} at N_PROJ=16
./build/mnist_routed_bucket_multi --data /path/to/mnist

# Full: oracle + VOTE/SUM/PTM resolvers at every M checkpoint
./build/mnist_routed_bucket_multi --data /path/to/mnist --mode full

# Single M checkpoint at M=16 to check the target neighborhood
./build/mnist_routed_bucket_multi --data /path/to/mnist --mode full --single_m 16
```

Default run reproduces the Axis 6 measurement byte-for-byte: **M=32 SUM reaches 97.24%** on deskewed MNIST at N_PROJ=16 — the first routed architecture in the project to exceed 97%.

### Single-table routed bucket (Axis 5)

```bash
# Tunes MAX_RADIUS × MIN_CANDIDATES at M=1 with H1 filter + H2+H3+H4 resolver
./build/mnist_routed_bucket --data /path/to/mnist
```

Default run reproduces the Axis 5 measurement: **82.58% at 9.9 μs/query** (MAX_R=2, MIN_C=100).

### Running tests

```bash
ctest --test-dir build
```

All 11 test binaries should pass: 5 `m4t` substrate tests, `glyph_wrapper` (alias surface), `glyph_libglyph` (20 unit tests covering RNG, bucket, multi-probe, resolvers), and 4 integration tests (`m4t_ternary_matmul`, `m4t_trit_pack`, `routed_tool_smoke`, `multi_smoke`).

## Documentation map

| File | Purpose |
|---|---|
| [`NORTH_STAR.md`](NORTH_STAR.md) | The vision. Why base-3, why routing, what the end-game is not. Re-read when base-2 gravity pulls. |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | Consolidated measurements and what they mean. Six axes covering accuracy, speed, inspectability, cascade architecture (dense + routed), signature-as-address, and multi-table composition. |
| [`docs/LIBGLYPH.md`](docs/LIBGLYPH.md) | `libglyph` library overview — module descriptions, usage flow, how to write a new consumer. |
| [`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md) | Every parameter across every experiment. Reference for reproduction. |
| [`docs/THESIS.md`](docs/THESIS.md) | What would falsify the thesis. Current empirical state. Benchmark bed open questions. |
| [`m4t/docs/M4T_SUBSTRATE.md`](m4t/docs/M4T_SUBSTRATE.md) | The substrate specification. 18 numbered sections, traceable to conversation. |
| [`docs/REMEDIATION_PLAN.md`](docs/REMEDIATION_PLAN.md) | Red-team findings and remediation status (first-light round; later rounds tracked in CHANGELOG). |
| [`CHANGELOG.md`](CHANGELOG.md) | Notable changes since the ground-zero rebuild. |
| [`m4t/README.md`](m4t/README.md) | Substrate-layer build and surface. |
| [`archive/README.md`](archive/README.md) | What's in the archive and why. |
| `journal/fashion_mnist_*.md` | Fashion-MNIST generalization, atomics diagnosis, density-sweep experiments. |
| `journal/` | LMM-cycle research artifacts (raw → nodes → reflect → synthesize). |

## Headline results (deskewed MNIST, single seed unless noted)

The architecture went through several phases. Numbers below reflect the current state after the routing-native refactor; see [`docs/FINDINGS.md`](docs/FINDINGS.md) for the full axis-by-axis story.

### Routed production architecture (Axis 5 / 6)

| Consumer | Config | Accuracy | ms/query | Architecture |
|---|---|---|---|---|
| `mnist_routed_bucket` | N_PROJ=16, MAX_R=2, MIN_C=100 | **82.58%** | ~0.01 | Single-table bucket-indexed LSH (signature-as-address). First genuinely routed consumer. |
| `mnist_routed_bucket_multi` | N_PROJ=16, M=16, SUM | **96.13%** | ~0.67 | 16 independent bucket tables + union-merge + summed-distance resolver. |
| `mnist_routed_bucket_multi` | **N_PROJ=16, M=32, SUM** | **97.24%** | ~1.92 | Target crossing. First routed architecture to exceed 97%. |
| `mnist_routed_bucket_multi` | N_PROJ=16, M=64, SUM | **97.31%** | ~4.13 | Diminishing returns regime. |

Multi-table routed bucket at M=32 (512 total signature trits) matches or slightly beats the pure-signature scaling curve at equivalent total bits (pure N_PROJ=512 is 97.06%; M=32 SUM is +0.18 points). Wall-time cost is ~2× faster than an equivalent dense N_PROJ=512 scan. Zero dense scans anywhere in the pipeline.

### Fashion-MNIST generalization (same architecture, no deskew)

| Consumer | Config | Accuracy | Notes |
|---|---|---|---|
| `mnist_routed_bucket_multi` | N_PROJ=16, M=64, d=0.33, SUM | **85.15%** | Baseline at balanced base-3 density |
| `mnist_routed_bucket_multi` | N_PROJ=16, M=64, d=0.25, SUM | **85.54%** | Dataset-optimal density (multi-seed confirmed, p<0.02) |

The architecture generalizes without code changes. Resolver gap is ~6× wider than MNIST, concentrated in the upper-body-garment cluster (classes 0/2/4/6: T-shirt, Pullover, Coat, Shirt). Atomics diagnosis (`tools/fashion_atomics.c`) shows the per-table min-Hamming gap is −0.036 bits with 65% of (query, table) pairs tied — the projection layer cannot discriminate these classes at per-table resolution. See `journal/fashion_mnist_atomics.md`.

### Historical reference (research scaffolding — O(N) dense outer loop with routed kernels)

The cascade tools listed in the architecture block above run routing primitives inside a dense outer loop. Their numbers were useful for producing the atomic probes that motivated the bucket architecture, but they are **not** production consumers. Retained for historical context:

| Config | Accuracy | Notes |
|---|---|---|
| `mnist_routed_knn` N=4096 k=5 rank-wt | 97.99 ± 0.01% (3 seeds) | Pre-bucket era headline; dense O(N) scan with routed kernels |
| `mnist_routed_knn` N=2048 k=3 majority | 97.79 ± 0.05% (3 seeds) | Sweet-spot scaffolding configuration |
| Dense pixel k-NN (classical baseline) | 97.16% | Control — deskewed pixels, L1 k-NN |

The Axis 5 reframe (`journal/routed_bucket_consumer.md`) explains why every cascade tool is scaffolding: each runs `m4t_popcount_dist` in an O(N_train) outer loop per query, which is dense architectural shape with routed kernels. The bucket consumers use the signature as a hash-table key — O(1) amortized per query, zero dense work at the filter stage.

## Origin

Forked from trix-z (ternary-routed transformer research). The original C kernels live in `archive/reference-code/` — quarantined because they contained float paths that do not belong in M4T.

## License

[MIT](LICENSE).
