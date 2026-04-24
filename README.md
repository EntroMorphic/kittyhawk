# GLYPH

A routing-first ternary compute stack for Apple Silicon. Built on the thesis that base-2 systems ignore one-third of the natural signal — the structural zero — and that base-3 silicon primitives (TBL, masked-VCNT, SDOT) are already ternary-shaped underneath the base-2 framings that pave them over.

**Start here:** [`NORTH_STAR.md`](NORTH_STAR.md) — the compass.

---

## Status

Ground-zero rebuild completed (initiated 2026-04-14). The prior implementation collapsed Multi-Trit Floating Point into a fixed-point reading with a shared global scale; the rebuild restored the F in MTFP (mantissa cells + per-block exponent metadata) and put routing primitives first. Since then the project has closed out six remediation rounds, converted every dense-resolver cascade tool to routing-native primitives, rebuilt the architecture with the signature-as-address reframe, broken 97% accuracy on deskewed MNIST using a purely routed consumer at N_PROJ=16, and extended to CIFAR-10 and Fashion-MNIST via direct ternary quantization with GSH (Global Signature Hash) selective scoring.

- **Substrate spec:** locked. See [`m4t/docs/M4T_SUBSTRATE.md`](m4t/docs/M4T_SUBSTRATE.md).
- **Numeric core:** block-native primitives for MTFP19, compile clean under repo-root `-Werror`, all tests pass.
- **Routing surface:** trit pack/unpack, TBL-based trit ops, masked-VCNT reducers, threshold-based signature extraction, ternary matmul (MTFP19 and SDOT-native MTFP4 paths).
- **Signature path:** direct ternary quantization is the preferred production path. Pixel intensities are quantized directly to {-1, 0, +1} via density-calibrated thresholds — no random projection matrix required. Optional gradient features extend discrimination to texture-rich datasets (CIFAR-10, Fashion-MNIST).
- **Consumer library (`libglyph`):** higher-level routed k-NN infrastructure sitting on top of `libm4t` — MNIST/Fashion-MNIST/CIFAR-10 loader with optional normalization, signature builder, bucket index, ternary multi-probe, resolver variants, CLI hyperparameter parser.
- **Production consumers (default build, rule-compliant):**
  - `direct_lsh` — direct ternary quantization consumer with GSH selective scoring; supports MNIST, Fashion-MNIST, and CIFAR-10 (production best across all three datasets)
  - `structured_lsh`, `structured_gsh`, `sstt_precog`, `ig_scored`, `inverted_ig`, `block_distance` — additional direct-signature or structured-signature consumers
- **Legacy random-projection consumers (opt-in, `-DGLYPH_BUILD_LEGACY_RP=ON`):**
  - `mnist_routed_bucket` — single-table bucket-indexed LSH, signature-as-address (Axis 5)
  - `mnist_routed_bucket_multi` — multi-table bucket-indexed LSH (Axis 6); **breaks 97% at N_PROJ=16** — retained for benchmark reproducibility only (see `docs/LIBGLYPH.md`)
- **Architecture discipline:** every active routed consumer is zero-dense-scan at the application level; cascade tools are retained as research scaffolding.
- **Red-team:** six rounds plus a full libglyph refactor red-team complete. See [`docs/REMEDIATION_PLAN.md`](docs/REMEDIATION_PLAN.md) and recent `CHANGELOG.md` entries.
- **Tests:** **14/14 ctest binaries** passing in the default build (`m4t_*` substrate tests, `glyph_wrapper`, `glyph_libglyph` unit tests, plus 5 `train_*` routed-autodiff tests). 16/16 with `-DGLYPH_BUILD_LEGACY_RP=ON` (adds `routed_tool_smoke` and `multi_smoke` against the legacy random-projection consumers).
- **Routed autodiff MVP (`libtrain.a`):** consumer-layer training scaffolding per NORTH_STAR §13. Scalar forward + backward through `tlinear` (dense ternary linear) and `rroute` (routed top-k dispatch), hysteresis-aware re-quantization of float latents to trits, 5 tests covering gradient checks, convergence, edge cases, and expert-collapse diagnosis on a 10-class toy. Opt-in via `-DGLYPH_BUILD_TRAIN=ON` (default ON). See [`train/README.md`](train/README.md).
- **Substrate distance refinement (2026-04-24):** `go_probe` cycle identified a density-scaling bias in raw trit Hamming that `hamming_norm = H · 1024 / (|a|₀ + |b|₀ + 1)` corrects. +48pp on Go phase-ID (40% → 88%) via a one-line fix. Red-team confirmed the result under game-wise split (no within-game leakage), but also showed the gain is mostly density-recovery, not a new structural axis. Image-pipeline measurement on MNIST/Fashion/CIFAR is the next decisive test (queued). See [`journal/substrate_distance_refinement_closeout.md`](journal/substrate_distance_refinement_closeout.md).

## Architecture

```
m4t/                  — the substrate (libm4t.a). Routing-first ternary kernels.
  src/                  numeric core, routing primitives, ternary matmul, trit ops/pack/reducers
  tests/                7 test binaries, hand-derived integer golden values
  tools/                dev-only tools (trit_golden, profile) — opt-in M4T_BUILD_TOOLS=ON
  docs/                 substrate specification
train/                — libtrain.a. Routed autodiff MVP (consumer-layer, §13).
  src/                  backward_linear, backward_routed, requantize (hysteresis)
  tests/                5 tests: gradient checks, 2- and 10-class toys, edge cases
  README.md             scope, primitives, MVP findings, hyperparameters
src/                  — libglyph (libglyph.a). Consumer-side routed k-NN infrastructure.
  glyph_dataset.{h,c}   dataset loader + deskew + normalization + gradients
  glyph_rng.{h,c}       xoshiro128+ RNG
  glyph_sig.{h,c}       direct quantization + random ternary projection + τ calibration
  glyph_bucket.{h,c}    sorted bucket index keyed on packed-trit signatures
  glyph_multiprobe.{h,c} ternary Hamming neighbor enumeration (radius 0, 1, 2)
  glyph_probe.{h,c}     shared multi-probe candidate collection (probe/reset/table)
  glyph_resolver.{h,c}  7 resolver variants: VOTE, SUM, SUM-NEON4, PTM, KNN, voteweighted, radiusaware
  glyph_config.{h,c}    hyperparameter struct + CLI long-option parser
  glyph_*.h             thin wrapper headers that alias m4t_* into glyph_* namespace
tools/                — CLI consumer tools built on libglyph or standalone.
                         Default build (rule-compliant): direct_lsh, structured_lsh,
                           structured_gsh, sstt_precog, ig_scored, inverted_ig,
                           block_distance, csa_classifier, go_probe
                         Legacy (random-projection, opt-in -DGLYPH_BUILD_LEGACY_RP=ON,
                           retained for Axis 5/6 benchmark reproducibility):
                           mnist_routed_bucket{,_multi}, mnist_cascade_*,
                           mnist_routed_{knn,lattice,trace,weighted,amplified},
                           mnist_trit_lattice, mnist_full_sweep, mnist_resolver_sweep,
                           mnist_local_*, mnist_lvg_*, mnist_probe_nproj16,
                           fashion_atomics, cifar_seed_overlap, dynamic_nproj,
                           subsetted_multi, bruteforce_nproj, layered_lsh,
                           specialist_rerank, centroid_routed, conv_lsh
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

Binary floating point (IEEE-754 / float / double / float16 / bfloat16) is banned in every runtime kernel of `libm4t` and in every per-query / per-batch path of `libglyph`. Sanctioned non-runtime float sites are enumerated in [`m4t/docs/M4T_SUBSTRATE.md`](m4t/docs/M4T_SUBSTRATE.md) §12: archived build-time LUT generator, microbenchmark display math, and one-shot dataset ingestion.

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

### Direct LSH (production best, direct ternary quantization)

```bash
# CIFAR-10: direct quantization with gradient features and normalization
./build/direct_lsh --data /path/to/cifar10 --no_deskew --normalize --density 0.395 --gradients --m_max 64

# Fashion-MNIST: same configuration
./build/direct_lsh --data /path/to/fashion-mnist --no_deskew --normalize --density 0.395 --gradients --m_max 64

# MNIST: deskewed pixels only, no gradients or normalization needed
./build/direct_lsh --data /path/to/mnist --density 0.10 --m_max 64
```

Default runs reproduce the production measurements with GSH selective scoring.

### go_probe — base-3-native benchmark probing (Go positions)

```bash
# Requires a directory of 19×19 SGF files on disk.
./build/go_probe /path/to/sgf_dir --max_games 2000 --sample_every 5 \
    --encoding {raw,contrast3} \
    --metric   {hamming,hamming_norm} \
    --task     {phase,same_game} \
    --split    {position,game}
```

Standalone SGF parser + Go rules engine + brute-force Hamming k-NN on raw 361-trit position state. Used for the `base3_benchmarks` and `substrate_distance_refinement` cycles; identified a density-scaling bias in raw trit Hamming and a one-line fix (`hamming_norm`) that recovers +48pp on Go phase-ID under a leakage-free game-wise split. See `journal/base3_go_probe.md` and `journal/substrate_distance_refinement_closeout.md`.

### Multi-table routed bucket (Axis 6, legacy random-projection path — opt-in)

```bash
# Reconfigure with the legacy flag, rebuild:
cmake -S . -B build-legacy -DGLYPH_BUILD_LEGACY_RP=ON
cmake --build build-legacy -j

# Default: oracle pass over M ∈ {1,2,4,8,16,32,64} at N_PROJ=16
./build-legacy/mnist_routed_bucket_multi --data /path/to/mnist

# Full: oracle + VOTE/SUM/PTM resolvers at every M checkpoint
./build-legacy/mnist_routed_bucket_multi --data /path/to/mnist --mode full

# Single M checkpoint at M=16 to check the target neighborhood
./build-legacy/mnist_routed_bucket_multi --data /path/to/mnist --mode full --single_m 16
```

Default run reproduces the Axis 6 measurement byte-for-byte: **M=32 SUM reaches 97.24%** on deskewed MNIST at N_PROJ=16 — the first routed architecture in the project to exceed 97%. The consumer uses random ternary projection weights; retained only for benchmark reproducibility. See `docs/LIBGLYPH.md` "Legacy random-projection consumers" section for the full list of 26 legacy tools.

### Single-table routed bucket (Axis 5, legacy — opt-in)

```bash
# Requires -DGLYPH_BUILD_LEGACY_RP=ON
./build-legacy/mnist_routed_bucket --data /path/to/mnist
```

Default run reproduces the Axis 5 measurement: **82.58% at 9.9 μs/query** (MAX_R=2, MIN_C=100).

### Running tests

```bash
ctest --test-dir build
```

Default build: **14/14 tests** pass — 7 `m4t` substrate tests (`m4t_mtfp`, `m4t_trit_ops`, `m4t_trit_reducers`, `m4t_route`, `m4t_mtfp4`, `m4t_ternary_matmul`, `m4t_trit_pack`), `glyph_wrapper` (alias surface), `glyph_libglyph` (20 unit tests covering RNG, bucket, multi-probe, resolvers), and 5 `train_*` routed-autodiff tests (`train_gradient_linear`, `train_gradient_routed`, `train_toy_convergence`, `train_toy_10class`, `train_edge_cases`).

Legacy opt-in build (`-DGLYPH_BUILD_LEGACY_RP=ON`): **16/16 tests** — adds `routed_tool_smoke` and `multi_smoke` against the legacy random-projection consumers.

## Documentation map

| File | Purpose |
|---|---|
| [`NORTH_STAR.md`](NORTH_STAR.md) | The vision. Why base-3, why routing, what the end-game is not. Re-read when base-2 gravity pulls. |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | Consolidated measurements and what they mean. Eight axes covering accuracy, speed, inspectability, cascade architecture, signature-as-address, multi-table composition, Fashion-MNIST, and CIFAR-10. |
| [`docs/LIBGLYPH.md`](docs/LIBGLYPH.md) | `libglyph` library overview — module descriptions, usage flow, how to write a new consumer. |
| [`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md) | Every parameter across every experiment. Reference for reproduction. |
| [`docs/THESIS.md`](docs/THESIS.md) | What would falsify the thesis. Current empirical state. Benchmark bed open questions. |
| [`m4t/docs/M4T_SUBSTRATE.md`](m4t/docs/M4T_SUBSTRATE.md) | The substrate specification. 18 numbered sections, traceable to conversation. |
| [`docs/REMEDIATION_PLAN.md`](docs/REMEDIATION_PLAN.md) | Red-team findings and remediation status (first-light round; later rounds tracked in CHANGELOG). |
| [`CHANGELOG.md`](CHANGELOG.md) | Notable changes since the ground-zero rebuild. |
| [`m4t/README.md`](m4t/README.md) | Substrate-layer build and surface. |
| [`train/README.md`](train/README.md) | Routed autodiff MVP — scope, primitives, findings, hyperparameters. |
| [`archive/README.md`](archive/README.md) | What's in the archive and why. |
| `journal/direct_lsh_production.md` | Direct ternary quantization: design, measurements, GSH selective scoring. |
| `journal/normalization_first_light.md` | Per-channel normalization breakthrough for CIFAR-10 and Fashion-MNIST. |
| `journal/cifar10_nproj_ceiling.md` | CIFAR-10 projection ceiling analysis — why 46% and what the gap means. |
| [`docs/DYNAMIC_NPROJ.md`](docs/DYNAMIC_NPROJ.md) | Dynamic N_PROJ exploration and per-dataset tuning. |
| [`docs/LATTICE_GEOMETRY_RESOLVER.md`](docs/LATTICE_GEOMETRY_RESOLVER.md) | Lattice geometry resolver design and integration notes. |
| `journal/fashion_mnist_*.md` | Fashion-MNIST generalization, atomics diagnosis, density-sweep experiments. |
| `journal/` | LMM-cycle research artifacts (raw → nodes → reflect → synthesize). |

## Headline results

The architecture went through several phases. Numbers below reflect the current state after the routing-native refactor; see [`docs/FINDINGS.md`](docs/FINDINGS.md) for the full axis-by-axis story.

### Production (direct quantization path)

| Dataset | Consumer | Config | Accuracy | vs SSTT | Notes |
|---|---|---|---|---|---|
| CIFAR-10 | `direct_lsh` | d=0.395, gradients, normalize, M=64, selective | **46.63%** | ~53% | First CIFAR-10 result; gap is projection ceiling (see `journal/cifar10_nproj_ceiling.md`) |
| Fashion-MNIST | `direct_lsh` | d=0.395, gradients, normalize, M=64, selective | **87.95%** | 86.54% | **Glyph wins** — +1.41 points over SSTT |
| MNIST | `direct_lsh` | d=0.10, M=64, selective | **97.18%** | 97.53% | Tied (within noise); 97.23% without gradients |

SSTT = Self-Supervised Ternary Transformers (published baseline for ternary-native classification).

### Legacy (random projection path, Axis 5 / 6)

| Consumer | Config | Accuracy | ms/query | Architecture |
|---|---|---|---|---|
| `mnist_routed_bucket` | N_PROJ=16, MAX_R=2, MIN_C=100 | **82.58%** | ~0.01 | Single-table bucket-indexed LSH (signature-as-address). First genuinely routed consumer. |
| `mnist_routed_bucket_multi` | N_PROJ=16, M=16, SUM | **96.13%** | ~0.67 | 16 independent bucket tables + union-merge + summed-distance resolver. |
| `mnist_routed_bucket_multi` | **N_PROJ=16, M=32, SUM** | **97.24%** | ~1.92 | Target crossing. First routed architecture to exceed 97%. |
| `mnist_routed_bucket_multi` | N_PROJ=16, M=64, SUM | **97.31%** | ~4.13 | Diminishing returns regime. |

Multi-table routed bucket at M=32 (512 total signature trits) matches or slightly beats the pure-signature scaling curve at equivalent total bits (pure N_PROJ=512 is 97.06%; M=32 SUM is +0.18 points). Wall-time cost is ~2× faster than an equivalent dense N_PROJ=512 scan. Zero dense scans anywhere in the pipeline.

### Fashion-MNIST legacy (random projection path, no deskew)

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
