---
title: libglyph — consumer-side routed k-NN primitives
status: As of 2026-04-15 (Axis 5/6 production library)
companion: m4t/docs/M4T_SUBSTRATE.md · docs/FINDINGS.md · docs/HYPERPARAMETERS.md
---

# libglyph

`libglyph` is the consumer-side library that sits on top of `libm4t`. The substrate (m4t) provides ternary kernels, packed-trit signatures, and `popcount_dist`. libglyph adds the higher-level infrastructure every routed-k-NN consumer needs — dataset loading, signature building, bucket indexing, ternary multi-probe enumeration, resolver variants, CLI hyperparameter parsing.

Before this library existed, every tool in `tools/` embedded its own MNIST loader, RNG, deskew, random-projection build, density calibration, and bucket index. The duplication was fine for research scaffolding but bad for production: changing a hyperparameter meant editing source in multiple places, and architectural invariants drifted between tools. The Axis 5 / Axis 6 refactor consolidated everything into `libglyph` and rewrote the production consumers as thin CLI wrappers.

**Design shape:** libglyph is a static library (`libglyph.a`) with a flat `src/glyph_*.{h,c}` layout. Every public function has a `glyph_` prefix. The library is ternary-routed end-to-end — no float, no dense scans at the application level, no hidden pixel-space fallbacks.

---

## Layered view

```
┌─────────────────────────────────────────────────────────┐
│  tools/mnist_routed_bucket_multi.c   (Axis 6 consumer)  │
│  tools/mnist_routed_bucket.c         (Axis 5 consumer)  │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  libglyph  (7 modules, ~900 lines of C)                 │
│  ─────────────────────────────────────────────────      │
│  glyph_dataset     MNIST IDX loader + deskew            │
│  glyph_rng         xoshiro128+ RNG                      │
│  glyph_sig         random ternary proj + tau calib      │
│  glyph_bucket      sorted bucket index + lower_bound    │
│  glyph_multiprobe  ternary Hamming neighbor enum        │
│  glyph_resolver    VOTE / SUM / PTM candidate scorers   │
│  glyph_config      hyperparameter struct + CLI parser   │
│                                                         │
│  glyph_*.h         thin wrapper headers aliasing m4t_*  │
│                    into the glyph namespace (legacy)    │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  libm4t  (the substrate — m4t/src/m4t_*.{h,c})          │
│  ─────────────────────────────────────────────────      │
│  m4t_mtfp          block-native MTFP19 arithmetic       │
│  m4t_ternary_matmul  ternary projection kernel          │
│  m4t_route         threshold_extract, apply_signed, ... │
│  m4t_trit_pack     pack/unpack + popcount_dist          │
│  m4t_trit_ops      element-wise TBL-based trit ops      │
│  m4t_trit_reducers masked-VCNT reductions               │
│  m4t_mtfp4         SDOT-native int8 path                │
└─────────────────────────────────────────────────────────┘
```

Every libglyph module depends only on m4t primitives and on other libglyph modules. The dependency graph is a DAG (no cycles). Consumers depend on libglyph, which transitively pulls m4t.

---

## Module reference

### `glyph_dataset` — MNIST IDX loader + deskew

**Public header:** `src/glyph_dataset.h`

**Struct:** `glyph_dataset_t` owns MTFP-encoded pixel data and integer labels for train and test splits. Allocated by `glyph_dataset_load_mnist`, freed by `glyph_dataset_free`.

**Functions:**

- `glyph_dataset_load_mnist(ds, dir)` — reads four IDX files from `<dir>`:
  `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`. Validates IDX magic numbers (0x00000803 for images, 0x00000801 for labels). Returns 0 on success, non-zero with a diagnostic to stderr on failure.
- `glyph_dataset_deskew(ds)` — applies integer-moment shear correction to every image in train and test. Zero float; uses int64 image moments. Idempotent.
- `glyph_dataset_free(ds)` — releases all heap state. Safe to call multiple times.

**Typical use:**

```c
glyph_dataset_t ds;
if (glyph_dataset_load_mnist(&ds, "/path/to/mnist") != 0) {
    /* diagnostic already printed */
    return 1;
}
glyph_dataset_deskew(&ds);
/* ... use ds.x_train, ds.y_train, ds.x_test, ds.y_test ... */
glyph_dataset_free(&ds);
```

### `glyph_rng` — xoshiro128+ RNG

**Public header:** `src/glyph_rng.h`

Small, fast, deterministic RNG used to generate random ternary projection matrices. This is the "plus" variant of Blackman and Vigna's xoshiro128 family — state update is the standard xoshiro128 step (xor + shift + rotate), output is `s[0] + s[3]`.

**IMPORTANT:** xoshiro requires at least one non-zero state element. `glyph_rng_seed` asserts this at runtime.

**Functions:**

- `glyph_rng_seed(r, a, b, c, d)` — seed the generator with a four-element state. Asserts non-zero.
- `glyph_rng_next(r)` — draw the next 32-bit value. Deterministic given the same seed quadruple.

### `glyph_sig` — ternary signatures

**Public header:** `src/glyph_sig.h`

Two signature paths:

**Direct quantization (preferred for image classification):**

- `glyph_sig_quantize(x, n_dims, tau, out_sig)` — quantize each input dimension to a trit via per-value thresholding. Each trit represents a SPECIFIC input value. Preserves spatial identity.
- `glyph_sig_quantize_batch(x_batch, n, n_dims, tau, out_sigs)` — quantize n vectors.
- `glyph_sig_quantize_tau(x_sample, n_sample, n_dims, density)` — compute τ from a calibration sample at the given density.

**Random projection (legacy, non-image domains only):**

- `glyph_sig_builder_init(...)` — generates a random ternary projection matrix. Each trit is a random mixture of ~D/3 input dimensions. **DESTROYS spatial structure. Do NOT use for image classification.** Empirically proven inferior on MNIST, Fashion-MNIST, and CIFAR-10.
- `glyph_sig_encode(sb, x, out_sig)` — encode via random projection.
- `glyph_sig_encode_batch(sb, x_batch, n, out_sigs)` — batch encode.
- `glyph_sig_builder_free(sb)` — release heap state.

**Typical use (image classification — direct quantization):**

```c
/* Normalize the dataset first. */
glyph_dataset_normalize(&ds);

/* Compute tau from the normalized training data. */
int64_t tau = glyph_sig_quantize_tau(ds.x_train, 1000,
                                      ds.input_dim, 0.60);

/* Quantize all training and test images. */
int sig_bytes = M4T_TRIT_PACKED_BYTES(ds.input_dim);
uint8_t* train_sigs = calloc((size_t)ds.n_train * sig_bytes, 1);
glyph_sig_quantize_batch(ds.x_train, ds.n_train,
                          ds.input_dim, tau, train_sigs);

/* ... use train_sigs with glyph_bucket, glyph_multiprobe,
 *     glyph_resolver as usual ... */
free(train_sigs);
```

### `glyph_bucket` — sorted bucket index

**Public header:** `src/glyph_bucket.h`

The bucket table is libglyph's production index structure for routed k-NN. Given N prototypes and their packed-trit signatures, it sorts `(sig_key, proto_idx)` pairs by key so that lookup is a binary search in O(log N) and exact-match buckets are contiguous runs.

**Current constraint:** 4-byte signatures (N_PROJ=16) only. Longer signatures require a different key type. Named as a library limitation in the header.

**Functions:**

- `glyph_sig_to_key_u32(sig)` — pack a 4-byte signature as a little-endian uint32.
- `glyph_bucket_build(bt, sigs, n_entries, sig_bytes)` — sort `(sig_to_key(sig), proto_idx)` pairs. Returns 0 on success. Rejects `sig_bytes != 4`.
- `glyph_bucket_lower_bound(bt, target)` — binary search for the first entry with `key >= target`. Returns the insertion position in `[0, n_entries]`.
- `glyph_bucket_count_distinct(bt)` — diagnostic; returns the number of distinct keys in the sorted table.
- `glyph_bucket_table_free(bt)` — release heap state.

**Lookup pattern (matching-run scan):**

```c
uint32_t query_key = glyph_sig_to_key_u32(query_sig);
int lb = glyph_bucket_lower_bound(&bt, query_key);
if (lb < bt.n_entries && bt.entries[lb].key == query_key) {
    /* Scan forward while keys still match — same-key entries form a
     * contiguous run because the table is sorted. */
    for (int i = lb; i < bt.n_entries && bt.entries[i].key == query_key; i++) {
        int proto_idx = bt.entries[i].proto_idx;
        /* ... do something with proto_idx ... */
    }
}
```

### `glyph_multiprobe` — ternary Hamming multi-probe

**Public header:** `src/glyph_multiprobe.h`

Multi-probe LSH enumerates signatures at ternary Hamming cost 0, 1, 2 around a query signature. The ternary cost function is:

| transition | cost |
|---|---|
| same trit (any state) | 0 |
| 0 ↔ ±1 | 1 |
| +1 ↔ −1 | 2 |

**Functions:**

- `glyph_read_trit(sig, j)` — read trit j from a packed signature; returns `-1`, `0`, or `+1`.
- `glyph_write_trit(sig, j, t)` — write trit j into a packed signature (in-place).
- `glyph_multiprobe_enumerate(query_sig, n_proj, sig_bytes, radius, scratch, cb, ctx)` — enumerate every neighbor at EXACTLY ternary Hamming cost equal to `radius` (0, 1, or 2), calling `cb` for each probe signature. `cb` may return non-zero to stop enumeration early.

**Radius 0** produces one probe (the query itself).

**Radius 1** produces approximately 16 × 1.67 ≈ 27 probes per query at density 0.33. At each position, if the trit is 0 the probe writes `+1` and `-1` (two cost-1 moves); if the trit is ±1 the probe writes `0` (one cost-1 move; the sign flip is cost 2).

**Radius 2** produces approximately 340 probes per query. Two subcases:
- (a) single sign-flip at a non-zero position (+1 → −1 or −1 → +1, cost 2 each)
- (b) two distinct cost-1 moves on different positions (cost 1 + 1 = 2)

**Callback contract:**

```c
static int my_probe_callback(const uint8_t* probe_sig, void* ctx) {
    my_ctx_t* c = (my_ctx_t*)ctx;
    /* ... look up probe_sig in a bucket index, collect candidates ... */
    if (c->candidates_full) return 1;   /* stop enumeration */
    return 0;                            /* continue */
}

uint8_t scratch[4];
glyph_multiprobe_enumerate(
    query_sig, n_proj=16, sig_bytes=4, radius=2,
    scratch, my_probe_callback, &my_ctx);
```

### `glyph_resolver` — candidate-set scorers

**Public header:** `src/glyph_resolver.h`

A resolver reads a candidate union (produced by multi-table bucket lookup + multi-probe) and returns a predicted class label. Six variants are provided; `SUM` dominates empirically on MNIST in every measured configuration at M ≥ 2.

**Struct:** `glyph_union_t` borrows a hit-list (array of prototype indices currently in the union), a dense vote array (indexed by prototype index, sized to `n_train`), the shared training labels, and a class cardinality. See the header for the full lifecycle contract — especially the "lazy zero" pattern for reusing the votes array across queries.

**Functions (production):**

- `glyph_resolver_vote(u)` — argmax class by summed vote weight over the union. O(n_hit) time, no distance arithmetic. Weakest resolver (saturates at ~89.77% at M=64 on MNIST).
- `glyph_resolver_sum(u, m_active, sig_bytes, train_sigs, query_sigs, mask)` — argmin candidate by `Σ_m popcount_dist(q_sig_m, cand_sig_m)` across all active tables. O(n_hit × m_active) popcount_dist calls. Best resolver at every M ≥ 2 on MNIST.
- `glyph_resolver_sum_neon4(...)` — NEON-batched SUM variant for sig_bytes=4 (N_PROJ=16). Processes 4 candidates per 16-byte vector. Bit-exact equivalent to `glyph_resolver_sum` for any mask value. Falls back to scalar SUM on non-NEON targets. 1.2-1.3× faster than scalar on M-series.
- `glyph_resolver_per_table_majority(u, m_active, sig_bytes, train_sigs, query_sigs, mask)` — per-table 1-NN within the union; majority-vote the M labels. Middle performer.

**Functions (research / falsified variants):**

- `glyph_resolver_sum_voteweighted(...)` — scores each candidate as `sum_dist / (1 + votes[c])`, folding the filter-stage vote count into the resolver ranking. Integer-scaled (×1024) to avoid float. Phase A experiment: falsified on both MNIST and Fashion-MNIST — either neutral or harmful vs scalar SUM. Per-class instrumentation from this experiment revealed the Fashion-MNIST upper-body-garment cluster concentration.
- `glyph_resolver_sum_radiusaware(u, ..., min_radius, lambda)` — scores each candidate as `sum_dist + lambda × min_radius[c]`, penalizing candidates reachable only via deep multi-probe expansion. Phase B.1 experiment: falsified with monotone degradation as λ increases, confirming that multi-probe radius is a coarsening of information already present in sum_dist.

Both research variants are wired into the tool via `--resolver_sum {voteweighted,radiusaware}` and remain as documented negative results.

**Class-cardinality cap:** `GLYPH_MAX_CLASSES = 256` (stack buffer size for class tallies). Runtime-asserted at each resolver entry point. Covers MNIST (10), Fashion-MNIST (10), EMNIST (47), CIFAR-100 (100), and smaller benchmarks. Larger cardinalities require either bumping the constant or an API change.

### `glyph_config` — CLI hyperparameter parser

**Public header:** `src/glyph_config.h`

**Struct:** `glyph_config_t` holds every hyperparameter that previously required source edits. See [`docs/HYPERPARAMETERS.md`](HYPERPARAMETERS.md) for the full flag list.

**Functions:**

- `glyph_config_defaults(cfg)` — initialize with Phase 3 defaults (N_PROJ=16, density=0.33, M_MAX=64, MAX_RADIUS=2, MIN_CANDS=50, base_seed=42,123,456,789, mode=oracle).
- `glyph_config_parse_argv(cfg, argc, argv)` — parse command-line arguments. Uses `strtol`/`strtod` with strict validation — rejects non-numeric input, out-of-range values, all-zero seeds. Returns 0 on success, 1 on usage error (diagnostic to stderr), -1 when `--help` was requested (usage printed to stdout).
- `glyph_config_print_usage(progname)` — print the usage block to stdout.

**Phase B.1/B.2 fields (resolver variants + density scheduling):**

| Field | CLI flag | Default | Role |
|---|---|---|---|
| `resolver_sum` | `--resolver_sum` | `"scalar"` | SUM resolver implementation: `scalar`, `neon4`, `voteweighted`, `radiusaware`. |
| `radius_lambda` | `--radius_lambda` | 8 | Penalty coefficient for `radiusaware`. Only consulted when `resolver_sum == "radiusaware"`. |
| `density_schedule` | `--density_schedule` | `"fixed"` | `fixed` = all tables use `--density`; `mixed` = round-robin over `--density_triple`. |
| `density_triple[3]` | `--density_triple` | `0.25,0.33,0.40` | Three densities for mixed schedule. Table m uses `density_triple[m % 3]`. |

See `journal/fashion_mnist_atomics.md` and `journal/fashion_mnist_density_sweep.md` for the experiments that motivated these fields.

---

## Writing a new consumer on libglyph

A new routed k-NN consumer follows this skeleton:

```c
#include "glyph_config.h"
#include "glyph_dataset.h"
#include "glyph_sig.h"
#include "glyph_bucket.h"
#include "glyph_multiprobe.h"
#include "glyph_resolver.h"

int main(int argc, char** argv) {
    /* 1. Parse hyperparameters from argv. */
    glyph_config_t cfg;
    int rc = glyph_config_parse_argv(&cfg, argc, argv);
    if (rc != 0) return (rc < 0) ? 0 : 1;

    /* 2. Load dataset. */
    glyph_dataset_t ds;
    if (glyph_dataset_load_mnist(&ds, cfg.data_dir) != 0) return 1;
    glyph_dataset_deskew(&ds);

    /* 3. Normalize (required for CIFAR-10; neutral on MNIST). */
    if (cfg.normalize) glyph_dataset_normalize(&ds);

    /* 4. Direct ternary quantization — each trit = one input dimension.
     * Do NOT use glyph_sig_builder_init for image classification;
     * random projections destroy spatial structure and are strictly
     * inferior on every measured image dataset. */
    int sig_bytes = M4T_TRIT_PACKED_BYTES(ds.input_dim);
    int64_t tau = glyph_sig_quantize_tau(
        ds.x_train, (ds.n_train < 1000 ? ds.n_train : 1000),
        ds.input_dim, cfg.density);
    uint8_t* train_sigs = calloc((size_t)ds.n_train * sig_bytes, 1);
    uint8_t* test_sigs  = calloc((size_t)ds.n_test  * sig_bytes, 1);
    glyph_sig_quantize_batch(ds.x_train, ds.n_train,
                              ds.input_dim, tau, train_sigs);
    glyph_sig_quantize_batch(ds.x_test, ds.n_test,
                              ds.input_dim, tau, test_sigs);

    /* 5. Build bucket index on train signatures. */
    glyph_bucket_table_t bt = {0};
    glyph_bucket_build(&bt, train_sigs, ds.n_train, sig_bytes);

    /* 6. Per-query probe state (votes + hit_list). Reused across queries. */
    uint16_t* votes    = calloc((size_t)ds.n_train, sizeof(uint16_t));
    int32_t*  hit_list = malloc((size_t)cfg.max_union * sizeof(int32_t));
    int n_hit = 0;
    uint8_t scratch[4];
    uint8_t mask[4]; memset(mask, 0xff, 4);

    /* 7. For each test query: multi-probe, score, predict. */
    int correct = 0;
    for (int s = 0; s < ds.n_test; s++) {
        /* Lazy-zero the votes touched by the previous query. */
        for (int j = 0; j < n_hit; j++) votes[hit_list[j]] = 0;
        n_hit = 0;

        /* Build candidate union via multi-probe. Your callback should
         * read bt, look up each probe, and append unique proto_idxs to
         * hit_list / increment votes[idx]. */
        /* glyph_multiprobe_enumerate(...); */

        /* Resolve. */
        glyph_union_t u = { hit_list, n_hit, votes, ds.y_train, /*n_classes=*/10 };
        int pred = glyph_resolver_sum(/* ... */);
        if (pred == ds.y_test[s]) correct++;
    }

    /* 8. Cleanup. */
    free(votes); free(hit_list);
    glyph_bucket_table_free(&bt);
    free(train_sigs); free(test_sigs);
    glyph_sig_builder_free(&sb);
    glyph_dataset_free(&ds);
    return 0;
}
```

For a complete working example see `tools/mnist_routed_bucket.c` (single-table, ~250 lines) or `tools/mnist_routed_bucket_multi.c` (multi-table, ~300 lines).

---

## Linking against libglyph

In CMake:

```cmake
add_executable(my_tool tools/my_tool.c)
target_link_libraries(my_tool PRIVATE glyph)
```

`glyph` is declared as a STATIC library in the top-level `CMakeLists.txt`. It transitively pulls `m4t`, so consumers never need to link m4t explicitly.

---

## Tests

`libglyph` has a unit test suite in `tests/test_glyph_libglyph.c` covering 20 tests across four modules:

- **glyph_rng (3 tests):** determinism, seed divergence, mod-3 uniformity.
- **glyph_bucket (6 tests):** empty build, single entry, collision runs, sig_bytes validation, lower_bound gap cases, sig_to_key endianness.
- **glyph_multiprobe (8 tests):** trit read/write roundtrip, non-neighbor preservation, radius 0/1/2 probe counts with exact expected values (e.g. 480 probes at r=2 all-zero, 451 probes at r=2 single-plus), callback early-stop.
- **glyph_resolver (4 tests):** VOTE simple majority, tie-break behavior, weighted votes, SUM 1-NN with hand-built signatures.

The multi-probe tests pin the ternary Hamming enumeration math to exact probe counts — any refactor that miscounts neighbor sets will fail loudly.

Run with `ctest --test-dir build`; the test is registered as `glyph_libglyph`.

---

## Limitations (named, not hidden)

- **Bucket key width:** 4 bytes only. The fused-filter variant needs `uint64_t` keys (8-byte concatenated signatures). Named in `src/glyph_bucket.h`. Future generalization.
- **Dataset loader:** MNIST IDX only. CIFAR-10 and other benchmarks need a new loader; the rest of libglyph is dataset-agnostic once `glyph_dataset_t` is populated with MTFP-encoded pixels and integer labels.
- **Class cardinality:** capped at 256. Covers every current benchmark; larger caps or API changes required for ImageNet-scale class cardinalities.
- **Config struct:** single flat `glyph_config_t` shared across all tools. Tool-specific flags (`--m_max`, `--single_m` are multi-table-only) are silently ignored by tools that don't use them. A per-tool config split is a future polish.
- **Resolver scoring:** six variants (VOTE, SUM, SUM-NEON4, PTM, voteweighted, radiusaware) with runtime-selected dispatch via `--resolver_sum`. Two research variants (voteweighted, radiusaware) are falsified negative results retained as infrastructure. Richer resolvers (per-table normalization, calibrated distance) would require API additions.

## Related documentation

- [`docs/FINDINGS.md`](FINDINGS.md) — empirical measurements and axes, including Axis 5 (signature-as-address) and Axis 6 (multi-table composition).
- [`docs/HYPERPARAMETERS.md`](HYPERPARAMETERS.md) — full hyperparameter reference including libglyph CLI flags.
- [`docs/THESIS.md`](THESIS.md) — what would falsify the thesis; current empirical state.
- [`m4t/docs/M4T_SUBSTRATE.md`](../m4t/docs/M4T_SUBSTRATE.md) — substrate specification (what libglyph sits on).
- `journal/routed_bucket_consumer.md` — Axis 5 architectural correction (dense outer loop → signature-as-address).
- `journal/break_97_nproj16_{raw,nodes,reflect,synthesize}.md` — LMM cycle that designed the multi-table composition.
- `journal/break_97_nproj16_phase3_results.md` — Axis 6 full measurement.
- `journal/fashion_mnist_first_light.md` — Fashion-MNIST generalization: 85.15% at M=64, resolver gap 6× wider than MNIST.
- `journal/fashion_mnist_atomics.md` — three diagnostic atoms on the upper-body-cluster failure mode + magnet audit.
- `journal/fashion_mnist_density_sweep.md` — Phase B.2 density-mixing experiment (falsified) + per-dataset density tuning (confirmed, multi-seed).
