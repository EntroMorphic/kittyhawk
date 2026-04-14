---
title: Remediation Plan — First-Light Red-Team Findings
opened: 2026-04-14
scope: commit 95f5bab ("First light — minimal MTFP core")
source: red-team conducted 2026-04-14 against the first-light increment
---

# Remediation Plan

Tracks findings from the red-team of the first-light rebuild increment. Each item has a severity, specific remediation, and completion criterion.

Severity: **H** (high — blocks real spec progress), **M** (medium — correctness or claim integrity), **L** (low — hygiene).

---

## H1. The rebuild didn't rebuild the shape — it rebuilt the documentation

**Finding.** The spec's load-bearing novelty is per-block exponent as sidecar metadata, with the 16-byte block as the atomic unit. First-light has no block type, no block-native primitives, no SoA tensor abstraction. `m4t_mtfp_t` is still `int32_t` operated on in cell-native vector ops.

**Remediation.**
- [x] Define block-native atomic primitives `m4t_mtfp_block_add` / `_sub` that operate on exactly `M4T_MTFP_CELLS_PER_BLOCK` cells (one NEON vector).
- [x] Rewrite `vec_add_inplace` / `_sub_inplace` as compositions: loop of block ops + scalar tail.
- [x] Add `_Static_assert(sizeof(m4t_mtfp_t) * M4T_MTFP_CELLS_PER_BLOCK == M4T_BLOCK_BYTES, ...)`.

**Complete when.** Block-native primitives exist as the substrate's atomic unit; vec ops are compound.

---

## H2. The new primitives have zero direct tests

**Finding.** `vec_zero`, `vec_add_inplace`, `vec_sub_inplace`, `clamp64` exercised only transitively through `test_m4t_route.c`. No direct tests for boundary saturation, n=0, scalar-tail, aliasing, NEON/scalar equivalence.

**Remediation.**
- [x] Create `m4t/tests/test_m4t_mtfp.c`.
- [x] Test `clamp64` at ±MAX_VAL, ±(MAX_VAL+1), 0, INT64 extremes.
- [x] Test `vec_zero` at n=0, 1, 4, 7, 16, 1024.
- [x] Test `block_add` / `_sub`: positive saturation, negative saturation, all-zero, mixed-sign, identity, aliasing.
- [x] Test `vec_add_inplace` / `_sub_inplace`: NEON-only (n=4), scalar-only (n=3), mixed (n=5, 7), empty (n=0), large (n=1024), aliasing.
- [x] Register in CMake, confirm pass.

**Complete when.** New binary passes with ≥15 distinct assertions.

---

## H3. §8.5 "widen, don't round" silently doesn't apply to fixed-output vec ops

**Finding.** `vec_add_inplace` outputs MTFP19; widening to MTFP39 would break the output type. Current code saturates. Saturation is neither widening nor rounding — the spec didn't name this case.

**Remediation.**
- [x] Edit `m4t/docs/M4T_SUBSTRATE.md` §8.5 to distinguish three cases: widen (output admits wider cell), saturate (fixed output), round (named cross-block opt-in only).
- [x] Specify saturation is *informative* (sets status flag under §14.4 opt-in), not silent.
- [x] Cross-reference §8.5 from the new block-primitive headers.

**Complete when.** Spec names saturation explicitly; code matches spec without contradiction.

---

## H4. Consumer-inference instead of consumer-choice

**Finding.** Primitives selected by grep of kept-file callers, not by naming a consumer. Same failure mode as the prior dev's drift.

**Remediation.**
- [x] Edit `docs/THESIS.md` §3 to name `tools/mnist_trit_lattice.c` as the provisional primary consumer.
- [x] Add a `consumer-demand` line to each new block-primitive header citing the consumer.
- [x] Update `feedback_working_style.md` memory: "primitive selection by grep" is a failure mode.

**Complete when.** Current consumer named; every new primitive cites consumer demand.

---

## M1. Block-geometry constants are orphaned

**Remediation.**
- [x] Use `M4T_MTFP_CELLS_PER_BLOCK` in block-primitive signatures (array size).
- [x] Use `M4T_BLOCK_BYTES` in a `_Static_assert`.
- [x] `_Static_assert(M4T_MTFP_CELLS_PER_BLOCK == 4)` in `test_m4t_mtfp.c`.

**Complete when.** No added constant unused.

---

## M2. Same-block contract is aspirational, not enforced

**Remediation.**
- [x] Block-native primitives enforce same-block by construction (signature takes exactly one block).
- [x] Vec primitives document their single-tensor contract in the header.
- [ ] *Deferred:* block-aware tensor type. Not implemented until a consumer drives it. Note in substrate §14.

---

## M3. Overstated commit claims

**Remediation.**
- [x] Fix unused-variable warning in `tools/mnist_trit_lattice.c:155`.
- [x] Add note to `feedback_working_style.md`: commit claims distinguish "builds" / "tests pass" / "measured faster."
- [ ] *Deferred:* NEON benchmark in `m4t/tools/`. Add when a consumer's performance matters.

**Complete when.** LSH tool compiles with zero warnings; future commits observe the builds/tests/measured distinction.

---

## M4. `vec_add_inplace` trust-boundary

**Remediation.**
- [x] Document in-range input precondition in the header.
- [x] `_Static_assert((int64_t)M4T_MTFP_MAX_VAL * 2 < INT32_MAX, ...)`.
- [ ] *Deferred:* debug-mode bounds-check. Not in this pass.

**Complete when.** Precondition named; compile-time assert catches config drift.

---

## L1. Aliasing (dst == a) undocumented

**Remediation.**
- [x] Document that `vec_add_inplace(dst, dst, n)` computes `dst[i] *= 2` with saturation; `_sub_inplace` → zeros.
- [x] Aliasing test case in `test_m4t_mtfp.c`.

---

## L2. `clamp64` `static inline` in header

**Resolution.** No action. `static inline` is correct for a 4-instruction function. Logged so it isn't re-raised.

---

## L3. Scalar tail and NEON use different clamp implementations

**Remediation.**
- [x] Test exercising both paths with inputs that hit the saturation boundary (n=5: NEON + scalar tail).

---

## Execution order

Doc edits first (fast, low risk), then code + tests:

1. H3 + H4 doc edits.
2. M3 warning fix.
3. Memory update.
4. H1 + M1 + M2 + M4 + L1 + L3: block primitives, vec rewrite, static asserts, header docs.
5. H2 + L1 test binary.
6. Build + test + commit.

---

## Completion

Check off items as they land. When all unchecked items (non-deferred) are done, the remediation is complete and the increment can be claimed as spec-realizing rather than legacy-compatible.
