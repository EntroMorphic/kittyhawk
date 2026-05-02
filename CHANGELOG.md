# Changelog

Notable changes to Glyph since the 2026-05-01 ground-zero rebuild. Older entries are preserved in `01MAY26_archived/CHANGELOG.md`.

## [Unreleased]

### Added
- `01MAY26_archived/` snapshot of the prior implementation (gitignored, retained on disk as reference).
- Repository scaffolding: `LICENSE`, `README.md`, `CONTRIBUTING.md`, `NORTH_STAR.md`, `.github/` (workflows, PR template, issue templates, CODEOWNERS), top-level CMake, `docs/` skeleton (`THESIS.md`, `FINDINGS.md`).
- `docs/REMEDIATION_PLAN.md` — kernel rebuild plan covering tiers 2 (route primitives + MTFP19 mantissa arithmetic) and 3 (cross-exponent kernel + MTFP4 + ternary matmul, all consumer-gated).
- `docs/REMEDIATION_PLAN_REDTEAM.md` — adversarial review of the plan; 12 findings; substantive ones (T2, T10) reshaped tier 3 to begin with a consumer-discovery cycle rather than a design memo.
- `docs/DESIGN_X-EXPO.md` — design exploration for the cross-exponent kernel. Forward-looking specification, not a build commitment.
- `journal/xexpo_design_{raw,nodes,reflect,synthesize,closeout}.md` — first LMM cycle in the rebuild, applied to the cross-exponent kernel design and its intent. Cycle surfaced two design findings: (a) the saturation-contract error bound was too tight by a factor of 3 (`≤ 3^(e_d − 1)` should be `< 3^e_d`, with truncate-toward-zero rounding rule explicitly stated), and (b) the design's primary API should be accumulator-shaped, not pairwise — the cited consumers naturally accumulate, and `m4t_route_apply_signed` is already an accumulator at the trivial `e_running == e_new` case. The cross-exp kernel is its generalization.
- **Tier 1 kernel lift** (pure base-3 layer): `m4t_types.h`, `m4t_internal.h`, `m4t_trit_pack`, `m4t_trit_ops`, `m4t_trit_reducers` + their tests. 3/3 ctest binaries green.
- **Tier 2 hygiene pass**: lifted `m4t_mtfp.{h,c}` (MTFP19 mantissa arithmetic at one shared block exponent) and `m4t_route.{h,c}` (five route primitives). Added missing input asserts on `distance_batch` (T, sig_dim) and `apply_signed` (k, dim, per-decision tile_idx and sign). Added `m4t_route_decisions_emit_coverage` helper for §18 testability. 5/5 ctest binaries green.

### Changed
- `docs/DESIGN_X-EXPO.md` — revised per `journal/xexpo_design_closeout.md`: primary API became `m4t_mtfp_vec_accum_aligning` (stateful accumulator), pairwise `vec_add_aligning` retained as thin convenience wrapper; saturation-contract bound corrected to strict `< 3^e_result` with truncate-toward-zero rounding rule named; `Δ ≥ 19` softened from hard assertion to documented degenerate behavior; property tests refit to sequence-shaped accumulator semantics.
- `docs/REMEDIATION_PLAN.md` — cycle protocol gained explicit call-pattern measurement (pairwise-vs-accumulator) with both static-analysis and API-shape-sketch evidence sources, plus a §14.2 spec re-read prerequisite.
- `CONTRIBUTING.md` — added principle 7: substrate-level specs are upstream of kernel designs.

### Deferred
- **Tier 3** (`m4t_mtfp_vec_accum_aligning` cross-exponent accumulator, `m4t_mtfp4.*` SDOT path, `m4t_ternary_matmul.*`): pending the consumer-discovery cycle. The substrate is currently *MTFP-capable, fixed-point-in-practice*. See `docs/REMEDIATION_PLAN.md` for the cycle's pre-committed decision endpoints.

## [2026-05-01 — tier 3a: cross-exponent accumulator built]

Owner-authorized direct build, skipping the consumer-discovery cycle (codified principle 5 reading: named consumer demand suffices, not measured cost). Per principle 7, the substrate spec was re-read before implementation — that re-read changed two design choices the LMM cycle had specified.

### Added
- **`m4t_mtfp_vec_accum_aligning`** (canonical) and **`m4t_mtfp_vec_add_aligning`** (pairwise wrapper) in `m4t/src/m4t_mtfp.{h,c}`. Path A alignment, **base-3 round-to-nearest** (§8.2; the original LMM design specified truncate-toward-zero — overridden by spec re-read). Per-cell status flags with `M4T_FLAG_SATURATED` (bit 0) and `M4T_FLAG_ROUNDED` (bit 1), sticky-OR'd across calls (§14.4).
- `M4T_FLAG_SATURATED` and `M4T_FLAG_ROUNDED` macros in `m4t/src/m4t_mtfp.h`.
- `m4t/tests/test_m4t_mtfp_accum_aligning.c` — six property tests at 10,000 samples each, with a bit-exact `int64` reference implementation as the oracle. No floating-point in the test path.
- `m4t/CMakeLists.txt`: register the new test as `m4t_mtfp_accum_aligning`.

### Changed
- `m4t/docs/M4T_SUBSTRATE.md` §14.2 status: DEFERRED → IMPLEMENTED. Document table updated. §8.2 reference status updated.
- `docs/DESIGN_X-EXPO.md` revised to reflect the spec-driven changes: round-to-nearest replaces truncate, flag layout includes ROUNDED bit, parameter rename `new` → `addend` (C++ portability), bit-exact reference replaces fp decode oracle.
- `m4t/README.md` "Live surface" extended with the cross-exponent accumulator section. Tier-3 surfaces split into 3a (built) and 3b (pending). Tests table extended to 6 binaries.

### Substrate status

The substrate is now **floating-point in base 3 at per-tensor exponent granularity**. The cross-exponent kernel that distinguishes MTFP from fixed-point exists, ships under property-test coverage, and honors the spec's "named opt-in for the lossy path" framing. Per-block exponent storage (§7's stated intent) remains a separate kernel deferred until a consumer asks. Tier 3b (MTFP4 SDOT and ternary matmul) remains consumer-gated.

## [2026-05-01 — tier 3a: kernel red-team remediation]

Adversarial pass over the cross-exponent accumulator surfaced 14 findings (5 high, 5 medium, 4 low). All remediated in this commit. Recorded in `journal/xexpo_kernel_redteam.md`.

### Changed (kernel and tests)

- **Flag layout (H1):** migrated from per-cell `uint8_t[]` to per-block `uint8_t[M4T_FLAG_BYTES(n)]`, matching `M4T_SUBSTRATE.md` §14.4 verbatim. Each byte encodes 2 events × 4 cells. Added helpers `M4T_FLAG_BYTES(n)` and `m4t_flag_test(flags, cell, event)`.
- **Round-to-nearest-even invariant (H2):** added compile-time `_Static_assert` on every `M4T_POW3_*` constant verifying odd-LSB; runtime `assert(s & 1)` in `m4t_pow3_round_div`. Documents and enforces that ties cannot occur.
- **Aliasing test (H3):** renamed misleading `prop_accum_aligning_aliasing` → `prop_accum_determinism`. Added genuine aliasing test `prop_add_dst_alias_a` exercising the wrapper's `dst == a` path.
- **Wrapper assertions (M5):** added `assert(dst != b)` to `vec_add_aligning` and `vec_sub_aligning`.
- **n=0 contract (M1, kernel bug discovered):** added `if (n == 0) return;` at the top of the accumulator. Original code updated `*running_exp` even with n=0; the new `prop_accum_n_zero` test caught this.
- **Coverage hardening (M1, M3, M4):** expanded property tests from 6 to 14:
  - `prop_accum_partial_block` — trailing-block flag bits past n stay zero.
  - `prop_accum_long_sequence` — 200 sequences × K=256 calls.
  - `prop_accum_boundary` — curated edge cases (MAX_VAL, 0, Δ ∈ {0,1,19,20}, n=1).
  - `prop_accum_n_zero` — n=0 no-op contract.
  - `prop_add_dst_alias_a` — wrapper aliasing.
  - `prop_add_out_e_nullable` — wrapper accepts NULL out_e.
  - Saturation-targeted RNG (`rand_mantissa_near_max`) in `prop_accum_flags`.

### Added (new kernel)

- **`m4t_mtfp_vec_sub_aligning` (L2):** pairwise subtract wrapper, sibling of `vec_add_aligning`. Negates `b` inline within the four-case structure (no temporary buffer). Property-tested via `prop_sub_via_negation` (matches `add(a, neg(b))`) and `prop_sub_self` (`sub(x, x) == 0`).

### Changed (documentation)

- `docs/DESIGN_X-EXPO.md` — flag layout section rewritten for per-block; property-test table expanded to 14 entries; subtract wrapper documented; per-cell-deviation history noted.
- `m4t/README.md` — header clarified to distinguish NEON vs scalar kernels (L3); cross-exp section updated for per-block flags + sub wrapper; `apply_signed`-as-degenerate-case framing tightened (L4) — generalizes the *arithmetic*, not the routing semantics.
- `m4t/docs/M4T_SUBSTRATE.md` §14.2 amendment expanded with the per-block layout and odd-divisor invariant; §14.4 disambiguated to "1 byte per block."
- `journal/xexpo_spec_amend.md` (H4) — lightweight synthesize cycle documenting the §14.2 + §14.4 amendments per principle 7.
- `journal/xexpo_kernel_redteam.md` — full red-team record, all findings, all remediations.

### Substrate status

Unchanged from the prior tier-3a entry: floating-point in base 3 at per-tensor exponent granularity. The remediation hardened the implementation; the substrate's overall capability is the same. 6/6 ctest binaries green from clean rebuild under `-Werror`; 14 properties pass at full sample counts.

### Changed
- (none — ground zero state; tiers 1 and 2 are first landings.)

### Removed
- (nothing — DELETE=never; see `01MAY26_archived/`).
