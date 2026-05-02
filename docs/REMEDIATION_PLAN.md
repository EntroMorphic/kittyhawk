---
title: Substrate Remediation Plan — Tiers 2 & 3
date: 2026-05-01
scope: m4t kernel rebuild after the ground-zero reset
status: EXECUTED 2026-05-01 — owner authorization overrode the consumer-discovery cycle gate after tier 2; tier 3 built directly under the named-consumer-demand reading of principle 5. All tiers shipped, all kernels red-teamed (see `journal/xexpo_kernel_redteam.md`, `journal/m4t_matmul_redteam.md`). This document is preserved as the original plan; the actual narrative is in `CHANGELOG.md`.
companions: 01MAY26_archived/REVIEWED.md · docs/REMEDIATION_PLAN_REDTEAM.md · m4t/docs/M4T_SUBSTRATE.md · NORTH_STAR.md · CHANGELOG.md
---

# Remediation Plan

> **Status note (2026-05-01).** This plan was written before tier 3 began. Its body describes a consumer-discovery cycle that would gate tier 3 against measured consumer demand. **That gate was overridden** by owner direction ("the consumer wall was holding back progress"), which read principle 5 as "named consumer demand suffices, not measured." Tier 3a, 3b, and 3c all shipped under that reading, with substrate-spec re-reads honoring principle 7 and adversarial red-team passes after each kernel landed. The body below is preserved as the original plan; the **CHANGELOG** records what actually happened.

The 2026-05-01 audit (preserved as `01MAY26_archived/REVIEWED.md`) categorized the m4t kernels into three tiers by substrate trustworthiness:

- **Tier 1 — Lift verbatim.** `m4t_types.h`, `m4t_internal.h`, `m4t_trit_pack`, `m4t_trit_ops`, `m4t_trit_reducers`. Pure base-3, no MTFP entanglement, no design debt. **DONE 2026-05-01 (3/3 ctest binaries green under `-Werror`).**

- **Tier 2 — Hygiene pass + route primitives.** The five `m4t_route` primitives (`threshold_extract`, `distance_batch`, `topk_abs`, `apply_signed`, `signature_update`) plus the same-block-exponent MTFP19 mantissa arithmetic (`m4t_mtfp.{h,c}`) that `apply_signed` consumes. **DONE 2026-05-01 (5/5 ctest binaries green under `-Werror`).** Contract blocks header-resident; input asserts in place; `m4t_route_decisions_emit_coverage` helper makes §18 contract testable at the call site.

- **Tier 3 — The cross-exponent kernel and the consumer-gated MTFP surface.** `m4t_mtfp_vec_add_aligning` (named in `M4T_SUBSTRATE.md` §14.2 but not built), plus the consumer-gated returns of `m4t_mtfp4.*` (SDOT-native MTFP4 matmul + conversions) and `m4t_ternary_matmul.*` (MTFP19 × packed-ternary matmul). All three return only when a measured consumer demands them.

  The cross-exponent kernel is what makes the substrate genuinely MTFP rather than fixed-point-with-conversions. Until it lands and a consumer drives it, the substrate's "F" lives only in the cell-width conversions and the SDOT case-W output widening — operationally fixed-point with a configurable global scale.

  **Tier 3 has a structural prerequisite the original framing missed:** all three candidate consumers are currently archived. The consumer-discovery cycle cannot run on a consumer that doesn't exist. The first phase of tier 3 is a consumer-side rebuild — minimum surface needed to drive a real-data measurement.

This document plans tiers 2 and 3. Tiers 1 and 2 are done.

---

## Tier 2 — Hygiene pass

### Surface

`m4t_route.{threshold_extract, distance_batch, topk_abs, apply_signed, signature_update}` plus their MTFP19 mantissa-arithmetic dependency (`m4t_mtfp.{h,c}` — `block_add`, `block_sub`, `vec_add_inplace`, `vec_sub_inplace`, `vec_zero`, `clamp64`).

### Problem

The §18 emission-coverage contracts and the same-block-exponent assumption are documented in the substrate spec, not in the headers consumers actually read. Two specific risks:

1. **Silent two-state degradation.** A consumer constructs decisions or feeds packed buffers without honoring the input-class contract; the primitive's three-state behavior degrades to two states without raising an error. This is exactly the failure mode `threshold_extract` was carved out from `sign_extract` to prevent — and the carve-out is documented in the spec, not the header.

2. **Block-exponent drift.** A consumer mixes mantissas from different block exponents under `apply_signed`. The primitive sums the mantissas as integers; the result is meaningless unless the exponents matched. There is currently no header-resident statement of this requirement.

### Approach

1. **Promote each contract from spec-pointer to header-resident `@requires` block.** Every header gets a precondition section that says, in two or three lines, what the caller must guarantee — no more spec-§-pointers as the only documentation.

2. **Add per-primitive runtime asserts under `M4T_DEBUG`.** Active in dev builds, compiled out in release. Specifically:
   - `threshold_extract`: assert `tau >= 0`, `n >= 0`.
   - `distance_batch`: assert `T >= 0` and `sig_dim >= 0`.
   - `topk_abs`: assert `T <= M4T_ROUTE_MAX_T`, `0 <= k <= T`.
   - `apply_signed`: assert each `decisions[i].tile_idx >= -1` and `decisions[i].sign ∈ {-1, 0, +1}` and `dim >= 0`. (The upper-bound on `tile_idx` is implicit in the caller's `tile_outs` buffer; checking it would require a `T` parameter the prior signature does not have. Per red-team T4: keep the signature stable.)
   - `signature_update`: assert `T >= 1`, `H >= 1`, `D >= 1`.

3. **Add an emission-coverage helper** (`m4t_route_decisions_emit_coverage`): given a `m4t_route_decision_t[]`, return three booleans for whether `+1`, `0`, and `-1` sign states all appeared. Consumers' integration tests use this helper to demonstrate the input-class contract is honored *at the call site*. This makes coverage a positive obligation on the consumer, not an unstated assumption.

### Decision endpoints

- All five primitives have header-resident contracts AND (under M4T_DEBUG) runtime asserts AND the coverage helper exists with at least one consumer-test using it: **tier 2 done**.
- Any contract is genuinely under-specified (the spec text is ambiguous and the right behavior is unclear): **pause, open a journal cycle.**
- API change to `apply_signed` (adding `T` parameter) breaks an active consumer: address in the consumer rather than weakening the assert.

### Cost estimate

Half a day, capped at one day. Deltas are documentation + asserts + one helper; no algorithmic change. If a contract is genuinely under-specified, pause at the cap and open a journal cycle.

---

## Tier 3 — Cross-exponent kernel + consumer-gated MTFP surface

### Surface

`m4t_mtfp.*`, `m4t_mtfp4.*`, `m4t_ternary_matmul.*`, plus the new `m4t_mtfp_vec_add_aligning`.

### Problem

Every active arithmetic kernel runs at one shared block exponent. The "F" in MTFP currently lives in three places:

1. Cell-width conversions (`mtfp19↔mtfp4`) carry an explicit ×6561 / ÷6561 scale shift.
2. The SDOT matmul's case-W output widening (MTFP4·trit → MTFP19) is exact by construction.
3. The mantissa types support per-block exponent metadata — *if the consumer manages it*.

What's missing: any kernel that takes two MTFP tensors at different block exponents and produces an aligned result. Until this kernel exists and a consumer drives it, the substrate is operationally fixed-point with a configurable global scale.

### Prerequisite — consumer-side rebuild

The consumer-discovery cycle below requires a *measurable* consumer. All three candidate consumers are currently archived in `01MAY26_archived/`:

- `libglyph` (bucket index, multi-probe, resolvers) — archived.
- `libtrain` (`tlinear`, `rroute_*` autodiff primitives) — archived.
- The tools that drive `apply_signed` end-to-end — archived.

**The cycle cannot proceed until at least one consumer is back online.** The original tier-3 framing skipped this dependency; this revision makes it explicit and load-bearing.

#### Three rebuild options, ranked by lift cost

| Option | Surface | Lines | Real-data? | Notes |
|---|---|---|---|---|
| **A. Multi-table SUM resolver** | Subset of libglyph (`glyph_bucket`, `glyph_multiprobe`, `glyph_resolver`, minimal `glyph_dataset` for MNIST) plus `mnist_routed_bucket_multi` | ~1.5k | Yes (MNIST) | No SGD, no float latents, just signature-driven retrieval and integer-mantissa accumulation. Easiest path to a real-data measurement. |
| **B. Multi-tile routed accumulation tool** | A small benchmark that drives `apply_signed` with intentionally heterogeneous tile outputs | ~200 | No (synthetic) | Smallest code, but the cycle's gate ("demand must be measurable on real data") may not consider this real enough. |
| **C. Routed autodiff** | Lift `libtrain` + `trained_classifier` consumer | ~600 | Yes | Highest cost. Prior cycle measured nothing about precision loss from collapsed `block_exp`, so the question is open but expensive to answer. |

**Recommendation: Option A — multi-table SUM.** Real data, real benchmark, modest lift, instrumentable per-table. B and C stay as later options if the SUM measurement is inconclusive.

#### What "rebuilding multi-table SUM" means concretely

- Verbatim lift from `01MAY26_archived/src/`: `glyph_rng.{h,c}`, `glyph_bucket.{h,c}`, `glyph_multiprobe.{h,c}`, `glyph_probe.{h,c}`, `glyph_resolver.{h,c}`, `glyph_dataset.{h,c}` (only the MNIST loader path), `glyph_sig.{h,c}` (only the random-projection path needed by the legacy consumer).
- Verbatim lift of `01MAY26_archived/tools/mnist_routed_bucket_multi.c`. This is the tool that hit 97.24% on deskewed MNIST.
- Wire libglyph into top-level `CMakeLists.txt`. Re-enable the legacy random-projection consumer flag (`GLYPH_BUILD_LEGACY_RP`) for this one tool. **Discipline note:** random projections in libglyph land for measurement only; they do not become part of the substrate's claim. The flag stays opt-in.
- Smoke gate: regression test that reproduces the prior `97.24%` byte-for-byte at the same seed and config.

Cost: 1-2 days for the lift, including build wiring and the regression smoke test.

### Discipline check (the gate before any kernel design)

Per red-team T2: a *named* consumer is not enough; the demand must be measurable. Tier 3 begins with a **consumer-discovery cycle** running on a rebuilt consumer (per the prerequisite above), not a design memo.

Candidate consumers — each is a hypothesis until measured:

1. **Multi-table SUM resolver** where per-table distance scores are accumulated (rebuild option A). The prior `mnist_routed_bucket_multi` hit 97.24% with single-block-exponent arithmetic; the question is whether harder benchmarks (or even MNIST itself) pay a measurable cost from collapsing per-table magnitudes.

2. **Multi-tile routed accumulation** where per-tile activation magnitudes legitimately differ (rebuild option B, synthetic). `apply_signed` assumes all tile outputs share one block exponent; the question is whether real routed classifiers exhibit per-tile magnitude heterogeneity that costs precision.

3. **Routed autodiff gradient accumulation** across tiles whose scales differ (rebuild option C). The prior libtrain MVP collapsed to one `block_exp`; the question is whether gradient precision suffered measurably.

The cycle's deliverable: for the first candidate that has a rebuilt consumer (per the prerequisite), a measurement that establishes whether the same-block-exponent assumption costs anything on real data. **A candidate becomes a "named consumer" only when a measurement shows it pays a real cost.** If no candidate measures positive, tier 3 stays at "MTFP-capable, fixed-point-in-practice" and the documentation reflects that.

### Consumer-discovery cycle — measurement protocol

The cycle runs as a standard LMM cycle (`raw → nodes → reflect → synthesize`). Concretely, on the rebuilt multi-table SUM consumer:

#### Instrumentation

Add per-table distance-distribution logging to `mnist_routed_bucket_multi`. For each query:

- Log `min_dist[t]`, `max_dist[t]`, `mean_dist[t]`, `std_dist[t]` for each table `t` ∈ [0, M).
- Log the *spread ratio* across tables: `max_t(max_dist[t]) / max(min_t(max_dist[t]), 1)`. Larger ratios mean per-table magnitudes are heterogeneous.

If logging shows distances within ~2× of each other across all tables (low spread), per-table block_exp drift is small and the same-block-exponent assumption costs nothing — independent of the accuracy comparison below. If the spread is >10×, the assumption may collapse useful magnitude information; the accuracy comparison is the deciding measurement.

#### Comparison

Run the consumer in two modes on the same benchmark:

- **Mode A (current substrate):** all per-table distances accumulated at one shared `block_exp`. Saturating clamp where individual cells overflow.
- **Mode B (oracle):** identical accumulation, computed in `int64` *outside* the substrate (test-only path; the test's binary-int64 math is sanctioned because tests are not runtime kernels per `M4T_SUBSTRATE.md` §12). No saturation; full precision.

The accuracy delta `Δ = acc(Mode B) − acc(Mode A)` is the cost of the same-block-exponent assumption on this consumer. Run on MNIST first; if the prior `data/cifar10` and `data/fashion-mnist` directories are restored, run those too.

#### Pass thresholds

| Δ | Verdict |
|---|---|
| < 0.5pp | **No qualifying consumer.** Substrate stays MTFP-capable, fixed-point-in-practice. |
| 0.5–2.0pp | **Marginal consumer.** Document the cost; design memo conditional on Δ being reproducible across ≥3 seeds. |
| > 2.0pp | **Qualifying consumer.** Cross-exponent kernel earns its design. Open §14.2 review + design memo. |

Thresholds match the discipline used in prior cycles (e.g., the substrate-distance-refinement image-pipeline gate's NEGATIVE verdict at Δ = −2.65pp on Fashion-MNIST).

#### Call-pattern measurement (added 2026-05-01 per `xexpo_design_closeout.md`)

Beyond the accuracy delta, the cycle must decide whether the kernel's primary API is **pairwise** (one-shot, two distinct buffers) or **accumulator** (running buffer reused across iterations, exponent may drift). The closeout review identified this as load-bearing for the design and architecturally deeper than a signature change — the accumulator is a stateful primitive with the renormalize step embedded.

**Evidence sources (BOTH required, not either-or):**

1. **Static analysis of archived consumers.** Read `01MAY26_archived/tools/mnist_routed_bucket_multi.c` (and any other rebuild-option-A consumer that lifts) and trace every site that combines two or more MTFP19 buffers. Categorize each site:
   - **Pairwise** — one shot, two distinct input buffers, no temporal dependency, result consumed once.
   - **Accumulator** — running buffer reused across iterations, with the iteration count `k > 2` typical, and the running's exponent could legitimately drift if cross-exp arithmetic were available.

2. **API-shape sketch.** For each identified site, write the call expression under both APIs (pairwise and accumulator). The criterion: which API requires fewer working-buffer manipulations to express the consumer's natural computation? Specifically, count the lines of caller code per site under each API. Lower line count is the more natural fit.

**Verdict rule:**

| Pattern | Decision |
|---|---|
| Accumulator is the more natural reading at >50% of sites | Accumulator is the primary API; pairwise is the n=1-add convenience wrapper |
| Accumulator wins at <50% of sites BUT at any hot-path site (per profile) | Accumulator is still the primary API — hot-path naturalness dominates |
| Accumulator wins at <50% of sites AND no hot-path site is accumulator-natural | Pairwise stays as primary |
| Mixed evidence between sources 1 and 2 | Pause. Do not commit; record the disagreement in the cycle's REFLECT phase and reconsider |

**Architectural anchor.** The tier-2 primitive `m4t_route_apply_signed` is *already an accumulator*, restricted to `e_running == e_new`. The cross-exp kernel is its generalization. Sites that look like apply_signed loops are accumulator-shaped by definition. This anchor narrows the analysis: if a consumer's combine-multiple-MTFP-buffers site looks like apply_signed's loop body, it is already evidence for the accumulator.

#### Spec re-read prerequisite

Per `CONTRIBUTING.md` principle 7 (substrate-level specs are upstream of kernel designs), the cycle's RAW phase MUST include a re-read of `m4t/docs/M4T_SUBSTRATE.md` §14.2. The cycle records:

- What §14.2 actually says about cross-exp arithmetic semantics.
- Which assumptions the design (`docs/DESIGN_X-EXPO.md`) honored vs. quietly amended.
- Whether any §14.2 constraint contradicts the design as written.

Spec contradictions block the design memo until resolved (either the design changes or the spec amends through a journal cycle).

### Design (sketch — full design lives in a journal cycle, not this plan)

```c
/* dst[i] = decode(a[i], a_block_exp) + decode(b[i], b_block_exp),
 * re-encoded at result_block_exp. The smaller-exponent operand is
 * rescaled by 3^Δ before the add; the larger exponent is preserved.
 * Saturation at ±MAX_VAL is per-cell and informative (Case S).
 *
 * If 3^Δ × |smaller_operand[i]| would overflow MAX_VAL even before the
 * add, the cell saturates pre-add. With sat_flags non-NULL, the
 * corresponding cell flag is set (§14.4 saturation tracking opt-in).
 */
void m4t_mtfp_vec_add_aligning(
    m4t_mtfp_t* dst,
    int8_t* result_block_exp,           /* out */
    const m4t_mtfp_t* a, int8_t a_block_exp,
    const m4t_mtfp_t* b, int8_t b_block_exp,
    uint8_t* sat_flags,                 /* nullable */
    int n);
```

Open design questions (each answered in the cycle, not here):

- **Block-exponent storage granularity.** Per-block (one int8 per 4-cell block) or per-tensor (one int8 per call)? The cycle's instrumentation reveals which one the consumer actually needs. Per-block is the spec's intent; per-tensor is what most plausible consumers want for the MVP.
- **`block_exp` integer width.** int8 covers exponents in `[−128, 127]`. MTFP19's mantissa range corresponds to exponents up to ~9 in scientific notation; int8 is comfortable. If the consumer's distribution surfaces exponents beyond ±19, revisit.
- **Δ overflow.** If `Δ = max(a_block_exp, b_block_exp) − min(...)` is large (>19), the smaller operand effectively becomes zero post-rescale. Document as expected loss; do not raise an error.
- **Saturation strategy.** Case S (saturate, fixed output type) is the substrate's current default. Confirm Case S is right, or identify when Case W (output widening to MTFP39) earns the wider buffer.

### Test infrastructure (genuinely new)

The current m4t test suite is golden-value: hand-derived expected outputs, exact integer comparison, zero float in tests. Property-based testing for `vec_add_aligning` is a new pattern; this section specifies what the infrastructure looks like before it is built.

#### Decode oracle (test-only, binary float sanctioned per §12)

```c
/* Test-only. NOT linked into libm4t. Lives in m4t/tests/ alongside
 * test_m4t_mtfp_vec_add_aligning.c. Uses double per §12 sanction
 * (test path; not a runtime kernel; not per-query consumer code). */
static double mtfp_decode_to_double(m4t_mtfp_t mantissa, int8_t block_exp) {
    return (double)mantissa * pow(3.0, (double)block_exp);
}
```

#### "Within saturation tolerance" — precise definition

For inputs `(a, e_a, b, e_b)` and result `(d, e_d)` produced by the kernel:

```
real_sum  = decode(a, e_a) + decode(b, e_b)
substrate = decode(d, e_d)
saturated = the kernel reports any sat_flag bit set for this cell

if !saturated:
    require |real_sum − substrate| ≤ 3^(e_d − 1)         /* half-trit precision */
if saturated:
    require sign(substrate) == sign(real_sum)
    require |substrate| == M4T_MTFP_MAX_VAL × 3^e_d      /* clamped to bound */
```

Half-trit precision at the result block_exp is the tightest bound the kernel can promise without Case-W widening. The substrate-spec amendment (if any) records this formally.

#### Sample-count gate

10 000 random `(a, b, e_a, e_b)` per property, drawn from operand distributions matching the consumer's instrumentation (don't test exponents the consumer never emits). Pass the gate if **10 000 / 10 000** satisfy the tolerance. Anything less is a failed test, not a flaky test.

#### CI integration

Property tests run slower than golden-value (millisecond → tens of milliseconds per test). Five property tests at <100ms each grow the `ctest` budget by <1 second — tolerable. Run them inline with the existing suite; revisit if the count grows past ~20.

### Validation (the four properties)

1. **`prop_add_aligning_correctness`** — for 10 000 random `(a, b, e_a, e_b)` within the consumer-instrumented operand space, the result satisfies the saturation-tolerance bound above. Tightest correctness gate.
2. **`prop_add_aligning_roundtrip`** — `add_aligning(x, e, neg(x), e)` produces `dst = 0` for any `x, e`. Tightens against asymmetric saturation bugs.
3. **`prop_add_aligning_aliasing`** — `dst` aliasing `a` or `b` produces results identical (to the bit) to the non-aliased call. Each random sample run twice, compared.
4. **`prop_add_aligning_sat_flags`** — saturating inputs set the corresponding `sat_flags` bits; non-saturating inputs leave them clear. False-positive flag and false-negative flag both fail.

### Decision endpoints

- Kernel passes all four validations AND consumer-discovery cycle Δ > 2.0pp at the cycle's gate: **tier 3 done**; substrate is genuinely MTFP.
- Kernel passes validations but cycle showed Δ < 0.5pp on every measured consumer: **stays at synthetic-only.** Substrate documented as MTFP-capable, fixed-point-in-practice. Real result, not failure.
- Cycle showed 0.5pp ≤ Δ < 2.0pp: **marginal verdict.** Document the cost and the consumer; design + implement only if a 3-seed re-run confirms the delta is reproducible. Do not promote on a single noisy measurement.
- Kernel cannot pass property-based test without unbounded saturation: **open a journal cycle** on Case S vs Case W. Do not work around with hidden float math.

### Cost estimate (revised — includes the prerequisite)

| Phase | Days | Conditional? |
|---|---|---|
| Consumer-side rebuild (multi-table SUM subset of libglyph + `mnist_routed_bucket_multi` lift + 97.24% regression smoke) | 1–2 | No |
| Consumer-discovery cycle (instrumentation + measurement + journal cycle) | 2–3 | No |
| §14.2 review + design memo | 1 | Yes — Δ ≥ 0.5pp at cycle's gate |
| Kernel implementation + property-based tests + consumer integration | 2–3 | Yes — design memo lands |

**Realistic total wall-clock if everything fires: 6–9 days of focused work.** If the cycle says no qualifying consumer (Δ < 0.5pp on every measured consumer): **3–5 days**, ending with the substrate's documented status updated to "MTFP-capable, fixed-point-in-practice."

---

## Build order

1. **Tier 1 lift** — DONE 2026-05-01. 3/3 ctest binaries green.
2. **Tier 2 hygiene pass** — DONE 2026-05-01. 5/5 ctest binaries green.
3. **Tier 3a — consumer-side rebuild** (multi-table SUM subset of libglyph + `mnist_routed_bucket_multi` lift + 97.24% regression smoke). 1–2 days. Lands as one commit. Re-enables `GLYPH_BUILD_LEGACY_RP` (opt-in, measurement-only).
4. **Tier 3b — consumer-discovery cycle**. 2–3 days. Lands as a journal cycle (`raw → nodes → reflect → synthesize`). Decides whether tier 3 promotes via the Δ thresholds above.
5. **Tier 3c — §14.2 review + design memo** — *conditional on step 4 producing Δ ≥ 0.5pp*. 1 day. Lands as a substrate-spec amendment + design doc.
6. **Tier 3d — kernel implementation + property-based tests + consumer integration** — *conditional on step 5*. 2–3 days.

Each step lands as a separate commit with the discipline checklist on the PR. CI must stay green between steps (red-team T9).

#### Wall-clock summary

- Best case (no qualifying consumer): steps 1+2 (done) + 3a + 3b = **3–5 days remaining**, ending with substrate status updated to "MTFP-capable, fixed-point-in-practice."
- Full path (qualifying consumer found, kernel earns its design): **6–9 days remaining**, ending with the first FINDINGS axis: "substrate is genuinely MTFP, with [consumer] driving the cross-exponent kernel at Δ = [n]pp recovered."

## What stays "MTFP-capable, fixed-point-in-practice" looks like

If step 3 produces no qualifying consumer, the substrate ships with this documented status:

- `m4t/README.md`'s status section: "MTFP-capable substrate, fixed-point-in-practice — no consumer has yet driven the cross-exponent kernel. The mantissa types support per-block exponent metadata; the conversion routines (`mtfp19↔mtfp4`) carry explicit ×3^k scale shifts; the SDOT case-W matmul lands MTFP4·trit exactly in MTFP19. Cross-exponent arithmetic at one shared block_exp does not exist as a primitive."
- `m4t/docs/M4T_SUBSTRATE.md` §14.2 amended with the consumer-discovery cycle's findings.
- A pinned issue: "MTFP cross-exponent kernel — open until a consumer demands it."

This is the substrate's *honest current state*. Calling it complete would be a discipline violation.

## What this plan deliberately does not decide

- **Which higher-layer consumers come back online beyond the multi-table SUM rebuild required by tier 3a, and in what order.** Other libglyph surfaces, libtrain, and the broader tools tree are separate plans. The 3a rebuild is scoped narrowly to what the consumer-discovery cycle needs.
- **Whether to lift the prior libtrain MVP verbatim.** That's a tier-1-style audit on the train tree, not a kernel question. If the multi-table SUM cycle is inconclusive (Δ in the marginal band), libtrain rebuild may follow as cycle-3b.bis on rebuild option C — but only then.
- **Whether the rebuild adopts a new benchmark.** `docs/THESIS.md` open question 2; addressed there, not here.
- **MTFP9 status** (red-team T5). MTFP9 (16-bit, 9 trits) is dropped from the active substrate until a consumer asks. The type stays in `m4t_types.h`; no kernels appear until demanded.
- **Observability** (red-team T12). Aggregating `sat_flags` across a run is a future cycle. The asserts and per-cell flags are sufficient for development; production observability is its own design.

## Memory updates after each step

- After tier 2: update `m4t/README.md`'s "Live surface" section to link the contract blocks; no FINDINGS axis (housekeeping).
- After tier 3 consumer-discovery cycle: update `m4t/docs/M4T_SUBSTRATE.md` §14.2 with the cycle's findings; add a journal pointer; update memory `feedback_*.md` with whatever discipline lessons emerge.
- After tier 3 implementation (if it fires): add the first FINDINGS axis — substrate is genuinely MTFP, with a concrete consumer demonstrating it.
