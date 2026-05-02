---
title: Cross-exponent MTFP add — design
date: 2026-05-01 (revised 2026-05-01 per xexpo_design_closeout.md; implemented same day)
status: IMPLEMENTED. The kernel ships in m4t/src/m4t_mtfp.{h,c} with property tests at 10,000 samples × 6 properties × bit-exact int64 reference. Owner authorized direct build (skipping the consumer-discovery cycle) under the codified discipline rule (principle 5 — named consumer demand, not measured). Spec re-read of `M4T_SUBSTRATE.md` §14.2 + §8.2 drove two design changes from the original synthesize phase: rounding rule is base-3 round-to-nearest (§8.2) instead of truncate-toward-zero, and the flag layout carries both SATURATED and ROUNDED bits (§14.2 + §14.4). Both changes are reflected below.
companions: REMEDIATION_PLAN.md · m4t/docs/M4T_SUBSTRATE.md (§14.2, §8.2, §14.4) · NORTH_STAR.md · journal/xexpo_design_closeout.md · m4t/src/m4t_mtfp.{h,c} · m4t/tests/test_m4t_mtfp_accum_aligning.c
---

# `m4t_mtfp_vec_accum_aligning` — design

## Premise

The substrate's missing kernel: accumulate MTFP19 contributions into a running buffer when contributions may carry **different `block_exp` values** than the running buffer.

Without this, every consumer that combines MTFP tensors of different scale must:
- Force one shared `block_exp` upstream (current state — the substrate is fixed-point-with-conversions), or
- Bypass the substrate and do the rescale in scalar/binary-float code (discipline violation), or
- Avoid the combination entirely (architectural distortion).

This kernel is the one piece between "MTFP-capable" and "genuinely floating in base 3."

## Architectural insight — `apply_signed` is the degenerate case

The tier-2 primitive `m4t_route_apply_signed` is *already an accumulator*, restricted to the trivial `e_running == e_new` case:

```c
/* Excerpt from current m4t_route.c apply_signed. */
for (int sel = 0; sel < k; sel++) {
    if (sign == 1)       m4t_mtfp_vec_add_inplace(result, tile, dim);
    else if (sign == -1) m4t_mtfp_vec_sub_inplace(result, tile, dim);
}
```

The body accumulates `k` tile contributions into `result`. Today every contribution must share the running's `block_exp`. The cross-exponent kernel designed here is what `apply_signed` *becomes* when block_exp drift is allowed across contributions — the same primitive shape, with the same-block-exp constraint dropped.

This reframes the kernel's role from "new primitive" to "generalization of a primitive that already ships." The cycle's hypothesis is therefore: **does removing the same-block-exp constraint on apply_signed-shaped accumulation pay measurable returns?**

## Two paths, only one defensible

Given a running mantissa `running[]` at exponent `e_running` and a new contribution `new[]` at exponent `e_new`, the result lives at exponent `e_result`. Two natural choices:

### Path A — `e_result = max(e_running, e_new)` (align up)

The smaller-exponent operand is rescaled by **dividing** its mantissa by `3^Δ` where `Δ = e_result − e_smaller`. The result lives at the larger scale.

- Pre-rescale overflow: impossible. Division shrinks.
- Post-add overflow: possible if both operands are near `MAX_VAL` after alignment. Saturate; flag.
- Precision behavior: smaller operand loses low-order trits to integer truncation. When `Δ` is large, the smaller operand silently zeroes — preserving the dominant magnitude.
- Dynamic range: large. Result can represent any value in MTFP19 at `e_result`.

### Path B — `e_result = min(e_running, e_new)` (align down)

The larger-exponent operand is rescaled by **multiplying** its mantissa by `3^Δ`.

- Pre-rescale overflow: very likely. `3^Δ × |larger_operand|` exceeds `MAX_VAL` for any operand near full scale at `Δ ≥ 4`. The larger operand saturates *before* the add — catastrophic, because the larger operand carries the dominant magnitude.
- Post-add overflow: possible.
- Precision behavior: preserves smaller operand's precision when result fits — but at the cost of the larger operand frequently saturating to noise.
- Dynamic range: collapsed.

### Decision: **Path A**

Path A is forced by geometry, not idiom: alignment-to-larger is the only positional-arithmetic choice that does not catastrophically saturate the larger operand. The larger operand carries the dominant magnitude; corrupting it is unrecoverable. Path B is a non-starter for any base — the substrate's choice would be the same in base-2, base-3, or any positional system.

(Historical anchor: this is the same choice IEEE-754 makes for binary float alignment. The shared answer comes from shared geometry, not from importing a base-2 idiom.)

The named cost — smaller operand vanishes when scale gap is large — is inherent. Consumers wanting to preserve both magnitudes simultaneously must request a Case-W output type (MTFP39) in a future kernel; that is *not* this kernel.

## Algorithm — accumulator-shaped primary

The accumulator's invariant: at any moment between calls, `|running[i]| ≤ MAX_VAL at running_exp` for every cell. The kernel maintains this across calls by potentially growing `running_exp` upward when contributions push the result out of range at the current exponent.

Four case branches per call:

```
Case addend_exp == running_exp:
    /* Same-block-exp add. Degenerates to vec_add_inplace semantics. */
    for i: running[i] = clamp(running[i] + addend[i])

Case addend_exp > running_exp:
    /* Grow running_exp upward. Running mantissa rescales down. */
    Δ = addend_exp - running_exp
    s = pow3(Δ)
    for i: running[i] = clamp(round_nearest(running[i], s) + addend[i])
    running_exp = addend_exp

Case addend_exp < running_exp:
    /* Addend rescales down to running_exp. running_exp unchanged. */
    Δ = running_exp - addend_exp
    s = pow3(Δ)
    for i: running[i] = clamp(running[i] + round_nearest(addend[i], s))

Degenerate (|Δ| ≥ 20):
    /* Smaller side rounds to zero by the math; pass the larger side
     * through. Mark ROUNDED for any cell where the rescaled side was
     * non-zero. */
```

Exactly one rescale happens per call (or zero in the same-block case). The rescale uses **base-3 round-to-nearest** (§8.2). Powers of 3 are odd, so the "halfway" point `s/2` is never an integer remainder; round-to-nearest is unambiguous and the worst-case truncation in mantissa units is `(s−1)/(2s) < 1/2`.

The pairwise primitive `vec_add_aligning(dst, &out_e, a, e_a, b, e_b)` is implemented as a thin wrapper that copies `a → dst`, sets `e = e_a`, calls the accumulator with `(b, e_b)`, and writes back `*out_e`. It ships only as a convenience; the accumulator is the canonical primitive.

## Saturation semantics (precise)

Per-call rescale rounding error is bounded **bit-exactly** by `(s−1)/(2s) < 1/2` mantissa units at the result exponent, with `s = 3^Δ`. In real numbers: at most `(3^e_result − 3^e_smaller) / 2 < (1/2) · 3^e_result`.

For the accumulator's per-call contract, with **base-3 round-to-nearest**:

```
For each cell i:
  let unsat = (rescaled larger operand mantissa) + (rescaled smaller operand mantissa)
              computed in int64

  if |unsat| ≤ MAX_VAL:
      running_after[i] == clamp(unsat) == unsat
      flags[i] |= ROUNDED iff this call's rescale produced a non-zero remainder
                              for cell i

  if |unsat| > MAX_VAL:
      running_after[i] ∈ {+MAX_VAL, −MAX_VAL}
      sign(running_after[i]) == sign(unsat)
      flags[i] |= SATURATED
      flags[i] |= ROUNDED if rescale rounded
```

The property test verifies this **bit-exactly** against an int64 reference implementation — no floating-point oracle, no tolerance. Any kernel deviation from the reference produces a different `running[i]`, which fails the test.

This is the **substrate's saturation contract for cross-exponent accumulation** under §8.2 and §14.2. Case S (saturate, fixed output type) per §8.5; Case W (widen output to MTFP39) is a future kernel that has not been built.

## API (as implemented in `m4t_mtfp.h`)

```c
#define M4T_FLAG_SATURATED  ((uint8_t)0x01)
#define M4T_FLAG_ROUNDED    ((uint8_t)0x02)

void m4t_mtfp_vec_accum_aligning(
    m4t_mtfp_t*       running,
    int8_t*           running_exp,    /* in-out */
    const m4t_mtfp_t* addend,
    int8_t            addend_exp,
    uint8_t*          flags,          /* nullable, length n; bits sticky-OR'd */
    int               n
);

void m4t_mtfp_vec_add_aligning(
    m4t_mtfp_t*       dst,
    int8_t*           out_e,           /* result exponent, written */
    const m4t_mtfp_t* a, int8_t        e_a,
    const m4t_mtfp_t* b, int8_t        e_b,
    uint8_t*          flags,
    int               n
);
```

The header carries the full preconditions, sticky-flag semantics, and aliasing contract. Parameter `addend` is used (not `new`) to preserve C++ portability of the `extern "C"` block.

### Why the running exponent is in-out

Unlike the pairwise design's deterministic `e_d = max(e_a, e_b)`, the accumulator's exponent is *stateful*: it grows across calls and the consumer needs to know its current value to decode the running buffer. The in-out parameter is unavoidable and load-bearing — there is no derivation rule the consumer can apply locally.

This is the *renormalize* operation embedded in the primitive at the natural boundary: when a contribution's scale exceeds the running's representable range, the running rescales itself. Consumers don't call a separate `renormalize()`; the accumulator does it.

### Why `int8_t` for exponent

`int8_t` covers exponents in `[−128, 127]`. MTFP19's full mantissa range corresponds to `3^19 ≈ 1.16e9`; the practical exponent range fits comfortably. Larger types add storage cost without representable benefit.

## Storage granularity (per-tensor, not per-block)

Two options were considered:

1. **Per-tensor** (one `int8_t` exponent for the whole `running[]` or `new[]` array — what the API specifies).
2. **Per-block** (one `int8_t` per 4-cell MTFP19 block, stored in a sidecar array).

Per-block is the substrate spec's stated intent (§7). But the consumers identified for the discovery cycle all emit *one tensor per logical scale*:

- Multi-table SUM resolver: one distance vector per table; all distances in one table share the projection scale.
- Multi-tile routed accumulation: one MTFP19 output per tile; all cells of one tile share the matmul scale.
- Routed autodiff gradients: one gradient tensor per parameter; all cells share the optimizer's per-parameter scale.

Within each logical tensor, scale is uniform. **Per-block sidecar exponents would carry zero information** for these consumers and add 25% storage overhead.

**Decision:** MVP is per-tensor. Per-block becomes a separate kernel only if the consumer-discovery cycle surfaces a tensor whose internal scale legitimately varies across blocks. The plan's open question is provisionally answered.

## Flag layout (§14.4 status array — per-block)

**One `uint8_t` per MTFP19 block** (4 cells per block). For an n-cell tensor, the flags array has `M4T_FLAG_BYTES(n) = ceil(n / 4)` bytes. Each byte encodes two events × four cells:

| Bits | Cell within block | Events |
|---|---|---|
| 0–1 | cell 0 | bit 0 SATURATED, bit 1 ROUNDED |
| 2–3 | cell 1 | bit 2 SATURATED, bit 3 ROUNDED |
| 4–5 | cell 2 | bit 4 SATURATED, bit 5 ROUNDED |
| 6–7 | cell 3 | bit 6 SATURATED, bit 7 ROUNDED |

Bits OR'd in across calls; never cleared by the kernel. Caller initializes via `memset(flags, 0, M4T_FLAG_BYTES(n))` and clears manually as needed. Pass `NULL` to disable tracking entirely.

Helper: `m4t_flag_test(flags, cell_index, M4T_FLAG_SATURATED | M4T_FLAG_ROUNDED)` returns non-zero iff the corresponding event(s) fired for that cell.

This is the §14.4 spec layout verbatim — "1-byte status array per block." An earlier draft used per-cell bytes (one `uint8_t` per cell); the red-team flagged this as a spec deviation and the per-block layout replaced it.

## Aliasing

`running` and `addend` **must not alias** in the accumulator API. The kernel may read `addend[i]` after writing `running[i]` (depending on which operand needs rescale), so aliasing is unsafe.

The pairwise wrapper enforces aliasing safety internally via the `dst = a` initialization step. If the consumer needs `dst == a`, the wrapper handles it; if `dst == b`, it does not (a separate wrapper variant could, but is not built until requested).

The property test verifies determinism by running two parallel kernel invocations with identical inputs and confirming bit-identical results (catching any nondeterministic state leak through the kernel's internal accumulator path).

## Implementation

Scalar MVP. Source: `m4t/src/m4t_mtfp.c`. ~150 lines, no NEON, no fp.

The implementation includes a static `M4T_POW3_TABLE[20]` (powers of 3 from 3^0 through 3^19) and a `m4t_pow3_round_div(M, s, &had_remainder)` helper that performs base-3 round-to-nearest. The accumulator switches on `(addend_exp − running_exp)` sign and handles four branches (same-exp, grow up, addend rescales down, degenerate edge).

The kernel runs at `O(n × 1 division-and-add)` per call. No primitive is called more than once per cell per call.

### NEON consideration (deferred)

ARM NEON has no integer-divide instruction. A vectorized version would either:

1. **Multiply by reciprocal.** Precompute `recip[Δ] ≈ 2^32 / 3^Δ`, then `aligned = (mantissa * recip) >> 32`. Standard libdivide pattern. Adds rounding subtlety the property test must verify against the scalar reference.
2. **Scalar inner loop.** Accept that the kernel runs in accumulation phases, not distance loops — no profile evidence yet that it's a hot path.

**MVP is scalar.** Vectorization is its own cycle, gated on profile evidence from a real consumer.

## Property tests (bit-exact, sequence-shaped)

Located in `m4t/tests/test_m4t_mtfp_accum_aligning.c`. The tests use a **bit-exact int64 reference implementation** of the kernel as the oracle — no fp, no tolerances. Any kernel deviation produces a different `int32` result or per-block flag byte, which fails the comparison.

After the red-team pass, the test suite expanded from 6 properties to **14**, with saturation-targeted distributions for flag-fire coverage and curated boundary cases for explicit edge-case enumeration.

| # | Property | Coverage |
|---|---|---|
| 1 | `prop_accum_correctness` | 10 000 sequences × 1–16 calls × 1–64 cells; bit-exact vs reference |
| 2 | `prop_accum_invariant` | Same shape; `|running[i]| ≤ MAX_VAL` after every call |
| 3 | `prop_accum_determinism` | Two parallel kernel invocations agree bit-exactly (renamed from "aliasing" — the original test was a determinism check) |
| 4 | `prop_accum_flags` | Saturation-targeted operands; per-block flag bytes match reference exactly |
| 5 | `prop_accum_partial_block` | Trailing-block bits past `n` stay zero |
| 6 | `prop_accum_long_sequence` | 200 sequences × 256 calls; invariant + correctness across long sequences |
| 7 | `prop_accum_boundary` | Curated cases: M ∈ {0, ±MAX_VAL}, Δ ∈ {0, 1, 19, 20}, n ∈ {1, 4} |
| 8 | `prop_accum_n_zero` | n=0 is a clean no-op (running, exponent unchanged) |
| 9 | `prop_add_via_wrapper` | Wrapper bit-identical to manual setup + accumulator |
| 10 | `prop_add_roundtrip` | x + neg(x) at same exp → 0 |
| 11 | `prop_add_dst_alias_a` | Wrapper with `dst == a` matches non-aliased path (genuine aliasing test) |
| 12 | `prop_add_out_e_nullable` | Wrapper accepts `out_e == NULL` |
| 13 | `prop_sub_via_negation` | `sub(a, b)` matches `add(a, neg(b))` at the storage layer |
| 14 | `prop_sub_self` | `sub(x, x)` at same exp → 0 |

All 14 pass on the build.

## Open questions (now post-implementation; cycle becomes a usage study)

The kernel ships. The questions the consumer-discovery cycle was going to test are now usage-study questions instead — the kernel's behavior is fixed, the question is which consumer call patterns hit which branches:

1. **What `Δ` distribution do real consumers produce?** Determines whether the kernel's accuracy promises are tight or slack on real data.
2. **Is the saturation rate non-trivial?** Determines whether `M4T_FLAG_SATURATED` is informative or dead infrastructure.
3. **Is the rounding rate non-trivial?** Same question for `M4T_FLAG_ROUNDED`.
4. **Pairwise vs accumulator usage at real call sites.** The wrapper is a convenience; if consumers always go through the wrapper, the accumulator API is over-engineered for the typical call pattern.
5. **Does any consumer want downward exponent migration?** Currently unsupported. If a consumer needs it, separate kernel.

These are study questions, not decision gates. The kernel is built.

## What this kernel does not decide

- **Per-block exponent variant.** Per-tensor only. Per-block is a separate kernel if a consumer ever asks.
- **Case-W variant.** `vec_accum_aligning_widening` with output in MTFP39 (no saturation) is a separate kernel.
- **Downward exponent migration.** Upward-only.
- **NEON vectorization.** Scalar. Vectorize on profile evidence.
- **MTFP4 / MTFP9 variants.** Tier 3 also-deferred surfaces.
- **Subtraction.** `m4t_mtfp_vec_sub_aligning` ships alongside the add wrapper. Negates the addend inline (no temporary buffer) by mirroring the four-case structure with sign flips on every read of `b`.

## Spec amendment

`m4t/docs/M4T_SUBSTRATE.md` §14.2 is updated alongside this build to:

- Replace the DEFERRED status with IMPLEMENTED, pointing to `m4t_mtfp.h` and the property-test file.
- Confirm Path A alignment (max-exponent target) as the choice.
- Confirm base-3 round-to-nearest as the rule (§8.2 default).
- Document the per-tensor exponent granularity choice.
- Cross-reference `journal/xexpo_design_*` as the LMM cycle that scoped the design.
