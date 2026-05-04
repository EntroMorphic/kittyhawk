# Synthesis: Elemental Floor Audit — Close the Floor with shift3 + select

## Architecture

The substrate's elemental floor is **5 ops + 3 constants**:

```
ops:       add, neg, shift3, sign, select
constants: -1, 0, +1
```

The substrate currently has: `add, neg, sign` (as `m4t_route_threshold_extract` at tau=0). It has `shift3` implicitly (in MTFP per-block exponent storage) but not as a runtime kernel. It has `select` partially (`m4t_route_apply_signed` does decision-controlled accumulation) but not as a clean trit-controlled mux at the cell level.

**Closing the floor means:**

1. Add `m4t_mtfp_shift3` as an explicit kernel.
2. Add `m4t_route_select` as a clean trit-controlled cell-level mux.
3. Audit existing composite kernels (`m4t_trit_mul`, `m4t_trit_sub`, `m4t_trit_max`, `m4t_trit_min`, `m4t_trit_eq`, etc.) and document them as composite in their headers — kernels stay for performance, but their composite status is named.

Vision claim #1 is then satisfied with a defensible, audited floor.

## Key Decisions

**D1: 5 ops + 3 constants is the elemental floor.** [REFLECT core insight]
add, neg, shift3, sign, select. neg is engineering-pragmatic-elemental (mathematically derivable from select + constants but kept as primitive for speed). Everything else is composite.

**D2: Composite kernels stay; documentation names them composite.** [REFLECT T4]
mul, sub, max, min, eq, ternary_matmul, SDOT all remain as performance kernels. Header comments name them as composite. No code is removed.

**D3: shift3 by negative k uses round-to-nearest-even.** [REFLECT remaining question 1]
Matches the substrate's existing rounding discipline (cross-exponent accumulator). 3^|k| is odd, so the odd-divisor lemma applies — ties are impossible at integer mantissas.

**D4: select API is width-uniform.** [REFLECT remaining question 2]
All three inputs and output are the same cell width. Polymorphism deferred unless a consumer demands it.

**D5: No renames of existing kernels.** [REFLECT remaining question 3]
`m4t_route_threshold_extract` keeps its name; documentation adds "this is the substrate's `sign` primitive when called with tau=0." Lower disruption.

## Implementation Spec

### `m4t_mtfp_shift3` — base-3 positional shift

```c
/* Multiply (or divide) cell `a` by 3^k.
 *
 * Positive k: multiply (shift up). Increments per-block exponent by k.
 *   No mantissa change.
 *   Saturates: if mantissa exceeds MAX_VAL after exponent shift, clamps.
 *
 * Negative k: divide (shift down). Decrements per-block exponent by |k|.
 *   Mantissa unchanged but represents a smaller value.
 *   When converting to a fixed exponent (e.g., before printing), rounds
 *   the mantissa via base-3 round-to-nearest-even (existing substrate
 *   discipline; 3^|k| is odd, so no halfway ties).
 *
 * k = 0: identity.
 *
 * Substrate-discipline: this is a positional operation, not arithmetic.
 * No primitives are called; the implementation manipulates the per-block
 * exponent metadata directly.
 *
 * Preconditions:
 *   |k| ≤ 19 (the MTFP19 trit count); for larger shifts, value collapses
 *   to ±MAX_VAL (positive k) or 0 (negative k).
 */
void m4t_mtfp_shift3(
    m4t_mtfp_t* dst,
    const m4t_mtfp_t* src,
    int8_t* dst_exp,
    int8_t src_exp,
    int k,
    int n_cells);
```

Implementation: ~30 lines of code (just exponent adjustment + mantissa clamp on overflow). NEON-friendly (parallel exponent updates and clamps).

### `m4t_route_select` — trit-controlled mux at cell level

```c
/* For each cell position i: if c[i] = +1 → out[i] = a[i]
 *                            if c[i] = -1 → out[i] = b[i]
 *                            if c[i] =  0 → out[i] = d[i]
 *
 * c is a packed-trit control vector of length n_cells.
 * a, b, d are arrays of n_cells MTFP cells.
 *
 * Substrate-discipline: pure routing; no arithmetic. NEON-vectorizable
 * via vbslq_s32 with masks derived from c's trit codes.
 */
void m4t_route_select(
    m4t_mtfp_t* out,
    const uint8_t* c_packed,
    const m4t_mtfp_t* a,
    const m4t_mtfp_t* b,
    const m4t_mtfp_t* d,
    int n_cells);
```

Implementation: ~50 lines including NEON path. Mask derivation per trit + bit-select per cell.

### Documentation pass on composite kernels

Add to each composite kernel's header a "Status: composite" comment with the elementals it decomposes to. Examples:

- `m4t_trit_mul`: composite from `{select, shift3, add}` via repeated trit-conditional add. Performance kernel; ~28 NEON instructions per 64 trits.
- `m4t_trit_sub`: composite from `{add, neg}`. Performance kernel; saturating semantics.
- `m4t_trit_max`/`m4t_trit_min`: composite from `{sub, sign, select}`. Performance kernel; TBL-based.
- `m4t_trit_eq`: composite from `{sub, sign}` (eq(a,b) = (sign(a-b) == 0)). Performance kernel; TBL-based.
- `m4t_mtfp_ternary_matmul_bt`, `m4t_mtfp4_sdot_matmul_bt`: composite from `{add, neg, shift3, select}` (matmul = repeated conditional-add over weights). Performance kernels; NEON-vectorized.

Just header comments — no code changes for these.

## Pre-committed gates

(Per H4 discipline rule from prior cycles.)

**G1 (shift3 correctness):** for 100 random (cell, k) pairs with k ∈ [-19, +19], `shift3(cell, k)` produces a value matching `cell * 3^k` (computed via int64 reference) within rounding tolerance for negative k.

**G2 (select correctness):** for 100 random (c, a, b, d) tuples with c packed-trit control, `select(c, a, b, d)` produces output where each cell matches the algorithmic specification (a if c=+1, b if c=-1, d if c=0).

**G3 (composite kernel re-implementation via elementals):** for at least one composite kernel (e.g., `m4t_trit_neg`), demonstrate equivalence to the elemental composition (`select` per trit). Bit-equivalence on 100 random inputs.

**G4 (no regression):** all 14 existing ctest binaries still PASS. All existing probe binaries still PASS at their original verdicts. The two new primitives don't break anything.

## Substrate-discipline notes

- The two new primitives are foundational and justified by vision claim #1's elemental-floor analysis. Per the retired "no consumer demand" rule, foundational primitives don't need a downstream consumer.
- `shift3` interacts with the substrate's MTFP per-block exponent representation. Implementation must respect §7 (block-exponent metadata) and §14.2 (rounding rules) of the substrate spec.
- `select` is pure routing; no arithmetic, no rounding, no saturation. Cleanest possible primitive.
- Existing kernels are unchanged in code; only header documentation gets composite-status annotations.

## What this synthesis does NOT do

- Does not add exp, log, sin, cos, sqrt, or any other transcendentals. Those are consumer-level constructions; if needed, they get built on top of the elemental floor.
- Does not add division. Composite from `{shift3, sub, select, sign}` via long division. If a hot path needs fast div, a kernel can be added later as a performance composite.
- Does not remove or rename any existing kernels. Documentation only.
- Does not address vision claim #2 (scope gap) or vision claim #3 (substrate-distinctness in consumer). Those are separate tracks.

## Loop-back triggers (per LMM)

- **Back to RAW** if `m4t_mtfp_shift3` or `m4t_route_select` reveal a substrate-spec issue not anticipated here (e.g., shift3 needs to interact with the cross-exponent accumulator in a non-obvious way).
- **Back to NODES** if implementing the composite re-derivation (G3) shows that a "composite" op needs an elemental we haven't named yet (e.g., a "carry-propagation" primitive at the trit level).
- **Back to REFLECT** if owner pushes back that 5 ops + 3 constants is too many or too few.
- **Run a full new cycle** if the implementation reveals that the cell-level abstraction was wrong and trit-level analysis is what's actually needed.

## Action plan

1. Write this synthesize (done).
2. Implement `m4t_mtfp_shift3` in `m4t/src/m4t_mtfp.{h,c}` with property tests in `m4t/tests/test_m4t_shift3.c`.
3. Implement `m4t_route_select` in `m4t/src/m4t_route.{h,c}` with property tests in `m4t/tests/test_m4t_route_select.c` (or extend existing test_m4t_route).
4. Run all gates G1–G4. Verify substrate spec consistency.
5. Documentation pass: add composite-status comments to existing composite kernels.
6. CHANGELOG entry. Closeout doc.

Total budget: ~1 week of focused work. Small scope, foundational result.

## What this closes

- **Vision claim #1 is substantively addressed** for the first time. The elemental floor is named, audited, and (after implementation) shipped. The substrate has the operations the foundation requires.
- **The "is exp/log primitive" question is resolved** as "no, they're composite, build them at consumer level if needed."
- **The "missing consumer" framing is gone** — this work is justified by the foundation directly.
- **The substrate's identity becomes cleaner**: a small named set of elementals + a curated set of performance composites + a documented separation between them.
