# Nodes of Interest: Elemental Floor Audit

## Node 1: The right level of abstraction is CELL, not TRIT
At the trit level, "add" doesn't return a trit — it returns a value in {-2..2} that needs carry handling. Trit-level add is really add-with-carry-propagation, which is fundamentally a multi-trit operation. So the elemental analysis must be at the cell level.
**Why it matters:** answers about what's elemental change between trit and cell levels. We commit to cell.

## Node 2: The cell-level elemental floor is 4–5 ops
1. **add(a, b)** — cell-level addition. Atomic.
2. **neg(a)** — cell-level negation. Atomic at cell level (derivable at trit level from select).
3. **shift3(a, k)** — multiply by 3^k. Positional shift; atomic at cell level (the substrate's per-block exponent makes this conceptually free, but not exposed as a runtime op).
4. **sign(a)** — read sign of cell as a trit. Atomic; the substrate's `m4t_route_threshold_extract` at tau=0 is essentially this.
5. **select(c, a, b, d)** — route one of three values based on control trit c. Atomic; the conditional/branching primitive.
**Why it matters:** this is the actual answer to vision claim #1. Everything else is composite.

## Node 3: select is genuinely irreducible (the algebraic attempt is circular)
Algebraic attempt: `select(c, a, b, d) = (a + b)*c²/2 + (a - b)*c/2 + d*(1 - c²)`. Needs mul and div. Mul and div are composite from {add, neg, shift3, sign, select}. **Circular.** Without select, you can't build mul; without mul, you can't build select algebraically. select stands.
**Why it matters:** confirms select is in the floor.

## Node 4: neg might be derivable from select + constants
Trit-level neg is "swap +1 and -1, leave 0." Trivially expressible as `select(t, -1, +1, 0)` if -1 and +1 are available constants. Cell-level neg follows by trit-wise application + position preserved.
So neg is derivable from {select, constants}. NOT elemental in strict mathematical sense.
**Why it matters:** the floor might be 4 ops (add, shift3, sign, select) plus the constant -1 (and +1, 0). Or 5 if we keep neg as a primitive for performance.

## Node 5: sign at cell level might be composite (find-leading-nonzero)
Cell-level sign requires reading the highest-magnitude trit position. That's a "find leading nonzero" operation, which can be built by iterating over trit positions with trit-level sign-tests.
But: at trit level, sign(t) = t. The trit IS its sign. So sign-of-trit is free (no operation, just observation).
At cell level: sign(cell) = sign of the highest non-zero trit. Iteration + trit-level select.
**Why it matters:** depending on iteration tolerance, sign-at-cell-level could be considered either elemental or composite. Pragmatically, the substrate exposes it as a kernel (`m4t_route_threshold_extract`) so it's atomic at the substrate API level even if mathematically composite.

## Node 6: shift3 is not exposed as a runtime op despite being implicit
The substrate's MTFP per-block exponent is shift3 stored as metadata. Block_add updates the exponent implicitly when needed. Cross-exponent accumulator (`m4t_mtfp_vec_accum_aligning`) does the rescaling. But there's no kernel that says "take this cell, multiply by 3^k, return result."
shift3 needs to be exposed if it's elemental. The implementation is essentially "increment block exponent by k, optionally clamp/round mantissa if shifting down beyond precision."
**Why it matters:** missing primitive in the substrate. Should be added.

## Node 7: Existing composite kernels stay for performance but need documentation
The substrate has kernels for many composite ops:
- `m4t_trit_mul`, `m4t_mtfp_ternary_matmul_bt`, `m4t_mtfp4_sdot_matmul_bt` (mul: composite from {add, shift3, select})
- `m4t_mtfp_block_sub` (sub: composite from {add, neg})
- `m4t_trit_max`, `m4t_trit_min` (max/min: composite from {sub, sign, select})
- `m4t_trit_eq` (eq: composite from {sub, sign})
These are SHORTCUTS for composite ops, not elementals. They should remain in the substrate (performance) but be documented as composite.
**Why it matters:** the substrate's API surface mixes elemental and composite; the audit clarifies which is which.

## Node 8: The constant set matters
Even with the 4-5 op floor, you need primitive CONSTANTS to bootstrap. At minimum: {-1, 0, +1} (the trit values themselves). Without these as available values, you can't compute anything.
**Why it matters:** the "floor" is ops + constants, not ops alone.

## Node 9: shift3 needs precision specification
shift3(a, k):
- k > 0: a *= 3^k. May overflow for large k → saturation.
- k < 0: a /= 3^|k|. Loses precision (lowest |k| trits).
- k = 0: identity.

For k > 0 in MTFP land, just increment block exponent — no mantissa change. For k < 0, mantissa must round. Need to specify rounding rule.
**Why it matters:** shift3's API needs to be precise (k range, rounding rule, saturation behavior).

## Node 10: select API at cell level has design choices
select(c, a, b, d) where c is a single trit and a, b, d are cells:
- Returns a if c = +1, b if c = -1, d if c = 0
- Width-uniform: all of a, b, d, output are the same cell width
- No saturation (no arithmetic happens; just routing)

But what about VECTOR select where c is a vector of trits? `select_vec(c[i], a[i], b[i], d[i])` per position. That's just element-wise select. Could be a separate kernel or implicit.
**Why it matters:** the substrate currently has signature-based selection in `m4t_route_apply_signed` but not a clean "trit-controlled mux" primitive. The new primitive's API needs design.

---

## Tension Summary

- **T1 (level of abstraction):** trit vs cell. Resolved by Node 1 — cell.
- **T2 (neg in or out):** Node 4 says neg is derivable from select + constants. Tension between "5 ops, neg primitive for perf" vs "4 ops, neg derived." Engineering choice.
- **T3 (sign elemental or composite at cell level):** Node 5 says it's composite if iteration is tolerated, atomic if exposed as kernel. Pragmatic answer: keep it atomic at the substrate API.
- **T4 (composite kernels stay or go):** Node 7 says stay for performance, document as composite. No real tension; just discipline.

## Dependencies

- **D1:** shift3 needs an API spec (Node 9) before implementation.
- **D2:** select needs an API spec (Node 10) before implementation.
- **D3:** Documentation of composite kernels can proceed independently of new primitive implementation.
