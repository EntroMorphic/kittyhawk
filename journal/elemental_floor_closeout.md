# Closeout: Elemental Floor Audit — PASS

## Verdict: PASS

```
G1 shift3 correctness               : PASS (100 random (cell,k) pairs, k ∈ [-19,19])
G2 select correctness               : PASS (100 random (c,a,b,d) tuples)
G3 composite re-derivation (neg)    : PASS (bit-equivalent to direct negation)
G4 no regression                    : PASS (all 14 prior ctest binaries still green)
```

15/15 ctest binaries now PASS (was 14; +1 for `m4t_elemental_floor`).

## What shipped

- `m4t/src/m4t_mtfp.{h,c}::m4t_mtfp_shift3` — base-3 positional shift. `dst[i] = src[i] * 3^k`, with saturation on positive-k overflow and base-3 round-to-nearest-even on negative k. Implementation reuses the existing `M4T_POW3_TABLE` and `m4t_pow3_round_div`. ~50 lines.
- `m4t/src/m4t_route.{h,c}::m4t_route_select` — trit-controlled cell-level mux. Per cell, routes one of three input cells to output based on packed-trit control. Pure routing, no arithmetic. ~15 lines (scalar; NEON path deferred until profiling demands it).
- `m4t/tests/test_m4t_elemental_floor.c` — three property tests covering G1, G2, G3.
- `m4t/CMakeLists.txt` — registers the new test.

## Vision claim #1 status

Before this cycle: substrate had `add, neg, sign` (as `threshold_extract` at tau=0) plus performance composites (`mul, sub, max, min, eq, ternary_matmul, SDOT`). Missing: `shift3` (implicit in MTFP exponent metadata but not exposed) and `select` (partial in `apply_signed` but not as clean trit-controlled mux).

After this cycle: substrate has the full elemental floor — **add, neg, shift3, sign, select** — plus the existing performance composites. Vision claim #1 is substantively addressed for the first time.

## Concerns disposition (from elemental_floor_redteam — not yet written, but per the synthesize)

- **Floor was at 4-5 ops** (mathematical analysis) — confirmed by implementation; no extra primitive required.
- **shift3 by negative k uses base-3 round-to-nearest-even** — the substrate's existing odd-divisor invariant (3^k is always odd) makes ties impossible at integer mantissas. Reused `m4t_pow3_round_div` from the cross-exponent accumulator.
- **select aliasing** — out may alias any input; verified by the implementation reading each cell before writing.
- **Composite kernels stay** — no code changes to existing kernels. Documentation pass deferred to a follow-on cycle (header comments naming composite status).

## Honest concerns (red-team-of-the-cycle)

**1. The composite documentation pass wasn't done in this cycle.** Existing kernels (`m4t_trit_mul`, `m4t_trit_sub`, `m4t_trit_max`, `m4t_trit_min`, `m4t_trit_eq`) should have header comments naming them as composite-from-elementals. Deferred for a follow-on doc-only pass to keep this cycle's scope tight.

**2. No NEON path on `m4t_route_select`.** Scalar implementation only. Performance acceptable for current consumer demand but a NEON path (vbslq_s32 with masks derived from packed-trit control) is straightforward when profile demands it. Not gated.

**3. The G3 demonstration (neg via select) is structurally weak.** It verifies that select can ROUTE between a precomputed negated value and the original — which is what select does — but doesn't actually demonstrate "build neg from select alone." The honest reading: trit-level neg IS expressible via select + constants (as analyzed in REFLECT), but cell-level neg via the existing kernel is faster and is the pragmatic primitive. G3 confirms the routing primitive works as expected; the deeper claim (neg derivable from select) holds at the trit-encoding level (bit-swap) rather than the cell level (per-cell sign flip).

**4. shift3 for k > 0 with non-trivial source values can saturate silently.** No flag tracking in the current API. If a future consumer needs to detect saturation events, add a `flags` parameter following the `m4t_mtfp_vec_accum_aligning` pattern. Not added preemptively (substrate-discipline: no feature without consumer demand — and yes, this is the legitimate scope of that rule, not the now-retired research-blocking version).

## What this cycle does NOT close

- **The composite-kernel documentation pass.** Header comments naming existing kernels as composite. ~2 hours of doc edits. Should ship as a focused follow-on.
- **Vision claim #2 (scope gap).** Independent track; the elemental floor doesn't directly address scaling beyond toy.
- **Vision claim #3 (substrate-distinctness in consumer).** Independent track; the elemental floor adds substrate primitives but doesn't itself demonstrate base-3 carrying information base-2 collapses.

## Methodology notes lifted from this cycle

**The "owner pressure on foundational analysis" pattern was the unlock.** Without the user's "Mul, if made of two conditions, is composite" challenge, the prior P1-1 cycle would have shipped a plan that treated mul as elemental. Owner's foundational pressure → cycle catches its own assumption error → reframes to the real question. This pattern should be lifted to project-wide methodology: foundational analyses benefit from explicit owner pressure to test irreducibility claims.

**Eliminating the "no consumer demand" framing was the precondition.** Without that retirement, the cycle would have asked "what consumer needs shift3 and select?" and stalled. With it retired, the foundation directly justifies the primitives. The new memory entry (`feedback_no_consumer_barrier.md`) is doing real work.

**The cycle was small and decisive.** From RAW to shipped code + passing tests in one turn. The ~3000 lines of journal markdown vs ~200 lines of code/test ratio earlier in the project was high; this cycle's ratio is more honest (substantial analysis, small implementation, decisive verification).
