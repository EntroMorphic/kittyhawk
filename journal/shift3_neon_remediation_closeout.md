# CLOSEOUT: shift3 NEON cycle remediation (100/100)

Per `journal/shift3_neon_remediation_precommit.md`. All 12 R-G gates PASS. The 13 red-team findings (3 critical, 2 high, 4 medium, 4 low) are closed.

## Verdict: PASS — all 12 gates closed

```
R-G1  (m4t_mtfp_shift3_scalar_ref exposed)              : PASS
R-G2  (NEON path extracted to static helper)            : PASS
R-G3  (test uses scalar_ref oracle, not the productized fn) : PASS — 18/18 ctest
R-G4  (perf bench uses scalar_ref baseline)             : PASS — speedup > 1
R-G5  (exhaustive bit-exact vs scalar_ref)              : PASS — 22.08e9 test points
R-G6  (prototype kernel copy removed from test file)    : PASS — single source of truth
R-G7  (re-measured speedup; correct numbers)            : PASS — 9.6× BATCHED, 1.6× TIGHT-LOOP
R-G8  (file renamed: _proto.c → test_m4t_shift3_neon.c) : PASS — git mv preserved history
R-G9  (exhaustive verify documented + invocable)        : PASS — `./test_m4t_shift3_neon x`
R-G10 (closeout framing softened)                       : PASS — multiply direction acknowledged
R-G11 (M4T_SUBSTRATE.md tree updated)                   : PASS — m4t_pow3_magic.h, tools/, etc.
R-G12 (vqrdmulh-pivot comment in production code)       : PASS — already added during R-G2
```

## Per-finding disposition (against the 13 red-team findings)

| ID | Finding | Closed by | Outcome |
|----|---------|-----------|---------|
| **C1** | Post-G6 bit-exact test became NEON-vs-NEON tautology | R-G1 + R-G3 + R-G5 | Test now compares production NEON vs `m4t_mtfp_shift3_scalar_ref` (always-scalar). 22.08e9 exhaustive bit-exact PASS. |
| **C2** | Post-G6 perf comparison showed 1.0× BATCHED | R-G1 + R-G4 + R-G7 | Bench compares production vs scalar_ref. **Real speedup: 9.6× BATCHED, 1.6× TIGHT-LOOP.** |
| **C3** | LTO DCE'd the scalar fallback; no scalar oracle accessible | R-G1 | `m4t_mtfp_shift3_scalar_ref` is referenced by the test, so LTO can't eliminate it. Substrate now ships both NEON path and scalar reference. |
| **H1** | Two copies of NEON kernel (production + prototype) | R-G6 | Prototype copy deleted. Single NEON kernel in `m4t_mtfp.c::shift3_div_neon`. |
| **H2** | "Production speedup" claim was inferred, not measured | R-G7 | Re-measured: 9.6× BATCHED holds; 1.6× TIGHT-LOOP corrects the prior 6.3× claim (which was an inlining-asymmetry artifact). |
| **M1** | Test file named `_proto.c` despite being real ctest | R-G8 | `git mv test_m4t_shift3_neon_proto.c test_m4t_shift3_neon.c` |
| **M2** | NEON kernel inlined in `m4t_mtfp_shift3` | R-G2 | Extracted to `static void shift3_div_neon(...)` in m4t_mtfp.c. Cleaner separation. |
| **M3** | G1 exhaustive verify not in CI; magic-table regen could ship wrong values | R-G9 | Exhaustive invokable via `./test_m4t_shift3_neon x` (~25s). Documented in CMakeLists comment. |
| **M4** | Closeout's "no slow direction" framing overstated | R-G10 | Closeout updated: divide direction is NEON; multiply direction partly auto-vectorized but has further headroom. |
| **L1** | M4T_SUBSTRATE.md tree didn't list new files | R-G11 | Added m4t_pow3_magic.h, tools/gen_pow3_magic.c, test_m4t_shift3_neon.c, bench_m4t_lto.c. |
| **L2** | vqrdmulh-pivot reasoning lived only in journals | R-G12 | Added inline comment in m4t_mtfp.c::shift3_div_neon explaining the pivot + journal pointers. |

## What shipped

- `m4t/src/m4t_mtfp.h` — declared `m4t_mtfp_shift3_scalar_ref` (test-only oracle).
- `m4t/src/m4t_mtfp.c` — refactored: scalar divide loop in `static shift3_div_scalar`; NEON divide loop in `static shift3_div_neon` with extensive comment explaining the vqrdmulh-vs-vmull pivot. `m4t_mtfp_shift3` dispatches between them. New `m4t_mtfp_shift3_scalar_ref` always uses scalar.
- `m4t/tests/test_m4t_shift3_neon.c` (renamed from `_proto.c`) — uses `m4t_mtfp_shift3_scalar_ref` as the oracle. Prototype kernel copy deleted. Exhaustive mode preserved (`./test_m4t_shift3_neon x`).
- `m4t/CMakeLists.txt` — updated for renamed file; comment block explains exhaustive invocation.
- `m4t/docs/M4T_SUBSTRATE.md` — tree listing updated for new files (R-G11).
- `journal/shift3_neon_closeout.md` — softened framing (R-G10), forward-pointer to this remediation.

## Re-measured speedup (R-G7 — corrected numbers)

```
Production m4t_mtfp_shift3 (NEON dispatch) vs m4t_mtfp_shift3_scalar_ref (always-scalar)
Min-of-5 sampling.

Shape A (BATCHED): n=4096 per call, 200 calls
  k= 1: scalar_ref 0.759 ns/elem  → production 0.079 ns/elem  =  9.6× speedup
  k= 7: scalar_ref 0.747 ns/elem  → production 0.078 ns/elem  =  9.6× speedup
  k=13: scalar_ref 0.748 ns/elem  → production 0.078 ns/elem  =  9.6× speedup
  k=19: scalar_ref 0.719 ns/elem  → production 0.078 ns/elem  =  9.2× speedup

Shape B (TIGHT-LOOP): n=4 per call, 200K calls
  k= 1: scalar_ref 0.797 ns/elem  → production 0.494 ns/elem  =  1.6× speedup
  k= 7: scalar_ref 0.795 ns/elem  → production 0.495 ns/elem  =  1.6× speedup
  k=13: scalar_ref 0.797 ns/elem  → production 0.495 ns/elem  =  1.6× speedup
  k=19: scalar_ref 0.790 ns/elem  → production 0.494 ns/elem  =  1.6× speedup
```

**Honest framing:** the BATCHED 9.6× is real and matches the original headline claim. The TIGHT-LOOP 6.3× from the original cycle was wrong — it was measuring the prototype's inlinable copy against the substrate-boundary call, not a fair comparison. The real TIGHT-LOOP speedup (production vs scalar_ref, both through the substrate API) is 1.6×. The function-call overhead dominates at small n.

## Methodology lifted (cycle-level lesson)

**The structural error in the original cycle:** when productionization REPLACES the function the bit-exact gate was comparing against, the gate's verification expires the moment productionization runs. The original cycle's G1 verified prototype-NEON vs scalar-substrate. G6 productionized: replaced scalar-substrate with NEON. G8 only smoke-tested at the consumer level. The bit-exact gate was structurally invalidated.

**Lesson lifted to project rules:**

> When a productionization gate replaces the implementation under test, the bit-exact verification gate must:
> (a) Run AFTER productionization, AND
> (b) Compare against a separately-preserved reference oracle that productionization does NOT replace.
>
> Pre-productionization "bit-exact" claims do not transfer to post-productionization unless one of these conditions holds.

The remediation closes the gap by exposing `m4t_mtfp_shift3_scalar_ref` as a permanent reference oracle. Future shift3 modifications can be verified against it without losing the comparison baseline.

## Honest concerns from this cycle

**1. The exhaustive verify is still gated behind a CLI argument, not a CI label.** R-G9 documented the invocation but didn't wire CI to run it periodically. If someone regenerates the magic table without manual exhaustive verify, the sample test (still ctest) might pass even with wrong constants. Mitigation: documentation + `gen_pow3_magic.c` itself does its own exhaustive verify at table-generation time, so the table can't ship wrong from the generator.

**2. The TIGHT-LOOP 1.6× number is workload-shape-bound, like the BATCHED 9.6×.** A future consumer that calls shift3 in a per-element loop (n=1) would see even less benefit than 1.6×, perhaps no benefit. Per CONTRIBUTING.md scope-match rule (workload-shape caveat), shift3 perf claims should always declare the shape. Both numbers do this in the bench output; the closeout/CHANGELOG also do.

**3. R-G2's helper extraction creates a slight tail-handling code duplication** between `shift3_div_neon` (which has its own scalar tail loop) and `shift3_div_scalar`. They handle different paths; deduplicating would require restructuring. Acceptable as-is for prototype-level engineering; could revisit if either changes.

**4. m4t_mtfp_shift3_scalar_ref is in the public API but production code mustn't call it.** Convention is documented in the header comment, but there's no compile-time enforcement. A future caller could accidentally use scalar_ref where they meant the production function. Mitigation: the name is verbose and self-documenting (`_scalar_ref`).

## Status

CLOSED — 13/13 red-team findings remediated; 12/12 R-G gates PASS; 18/18 ctest.

The bit-exact verification gate is now structurally sound: production NEON path is verified bit-exact against an always-scalar reference (`m4t_mtfp_shift3_scalar_ref`) across the full 22.08 × 10⁹-point input space. Real speedup (re-measured): 9.6× BATCHED, 1.6× TIGHT-LOOP. The cycle's verification structure no longer rots under productionization.

Followups (already deferred from the original cycle): cross-exp accumulator per-cell-varying-k variant; multiply-direction NEON optimization; vqrdmulhq-based further speedup; CI integration for magic-table drift check.
