# RED-TEAM: shift3 NEON cycle

Cold-eye review of `journal/shift3_neon_closeout.md` and the productionized state. The cycle's 8 gates passed; this red-team examines whether the gates *actually proved what they claimed* and whether there are issues that the gate design didn't catch.

## Critical findings (cycle's verdict materially weaker than claimed)

### C1: Post-G6, the bit-exact test became NEON-vs-NEON (tautology)

The pre-G6 G1 verified `prototype-NEON-kernel == m4t_mtfp_shift3-scalar` across 22.08 × 10⁹ test points. Strong evidence.

**After G6 productionization, `m4t_mtfp_shift3` no longer uses scalar.** Its divide-direction path now runs the NEON code. So the test's "ref" call (`m4t_mtfp_shift3`) returns NEON output, and the test's "neon" call (`m4t_shift3_div_neon`, the prototype copy) also returns NEON output. The comparison is now between two copies of the same NEON algorithm.

If both copies are byte-identical in source, they match by construction — the test passes trivially without proving anything against the scalar oracle.

The strongest correctness gate (G1) is now structurally compromised. Any future regression introduced into BOTH copies simultaneously (e.g., a wrong magic-table value) would pass the test silently.

### C2: Post-G6, the perf comparison became NEON-vs-NEON (1.0× BATCHED)

Empirically verified by re-running the test post-G6:

```
shape=BATCHED       k= 1 n=4096 calls=200 : scalar=0.07 ns/elem  neon=0.07 ns/elem  speedup=1.0x
shape=BATCHED       k= 7 n=4096 calls=200 : scalar=0.07 ns/elem  neon=0.07 ns/elem  speedup=1.0x
shape=BATCHED       k=13 n=4096 calls=200 : scalar=0.07 ns/elem  neon=0.07 ns/elem  speedup=1.0x
shape=BATCHED       k=19 n=4096 calls=200 : scalar=0.07 ns/elem  neon=0.07 ns/elem  speedup=1.0x
```

The "scalar" measurement now calls `m4t_mtfp_shift3` which has been productionized to NEON. Both paths run the same code through different inlining contexts, hence ~1.0×.

The closeout's headline "9.5× speedup BATCHED" was a pre-G6 measurement and remains historically accurate as "prototype-NEON vs old-scalar-substrate." But it is **not** a measurement of the current production state. The CHANGELOG and CLOSEOUT present it as the production speedup, which is misleading.

### C3: LTO dead-code-eliminated the scalar fallback path

`nm`/`otool` on the post-G6 binary shows the scalar fallback (`m4t_pow3_round_div` + the scalar loop) is GONE — `sdiv` count is 0. LTO observed that `M4T_HAS_NEON` is always true at compile time, so the `#else` branch is unreachable in this build configuration.

Consequence: there is now no way to invoke the scalar reference from outside the substrate. The test cannot recover the scalar oracle for verification. Any future bit-exact test would need to either (a) reimplement the scalar logic in the test file, (b) build a separate non-NEON-flavored library, or (c) be supplied a scalar-reference API by the substrate.

This is the structural root cause of C1 and C2.

## High-severity findings (should fix)

### H1: Two copies of the NEON kernel — drift hazard

The same kernel exists in:
- `m4t/src/m4t_mtfp.c::m4t_mtfp_shift3` (production)
- `m4t/tests/test_m4t_shift3_neon_proto.c::m4t_shift3_div_neon` (prototype)

G7 fixed this for the magic table (single header). It did NOT fix it for the kernel code itself. If a future engineer tunes the production NEON code (e.g., switches to `vqdmlal_lane_s32`), the prototype copy silently drifts. The bit-exact test (C1) wouldn't catch a drift between the two copies that affected ALL k values uniformly, because the two-copies-of-NEON comparison would still match if both happen to compute the same wrong answer.

The drift hazard isn't theoretical; the H1 issue surfaced because I didn't apply G7's lesson to the kernel code.

### H2: The "9.5× production speedup" claim is supported by inference, not measurement

The closeout asserts the production substrate runs at NEON throughput post-G6. The supporting evidence is:
- G4 disasm (pre-G6 prototype shows NEON ops) — verified
- Post-G6 binary contains `smlal/sshl` ops in `m4t_mtfp_shift3` — verified now

The MISSING measurement: production-NEON vs an actual scalar-built version of the same substrate. Without that comparison, "9.5× speedup" describes the prototype's relationship to the OLD scalar; we don't have a current measurement.

To produce the correct number post-G6, we'd need either:
- A scalar-reference function callable from the test
- A second build of `libm4t` with `M4T_HAS_NEON=0` to compare against

Neither exists today.

## Medium-severity findings

### M1: Test file is named `*_proto.c` but it's a real ctest

`m4t/tests/test_m4t_shift3_neon_proto.c` should be `test_m4t_shift3_neon.c`. The "_proto" suffix records the prototype lineage but misleads readers about its current role.

### M2: Production NEON kernel is inlined in the body of `m4t_mtfp_shift3`

~30 lines of NEON code mixed into a function that also handles k=0, k≥20, k>0 directions. Hard to audit. Should be a `static` helper (`shift3_div_neon_path` or similar) called from the divide-direction branch.

### M3: G1 exhaustive verify is not in CI; magic-table regeneration could ship wrong values silently

The post-G6 ctest only runs the sample-based check (700K points). G1 (1.16e9 × 19 = 22.08e9 exhaustive) is gated behind `./test_m4t_shift3_neon x` and not invoked by ctest. If anyone re-runs `gen_pow3_magic.c` and the table values change for any reason (search algorithm tweak, etc.), the sample test would still pass, and the regression could escape into production.

CI lacks a periodic-or-manual "exhaustive" verification job.

### M4: Closeout overstates the speedup framing

> "The substrate's elemental floor primitive `shift3` no longer has a slow direction."

The divide direction is now NEON. The multiply direction is partly auto-vectorized (per G4 finding) but uses scalar 64-bit muls inside NEON shuffling — not true SIMD multiply. So "no longer has a slow direction" is overstated. The accurate framing: "the divide direction now runs at NEON throughput; the multiply direction is partly vectorized but has further headroom."

## Low-severity findings

### L1: Substrate doc tree listing not updated

`m4t/docs/M4T_SUBSTRATE.md` lists the substrate tree but doesn't include:
- `m4t/src/m4t_pow3_magic.h` (newly added)
- `m4t/tools/gen_pow3_magic.c` (newly added)
- `m4t/tests/test_m4t_shift3_neon_proto.c` (newly added)

### L2: The vqrdmulh-pivot reasoning lives only in journals

The closeout names "compound rounding" as the reason for the pivot, but the production substrate's NEON kernel comment doesn't reference this. A future engineer wondering "why not use vqrdmulhq for fewer cycles" has no signpost in the production source.

## Methodology issue surfaced by this red-team

**The cycle's gates were each well-designed individually but didn't compose into end-to-end production verification.** Specifically:

- G1 verified prototype-NEON vs (then-current) scalar-substrate.
- G6 productionized: replaced scalar-substrate with NEON.
- G8 smoke-tested production binaries (correct outputs) but did NOT re-verify the bit-exact gate against a scalar oracle.

The order-of-execution put G6 AFTER G1, so G1's verification became invalid the moment G6 ran. G8 covered consumer-level "no regression," not substrate-internal "still bit-exact." The gap is between G1 and G8 — there's no gate that verifies "post-productionization, the production NEON path matches the original scalar oracle."

**Lesson:** when productionization replaces a function's implementation, the bit-exact gate must run AFTER the replacement, against a separately-preserved oracle (not against the now-replaced function).

## Remediation plan summary (sketch — full plan in pre-commit)

1. **Expose `m4t_mtfp_shift3_scalar_ref` in the substrate API.** Always uses scalar path. Production never calls it; tests use it as the oracle.
2. **Update the test** to compare `m4t_mtfp_shift3` (NEON) vs `m4t_mtfp_shift3_scalar_ref` (scalar). Closes C1.
3. **Update the perf bench** to compare the two functions. Closes C2; produces correct post-G6 speedup number.
4. **Re-run G1 exhaustive** against the scalar reference function. Closes the C1/C3 root cause.
5. **Remove the prototype copy** of the NEON kernel from the test file (no longer needed; production IS the kernel under test). Closes H1.
6. **Re-measure G5** with the corrected comparison; update CLOSEOUT and CHANGELOG with the actual production speedup. Closes H2.
7. **Rename test file** to `test_m4t_shift3_neon.c`. Closes M1.
8. **Extract the NEON path** into a `static` helper inside m4t_mtfp.c. Closes M2.
9. **Add a CI-skipped "exhaustive verify" target** (e.g., `m4t_shift3_neon_exhaustive` ctest entry, gated by an environment variable or a separate ctest label) so it can be invoked deliberately. Closes M3.
10. **Soften closeout framing** about "no slow direction." Closes M4.
11. **Update substrate doc tree.** Closes L1.
12. **Add a code comment** in the production NEON kernel pointing to the journal for the vqrdmulh-pivot reasoning. Closes L2.

## Status

13 findings (3 critical, 2 high, 4 medium, 4 low) — none break correctness today, but C1/C2/C3 collectively mean the closeout's strongest correctness and perf claims aren't currently supported by post-G6 evidence. The substrate's behavior is correct (G8 confirmed via consumer probes) but the cycle's verification structure has rotted under productionization.

Remediation cycle should be small (most findings share root causes) and tight (8 sub-gates, executable in one session).
