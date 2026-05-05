# PRE-COMMIT: cross-exp accumulator routing remediation (100/100)

Per `journal/cross_exp_accum_routing_redteam.md` — 10 findings (2 critical, 3 high, 4 medium, 3 low). Locks in 8 R-G gates BEFORE execution.

## Verdict commitment

I commit to PASS verdicts on every gate below. If any FAIL during execution, the cycle stops there and the failure is recorded honestly.

## Gates

| ID | Closes | What | PASS bar |
|----|--------|------|----------|
| **R-G1** | H1 | Rewrite same-exp + flags!=NULL branch to use NEON. The cross-exp NEON pipeline computes flags via NEON compare — when delta=0, it degenerates to "no divide, just add+clamp+flags" — that's exactly what same-exp+flags needs. Refactor `accum_aligning_neon_block` to handle delta=0 OR write a same-exp+flags helper. NO scalar fallback in the production path. | Bit-exact preserved (A-G4 still passes); same-exp + flags NEON path verified by disasm |
| **R-G2** | C1 | Add constructed cross-exp saturation case to test_m4t_accum_aligning_neon. Pick a config where aligned + other > MAX_VAL, verify both clamp and SATURATED bit match. | New test case PASSes; saturation actually triggered |
| **R-G3** | C2 | Remove `m4t_mtfp_vec_accum_aligning_neon` from public API. Inline its dispatcher logic into `m4t_mtfp_vec_accum_aligning`. Update tests to call the production function (already do, just need to remove the redundant `_neon` calls in test_m4t_accum_aligning_neon). | Symbol absent from `nm`; ctest still passes |
| **R-G4** | M1 | Disasm `m4t_mtfp_vec_accum_aligning` post-cleanup; check whether `accum_aligning_neon_block` is inlined or called via `bl`. Document. | otool output captured; inlining state recorded |
| **R-G5** | L2 | Amend `journal/cross_exp_accum_routing_closeout.md`'s "all lessons applied" claim. Add a redteam-correction section documenting H1 violation and its remediation. | Closeout amended with correction note |
| **R-G6** | H2 + methodology lift | Add to CONTRIBUTING.md: (a) "REFLECT NEON-vs-scalar estimates should bound by compiler auto-vectorization" — extend the throughput-microbench-discipline checklist; (b) "Audit-time application of no-scalar rule to inherited code" — extend the post-commit doc-currency checklist. | New checklist items present; reference back to the cross-exp cycle as the lifted lesson |
| **R-G7** | regression | Run full ctest post-R-G1/R-G2/R-G3 changes. 20/20 still PASS; new cross-exp saturation case included. | 20/20 ctest PASS |
| **R-G8** | scope close | Closeout doc + CHANGELOG entry + commit + push + CI. | Done with green CI |

## Order of execution

R-G1 (H1 fix is the load-bearing one) → R-G2 (cross-exp sat case) → R-G3 (`_neon` API cleanup) → R-G4 (disasm verification) → R-G5/R-G6 (doc updates in parallel) → R-G7 (ctest re-verify) → R-G8 (closeout + commit + push).

## Risk register

- **R1 (R-G1 NEON path harder than expected):** writing a delta=0-aware NEON helper might require more code than expected. Mitigation: degenerate accum_aligning_neon_block to handle delta=0 (skip the divide; aligned = val unchanged; ROUNDED bit always 0). Same code structure, conditional on delta.
- **R2 (R-G3 reveals other consumers of `_neon`):** if any other code calls `m4t_mtfp_vec_accum_aligning_neon` directly, removing it breaks them. Mitigation: grep first, update consumers (only the test should use it).
- **R3 (R-G2 cross-exp sat construction harder than expected):** finding inputs that drive sum > MAX_VAL in cross-exp branches. Mitigation: pick aligned at boundary (val = MAX_VAL/3 - small, divisor s = 3, aligned ≈ MAX_VAL/3, other = MAX_VAL → sum = 4×MAX_VAL/3 > MAX_VAL). Compute the input val that yields the desired aligned via val = aligned * 3 + remainder.

## Out of scope

- **M2 cross-cutting audit** (block_add, block_sub, ternary_dot dispatch dead `#else`): still its own future cycle. This remediation covers only the cross-exp accumulator's own violations.
- **M4 cache-effects characterization** (n=4096 < n=64 speedup): perf optimization, not a correctness gap.
- **H3 sample-size expansion**: 1030+ configs is sufficient; expanding to 10000 is a nice-to-have, not load-bearing.
- **L3 baseline shape difference**: minor; A-G2 baseline and A-G6 perf bench have different setups; documenting the comparison gap doesn't change anything.

## Done when

R-G1 through R-G8 PASS. CLOSEOUT records per-gate verdict, the H1 fix in detail, and any methodology lifted.

## Status

Pre-committed. Beginning R-G1.
