# Pre-Commit: Expression Routing Remediation

Per `journal/expression_routing_redteam.md`. Each finding gets a disposition committed BEFORE any remediation code is written. This document is the discipline anchor.

## Per-finding disposition

| ID | Severity | Disposition | Verification |
|----|----------|-------------|--------------|
| C1 | Critical | FIX — subagent-blind probe set | Subagent constructs ≥15 arity-1 + ≥15 arity-2 probes with predicted target classes, knowing only bank representative names. Their predictions are run through the routing kernel. **Pre-commit gate: ≥70% match between subagent prediction and routing landed-class.** Below 70%: the routing rule is not aligned with independent mathematical intuition and the verdict is overturned. |
| C2 | Critical | FIX — subagent probes (subsumes C1) | Subagent does NOT see signature math. Their prediction method is mathematical intuition, not Hamming arithmetic. Match rate above 70% is non-tautological evidence. |
| H1 | High | FIX — multi-input-set sweep | Run probe with 4 input sets: (A) original, (B) random sample, (C) all positive, (D) coarsely spaced powers of 10. **Pre-commit gate: PASS criteria met under ≥3/4 input sets.** Failures get reported per set. |
| H2 | High | FIX — random-tree bank | Replace curated arity-1 bank with random expression trees (bounded depth). Report merger rate. **Pre-commit: no gate (this is data, not pass/fail).** Merger rate is the finding. |
| H3 | High | FIX — scale-collapse probe section | Add ~10 probes per arity that test magnitude-scaling: `2*x → x`, `100*(x+y) → x+y`, etc. **Pre-commit: expected outcome is 100% routing to the unscaled class** (because sign-only signatures must collapse magnitudes). The PASS confirms a known limit; reporting it honestly is the fix. |
| H4 | High | METHODOLOGY FIX — pre-commit ALL gates upfront | This document is the fix. All remediation gates committed BEFORE code. No gate revision after results. |
| M1 | Medium | FIX — inter-class distance diagnostics | `expr_bank_build` reports the minimum and mean inter-class signature distance. **Pre-commit: no gate (diagnostic).** If min distance is < 4 (one quarter sig_dim) for either arity, flag it. |
| M2 | Medium | FIX — multi-seed wrapper | Random-bank and random-input-set runs use seeds {0,1,2,3,4}. Report mean ± stddev of PASS rates. **Pre-commit gate: 5-seed mean PASS rate ≥ 70% with stddev ≤ 15pp.** |
| M3 | Medium | DOCUMENT for P1 | Note in remediation closeout: P1-1 (primitives floor) design must let the third state carry information beyond exact-zero (wildcard, weak/strong magnitude bands). Won't fix in P0. |
| M4 | Medium | FIX — amend closeout headline | Replace "vision claim #2 mechanism is operational" with "behavior-based equivalence-class lookup is operational at toy scale on hand-designed banks." |
| M5 | Medium | DOCUMENT for P1 | Note: cost-blindness from D5 must be addressed by P1 design. Either carry cost as metadata or restrict bank to identity-lookup. Won't fix in P0. |
| L1 | Low | FIX — `tile_to_class` indirection | Add field to `expr_bank_t`. Identity today; documents the semantic distinction. |
| L2 | Low | DEFER — no consumer demand | Substrate-discipline rule: no primitive without named consumer demand. memcmp is fine; revisit if a hot-path consumer needs `m4t_route_signature_equal`. |
| L3 | Low | SUBSUMED by M2 | Multi-seed scales probe count; per-class floor regains teeth. |
| L4 | Low | FIX — code comments | Document scale-coupling in remediation probe code. |

## Pre-committed gates summary

A remediation **PASS** requires all of:

1. **C1 gate** (subagent blind): ≥70% match between subagent predictions and routing on at least 30 probes (≥15 per arity).
2. **H1 gate** (multi-input-set): probe PASS criteria met under ≥3/4 input sets.
3. **H3 verification**: scale-collapse probes route to unscaled class as expected (100%, by construction). Documented as a known limit.
4. **M1 diagnostic**: inter-class distances reported; flags raised if min < 4.
5. **M2 gate** (multi-seed): 5-seed mean PASS rate ≥ 70% with stddev ≤ 15pp on the random-bank track.

A remediation **WEAK** is any combination not meeting all five but not FAILing any.

A remediation **FAIL** is any of:
- C1 below 50%
- H1 PASS in ≤1/4 input sets
- M2 mean PASS rate ≤ 50%

## Fixes that don't require gates

- H2 (random bank): produces a merger rate; report it.
- H4 (this doc): the methodology fix is this document existing before code.
- M3, M5 (P1 notes): documented in remediation closeout.
- M4 (closeout amendment): one-line rewrite.
- L1, L2, L4: housekeeping.

## What this remediation deliberately does NOT do

- Does not re-run the original probe (the easy and HARD gate from `expression_routing_closeout.md`). That PASS stands on its own narrow terms; this remediation tests broader claims.
- Does not change the signature derivation rule. The rule is sign-extract at tau=0; the red-team established what that rule does and doesn't capture. Changing it would be a different cycle.
- Does not implement P1-1 or P1-2.
- Does not address whether the verdict is fundamentally generalizable beyond toy scale. That requires a real-data benchmark (vision claim #2 needs eventual non-toy evidence; not in scope here).

## Order of execution

1. Write this doc (in progress).
2. Spawn subagent for blind probes (in background, parallel with code work).
3. Code framework: random expression tree generator, random input generator, inter-class distance, `tile_to_class` indirection, multi-seed wrapper.
4. Probe sections: scale-collapse, multi-input-set, random-bank.
5. Integrate subagent's blind probes when they return.
6. Build remediation binary; run all sections.
7. Verify against the pre-committed gates above.
8. Closeout doc with per-finding disposition and final verdict.
