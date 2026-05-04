# Closeout: Tier 2 Red-Team Remediation

Per `journal/tier2_remediation_precommit.md` against the 13 findings in `journal/tier2_perf_redteam.md`.

## Verdict: PASS — with major finding that the original Tier 2 verdict on T2-B was wrong

```
R-G1 select speedup        : PASS (3/3 distributions, fair comparison)
R-G2 conf-dist diagnostic   : MEASURED (branchless is 1.8-2.5x FASTER than branchy across distributions)
R-G3 T2-C path exercised    : PASS (test_accum_same_exp_flags_null PASS)
R-G4 direction-of-effect    : PASS (CONSISTENT across distributions for both candidates)
R-G5 cache-defeat verified  : INCONCLUSIVE (verification harness has design flaw; documented)
R-G6 no regression          : PASS (16/16 ctest binaries green; was 15, +1 for new T2-C test)
R-G7 gate-design red-team   : PASS (precommit doc explicitly red-teamed gates before code)
```

## Per-finding disposition (13/13)

| ID | Original concern | Disposition |
|----|------------------|-------------|
| **C1** | T2-A measurement unfair (inlined ref vs lib call) | **CLOSED.** Scalar reference moved into lib (`m4t_route_select_scalar_ref`); both versions called via lib boundary. Fair re-measurement: NEON beats scalar by **5.57× / 1.82× / 2.97×** across random / structured / sparse-zero distributions. The original "2.55×" was a lower bound — fair comparison shows NEON is genuinely much better than the artifact suggested. |
| **H1** | T2-C's new fast path (flags=NULL) untested | **CLOSED.** Added `test_accum_same_exp_flags_null` in `test_m4t_elemental_floor.c`. Exercises the new code path against int64 reference on 50 trials. PASS. |
| **H2** | (subsumed by H1) | **CLOSED.** G7 "no regression" claim now has teeth for the T2-C path. |
| **H3** | Methodology lesson should apply retroactively to T2-A | **CLOSED.** T2-A verdict updated based on R-G1 fair measurement. Original T2-A closeout's "2.55×" demoted to "lower bound under unfair comparison; true speedup 1.8–5.6× per fair re-measurement." |
| **M1** | `clock()` resolution marginal | **CLOSED.** Switched to `clock_gettime(CLOCK_MONOTONIC)`. ns resolution; sufficient for sub-µs operations. |
| **M2** | Single data distribution | **CLOSED.** Three distributions tested: random (per-trit-state-uniform), structured (alternating ternary pattern), sparse-zero (90% zeros, 10% ±1). |
| **M3** | Cache + branch-prediction make repeat-call measurements optimistic | **PARTIALLY CLOSED.** Pool of 8 distinct data arrays cycled by pseudo-random index per iteration. R-G5 verification harness intended to detect steady-state but has a design flaw (see "Honest concerns" below) — we mitigated the issue but can't verify it's fully defeated. |
| **M4** | T2-B fair re-measurement named but not committed | **CLOSED via R-G2.** Branchless variant added to lib (`m4t_route_confidence_weighted_dist_branchless`); both versions timed via fair lib-boundary calls. **Result: branchless is 1.8–2.5× FASTER than branchy across all distributions.** This OVERTURNS the original T2-B revert decision (see "Major finding" below). |
| **M5** | Pre-commit gate-design wasn't red-teamed | **CLOSED.** This remediation's pre-commit (`tier2_remediation_precommit.md`) explicitly red-teamed the gate design before code, naming Risks A–E. |
| **M6** | Reverted T2-B comment too long (13 lines) | **CLOSED.** Trimmed to 2 lines pointing at the journal. |
| **L1** | Perf harness not visible to ctest users | **CLOSED.** CMakeLists comment added explicitly noting the bench is not a ctest binary and how to run it. |
| **L2** | `test_m4t_tier2_perf.c` naming misleading | **CLOSED.** Renamed to `bench_m4t_tier2_perf.c`. |
| **L3** | Open items not prioritized | **CLOSED via this closeout.** "What stays open" section below has per-item priorities. |

**12 fully closed, 1 partial (M3) with documented residual gap.**

## Major finding from R-G2: T2-B's revert was based on bad data

The original Tier 2 cycle's measurement showed the branchless `confidence_weighted_dist` variant as 2.9× SLOWER than branchy. The cycle reverted on that basis. The post-revert red-team identified the inline-vs-lib-call asymmetry as the likely artifact source.

This remediation's fair re-measurement (both versions in lib, both called through identical boundaries):

| Distribution | Branchy (lib) | Branchless (lib) | Branchless speedup |
|--------------|---------------|-------------------|-----|
| Random       | 2.596 ms      | 1.027 ms          | **2.53× faster** |
| Structured   | 1.120 ms      | 0.438 ms          | **2.56× faster** |
| Sparse-zero  | 1.415 ms      | 0.781 ms          | **1.81× faster** |

**The branchless version is genuinely faster across all distributions, including the sparse-zero distribution that supposedly favored the branchy "early-exit" optimization.**

The substrate currently has the slower (branchy) production version because of the original bad measurement. **Per the project's substrate-discipline (use the correct, fastest verified code), the production version should be flipped to branchless.** Per the project's H4 discipline (don't shift gates after results), R-G2 was pre-committed as "diagnostic only — no PASS/FAIL gate," so the flip isn't automatic — owner must decide.

**Recommended action:** flip `m4t_route_confidence_weighted_dist`'s production implementation to the branchless version. Keep the branchy version as `_branchy_ref` for benchmarking continuity. Code change is ~10 lines. Owner authorization required because R-G2 was diagnostic, not gated.

## R-G1 update — T2-A's true speedup

Fair re-measurement of T2-A select: NEON beats scalar by **5.57× / 1.82× / 2.97×** across random / structured / sparse-zero. The previous Tier 2 closeout reported 2.55× and called it "real." Both numbers describe the same code; the gap is purely measurement methodology.

The new numbers update the T2-A verdict from "PASS at 2.55×" to "PASS at 1.8–5.6× depending on data, with random workloads showing the largest NEON win."

## Honest concerns (red-team-of-the-remediation)

**1. R-G5's cache-defeat verification doesn't actually verify what it claims.** The verification runs `measure_select` twice in succession (`t1` and `t2`), expecting the first run to be unprimed and the second to be primed. But the harness has been running other benchmarks BEFORE the verification block, so `t1` is already warmed up. The 1.00 ratio I observed is uninformative — it confirms the two consecutive runs are equally warm, not that we've defeated cache. **The pool-of-8 + pseudo-random indexing IS a real mitigation; we just can't verify how much it helped.** Real fix would require a fresh process per trial or explicit cache flushing — both expensive.

**2. The fair comparison's lib-call overhead is real but identical for both versions.** Each call has ~5-10ns of overhead. For the conf-dist measurements where the algorithmic work is also on the order of 5-30ns per call, lib-call overhead dominates the absolute timings. The RATIOS are still meaningful (both versions pay the same overhead), but the absolute "this is X ns per signature comparison" numbers are inflated. Documented but not fixed.

**3. The pre-commit's R-G7 ("gate-design red-team applied retroactively") is partially circular.** It says "this pre-commit doc must list specific risks of the gate design itself BEFORE measurement." That's met — but it's the kind of meta-rule that's easier to game than to verify. A future cycle could write a perfunctory risks-list and call R-G7 satisfied. The discipline rule is fine; the verification is weak.

**4. R-G2 changed the substrate-relevant question without committing to act.** The fair measurement clearly shows branchless wins. But per H4 (don't shift gates), I can't unilaterally flip the production code. The owner has to decide. So we have honest data sitting in the closeout pending an owner-action that the cycle itself can't take. Defensible discipline; awkward state.

**5. Three distributions tested, but they're all I designed.** Cooperative-author bias from prior cycles applies here too. A truly adversarial distribution (designed to fool branch prediction or cache, or to favor one version specifically) wasn't tested. R-G4 "direction-of-effect consistent" PASSes for these three; might fail on a fourth.

## What stays open (PRIORITIZED per L3)

| Priority | Item | Cost estimate |
|----------|------|---------------|
| **HIGH** | **Owner decision on flipping `confidence_weighted_dist` production to branchless** based on R-G2 finding (1.8–2.5× speedup). | <1 day if approved (10-line change + verify). |
| MEDIUM | Cache-defeat verification rewrite (M3 follow-on). Either fork-per-trial or explicit cache invalidation. | ~1 day. Defers until perf-measurement work matters more. |
| MEDIUM | Adversarial data distributions for perf gates (R-G4 follow-on). | ~1 day. Defers until perf-measurement matters more. |
| LOW | NEON-vector across multiple bytes for `confidence_weighted_dist` (Tier 2.5 from prior closeout). Only worth doing if branchless win is committed AND real workloads show it as a hot path. | ~3-5 days. |
| LOW | Magic-number-multiply vectorization of `m4t_pow3_round_div` (Tier 2.5 from prior closeout). Would unlock NEON paths for accum_aligning rescale branches. | ~1-2 weeks. Real numerical-methods work. |

## Substrate-discipline notes

- All correctness gates passed at every step. 16/16 ctest binaries PASS (was 15; added test_accum_same_exp_flags_null for R-G3).
- Both new lib functions (`m4t_route_select_scalar_ref`, `m4t_route_confidence_weighted_dist_branchless`) are documented as "for benchmarking, NOT for production use" in their headers. Substrate-discipline: they exist because the discipline of fair measurement requires them, not because consumers should call them.
- Production code (`m4t_route_select`, `m4t_route_confidence_weighted_dist`) is unchanged in behavior from the prior Tier 2 closeout. Production semantics are stable; this remediation only ADDED reference variants and improved the perf harness.
- Pre-commit doc honored: every gate stated before code; no post-hoc gate revision.

## Methodology lifted to project rules

**1. Pre-commit gates with measurement components must specify HOW the measurement is collected.** Both versions through identical machinery; specify call mechanics (lib-boundary, inlined, function-pointer-indirect, etc.). Cross-mechanism comparisons are unreliable.

**2. Reference variants for fair benchmarking go in the lib, not in the test file.** Benchmark harness compares two lib functions through equivalent call paths. Documented as "for benchmarking, NOT production."

**3. Cache-defeat verification needs adversarial design, not naive consecutive-runs comparison.** Fresh-process-per-trial or explicit cache-invalidation are the real options. The "consecutive runs match" check is uninformative.

**4. Diagnostic gates that surface major findings require explicit owner-action protocols.** R-G2 was diagnostic-only (per pre-commit) but produced a finding that overturns prior production code. The cycle can deliver the data; the substrate change requires owner authorization. Naming this protocol explicitly avoids future ambiguity.

## Status

CLOSED — 12/13 findings closed, 1 partial (M3) documented. Production code unchanged pending owner decision on T2-B re-flip.
