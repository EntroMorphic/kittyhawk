# Red-Team: Tier 2 NEON Underuse Remediation

Adversarial pass on the Tier 2 closeout. The closeout already named some concerns honestly; this document sharpens them and adds findings the closeout missed.

Findings: 1 critical, 3 high, 6 medium, 3 low.

---

## C1 — T2-A's "2.55x speedup" measurement has the SAME unfair-comparison flaw the closeout flagged for T2-B

The closeout asserts:
> G2 timing measurement is fair because both paths go through the lib-call boundary with the same overhead.

**This is wrong by the closeout's own T2-B logic.** Look at the perf harness:

```c
static void route_select_scalar_ref(...)  { ... }   // static, IN THE TEST FILE
...
extern void m4t_route_select(...);                   // lib function, EXTERNAL
```

When the compiler compiles `test_m4t_tier2_perf.c`:
- `route_select_scalar_ref` is **inlinable** into `measure_select` (static, same TU)
- `m4t_route_select` is **not inlinable** (lives in libm4t.a, external)

This is **exactly the same asymmetry** the closeout correctly identified for T2-B. The closeout treated the two cases inconsistently: T2-B got the "measurement was unfair" disposition; T2-A got the "speedup is real" disposition. The methodology lesson the closeout itself lifted ("perf gates need to specify HOW the timing is collected") applies retroactively to T2-A.

**What this means for the verdict:** the 2.55x might be the speedup AFTER paying lib-call overhead the scalar version doesn't pay. The true algorithmic speedup of NEON-over-scalar could be larger (if call overhead is significant) or the measured speedup could be inflated (less likely, since both versions do similar I/O). We don't actually know.

**Recommended:** apply the same fair-comparison standard to T2-A. Either inline the lib version into the harness (#include the .c, mark static), or put the scalar reference in the lib and call both via lib boundary. Re-measure. The 2.55x is probably real direction-of-effect (NEON processing 4 cells per cycle vs scalar processing 1 should give >4x algorithmic gain), but the specific number isn't trustworthy.

---

## H1 — T2-C's new fast path may be UNTESTED by ctest

`m4t_mtfp_vec_accum_aligning` now has two same-exp branches:
- `flags == NULL`: calls `m4t_mtfp_vec_add_inplace` (NEW path, NEON)
- `flags != NULL`: scalar per-cell loop (ORIGINAL path)

The `test_m4t_mtfp_accum_aligning` suite (per its description in the CMakeLists comment: "14 properties × 10k random samples each, bit-exact int64 reference") was written when flags-tracking was the primary use case. **It's likely the test passes flags=non-NULL exclusively, which means the new fast path never runs in ctest.**

If true, "G5 correctness preserved" is trivially PASSing because the new code never executes during the test. The G7 "no regression" claim has the same emptiness for this code path.

**Recommended:** verify whether the test exercises the flags=NULL branch. If not, add a property test that does. Until verified, treat T2-C as "structurally landed but untested."

---

## H2 — The G6 "verifiable by code review" gate is the weakest gate in the cycle

G6: "same-exp branch demonstrably uses NEON (call goes through `m4t_mtfp_vec_add_inplace`) — verifiable by code review. No timing gate."

This is a manual-eyeball check, not a verification. Code review can miss things; static analysis can't catch behavioral equivalence; the gate doesn't exercise the path.

A small property test (build a flags=NULL accum scenario, run, compare to int64 reference) would have actually exercised the new code and caught any divergence. Less than an hour of work; would have closed H1 too.

**Recommended:** add the property test as part of H1's remediation. Convert G6 from "code-review verifies" to "test-driven correctness."

---

## H3 — The methodology lesson the closeout lifted ("perf measurements need gates specifying HOW") was the right lesson but the wrong scope

The closeout lifts:
> Future perf gates should specify identical call mechanics for both versions.

This is correct as a forward-looking rule. But the closeout doesn't apply the rule retroactively to T2-A's verdict. Per H4 from the prior remediation cycle (now project-wide rule): pre-committed gates shouldn't shift after results land. The corollary for this case: methodology lessons should apply retroactively if they invalidate prior measurements in the same cycle.

T2-A's measurement was the same shape as T2-B's. T2-B got "measurement was unfair." T2-A got "speedup is real." Those dispositions are inconsistent.

**Recommended:** the Tier 2 closeout should be amended to demote T2-A's verdict from "PASS at 2.55x measured" to "PASS direction-of-effect; specific magnitude unverified pending fair-comparison re-measurement." Same standard as T2-B.

---

## M1 — `clock()` has limited resolution that may be marginal for the fastest operations

`CLOCKS_PER_SEC` on macOS is typically 1,000,000 (1µs resolution). For 100K iterations of a function taking ~50ns each, total time is ~5ms = 5000 clock ticks. Resolution is fine for that case.

But for the conf-dist branchy version (0.517ms total over 100K iter = 5ns per call), we're at 517 clock ticks total — close to the noise floor. A 10% jitter in clock() reading is significant relative to the underlying signal.

`clock_gettime(CLOCK_MONOTONIC, ...)` gives nanosecond resolution and would be more appropriate for sub-µs measurements.

**Severity:** medium. Affects the precision of the measurement, not the direction. For T2-B's already-discredited measurement, doesn't change the conclusion. For any future fair-comparison re-measurement, switch to clock_gettime.

---

## M2 — Test data distribution is not representative of real workloads

The perf harness fills control bytes with `c[i] = (i * 0x55) ^ 0xAA`. This produces very specific bit patterns:
- i=0: 0x00 (all 4 fields = 0)
- i=1: 0xFF (all 4 fields = reserved 0b11)
- i=2: 0xAA (each field = 0b10 = trit -1)
- i=3: 0x55 (each field = 0b01 = trit +1)
- ...

This is **not random trit distribution.** Real workloads have trit distributions determined by their data — for the gesh consumer, signatures derived from class-mean expressions have specific zero-richness; for random expressions, ~33% per trit state.

Branch prediction performance depends heavily on the data pattern. The "scalar baseline" might be artificially fast (predictable repeating pattern) or slow (worst-case pattern).

**Severity:** medium. The 2.55x speedup is for THIS specific synthetic pattern; real workloads may give different ratios.

**Recommended:** any future perf re-measurement should use multiple data distributions: deterministic-random (xorshift32-seeded), all-positive, all-negative, sparse-zero, dense-zero. Report ratios across distributions.

---

## M3 — Cache effects + branch prediction make repeat-call measurements unrealistic

Each `measure_select` call uses the same arrays for all 100K iterations. After the first ~10 iterations:
- Data is hot in L1 cache (no cache pressure)
- Branch predictor has learned the exact pattern (zero mispredictions)
- The CPU is in steady-state on this exact workload

Real workloads have:
- Cache misses (different signatures per query)
- Branch mispredictions (varying trit patterns)
- Pipeline disruptions

**Severity:** medium. The measured speedup is best-case for both versions; the relative speedup might be similar in real workloads, but absolute timings don't transfer.

**Recommended:** for future measurements, randomize data per iteration (or use a pool of N random arrays cycled through). More expensive per measurement but more honest.

---

## M4 — T2-B's "fair re-measurement" is named as follow-on but not committed to

The closeout lists T2-B's true speedup as "unknown" and proposes a fair re-measurement as Tier 2.5. But there's no commitment to actually doing it. Without that follow-on:
- The substrate has the original branchy code (conservative)
- We have no measurement supporting the original being optimal
- The branchless version is gone (reverted) without ever being fairly evaluated
- T2-B is permanently in a "we don't actually know" state

**Severity:** medium. The substrate is in a defensible state, but a real piece of the original Tier 2 list is unresolved.

**Recommended:** schedule T2.5 with a specific deadline, or explicitly close T2-B with "deferred indefinitely; not re-evaluating."

---

## M5 — The pre-commit's gate-design wasn't itself red-teamed before code

H4 from the prior remediation lifted "pre-commit ALL gates upfront" to project-wide rule. But it didn't lift "red-team the gates themselves before committing." The Tier 2 pre-commit named G2 and G4 as speedup gates without specifying the measurement methodology — that gap surfaced only in retrospect.

A red-team-of-the-pre-commit would have asked: "How exactly are you going to measure speedup? What's the comparison apparatus? Could it have inlining asymmetries?" That kind of pre-execution review would have caught the unfair-comparison flaw before code shipped.

**Severity:** medium. Methodology gap, not a finding about the work itself.

**Recommended:** lift to project methodology: pre-commits with measurement gates should include a "harness design" sub-section that specifies HOW the measurement is collected, vetted before code.

---

## M6 — The reverted T2-B code carries a 13-line inline comment about a decision that was made then unmade

The revert added a comment block in `m4t_route.c`:

```c
/* Confidence weight: per-position scan with early-exit on non-opposite.
 *
 * NOTE: a branchless per-byte version was attempted as Tier 2 remediation
 * (T2-B in journal/tier2_perf_precommit.md). Measured 2.9x SLOWER than
 * this branchy version for typical sparse-opposite-mismatch workloads
 * ...
 * Reverted; documented the FAIL in journal/tier2_perf_closeout.md. */
```

Per the project's "ship-with-FAIL" discipline, keeping the failed-experiment note is appropriate. But:
- The comment is 13 lines on a 30-line function
- The reasoning it documents has been partly invalidated by the post-revert red-team (the "2.9x slower" was an artifact)
- A 1-line pointer to the journal would carry the same documentation value

**Severity:** medium-low. Code reads slightly less cleanly. Not a substantive issue.

**Recommended:** trim the inline comment to 2-3 lines pointing at the journal, or update it to reflect the post-revert red-team's finding (artifact, not real slowdown).

---

## L1 — Perf harness is not in ctest; easy for future contributors to forget

`test_m4t_tier2_perf` is a build target but not a ctest binary. Running `ctest` doesn't include it. Future contributors might miss its existence; bit-rot risk.

**Recommended:** add a comment in CMakeLists pointing future readers at it. Or wrap it in a separate "perf" ctest target that's opt-in.

---

## L2 — File naming `test_m4t_tier2_perf.c` is misleading

It lives in `tests/` and starts with `test_`, but it isn't a test in the ctest sense — it's a measurement tool that always returns 0 (PASS) regardless of measurement outcome. Naming convention says "test" but semantics are "bench" or "perf."

**Recommended:** rename to `bench_m4t_tier2_perf.c` or move to a `bench/` subdirectory. Convention nit.

---

## L3 — "What stays open" in the closeout lists 3 items but doesn't prioritize them

The closeout lists T2-B fair re-measurement, NEON across multiple bytes for conf-dist, and magic-number-multiply vectorization as open items. No priority order, no rough cost-benefit analysis. Future readers can't easily decide which to tackle.

**Recommended:** add a one-line priority + cost estimate per open item.

---

## Summary

| ID | Severity | Status |
|----|----------|--------|
| C1 | Critical | T2-A measurement has same unfair-comparison flaw as T2-B; closeout treated them inconsistently |
| H1 | High | T2-C's new fast path likely untested by ctest (test exercises flags=non-NULL only) |
| H2 | High | (subsumed by H1) G7 "no regression" is empty for the new T2-C path |
| H3 | High | Methodology lesson lifted but not applied retroactively to T2-A's verdict |
| M1 | Medium | clock() resolution marginal for sub-µs operations |
| M2 | Medium | Test data distribution not representative of real workloads |
| M3 | Medium | Cache + branch prediction make repeat-call measurements optimistic |
| M4 | Medium | T2-B fair re-measurement named but not committed |
| M5 | Medium | Pre-commit's gate-design wasn't red-teamed |
| M6 | Medium-low | Reverted T2-B inline comment is over-long given post-revert red-team |
| L1 | Low | Perf harness not in ctest |
| L2 | Low | `test_m4t_tier2_perf.c` naming convention nit |
| L3 | Low | Open items not prioritized |

## What this red-team changes about the verdict

The Tier 2 closeout claimed:
- T2-A: PASS at 2.55x measured
- T2-B: REVERTED (measurement artifact)
- T2-C: PASS structurally

The honest revised disposition:
- **T2-A: PASS direction-of-effect; specific magnitude unverified.** The 2.55x measurement has the same fairness flaw as T2-B. NEON over scalar should give >4x algorithmic speedup; whether the measured 2.55x is correct, inflated, or deflated isn't known.
- **T2-B: REVERTED with documented methodology gap.** Substrate code is in defensible state; true comparative speed unknown.
- **T2-C: PASS structurally; new fast path possibly UNTESTED.** Until H1 is verified or addressed, the verdict relies on code review rather than test execution.
- **G7 "no regression" still holds at the suite level** (15/15 ctest binaries PASS), but the granular claim "T2-C didn't regress its own path" depends on H1.

**Recommended next move:** address H1 first (~1 hour: small property test for flags=NULL accum). Then optionally a fair-comparison re-measurement of both T2-A and T2-B (~1 day) to confirm or refute the measured speedups. Methodology improvements (M5) should be lifted to the next pre-commit doc that uses perf gates.
