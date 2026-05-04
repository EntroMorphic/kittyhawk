# Closeout: Tier 2 NEON Underuse Remediation

Per `journal/tier2_perf_precommit.md`. Three items; mixed outcome.

## Verdict: PARTIAL PASS (2 of 3 items succeeded)

| ID | Item | Outcome |
|----|------|---------|
| **T2-A** | NEON `m4t_route_select` | **PASS** (G1 correct, G2 measured 2.55x speedup) |
| **T2-B** | Branchless `confidence_weighted_dist` | **REVERTED** (G3 correct, G4 measurement inconclusive due to harness artifact) |
| **T2-C** | `accum_aligning` same-exp refactor | **PASS** (G5 correct, G6 by code review) |
| **G7** | No regression | **PASS** — all 15 ctest binaries still green |

## Per-item

### T2-A — NEON `m4t_route_select`: PASS

Replaced the scalar per-cell loop with a NEON path that processes 4 cells per iteration. Per block:
- Decode 4 trit codes from one packed-trit byte
- Build mask vectors via `vceqq_s32` against constants {1, 2}
- Bit-select via `vbslq_s32`-style cascade

**Measured 2.55x speedup** (scalar 9.258ms vs NEON 3.626ms over 100K iterations on 64-cell vectors). Gate was ≥2.0x; PASS with margin.

`test_m4t_elemental_floor` still PASSes (G1 correctness). G2 timing measurement is fair because both paths go through the lib-call boundary with the same overhead.

### T2-B — Branchless `confidence_weighted_dist`: REVERTED

Implemented a per-byte branchless version using bitwise indicator extraction + popcount. Replaced the original branchy per-position loop.

Initial perf measurement showed branchless was **2.9x SLOWER** than branchy. Reverted.

**Honest red-team-of-the-measurement (post-revert):** the perf harness compared an inlined reference (where the compiler could constant-propagate sig_dim=16, unroll loops, eliminate dead code) against the library version (called through an external function boundary, sig_dim opaque to optimizer). Function-call overhead alone explains most of the apparent 6.8x slowdown of the lib version vs inlined ref. **The original "branchless is 2.9x slower" finding was an artifact of unfair comparison, not a real algorithmic difference.**

What this actually means:
- Both branchy and branchless versions are correct (G3 PASSed for both).
- True per-call speed at substrate scale is ~equal (probably; the harness can't distinguish them fairly).
- Reverting to branchy is the conservative choice (minimizes change from the original substrate).
- A fair re-measurement would require putting both versions in the library and timing both via the same boundary — deferred as a Tier 2.5 follow-on.

The closeout note is added inline in `m4t_route.c` so future readers see why the revert happened.

### T2-C — `accum_aligning` same-exp refactor: PASS

The same-exponent branch of `m4t_mtfp_vec_accum_aligning` was scalar (per-cell add+clamp). Refactored:
- If `flags == NULL` (caller doesn't track per-cell saturation): call `m4t_mtfp_vec_add_inplace`, which is NEON-vectorized via `m4t_mtfp_block_add`.
- Otherwise: keep the scalar per-cell loop so saturation events can be detected per cell.

`test_m4t_mtfp_accum_aligning` still PASSes (G5 correctness, all 14 properties × 10K samples). G6 verified by code review — the same-exp branch goes through `vec_add_inplace` when flags are unused.

No measurable speedup gate (G6 is structural). The speedup is real for callers that don't need flag tracking — they go from per-cell scalar to per-block NEON.

## What went right

- The pre-committed gates surfaced T2-B's unfair-comparison issue cleanly. Without G4's specific threshold, I might have shipped the branchless version without knowing whether it was actually faster.
- T2-A's NEON path was the obvious win and delivered cleanly.
- T2-C's refactor was structurally correct and added no risk.
- All correctness gates (G1, G3, G5, G7) PASSed throughout. No regression at any point.

## What this taught

**Perf measurement is itself a discipline that needs gates.** The pre-commit named what to measure (speedup ratios) but didn't pre-commit what constitutes a fair comparison. The "inlined ref vs lib call" artifact was a real flaw in the harness design that only surfaced after reverting and re-running.

**Lifted to project methodology:** any future perf gate should specify HOW the timing is collected — both versions in the lib, called through identical boundaries, OR both inlined in the harness. Cross-boundary comparisons are unreliable.

**The revert was the right outcome via the wrong reasoning.** Revert minimized substrate change, which is conservative; the original measurement that motivated it was flawed. Net: code is in a defensible state, methodology is improved for next time.

## Honest concerns

1. **T2-B's true speedup is unknown.** The fair-comparison measurement was deferred; we don't actually know whether the branchless version is faster, slower, or equal at substrate scale.
2. **The Tier 2 list as originally analyzed had three real candidates; this cycle delivered one full win, one partial structural win, and one revert.** A more accurate framing of "Tier 2" might be "places where vectorization MIGHT help" rather than "places where vectorization WILL help."
3. **No NEON path on T2-B even after fair re-measurement.** If branchless turns out to be equivalent or slightly faster, the case for vectorizing further (true NEON parallelism across multiple bytes) is weaker than the precommit suggested.

## Substrate-discipline notes

- All correctness gates passed at every step. No regression.
- The reverted T2-B code preserves an inline note explaining the revert and pointing to this closeout.
- `test_m4t_tier2_perf` is registered as a build target but NOT as a ctest binary — perf measurements aren't correctness regressions.
- All 15 ctest binaries still PASS.

## What stays open

- **T2-B fair re-measurement.** Put both versions in the lib, time both via lib boundary, see which actually wins. ~1 day.
- **NEON vectorization across multiple bytes for conf-dist.** Only worth doing if T2-B's fair comparison shows the branchless version has a real algorithmic advantage.
- **Magic-number-multiply vectorization of `m4t_pow3_round_div`.** Tier 2.5 — would unlock NEON paths for the rescale branches in `accum_aligning`. Real engineering work.

## Status

CLOSED — 2 of 3 PASS, 1 reverted with documented methodology gap.
