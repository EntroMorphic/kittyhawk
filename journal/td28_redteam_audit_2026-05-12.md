# Full red-team audit of the Glyph codebase — synthesis

Four parallel audit agents covered four slices:
- **Audit-1**: libm4t substrate kernel semantics
- **Audit-2**: BitNet production inference hotpath
- **Audit-3**: experiments/ Python infrastructure (non-γ)
- **Audit-4**: vision-claim ↔ shipping-code consistency

Below: findings consolidated by severity, with my verification where I cross-checked, and a remediation order.

## Findings by severity

### HIGH (action needed)

None confirmed. The one HIGH from Audit-1 (`m4t_trit_sparsity`/`counts`
silently misbehave on reserved 0b11 codes) is a **defense-in-depth gap**
rather than an active bug: the pack functions never emit 0b11, so the
reducers only see it if an external caller constructs a buffer
incorrectly. Worth a runtime guard or static assertion but not an
active hazard.

### MEDIUM

1. **`m4t_route_select` asserts `n_cells > 0` without documenting it**
   (`m4t/src/m4t_route.c:469,543`). Callers composing this primitive
   may hit the assert without the header indicating the precondition.

2. **2-bit trit encoding invariant is compile-time guarded
   (`M4T_TRIT_PACKED_BYTES` math) but not runtime-asserted at
   buffer boundaries** (`m4t_trit_pack.h:87-94` header warning).
   If anyone hand-builds a packed buffer with 0b11 or 1-bit-sign-only
   codes, `m4t_popcount_dist` silently degrades to a different
   distance — the exact silent-degradation failure the td28
   discovery was about, on the other side of the API.

3. **`m4t_route_topk_abs` uses `uint64_t used` bitmask for ≤64 tiles;
   T > 64 only asserts in debug** (`m4t/src/m4t_route.c:310-351`). At
   release-time, T > 64 would silently produce wrong results. T is
   currently never user-controlled in production but this is a hidden
   trap.

4. **Branchless and branching `confidence_weighted_dist` variants
   coexist with no test gate ensuring bit-exact equivalence**
   (`m4t/src/m4t_route.c:81-130` branching, `:554-597` branchless).
   Future optimization to one path could drift from the other
   silently.

### LOW

5. **`gesh/bitnet/README.md` line 180-182 claims sparse attention is
   "experimental, not on the production hot path"** but the harness
   (`bitnet_harness.c:1139-1140`) was promoted to NEON-production on
   2026-05-12. Documentation drift only; the production code is correct.

6. **Aliasing assertions on writable output buffers not yet
   transferred** to BitNet forward path (per CONTRIBUTING.md §63
   pattern from Gesh Phase A.1). Currently safe because of
   single-threaded inference and independent scratch buffers, but
   the pattern transfer is overdue.

### VISION-LEVEL (not code bugs — design questions)

7. **The "six-primitive floor" foundation claim lists ~6 primitives
   including exp and log; the substrate ships add/sub/mul/neg/max/min/eq
   but exp/log are absent.** This is **straightforwardly: the work
   hasn't been done yet**. The vision claim #1 explicitly names exp
   and log; the user has disclaimed (with extreme prejudice — see
   `feedback_no_consumer_barrier.md`) the "no primitive without demand"
   framing for foundational primitives. The vision's naming IS the
   demand. Original framing of this finding cited that rule and was
   wrong; corrected here.

   The honest gap: exp and log are owed to the foundation; the
   substrate currently has 4 of the named 6. Plan and ship them, or
   refine the foundation statement if "six" was approximate.

   *Audit drift note: Audit-4 surfaced this against `CONTRIBUTING.md:15`
   verbatim ("No primitive without named consumer demand"). I
   synthesized that into the finding without checking memory. The
   subagent didn't have access to `feedback_no_consumer_barrier.md`;
   I did, and missed it. See also remediation item 8 below.*

8. **The Phase β/γ/δ/ε arc's "L1 vs Hamming" narrative is a
   methodological artifact of comparing two Python implementations
   when production is already on L1.** Already documented in
   `td28_l1_already_in_production_2026-05-12.md`; flagging here as
   a documentation-consolidation need rather than a new finding.

## Audit-3 findings I VERIFIED AGAINST AND REJECTED

The Python-code audit returned two findings flagged "CRITICAL":

- **`run_phase_beta.py:181-182`**: `n_total = int(np.sum(flat <= t1)) * 2`
  claimed to inflate counts and bias d̂.
- **`heterogeneity_check.py:48`**: same shape.

**Both invalid on inspection:**

1. In `run_phase_beta.py`, `n_total`/`k_total` are computed AFTER
   `estimate_id_fixed_radii` returns `d`. They are returned as
   bookkeeping summary stats — they do NOT feed into d_hat.
   `estimate_id_fixed_radii` does its own per-point counting
   internally (verified at `experiments/phase_alpha/m1_estimator_v2.py:140-175`).

2. In both files, the `* 2` multiplier applies to BOTH `n_total` AND
   `k_total`. Even when these DO feed into the target ratio (heterogeneity
   case), the ratio is invariant under the constant factor. d_hat is
   unaffected.

   Verified empirically: `n=3, k=7 → 3/7 = 0.4286 ≡ 6/14 = 0.4286`.

The audit agent applied template "multiplied-by-2-is-a-double-count" reasoning
without checking whether the ratio invariance preserves d_hat. Recording
this here because the pattern (a subagent declaring CRITICAL based on
shallow inspection) matches my own session's overclaim history — the
audit needed an audit.

## Cross-cutting themes

**Theme 1: The codebase is in good shape.** Across all four audits, the
production hotpath has no bugs; substrate kernels are well-documented
and consistent with their headers; the recent td28 misalignment is
documented and memory-encoded.

**Theme 2: Defense-in-depth is the biggest opportunity area.** Most
remaining concerns are "this WOULD silently fail IF an external caller
violated an unwritten contract." None are firing now, but each is a
landmine for future development.

**Theme 3: Documentation lags code.** The README.md, the substrate-claim
journal series, and the foundation claim about six primitives all have
drift relative to current code. The fix is a single doc-consolidation
pass, not new code.

**Theme 4: The audit-the-audit pattern is real.** Audit-3's false-
critical findings show that even adversarial review can pattern-match
without verifying. Reinforces the `feedback_spot_check_before_verdict`
memory: always trace findings to ground truth before acting.

## Remediation priority (recommended)

If we were to remediate, in order of leverage:

1. **Update `gesh/bitnet/README.md`** to reflect sparse-attention
   production status (5-line edit, LOW).
2. **Document `m4t_route_select` `n_cells > 0` precondition** in
   header (MEDIUM, 1-line edit).
3. **Add runtime assert** in `m4t_trit_sparsity`/`counts` rejecting
   0b11 codes when validation builds are enabled — or a static
   assertion linking pack contract to reduce expectation (MEDIUM,
   2-line addition + 1 unit test).
4. **Promote `T > M4T_ROUTE_MAX_T` check** in `m4t_route_topk_abs`
   to release-mode (MEDIUM, change `assert` to runtime check).
5. **Add bit-exact-equivalence test** for branching vs branchless
   `confidence_weighted_dist` (MEDIUM, 1 unit test).
6. **Consolidate the substrate-claim arc's documentation** into one
   "what the substrate does, with empirical margins" doc; the journal
   series is now 8+ files with corrections that need a single
   readable summary (LOW priority, MEDIUM effort).
7. **Resolve the six-primitive foundation claim**: either commit to
   delivering exp/log or refine the claim's wording (VISION-level
   decision). The vision claim names them; the work is owed.

8. **Amend `CONTRIBUTING.md:15` and `README.md:63`** — the "no
   primitive without named consumer demand" rule keeps causing audit
   drift. The user has disclaimed it (with extreme prejudice) for
   foundational primitives (the named six in vision claim #1). The
   rule still applies to derived/composite kernels. Scope it
   explicitly in the doc, or remove it if it causes more drift than
   it prevents. Memory `feedback_no_consumer_barrier.md` has the full
   disclaimer.

None of these are urgent. Items 1-5 could ship in a single
"defensive hardening" commit.

## What this audit did NOT cover

- The 100+ journal files in `journal/` (only sampled; many older
  ones may contain claims that the recent corrections now
  contradict).
- The `gesh/src/` and `gesh/bench/` consumer-side code outside the
  BitNet harness.
- The MTFP arithmetic kernels (`m4t_mtfp.c` — 2400+ lines) in depth.
  Cross-exp accumulator and ternary matmul were spot-checked but
  not exhaustively audited.
- Build system / CMake configuration.
- Test coverage analysis (which lines of production code do tests
  actually exercise).

A follow-up audit pass on any of these would be straightforward to
delegate.

## Discipline note

The Phase α→ε arc was a 19-misalignment journey through escalating
red-teams. The full audit produces ZERO new firing bugs and 5
defense-in-depth gaps. That's an honest assessment: **the recent
self-criticism wasn't picking at noise — the bugs were real and got
fixed across that work.** What's left in the codebase is structurally
sound. The next 100 misalignments will require a different lens
(application-level, integration, scale) — not more inspection of
the same upstream code.
