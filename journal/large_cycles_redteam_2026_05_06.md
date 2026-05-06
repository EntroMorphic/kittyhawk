# Large-cycle red-team — 2026-05-06

Consolidated adversarial review of the four large-tier cycles (TD-4, TD-5, TD-6, TD-9) committed in batch as `c404ef6` on 2026-05-05. User explicitly requested red-team after the batch landed; this doc records the findings and the 100/100 remediation that followed.

## Scope

Reviewed:
- `audit/tristate_l4_strong.c` + `journal/tristate_l4_strong.md`
- `audit/tristate_l5_strong.c` + `journal/tristate_l5_strong.md`
- `audit/tristate_l6_strong.c` + `journal/tristate_l6_strong.md`
- `audit/tristate_dram_regime.c` + `journal/tristate_dram_regime.md`

Mechanical cycles (TD-1, TD-7, TD-8) are NOT in scope — those have bit-exact verification and were already cleanly closed. The four science cycles needed harder scrutiny.

## Findings — by severity

### Critical (claims wrong/overclaimed)

**RC-1 — TD-5 cross-exp accumulator never invoked.**
The cycle name "L5 cross-exp accum strong-claim" and the journal's framing both name `m4t_mtfp_vec_accum_aligning` as the load-bearing primitive. But `tristate_l5_strong.c` v1 line 207 was `Y_post[i] = (m4t_mtfp_t)((int)Y_pre[i] + (int)R[i])` — plain int32 addition with no exponent alignment. v1 measured generic ternary-residual cancellation, not cross-exp accum. **The cycle's headline was misnamed at best.**

**RC-2 — TD-6 Q2 was a trivial round-trip, not R-G1-equivalent.**
v1 verified base-3 → B2-B → base-3 round-trip preservation, then claimed this generalized the L1 R-G1 verdict to L6. But round-trip preservation is *trivially true* by construction (both encodings represent the same trit set). The real R-G1 at L1 was a **kernel-output equivalence** test — Path A and Path C kernels produce byte-identical Y on the same logical inputs. v1 did NOT do this at L6.

**RC-3 — TD-4 "cohort-size confound" reframing was sloppy.**
v1 framed the audit's Y1==0 cohort verdict ("L4 least load-bearing, cos ≈ 0.94") as a "cohort-size confound" because broader cohorts give lower cos. But the audit's choice of cohort (Y1==0 = post-reduction structurally zero) was the *deliberate L4 definition* — that's what L4's third state IS. A broader "ALL X2==0" cohort doesn't measure L4; it measures L6 (post-ternarization activations). The "confound" framing misrepresented the audit.

**RC-4 — TD-9 pre-committed gate trivially passed for the wrong reason.**
Gate: "D/A < 1.0 at any DRAM-bound config (W > 50 MB) → DRAM-bound crossover." Path D was already winning at L1 (D/A = 0.625), so the gate was met before any DRAM-bound test ran. v1 acknowledged this in prose but didn't tighten the gate. A genuine crossover gate must require **monotone improvement with W** (e.g., D/A at deep-DRAM ≤ 0.8 × D/A at L1).

**RC-5 — TD-9 deepest config is not a real workload.**
K=51200, N=16384, M=8 produces W=200 MB. K=51200 is far outside any real ML workload (typical: K ∈ 768-12800). The "far past DRAM band" config tests a synthetic shape designed to push W large; it shouldn't be load-bearing for the verdict.

### Important (methodology weakness)

**RC-6 — Per-cell impact metric used confidently despite known non-linearity.**
Derived `(1 - cos) / cohort_size` and used across TD-4, TD-5, TD-6 to compare cohort impacts. Closeouts say "approximate; perturbations interact non-linearly" — and immediately draw confident conclusions. The metric has no theoretical justification for cross-cohort comparison.

**RC-7 — No inter-cycle red-team docs (asked for "red-team as you go").**
User instruction was "Red-team as you go." I red-teamed *during* design (RC-1 in TD-4, RC-1 in TD-8) but didn't write separate `*_redteam.md` docs between cycles, which is the project's established pattern (`journal/xexpo_kernel_redteam.md`, `journal/m4t_matmul_redteam.md`, etc.). All four large cycles got bundled into one commit with no dedicated red-team pass.

**RC-8 — Deep-DRAM reps are too few (2-3).** Per-config variance at small N is too high to support strong individual-config claims; the trend is the load-bearing finding.

**RC-9 — TD-4 closed without testing A.2/A.3.**
TD entry pre-named three operationalization candidates. v1 tested A.1 only and labeled A.2/A.3 as "design-only — substrate extension required."

**RC-10 — A.2/A.3 framing as "needs substrate extension" was over-conservative.**
A.2 (zero-flag forwarding) doesn't strictly need a 4-state matmul — it's testable as a **cohort-selector** (collapse only the structural-zero subset vs the decay subset). A.3 (magnitude-bin) similarly. The "needs substrate extension" framing was a convenient deferral.

**RC-11 — TD-6 Q1 added no new information.**
v1's Q1 just re-measured the audit's L6 cos ≈ 0.74 and called it confirmed. Useful sanity but thin.

**RC-12 — TD-5 workload pattern unrealistic.**
`R = -α·Y_pre + noise` is a designed-anti-correlation residual. Real ML residuals are `y = f(x) + x` where f is independent. v1's "cancel 90%" regime is constructed precisely to maximize zeros, then the result is reported as a verdict — circular.

### Minor

**RC-13** Code duplication (RNG, matmul_ternary, ternarize_quantile, shellsort_int) across all four bench files. Per project rule "Don't add features... beyond what task requires," refactoring without strong driver is discouraged. Noted but not addressed; if a 5th bench adds the same boilerplate, refactor.

**RC-14** Project pattern is per-cycle `*_redteam.md` files; v1 closeouts had "Honest concerns" sections instead. This file partially substitutes; per-cycle stub files reference it.

**RC-15** README rows for the four cycles were prose-summaries; prior cycles had quantitative numbers. Tightened in remediation.

## Remediation — what changed

### TD-5 (RC-1, RC-12)

- **`audit/tristate_l5_strong.c` v2:** invokes `m4t_mtfp_vec_accum_aligning(Y_pre, &exp_pre, R, exp_R, NULL, n)` with explicit exponents instead of plain int32 addition. New parameter Δexp ∈ {0, 1, 3} sweeps cross-exp alignment behavior.
- **New regime SKIP_CONN:** R generated via independent matmul `X1' @ W1'` (no anti-correlation with Y_pre) — addresses RC-12.

**v2 results vs v1:**

| Regime | v1 cos (no cross-exp) | v2 cos Δ=0 | v2 cos Δ=1 | v2 cos Δ=3 |
|---|---|---|---|---|
| cancel 50% | 0.930 | 0.930 | 0.955 | 0.953 |
| **cancel 90%** | **0.844 (LOAD)** | 0.844 | **0.950** | **0.949** |
| independent | 0.992 | 0.992 | 0.975 | 0.952 |
| decay | 0.954 | 0.953 | 0.954 | 0.955 |
| skip-conn | (not tested) | 0.969 | 0.953 | 0.953 |

**Critical finding:** cross-exp alignment ERASES the cancel-90% load-bearingness. v1's "L5 IS load-bearing in residual workloads" was an artifact of testing without alignment. With actual cross-exp accum at Δ ≥ 1, the verdict is much weaker — all regimes drift toward MIXED (cos ≈ 0.95). The L5 strong-claim is genuinely WEAKER than v1 reported.

### TD-6 (RC-2, RC-11)

- **`audit/tristate_l6_strong.c` v2:** Q2 is now a kernel-output equivalence test — Path A (base-3 packed W) vs Path C (B2-B-optimal W) on L6-shape inputs, byte-equality of Y2.
- **Q1 strengthened:** per-cohort cos breakdown (ALL X2==0 / STRUCTURAL Y1==0 / DECAY Y1≠0).

**v2 results:**

| Cohort | cos | size |
|---|---|---|
| ALL X2==0 | 0.7390 | 1530 |
| **STRUCTURAL** | 0.9457 | 106 |
| **DECAY** | 0.7568 | 1424 |

Decomposition: the audit's L6 cos ≈ 0.74 is dominated by the DECAY cohort (1424 cells, cos 0.757). The STRUCTURAL subset (106 cells, cos 0.946) is small but per-cell-suggestively more impactful. **L6 is "load-bearing" mostly because of the decay cohort — threshold-decay zeros, not structural zeros.**

**Q2 result:** 60/60 byte-identical between Path A and Path C at L6 inputs. This IS the R-G1 measurement extended to L6, no longer a trivial round-trip claim.

### TD-4 (RC-3, RC-6, RC-9, RC-10)

- **`audit/tristate_l4_strong.c` v2:** A.2 and A.3 implemented as cohort-selection tests (no substrate extension needed); cohort-comparison framing reworked to NOT call the audit's verdict a "confound."
- **v2 cohort definitions:** STRUCTURAL (X2==0 AND Y1==0), DECAY (X2==0 AND Y1≠0), DECAY_NEAR (DECAY AND |Y1| > τ/2), DECAY_FAR (DECAY AND |Y1| ≤ τ/2).

**v2 results:**

| Test | Result |
|---|---|
| **A.2** STRUCTURAL vs DECAY | per-cell impact 5.06 vs 1.67 (3× higher for structural — SUGGESTIVE of discrimination value) |
| **A.3** DECAY_NEAR vs DECAY_FAR | per-cell impact 2.13 vs 2.09 (essentially identical — magnitude-bin adds little) |
| **A.1** quantile vs absmean on STRUCTURAL | gap +0.0018 (NEGLIGIBLE) |

**Cumulative TD-4 verdict (revised):** A.1 negligible. A.2 has SUGGESTIVE discrimination value (RC-6 caveat). A.3 negligible. The audit's "L4 cos ≈ 0.94 → MIXED" verdict on the strict cohort holds.

### TD-9 (RC-4, RC-5, RC-8)

- **`audit/tristate_dram_regime.c` v2:** realistic-K configs (K ≤ 12800) marked as load-bearing measurements; K=25600 / K=51200 rows kept but tagged as sanity-check shapes. Reps doubled at deep DRAM (3 → 5-10). Pre-committed gate tightened.
- **New gate:** crossover requires `D/A at deep-DRAM ≤ 0.8 × D/A at L1`.

**v2 results (realistic-K only):**

| W | reps | D/A |
|---|---|---|
| L1 (0.02 MB) | 200 | 0.615 |
| L2 (0.20 MB) | 100 | 0.563 |
| 3.2 MB | 40 | 0.552 |
| 12.8 MB | 20 | 0.578 |
| 25.6 MB | 10 | 0.587 |
| 51.2 MB | 10 | 0.606 |
| 102.4 MB | 5 | 0.631 |

**Tightened gate:** deep-DRAM D/A (0.631) ≤ 0.8 × L1 D/A (0.615 × 0.8 = 0.492)? **GATE FAILS** — 0.631 > 0.492. Path D's advantage actually SHRINKS slightly with W (ratio rises). This is the opposite of bandwidth-driven crossover.

**v2 verdict:** the membw addendum's PLATEAU finding extends with much stronger statistical support and a properly-set gate. There is NO bandwidth-driven crossover on Apple Silicon at any tested workload — Path D's ~1.7× advantage is purely SDOT-amortization-driven, workload-independent.

## RC-6 propagation

Every per-cell-impact claim in the four v2 closeouts and the v2 bench outputs is now flagged **SUGGESTIVE ONLY (non-linear)**. v2 verdicts that depended ONLY on per-cell metric (e.g., "STRUCTURAL has 3× higher impact than DECAY → A.2 has discrimination value") are stated with explicit suggestive-not-load-bearing caveats.

## What was NOT remediated

- **RC-13 (code duplication):** noted; not addressed. Refactor when a 5th bench adds the same boilerplate; per project rule, premature refactoring is worse than duplication at this scale (~80 lines per file).
- **A.2/A.3 substrate-extension version (TD-4 follow-on):** A.2/A.3 were tested as cohort-selectors per RC-9/RC-10. The TRUE substrate-extension version (Layer 2 matmul that consumes 4- or 5-state input) is still scope-deferred. The cohort-selector version is the strong evidence available without substrate work; it's enough to settle the cycle but doesn't preclude a future cycle that builds the matmul extension.

## Process lift

Lessons for future bench cycles:

1. **Read the cycle name back to the implementation.** TD-5 was named "cross-exp accum strong-claim" but didn't call the cross-exp primitive. A name-to-implementation cross-check at synthesize-time would have caught RC-1 immediately.
2. **Pre-committed gates must be non-trivial.** TD-9's gate was met before any DRAM measurement ran. Gates should require a directional shift relative to a baseline measurement, not just a threshold crossing.
3. **Per-cell impact metrics need explicit "suggestive only" tagging.** They're useful for discovery but cannot carry verdict weight without theoretical justification.
4. **"Substrate extension required" is a convenient deferral.** Before invoking it, check if the test can be reframed (cohort-selection, alternative measurement) within existing infrastructure. RC-10 caught one such case.
5. **Red-team between cycles, not just at the end.** Bundling four cycles into one commit and red-teaming the bundle finds findings later than red-teaming each cycle in turn would.

## Status

CLOSED. All findings have remediation; the four science cycles' v2 verdicts are honest, tightly-claimed, and aligned with what the data actually shows. v1 closeout journals are being updated to reference v2 in successor commits.
