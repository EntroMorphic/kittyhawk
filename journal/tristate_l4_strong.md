# L4 strong-claim cycle (TD-4)

Closes TD-4 from `docs/TECHNICAL_DEBT.md`. Per `journal/tristate_op_closeout.md` Track A.

## Question

Can L4's third state (post-reduction Y1 mantissa zeros) be made MORE load-bearing under a different operationalization rule than the audit's quantile-based ternarization?

Three pre-named candidates:
- **A.1** absmean ternarization (BitNet b1.58 rule): τ = mean(|Y|).
- **A.2** stateful zero-flag forwarding.
- **A.3** two-channel sign + magnitude split.

## RC-1 (caught during initial design)

A.2 and A.3 require richer per-cell state (extra flag bit or two channels) AND a Layer 2 matmul that consumes the augmented state. Just adding the augmentation to X2 without changing Layer 2's decode is invisible — Y2 is unchanged.

To test A.2/A.3 properly we'd need new matmul kernels (4-state or 5-state input). That's substantially out of scope for a single-cycle TD-4 closure.

**Honest framing:** A.2 and A.3 are documented as design-only. A.1 is the only candidate that's a pure RULE change and therefore directly testable with the existing 3-state matmul.

## Two-axis test (kept axes separate to avoid confound)

### PART 1 — Cohort-definition sensitivity (rule fixed = quantile)

Tests how much the L4 verdict depends on what "the third state at L4" means. Three cohort definitions:

| Cohort | Mean cos | Mean cohort | Per-cell impact (×10000) |
|---|---|---|---|
| ALL X2==0 cells | 0.733 | 1530 | 1.747 |
| Y1==0 EXACTLY (audit's L4 def) | **0.946** | 106 | **5.060** |
| NEAR-threshold zeros | 0.843 | 735 | 2.134 |

**Finding:** Confound — cohort size differs across definitions. The audit's Y1==0 cohort is the SMALLEST; its high cos (0.946 → "MIXED") was driven by cohort size, not per-cell weakness. **Per cell, the Y1==0 cohort actually has the HIGHEST impact** (5.06×, vs 1.7-2.1× for the other definitions).

**Reframe of audit verdict:** L4's third state is small in COUNT (few exact-zero cells per workload) but each one is highly load-bearing (3× per-cell impact vs the other cohorts). The audit's "least load-bearing" verdict was driven by cohort size, not intrinsic weakness.

### PART 2 — A.1 test (rule comparison on L4 cohort)

Holding the cohort fixed (Y1==0 only — same cells across both runs), does the absmean rule make L4 more load-bearing than quantile?

| Rule | cos | Cohort | Verdict |
|---|---|---|---|
| Quantile | 0.946 | 106 | MIXED |
| Absmean (A.1) | 0.944 | 106 | MIXED |
| Gap | +0.002 | — | NEGLIGIBLE |

**A.1 verdict:** the absmean rule does NOT meaningfully change L4's load-bearingness on the Y1==0 cohort. Gap is well below the 0.05 verdict threshold.

## Cumulative verdict (TD-4)

1. **Per-cell, L4's third state IS load-bearing** (impact 5.06 ×10000, the highest of any cohort tested). The audit's "least load-bearing" verdict was a cohort-size artifact.
2. **A.1 does not improve L4's load-bearingness.** Quantile and absmean rules produce essentially identical cos on the L4 cohort.
3. **A.2 and A.3 are design-only.** Both require Layer 2 substrate extensions (4- or 5-state input matmul) that are outside this cycle's scope.

**TD-4 status: CLOSED with mixed verdict.** L4 is more load-bearing than the audit suggested when judged per-cell, but no operationalization rule explored here changes the cohort-level cos meaningfully. A.2 and A.3 remain open follow-ons (would extend the substrate, not just change rules).

## Honest concerns

1. **Per-cell impact metric is approximate.** (1 - cos) / cohort_size assumes linear additivity of per-cell perturbations on Y2; in practice perturbations interact non-linearly.
2. **Workload is GEMM-only.** The audit's workload is a 2-layer ternary GEMM. Real ML workloads have layer norms, residuals, attention — different L4 dynamics.
3. **A.2 / A.3 deferred is a real scope gap.** If either substrate extension lands in a future cycle, those candidates should be retested.

## Cross-references

- Bench source: `audit/tristate_l4_strong.c`
- Original audit: `journal/tristate_op_closeout.md`
- TD entry: `docs/TECHNICAL_DEBT.md` TD-4 (now removed)
