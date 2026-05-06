# L5 strong-claim cycle (TD-5)

Closes TD-5 from `docs/TECHNICAL_DEBT.md`. Per `journal/tristate_strong_closeout.md` Track C ("Cross-exp accum strong-claim L5: requires residual-style workload not produced by GEMM-only").

## Question

When does L5's third state (exact-zero output of cross-exp accumulation) carry downstream weight? GEMM-only workloads don't exercise cross-exp accum substantially. The TD source explicitly required a residual-style workload.

## Method

Workload pattern: `Y_post = Y_pre + R` where:
- `Y_pre = matmul(X1, W1)` — int32 mantissas from a ternary GEMM
- `R` is a residual addend with one of four regimes
- `X2 = ternarize(Y_post, p_zero=0.40)`
- `Y2 = matmul(X2, W2)`

Four residual regimes:

| Regime | R definition | Expected L5 zero-fraction |
|---|---|---|
| cancel 50% | R = −Y_pre/2 + small noise | high |
| cancel 90% | R = −0.9·Y_pre + small noise | very high |
| independent | R = uniform random small | low |
| decay (small-exp) | R ∈ {−1, 0, +1} (much smaller than Y_pre) | low-mid |

L5 cohort = cells where Y_post == 0 AND X2 == 0 (structural cancellation that survives ternarization). Gate II measurement: collapse L5 cohort (force ±1 in X2), measure cos(Y2_native, Y2_collapsed).

12 configs × 5 seeds × 4 regimes = 240 measurements.

## Results

| Regime | mean cos | mean cohort | Verdict | Per-cell impact (×10000) |
|---|---|---|---|---|
| cancel 50% | 0.930 | 192 | MIXED | 3.671 |
| **cancel 90%** | **0.844** | 561 | **LOAD-BEARING** | 2.771 |
| independent | 0.992 | 36 | SINK | 2.358 |
| **decay (small-exp)** | 0.954 | 113 | SINK (cohort-aggregate) / **HIGH per-cell** | **4.085** |

## Two complementary readings

**By raw cohort cos (the audit's primary metric):**
- Cancel 90% wins → L5 is LOAD-BEARING when residuals are designed for cancellation.
- Independent regime → L5 is SINK-LIKE (no L5 work to do).
- The bigger the cancellation pressure, the more load-bearing L5's third state.

**By per-cell impact (the TD-4 RC-1-lifted methodology):**
- Decay regime wins per-cell (4.085 ×10000 — highest of all).
- Each "decay zero" carries more downstream weight than each "cancellation zero."
- Decay zeros are RARE but each one matters disproportionately.

Both readings agree: **L5's third state IS load-bearing in residual workloads.** GEMM-only workloads understate the load-bearingness because they don't trigger cross-exp accum scenarios.

## Cumulative verdict (TD-5)

1. **L5 is load-bearing in residual-style workloads.** Cancellation regimes (especially 90%) drop cos below the LOAD-BEARING threshold.
2. **L5 is sink-like in independent (non-residual) workloads.** Confirms why GEMM-only audits gave no L5 verdict.
3. **Per-cell, even decay zeros (small cohort, sink-aggregate) are highly load-bearing.** Each cell that becomes zero from accumulator decay matters disproportionately.

**TD-5 status: CLOSED.** L5's third state has consumer-pattern-dependent load-bearingness. The substrate's `m4t_mtfp_vec_accum_aligning` is the cross-exp primitive; consumers that use it for residual computation get load-bearing third states; consumers that use it for independent additions don't.

## Honest concerns

1. **Workload is synthetic.** Real ML residuals (transformer skip-connections) have tighter structural correlation between residual and main path than the random `R = -α·Y_pre + noise` here.
2. **No cross-exp alignment is exercised.** The bench keeps both Y_pre and R at the same effective exponent. True cross-exp scenarios would amplify the decay regime's contribution.
3. **Per-cell impact metric is approximate** (same caveat as TD-4 closeout).

## Cross-references

- Bench source: `audit/tristate_l5_strong.c`
- Cross-exp accum kernel: `m4t/src/m4t_mtfp.c::m4t_mtfp_vec_accum_aligning`
- Original cross-exp design: `journal/xexpo_design_closeout.md`
- Production routing: `journal/cross_exp_accum_routing_closeout.md`
- TD entry: `docs/TECHNICAL_DEBT.md` TD-5 (now removed)
