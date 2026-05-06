# L5 strong-claim cycle (TD-5)

Closes TD-5 from `docs/TECHNICAL_DEBT.md`. Per `journal/tristate_strong_closeout.md` Track C.

## v2 (REMEDIATED 2026-05-06) — supersedes v1

This file documents the v2 cycle. v1 had a critical RC-1 error (the bench did not actually invoke `m4t_mtfp_vec_accum_aligning` despite the cycle being named for cross-exp accum). Per `journal/large_cycles_redteam_2026_05_06.md`, v2 fixes RC-1 (uses cross-exp primitive with explicit exponents) and RC-12 (adds an honest skip-connection regime with no anti-correlation).

## Question

When does L5's third state (exact-zero output of cross-exp accumulation) carry downstream weight? GEMM-only workloads don't exercise cross-exp accum; the TD source explicitly required a residual-style workload.

## Method (v2)

Workload pattern:
- `Y_pre = matmul(X1, W1)` — ternary GEMM, int32 mantissas at exp=0
- `R` generated per regime, at exp = (running_exp − Δexp)
- `m4t_mtfp_vec_accum_aligning(Y_pre, &exp, R, addend_exp, NULL, n)` — substrate's actual cross-exp primitive
- `X2 = ternarize_quantile(Y_post, p_zero=0.40)`
- `Y2 = matmul(X2, W2)`

Five regimes × three Δexp × six configs × five seeds = 450 measurements.

**Regimes:**
- cancel 50% : `R = -Y_pre/2 + small noise`
- cancel 90% : `R = -0.9·Y_pre + small noise`
- independent : `R = uniform random small`
- decay : `R ∈ {-1, 0, +1}` (much smaller than Y_pre)
- skip-conn (RC-12): `R = matmul(X1', W1')` where X1', W1' are independent ternary tensors

**Δexp variants:** 0 (no alignment, plain addition), 1 (mild rescale), 3 (stronger rescale).

L5 cohort: cells where `Y_post == 0 AND X2 == 0`. Gate II measurement: collapse cohort, compute `cos(Y2_native, Y2_collapsed)`.

## Results (v2)

Mean cos by (regime, Δexp):

| Regime | Δ=0 | Δ=1 | Δ=3 |
|---|---|---|---|
| cancel 50% | 0.9305 (MIXED) | 0.9547 (SINK) | 0.9528 (SINK) |
| cancel 90% | **0.8443 (LOAD)** | 0.9495 (MIXED) | 0.9492 (MIXED) |
| independent | 0.9919 (SINK) | 0.9748 (SINK) | 0.9524 (SINK) |
| decay (small) | 0.9525 (SINK) | 0.9539 (SINK) | 0.9550 (SINK) |
| skip-conn (real) | 0.9685 (SINK) | 0.9531 (SINK) | 0.9531 (SINK) |

Cross-exp alignment effect (gap from Δ=0 baseline):

| Regime | Δ=0 → Δ=1 | Δ=0 → Δ=3 |
|---|---|---|
| cancel 50% | +0.0242 | +0.0223 |
| **cancel 90%** | **+0.1052** | **+0.1048** |
| independent | -0.0170 | -0.0394 |
| decay | +0.0014 | +0.0025 |
| skip-conn | -0.0154 | -0.0154 |

## Verdict (v2)

**The L5 strong claim is WEAKER than v1 reported.**

- At Δ=0 (no cross-exp alignment), the cancel-90% regime gives cos = 0.844 (LOAD-BEARING). This is the only LOAD verdict in the matrix.
- At Δ ≥ 1 (true cross-exp), the cancel-90% verdict moves to MIXED (cos 0.95). Cross-exp alignment ERASES the cancellation-driven load-bearingness.
- All other regimes hover in the SINK band across all Δ values.
- The skip-connection regime (real residual pattern) is firmly SINK at all Δ.

**Why alignment erases the signal:** the cross-exp primitive rounds during alignment. A cell that was "exactly zero from cancellation" before alignment may become non-zero post-alignment (and vice versa) — the rounded sum carries different information than the unrounded sum. Over many cells, this washes out the structural signal that the third state was supposed to encode.

**TD-5 status: CLOSED.** L5's third state has NARROW load-bearingness:
- LOAD-BEARING only when (a) the workload exhibits strong structural cancellation AND (b) cross-exp alignment is not invoked. In other words, L5 is load-bearing in *same-exponent* residual workloads with cancellation, not cross-exp ones.
- SINK in real-residual (skip-connection) workloads, in independent-residual workloads, and in decay-only workloads, regardless of Δexp.

## Honest concerns

1. **Per-cell impact metric is SUGGESTIVE only** (RC-6). Per-cell tables in the bench output are non-linear; they're not load-bearing for the verdict.
2. **The "cancel 90%" regime is engineered.** R = -0.9·Y_pre is not a natural ML pattern; it requires structural anti-correlation.
3. **Skip-connection still uses ternary GEMM for R**, not a learned residual. Real transformer skip-connections may differ further.
4. **Δexp ∈ {0, 1, 3} is a sparse sweep.** Larger Δ (≥ 20 → degenerate truncation) not tested; the "interesting" range is 0-5 where alignment is non-trivial.

## Cross-references

- Bench source: `audit/tristate_l5_strong.c` (v2)
- Cross-exp accum kernel: `m4t/src/m4t_mtfp.c::m4t_mtfp_vec_accum_aligning`
- Original cross-exp design: `journal/xexpo_design_closeout.md`
- Production routing: `journal/cross_exp_accum_routing_closeout.md`
- Red-team: `journal/large_cycles_redteam_2026_05_06.md` (RC-1, RC-12)
- TD entry: `docs/TECHNICAL_DEBT.md` TD-5 (now removed)

## v1 archived

v1 cycle results (Δ=0 implicitly, plain int32 addition) were:
- cancel 90%: cos 0.844 (LOAD-BEARING)
- decay: cos 0.954
- skip-conn: not measured

v1's verdict "L5 IS load-bearing in residual workloads" was correct *for same-exp residuals* but did not test the cross-exp behavior. v2 corrects the scope.
