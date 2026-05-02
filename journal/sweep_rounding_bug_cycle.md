---
cycle: sweep_rounding_bug
phase: RAW + CLOSEOUT (lightweight; no NODES/REFLECT/SYNTHESIZE — bug discovery, not design)
date: 2026-05-02
scope: discovery and remediation of the integer-percent floor bias in sweep_dims.c::eval_test_accuracy
companions: gesh/bench/sweep_dims.c · gesh/bench/finding3_probe.c · gesh/docs/sweep_dims_results.md · journal/gesh_sdot_finding3_redteam.md
status: bug found, fix landed, cascade complete
---

# Sweep_dims rounding-bug discovery cycle

A lightweight LMM cycle recording a methodology bug discovery that was surfaced as a side-effect of the Finding 3 high-seed measurement. Per the SDOT-finding3 red-team's H2 (no LMM cycle for the bug discovery), this entry recapitulates the discovery trail and the remediation. Bug discoveries that change published numbers warrant a journal cycle, even a light one.

## RAW — what was observed

While auditing the Finding 3 high-seed probe's output, the **5-seed sub-mean cross-check disagreed with the published `sweep_dims_results.md` numbers**:

```
finding3 5-seed sub-mean      sweep_dims_results.md (5 seeds)    drift
sig=2:  16.3 / 21.4%          15.6 / 21.0%                       +0.7 / +0.4 pp
sig=4:  21.8 / 27.2%          21.2 / 26.8%                       +0.6 / +0.4 pp
sig=8:  32.3 / 36.6%          31.8 / 36.2%                       +0.5 / +0.4 pp
```

Both probes used **identical seeds** (verified — the first 5 seed positions in `finding3_probe.c::init_seeds` and `train_seeds` arrays are the same as `sweep_dims.c::init_seeds` and `train_seeds`). Both used identical synthetic data (`cfg.seed = 0xdeadbeefu`, sample seeds `0x11111111u`/`0x22222222u`). Both used identical training configurations (50 epochs, batch=128, intra-epoch refresh per the red-team).

Despite all that, the means differ by 0.4–0.7 pp consistently in the same direction (finding3 > sweep_dims).

## What the disagreement could be

Initial hypotheses considered:
1. Seed lists differ (verified: identical).
2. Training config differs (verified: identical).
3. Data differs (verified: identical synthetic generator output).
4. Kernel path differs (verified: both use the SDOT-routed path post-cleanup).
5. **Eval rounding differs** (← this turned out to be it).

## The bug

`sweep_dims.c::eval_test_accuracy`:
```c
int correct = 0;
for (int i = 0; i < n_test; i++) {
    if (preds[i] == test_lbl[i]) correct++;
}
return (correct * 100) / n_test;   /* int division → floors to integer percent */
```

For n_test = 500 and `correct = 78` (true accuracy 15.6%), this returns `78 * 100 / 500 = 15`. The 0.6 pp fractional part is lost. Across 5 seeds with means around 15.6–21.0%, the per-seed flooring biases the 5-seed mean **systematically downward** by ~0.5 pp.

`finding3_probe.c::eval_accuracy_pm` returns permille (`(correct * 1000) / n_test`), preserving 0.1 pp resolution. Across 5 seeds the loss is ~0.05 pp instead of ~0.5 pp.

The disagreement was the int-percent floor bias.

## Why this matters

The bias affects every published number from `sweep_dims_results.md`:
- `sweep_dims_results.md` 12-cell sig_dim sweep table.
- `CHANGELOG.md` cross-references to those numbers (multiple entries).
- `gesh/README.md` Phase A.2 status block.

Magnitude varies: at low sig_dim with ~3 pp seed stddev, the 5-seed mean drift is ~0.5 pp. At high sig_dim near saturation (sig_dim ≥ 384), the per-seed value is uniformly ~96-99%; flooring still happens but flat-cap means the per-seed values were already at integer boundaries. Bias minimal there.

The most-affected cells are sig_dim ∈ {2, 4, 8} — the capacity-floor regime where Finding 3 lives. The original `sweep_dims_results.md` reported "21.0% trained" at sig_dim = 2; permille gives "21.4%" at the same 5 seeds. Drift +0.4 pp.

The drift doesn't change *direction* of any finding (compression-regime gain still positive, expansion saturation still monotone). It changes the *magnitude* by a consistent ~0.5 pp at the resolution boundary.

## Remediation

**Code fix (sweep_dims.c, ~5 lines):**
```c
return (correct * 1000) / n_test;   /* permille */
```
And the print path divides by 10 to render as percent.

Plus a determinism-check on the identity baseline (per M3): run identity twice; assert bit-equal.

**Doc cascade:**
- `gesh/docs/sweep_dims_results.md` — re-run, replace 12-cell table with permille values.
- `gesh/README.md` — Phase A.2 status numbers.
- `CHANGELOG.md` — cross-references.

**Cross-check:** post-fix, the 5-seed sub-mean of `finding3_probe.c` matches `sweep_dims_results.md`'s 5-seed cell exactly. ✓

## CLOSEOUT — methodology lesson

Two layers of lesson:

1. **Direct lesson:** integer-percent rounding in multi-seed eval introduces a per-seed floor bias that systematically under-reports the mean by half the rounding step. Permille (or floating-point) precision is needed for any benchmark whose claims rest on differences smaller than 1 pp.

2. **Meta-lesson:** the only reason this surfaced is that the Finding 3 probe's 5-seed sub-mean was deliberately constructed as a cross-check of the `sweep_dims` published numbers. **Cross-checks against established baselines are how silent methodology bugs get caught.** A probe that doesn't cross-check against existing measurements wouldn't have surfaced this; the published numbers would have continued to be ~0.5 pp low forever.

Worth promoting as a methodology rule: any new measurement that overlaps with an existing measurement's domain (same seeds, same data, same metric) should explicitly cross-check the overlap. Any disagreement that exceeds floating-point precision deserves investigation.

## What this cycle was NOT

Not a design cycle. No NODES, REFLECT, or SYNTHESIZE phases — the bug-discovery shape is RAW (observation) → fix → CLOSEOUT (lesson). The RAW/CLOSEOUT pairing is the appropriate weight for a methodology-bug cycle; the heavyweight LMM cycle structure is reserved for design questions.

## Loop-back triggers

- **No loop-back if:** the cross-check post-fix confirms the 5-seed sub-mean match. Confirmed.
- **Loop-back to RAW if:** another disagreement surfaces between probes that should produce identical output. Procedure: run the cross-check, dig into the source of disagreement, fix or document.
- **Loop-back to NODES if:** the rounding bug was somehow not the only source of the published-number drift. Verified post-fix: the only drift was the rounding bias.
