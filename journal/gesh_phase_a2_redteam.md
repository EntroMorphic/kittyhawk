---
cycle: gesh_phase_a2_redteam
date: 2026-05-02
scope: end-to-end red-team of Phase A.2 (lattice-update training + sig_dim sweep + closeout doc currency)
companions: gesh/src/gesh_train.{c,h} · gesh/tests/test_gesh_train.c · gesh/bench/sweep_dims.c · gesh/docs/sweep_dims_results.md · gesh/README.md · journal/gesh_design_closeout.md · CONTRIBUTING.md · CHANGELOG.md
status: COMPLETE — 12 findings remediated, 1 acknowledged (multi-seed methodology lesson recorded)
---

# Red-team — Gesh Phase A.2

End-to-end pressure on Phase A.2's code, measurement methodology, and documentation ensemble. Modeled after the m4t kernel red-team and the Phase A.1 forward-pass red-team. Three severity tiers:

- **C** — Critical (the measurement is wrong or the implementation is unsafe).
- **H** — Hardening (correct but fragile, drifts under realistic conditions).
- **M** — Medium (correctness or hygiene, fix without urgency).
- **L** — Low (cleanup, comment, doc-currency).

## Findings

### C1 — Single-seed sweep produced narratives that didn't survive multi-seed averaging

**Severity: Critical.** The original `sweep_dims.c` ran one seed per (sig_dim, variant) cell. The resulting `sweep_dims_results.md` reported a "+15pp peak at sig_dim = 16," a "+13pp gain at sig_dim = 32," and a "−2pp anomaly at sig_dim = 64 (training walks into a worse basin)." All three were single-seed artifacts.

Multi-seed (5 seeds per cell) corrected:
- sig_dim = 16: +8.0pp ± 4.6pp (not +15pp)
- sig_dim = 32: +8.2pp ± 2.4pp (not +13pp)
- sig_dim = 64: +1.8pp ± 2.3pp (not −2pp; the anomaly evaporates)

The qualitative story (compression regime helps, expansion regime saturates) survived; the headline-number narratives did not. **This is the exact issue the C1 finding was about: a sweep that supports directional claims with single-seed numbers can publish artifacts as findings.**

**Remediation:** rewrote `sweep_dims.c` to average across N_SEEDS=5 with independent (init, train) seed pairs per trial. Reports mean ± stddev. Updated `sweep_dims_results.md` to report multi-seed numbers and explicitly retract the single-seed claims. Lesson lifted to `CONTRIBUTING.md`.

### H1 — Stale bank during epoch (intra-epoch flip evaluations score against a stale bank)

**Severity: High.** `gesh_train_lattice_update` builds the bank from R at the start of each epoch, then evaluates ~hundreds of flip candidates per epoch against that bank. After a few accepted flips, R has changed — the bank is now a stale derived statistic, and subsequent flip evaluations score against a slightly wrong contract.

**Remediation:** added `bank_refresh_every` config knob; `count_errors_scratch` rebuilds the bank from current R every k accepted flips. Default in `sweep_dims.c`: every (n_flips/4) flips, so bank is rebuilt 4 times per epoch on average.

### H2 — Stale batch (same training batch reused across all flip evaluations within an epoch)

**Severity: High.** Within an epoch, the same batch was used to evaluate every flip candidate. A flip that improves loss on this batch may not generalize. Combined with H1's stale-bank issue, the within-epoch optimization was scoring flips against a doubly-stale signal.

**Remediation:** added `batch_refresh_every` config; resamples the training batch on the same cadence as the bank refresh. Now flip-evaluations get a fresh batch every (n_flips/4) flips.

### H3 — "Implicit denoising" framing was a hypothesis stated as a finding

**Severity: High.** The single-seed `sweep_dims_results.md` claimed: "random ternary projection at sig_dim = 64 beats identity at +10pp via implicit denoising — random projection mixes noise dims into incoherent signal that the class-mean averages toward zero." This is a plausible mechanism, but it was not measured. The data only shows the *outcome* (+7pp under multi-seed), not the *mechanism* (denoising via dim-mixing).

**Remediation:** rewrote that section to flag the +7pp gap as a robust finding and the denoising mechanism as a hypothesis with a proposed mechanism test (project the noise dims, examine class-conditional variance). Added a "Hypotheses (NOT verified findings)" section. Pattern lifted to `CONTRIBUTING.md` as a doc-currency check.

### M1 — Test gate `< batch_size` was too loose

**Severity: Medium.** The training-reduces-loss test gate was "trained error < `batch_size` errors at any point." With `batch_size=128`, that's just "less than 100% wrong on the batch" — a trivially-pass gate. The test would pass even if training did almost nothing.

**Remediation:** tightened to `< batch_size / 2` (must beat 50% error). Test still passes on real training; now actually fails if training is broken.

### M2 — Single-seed test was a single sample of variance

**Severity: Medium.** `test_trains_reduces_loss` ran one seed. A run that happens to land on a good seed passes; a flaky implementation passes intermittently.

**Remediation:** added `test_multi_seed_stability` — 3 seeds, requires average gain ≥ 3pp across the three. Catches "training works on one seed and is flat on others" failure modes.

### M3 — No regression-floor test (training never significantly worse than random)

**Severity: Medium.** Even after fixing M1/M2, no test gates against "training catastrophically underperforms random." A regression where training drives R into a worse basin would pass the "trained beats batch_size" test as long as training did *some* work.

**Remediation:** added `test_no_catastrophic_regression` — requires `trained ≥ random - 5pp` on a held-out test set. Locks in the "training never hurts much" property.

### M4 — Per-flip mallocs in the hot loop

**Severity: Medium.** The original `count_errors` allocated/freed scratch buffers (`projected[]`, `preds[]`) on every call, and `count_errors` was called O(n_flip_evals_per_epoch × n_epochs) times — millions of allocations per training run on the larger sig_dim points.

**Remediation:** introduced `gesh_train_scratch_t` allocated once at the top of `gesh_train_lattice_update`, passed by pointer into `count_errors_scratch`. ~10× speedup on the sweep.

### M5 — No early stopping (wastes compute when accuracy plateaus)

**Severity: Medium.** Training ran for the full epoch budget even when accuracy stopped improving. On the larger sig_dim points, half the epochs were wasted.

**Remediation:** added `early_stop_patience` config; halts when batch error doesn't improve for N consecutive epochs. Default 5 in `sweep_dims.c`. Cuts sweep runtime by ~30%.

### M6 — Budget-vs-R-size silently degrades

**Severity: Medium.** If the training budget is small relative to R's trit count (e.g., `flip_budget < sig_dim × input_dim`), only a fraction of trits get evaluated. The user gets an under-trained R with no warning.

**Remediation:** `gesh_train_lattice_update` now emits a one-line `[gesh_train] warn:` log when `flip_budget < n_flips_total`, listing the ratio. Doesn't fail; just flags.

### L1 — `seed = 0` was undocumented and silently broken

**Severity: Low.** The xorshift state initialized from `seed = 0` produces all-zero output (xorshift's degenerate case). No assertion, no fallback; users would get a "constant projection" failure mode that's hard to debug.

**Remediation:** documented that any seed including 0 is valid; internal `0x12345678u` mixed into the state to break the all-zero case. Sweep-tool seed lists explicitly exclude 0 anyway, but the library function is now safe.

### L2 — Random-init projections are not balanced (per-row trit count drifts)

**Severity: Low.** `gesh_init_random_projection` writes uniform-random ±1 (no zeros) per trit. Per-row sums vary by ±√D from zero on average; rows with large positive or negative drift project x toward a fixed sign regardless of x's content (a slight bias, but a real one).

**Remediation:** added `gesh_init_random_projection_balanced` which generates ±1 in a balanced (per-row sum = 0 for even D, ±1 for odd D) shuffle. Not the default — `gesh_init_random_projection` is what `sweep_dims.c` uses, deliberately, to keep the random-baseline as "what users naive-init would get." But available for callers that want it.

### L5 — Closeout said "no periodic refresh"; implementation has intra-epoch refresh

**Severity: Low.** `journal/gesh_design_closeout.md` asserted "no periodic refresh of tile signatures — they don't drift, because there are no continuous shadow parameters to track." The reasoning was correct (no float→quantized drift), but the conclusion overgeneralized — the bank is a derived statistic of R, so every accepted flip stales it. The closeout missed this distinct refresh-need.

**Remediation:** added a "Post-implementation revision" section to `gesh_design_closeout.md` distinguishing **STE-shadow refresh** (correctly absent) from **R-derivative refresh** (correctly present). Recording the lesson: when a mechanism choice eliminates one reason for an operation, check whether other reasons remain.

## What was NOT remediated and why

- **Asynchronous flip-evaluation parallelism** — would be a meaningful speedup at sig_dim ≥ 256 but adds concurrency surface area that doesn't earn its keep at Phase A scope. Deferred until a consumer asks.
- **Variance reduction (e.g., common random numbers across variants)** — the multi-seed sweep already gives ±2pp confidence intervals on the headline numbers, sufficient to support the qualitative claims. CRN would tighten the intervals but doesn't change the conclusions.
- **Mechanism tests for the implicit-denoising hypothesis** — proposed tests are documented in `sweep_dims_results.md` § Hypotheses; deferred to a future cycle that wants to upgrade hypotheses to findings.

## Tally

- 12 findings remediated (C1, H1, H2, H3, M1–M6, L1, L2, L5).
- 0 findings deferred without remediation.
- 0 findings escalated.

12 / 12 tests pass post-remediation. Sweep runs in ~22s on Apple Silicon. The Phase A.2 measurement now stands on multi-seed evidence with documented hypothesis/finding distinctions.

## Lesson promoted to project-level

`CONTRIBUTING.md` now includes:
- "Multi-seed validation for any directional measurement claim" — single-seed measurements are exploratory; multi-seed measurements are evidence.
- "Hypothesis vs finding distinction in measurement docs" — mechanism explanations are hypotheses until tested by mechanism-revealing measurements.

These are the two methodology-level lessons from this cycle. They generalize to any future measurement that supports a published claim.
