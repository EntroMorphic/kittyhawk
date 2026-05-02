---
cycle: gesh_phase_b_redteam
date: 2026-05-02
scope: end-to-end red-team of Phase B probe (Gate 1 MNIST + Gate 2 H1 mechanism + closeout ensemble)
companions: gesh/bench/{image_canon.{c,h}, mnist_probe.c, denoise_probe.c} · gesh/docs/phase_b_gate{1,2}_results.md · journal/gesh_phase_b_probe_closeout.md · CHANGELOG.md
status: 13 findings (3 critical, 6 high, 3 medium, 1 low) — methodology gaps dominate
---

# Red-team — Gesh Phase B probe

The Phase B probe ran cleanly to clear gate verdicts. Pressure-testing surfaces that **the verdicts themselves are honest, but the closeout's narrative around them is undersupported by the measurements taken**. Most findings are about the gap between what was measured and what's claimed downstream, not bugs in the probes.

Severity tiers (matches prior red-teams):
- **C** — Critical. The claim or measurement is structurally unsupported.
- **H** — Hardening. Correct but fragile, or correct in narrow conditions and over-generalized.
- **M** — Medium. Correctness or hygiene fix without urgency.
- **L** — Low. Cleanup, comment, doc.

## Findings

### C1 — Closeout asserts "consumer architecture is the bottleneck" without measurement-controlling alternatives

**Severity: Critical.** The Gate 1 closeout (`journal/gesh_phase_b_probe_closeout.md`) and the gate doc (`gesh/docs/phase_b_gate1_results.md`) both attribute the FAIL to the consumer architecture: *"the Gesh-Phase-A consumer (single class-mean bank, top_k=1) is too weak to extract MNIST-level structure."*

This is a hypothesis dressed up as a finding. The Gate 1 measurement supports "trained Gesh hits 51–55% on MNIST in this configuration." It does NOT support the causal claim about which configuration variable is responsible. Plausible alternative causes that were not measurement-controlled:

- **Undertraining (see H1 below):** flip budget was 20k over R's ~100k trits, ~20% coverage on average per training run. The synthetic peak at sig_dim ≤ 32 used 5× coverage. Training may have done nothing detectable because we never visited 80% of the trits, not because the consumer is too weak.
- **Sample size:** n_train was subsampled to 2000 from MNIST's 60k. Per-class sample count drops from ~6000 to ~200. Class-mean bank quality scales with √n_per_class. The bank itself may have been too noisy to support training, independent of the consumer architecture.
- **Tau calibration:** tau=26687 yields ~60% structural-zero rate. This was lifted from archive convention without probe-specific calibration. Different density may give a different signal-to-noise floor.
- **Single random R seed family:** all 3 multi-seed runs use the same training/test subsamples (see H3). The "training adds nothing" verdict is observed against one data realization.

The next-cycle recommendation (**Path A: richer consumer**) is gated on the architecture-bottleneck claim. **If the actual cause is undertraining or sample size, Path A wastes a cycle.**

**Remediation (recommended, not done):**
- Three follow-up measurements that *would* disambiguate, each independently cheap:
  1. Gate 1.1 — re-run at 200k flip budget (10× current). If trained still ≤ random within noise, undertraining ruled out.
  2. Gate 1.2 — re-run at full n_train=60000 (30× current). If trained still ≤ random, sample size ruled out.
  3. Gate 1.3 — re-run with archive's `mnist_routed_bucket_multi` consumer (known-good 97.24%) using the same random R. If accuracy jumps, consumer architecture confirmed as bottleneck.
- The closeout should be rewritten with the architecture-bottleneck claim demoted to hypothesis pending these checks. **It currently presents one hypothesis as the verdict.**

### C2 — "C2 transfers cleanly to MNIST" conflates regimes

**Severity: Critical.** The closeout and CHANGELOG repeatedly cite a **+7.3pp gap** on MNIST as evidence that the synthetic's C2 finding transfers. The synthetic's C2 is precisely:

> **C2 (synthetic):** random ternary R at sig_dim = D = 64 outperforms identity at sig_dim = D = 64 by **+7.4pp**.

The MNIST measurement compares:
- Random R at sig_dim = **128** (compression regime, sig_dim < D = 784)
- Identity at sig_dim = D = **784**

These are at **different sig_dims**. The MNIST gap is between random-in-compression and identity-at-D, not between random-at-D and identity-at-D. **C2 was never tested on MNIST as constructed.** The matching ~7pp magnitude is suggestive but does not establish that C2 transfers; it shows a *different* finding (random projection in compression beats identity at full dim) with a coincidentally similar magnitude.

To test C2 on MNIST faithfully, the probe should have measured **random R at sig_dim = D = 784 on MNIST**, then compared to identity at sig_dim = 784. That measurement does not exist.

**Remediation (recommended, not done):**
- Add `random R at sig_dim = D = 784` cell to `mnist_probe.c`. One additional trial per seed, ~1 minute of runtime.
- Update closeout and CHANGELOG to specify "the **compression-vs-identity** gap is +7.3pp on MNIST; whether random@D beats identity@D on MNIST (the actual C2 regime) is unmeasured."

### C3 — Single-config Gate 1 cannot support the architecture-bottleneck narrative

**Severity: Critical.** The Phase A.2 red-team's C1 finding said: *single-seed measurements produce single-seed narratives that don't survive multi-seed averaging*. The Phase B probe applies multi-seed within a single config but runs **two configs** (sig_dim ∈ {128, 256}). The closeout's narrative about consumer architecture rests on this 2-cell measurement.

The synthetic sweep had **12 cells × 5 seeds**. The MNIST probe has **2 cells × 3 seeds**. The Gate 1 verdict is appropriately conservative *for the cells measured*, but the closeout extrapolates beyond them: *"Gesh-Phase-A consumer's expressivity ceiling on MNIST is 50–55%, regardless of projection mechanism."* The "regardless of projection mechanism" is the load-bearing phrase, and it's tested at exactly two projection sizes.

**Remediation (recommended, not done):**
- Sweep at least 4 sig_dims on MNIST (e.g., 64, 128, 256, 512) before claiming a "ceiling." A real ceiling is a curve plateau, not two adjacent points.
- If accuracy *does* plateau at 50–55% across sig_dims, the architecture-bottleneck claim is closer to supported. If accuracy keeps rising with sig_dim, the claim is falsified — the consumer can express more, the question becomes how much.

## High-severity findings

### H1 — Flip budget on MNIST is ~50× sparser than the synthetic baseline

**Severity: High.** Budget choice rationale was not surfaced in either the probe code or the doc. Math:

- **Synthetic:** sig_dim=128 (where peak gain was measured), R has 128 × 64 = 8192 trits. Sweep used budget = 5 × sig_dim × D = 40,960 flip-evals → **5× coverage of R's trits**. Each trit visited ~5 times on average.
- **MNIST:** sig_dim=128, R has 128 × 784 = 100,352 trits. Probe used flat budget = 20,000 flip-evals → **0.2× coverage**. Most trits never visited.

The trained-vs-random gain in the synthetic sweep landed at +5–8pp at the same sig_dim with 5× coverage. The MNIST measurement at 0.2× coverage shows +0.5–0.8pp. **It is not possible from this data to distinguish "training adds nothing because the consumer is weak" from "training adds nothing because we barely visited any trits."**

This is the primary mechanism behind C1's critique. Cheap to remediate: re-run with budget = 5 × sig_dim × D (= 502,400 for sig_dim=128). Probably 30–60 minutes per seed; well within probe budget.

### H2 — n_train subsample to 2000 may starve the bank

**Severity: High.** MNIST has 60,000 training samples. The probe subsamples to 2000 *to match the synthetic's runtime parity* (per a comment in `mnist_probe.c`). Per-class sample count: ~200.

Bank construction averages ~200 ternary signatures per class to form the class-mean. With 200 samples averaging out per-trit noise, the per-trit class-mean has approximately √200 ≈ 14× SNR over a single sample. This may be sufficient for synthetic data with K=16 informative dims and clean noise structure; on MNIST with 784 dims of correlated, structured signal, the bank may be too noisy at this sample size. The full 60k would give √6000 ≈ 77× SNR — 5× tighter banks.

The closeout claims the consumer architecture caps at 50–55%. **It might cap at 50–55% only at n_train=2000; with full n_train, the same consumer might hit 70–80% or higher.** Untested.

The prior-cycle's `mnist_routed_bucket_multi M=32 SUM at 97.24%` (per memory) almost certainly used full n_train. Comparing apples to oranges.

### H3 — Multi-seed varies (init, train) but not data realization

**Severity: High.** The 3 multi-seed runs use independent (init_R, train_batch) seed pairs but **the same training and test subsamples** (subsample seeds `0xa5a5a5a5u` and `0xc0ffeedu` are constants in `mnist_probe.c`). All 3 runs see the same 2000-of-60000 training set and same 2000-of-10000 test set.

**Variance the multi-seed measurement captures:** initialization of R, training-batch sequencing.
**Variance the multi-seed measurement does NOT capture:** which 2000 training samples were drawn, which 2000 test samples were drawn.

The synthetic was parameterized; varying `cfg.seed` would resample the data realization. The MNIST probe doesn't expose subsample seeds. This was flagged as a methodology limit in `gesh_findings_reflect.md` — but that flag is not surfaced in the probe doc itself, only in a prior cycle's REFLECT.

For Gate 1's pre-committed thresholds (95% / +2pp), this likely doesn't change the verdict. For Gate 1's narrative ("training adds nothing on MNIST"), it's a weaker claim — training adds nothing *on this specific 2000-sample realization*. A different draw might produce different training-vs-random gaps.

### H4 — Identity baseline is single-trial deterministic

**Severity: High.** Gate 1 reports identity@784 = 43.4% as a single number. It's deterministic (no projection seed), but the **bank construction** depends on which 2000 training samples were drawn. With a different subsample, identity might hit 41% or 46%. The +7.3pp gap is computed against this single-realization baseline.

The synthetic's identity baseline had the same property and was acknowledged. The MNIST probe inherits the same single-trial weakness without surfacing it.

### H5 — Gate 2's p-value claim overclaims, but the effect-size claim is sound

**Severity: High.** The Gate 2 doc cites *"Pearson r = +0.892, t = 157.89 (df = 6398), p << 0.001."* With df=6398 and t=157.89, the nominal p is so small it underflows ordinary representation. But the t-statistic assumes IID observations — they aren't (within a single R sample, the 64 output dims are correlated through R's row-sum structure and through the prototype geometry). The properly-corrected effective df is closer to N_R_SAMPLES × (some structural correction) than to 6400.

The **effect size** (r = +0.89) is the load-bearing number. r = +0.89 is a very strong correlation regardless of df, and it's robust to the IID-violation correction. The doc should lead with effect size; the p-value is over-precise.

This doesn't change the Gate 2 verdict (PASS) — the mechanism is well-supported by the strong effect size and the monotone stratification. It changes how the verdict should be cited: "strong positive correlation (r=0.89), monotone tertile stratification" beats "p << 0.001."

### H6 — Closeout F2 generalizes from N=2 data points

**Severity: High.** The closeout's "Findings" section asserts:

> **F2 — Random ternary projection's "+7pp over identity" is a substrate-level property, not a consumer-architecture property. It transfers across input distributions, dimensionalities, and ceiling regimes.**

This generalization is from **two** data points (synthetic + MNIST). "Transfers across input distributions" with N=2 is an inductive leap. The +7.3pp gap on MNIST is also *not at C2's regime* (per C2 finding above), so even the N=2 claim has structural concerns.

The closeout's NEXT cycle recommendation leans on F2 ("the substrate-level finding survives, so attack the consumer"). If F2 turns out to be coincidence — if random@D on MNIST hits, say, 60% rather than the 76% extrapolation — the next-cycle reasoning weakens.

**Remediation (recommended, not done):** demote F2 to "F2-hypothesis: the +7pp gap may be a robust substrate-level property; this is supported by 2 transfer points and 1 mechanism test (Gate 2). A confirmatory measurement on a third domain (e.g., Fashion-MNIST or a Go-position probe) would upgrade F2 from hypothesis to finding."

## Medium findings

### M1 — `mnist_probe.c::subsample` comment lies about the algorithm

**Severity: Medium.** The function comment says *"Random-without-replacement via Floyd's algorithm."* The implementation:

```c
for (int i = 0; i < dst_n; i++) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    picks[i] = (int)(s % (uint32_t)src_n);
}
```

This is **with-replacement uniform sampling**. Floyd's algorithm uses a hash set to track already-picked indices and re-rolls collisions. The probe's implementation does no such thing; the trailing comment "*Note: not strictly without-replacement; for n_train=2000 of 60000 the collision rate is ~3% per pick which is within noise tolerance*" partially admits this but mischaracterizes the rate.

Actual collision rate: for 2000 picks from 60000 with replacement, expected duplicates ≈ 2000²/(2 × 60000) ≈ 33 duplicates (~1.6% of the subsample is duplicated). Not catastrophic — but the comment claiming "Floyd's" should be removed and replaced with "uniform sampling with replacement; expected ~1.6% duplicate rate at this scale."

**Remediation (5-line fix):** replace the comment. Or implement Floyd's properly if without-replacement matters (it doesn't materially at this scale).

### M2 — Aliasing assertions missing on writeable outputs in `image_canon.c`

**Severity: Medium.** Per `CONTRIBUTING.md`'s post-Phase-A.1 doc-currency checklist:

> **Aliasing assertions on every writable output.** Pattern set by the m4t kernel red-teams: any function that writes to a caller-provided buffer asserts the output doesn't alias any const input.

`image_canon_quantize_unpacked_batch(x_batch, n, n_dims, tau, out_trits)` writes to `out_trits` from `x_batch`. No `assert((const void*)out_trits != (const void*)x_batch)`.

`image_canon_normalize` writes in-place — by design. Still: callers might double-call. No idempotency guard or assertion.

Same gap as the Phase A.1 red-team caught in `gesh_train_lattice_update`. Pattern transfer that wasn't applied here.

**Remediation:** add asserts; ~3 lines.

### M3 — No regression test for image_canon, mnist_probe, denoise_probe

**Severity: Medium.** ctest currently runs 12 binaries; none cover the new code. If a substrate change (e.g., M4T_MTFP_SCALE redefined) breaks the IDX→MTFP conversion, only running the binary surfaces it. If the IDX magic numbers parser regresses (e.g., endianness assumption broke after a portable-int refactor), the binary segfaults silently in CI.

Minimal smoke test: `test_image_canon` with a 4-image synthetic IDX file written to /tmp, loaded, normalized, quantized; assert basic invariants (count, range, structural-zero rate). 50 lines, registered in `gesh/CMakeLists.txt`.

**Remediation:** the smoke test is straightforward but not present. Worth adding before the next cycle relies on `image_canon` further.

### M4 — Path A's pre-committed gate is missing

**Severity: Medium.** The closeout recommends Path A (richer consumer + Gate 1 re-run) but does not specify the new Gate 1's PASS bar. Implicit assumption: same 95%/+2pp threshold. But a richer consumer at the same threshold is a different claim — multi-table LSH was already at 97% in the prior cycle. Hitting 95% with multi-table LSH would not validate Gesh's contribution; it would validate the consumer.

The right Gate 1.A criterion needs to either:
- Specify Gesh's *delta* over the archive consumer (i.e., random R + multi-table LSH vs same-multi-table-LSH-without-Gesh), or
- Specify the substrate-claim test (Gesh + multi-table LSH ≥ archive baseline at comparable parameters).

The closeout punts on this. Without a pre-committed gate, Path A's interpretation will drift to fit whatever Gate 1.A produces.

**Remediation:** a 5-line section in the closeout pre-committing Gate 1.A's PASS bar before any code is written.

## Low findings

### L1 — `MNIST_DIR` hardcoded to gitignored archive path

**Severity: Low.** The default `MNIST_DIR` in `mnist_probe.c` points at `01MAY26_archived/data/mnist`. Per `feedback_delete_never`, the archive is preserved on disk, so the path resolves. But:
- The path is gitignored — anyone cloning fresh won't have the data.
- A future cycle that adds CIFAR-10 or other datasets will want a non-archive data location.

The override-via-argv works fine. The default just isn't portable.

**Remediation:** non-urgent. Could move data files (or a symlink) under `gesh/data/mnist/` and update the default. Or document the dependency on archive presence.

## What was NOT pressure-tested

- **The image_canon pipeline correctness** (does the MTFP encoding match what a Python reference would produce per-pixel?). The probe's output is sensitive to this; an off-by-one in the integer rescaling could shift the entire accuracy regime. No reference comparison was run.
- **The lattice-update mechanism's behavior at MNIST scale with sufficient budget.** This is the H1-recommended follow-up. Not done in this red-team.
- **The Gate 2 mechanism's transfer to MNIST.** Currently demonstrated only on the synthetic. The closeout claims H1 "likely generalizes" — untested.
- **Comparison to a `top_k > 1` configuration.** The probe uses top_k=1. Synthetic numbers were also at top_k=1, so apples-to-apples. But on MNIST with 10 classes and noisy 200-sample banks, top_k=3 or top_k=5 might surface different ceilings. Not tested.

## Tally

| Tier | Count | Findings                              |
|------|-------|---------------------------------------|
| C    | 3     | C1 (causal attribution), C2 (regime conflation), C3 (extrapolation from 2 cells) |
| H    | 6     | H1–H6                                 |
| M    | 4     | M1 (comment lie), M2 (aliasing asserts), M3 (no smoke test), M4 (next-gate missing) |
| L    | 1     | L1 (hardcoded path)                   |

**Total: 14.** Most findings cluster around **methodology gaps in the closeout's narrative**, not bugs in the probes themselves. The probes ran honestly; the verdicts are the data; the closeout overcommits on causal attribution and on inductive generalization.

## Recommended remediation order (if user authorizes)

1. **C1 + H1 + H2 (paired):** rerun Gate 1 with 10× flip budget *and* full n_train. If trained still ≤ random, the architecture-bottleneck claim survives. If trained suddenly hits 70–90%, the closeout's narrative is falsified and Path A's framing changes.
2. **C2:** add the `random@sig_dim=784` cell to MNIST probe. Tests C2 on MNIST faithfully. ~1 min of runtime.
3. **C3 + H6:** sweep more sig_dims on MNIST (64, 128, 256, 512) for shape-of-curve, not point-of-claim.
4. **M1 + M2:** code cleanup. ~10 lines total.
5. **M3:** smoke tests for new code.
6. **M4:** rewrite the closeout's Path A section with a pre-committed Gate 1.A.

## What this red-team does NOT recommend

- Re-running the probes to "fix" the FAIL verdict by tuning hyperparameters. The pre-commit-and-honor methodology is intact; the failure is real *for this configuration*. The remediation makes the configuration more rigorous, not the verdict more favorable.
- Skipping Path A. If the architecture-bottleneck claim survives the C1+H1+H2 follow-up, Path A is still the right move. If it doesn't survive, *the next cycle's plan changes* but the framework holds.
- Adding more synthetic measurements. The synthetic has been measured exhaustively. Next-cycle work needs to be on real data, even with the methodology gaps surfaced here.

## Methodology lesson

The **prior** Phase A.2 red-team caught single-seed measurements producing single-seed narratives. This red-team catches a structurally similar issue at a higher level: **single-config measurements producing single-config narratives**. Multi-seed validates that one config's number isn't a seed artifact. Multi-config validates that one config's *story* isn't a config artifact. The Phase B probe has multi-seed but not multi-config. Worth promoting to a CONTRIBUTING.md checklist item: any closeout that asserts a *causal mechanism* needs measurements at multiple configurations of the variables being attributed, not just multiple seeds at one configuration.

Or shorter: **multi-seed gates the cell; multi-config gates the story.**
