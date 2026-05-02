---
cycle: gesh_sdot_finding3_redteam
date: 2026-05-02
scope: end-to-end red-team of the SDOT fix (gesh_project routing) and the Finding 3 high-seed re-measurement
companions: gesh/src/gesh_project.c · gesh/bench/{finding3_probe.c, denoise_probe.c} · gesh/docs/{finding3_high_seed_results.md, sweep_dims_results.md} · CHANGELOG.md · journal/gesh_substrate_discipline_cleanup.md
status: 14 findings (3 critical, 6 high, 4 medium, 1 low) — substrate-scope reach + uncorrected rounding bias + outcome-vs-mechanism conflation dominate
---

# Red-team — SDOT fix and Finding 3 high-seed measurement

The SDOT fix made the substrate-discipline cleanup actually fast. The Finding 3 high-seed measurement hardened a soft claim into a sharper one. Both moves are net-positive. **Pressure-testing surfaces that the *narrative around them* is, in places, ahead of what the data supports** — and that the SDOT routing is using a kernel slightly outside its declared scope.

Severity tiers (matches prior red-teams):
- **C** — Critical. Claim or measurement is structurally unsupported, OR substrate-discipline is reached past its limits.
- **H** — Hardening. Correct but fragile, narrowly conditioned, or over-generalized.
- **M** — Medium. Hygiene fix without urgency.
- **L** — Low. Cleanup, comment, framing.

## Critical findings

### C1 — `m4t_mtfp4_sdot_matmul_bt` is being used outside its labeled input class

**Severity: Critical (substrate-discipline reach).** The kernel header reads:

> *MTFP4 × packed-trit weights → MTFP19 via SDOT.*
> *Activations X are `m4t_mtfp4_t` mantissas (int8, range ±40).*

The new `gesh_project.c` calls it with `(const m4t_mtfp4_t*)x_batch` where `x_batch` is `m4t_trit_t* (= int8_t*)`. Ternary trits {-1, 0, +1} fit numerically inside MTFP4's ±40 range, so the kernel produces correct results — verified by 33/33 bit-equivalence checks in `test_gesh_project`. **But the kernel was specified, tested, and substrate-spec'd for MTFP4 activations, not ternary.** We're reaching past its scope.

Substrate-discipline implications:
- Per principle 7 ("substrate-level specs are upstream of kernel designs"), reaching past a kernel's labeled scope without a spec amendment is a discipline violation.
- The kernel's debug-build asserts only spot-check that *weights* are valid trits, not that *activations* are within MTFP4 range. Under `-DNDEBUG` (Release), no runtime check exists.
- The kernel's bit-equivalence with the reference open-coded path is real for our specific input range, but tomorrow's caller could pass an out-of-MTFP4-range int8 activation and not know.

**Three remediation paths, in increasing order of substrate-cleanliness:**

1. **Document the reach.** Update `m4t_mtfp4.h` to note that ternary activations are also accepted (since they're a subset of MTFP4 range), with explicit text in the input-class contract. Lightweight; doesn't add code; clarifies intent. Honors principle 7 by amending the spec when the kernel is used outside its original class.

2. **Add a thin wrapper kernel `m4t_trit_x_trit_dot_matmul_bt`** in libm4t that delegates to `m4t_mtfp4_sdot_matmul_bt` internally but exposes a substrate-clean ternary × ternary API. Three lines of code; documents the ternary × ternary input class as a first-class operation; principle-5 consumer demand exists.

3. **Add a dedicated ternary × ternary SDOT kernel** with its own range bounds and tests. Most substrate-clean, most code; the SDOT path itself is identical, so it's mostly an API surface decision.

The current code works correctly within its narrow input range. The discipline question is whether we want libm4t's API to *say* what it actually accepts. (1) at minimum is needed before the next substrate-claim publication.

### C2 — Uncorrected rounding bias in `sweep_dims.c`; published numbers are stale

**Severity: Critical.** The Finding 3 probe surfaced that `sweep_dims.c::eval_test_accuracy` returns `(correct * 100) / n_test` — int division, flooring each seed's percent. Across 5 seeds, this systematically under-reports by ~0.5 pp; for trained mean at sig_dim = 2, the 5-seed bias is **+1.7 pp** (5-seed sweep claimed 21.0% vs 30-seed permille 19.3%).

**The fix wasn't applied to `sweep_dims.c`.** All sweep_dims numbers published in:
- `gesh/docs/sweep_dims_results.md` (the 12-cell table through sig_dim = 1024)
- `CHANGELOG.md` (multiple entries citing those numbers)
- `gesh/README.md` (Phase A.2 status block)

…still carry up to ~1.7 pp downward bias on the trained means. The capacity-floor cells (sig_dim ≤ 8) are the most affected; the saturation cells (sig_dim ≥ 256) are unaffected (accuracy near the test-set ceiling, integer rounding minimal effect).

The doc-currency issue cascades. Every claim in `sweep_dims_results.md` rests on numbers known to be biased. The cleanup is small (5 lines) but propagates to multiple documents. **Currently the only "correct" measurement is `finding3_probe.c` at sig_dim ∈ {2, 4, 8}**; everything else needs to be re-run at permille precision before being cited as authoritative.

**Remediation:** change `(correct * 100) / n_test` to `(correct * 1000) / n_test`, divide by 10 at print time. Re-run sweep_dims. Update `sweep_dims_results.md`, `CHANGELOG`, README. Cascade is non-trivial (~5 docs) but mechanical.

### C3 — Capacity-floor *mechanism* is asserted but not measured

**Severity: Critical.** Finding 3's framing: *"With 3² = 9 distinct ternary signatures vs C = 10 classes, this is information-theoretically limited. The capacity-bounded behavior is now a finding, not a hypothesis."*

The 30-seed measurement demonstrates the **outcome**: monotone climb 19.3 → 27.0 → 35.9 across sig_dim ∈ {2, 4, 8} with non-overlapping CIs. It does NOT demonstrate the *cause*. Plausible alternative explanations consistent with the same outcome:

- **Training-budget effect.** At sig_dim = 2, budget = 5 × 2 × 64 = 640 flip-evals; R has 128 trits, so coverage is ~5×. At sig_dim = 8, budget = 2560 with 512 trits, coverage ~5×. Coverage is matched. ✓ Budget effect ruled out.
- **Bank-construction effect.** Class-mean bank with 200 samples projected to 2 trits hits the noise floor faster than projection to 8 trits. The bank quality might cap accuracy independent of the capacity argument.
- **Init-distribution effect.** Random ternary R at sig_dim = 2 has 3² = 9 possible R configurations per output dim (modulo sign-of-all-zero); at sig_dim = 8 there are 3⁸. Init landing on a "good" R is rarer at low sig_dim.
- **The actual capacity argument.** With 9 signatures and 10 classes, at least one class collides by pigeonhole. But the *measurable* effect of this collision is unknown — does the bank handle the collision via tie-break, or does the colliding class get classified as its neighbor, or is it spread across multiple signatures?

The probe doesn't distinguish these. **The "capacity-bounded behavior is now a finding" claim overstates what the data demonstrates.** What the data shows: trained accuracy at low sig_dim has a robustly low ceiling that climbs monotonically with sig_dim. That's an *observation* consistent with the capacity argument; it's not a confirmation of the capacity *mechanism*.

To upgrade outcome → mechanism, the probe would need to:
- Examine the trained R at sig_dim = 2 and report whether the 10 class-mean tiles span 9 vs 10 distinct signatures (pigeonhole prediction: must be ≤ 9).
- Compute the per-class confusion matrix at sig_dim = 2 and identify whether the predicted "collision" pattern (one class gets ≤ 50% accuracy because it shares a signature with another) actually appears.
- Repeat at sig_dim = 4 (3⁴ = 81 signatures, no pigeonhole forcing) and sig_dim = 8 (6561 sigs, plenty) to verify the mechanism's onset.

Without these, the "capacity floor" framing is a hypothesis, not a finding — the same H3 / hypothesis-vs-finding distinction the Phase A.2 red-team caught for "implicit denoising via random ternary projection." Both are plausible mechanisms, both demonstrate consistent outcomes, both would be findings if the mechanism were directly tested.

## High-severity findings

### H1 — Cast `(const m4t_mtfp4_t*)x_batch` is type-punning across nominally-distinct types

**Severity: High.** `m4t_trit_t = int8_t`, `m4t_mtfp4_t = int8_t`. They share an underlying type, so the cast is identity at the bit level. C standard: pointer casts between types of compatible representation are allowed.

**But:** the cast is silently coupling two semantic types. If a future refactor changes `m4t_mtfp4_t` to a different cell type (e.g., adds a tag, uses int16, etc.), this cast becomes a bug. The compiler accepts it today; tomorrow it might compile through and silently corrupt.

**Remediation:** add a `static_assert(sizeof(m4t_trit_t) == sizeof(m4t_mtfp4_t), "...")` in `gesh_project.c` near the cast site, plus a comment explaining the deliberate type-pun. Or — better — wrap the cast in an inline helper that documents the bit-compatibility assumption. ~3 lines.

### H2 — The "investigation that revealed the rounding bug" was not logged as an LMM cycle

**Severity: High.** The 5-seed-sub-mean disagreement led me to dig into `sweep_dims::eval_test_accuracy` and find the int-percent floor bias. That investigation surfaced a real methodology issue (now in `finding3_high_seed_results.md`'s methodology note and CHANGELOG). But there's no LMM journal cycle recording:
- The observation (sub-mean disagreed with published numbers).
- The hypothesis chain (kernel/seed/data inspection).
- The synthesize action (use permille; flag sweep_dims).

Per project discipline, methodology bugs that change published numbers warrant a journal entry. The cycle convention is `journal/<scope>_<phase>.md` with RAW → NODES → REFLECT → SYNTHESIZE → CLOSEOUT. The current handling — note in a results doc + a CHANGELOG paragraph — is lighter than the project's established cycle bar.

**Remediation:** lightweight journal cycle (RAW + closeout, skip the middle phases) recording the bug-discovery, the cross-check that exposed it, and the planned cascade. 1 file.

### H3 — Data-realization variance still unsampled

**Severity: High.** The Finding 3 high-seed probe runs 30 seed-pairs over (init_R, train_batch). The synthetic data realization stays fixed (`cfg.seed = 0xdeadbeef`, train sample seed `0x11111111u`, test `0x22222222u`). All 30 trials see the same 2000 train + 500 test samples.

This is the same H3 issue the Phase B red-team flagged. The probe document (`finding3_high_seed_results.md`) does NOT acknowledge this limit explicitly. Reads as if multi-seed = multi-everything-variance; actually = multi-(init, train).

**Remediation:** a sentence in `finding3_high_seed_results.md` near the methodology note: *"30-seed measurement varies (init_R, train_batch); data realization (`cfg.seed`, sample seeds) stays fixed. Variance from data-resampling is unsampled per the H3 limit acknowledged in `journal/gesh_phase_b_redteam.md`."*

### H4 — Speedup numbers in CHANGELOG are cross-branch, not apples-to-apples

**Severity: High.** The CHANGELOG quotes:

> | measurement | open-coded (pre-clean) | packed-trit kernel | SDOT kernel |
> |-------------|-------------------------|---------------------|--------------|
> | MNIST ablation total | 210s | 1740s | **156s** |

The "open-coded" 210s was measured **before the substrate-discipline cleanup**, on a code path that no longer exists. The "packed-trit" 1740s and "SDOT" 156s were measured post-cleanup with otherwise-identical code. **The 210s ↔ 156s comparison spans two different code branches**; differences could come from:
- Compiler optimization decisions on different surrounding code.
- Allocation patterns in the wrappers (changed during cleanup).
- Cache/ICache behavior with different code layouts.

Probably the SDOT path is genuinely faster than open-coded, but the timing comparison as written overstates the rigor. To be apples-to-apples we'd need to measure all three on the same checkout (revive the open-coded path temporarily for benchmark only).

**Remediation:** rewrite the CHANGELOG table to mark the "open-coded" column as approximate / cross-branch. Or run a side-by-side comparison and replace with same-checkout numbers. Annotation is sufficient.

### H5 — Test_gesh_train shows SDOT is *slower* than open-coded for small workloads

**Severity: High.** Pre-cleanup test runtime: 0.84s. Post-cleanup w/ SDOT: 1.14s. **36% slower.**

The CHANGELOG framing: *"SDOT path is faster than the original open-coded compiler-vectorized loop AND fully on-substrate."* This is true for the **MNIST-scale** matmul (D=784, sig=128, batch=128). It's **false** for the **synthetic-test scale** matmul (D=64, sig=16, batch=64), where allocation overhead and threshold-extract per-row int64 widening dominate vs the cheap matmul.

Both regimes are bit-equivalent. But the speedup story has a regime where SDOT loses, and the framing currently doesn't surface that.

**Remediation:** in the CHANGELOG, note the regime split: *"SDOT path is faster than open-coded for matmuls large enough that NEON dominates per-call setup cost (MNIST-scale: 1.3× faster). For small synthetic-scale matmuls, the threshold-extract widen-and-unpack overhead can flip the comparison (test_gesh_train: 1.4× slower than pre-cleanup open-coded)."*

A scratch-aware threshold-extract would close this gap. Not done; not urgent given small-scale isn't a substrate-claim hot path.

### H6 — Finding 3 doc says "finding, not hypothesis" but the *trained-gain* sub-claim is shakier

**Severity: High.** The doc states:

> *"All gains are statistically positive (lower CI bound > 0). C1 (lattice update earns gain in compression) holds at the capacity floor."*

Examining the gain CIs:
- sig_dim = 2: gain +3.5 pp, CI ±1.5 pp → lower bound +2.0 pp
- sig_dim = 4: gain +4.6 pp, CI ±1.5 pp → lower bound +3.1 pp
- sig_dim = 8: gain +5.1 pp, CI ±1.7 pp → lower bound +3.4 pp

All lower bounds positive. Statistically the gain holds. **But the gain CIs were computed by treating random and trained as independent; they're not.** Both use the same init_seed; trained additionally uses train_seed. The (init_seed-dependent) variance is shared.

The correct gain CI is on the per-seed *paired difference* (trained_pm[s] - random_pm[s]), not from independent stddev sums. Computed pair-wise, the gain CI would be tighter (paired-sample t-test). The "lower bound > 0" claim survives — paired CIs are tighter, not looser — but the methodology in the probe is loose.

**Remediation:** add per-seed paired-difference CI to `finding3_probe.c` output. ~10 lines.

## Medium findings

### M1 — `finding3_probe.c` seeds 6–25 have arithmetic-suffix patterns

The 25 fresh seed pairs include `0x10203040, 0x50607080, 0x90a0b0c0, 0xd0e0f000, 0x11223344, 0x55667788, 0x99aabbcc, 0xddeeff00`. These are byte-stride patterns, not unstructured random hex. Worth nothing for xorshift collision risk, but seeds with structure can interact unexpectedly with state-evolving generators.

**Remediation:** replace 8 of the 25 with truly-random hex (e.g., from `/dev/urandom` once, hardcoded). 1-line change.

### M2 — No outlier detection in 30-seed Finding 3 measurement

Reported `min` and `max` per cell:
- sig_dim = 2 trained: min 12.0%, max 25.2% — 13.2 pp range across 30 seeds.
- sig_dim = 4 trained: min 22.4%, max 37.6% — 15.2 pp range.

Per-seed range is wide. If 1 of 30 seeds landed at 12.0% (vs cluster at 18-22%), the mean is biased. With 30 seeds the influence is small (~3% weight), but median or trimmed-mean would be more robust to outliers.

**Remediation:** report median and trimmed-mean (10% trim) alongside arithmetic mean. ~5 lines.

### M3 — Single-trial identity at sig_dim=64 (69%) is still un-multi-seeded

Inherited limitation from the original sweep. Identity has no projection seed; bank construction is over the (fixed) train set. So identity should produce a single deterministic number. Verifying this would require running identity twice and confirming the result is bit-identical. Easy check, not done.

**Remediation:** a deterministic-trial assertion in the sweep's identity path would catch any non-determinism that creeps in. ~3 lines.

### M4 — `gesh_project_scratch_t` carries dead fields after SDOT cleanup

The struct still declares `R_packed` and `X_mtfp` fields, kept NULL by the SDOT-path init. Dead fields signal "this code recently changed" but should be removed for hygiene.

**Remediation:** remove the dead fields, update the only caller (gesh_train's `gesh_train_scratch_t`). Both files affected; ~10 lines changed.

## Low findings

### L1 — CHANGELOG "fully on-substrate" framing has caveats

The phrase "100% on-substrate via libm4t kernels" is technically accurate but overlooks:
- The MTFP4 SDOT kernel is being used outside its labeled scope (C1).
- The threshold-extract per-row widen path still allocates per call in non-scratch wrappers.
- `image_canon::normalize_one` is not on-substrate (sanctioned per §12 but worth flagging in the headline phrase).

**Remediation:** soften the headline to "every multiply-accumulate and sign-threshold runs through libm4t kernels" — matches what's actually true.

## What was NOT pressure-tested

- **Whether `m4t_mtfp4_sdot_matmul_bt` actually uses SDOT on this build.** The kernel is `#if M4T_HAS_NEON && defined(__ARM_FEATURE_DOTPROD)`-gated. Without verifying compile-time defines, the kernel might silently fall back to scalar for ternary × ternary if the dotprod feature isn't defined. Check via `objdump -d` for `sdot` instruction; not done.
- **The `image_canon::quantize` cleanup correctness on non-trivial pixel data.** `test_image_canon` covers small synthetic IDX; no test runs against real MNIST and validates that the kernel-routed quantizer produces the same trit signatures as the open-coded path on real images.
- **The Finding 3 "30-seed sub-mean of first 5 should match sweep_dims" cross-check** assumes integer rounding is the only drift source. If there were a deeper bug (different randomness path, different data realization), the rounding explanation would mask it. Worth a hard equality test rather than a 0.5-pp tolerance.

## Tally

| Tier | Count | Findings |
|------|-------|----------|
| C    | 3     | C1 (kernel scope reach), C2 (uncorrected rounding in sweep_dims), C3 (capacity-floor mechanism unconfirmed) |
| H    | 6     | H1 (type-pun cast), H2 (no LMM cycle for the bug discovery), H3 (data-realization variance unsampled), H4 (cross-branch speedup numbers), H5 (SDOT slower at small scale), H6 (gain CI not paired) |
| M    | 4     | M1 (patterned seeds), M2 (no outlier detection), M3 (single-trial identity), M4 (dead struct fields) |
| L    | 1     | L1 (framing caveats) |

**Total: 14.** The substantive findings cluster around (a) using the SDOT kernel slightly past its labeled scope, (b) the rounding bias in `sweep_dims.c` propagating to all published numbers without cleanup, and (c) Finding 3's framing as "now a finding, not hypothesis" overstating what the outcome data demonstrates without mechanism-level verification.

## Recommended remediation order

1. **C2 + H4 paired:** fix `sweep_dims.c` to permille precision, re-run, update `sweep_dims_results.md`, `CHANGELOG`, README. Cascades but mechanical.
2. **C1:** at minimum, document the MTFP4-SDOT kernel's accepted-input-class extension to ternary (header text). Optionally, add a thin wrapper kernel `m4t_trit_x_trit_dot_matmul_bt` for substrate-cleanliness.
3. **C3 + H6:** add per-class confusion-matrix output to `finding3_probe.c` to confirm the pigeonhole prediction. Add paired-difference CI for the gain.
4. **H1 + H5:** add `static_assert` for the type-pun, document the regime split in CHANGELOG.
5. **H2:** lightweight LMM cycle for the rounding-bug discovery.
6. **M1 — M4:** code/doc hygiene.

## Methodology note: meta-pattern across red-teams

Across four red-teams in this codebase:

- **Phase A.2 red-team:** caught single-seed → seed-noise narrative artifact.
- **Phase B red-team:** caught single-config → config-confound causal artifact.
- **Phase B kernel-use audit:** caught open-coded → substrate-bypass.
- **This red-team:** catches in-scope-kernel → out-of-scope-kernel-application.

The pattern at each level: a **lower-N measurement is used to support a higher-N claim**. Single seed claims about populations. Single config claims about mechanisms. Hand-coded loops claim substrate-claims. **In-scope kernels claim out-of-scope behavior.**

The discipline rule is the same at each level: *the N of the support must match the N of the claim*. This generalizes the Phase A.2 + Phase B + Phase B-kernel rules into a single principle worth promoting:

> **Match the scope of evidence to the scope of claim.** Single seeds do not measure populations. Single configurations do not establish mechanisms. Substrate-shaped re-implementations are not substrate measurements. Kernels labeled for class A do not establish behavior on class B without scope-extension. Each gap is a methodology debt; each cleanup adds a checklist item.

Worth a CONTRIBUTING.md entry that subsumes the prior three rules under this single meta-pattern.
