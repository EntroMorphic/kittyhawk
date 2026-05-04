# Closeout: V4 Residual #2 — Tight bound data dependency

Per the V4 closeout's honest concern #2: the V4 hardcoded `tight_bound = 10*dim` is data-dependent. It holds for the synthetic test data (`pixel(i,j) = (i*7+j*11) & 0xff`) but if the test data changes to something with a different per-image standard deviation, the SCALE/sd amplification shifts and the bound silently becomes wrong.

## Verdict: CLOSED

```
Cycle: remediate → red-team → fix red-team → positive control → recalibration validation → doc/commit
Result: tight bound is now per-image, derived from the actual data's pre-normalize sd. Auto-recalibrates.
```

## What shipped

`gesh/tests/test_image_canon.c`:

- New file-scope helpers: `test_isqrt64` (Newton iteration integer sqrt) and `derive_tight_bound`.
- `test_normalize_invariants` now allocates a per-image `tight_bounds[]` array, computes each image's bound BEFORE calling `image_canon_normalize` (which destroys the input), then iterates the normalized images comparing each `sum` against its own derived bound.
- The loose bound (`dim * SCALE / 10`) is preserved as a data-independent backstop with explicit role documentation: "primarily serves as a safety net if `derive_tight_bound` itself has a future bug."

## Math derivation (from in-source comment)

Working through `image_canon.c::normalize_one`:

- Step (a) **centering**: `img[d] -= sum/dim`. Residual `R = sum - dim * floor(sum/dim) = sum mod dim`. Bounded `|R| < dim`.
- Step (b) **rescaling**: `img[d] = floor(img[d] * SCALE / sd)`. Per-element truncation drift ≤ 1 (in absolute terms). Summed across `dim` elements: ≤ dim. The centering residual `R` ALSO gets multiplied by SCALE/sd_real, contributing ≤ |R| * SCALE/sd_real ≤ dim * SCALE/sd_real.
- Total: `|sum after normalize| ≤ dim * (1 + SCALE/sd_real)`.

Two integer-truncation pessimizations between `sd_real` and what we compute:
1. `var = sq/dim` and `sd = isqrt(var)` both truncate. Computed `sd` is a LOWER bound on `sd_real`.
2. `floor(SCALE/sd_computed)` is therefore an UPPER bound on `SCALE/sd_real`. Adding `+1` handles `floor(SCALE/sd)` itself truncating: `floor(SCALE/sd) ≤ SCALE/sd_real ≤ floor(SCALE/sd) + 1`.

Final formula:
```
scale_over_sd_ub = floor(SCALE / sd_computed) + 1
bound = 2 * dim * (1 + scale_over_sd_ub)        # 2x safety factor
```

For `dim=16`, `sd ≈ SCALE/5`: `scale_over_sd_ub = 6`, `bound = 224`.
Observed drift on this synthetic ≤ 76. Headroom ≈ 2.95×.

## Red-team and remediation

The red-team caught four findings, all addressed:

| ID | Finding | Disposition |
|----|---------|-------------|
| R-A | Code computed `2 * dim * (2 + scale_over_sd)` but math gives `2 * dim * (1 + scale_over_sd_ub)`. Off-by-one made bound 1.14× looser than necessary (256 vs 224). | **FIXED** — tightened to match math. |
| R-B | Comment "+1 on (SCALE/sd) handles integer-truncation" didn't explain it as an upper-bound trick; future reader could think it's redundant safety. | **FIXED** — comment now explicitly walks through the two truncation layers and why the +1 is structurally required, not a guard. |
| R-C | Integer `var = sq/dim` ALSO truncates, so computed `sd` is a lower bound on true sd in addition to the explicit `+1` on SCALE/sd. Worth documenting. | **FIXED** — comment now names this as an additional source of conservativeness. |
| R-D | Loose bound (94K) is now redundant — tight (224) fails first for any realistic data. Should be acknowledged. | **FIXED** — loose bound's comment renamed to "data-independent backstop" and explains its role as safety net against `derive_tight_bound` regressing. |

## Validation

**Positive control:** `/tmp/positive_control.c` (scratch, not committed). Computed baseline post-normalize sum = 5 with bound = 224. Injected +15 per pixel: sum became 245 > 224. **Tight check correctly fires.** Confirms the bound is meaningful, not vacuous.

**Recalibration validation:** `/tmp/recalibration_test.c` (scratch, not committed). Computed bounds across four data scenarios:

| Scenario | Description | Bound |
|---|---|---|
| A | Original synthetic pattern (sd ≈ SCALE/5) | 224 |
| B | Low-sd: slowly varying around SCALE/2 | 472,448 |
| C | High-sd: alternating 0/SCALE | 128 |
| D | Uniform (sd = 0) | 32 |

The bound auto-recalibrates by orders of magnitude across data shapes. Low-sd data correctly produces a much larger bound (rescaling amplifies any drift massively when sd is tiny); high-sd data produces a smaller bound (less amplification). The `sd == 0` edge case correctly returns the floor `2*dim` since `normalize_one` early-returns and post-normalize sum = 0.

## Regression check

`16/16 ctest binaries PASS` after the V4-residual-2 changes. No collateral damage in any other test.

## What's now structurally true

**Future test data changes auto-recalibrate the bound.** The V4 hardcoded `10*dim` was pinned to one specific synthetic pixel pattern. The new derivation walks through the actual data's centering and sd computation; any change to the synthetic — different pixel pattern, different `IMG_W`/`IMG_H`, different `N_TRAIN` — produces a per-image bound appropriate for that data. No silent bound-drift regression possible.

## Honest concerns from this cycle

**1. The 2× safety factor is empirical-ish.** It's documented as a safety factor without rigorous justification. In principle, the math gives an exact upper bound (no safety factor needed). The 2× exists to swallow:
- Per-element truncation that might cumulatively bias (very unlikely with random-ish data, more likely with adversarial patterns).
- Slight asymmetry between true sd and our integer approximation in edge cases.
- A judgment call about how strict "2× regression catch" should be.

For arbitrary adversarial test data, the 2× factor may not be enough. Risk: low for image-shaped data; medium if someone uses pathological synthetic.

**2. The two-pass structure (compute bounds, then normalize, then check) doubles the data traversal in the test.** Cheap for tiny test data; would matter if the test were ported to full-MNIST (10K images × 784 pixels). Defer until that becomes real.

**3. The math derivation assumes integer division truncates toward zero (C99 / C11 behavior).** If we ever cross-compile to a target with different rounding semantics, the formula needs review. Not a current concern.

## Status

CLOSED — V4 residual #2 (tight bound data dependency) is structurally remediated. The bound is now per-image, derived from the actual data, and validated against three controls (positive injection, recalibration across scenarios, full ctest regression). Future test-data changes won't silently invalidate the bound.
