# L4 strong-claim cycle (TD-4)

Closes TD-4 from `docs/TECHNICAL_DEBT.md`. Per `journal/tristate_op_closeout.md` Track A.

## v2 (REMEDIATED 2026-05-06) — supersedes v1

This file documents the v2 cycle. Per `journal/large_cycles_redteam_2026_05_06.md`:
- RC-3: v1's "cohort-size confound" framing misrepresented the audit's deliberate L4 definition.
- RC-9 / RC-10: v1 deferred A.2 and A.3 too aggressively. They're testable as cohort-selection rules without a substrate extension.
- RC-6: per-cell impact metric flagged as SUGGESTIVE.

## Question

Can L4's third state (post-reduction Y1 mantissa zeros) be made MORE load-bearing under a different operationalization?

Three pre-named candidates from `tristate_op_closeout.md`:
- **A.1** absmean ternarization (BitNet b1.58 rule)
- **A.2** stateful zero-flag forwarding (distinguish structural-zero from decay-zero)
- **A.3** two-channel sign + magnitude split (or magnitude-bin)

## Method (v2)

All three candidates testable as cohort-selection or rule-swap within the existing 3-state matmul:

- **A.1** = rule swap (quantile → absmean), measure cos on STRUCTURAL cohort.
- **A.2** = cohort split: STRUCTURAL (X2==0 AND Y1==0) vs DECAY (X2==0 AND Y1≠0).
- **A.3** = cohort split within DECAY: DECAY_NEAR (|Y1| in (τ/2, τ]) vs DECAY_FAR (|Y1| ≤ τ/2).

12 configs × 5 seeds × 4 cohorts × 2 rules.

## Results (v2)

### Part 1 — A.2 zero-flag forwarding (cohort-selection)

| Cohort | mean cos | size | per-cell (×10000) [SUGG] |
|---|---|---|---|
| STRUCTURAL | 0.9462 | 106 | 5.060 |
| DECAY | 0.7627 | 1424 | 1.667 |

**A.2 verdict:** STRUCTURAL has ~3× higher per-cell impact than DECAY. Per RC-6 caveat, this is SUGGESTIVE only — but it does indicate that structural-zero cells differ from decay-zero cells in some downstream-relevant way. A.2's flag-forwarding has discrimination value, but the strength of the claim is bounded by the per-cell metric's reliability.

### Part 2 — A.3 magnitude-bin (subdivide DECAY)

| Cohort | mean cos | size | per-cell (×10000) [SUGG] |
|---|---|---|---|
| DECAY_NEAR (|Y1| in (τ/2, τ]) | 0.8431 | 735 | 2.134 |
| DECAY_FAR (|Y1| ≤ τ/2) | 0.8563 | 689 | 2.087 |

**A.3 verdict:** DECAY_NEAR and DECAY_FAR have essentially identical per-cell impact. A.3's magnitude binning adds little — within the decay subset, "barely below threshold" and "far below threshold" cells have the same downstream weight.

### Part 3 — A.1 absmean rule on STRUCTURAL cohort

| Rule | cos | size |
|---|---|---|
| Quantile | 0.9462 | 106 |
| Absmean (A.1) | 0.9444 | 106 |
| Gap | +0.0018 | — |

**A.1 verdict:** NEGLIGIBLE (gap << 0.05 threshold). Quantile and absmean rules give essentially identical cos on the L4 cohort. The choice of ternarization rule doesn't change L4's verdict.

## Cumulative verdict (v2)

L4 = post-reduction Y1 mantissa zeros (Y1==0 EXACTLY). This is the audit's deliberate definition, NOT a "cohort-size confound" (RC-3 fix).

The audit's verdict (cos ≈ 0.946 → MIXED) holds: this strict cohort is small (~7% of X2==0 cells) and the cohort-aggregate cos sits in the MIXED band.

**Of the three candidates:**
- **A.1** (absmean rule): NEGLIGIBLE effect on L4 verdict.
- **A.2** (zero-flag forwarding, as cohort-selector): SUGGESTIVE-of-discrimination (3× per-cell impact gap), but per-cell metric is RC-6-caveated. The cohort-selection version of A.2 ships the *evidence*; the substrate-extension version (Layer 2 matmul that consumes a 4-state input encoding `(trit, flag)`) would ship the *exploitation*. Substrate extension remains scope-deferred.
- **A.3** (magnitude-bin): NEGLIGIBLE within the DECAY subset.

**TD-4 status: CLOSED with mixed verdict.** A.1 doesn't change L4's verdict; A.2 has suggestive discrimination value (per-cell) but cohort-aggregate stays MIXED; A.3 adds nothing. The strongest finding is the structural-vs-decay per-cell disparity (3×), which is suggestive evidence that A.2's flag-forwarding is the most promising future direction — but it requires substrate work to exploit, and that's deferred.

## Honest concerns

1. **Per-cell impact metric is SUGGESTIVE only** (RC-6). Y2 perturbations are non-linearly composed; (1−cos)/cohort_size is a rough proxy.
2. **A.2 substrate-extension version not implemented.** v2 tests A.2 as cohort-selection; the substrate-extension version (4-state input matmul that decodes `(trit, flag)` per cell) would actually exploit the discrimination value the per-cell metric SUGGESTS exists. That's a separate cycle.
3. **Workload is GEMM-only.** Real ML L4 has layer norms / residuals; the cohort distribution may differ.
4. **Cohort sizes interact with verdict thresholds non-trivially.** STRUCTURAL has ~106 cells avg, DECAY has ~1424. A "MIXED" verdict at low cohort size and a "LOAD-BEARING" verdict at high cohort size aren't directly comparable.

## Cross-references

- Bench source: `audit/tristate_l4_strong.c` (v2)
- Original audit: `journal/tristate_op_closeout.md`
- Red-team: `journal/large_cycles_redteam_2026_05_06.md` (RC-3, RC-6, RC-9, RC-10)
- TD entry: `docs/TECHNICAL_DEBT.md` TD-4 (now removed)

## v1 archived

v1 framed Part 1 as "cohort-definition sensitivity (the audit's verdict was a confound)" — RC-3 caught that as misrepresentation. v1 also deferred A.2 and A.3 as "design-only — substrate extension required" — RC-9/RC-10 caught that as too conservative. v2 implements A.2 and A.3 as cohort-selectors and reframes the cohort comparison honestly.
