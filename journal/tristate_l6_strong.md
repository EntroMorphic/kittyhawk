# L6 strong-claim cycle (TD-6)

Closes TD-6 from `docs/TECHNICAL_DEBT.md`. Per `journal/p0_concern2_l2.md`.

## v2 (REMEDIATED 2026-05-06) — supersedes v1

This file documents the v2 cycle. v1 had two issues per `journal/large_cycles_redteam_2026_05_06.md`:
- RC-2: Q2 was a trivial round-trip preservation test, NOT R-G1-equivalent. v2 replaces Q2 with a real kernel-output equivalence test (Path A vs Path C kernels at L6 inputs, byte-equality of Y2).
- RC-11: Q1 was just a re-measurement of the audit's L6 cos. v2 strengthens Q1 with per-cohort breakdown (ALL X2==0 / STRUCTURAL Y1==0 / DECAY Y1≠0).

## Question

L6 = post-ternarization activations (the X2 cells consumed by Layer 2's matmul). Two questions, kept separate:

- **Q1 — Load-bearingness:** is L6's third state load-bearing per Gate II? Which subset (structural vs decay) carries the weight?
- **Q2 — Encoding-label equivalence at L6:** does base-3 vs B2-B-optimal encoding produce the same kernel-level Y?

## Method (v2)

12 configs × 5 seeds = 60 runs. Per run:

1. Generate ternary X1, W1, W2.
2. Y1 = X1 @ W1^T; X2 = quantile-ternarize(Y1, p_zero=a_zero).
3. **Q1:** measure cos with three cohort definitions:
   - ALL: collapse X2[i]==0
   - STRUCTURAL: collapse X2[i]==0 AND Y1[i]==0
   - DECAY: collapse X2[i]==0 AND Y1[i]≠0
4. **Q2 (new):** pack W2 in base-3 4-in-8 (Path A) and B2-B (Path C); run both kernels; compare Y2 byte-for-byte.

## Results (v2)

### Q1 — per-cohort breakdown

| Cohort | mean cos | mean cohort | per-cell impact (×10000) [SUGGESTIVE] |
|---|---|---|---|
| ALL X2==0 | 0.7390 (LOAD) | 1530 | 1.706 |
| STRUCTURAL | 0.9457 (MIXED) | 106 | 5.103 |
| DECAY | 0.7568 (LOAD) | 1424 | 1.709 |

**Decomposition:** the audit's reported L6 cos ≈ 0.74 is dominated by the DECAY cohort (1424 cells, cos 0.7568). The STRUCTURAL subset is small (106 cells, cos 0.9457) but per-cell-suggestively more impactful.

### Q2 — kernel-output equivalence at L6

Path A (base-3 packed W) vs Path C (B2-B-optimal W) at L6-shape inputs: **60 / 60 runs byte-identical Y**.

This is the R-G1 measurement extended to L6. Encoding labels (base-3 vs B2-B-optimal) are aliases at the kernel-output level on L6 inputs — confirms the L1 R-G1 verdict generalizes to L6 by direct evidence (was a symmetry argument).

## Verdict (v2)

**Q1: L6's third state IS load-bearing in aggregate (cos 0.7390 < 0.85), but the load-bearingness comes from the DECAY cohort, not STRUCTURAL.**

The structural subset (cells where Y1 was exactly zero) is small (~7% of X2==0 cells) and lands in MIXED territory (cos 0.946). The decay subset (cells where ternarize sent non-zero Y1 to zero) is large (~93%) and lands in LOAD-BEARING territory (cos 0.757) — almost identical to the unified ALL X2==0 cohort.

**Per-cell, structural cells are ~3× more impactful** than decay cells — but per-cell metric is SUGGESTIVE only (RC-6).

**Q2: encoding-label equivalence at L6 verified.** 60/60 byte-identical between base-3 and B2-B kernel outputs.

**TD-6 status: CLOSED.** L6 strong-claim verified explicitly:
- Aggregate L6 third state IS load-bearing (cos < 0.85).
- The DECAY cohort drives the verdict; the STRUCTURAL cohort is small and per-cell-suggestively-more-impactful but cohort-aggregate MIXED.
- Encoding labels are aliases at the kernel level on L6 inputs (Q2 verified).

## Honest concerns

1. **Per-cell metric SUGGESTIVE only** (RC-6).
2. **Q2 tests Path A vs Path C only.** A full R-G1-equivalent test would also include the disasm comparison (L1 cycle did this); at L6, kernels are the *same kernels* as L1 (same inner loop), so disasm equivalence transfers by construction.
3. **Workload is GEMM-only.** Real ML L6 has activation functions and layer norms before the next matmul; these may change the cohort distribution.

## Cross-references

- Bench source: `audit/tristate_l6_strong.c` (v2)
- Original audit: `journal/tristate_op_closeout.md` (cos_L6 ≈ 0.74)
- L1 R-G1: `journal/tristate_strong_redteam.md` R-G1
- L2 extension: `journal/p0_concern2_l2.md`
- Red-team: `journal/large_cycles_redteam_2026_05_06.md` (RC-2, RC-11)
- TD entry: `docs/TECHNICAL_DEBT.md` TD-6 (now removed)

## v1 archived

v1 Q2 was a base-3 ↔ B2-B encoding round-trip preservation test (60/60 preserved). Round-trip preservation is trivially true by construction (both encodings represent the same trit set). v1 incorrectly claimed this generalized R-G1 to L6 — it didn't (R-G1 is about kernel-level equivalence, not encoding round-trip). v2 replaces with the real kernel-level test.
