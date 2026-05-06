# L6 strong-claim cycle (TD-6)

Closes TD-6 from `docs/TECHNICAL_DEBT.md`. Per `journal/p0_concern2_l2.md`.

## Question

L6 = post-ternarization activations (the X2 cells consumed by Layer 2's matmul). The audit measured cos(L6) ≈ 0.74 (LOAD-BEARING) using ternary representation. The TD source notes the verdict "likely follows L1/L2 by structural symmetry but not directly measured" with respect to encoding-label equivalence.

Two questions, kept separate:

- **Q1 — Load-bearingness:** is L6's third state load-bearing per Gate II?
- **Q2 — Encoding-label equivalence at L6:** does base-3 vs B2-B encoding change Y2?

## RC-1 (caught pre-execution)

Y2's value is encoding-independent — base-3 and B2-B encode the same trits, only the storage/decode layout differs. So cos(Y2_native, Y2_collapsed) is identical at the OUTPUT level regardless of encoding. The encoding only affects kernel wall-clock, not Y2.

Q2 reduces to: does the round-trip (X2 → B2-B encoding → decode) preserve every cell's trit value? If yes, the encoding label is just a relabeling, and the L1 R-G1 verdict generalizes by direct evidence (was a symmetry argument).

## Method

12 configs × 5 seeds = 60 runs. Per run:

1. Generate ternary X1, W1, W2.
2. Y1 = X1 @ W1^T; X2 = quantile-ternarize(Y1, p_zero=a_zero).
3. **Q1**: collapse cohort = {i : X2[i] == 0}; Y2_test = X2_collapsed @ W2; cos(Y2_native, Y2_test).
4. **Q2**: encode X2 → (sign[], mask[]) via B2-B mapping; decode back → X2_rt; verify byte-for-byte match with X2.

## Results

### Q1 (load-bearingness)

Mean across 12 configs × 5 seeds:

| | mean cos | mean cohort | per-cell impact (×10000) |
|---|---|---|---|
| L6 | **0.7390** | 1530 | 1.706 |

**Verdict Q1: LOAD-BEARING** (cos < 0.85). Matches the original audit's reported cos_L6 ≈ 0.74 within RNG variance, providing independent confirmation.

### Q2 (encoding-label equivalence)

| | rate |
|---|---|
| Round-trip preservation | **60 / 60** (100%) |

**Verdict Q2: VERIFIED.** Base-3 and B2-B encode the same per-cell trit values; round-trip through B2-B preserves every cell. The L1 R-G1 verdict generalizes to L6 by direct round-trip evidence (was a symmetry argument).

## Cumulative verdict (TD-6)

1. **L6's third state IS load-bearing.** cos = 0.74, well below the 0.85 LOAD-BEARING threshold. Confirms the original audit.
2. **Encoding-label equivalence holds at L6.** Round-trip preserves all trit values across all 60 runs. The L1 R-G1 symmetry argument is now directly evidenced at L6, not just inferred.
3. **L6 strong-claim follows L1/L2's pattern.** Encoding labels are aliases; what matters is the trit values themselves. Implementation choice (base-3 storage vs B2-B sign+mask) at L6 is a wall-clock concern only, not an algorithmic one.

**TD-6 status: CLOSED.** L6 strong-claim explicitly measured; consistent with the L1/L2 verdicts.

## Honest concerns

1. **Q2's round-trip test is structural, not behavioral.** It verifies that B2-B can store the same trit values as base-3. It does NOT verify that a B2-B-native matmul kernel produces the same Y as a base-3-native matmul kernel — that's a separate measurement (R-G1 already did it for L1 by disasm comparison). TD-6 leans on the L1 R-G1 verdict for kernel-level equivalence.
2. **No wall-clock comparison at L6.** This cycle measures the algorithmic question only. Wall-clock at L6 follows the L1 strong-claim pattern (Path D wins by ~1.7×) by symmetry; explicit L6 wall-clock confirmation could be added in a future cycle if a consumer asks.
3. **Same cohort-size confound as TD-4.** L6's "all X2==0" cohort is the largest of any layer (~1530 cells avg). The cos = 0.74 is partly driven by cohort size. Per-cell impact (1.706) is comparable to L1/L2's per-cell levels.

## Cross-references

- Bench source: `audit/tristate_l6_strong.c`
- Original audit: `journal/tristate_op_closeout.md` (cos_L6 ≈ 0.74)
- L1/L2 strong claim: `journal/tristate_strong_*.md` series
- L1 R-G1 (encoding-label equivalence at L1): `journal/tristate_strong_redteam.md` R-G1
- L2 extension (P0-Concern-2): `journal/p0_concern2_l2.md`
- TD entry: `docs/TECHNICAL_DEBT.md` TD-6 (now removed)
