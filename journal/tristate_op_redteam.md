# RED-TEAM: tri-state operationalization audit

Cold-eye review of `journal/tristate_op_closeout.md` + `audit/tristate_audit.c`.

## Critical finding (warrants remediation + re-run)

### C1 — L4 collapse is structurally a no-op in high-act-zero configs

The headline finding ("L4 is the highest-leverage operationalization target") rests on `cos = 1.000` in the high-act-zero configs (a=0.60). On inspection, this is an **artifact of the L4 collapse design**, not a property of L4's third state.

**The flaw:**

```c
static void l4_collapse(m4t_mtfp_t* dst, const m4t_mtfp_t* src, int n, rng_t* r) {
    /* Compute median of non-zero |Y| */
    ...
    sub = nonzero_abs[nz / 2];   /* MEDIAN of non-zero |Y1| */
    ...
    for (int i = 0; i < n; i++) {
        if (src[i] == 0) dst[i] = (m4t_mtfp_t)(rng_sign(r) * sub);
        else             dst[i] = src[i];
    }
}
```

Then the collapsed Y1 is fed to `ternarize_quantile` with `target_zero_frac` = a (e.g., 0.60).

**What happens:**
1. The substituted values have magnitude = median of |non-zero Y1|.
2. After substitution, sorted |Y1| has the substituted values clustered at the ~50th percentile of non-zeros.
3. `ternarize_quantile` picks τ at the (a)-quantile of |collapsed Y1|. With a=0.60 and only ~5-13% of cells originally zero, τ sits ABOVE the median of non-zeros — meaning the substituted values fall BELOW τ and ternarize back to **0**.
4. Net effect: collapsed X2 ≡ native X2. cos(Y2_native, Y2_collapsed) = 1.000 by structural identity.

**The L4 collapse silently undoes itself via the downstream threshold.** The audit's reported `cos = 1.000` does NOT mean "L4's third state is invisible to downstream"; it means "this specific collapse design substitutes magnitudes the threshold reverses."

**Severity:** The headline finding ("L4 highest-leverage") and Track A's recommendation are based on this artifact. Must remediate.

**Remediation:** redesign L4 collapse so the substitution survives ternarization. Cleanest approach:
- Compute X2 via native ternarization on native Y1.
- Then OVERRIDE X2 at positions where Y1 was zero, forcing those cells to ±1 (random).
- This directly tests "if you forced L4's exact zeros to ±1 downstream, would Y2 change?" — the actual Gate II semantics for L4.

This bypasses any threshold interaction. Re-run after the fix.

## Doc-level concerns (no code change; documentation only)

### D1 — L3 collapse is mathematically equivalent to L1+L2 simultaneous collapse, not independent

For ternary inputs, X*W = 0 iff X==0 OR W==0. The L3 collapse "replace zero MAC products with random ±1" produces the same downstream output as "make BOTH X and W binary." So L3's measurement is L1+L2 simultaneous, not an independent third axis.

The closeout did note L3 as a derived measurement, but should be more explicit: L3's verdict is interpretable only relative to the union of L1+L2 collapse, not as an independent layer of evidence.

### D2 — L1 collapse modifies BOTH W1 and W2 (whole-weight intervention)

Current code:
```c
binary_collapse(w_test.W1, ...);
binary_collapse(w_test.W2, ...);
```

L1 represents "weight third-state across the model." This is a strong intervention — the cosine similarity then reflects cumulative effect through both layers. An alternative isolated-L1 measurement would collapse W1 only and leave W2 native. The audit picks the cumulative reading; should be documented as such.

### D3 — PRNG state shared across Gate II measurements within a seed

Each Gate II call (L1, L2, L3, L4, L6) advances the same RNG. Order-dependent entropy: the L6 measurement's randomness depends on what L1-L4 consumed first. Independent per-layer RNGs (seed XOR layer-id) would be cleaner, though the present design isn't biased — order-dependence affects per-seed values but mean over 5 seeds remains unbiased.

### D4 — Realism gate is trivial under direct generation

60/60 PASS is unsurprising: `gen_ternary` directly samples to target zero-fraction. The gate would only bite if a future cycle generates trits via more complex pipelines (e.g., real Gaussian → quantile-ternarize). For this audit, the gate added zero signal. Already noted in the closeout's "honest concerns."

### D5 — L2 / L6 collapse changes activation distribution substantially

L2 collapse: X1's zero-fraction goes from `act_zero_frac` (0.20 or 0.60) to 0. The distribution shift is large; cos(Y2_native, Y2_collapsed) reflects "the algorithm depends on having zeros in X1," not a more fine-grained "third state carries information." The Gate II semantics is still meaningful but blunter than ideal.

### D6 — Workload is small (M=8, P=8); per-seed variance probably substantial

Per-seed values fluctuate (e.g., cfg 0 cos_L1: 0.58, 0.65, 0.74, 0.74, 0.76 across 5 seeds). The 5-seed mean smooths but doesn't quantify uncertainty in the verdict. Standard deviation should ideally be reported alongside the mean. The closeout currently shows means only.

### D7 — No unit tests for measurement math

Entropy and cosine similarity are simple but unverified. A small test that feeds known distributions and checks the formulas would harden the result.

### D8 — "Highest-gap" framing is interpretation-dependent (already in closeout's honest concerns)

The closeout flagged this. After C1 remediation, the framing might shift entirely if the new L4 measurement looks different.

### D9 — Cross-layer interactions not measured

Each Gate II treats one layer's collapse independently. Real consumers might experience compounding effects (collapse L1 AND L4 simultaneously). The audit's per-layer values don't extrapolate to multi-layer collapse.

### D10 — L5 is genuinely uncovered

Already documented. The audit says nothing about cross-exp accum. Track C in the closeout addresses this.

## Severity classification

| ID  | Concern | Severity | Action |
|-----|---------|----------|--------|
| C1  | L4 collapse no-op artifact | **CRITICAL** | Remediate + re-run |
| D1  | L3 = L1+L2 simultaneous | DOC | Note in closeout |
| D2  | L1 collapses both layers' weights | DOC | Note in closeout |
| D3  | PRNG state shared | LOW | Could refactor; not biasing means |
| D4  | Realism gate trivial | DOC | Already noted |
| D5  | Distribution shift in L2/L6 collapse | DOC | Note interpretation caveat |
| D6  | Small workload, variance unreported | LOW | Add SD to closeout |
| D7  | No unit tests for math | LOW | Could add but not blocking |
| D8  | "Highest-gap" interpretation | DOC | Already noted; revisit post-C1 |
| D9  | No multi-layer interactions | DOC | Note in closeout scope |
| D10 | L5 deferred | DOC | Already in Track C |

## Remediation plan

1. **R-G1 (C1):** Rewrite L4 collapse to use the override-after-ternarize semantics. Re-run audit. Update closeout with new L4 finding.
2. **R-G2 (D1, D2, D5, D6, D9):** Add interpretation caveats to the closeout's "honest concerns" section and per-layer narratives.
3. **R-G3 (D3):** Optional — refactor to per-layer RNGs. Defer unless C1's fix changes the picture in a way that depends on RNG independence.

If R-G1's re-run shows the L4 finding HOLDS under the fixed collapse, the closeout's headline stands. If it shows L4 is actually load-bearing under the fixed collapse, the headline shifts (likely toward "L3 sparse-regime is the highest-gap-but-load-bearing layer" or "no layer is dramatically under-exploited").
