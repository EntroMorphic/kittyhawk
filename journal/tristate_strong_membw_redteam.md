# RED-TEAM: memory-bandwidth-regime addendum

Cold-eye review of `journal/tristate_strong_membw_addendum.md`.

## CRITICAL findings

### C1 — "Density advantage MANIFESTS" overclaim

The addendum's headline reads:

> "the density advantage MANIFESTS as kernel-cost reduction with increasing memory pressure"

But the data shows Path D **never beats Path A** within the tested K range. The ratio narrows from 1.95× → 1.16× — that's a **REDUCED PENALTY**, not a manifested advantage. Reduced penalty ≠ advantage.

Honest framing: "the density advantage's KERNEL-COST PENALTY narrows as memory pressure rises; full crossover (Path D < Path A) was not reached in tested regime."

The "manifests" framing leaks into the commit message and could be misread as "base-3 wins at memory-bandwidth-bound regimes." It doesn't — not in our data. The trajectory **predicts** crossover; the cycle didn't measure it.

**Severity: CRITICAL.** Verdict precision affects how the next-cycle scope is set. If readers think crossover was achieved, they'll wrongly assume base-3 has manifest kernel-cost advantage; the actual finding is "density-ceiling unconditional + kernel-cost penalty narrowing."

**Remediation:** reframe addendum + commit narrative to "trajectory toward crossover, not crossover itself."

### C2 — Cache-warming bias between consecutive kernel runs

The bench runs all kernels back-to-back on the same workload:
```c
ms_a    = run Path A REPS times
ms_b    = run Path B REPS times    /* cache is hot from Path A */
ms_skip = run Path B-skip REPS times
ms_opt  = run Path C REPS times
ms_sub  = run substrate REPS times
ms_5in8 = run Path D REPS times
```

Each kernel after the first finds W in L1/L2 from the prior kernel's run. The "memory-bandwidth-bound" framing assumes cold cache, but our measurements are warm-cache after the first rep.

For W=200KB at K=12800 (just exceeds L1 = 192KB):
- First rep of Path A: cold; reads from L2.
- Subsequent reps of Path A: warm in L1 (W fits if next rep doesn't evict).
- First rep of Path B: warm from Path A. NOT a fresh memory read.

This biases all "memory-bandwidth-bound" measurements toward warm-cache behavior. The 1.16× narrowing might partly reflect post-warm-up steady state, not actual bandwidth pressure.

**Severity: CRITICAL** for the validity of the bandwidth-bound interpretation.

**Remediation:** add cache-flush between kernels (touch a buffer larger than L2 to evict W and X). Re-run; report cold-vs-warm comparison.

### C3 — Tested regime never actually exceeds L2 (not DRAM-bound)

Apple M-series L2: 12-16 MB shared. At our largest config (K=51200, N=64), W = 800 KB. Well within L2. We never tested DRAM behavior.

The "trajectory predicts DRAM crossover" extrapolation is based on data that all happened in L1 or L2. We don't know whether DRAM behaves the same way; M-series unified memory architecture has unique characteristics.

**Severity: CRITICAL** for the strong-claim's claim of memory-bandwidth-regime relevance.

**Remediation:** add a config that pushes W beyond L2. Increase N (e.g., N=8192 with K=12800 → W = 25.6 MB exceeds 16 MB L2). Re-run; check if DRAM-bound regime confirms or refutes the extrapolation.

## HIGH-severity findings

### H1 — Standard deviation not reported

Bench reports MEAN ms only. Per-seed variance is unmeasured in the public output. Manual inspection shows CV ≈ 1% at K=51200 (low), but for runtime-budget-constrained K=12800, REPS=200 that may not hold.

**Remediation:** compute per-config SD; report in summary line.

### H2 — Substrate's 0.97-0.99× ratio is suspiciously close to 1.00×

The substrate has 4× the W bytes (8 bits/cell vs 2 bits/cell). At memory-bandwidth-bound regime, that should manifest as ~4× more cache pressure. Yet substrate is barely slower than Path A (0.97-0.99×).

Possible explanations:
- (a) Cache-warming bias (C2): subsequent kernels find W still in L2.
- (b) Fixed per-call overhead dominating: setup/teardown costs > the actual kernel cost difference.
- (c) Apple Silicon L2 bandwidth so high that 4× more bytes barely matters.

We don't know which. Without isolating these, the "substrate advantage collapses" claim isn't well-grounded.

**Remediation:** add cache-flush (addresses a); report SD (helps diagnose b); push beyond L2 (addresses c).

### H3 — Apples-to-apples comparison vs substrate is questionable

`m4t_ternary_dot_matmul_bt` delegates to `m4t_mtfp4_sdot_matmul_bt`, which has its own outer-loop scheduling, cache-prefetch hints, etc. Our hand-written kernels don't have these. The "0.77× substrate at L1-resident" might partly be substrate's better outer-loop scheduling rather than the no-decode advantage we attributed.

**Remediation:** note as caveat. To genuinely compare, would need to write a packed-W version of the substrate's outer-loop scheduling — significant effort. Out of scope for red-team.

## MEDIUM-severity findings

### M1 — Decode vs gather contributions to Path D's cost not separated

Path D's ~12 ops/16-cell breaks into:
- ~5 ops decode (vmovl, vmul-magic, vshr, vmovn, vqtbl per digit × 4 digits)
- ~5 ops X gather (vqtbl4q + vqtbl1q + vbslq per digit)
- ~2 ops shared (load + sdot)

We attributed Path D's penalty to "decode + strided gather" without separating which dominates. If gather is the bottleneck, a different X layout could reduce it. If decode is the bottleneck, only different packing helps.

**Remediation:** measure with X gather replaced by direct-load (which would give wrong outputs but show gather cost). Or skip — informative-but-not-essential.

### M2 — Cache topology assumptions are M-series-specific

CI runs may use different hardware (Apple M-series vs x86 vs ARM Linux). The "K=12800 exceeds L1" claim depends on which silicon the test is on. If CI's L1 is bigger (e.g., 256 KB), the regime crossover happens at different K. The tested regimes may not match the labels.

**Remediation:** add runtime cache-size detection? Or explicitly note in caveats. Likely doc-only.

### M3 — Per-config REPS scaling assumption

REPS scaled by 1/K to keep runtime bounded. But the per-iteration cost isn't strictly linear in K (bandwidth effects); the REPS scaling might give different statistical signal across configs.

**Remediation:** verify SD is bounded across configs; if it grows, increase REPS at large K.

## LOW-severity findings

### L1 — Bench config order is fixed
Sequential dependency between configs (cache state from earlier persists). Could randomize order or insert flush.

### L2 — No thermal/throttle check
Apple Silicon dynamically throttles. 12s total runtime is short, so throttling is unlikely, but no explicit check.

## Severity classification + remediation plan

| ID  | Concern | Severity | Action |
|-----|---------|----------|--------|
| C1  | "Manifests" overclaim | **CRITICAL** | Reframe verdict to "trajectory" |
| C2  | Cache-warming bias | **CRITICAL** | Add cache-flush between kernels |
| C3  | Never reaches DRAM-bound | **CRITICAL** | Add config N=8192 (W > L2) |
| H1  | SD not reported | HIGH | Compute + report per-config SD |
| H2  | Substrate ratio ≈ 1.00× | HIGH | Addressed by C2 + C3 |
| H3  | Substrate scheduling not matched | HIGH | Doc caveat |
| M1  | Decode vs gather attribution | DOC | Note as future work |
| M2  | Cache assumptions M-series specific | DOC | Note caveat |
| M3  | REPS scaling assumption | LOW | Verify SD bounded |
| L1  | Fixed config order | LOW | Cache flush addresses |
| L2  | No thermal check | LOW | Note caveat |

## Remediation execution

1. **R-G1 (C2):** Add `cache_flush()` helper that touches a 32 MB buffer (exceeds L2 on M-series); call between kernel runs. Each kernel starts cold relative to prior kernel.
2. **R-G2 (C3):** Add config K=12800, N=8192 (W = 25.6 MB exceeds 16 MB L2). DRAM-bound regime test. REPS=3 for runtime budget.
3. **R-G3 (H1):** Compute per-config SD alongside mean; report.
4. **R-G4 (C1):** Update addendum text to use "trajectory toward crossover" framing instead of "manifests."

After remediation: re-run, verify trends hold (or shift), update closeout.

## Predicted outcomes

- **R-G1 (cache flush):** ratios at K=12800 may drift slightly. The substrate's 0.97× ratio likely moves further from 1.00× (toward 0.85-0.95×) as warm-cache bias is removed. Path D's narrowing might be less dramatic but trajectorial direction preserved.
- **R-G2 (DRAM-bound):** at N=8192, K=12800 (W=25.6MB exceeding L2), expect to see PATH D actually CROSS OVER (< 1.0× of Path A) due to massive bandwidth savings (1.25× = 5.1MB less to fetch from DRAM per call). If crossover doesn't occur, the strong claim's bandwidth narrative is weakened; if it does, the trajectory's destination is confirmed.
- **R-G3 (SD):** likely shows variance is ~1-3% across all configs; minor noise floor.
- **R-G4 (reframe):** doc-only.

The DRAM-bound test is the most informative; if Path D wins there, the strong claim has empirical proof of the kernel-cost manifestation, not just trajectory.

## Status

Red-team identified 3 CRITICAL issues. Proceeding to remediation.
