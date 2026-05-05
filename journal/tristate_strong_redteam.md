# RED-TEAM: strong-claim test on L1

Cold-eye review of `journal/tristate_strong_closeout.md` + `audit/b2b_matmul.{h,c}`.

## CRITICAL findings

### C1 — B2-B-honest is a strawman; structural-advantage claim is fragile

The cycle's headline verdict ("base-3 wins on cost by 3 ops/block") rests on enforcing that B2-B "honestly" decodes sign and mask SEPARATELY. The closeout acknowledges:

> A more aggressively optimized B2-B might collapse to base-3 op count (via unified LUT), in which case the comparison becomes "base-3 vs base-3 with relabeled bits" — tautologically a tie. The "honest" B2-B that decodes sign and mask separately is the meaningful comparison...

But this acknowledgement is buried in caveats. The honest implication: **a skilled implementer faced with "store {-1, 0, +1} in 2 bits/cell" would NOT decode sign+mask separately**. They would use a 4-entry TBL — exactly the same op shape as Path A. At that point, "base-3 ternary" and "B2-B with relabeled bits" are byte-for-byte identical kernels.

The cycle's verdict can be restated more honestly:
- **Base-3 wins vs honest-B2-B by 3 ops/block.** TRUE.
- **Base-3 wins vs optimal-B2-B (unified TBL).** FALSE — they're identical.

The strong claim's structural argument ("base-3 has a fundamentally cheaper decode") is therefore **contingent on enforcing a specific B2-B implementation**. It is NOT a structural property of the encoding itself.

This is the major finding. Severity: **CRITICAL**, because it changes the verdict's strength from "structurally proven" to "strawman-comparison-dependent."

**Remediation:** add a Path C — B2-B-optimal kernel using unified TBL decode. Verify it has identical op count and bit-exact equivalence to Path A. Update closeout to report the verdict as: "base-3 beats honest-B2-B; base-3 ties optimal-B2-B; the strong-claim structural argument is contingent."

### C2 — Without external ground truth, all-NEON cross-check is internal-only

The cycle's verification is "all three NEON kernels produce identical Y." 60/60 bit-exact PASS. Per the closeout:

> NEON-vs-NEON cross-check is sufficient verification when no scalar reference exists.

But: if all three kernels share a logic bug (e.g., wrong DUP_IDX, wrong shift amount), they'd all produce the same WRONG Y and the cross-check would still PASS. Internal consistency != external correctness.

The substrate has `m4t_ternary_dot_matmul_bt`, which is externally validated (its own ctest, `test_m4t_ternary_matmul_neon`, runs `m4t_ternary_dot_matmul_bt` against `m4t_mtfp_ternary_matmul_bt_scalar_ref` for bit-exact verification). Adding a cross-check between MY kernels and the substrate's matmul would provide external grounding without violating the no-scalar-reference rule.

Severity: **CRITICAL** for verification rigor, **HIGH** for verdict robustness (if my kernels are wrong, they're consistently wrong, and the structural argument is bunk).

**Remediation:** add cross-check against `m4t_ternary_dot_matmul_bt` (externally validated NEON kernel from libm4t).

## HIGH-severity findings

### H1 — Skip kernel firing rate not empirically counted

The closeout claims `P(all 16 masked) ≈ 0.6^16 ≈ 2.8e-4 → skip rarely fires → skip overhead always exceeds benefit`. This is theoretical. The actual firing rate on the audit workload is unmeasured.

If the firing rate differs from theoretical (e.g., due to PRNG correlations or row-level structure), the skip kernel's verdict could shift. Should count empirically.

**Remediation:** instrument `b2b_skip_matmul_neon` to count skipped blocks; report fraction. (Or do a one-shot count outside the perf path.)

### H2 — Wall-clock numbers may be size-dependent in unmeasured ways

The cycle ran K∈{64, 256, 1024}. Real LLM hidden dims start at 4096+. Wall-clock ratios change with K (1.21× at K=64 → 1.49× at K=256). Extrapolation to large K is unverified.

For headline regime claims, the K=256 → K=1024 trend is ~stable. But for realistic LLM dims, an additional sweep might surface different regimes (cache effects, op fusion limits, etc.).

**Remediation:** EITHER sweep K=4096 or document the K-range scope explicitly in caveats.

### H3 — Op-count weights all NEON ops equally; pipeline reality is different

Counting `tbl + ushl + and + tbl + sdot` as "5 ops" treats them as equal cost. On Apple Silicon (M-series):
- SDOT: throughput ≈ 1/cycle, latency 4 cycles
- TBL: throughput ≈ 1/cycle, latency 1 cycle
- USHL/AND/USHR/BIC/MUL: throughput ≈ 2-4/cycle, latency 1-2 cycles
- LDR/LD1R: throughput ≈ 2/cycle, latency 4 cycles

The 3-op gap between Path A and Path B-honest is in cheap ops (and, ushr, mul); the 3-op gap between Path B-honest and Path B-skip is in cheaper ops (addv, fmov) plus a branch.

Wall-clock corroborates the gap, but op count overstates the structural significance. Pipeline dispatch + latency hiding might amortize the gap below the count-ratio prediction.

**Remediation:** report wall-clock as the primary metric; op count as cross-check. (Currently the cycle gates on op count; consider reframing.)

## MEDIUM-severity findings

### M1 — Memory bandwidth not measured

For K=1024, N=64, the W matrix is 64*256 = 16KB packed bytes — fits in L1 cache. For real LLMs (K=4096+), W can exceed L1 and start hitting L2/L3, where memory bandwidth becomes the bottleneck and decode-op differences amortize.

Cycle scope is 16KB-bound; real-world generalization needs explicit caveat or a sweep.

### M2 — Random ternary doesn't have structured sparsity; skip benefit untested in its natural regime

Real BitNet weights have structured sparsity (specific rows mostly zero, layer-wise patterns). The skip kernel's potential win (full-block skip) only manifests with structure. Cycle's random workload doesn't surface this; skip is "always slower" by design.

To honestly evaluate skip's value, would need real or structured-synthetic weights. Out of scope for the random-workload audit, but worth flagging.

### M3 — "Structural" framing in commit message overpromises

Commit message says: "the substrate's base-3 ternary representation has a STRUCTURAL advantage over base-2 sign + mask for L1." Per C1, this is contingent on the strawman B2-B definition. A more accurate framing: "Base-3 wins vs the canonical sign+mask base-2 implementation; ties optimal B2-B."

## LOW-severity findings

### L1 — Both packings waste the 0b11 state

Base-3 packing wastes 0b11 (reserved). B2-B-honest packing also wastes 0b11 (mask=1, sign=1 is redundant with mask=1, sign=0 since both encode 0). Theoretical density floor for {-1, 0, +1} is log2(3) ≈ 1.58 bits/cell; both kernels are at 2 bits/cell. Parity holds, but neither is theoretically optimal.

### L2 — "Honest expectation" of regime-dependent verdict was wrong

The synthesis pre-committed: "Honest expectation: base-3 likely wins on dense regimes, B2-B-skip likely wins on highly sparse regimes (skip amortizes mask overhead)." Actual data: base-3 wins uniformly across all regimes. Skip never wins.

This isn't wrong — data wins over expectation — but the cycle's framing should reflect the surprise rather than the prediction.

## Severity classification

| ID  | Concern | Severity | Action |
|-----|---------|----------|--------|
| C1  | B2-B-honest is a strawman | **CRITICAL** | Add Path C (B2-B-optimal); revise verdict |
| C2  | No external ground-truth verification | **CRITICAL** | Cross-check vs `m4t_ternary_dot_matmul_bt` |
| H1  | Skip firing rate empirical | HIGH | Instrument + report |
| H2  | Wall-clock K-range unsurveyed | HIGH | Doc caveat OR add K=4096 sweep |
| H3  | Op-count uniform weighting | HIGH | Reframe gating; primary on wall-clock |
| M1  | Memory bandwidth unmeasured | DOC | Note in caveats |
| M2  | Random workload, no structured sparsity | DOC | Note in caveats |
| M3  | "Structural" overpromise | DOC | Revise commit narrative |
| L1  | Theoretical-optimal packing | DOC | Already noted |
| L2  | Regime-dependent expectation wrong | DOC | Reframe in honest concerns |

## Remediation plan

1. **R-G1 (C1):** Add Path C kernel — B2-B-optimal using unified TBL. Show op count == Path A. Verify bit-exact equivalence. Update closeout's verdict to reflect "base-3 beats honest-B2-B but ties optimal-B2-B."

2. **R-G2 (C2):** Add cross-check in `tristate_strong_bench` against substrate's `m4t_ternary_dot_matmul_bt`. Verify all our kernels match the externally-validated substrate output.

3. **R-G3 (H1):** Instrument skip kernel to count blocks skipped vs processed. Report empirically.

4. **R-G4 (H2 + H3 + M1 + M2 + M3 + L2):** Add caveats / reframing. Lower-priority documentation pass; can do alongside R-G1/R-G2/R-G3.

After remediation, re-run; if verdict changes, document. If verdict stands, document the strengthening from external grounding.

## Predicted outcome after remediation

- Path C will have ~7 ops/block (matching Path A) and produce bit-exact equivalent Y.
- External cross-check against substrate kernel will PASS (my kernels are correct).
- Skip firing rate will be ~0% (matching theoretical 2.8e-4 prediction).
- Verdict shifts to: "Base-3 wins vs honest-B2-B (3-op advantage); ties optimal-B2-B; the structural-advantage framing is conditional on the B2-B implementation choice."

This is a more nuanced verdict but still POSITIVE for base-3 in practice (an honest implementer who hand-codes B2-B with separate sign/mask decode pays the 3-op cost). The structural argument WEAKENS but doesn't disappear.

## What this red-team does NOT find

- No bugs in any of the three kernels (none observed in cross-check; assertions hold).
- No issues with the workload generator or PRNG.
- No issues with the precision verification (60/60 PASS is solid for internal consistency; just lacks external grounding).
- No issues with the commit/CI process.

The red-team is about INTERPRETATION (C1, C2, M3) and rigor (H1, H2, H3) — not about the cycle's correctness. The kernels work; the verdict's BREADTH was overclaimed.
