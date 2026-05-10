---
title: Technical Debt
status: index of deferred work, organized by source channel
companions: NORTH_STAR.md · CONTRIBUTING.md · docs/THESIS.md · docs/FINDINGS.md · CHANGELOG.md
---

# Technical Debt

Centralized inventory of deferred work across the project. Each item is a real follow-on cycle, doc drift, or scope-deferred decision — NOT speculative future ideas.

## How to use this doc

- **When to add:** any cycle that closes with "deferred", "future work", or "out of scope" should add an entry here, with a pointer to the source cycle.
- **When to remove:** when an entry's source cycle is unblocked OR when the item is explicitly cancelled (with a journal entry recording the cancellation).
- **Priority hint** is informational — drives discussion, not a strict ordering. The actual order of work follows consumer demand and project rule (no primitive without named consumer demand for layered consumer code; foundational primitives can land any time).

The journal cycles are the source of truth. This doc is an index for navigation.

---

## Functional gaps (consumer-visible)

### TD-21 — M4T_SUBSTRATE.md doesn't cover Phase 2 BitNet primitives

**Source:** `journal/saturation_audit_complete_2026-05-09.md` doc-currency
review (item #6 of post-RMSNorm-fix remediation).
**State:** the substrate spec's §17 cross-reference ends at §20 (sub-2-bit
packing). Phase 2 added a substantial set of bx-aware primitives
(`m4t_mtfp_rmsnorm_bx`, `relu²_bx`, `elementwise_mul_bx`,
`bitlinear_scale_bx`, `bitlinear_scale_no_a8_bx`, `m4t_mtfp_softmax`,
`m4t_rope_apply`, `m4t_a8_quantize`/`dequantize`, `m4t_mtfp_rescale_bx`,
`m4t_mtfp_vec_scale`, `m4t_mtfp_vec_dot_i64`, `m4t_mtfp_attn_v_combine`)
that aren't in §17 or any later section. They exist in the headers and
m4t/README.md, but have no spec-section anchor.
**Impact:** per CONTRIBUTING.md principle 7, kernels need to trace back
to spec constraints. The Phase 2 BitNet kernels currently float as
"consumer-driven additions" without spec backing.
**Remediation:** lightweight `journal/bitnet_phase2_spec_amend.md` cycle
adding §21 (or wherever) for "Phase 2 BitNet inference primitives" with
the bx-tracking convention as the load-bearing spec content.
**Priority hint:** medium. Pre-existing spec drift; not introduced by
the recent RMSNorm work. No production impact.

### TD-20 — Substrate quality degradation vs HF on reasoning/code/structured tasks (PARTIALLY CLOSED 2026-05-10)

**Source:** `journal/inference_battery_v2_2026-05-09.md` (red-team finding).
**Original state:** expanded 24-prompt battery showed 5 substrate failures
(reason_word, math_div, code_loop, code_comment, json_format). HF handled
4 of 5 correctly.
**Resolution (partial):** hyperparameter sweep
(`journal/hp_sweep_2026-05-10.md`) found `BITNET_GATE_ACT_BX = 1` (down
from 2) recovers 4 of the 5 failures. The original tuning was on the
pre-RMSNorm-fix substrate where residual magnitudes were 6.5× smaller;
post-fix, the wider real-space range of BX=1 became load-bearing for
preventing outlier-cell quantization-binning in the FFN gate²·up product.
Strict 24-prompt pass rate 63% → ~80%, zero hard failures. Default
updated. Math_div remains indirect (gives algebraically equivalent "12 ×
12 = 144" instead of direct "12") — likely model-capability quirk under
greedy decoding rather than substrate issue.
**Remaining open:** factual_hamlet regression and gate1+fudge2 untested
(see TD-22).

### TD-22 — gate1 single-prompt regression + untested knob combinations

**Source:** `journal/hp_sweep_2026-05-10.md` Phase B.
**State:** `BITNET_GATE_ACT_BX = 1` (gate1) recovers 4 of 5 substrate-
specific failures but introduces one regression: `factual_hamlet` went
from "Shakespeare wrote Hamlet" to "(Hint: It's a famous play by a famous
playwright." The Phase A sweep also found `score_shift fudge = 2` and
`BITNET_FFN_BX = 8` as separate winners (4/5 each); their combinations
with gate1 weren't tested.
**Remediation:** sweep gate1+fudge2, gate1+fudge2+ffn8 on the failure
subset; if any combination preserves the gate1 wins AND restores
factual_hamlet, adopt it. If no combination recovers both, accept the
factual_hamlet regression as worth the broader gain.
**Priority hint:** medium. Not blocking; the gate1 default is already a
net positive.



### TD-19 — Late-layer `block_output` saturation in BitNet inference (CLOSED 2026-05-09)

**Source:** `journal/saturation_audit_2026-05-09.md` (red-team revision).
**Resolution:** closed by `journal/act_bx_sweep_2026-05-09.md`. Swept
`BITNET_ACT_BX ∈ {6, 7, 8}` post-fix. Lowering bx fully eliminates
block_output saturation (BX=7: 0 cells; BX=6: 0 cells) but trades it for
end-to-end quality regressions: BX=7 introduces a loop on
`reasoning_color`, BX=6 also flips the argmax on math (12 + 7 = 20 instead
of 19) due to 1-trit precision loss. **Keep BITNET_ACT_BX = 8.** The
0.034% saturation fraction is absorbed by downstream RMSNorm; substrate
metric improved at lower bx but end-to-end metric did not — the
saturations weren't load-bearing for quality.



### TD-2 — MTFP19-X variant of 5-in-8 packed matmul

**Source:** `journal/m4t_5in8_closeout.md` "Honest scope."
**State:** only ternary-X variant shipped (`m4t_ternary_5in8_matmul_bt`).
**Unblocks:** consumers wanting MTFP19 activations × sub-2-bit packed weights (parallel to existing `m4t_mtfp_ternary_matmul_bt` which uses 4-in-8).
**Priority hint:** low. No consumer demand named.

### TD-3 — §20 has no live consumer

**Source:** `journal/m4t_5in8_closeout.md`.
**State:** kernel + pack/unpack primitives shipped; nothing in `gesh/` or future consumers uses them.
**Unblocks:** validation that the §20 primitives match a consumer's needs.
**Priority hint:** scope-deferred. Per project rule (foundational primitives don't gate on consumer demand), shipping the primitive is OK; demand-gated cycles handle the consumer wiring.

---

## Open follow-on cycles (research)

(All initially-listed research cycles closed in 2026-05-05 batch — see CHANGELOG and `journal/tristate_l4_strong.md`, `journal/tristate_l5_strong.md`, `journal/tristate_l6_strong.md`, `journal/tristate_dram_regime.md`. New entries appear here as future cycles surface deferred work.)

---

## Housekeeping (doc drift)

---

## Spec-level deferrals

### TD-13 — `M4T_SUBSTRATE.md` §14.4 status array allocation

**Source:** spec §17 cross-reference.
**State:** "No status array allocated until consumer requests." Cross-exp accum kernel supports `flags` parameter; allocator not in libm4t.
**Unblocks:** consumers wanting persistent status tracking; per-tensor SATURATED/ROUNDED metadata.
**Priority hint:** scope-deferred per project rule.

### TD-14 — Restore prior-cycle LUT generator

**Source:** spec §11 / §13 / `01MAY26_archived/m4t/tools/`.
**State:** "Returns with the consumer that demands them; the prior cycle's LUT generator is preserved in `01MAY26_archived/m4t/tools/`."
**Unblocks:** smooth-nonlinearity LUTs (GELU, softmax tables) for consumers needing them.
**Priority hint:** scope-deferred per project rule.

---

## Open questions (project-level)

These come from `docs/THESIS.md` and remain open after the substrate-claim work landed. Track them here for visibility; they don't fit the "deferred follow-on cycle" framing — they're directional questions for the consumer-layer rebuild.

### TD-15 — What benchmark is the substrate's right arbiter?

**Source:** `docs/THESIS.md` "Open questions for the consumer-layer rebuild."
**Status:** unresolved. Image-canon benchmarks are base-2-framed; CIFAR-10 hits a representation tax base-3 doesn't close. **As of 2026-05-09**, BitNet b1.58-2B-4T inference (`gesh/bitnet/`) runs end-to-end on the substrate — one candidate arbiter for Part A of the thesis, though its claim shape doesn't speak directly to Part B (routing). See `journal/inference_battery_v2_2026-05-09.md` for characterization.
**Implication:** the consumer-layer rebuild needs to pick its arbiter deliberately. Default to image canon would not be a substrate-claim win.

### TD-16 — Is SDOT load-bearing for the substrate-claim?

**Source:** `docs/THESIS.md`.
**Status:** unresolved. SDOT is the hot path; consumers exercising SDOT are the substrate-claim demonstrations. Without consumers, the question is academic.

---

## Methodology debts (process)

### TD-17 — Audit cross-check should run early in port-to-libm4t cycles

**Source:** `journal/production_shoring_redteam.md` M1; methodology lift from Item 2.
**State:** observed that Item 2's audit cross-check was added late. Lift documented but not enforced.
**Unblocks:** future "port-to-libm4t" cycles avoiding the same late-cross-check pattern.
**Priority hint:** low. Self-discipline; could be added to CONTRIBUTING.md's post-commit checklist.

### TD-18 — Tile fairness in kernel comparisons

**Source:** `CONTRIBUTING.md` already-added methodology note (from `journal/p0_kernel_opt_redteam.md` P0-3 C1).
**State:** rule documented but not enforced; relies on cycle-level red-team to catch tile asymmetry.
**Unblocks:** automated check would prevent inflated headlines like "3× faster" → "actually 1.8× apples-to-apples."
**Priority hint:** low. Requires building a comparison-fairness audit harness; current pattern of "run red-team after first execution" works.

---

## Status

This doc was created 2026-05-05 in response to a session-level housekeeping question. Items are extracted from journal cycle closeouts + recent red-teams. Not exhaustive; new items added as cycles close with deferred work.

When a new cycle defers work, add an entry here with `Source:` pointing to the cycle's closeout file. When an item lands, remove it from this doc and note the closing commit / cycle.
