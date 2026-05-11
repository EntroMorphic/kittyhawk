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

### TD-20 — Substrate quality degradation vs HF on reasoning/code/structured tasks (CLOSED 2026-05-10)

**Source:** `journal/inference_battery_v2_2026-05-09.md` (red-team finding).
**Original state:** expanded 24-prompt battery showed 5 substrate failures
(reason_word, math_div, code_loop, code_comment, json_format). HF handled
4 of 5 correctly.
**Resolution:** two-stage retuning landed all 5. Stage 1
(`journal/hp_sweep_2026-05-10.md`): `BITNET_GATE_ACT_BX = 1` recovered
4/5 (reason_word, code_loop, code_comment, json_format). Stage 2
(`journal/math_div_atomics_2026-05-10.md`): atomics investigation of
math_div confirmed it was noise-accumulation rather than a single-kernel
bug; testing `gate1 + fudge=2` (the untested combination from TD-22)
recovered math_div ("12" direct) plus the gate1 regression
(factual_hamlet), bringing strict pass rate to ~22/24 (~92%). Both
defaults updated.

### TD-27 — Why does substrate-routed top-k outperform oracle top-k? (CLOSED 2026-05-10)

**Source:** `journal/loop_regularizer_atomics_2026-05-10.md` open follow-ups.
**Resolution:** `journal/td27_mechanism_2026-05-10.md` (commits in this
patch series). H2 confirmed: oracle's `|Q·K|` selection metric is
direction-blind, "wasting" sparsity budget on high-|negative-score|
positions that softmax suppresses anyway. Added a "posracle" arm
that picks top-k by SIGNED score (not |score|); on a 10-prompt
focused subset at k=4, posracle = 7/10 (vs oracle 3/10 vs routed 8/10).
Just changing the sort key from |score| to signed score recovered 4
of 5 oracle failures. The substrate-distinct contribution is
**direction-awareness as a native representation property** — trit
signatures encode direction (sign + zero) by construction, so signature-
distance selection is direction-aware automatically. Base-2 can
implement direction-aware sparse attention via posracle-like rules;
the substrate's contribution is making it native to the representation.
The remaining ~1-prompt routed > posracle gap (8/10 vs 7/10) is small
enough to be noise at n=10; could be H1 (representation robustness)
or H3 (head-coordination) but not pursued.

### TD-23 — Manual classification of Cycle 2's 456 outputs

**Source:** `journal/cycle2_full_battery_findings.md` honest caveats.
**State:** Cycle 2 full battery (24 prompts × 4 arms × 6 k = 456 runs)
used the loop heuristic for preliminary pass-rate counting. The
heuristic has known false positives and false negatives (per
`journal/math_div_atomics_2026-05-10.md`); manual strict-pass
classification of every output would refine the absolute numbers.
**Impact:** the EVIDENCE pattern (routed > random; gap widens with
sparsity) should survive manual classification, but the absolute
pass rates will shift. The pre-commit gate verdict (PASS) is unlikely
to change but the magnitudes will.
**Remediation:** ~2-4 hours of focused manual review reading each of
the 456 outputs and labeling ✓ / ⚠ / ✗.
**Priority hint:** medium. Refines the published numbers; doesn't
change the verdict.

### TD-24 — Compute-parity verification for Cycle 2

**Source:** `journal/cycle2_full_battery_findings.md` honest caveats.
**State:** Cycle 2 measured QUALITY (pass rates) at each (arm, k) but
not COMPUTE (wall-clock per token, FLOPs). Sparse attention SHOULD save
FLOPs at small k (linearly with k), but the implementation hasn't been
verified to deliver this. The "matched FLOPs" framing the synth chose
for Part-B operationalization needs empirical FLOP counting to land
cleanly.
**Impact:** without compute-parity verification, the Part-B claim is
"routed beats random on quality at small k" rather than the stronger
"routed beats random at matched compute."
**Remediation:** instrument the harness to count FLOPs per arm
(BitLinear ops, attention dot products, signature ops, distance batch
ops); run a wall-clock micro-bench on the sparse path; cross-check.
**Priority hint:** medium-high. Strengthens the Part-B claim
materially.

### TD-25 — Larger Part-B eval (1000+ prompts on standard benchmarks)

**Source:** `journal/cycle2_full_battery_findings.md` honest caveats.
**State:** The 24-prompt battery is small for "Part-B evidence"
claims. Standard NLP evals (MMLU, HellaSwag, etc.) use thousands of
prompts and have established difficulty / capability axes.
**Impact:** the Cycle 2 Part-B finding rests on n=24 prompts with
manual category selection. A larger eval would confirm or refute the
generalization.
**Remediation:** integrate a standard NLP eval harness with the
substrate; run dense vs routed at multiple k values; report per-task
breakdown.
**Priority hint:** medium. Solidifies an existing finding rather
than producing a new one.

### TD-26 — Cycle 3: routing-native attention with training (N1)

**Source:** `journal/partB_experiments_synth.md` Cycle 3 design;
`journal/cycle2_full_battery_findings.md` "Next moves."
**State:** The post-hoc Part-B test (N4) succeeded. The architectural
Part-B test (N1 = routing-native attention trained from scratch)
would test whether the gain amplifies under joint optimization.
**Prerequisites:** gradient kernels for the substrate primitives in
attention (bitlinear_scale_bx_backward, rmsnorm_bx_backward,
attn_v_combine_backward, route_topk_abs straight-through-or-REINFORCE
gradient); MTFP optimizer state representation; training loop
integration.
**Impact:** would establish whether Part-B is a post-hoc curiosity
or an architectural advantage. Big claim if it lands.
**Priority hint:** high (for thesis ambition); resource-intensive
(months of foundational kernel work).

### TD-22 — gate1 single-prompt regression + untested knob combinations (CLOSED 2026-05-10)

**Source:** `journal/hp_sweep_2026-05-10.md` Phase B.
**Resolution:** `journal/math_div_atomics_2026-05-10.md`. Atomics
investigation of the math_div holdout showed it was not a single-kernel
bug (per-layer ε grows ~5× per layer through compound noise; no localized
jump). Tested `gate1 + fudge=2` combination (untested in original sweep);
recovers math_div ("12" direct), recovers factual_hamlet ("Shakespeare
wrote Hamlet"), improves def_ml. Net +3/-0 vs gate1 alone (apparent
regressions on code_python / edge_question / code_comment all turned out
to be heuristic false positives — both produce valid output, just with
different repetition character). Strict pass rate ~19/24 → **~22/24 (92%)**.
Default updated: `score_shift += 2`.



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
