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

### TD-4 — L4 strong-claim cycle (audit Track A weak-deepening)

**Source:** `journal/tristate_op_closeout.md` Track A; concern 1/2 from `journal/p0_concern1_mechanism.md`.
**State:** L4 was the audit's least load-bearing layer (mean cos ≈ 0.94 post-redteam-fix). Three operationalization candidates pre-named:
  - A.1 absmean-threshold ternarization (BitNet b1.58 rule)
  - A.2 zero-flag forwarding (track structural-zero events)
  - A.3 two-channel sign+magnitude split
**Unblocks:** evidence that L4's third state can be made load-bearing OR confirmation that it can't (informative either way).
**Priority hint:** medium. Settles whether L4's "least load-bearing" verdict is fundamental or fixable.

### TD-5 — L5 strong-claim (cross-exp accumulator)

**Source:** `journal/tristate_strong_closeout.md` Track C.
**State:** L5 (cross-exp accum) requires a residual-style workload not produced by GEMM-only. Audit didn't measure; strong claim doesn't extend automatically.
**Unblocks:** L5's third-state utilization measured; strong claim verdict for cross-exp arithmetic.
**Priority hint:** low-medium. Requires designing a workload that exercises cross-exp accum naturally.

### TD-6 — L6 strong-claim cycle

**Source:** `journal/p0_concern2_l2.md` "What this does NOT establish."
**State:** L6 (post-ternarization activations) parallel to L1/L2 in shape; verdict likely follows L1/L2 by structural symmetry but not directly measured.
**Unblocks:** verdict generalization across all packing-relevant layers.
**Priority hint:** low. Symmetry argument is strong; explicit measurement would harden the claim but isn't required.

### TD-9 — DRAM-bound regime test (push beyond L2)

**Source:** `journal/tristate_strong_membw_addendum.md`; `journal/tristate_strong_membw_redteam.md` C3.
**State:** tested up to W = 25.6 MB (exceeds L1 + L2 partially). True DRAM-bound (W > L2 = 16 MB on M-series, ideally > L3 if it existed) requires N=2048+ or K > 1M. Not tested.
**Unblocks:** Whether sub-2-bit base-3's density advantage manifests at true DRAM-bound regime (membw addendum showed plateau, not crossover, within tested range).
**Priority hint:** low. Apple Silicon's unified memory bandwidth is generous enough that decode cost dominates at all reachable workloads; true crossover may be hardware-specific (older ARM, embedded, or non-Apple ARM).

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
**Status:** unresolved. Image-canon benchmarks are base-2-framed; CIFAR-10 hits a representation tax base-3 doesn't close; Go-position phase classification was a strong base-3-native signal in the prior cycle.
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
