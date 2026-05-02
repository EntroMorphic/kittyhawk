---
cycle: gesh_phase_a1_redteam
phase: CLOSEOUT
date: 2026-05-02
scope: post-build red-team of Gesh Phase A.1 (forward pass + bench + bank); 13 findings; 10 remediated, 3 explicitly deferred
companions: gesh/src/gesh_{bank,forward}.{c,h} · gesh/tests/test_*.c · journal/gesh_design_*.md
status: COMPLETE — H/M-tier remediated, L1/L2/L6 deferred with rationale
---

# Gesh Phase A.1 red-team

Adversarial pass over the Phase A.1 build. Same pattern as the m4t kernel red-teams: pre-Phase-A.2-training, sweep for issues that would compound during the lattice-update training loop's thousands of forward-pass calls.

13 findings (2 high, 5 medium, 6 low). H + M-tier remediated; L1/L2/L6 explicitly deferred with documented rationale.

## High-severity findings

### H1 — Aliasing assertions missing in `gesh_forward_classify` (FIXED)
The substrate's writable-output kernels all assert `dst != src`. The Gesh forward pass inherits the substrate's calling conventions; it should inherit the discipline pattern. Phase A.2's training loop will call this kernel thousands of times per epoch — silent corruption from aliased buffers would surface as garbled training, hard to debug.

**Fix:** added asserts that `out_predictions` doesn't alias `queries`, `bank->tiles_packed`, or `proj->R` (when non-NULL).

### H2 — `n_classes` derivation silently assumed dense non-negative labels (FIXED)
`n_classes = max(labels) + 1` works for the current bank constructor (`labels[c] = c`) but would silently misbehave if a future constructor used sparse labels, sentinel `-1`, or non-zero base.

**Fix:** added `assert(bank->labels[t] >= 0)` per tile in the forward pass; documented the dense-from-zero label convention in `gesh_bank.h` so future bank constructors know to honor it (or update both this assert and the n_classes derivation).

## Medium-severity findings

### M1 — Dead variables (FIXED)
`class_counts` in `gesh_bank_build_class_mean` (allocated, incremented, never read — leftover from a draft that did weighted averaging) and `n_classes_seen` in `gesh_forward_classify` (declared, silenced via `(void)`). Both removed; behavior unchanged. `-Wunused-but-set-variable` would have caught these if enabled.

### M2 — `sig_dim > 0` assert added (FIXED)
The forward pass asserted `top_k > 0 && top_k <= bank->n_tiles` and `bank->sig_dim == proj->sig_dim` but not `sig_dim > 0`. With `sig_dim == 0`, the mask construction degenerates and downstream behavior is undefined.

### M3 — Determinism test added (FIXED)
Pattern from the m4t kernel red-teams: every kernel needs a `prop_*_determinism` test. Two parallel calls on identical inputs must produce bit-identical outputs. Gesh forward pass now has this test; it confirms the function carries no hidden state across calls.

### M4 — Aliasing safety test added (FIXED)
Verifies that the canonical non-aliased call works correctly with deliberately-separated allocations. The forbidden-aliased case is enforced by H1's assert in debug builds; the test covers the legitimate path.

### M5 — `n_queries == 0` edge case test added (FIXED)
Same pattern that caught a real bug in the cross-exp kernel (the `n == 0` branch updating state spuriously). Phase A.1's forward pass returns 0 cleanly without modifying `out_predictions`; the test verifies via a sentinel value.

## Low-severity findings

### L3 — Class-balance test tolerance tightened (FIXED)
Was ±25%; now ±15%. With deterministic seed and N=2000, C=10, the tolerance covers ~2.2σ — well above any plausible drift. Tighter gate, same robustness.

### L4 — README/code drift on `m4t_route_threshold_extract` (FIXED)
The README documented the forward pass as using `m4t_route_threshold_extract` after the projection. The actual code uses inline `(acc > 0) ? +1 : (acc < 0) ? -1 : 0`. README updated to match the code; the inline sign-extract is functionally equivalent to `threshold_extract` with `tau = 0`.

### L5 — `gesh_bank.h` "future variants" clarification (FIXED)
Was ambiguous about which Phase introduces which variant. Updated to: Phase A.2 adds no new bank constructors (training operates on `R`, not the bank); Phase B+ may add k-means / PCA-derived / learned bank, each gated on a measured failure mode.

### L1 — Per-call malloc storm in `gesh_forward_classify` (DEFERRED)
Six heap allocations per call. Persistent-scratch variant for tight inference loops is future work. **Rationale for deferral:** Phase A.2's training loop calls `gesh_forward_classify` once per "evaluate-this-flip" — bounded per-step alloc count is fine; the perf impact is below the noise floor for a research probe. Promote to substrate scratch when profile shows the allocator in the hot path.

### L2 — `topk_smallest_indices` heap-allocates `buf_d` (DEFERRED)
Same rationale. Top_k bounded small (1–5 typical); could be stack-allocated. Defer to a perf cycle.

### L6 — No PCA-init projection test stub (DEFERRED)
Phase A.2 will add PCA-init alongside lattice-update. Test stub would mostly be a placeholder; cleaner to ship the real test with the real implementation. Tracked in Phase A.2 scope.

## Methodology meta-finding

This is the **fifth red-team in this codebase.** The substrate red-teams (xexpo, matmul) caught aliasing-assertion gaps; the Gesh red-team caught the same gap pattern in consumer code, despite the discipline being established at the substrate layer. Discipline transfer across architectural layers is INCOMPLETE — patterns established in substrate code don't automatically propagate to consumer code without an explicit checklist item.

**Remediation in CONTRIBUTING.md:** added a new entry to the post-commit doc-currency checklist: "Aliasing assertions on every writable output." Same idea as principle 7's "specs upstream of designs" but applied to consumer code: substrate discipline patterns are upstream of consumer code, and consumer code must inherit them deliberately.

The five red-team passes have surfaced this pattern repeatedly: each one is "the same kind of thing as last time, in a new context where the discipline didn't auto-transfer." Methodology drift across context shifts is a real failure mode; explicit checklists are the mitigation.

## Test surface growth

- `test_gesh_forward`: 4 → **7 tests** (added determinism, aliasing-safety, n_queries=0).
- `test_synth_proto`: tightened class-balance tolerance ±25% → ±15%.
- Total ctest binaries unchanged at 11; coverage hardened.

## Build state

- 11/11 ctest binaries green from clean rebuild under `-Werror`.
- Two new asserts in `gesh_forward_classify` (aliasing + label positivity).
- One new assert (`sig_dim > 0`).
- Two dead variables removed.
- README + `gesh_bank.h` docstring drift corrected.
- CONTRIBUTING.md methodology checklist extended.

## Loop-back triggers

- **Back to Phase A.1 NODES** if Phase A.2's lattice-update training surfaces additional latent issues in the forward pass that the red-team missed.
- **No loop-back** if Phase A.2 proceeds without forward-pass changes. The remediation closes Phase A.1's verification gap.
