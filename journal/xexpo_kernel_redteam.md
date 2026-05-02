---
cycle: xexpo_kernel_redteam
phase: CLOSEOUT (adversarial review of the kernel build, with all findings remediated)
date: 2026-05-01
scope: post-build red-team of m4t_mtfp_vec_accum_aligning + wrappers; 14 findings; all remediated
companions: m4t/src/m4t_mtfp.{h,c} · m4t/tests/test_m4t_mtfp_accum_aligning.c · docs/DESIGN_X-EXPO.md · journal/xexpo_spec_amend.md
status: COMPLETE — all findings remediated 2026-05-01
---

# Kernel red-team — `m4t_mtfp_vec_accum_aligning`

Adversarial pass over the cross-exponent accumulator built earlier in the same session. 14 findings across high / medium / low severity. All remediated in the same commit cycle.

## High-severity findings

### H1 — §14.4 spec deviation: per-cell flags vs per-block (FIXED)
The original implementation used per-cell flag bytes (one `uint8_t` per cell). §14.4 specifies "1-byte status array per block." Migration: flag layout reorganized to one byte per MTFP19 block, with bit-packing per cell inside each byte (`bits 0-1` cell 0, `bits 2-3` cell 1, etc.). New helpers: `M4T_FLAG_BYTES(n)` for sizing, `m4t_flag_test()` for reading. Consumer code updates from `flags[i]` (cell-indexed) to `m4t_flag_test(flags, i, EVENT)`.

### H2 — round-to-nearest-EVEN vs round-to-nearest (FIXED)
§8.2 specifies tie-breaking ("ties go to the mantissa whose least-significant trit is 0"). The implementation relied on the invariant "ties don't occur because powers of 3 are odd" without enforcing it. Fix: `_Static_assert` on every `M4T_POW3_*` constant verifying odd-LSB at compile time, plus runtime `assert(s & 1)` in the round-divide helper. Documents the invariant; catches accidental misuse if anyone later substitutes an even divisor.

### H3 — Aliasing test was actually a determinism test (FIXED)
The original `prop_accum_aligning_aliasing` ran two parallel kernel invocations on **separate** buffers and verified bit-identical results. That tests determinism, not aliasing. Renamed to `prop_accum_determinism`. Added a real aliasing test `prop_add_dst_alias_a` that exercises the wrapper's `dst == a` path and verifies output matches the non-aliased path.

### H4 — Spec amendment without a journal cycle (FIXED)
The §14.2 status change (DEFERRED → IMPLEMENTED) and the §14.4 disambiguation were inline edits to `m4t/docs/M4T_SUBSTRATE.md` without a journal cycle. Per principle 7, spec amendments require a journal cycle. Wrote `journal/xexpo_spec_amend.md` (lightweight synthesize-only cycle, since the amendment documents an existing implementation rather than revising substrate semantics).

### H5 — "Genuinely floating in base 3" overclaimed (FIXED)
The CHANGELOG and m4t/README claimed the substrate is now "genuinely floating in base 3." True at per-tensor exponent granularity, false at per-block. Per-block exponent storage (§7's stated intent) is not built. Tightened wording to "floating-point in base 3 at per-tensor exponent granularity, with per-block deferred."

## Medium-severity findings

### M1 — Coverage holes in property tests (FIXED)
The original 6 properties relied on uniform random sampling, missing:
- Saturation-targeted distribution → added `rand_mantissa_near_max()` helper, used in `prop_accum_flags`.
- Boundary-value cases (M ∈ {0, ±MAX_VAL}, Δ ∈ {0, 1, 19, 20}, n ∈ {1, 4}) → added `prop_accum_boundary` with curated cases.
- n=0 no-op → added `prop_accum_n_zero` (and fixed a kernel bug it caught: `running_exp` was being updated even with n=0).
- Long-sequence stress → added `prop_accum_long_sequence` (200 sequences × K=256 calls).
- Trailing-block bits past n stay zero → added `prop_accum_partial_block`.

### M2 — Reference and kernel can share bugs (ACKNOWLEDGED)
The bit-exact correctness test compares the kernel against an `int64` reference written at the same time. If the mental model of round-to-nearest is wrong, both implementations could share the same off-by-one. Mitigations:
- The reference is structurally simpler (no early returns, no flag-helper inlining), so most kernel bugs would diverge.
- Compile-time + runtime asserts independently verify the odd-divisor invariant.
- Boundary tests with hand-derived expected outputs add an external sanity check.
This is a known limitation of single-implementation testing in C; full mitigation would require an independent (e.g., Python) reference, which is out of scope for this build.

### M3 — Reserved-bits test (RESOLVED BY H1)
The original property tests masked to bits 0-1 only, leaving bits 2-7 unverified. With the per-block layout (H1), all 8 bits of each flag byte are now defined. The remaining concern — partial trailing blocks — is addressed by `prop_accum_partial_block`.

### M4 — `out_e == NULL` in wrapper untested (FIXED)
Added `prop_add_out_e_nullable` — verifies the wrapper accepts NULL `out_e` and produces correct `dst[]`.

### M5 — `dst == b` in wrapper unenforced (FIXED)
The contract forbids `dst == b` but the original implementation had no enforcement. Added `assert(dst != b)` to both wrapper functions (`vec_add_aligning` and `vec_sub_aligning`). Debug builds now catch the violation; release builds inherit the documented UB.

## Low-severity findings

### L1 — Unclear header doc wording (FIXED)
The phrase "running may equal flags's underlying buffer iff the consumer passes flags as NULL" was opaque. Replaced with explicit aliasing rules and a clear contract section in the per-block layout description.

### L2 — `vec_sub_aligning` not built (FIXED)
Built `m4t_mtfp_vec_sub_aligning` as a sibling of the add wrapper, with two new properties (`prop_sub_via_negation`, `prop_sub_self`) verifying it equals `add(a, neg(b))` at the storage layer and that `sub(x, x)` at same exp produces zero.

### L3 — README NEON-acceleration framing misleading for scalar kernels (FIXED)
The m4t/README opening claimed NEON-optimized without distinguishing kernels that actually use NEON from those that are scalar. Added a clarifier paragraph: "build requires aarch64 + NEON, and most kernels use NEON intrinsics — but a few are scalar (notably the cross-exp accumulator, since ARM has no integer-divide). Each module's docstring states whether its hot path is NEON or scalar."

### L4 — `apply_signed`-as-degenerate-case overclaimed (FIXED)
The closeout had stated `apply_signed` is the same-block-exp degenerate case of the cross-exp accumulator. True for the *arithmetic* (the mantissa adds are identical), but `apply_signed` also handles `sign ∈ {-1, 0, +1}` dispatch and sentinel-skip routing semantics that the cross-exp kernel does not replicate. Tightened the claim in `m4t/README.md` and `docs/DESIGN_X-EXPO.md`: the cross-exp kernel generalizes the *arithmetic*, not the routing semantics.

## Bonus finding caught during remediation

While adding `prop_accum_n_zero`, the test caught a real bug: the kernel was updating `*running_exp` to `addend_exp` when `addend_exp > running_exp` even with `n == 0`. This violated the contract "no work means no state change." Fixed by adding `if (n == 0) return;` at the top of the accumulator, before any state changes.

A test that wasn't in the original 6 properties caught a kernel bug. The red-team's coverage expansion (M1) earned its complexity.

## Outcome

Build passes 6/6 ctest binaries. The accumulator test now runs **14 properties** instead of 6, with bit-exact comparison against an `int64` reference. The kernel ships with:

- Per-block flag layout (§14.4 spec verbatim).
- Compile-time + runtime guards on the odd-divisor invariant.
- Aliasing assertions on the wrapper's `dst != b` contract.
- A real aliasing test (not just determinism).
- n=0 fast-path with an early return.
- `vec_sub_aligning` as a sibling of `vec_add_aligning`.

All claims in the m4t/README, CHANGELOG, and substrate spec §14.2 are now traceable to compile-time guards, property tests, or both.

## Methodology note

This red-team pass was authored by the same agent that built the kernel, in the same session. Independent review by a separate reviewer would catch a different distribution of issues (perspective bias). The findings here are what same-author adversarial review surfaces; future external review may find more.
