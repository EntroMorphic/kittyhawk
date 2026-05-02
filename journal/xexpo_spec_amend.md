---
cycle: xexpo_spec_amend
phase: SYNTHESIZE (lightweight)
date: 2026-05-01
scope: amendment to m4t/docs/M4T_SUBSTRATE.md §14.2 + §8.2 status, prompted by direct kernel implementation
companions: docs/DESIGN_X-EXPO.md · m4t/docs/M4T_SUBSTRATE.md · m4t/src/m4t_mtfp.{h,c} · journal/xexpo_design_closeout.md · journal/xexpo_kernel_redteam.md
---

# Spec amendment — §14.2 implementation + §14.4 layout disambiguation

## Context

The cross-exponent accumulator was built ahead of the consumer-discovery cycle under owner authorization (the codified principle-5 reading: named consumer demand suffices, not measured cost). The build necessitates two amendments to `m4t/docs/M4T_SUBSTRATE.md`:

1. **§14.2 status:** DEFERRED → IMPLEMENTED with the kernel's actual semantics.
2. **§14.4 layout:** clarified to one-byte-per-block with explicit per-cell bit packing (the spec's "1-byte status array per block" was implementation-ambiguous; the kernel resolves it).

Per principle 7, spec amendments require a journal cycle. This is that cycle, executed in lightweight form because the amendment is documenting an implementation-driven clarification, not changing substrate semantics.

## Amendments landed

### §14.2 (Cross-block add policy)

- **Status:** DEFERRED → IMPLEMENTED.
- **Surface:** `m4t_mtfp_vec_accum_aligning` (canonical), `m4t_mtfp_vec_add_aligning` and `m4t_mtfp_vec_sub_aligning` (pairwise wrappers).
- **Alignment:** Path A (max-exponent target, smaller-exp side rescales). Path B explicitly rejected per `journal/xexpo_design_closeout.md`.
- **Rounding:** base-3 round-to-nearest-even, satisfied vacuously because divisors are odd. Compile-time `_Static_assert` on `M4T_POW3_TABLE` enforces the invariant; runtime `assert(s & 1)` in `m4t_pow3_round_div` is the fallback.
- **Saturation:** Case S per §8.5. Same-block-exp degenerate case behaves identically to `m4t_mtfp_vec_add_inplace`.
- **Storage granularity:** per-tensor exponent (one `int8_t`). Per-block exponent storage per §7 remains a separate kernel until a consumer asks.

### §14.4 (Exponent status tracking) — clarification

The phrase "parallel 1-byte status array per block" was implementation-ambiguous. Two readings:

1. The array has one byte per block of data (n_blocks total bytes).
2. The array has one byte per cell, organized into blocks (n total bytes).

The kernel implements reading 1, with each byte encoding two events × four cells:

```
bits 0-1: cell 0 of block — bit 0 SATURATED, bit 1 ROUNDED
bits 2-3: cell 1 of block — bit 2 SATURATED, bit 3 ROUNDED
bits 4-5: cell 2 of block — bit 4 SATURATED, bit 5 ROUNDED
bits 6-7: cell 3 of block — bit 6 SATURATED, bit 7 ROUNDED
```

`M4T_FLAG_BYTES(n) = ceil(n / 4)` sizes the array. `m4t_flag_test(flags, cell, event)` reads bits.

The original implementation used per-cell layout (one byte per cell). The kernel red-team flagged this as a spec-language deviation; this amendment ships the per-block layout and disambiguates §14.4 to make reading 1 explicit.

### §8.2 (Cross-block add) — status table

Updated from "Deferred (§14.2)" to "IMPLEMENTED — see §14.2 (round-to-nearest, named opt-in)."

## Why a lightweight cycle

The full LMM cycle (raw → nodes → reflect → synthesize) is for substantive *research* questions where the answer isn't yet known. This amendment is a documentation update prompted by code that already exists and is property-tested. The decision (per-block flag layout) was already made by the kernel rebuild post-red-team; this cycle records the spec-side reflection.

A heavier cycle would be appropriate if the amendment were *changing* substrate semantics rather than documenting them. Future spec changes that genuinely revise behavior should run the full four phases.

## Loop-back triggers

- **Back to a full cycle** if any consumer's call pattern reveals that per-tensor exponent storage is insufficient and per-block storage is needed. That's a real semantic change.
- **Back to a full cycle** if the round-to-nearest-even invariant is ever broken (e.g., a future kernel introduces an even divisor). The static assertion would catch this at compile time, but the design would need re-examination.
