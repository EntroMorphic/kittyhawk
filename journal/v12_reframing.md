---
cycle: V12 — re-framing the "non-dense" condition
phase: clarification (no code change)
date: 2026-05-08
scope: closes V12 of pure-ternary architecture audit by re-framing
       what condition (3) "non-dense" actually requires. The
       original audit overreached; this entry restores the
       directive's actual scope.
companions: memory/feedback_pure_ternary_routed_architecture.md
            (the directive), journal/pure_ternary_audit_2026_05_08.md
            (the audit V12 was filed in).
---

# V12 — what "non-dense" actually means

## Original audit framing (overreach)

The audit filed V12 as:

> V12 [foundational] All matmul kernels operate dense over the
>   (i, j) output grid. None of them skip output cells where the
>   corresponding W column is structurally zero or where activation
>   is zero.

This extended condition (3) "non-dense" from per-K-cell skip to
per-(i,j)-output-grid skip. That extension was not in the directive.

## What the directive actually says

From memory/feedback_pure_ternary_routed_architecture.md:

> 3. **Non-dense.** Compute must skip cells where the trit
>    determines no contribution. Multiplication that produces zero
>    ("multiply-by-zero is free in vectorized math") is dense
>    compute, not skip. The substrate's compute should not perform
>    work on cells that route to nothing.

The "cells" here are **per-trit cells in the dot product** — each
W[k, j] contributes via a routing decision on its trit value.
"Multiply-by-zero is free in vectorized math" specifically refers
to the SDOT path where zero-trit lanes still execute the
multiplication. The condition is satisfied when the per-cell
operation dispatches on the trit value (mask+select) rather than
multiplying through.

V1 through V4 satisfy this:
- Per-cell, the trit value generates a mask (vceqq).
- The mask selects between X and 0 (vandq).
- Zero-trit lanes contribute 0 because the mask is 0, not because
  anything multiplied by 0.

That IS per-cell-K routing. It is non-dense per the directive.

## What the directive does NOT specify

Output-grid density (per-(i,j) skip) is a SEPARATE concept the
directive does not mention. Whether/when an output cell should be
skipped is a different question:

- For BitNet, every output cell contributes to the next layer; no
  skip predicate exists.
- For a future operation with structural sparsity at the output
  level (e.g., MoE expert gating, top-k attention), per-output
  skip would be relevant — but it would require a new architectural
  primitive (a routing predicate at the output level).

## Resolution

V12 is **closed by re-framing**: the condition the audit named
"foundational density" is not in the directive's "non-dense"
condition. V1-V4 satisfy condition (3) at the level the directive
specifies (per-K-cell dispatch).

If a future architectural decision adds an output-level routing
condition, that becomes a new primitive request — not a
remediation of the existing kernels.

## Audit status after V12 closure

  V1  ✓ closed  (m4t_ternary_5in8_matmul_bt_route, commit 6a4b3de + 1766521)
  V2  ✓ closed  (m4t_ternary_5in8_matmul_xpacked_bt_route, commit 57d8900)
  V3  ✓ closed  (m4t_mtfp4_sdot_matmul_bt_route, commit 2e416e3)
  V4  ✓ closed  (m4t_mtfp_ternary_matmul_bt_route, commit 09416f4)
  V5  ✓ closed  (delegates to V3 — no kernel of its own)
  V6  ✓ closed  (routed16 removed, commit dc788dd's predecessor)
  V7  ✓ closed  (rowskip switched to _bt_route, commit dc788dd)
  V8  ✓ closed  (V4's helper rebuild eliminated the K%16 scalar tail)
  V9  filed but not addressed (≤15 cells of overhead per call;
                                negligible; not in BitNet path)
  V10 filed but not addressed (mtfp.c element-wise scalar tails;
                                same character as V9; geometric tail
                                rule per CONTRIBUTING.md still permits)
  V11 filed but not addressed (m4t_trit_pack.c scalar tails; same
                                character; geometric tail allowed)
  V12 ✓ closed  (re-framing — no code change required)

V9-V11 remain open as documented future work. The user's directive
prioritized the matmul-shape violations (V1-V8); element-wise
auxiliary tails are acknowledged but not in the directive's
critical path.

## Architecture state after this audit

The substrate's matmul kernels now have architecture-conformant
routing-shaped siblings (V1-V4). The kernels with binary indicator
structures have been removed (V6 routed16) or switched to use the
conformant path (V7 rowskip). The auxiliary scalar tails (V9-V11)
remain as geometric tails permitted by CONTRIBUTING.md.

The substrate is structurally aligned with the pure-ternary, routed,
non-dense, no-binary-structures, no-scalar-ops architecture for
its primary compute path. The harness still calls the multiplicative
kernels by default; switching the BitNet harness to call _bt_route
is a separate cycle (not part of this audit) — it commits BitNet
inference to the architecture at the documented ~2.5× per-token
wall-time cost.
