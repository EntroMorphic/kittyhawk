---
cycle: pure-ternary architectural audit
phase: violation map
date: 2026-05-08
scope: per user directive 2026-05-08, the architecture is committed
       to pure ternary, operationally routed, non-dense, no binary
       structures, no scalar ops. This audit lists every place the
       current code violates these conditions.
companions: memory/feedback_pure_ternary_routed_architecture.md (the
            directive), recent kernel work that surfaced the gap
            (journal/cumulative_bench_2026_05_08.md and predecessors).
---

# Pure-ternary architecture audit — current violations

The five boundary conditions:
  1. **Pure ternary** — data semantics base-3 throughout
  2. **Routed** — per-cell dispatch on trit, not multiply-through
  3. **Non-dense** — skip where trit routes to nothing
  4. **No binary** — no binary indicator structures, no bit-packed
     binary encodings of trit data
  5. **No scalar ops** — already a project rule (reaffirmed)

Hardware-level binary (NEON instruction encodings, ALU ops) is the
substrate beneath the architecture, not in the architecture. Container
types (int8/int16/int32 holding trit-bounded values) are acceptable.
The violation is when the *architecture-level* code expresses ternary
semantics through binary operations or structures.

## Severity tiers

  S1 — load-bearing in BitNet's hot path (every inference token)
  S2 — exists in production code but not hot for BitNet
  S3 — internal/auxiliary, smaller exposure

## VIOLATIONS

### V1 [S1] m4t_ternary_5in8_matmul_bt — multiplicative compute (mult-by-trit)

  m4t/src/m4t_ternary_matmul.c, lines 487–763 (the dense matmul).
  Inner loop uses 5× `vdotq_s32` per j_tile per 80-trit chunk. Each
  SDOT multiplies through every cell (16 int8 × int8 → int32 acc).
  When trit = 0, the multiplication produces 0 but the cycle is
  spent. This is multiply-by-trit, not routed dispatch.

  Violations:
    (2) Routed — operation is multiply, not dispatch
    (3) Non-dense — every cell is processed, including zeros

  BitNet impact: this is the substrate's hot kernel. 26 ms / token
  end-to-end runs through this. Every BitLinear in BitNet inference
  hits this code.

### V2 [S1] m4t_ternary_5in8_matmul_xpacked_bt — same pattern (X-packed sibling)

  m4t/src/m4t_ternary_matmul.c, lines 884–1101.
  Identical structure to V1 — just adds a 5-in-8 X decode pass. Same
  vdotq_s32 SDOT inner loop, same multiplicative compute, same dense
  through-all-cells.

  Violations: (2), (3). Same as V1.

### V3 [S2] m4t_mtfp4_sdot_matmul_bt — same pattern at MTFP4 cell width

  m4t/src/m4t_mtfp4.c, lines 30–195.
  vdotq_s32 against ternary W with MTFP4 X. Mult-through every cell.

  Violations: (2), (3). Not in BitNet's path (BitNet uses int8 X via
  m4t_ternary_5in8_matmul_bt), but in the substrate's API.

### V4 [S2] m4t_mtfp_ternary_matmul_bt — vmlal_s32 multiplicative path

  m4t/src/m4t_ternary_matmul.c, lines 353–415.
  ternary_dot_vmlal / ternary_dot_vmlal_x4 helpers at lines 80–322.
  Decodes trit, sign-extends to int32, then multiplies via vmlal_s32.
  Every cell processed regardless of trit value.

  Violations: (2), (3). MTFP19-activations × packed-trit kernel.

### V5 [S2] m4t_ternary_dot_matmul_bt — wraps V3

  m4t/src/m4t_ternary_matmul.c, lines 338–351.
  Thin wrapper that delegates to m4t_mtfp4_sdot_matmul_bt. Inherits
  V3's violations.

### V6 [S2] m4t_ternary_routed16 (entire kernel) — binary indicator structure

  m4t/src/m4t_ternary_routed16.h, lines 67–82.
  m4t/src/m4t_ternary_routed16.c, lines 22–35.

  The tile struct stores ternary state as TWO PARALLEL BINARY SETS:
    idx_pos[16]: positions where trit = +1
    idx_neg[16]: positions where trit = -1
    (positions where trit = 0 are implicit — in neither set)

  This is binary indicator storage. Each position has two binary bits
  ("in pos?", "in neg?"), constrained to {(0,0), (1,0), (0,1)}, which
  encodes 3 states with bit-equivalent information — but the
  STRUCTURE is two binary sets, not one ternary value. Operations
  (`vqtbl2q_u8` gather, set membership) are binary-shaped.

  Violations: (1), (4). The "routed" framing in the kernel name is
  misleading: it routes via binary indicators, not via ternary
  dispatch.

### V7 [S2] m4t_ternary_rowskip — uses dense kernel internally

  m4t/src/m4t_ternary_rowskip.c, lines 188–210.
  Gathers X into X_compressed via `nonempty_idx`, then calls
  m4t_ternary_5in8_matmul_bt at compressed K. Inherits V1's
  multiplicative-compute violation. The "rowskip" is at the K-axis
  level (skipping all-zero rows of W); the per-cell compute remains
  multiplicative.

  Additionally: `nonempty_idx` is a binary indicator concept (which
  K positions are "non-empty"). It's an integer index list, not a
  binary mask, but the membership semantics are binary.

  Violations: (1) partial, (2), (3), (4) partial.

### V8 [S2] m4t_mtfp_ternary_matmul_bt scalar tail (ternary_dot_vmlal* K%16)

  m4t/src/m4t_ternary_matmul.c, lines 303–318 (in ternary_dot_vmlal_x4).
  Lines further down for ternary_dot_vmlal. Per-trit scalar mul-add
  for K%16 trailing trits. Filed for separate fix in
  journal/k80_audit_remediation.md but not addressed.

  Violations: (5).

### V9 [S3] m4t_mtfp4 element-wise conversion scalar tails

  m4t/src/m4t_mtfp4.c, lines 298, 340.
  Per-cell scalar tail in mtfp19→mtfp4 (n%4) and mtfp4→mtfp19 (n%16)
  conversions. Filed as "not worth fixing" in
  journal/k80_audit_remediation.md (≤15 cells, < 1% of work).

  Violations: (5). Small but present.

### V10 [S3] m4t_mtfp.c general scalar tails

  m4t/src/m4t_mtfp.c — multiple "Geometric scalar tail" comments
  (search confirmed at lines 313, 435, 735, etc.).
  Various MTFP19 element-wise ops: vec_zero, vec_add, vec_sub,
  shift3, rmsnorm, softmax, rope, vec_scale, rescale, etc.
  These are kernel infrastructure, not directly ternary, but they
  exist in production code paths and have scalar tails.

  Violations: (5). Project rule already permits "geometric scalar
  tail per project rule (sub-block scalar tails are allowed)" in
  CONTRIBUTING.md — but the user's new directive is stricter.

### V11 [S3] m4t_trit_pack.c scalar tails

  m4t/src/m4t_trit_pack.c, lines 157, 182, 448, 521.
  Pack/unpack scalar tails. The pack/unpack itself is base-3 encoding
  (acceptable per (1)), but the tail processing falls back to
  per-byte scalar ops.

  Violations: (5).

### V12 [foundational] All matmul kernels operate dense over zero cells

  Every matmul-shape kernel in m4t — V1, V2, V3, V4, V5 — produces
  output for every (i, j) cell of the output. None of them skip
  output cells where the corresponding W column is structurally
  zero or where activation is zero.

  Violations: (3) at the matmul level (not just per-trit).

  Note: the V6 routed16 attempted this at the K-axis level but used
  binary indicator storage (own violation). No kernel currently does
  per-output dispatch on a routing predicate.

## Counts

  S1 violations: 2 (V1, V2 — BitNet hot path)
  S2 violations: 6 (V3, V4, V5, V6, V7, V8)
  S3 violations: 4 (V9, V10, V11)
  Foundational: 1 (V12)

  Total kernels touching the violation set: ~13 distinct entry points.
  Lines of code affected (rough): the entire matmul.c (~1200 lines),
  most of mtfp4.c (~400 lines for the SDOT path), all of routed16
  (~270 lines), most of rowskip (~200 lines), and scattered tails
  across mtfp.c and trit_pack.c.

## What's NOT a violation (kept for clarity)

- **5-in-8 packing format** (m4t_trit_pack.c): genuine base-3
  encoding. byte = u₀ + 3u₁ + 9u₂ + 27u₃ + 81u₄ where uᵢ ∈ {0,1,2}.
  Each byte is a 5-digit base-3 number. Storage is base-3, not
  binary indicators. Acceptable per (1).

- **MTFP19 / MTFP4 / MTFP9 cell containers** (m4t_types.h): int8,
  int16, int32 containers holding trit-bounded values. Container
  binary, value ternary. Acceptable.

- **m4t_route.c primitives**: per the file, these implement actual
  routing decisions (threshold extract, top-k, etc.) on integer
  scores. The OPERATION is dispatch (select winners, route signals).
  These appear architecture-aligned. Need closer reading to confirm,
  but not flagged as violations on the audit pass.

- **Test scalar_ref oracles**: explicitly permitted by project rule.
  These are correctness gates for tests, not production paths.

## What this implies

The architecture as currently implemented is **substantially not the
declared architecture**. The substrate's hot kernels (V1, V2, V3, V4)
are multiplicative dense compute on ternary data — they produce the
right outputs but via operations the architecture excludes.

Closing the gap means rebuilding the matmul-shape kernels around
operational routing primitives. That's a deep, multi-cycle effort,
not a patch. Likely directions:

1. **Identify the ternary routing primitive on Apple Silicon.** The
   closest hardware approximation to per-cell ternary dispatch is
   `vbslq_s8` (bit-select per byte) which uses binary masks but
   selects values rather than multiplying. Selection between (X, 0)
   based on a positive-trit mask, plus selection between (-X, 0)
   based on a negative-trit mask, is closer to dispatch than
   multiplication — though the masks themselves are binary.

2. **Accept that hardware-level binary masks are the "binary substrate
   beneath the architecture"** the directive permits, and define the
   architecture-level ops in routing terms regardless of the underlying
   instruction.

3. **Or: accept that current Apple Silicon NEON cannot natively
   implement the architecture**, and treat the existing kernels as
   "executable substrate-equivalent for current hardware" while
   designing the architecture against a future predicated-SIMD
   target (SVE2, RISC-V V, etc.).

The directive forces a choice. The audit lays out where the choice
needs to be made.

## Recommended next steps (proposal)

1. Decide on the architecture/hardware boundary explicitly. The
   audit cannot proceed to remediation without knowing which
   interpretation of "no binary" is operative.

2. If interpretation is strict ("no binary structures at any level"):
   the substrate as built does not satisfy the architecture. We need
   either a different hardware target or an LMM cycle on what
   "ternary routing on binary hardware" means.

3. If interpretation is moderate ("architecture-level code must be
   ternary/routed; hardware primitives beneath are unavoidable"):
   the violations above need a restructuring pass — probably a
   ternary-routing kernel built with `vbslq_s8`-style per-lane
   selection rather than `vdotq_s32` multiplication, with bench
   to measure cost.

4. Revisit feedback_function_over_speed_no_scalar.md and
   feedback_substrate_claim_scope.md memories. The new directive
   layers on top; the existing rules remain valid but now have
   stricter peers.

This audit is a starting point for the architectural conversation, not
a remediation plan. The remediation plan depends on the boundary
decision in (1).
