---
cycle: V6 — routed16 removal
phase: deletion + archival reference
date: 2026-05-08
scope: closes V6 of pure-ternary architecture audit by removing the
       routed16 kernel. Architecturally non-conformant (binary
       indicator structure); empirically null (no BitNet operation
       reaches the 92% sparsity crossover); architecturally
       superseded (V1's _bt_route is the conformant path).
companions: memory/feedback_pure_ternary_routed_architecture.md
            (the directive), journal/pure_ternary_audit_2026_05_08.md
            (where V6 was filed), commit 960ee0e (the kernel's
            introduction — preserved in git history for archival
            reference).
---

# V6 — routed16 removal

## Why this kernel exists in git history

The routed16 kernel was the first attempt at "operationally routed"
ternary matmul on the substrate. It attempted to express ternary
sparsity as parallel binary indicator sets:

  uint8_t idx_pos[16];   // positions where W = +1
  uint8_t idx_neg[16];   // positions where W = -1

Each 16-trit window stored two binary subsets of K-positions; a trit
at position k is +1 iff k ∈ idx_pos, -1 iff k ∈ idx_neg, 0 iff
k ∉ (idx_pos ∪ idx_neg). The kernel used vqtbl2q_u8 gather to
selectively load X values per the active positions.

## Why it's removed

### Architectural

Per memory/feedback_pure_ternary_routed_architecture.md (2026-05-08),
the substrate's architecture is committed to:

  (1) Pure ternary
  (2) Routed
  (3) Non-dense
  (4) **No binary structures**
  (5) No scalar ops

routed16 violates condition (4). The idx_pos/idx_neg parallel sets
are binary indicators encoding ternary state — the structure
satisfies (1) information-theoretically (3 states from 2 bits with
"not both 1" constraint = log₂(3) bits/cell) but operationally
treats the data as binary set membership, not ternary dispatch.

### Empirical

Even setting the architectural commitment aside, routed16 never
delivered:

- BitNet weight sparsity caps at 50% (per
  journal/routed16_weight_structure.md). No BitLinear reaches the
  92% crossover where routed16 wins on K=6912 shapes.
- BitNet activation sparsity caps at ~87% per token (per
  journal/routed16_activation_sparsity_finding.md). 0/1680 samples
  cross the threshold.
- The kernel sat in libm4t as "infrastructure for an operation that
  may never appear" (per the synthesis at journal/routed16_synthesis.md).
  That operation hasn't appeared.

### Replacement

V1's m4t_ternary_5in8_matmul_bt_route is the architecture-conformant
dense ternary matmul. It dispatches per-cell on trit value via
mask + select (vceqq + vandq + vsubq + vaddlvq), no binary
indicators, no multiplication. Bit-exact to the original dense
kernel. ~2.85× slower; the cost of architectural conformance.

For sparse routing specifically, no BitNet operation requires it.
If a future operation ever needs 92%+ activation sparsity, the
routed16 design is recoverable via `git show 960ee0e:m4t/src/m4t_ternary_routed16.h`,
but it would need a ternary-native re-design before re-introduction.

## Removal scope

Files deleted:
  - m4t/src/m4t_ternary_routed16.h
  - m4t/src/m4t_ternary_routed16.c
  - m4t/tests/test_m4t_ternary_routed16.c

CMakeLists.txt: removed source from M4T_CORE_SRCS, removed test
target.

test_m4t_assert_live.c: removed include, violate_routed16 function,
and cases[] entry.

## What's preserved

- **Git history**: commit 960ee0e (kernel introduction), commits
  for the v1 kernel, red-team remediation, and rowskip cycle that
  used routed16 as a comparison point. All recoverable via git.
- **Journal entries**: 4 substantive analyses remain on disk
  documenting the kernel's design, atomics-level profile,
  weight-structure analysis, and activation-sparsity finding.
  These remain valid as empirical records.
- **Python script comments**: gesh/bitnet/scripts/* mention
  routed16 in comments referring to the journal entries. These
  are historical references, not code dependencies.

## Verification

  Build: clean (m4t target builds without routed16 source).
  Tests: 29/29 pass (was 30 — minus the removed routed16 test).
  ASAN+UBSAN: clean at halt_on_error=0 across all 29 tests.
  Dependencies: grep across .c/.h/CMakeLists/scripts confirms no
  remaining link-time references.

## V6 audit closed

The substrate no longer ships a kernel with binary indicator
structures encoding ternary state. The architecture-conformant
dense routing path is V1's _bt_route. Sparse routing is no longer
in the substrate's API — it can be re-introduced if and when a
sufficiently sparse operation appears in the model recipe.

## What this teaches

A kernel's empirical case can fail before its architectural
non-conformance is named. routed16 was empirically dead (no
operation crosses 92%) before the pure-ternary directive surfaced.
The directive made the architectural verdict explicit: the kernel
was always a structural mismatch, not just a low-yield optimization.

Removing it is the cleaner record. Keeping it as "infrastructure
waiting" was a hedge that the directive forecloses.
