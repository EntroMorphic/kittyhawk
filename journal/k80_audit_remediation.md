---
cycle: K%tile + K=0 + assert-coverage audit remediation
phase: cleanup of all K%80/K%16 scalar tails + UB across m4t kernels
date: 2026-05-07
scope: address the audit candidates filed at the end of the K%80
       fix remediation (commit 8217548). Five items: K=0 UB in
       sibling kernels, m4t_assert_coverage failure, xpacked K%80
       scalar tail, mtfp4 K%16 scalar tail (3 sites).
companions: journal/k80_fix_lmm.md (design source for the boundary-
            tile pattern), journal/k80_remediation.md (where this
            audit list was filed), journal/k16_mtfp4_BEFORE.txt,
            journal/k16_mtfp4_AFTER.txt.
---

# K%tile audit remediation closeout

## Items addressed

### 1. K=0 NULL+0 UB in sibling matmul kernels

After fixing m4t_ternary_5in8_matmul_bt's K=0 path in commit 8217548,
two sibling kernels still had the same latent UB:

  m4t_mtfp_ternary_matmul_bt              (m4t_ternary_matmul.c:357)
  m4t_ternary_5in8_matmul_xpacked_bt      (m4t_ternary_matmul.c:874)
  m4t_mtfp4_sdot_matmul_bt                (m4t_mtfp4.c:30)

Each had `if (M == 0 || N == 0) return;` early but no K==0 guard.
For K=0 with M>0, N>0: NULL+0 pointer arithmetic on X / W_packed.
Mathematically correct (zero-term dot product = 0) but UB.

Fix applied to all three: early return with memset(Y, 0, ...) for K=0.

UBSAN halt_on_error=0 sweep across all 30 ctest entries: clean.

### 2. m4t_assert_coverage meta-test failure

The check_assert_coverage.sh script verified that
test_m4t_assert_live.c's cases[] enumerated every substrate .c with
an assert(). After adding m4t_ternary_routed16.c and
m4t_ternary_rowskip.c (rowskip cycle, commit 08369c6), the meta-test
failed with "sources missing from cases[]".

Fix: added violate_routed16() and violate_rowskip() functions that
trip K=-1 assertions, plus corresponding cases[] entries.

m4t_assert_live now passes (all 7 cases SIGABRT as expected).
m4t_assert_coverage now passes (all 7 sources accounted for).

### 3. xpacked K%80 scalar tail

m4t_ternary_5in8_matmul_xpacked_bt had the same scalar tail pattern
as the dense kernel. Same fix applied: extend NEON tile body to
K_padded, stack-local zero-padded W per j_cell for the boundary tile,
delete scalar tail. Same pattern in both j_tile (4-output) and
j_tail (1-output) paths.

Test extended: K%80 ∈ {1..79} sweep + K=0 explicit case.
Bit-exact vs scalar_ref + cross-equivalence vs §20 dense kernel.

Performance: not separately benched (use case for xpacked is when X
is also ternary — applies to gesh consumers, not BitNet which uses
A8 int8 X). Expected behavior: same shape as dense fix — K%80=0
unchanged, K%80!=0 collapses to K%80=0 baseline cost.

### 4. mtfp4 K%16 scalar tail (m4t_mtfp4_sdot_matmul_bt)

Same boundary-tile pattern applied at K%16 granularity. Both j_tile
(4-output) and j_tail (1-output) paths.

**Bench shows this fix is mostly neutral** — unlike the K%80 fix:

  Shape          BEFORE        AFTER         Delta
  -------------+-------------+-------------+--------
  K=2560 K%16=0 0.099 ± .008  0.097 ± .010  -2% noise
  K=2575 K%16=15 0.078 ± .002 0.083 ± .004  +6%
  K=6925 K%16=13 0.231 ± .035 0.231 ± .022  0%
  K=15  (K<16)  0.003 ± .000  0.011 ± .001  +260% (real, µs scale)
  K=1           0.001 ± .000  0.006 ± .001  +500% (real, µs scale)

Reason for the difference vs K%80 fix:
  - The eliminated K%16 scalar tail was just `acc += xi[k] * wj[k]`
    — one cycle per cell. Net cost ≤ 15 cycles/output.
  - Boundary-tile setup (5 memcpy + 5 vld1q + 4 vdotq) is roughly
    cost-equivalent. No net win.
  - The K%80 fix eliminated per-trit divide-modulo decode, which
    was 2-3× more expensive per cell than mtfp4's plain mul-add.

Decision: keep the patch (code-consistency with dense kernel,
bit-exact preserved, ASAN-clean) but **document the regression
bound honestly** in the kernel comment. Same disposition as the
K%80=1 regression in commit 8217548 — small absolute cost, on
shapes BitNet doesn't use.

For BitNet specifically, K%16 = 0 always (K=2560, 6912, 640), so
the boundary tile NEVER fires — patch is a no-op for production.

### 5. Element-wise mtfp4 conversion scalar tails (NOT FIXED)

Two more scalar tails in m4t_mtfp4.c at lines 232 (mtfp19→mtfp4)
and 274 (mtfp4→mtfp19). These are element-wise conversions, not
matmuls. The scalar tail processes ≤ 15 cells per call regardless
of n.

For typical n (= layer dim, 2560+), the tail is < 1% of total
work. Not worth fixing. Documented as out-of-scope.

## Test additions

  test_m4t_ternary_5in8_xpacked.c:
    - K%80 sweep (K = 160 + km, km ∈ {1..79})
    - K=0 explicit
  test_m4t_mtfp4.c:
    - K%16 sweep (K_base ∈ {16, 32, 48, 64} × km ∈ {1..15})
    - K<16 cases (K ∈ {1, 5, 8, 15})
    - K=0 explicit
  test_m4t_assert_live.c:
    - violate_routed16, violate_rowskip + cases[] entries

All tests pass under ASAN+UBSAN halt_on_error=0.

## Coverage summary

After this remediation:
  m4t_ternary_5in8_matmul_bt           K%80 fix (commit f2eea9f, 8217548)
  m4t_ternary_5in8_matmul_xpacked_bt   K%80 fix (this commit)
  m4t_mtfp4_sdot_matmul_bt              K%16 fix (this commit)
  m4t_mtfp_ternary_matmul_bt           K=0 fix (this commit; K%16 in
                                       ternary_dot_vmlal* helper still
                                       open — separate cycle since not
                                       in BitNet's hot path)

  ternary_dot_vmlal_x4 / ternary_dot_vmlal scalar tails:
    Filed for separate cycle. Not in BitNet's path (uses MTFP19
    activations rather than int8 ternary). Lower priority.

## What this teaches

The K%80 fix's headline win on the dense 5-in-8 kernel (+24%) was
specific to that kernel's heavy per-trit decode. Applying the same
pattern blindly elsewhere doesn't guarantee a win — the cost
structure of the eliminated scalar tail matters.

For mtfp4_sdot, the scalar tail was so cheap (~1 cycle/cell) that
boundary-tile NEON setup cost was a wash. The fix adds
code-consistency value but no perf value.

For xpacked, the scalar tail is similar weight to the dense kernel
(per-trit divide-modulo decode), so a similar speedup is expected
but not benchmarked here (xpacked isn't in BitNet's path).

The right framing: this audit gives every matmul kernel a uniform
"no scalar in production" structure. Some entries deliver speedups,
some don't. Both are honest data.
