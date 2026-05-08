---
cycle: K%80 fix — red-team remediation
phase: post-cycle correction + audit
date: 2026-05-07
scope: address red-team findings on the K%80 fix (commit f2eea9f).
       Tests, docs, methodology, and one real bug surfaced.
companions: journal/k80_fix_lmm.md (design), journal/k80_fix_closeout.md
            (initial cycle close), this commit (remediation),
            journal/k80_bench_*_5runs*.txt (paired multi-run data).
---

# K%80 fix — remediation closeout

## What the red-team found

1. **K=1 / K%80=1 regression** — bound but not documented in the kernel
2. **K=0 implicit handling** — no explicit test, latent UB exposed by adding one
3. **M>1 K%80 sweep** — implicit only via existing tail tests, not in my new sweep
4. **`avail > 16` dead clamp** — math says unreachable; either remove or document
5. **BEFORE/AFTER bench was single-run** — couldn't separate signal from drift
6. **Original headline numbers used lucky-low AFTER samples** — overstated the win
7. **xpacked + mtfp4 kernels have the same scalar tail pattern** — out of scope but unrecorded

## Remediations applied

### 1. K=0 fix (real bug)

The K=0 test triggered 6 UBSAN warnings: "applying zero offset to null
pointer" at lines 562, 566, 580-583. NULL+0 pointer arithmetic is UB
even though the values are never dereferenced. The original kernel
(pre-patch) had the same latent issue; I just exposed it by writing
the K=0 test.

Fix: early return for K=0:
  ```c
  if (K == 0) {
      memset(Y, 0, (size_t)M * (size_t)N * sizeof(m4t_mtfp_t));
      return;
  }
  ```
Now UBSAN-clean across all tests at halt_on_error=0.

### 2. Test coverage extended

  - K=0 explicit test with sentinel-fill detection
  - M>1 K%80 sweep at K ∈ {161, 200, 239, 337, 479, 4, 33, 79} ×
    M ∈ {2, 4, 8} × N=64

All bit-exact vs scalar_ref.

### 3. Dead clamp → assert with math justification

The boundary tile's `if (avail > 16) avail = 16` and `if (avail < 0)
avail = 0` clamps were defensive-but-unreachable. Math:
  K = 80q + r, r ∈ [1, 79], byte_off = 16q, Kp = 16q + ceil(r/5),
  so avail = ceil(r/5) ∈ [1, 16].

Replaced with `assert(avail >= 1 && avail <= 16)` + math comment.
Confirmed pass under -UNDEBUG (test build).

### 4. Regression bound documented in kernel comment

Added explicit performance characteristics block:
  K%80 == 0:        unchanged (fast path)
  K%80 ∈ [4..79]:   collapses to K%80=0 baseline (former scalar
                    tail eliminated)
  K%80 ∈ [1..3]:    boundary tile fires for 1-3 real trits — minor
                    regression (~5% at K%80=1)
  K < ~10:          slight absolute regression (~µs); not realistic
                    BitLinear shape

### 5. Paired multi-run bench (5 BEFORE + 5 AFTER)

Reverted to commit 8a938bc, ran 5 fresh BEFORE benches. Restored
the patch + remediation, ran 5 fresh AFTER benches. Same machine,
same compile flags, sequential within minutes.

  Shape                            BEFORE mean (σ)    AFTER mean (σ)    Δ
  --------------------------------+------------------+-----------------+--------
  K=2560 N=2560 (q/o, K%80=0)      0.0777 (.0005)    0.0789 (.0008)   +1.5% noise
  K=2560 N=6912 (gate/up, K%80=0)  0.2130 (.0033)    0.2145 (.0034)   +0.7% noise
  K=6912 N=2560 (down, K%80=32)    0.2656 (.0009)    0.2182 (.0035)   -17.8% ← real
  K=2560 N=640  (k/v, K%80=0)      0.0199 (.0003)    0.0199 (.0003)    0%
  K=2400 (K%80=0)                  0.0730 (.0009)    0.0737 (.0010)   +1.0% noise
  K=2401 (K%80=1)                  0.0733 (.0008)    0.0765 (.0010)   +4.4% ← regression
  K=2440 (K%80=40)                 0.1457 (.0011)    0.0770 (.0012)   -47.1%
  K=2479 (K%80=79)                 0.2240 (.0023)    0.0773 (.0010)   -65.5%
  K=2480 (K%80=0)                  0.0759 (.0009)    0.0761 (.0009)   +0.3% noise

### 6. Honest revisions to original claims

The single-run bench in journal/k80_fix_closeout.md overstated
gains. Corrected numbers:

  Original (single-run)                   →  Revised (5-run mean)
  -------------------------------------------+-------------------------
  Down_proj (K=6912): "+24.5%"            →  "-17.8% time" / "+22% speedup"
  K=2479: "+189%"                          →  "+190%" (consistent)
  BitNet aggregate: "+6.2%"                →  "+5.1%" (recomputed from
                                              30 × 0.0474 ms / 27.98 ms)
  K%80=1: "-5%"                            →  "+4.4% regression" (was
                                              within σ before, now σ-distinguished)

The conclusion stands (down_proj substantially faster, scalar tail
eliminated, K%80=0 unchanged) but specific numbers were ~30%
overstated due to single-run AFTER landing on a lucky-low sample.

### 7. Audit candidates filed

The same K%80 (or K%16) scalar tail pattern exists in 4 other
kernels:

  m4t_ternary_5in8_matmul_xpacked_bt     (m4t_ternary_matmul.c:886)
  m4t_mtfp4_sdot_matmul_bt               (m4t_mtfp4.c:98)
  m4t_mtfp4_*-secondary loops            (m4t_mtfp4.c:232, 274)

Each is a separate audit. Same fix pattern likely applies (extend
NEON tile to ceil(K/tile)*tile, stack-local zero-padded data for
boundary tile). Out of scope for this cycle — filed for future.

## What this teaches (again)

The methodical structure caught what single-run benching would have
hidden. The original close-out was directionally correct but
quantitatively overstated; remediation got numbers right.

The K=0 UB is a humbling find. The kernel "works" for K=0 in the
sense of producing correct output (empty loops + zero accumulator =
zero output), but pointer arithmetic on NULL is UB even at zero
offset per the C standard. Compilers haven't exploited it (yet).
The fix is trivial; the lesson is that "tested with K%80!=0" doesn't
test "tested with K=0."

The +4.4% K%80=1 regression is an honest blemish. In BitNet (K ∈
{2560, 6912, 640}, K%80 ∈ {0, 32}), K%80=1 never fires. Consumers
with that shape would pay ~4% extra for the new code path's
boundary tile setup. Documented.

## Status after remediation

  - Kernel correct: 7 ternary tests pass + new K=0 + M>1 sweeps
  - UBSAN clean across all ternary tests at halt_on_error=0
  - Multi-run bench: BitNet down_proj -17.8% (σ-distinguished from noise)
  - BitNet aggregate: +5.1% (revised from +6.2%)
  - Documented regression bounds in kernel comment
  - Audit candidates filed for separate cycles
