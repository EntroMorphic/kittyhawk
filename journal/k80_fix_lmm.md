---
cycle: K%80 fix LMM
phase: ALL (RAW + NODES + REFLECT + SYNTHESIZE)
date: 2026-05-07
scope: design the patch to m4t_ternary_5in8_matmul_bt that eliminates
       its slow scalar tail when K%80 != 0. Surfaced via the rowskip
       cycle's bench data (rs_no_skip variant showed +4.9% on
       BitNet aggregate, all from tile-alignment side effect).
companions: journal/rowskip_kernel_synthesis.md (where this issue
            surfaced); commit b3f9eba (rowskip cycle close);
            m4t/src/m4t_ternary_matmul.c lines 499-672 (the kernel
            we're patching).
---

# K%80 fix — Lincoln Manifold Method cycle

## RAW

The dense kernel m4t_ternary_5in8_matmul_bt is fast at K%80==0 and
slow when K%80!=0. We learned this by accident, through the rowskip
v2 bench: when I padded K_compressed to the next multiple of 80
(originally as a fix to a regression), the bench showed +22%
speedup on K=6912 (K%80=32) BitLinears even when no rows were
actually skipped. The rs_no_skip control variant — which fabricated
W to have zero empty rows so rowskip's framework only added padding
without compressing anything — produced +4.9% aggregate vs dense.
That's a side effect of pure tile alignment.

Looking at the kernel: lines 506-619. K_aligned = K - (K%80). Tile
body runs k=0..K_aligned in 80-trit chunks. Then a "geometric scalar
tail" runs k=K_aligned..K trit by trit, multiplying each x[k] by a
decoded W trit (per-trit divide-by-POW3, modulo, conditional sign).
Per output cell, K%80 scalar ops. For K=6912 N=2560: 32×2560 = 82k
scalar ops per call. At ~2 cycles per scalar op (decode is roughly
that), 164k cycles ≈ 50 µs at 3.2 GHz. The full kernel for K=6912
runs around 270 µs. So the scalar tail is ~18% of total time.
Confirmed by the bench: rs_no_skip variant skipping the tail saves
22%, matching the estimate.

Why is the kernel like this? Per the comment "TD-1 (relaxation of
strict K%80+N%4 alignment): geometric scalar tail per project rule
(sub-block scalar tails are allowed)." So it's an explicit design
trade: keep the main body clean, accept a scalar tail for irregular
K. That's defensible when most callers have K%80==0 — but BitNet's
down_proj (K=6912, K%80=32) is exactly the case it doesn't cover
well. And down_proj is one of the heaviest BitLinears per token
(largest K).

The fix question is straightforward in spirit: process the boundary
80-trit tile through the same NEON path, with W and X zero-padded
beyond K. Bit-exact because zero-padded contributions are zero. The
trickiness is in the details: W_packed is allocated to exactly Kp
bytes by callers; reading past Kp is undefined. X_strided is
allocated to K5*5 = exactly K5*5 bytes; trits beyond K are
already zero-padded by the existing pre-permute loop, but the
buffer might be too small for the boundary tile's reads. So the fix
needs to either (a) extend the allocations, (b) use stack-local
buffers for the boundary tile, or (c) change the W_packed contract
to require padding.

Important: this fix would also make rowskip's main contribution
shrink. Currently rowskip's headline +6.12% is ~80% tile-alignment
+ ~20% pure row-skip. Fix this and the tile-alignment portion goes
to dense; rowskip drops to +1.55% (smart dispatch). That's fine —
it just sharpens the picture. The fix is the right thing to do
regardless of what it does to rowskip's narrative.

## Open questions

1. **Who pads — kernel or caller?** Caller-padded W_packed is
   cleaner (no per-call cost) but breaks the API contract: every
   existing caller would need to re-pack. Kernel-internal padding
   stays transparent but adds ~16N bytes of memcpy per call.
   For N=2560 that's 41 KB at memory bandwidth ≈ 1 µs. Negligible
   relative to the 50 µs we'd save. Both are acceptable;
   tradeoff is API surface vs internal complexity.

2. **What about the j_tail (single-output) loop's scalar tail?**
   Lines 663-669. Same issue, smaller impact (it only runs for
   N%4 outputs, e.g., 0-3 cells). Should the fix cover both, or
   only the main 4-j-cell tile body? Probably both for
   consistency — same approach scales.

3. **Are there other m4t kernels with K%80 scalar tails that
   would benefit from the same treatment?** A grep would tell.
   The principle is general: any NEON-tiled kernel with sub-block
   scalar tail is a candidate.

4. **How do we test bit-exactness?** Existing test
   test_m4t_ternary_matmul_neon already checks NEON vs scalar_ref.
   Need to extend with K values that hit every K%80 mod (1..79)
   to cover all boundary tile patterns. Also K<80 (entire kernel
   is boundary).

5. **Does the K=6912 win generalize across BitNet's actual call
   pattern?** Yes — every layer's down_proj is K=6912. Per inference
   token: 30 calls × 50 µs saved = 1.5 ms. Total per-token compute
   on substrate is ~28 ms (per the rowskip aggregate). So the
   per-call save × 30 calls = ~5% of per-token compute. Matches
   the rs_no_skip aggregate.

6. **Could the fix introduce a regression on K%80==0?** Only if we
   do something gratuitous. The fast path remains untouched if
   we conditionally enter the boundary code only when K%80!=0.
   Need a benchmark to confirm no degradation.

## NODES

- **N1 — Boundary-tile-only fix.** The main body (k=0..K_aligned)
  stays unchanged. Only the last NEON tile (k=K_aligned, processing
  [K_aligned, K_aligned+80)) needs special handling. Stack-local
  W buffer per j_cell, zero-padded past Kp. Cost: 4×16=64 bytes
  stack per j_tile, 1×16=16 bytes per j_tail iteration. Marginal.

- **N2 — X buffer extension.** X_strided is currently sized K5*5
  where K5 = (K+4)/5. The boundary tile reads up to
  X_d[d][K5_padded - 1] where K5_padded = K_padded/5. Need to
  malloc K5_padded*5 bytes instead of K5*5. Pre-permute loop
  already zero-fills past K; same loop just needs to fill more
  zeros. Trivial change.

- **N3 — W boundary-byte handling.** For boundary tile starting
  at k=K_aligned, we need 16 bytes of W per j_cell to cover trits
  [K_aligned, K_aligned+80). W_packed has bytes
  [j*Kp, j*Kp + Kp). Bytes available for the boundary tile:
  [j*Kp + K_aligned/5, j*Kp + Kp). Bytes needed: 16. If
  Kp - K_aligned/5 < 16, copy what's there to stack-local 16-byte
  buffer; zero the rest. The remaining trit positions (past K)
  have zero W trits, contributing 0 to the dot. Bit-exact.

- **N4 — Caller-pad alternative.** Change the contract: W_packed
  must be sized for K_padded trits (Kp_padded = (K_padded+4)/5
  bytes per row). Caller (encoder) zero-fills [Kp..Kp_padded).
  Kernel just trusts the caller and reads Kp_padded bytes. This
  eliminates the per-call stack-local W; cost is a one-time
  pack-time padding. Storage overhead: ≤9 extra bytes per row
  for K=6912 case = 23 KB per BitLinear. Negligible.

- **N5 — j_tail (single-output) parallel.** The j_tail loop
  (lines 630-672) has its own scalar tail. Same boundary-tile
  approach applies, with a single 16-byte stack buffer instead
  of 4. Should be patched together with the main body so the
  fix is complete.

- **N6 — Test surface.** Existing
  test_m4t_ternary_matmul_neon.c verifies NEON vs scalar_ref at
  some K values. To cover the boundary code, need K cases at
  every K%80 mod ∈ {1, 2, ..., 79}. Sweep would be 79 K values
  × representative shape. Plus K<80 (boundary IS the entire
  kernel) for K ∈ {1, 5, 10, 50, 79}.

- **N7 — Bench coverage.** Reuse /tmp/bench_rowskip_v2.c's
  rs_no_skip approach as a reference for the expected speedup.
  After the fix, dense kernel at K=6912 should match rs_no_skip
  performance (~0.215 ms vs current ~0.270 ms). Down_proj
  benchmark BEFORE → AFTER patch is the headline measurement.

- **N8 — Ripple effect on rowskip.** After this fix, rowskip's
  headline +6.12% drops to ~+1.55% (smart dispatch alone).
  rowskip becomes a marginal kernel kept only for L1 down_proj's
  43% case. That's a real consequence — the rowskip kernel is
  not obsoleted but its strategic value diminishes.

## Tensions

- **T1: API contract change vs internal complexity.**
  Option A (caller-padded W): breaks contract; every caller must
  update; one-time work; clean kernel.
  Option B (kernel-internal stack-local W): no contract change;
  per-call ~16N memcpy + extra alloc; slightly busier kernel code.

  Option A pays cost once at conversion time and runs faster
  forever. Option B keeps backward compatibility. The project
  has only one caller of m4t_ternary_5in8_matmul_bt today
  (BitNet harness via rowskip + direct), so the contract change
  is small in practice. But the m4t library is intended to be
  consumed by future projects; preserving stable contracts has
  value beyond the immediate change.

  **Resolution: B (kernel-internal) is the right default.** The
  per-call cost is small (1 µs vs 50 µs saved). Internal
  complexity is contained to one boundary tile branch. API
  contract preserved. If at some future point we have many
  callers, all converging on K%80!=0 use, we can add a
  caller-padded fast variant as a new entry point.

- **T2: scope creep risk vs methodical completeness.**
  The fix touches one kernel. Adjacent fixes (j_tail loop
  scalar tail, other m4t kernels with similar tails) are
  natural extensions. Including them = bigger PR but consistent
  treatment. Excluding = focused fix, but leaves visible
  inconsistency (why did we fix the main loop but not j_tail?).

  **Resolution: fix the entire m4t_ternary_5in8_matmul_bt
  function** (main body + j_tail). They share the scalar-tail
  pattern; fixing both keeps the kernel coherent.
  Other m4t kernels with similar tails are out of scope —
  separate audit / separate fix per kernel.

## REFLECT

**Why is the existing kernel slow on K%80!=0?**

Direct cause: the scalar tail is per-trit, with each trit doing
divide-by-POW3 + modulo + conditional-sign + multiply-add. ~2 cycles
per trit × 32 trits × 2560 outputs = ~50 µs per call on the K=6912
shape.

But that's not the whole answer. WHY is the tail scalar in the
first place?

→ Because the NEON tile body is structured around 16-trit chunks
× 5 digits = 80 trits. Sub-80-trit chunks don't fit the same
SDOT-tile structure cleanly. The original author chose a scalar
fallback for the "irregular" trits as the path of least resistance.

→ But that scalar fallback isn't actually irreducible. The
"irregular" trits could be processed through the same NEON tile
body if W and X are zero-padded past K. We just have to handle
the buffer sizing, which the original author punted on.

→ Why did they punt? Probably because: (a) most callers had
K%80==0, (b) extending allocations seemed like contract creep,
(c) the cost was acceptable for the use case at the time.

→ But for BitNet's down_proj (K=6912), it's not acceptable —
that's a structural slowdown of ~18% on one of the heaviest
BitLinears. We just hadn't measured it directly; we discovered
it through the rowskip side effect.

**Why didn't the rowskip cycle name this fix as the priority
work?**

Because rowskip's primary goal was exploiting BitNet's empty K-rows
for compute savings. The K%80 fix is orthogonal — it would help
ALL callers, not just rowskip. We caught it as a confound in the
rowskip bench (rs_no_skip variant), and the honest synthesis
called out that this is a separate, larger optimization
opportunity. But "out of scope" was the right move at THAT time
because we were committed to closing the rowskip cycle, and a
mid-cycle pivot would have muddied the analysis.

Now that we're at the K%80 fix decision, the question is: do we
build it? The data says yes — +4.9% across all 210 BitNet calls
per token, capturable via a focused kernel patch.

**Why should we trust the +4.9% projection?**

Because rs_no_skip variant directly measured it. The kernel
already saw what "padded K, no scalar tail" performance is — the
rowskip framework gave us that as a control. The patched dense
kernel should match rs_no_skip's numbers when K%80!=0; should be
identical to current dense when K%80==0 (no change).

There's a residual risk: rowskip's rs_no_skip path also includes
the gather overhead (X_compressed[c] = X[nonempty_idx[c]]). For a
no-empty-rows W, the gather is identity (nonempty_idx[i] = i),
but the compiler can't necessarily prove that and may still emit
the gather loop. This means rs_no_skip might be slightly slower
than a "pure tile-aligned dense" — meaning the fix could
deliver MORE than +4.9%.

Likely a few percent more. Won't know until benched.

## SYNTHESIZE

### Plan

1. **Patch m4t_ternary_5in8_matmul_bt** (file:
   m4t/src/m4t_ternary_matmul.c):

   a. Compute K_padded = ((K + 79) / 80) * 80 alongside K_aligned.

   b. Allocate X_strided to size K5_padded*5 instead of K5*5,
      where K5_padded = K_padded/5. Pre-permute loop already
      zero-fills past K — works as-is on the larger buffer.

   c. Restructure tile body: run k=0..K_padded step 80 (was
      k=0..K_aligned). For k < K_aligned: use existing W_packed
      directly. For k == K_aligned (boundary tile, only fires
      when K%80!=0): use 16-byte stack-local W buffer per j_cell,
      copied from W_packed[j*Kp + k/5 .. min(j*Kp + Kp, ...))
      with zero-fill.

   d. Delete the "geometric scalar tail" loop (lines 605-619).

   e. Same restructure for the j_tail (single-output) loop
      (lines 630-672). Same boundary handling, single stack
      buffer.

2. **Extend tests** (file: m4t/tests/test_m4t_ternary_matmul_neon.c):

   a. Add K values covering every K%80 ∈ {1..79} mod, at
      representative shape (e.g., M=1, N=64).
   b. Add K<80 cases: K ∈ {1, 5, 33, 79}.
   c. All bit-exact vs scalar_ref oracle.

3. **Bench** (new file: /tmp/bench_k80_fix.c):

   a. Pre-patch baseline: m4t_ternary_5in8_matmul_bt at K=6912 N=2560
      (BitNet down_proj shape).
   b. Post-patch: same kernel, same shape.
   c. Verify K%80==0 unchanged: m4t_ternary_5in8_matmul_bt at
      K=2560 N=2560 (BitNet q_proj shape).
   d. Real BitNet weights bench: re-run /tmp/bench_rowskip_v2.c
      and verify rs_no_skip ratio goes to ~1.0× (no longer faster
      than dense). The "tile-align bonus" should disappear into
      dense.

4. **Verify ASAN+UBSAN clean.**

5. **Update CONTRIBUTING.md** if it documents the geometric
   scalar tail rule. The rule still allows scalar tails — but
   we're removing this specific instance.

6. **Update rowskip's docs** (m4t/src/m4t_ternary_rowskip.h) to
   reflect that smart-dispatch with skip≥5% is the right policy
   (rowskip's tile-align bonus is now in dense itself).

### Risks

- **Bit-exactness regression**: any change to the inner loop is
  high risk. Mitigation: extensive K%80 sweep test, bit-exact
  vs scalar_ref oracle.

- **K=0 or K<80 edge cases**: K_aligned could be 0 if K<80.
  Need to ensure boundary-only path handles this without a
  separate code path.

- **Performance regression on K%80==0**: must keep the existing
  fast path bit-identical. Verify via bench: K=2560 BEFORE and
  AFTER patch should be within noise.

### Expected outcome

- BitNet aggregate: +4-5% per-token compute saved (matching the
  rs_no_skip measurement; possibly slightly more without gather
  overhead).
- rowskip's headline benefit drops to +1.55% (smart dispatch
  alone). Rowskip stays in libm4t but becomes marginal — kept
  primarily for L1 down_proj's 43% empty-row case.
- Other m4t kernels with similar K%80 scalar tails identified
  as candidates for the same treatment (separate audit).

### Decision

**Build the fix.** Per the data, this is a higher-value
optimization than rowskip itself (+4-5% vs rowskip's net +1.55%),
benefits all callers, and is contained to one kernel. The cycle
estimate: ~4 hours including tests and benches.
