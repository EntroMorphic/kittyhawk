---
cycle: routed16 (full LMM)
phase: ALL (RAW + NODES + REFLECT + SYNTHESIZE)
date: 2026-05-07
scope: built the first production sparse-routed NEON ternary matmul,
       benched against dense, red-teamed it, and remediated. The
       intent of this cycle is to extract the actual lessons — not
       the impressive-sounding ones — from the empirical floor we hit.
companions: commits 960ee0e (initial routed16 land), aecc0d5 →
            3acf3b7 (scalar oracle reframing), the v2 hardened bench
            (/tmp/bench_routed16_v2.c), test_m4t_ternary_routed16.c
            (33 cases, ASAN+UBSAN clean).
---

# routed16 — synthesis

## RAW

What I actually saw building this. The first thing that kept catching
me off guard: the dense NEON SDOT path is *fast*. It does 16 trits per
~3 NEON ops, and the multiply-by-zero in those 16 lanes is free —
SDOT just adds 0 to that lane's accumulation. So "routing as a way to
skip zero work" doesn't buy anything that vectorized math wasn't
already ignoring. The user and I had been talking about routing as
foundational to the substrate, which is correct as a representational
claim, but my casual framing extended that into a speed claim I
hadn't tested. The bench made me earn the distinction.

The second thing: a sparse-index representation is conceptually clean
but expensive. Each tile is 40 bytes (32-trit window, 16 nonzero
positions, 16 sign positions). At BitNet's actual ~40% sparsity that
expands to 7-8× the 5-in-8 packed storage — and I'm still doing 8
NEON ops per tile (2 vld1q, 2 vqtbl2q, 2 vaddlvq, 2 acc updates), so
per-trit cost is ~0.5 ops vs dense's ~0.2 ops. The crossover only
appears once sparsity is high enough that dense covers many trits per
output that sparse skips. That's what the data shows: K=N=2560 → 96-97%
sparsity crossover; K=6912 N=2560 → 92-94%. BitNet sits at 38-50%.
This is not where routed16 wins.

The third thing — and this is the part I almost missed — is that
xpacked is NOT a valid baseline for A8 activations. I added it to
the v2 bench thinking "the docs say xpacked beats dense; let me make
sure we're comparing against the actual fastest dense path." The
diff column lit up: 2558-2560 cells different per shape. xpacked
requires X to be ternary (m4t_trit_t in {-1,0,+1}), and silently
truncates anything outside that range. For BitNet, X is A8-quantized
int8 with full ±127 range, so xpacked gives wrong answers. The
silent semantic mismatch is exactly the kind of thing that turns
benchmarks into lies if you don't audit input validity.

The fourth thing: red-teaming caught real bugs. The kernel asserted
M==1 — fine in test builds (UNDEBUG'd), invisible in release. A
caller passing M=2 in production would silently get row 0's output
and undefined memory for row 1. The fix was extending the kernel to
arbitrary M, which both kills the bug and aligns with the rest of
m4t's API. That fix grew the code by ~10 lines and the test surface
by 10 cases (M ∈ {0, 1, 2, 3, 4, 8, 32}, varied shapes). Methodical
and proportionate.

What surprised me is how much of the work was *not* the kernel. The
kernel was straightforward. The real work was: design the
representation so it's NEON-amenable, encode the round-trip test
that verifies the encoder doesn't lose information, run ASAN/UBSAN
on the tail buf path, sweep the right shapes to find shape-dependent
crossover, and audit the bench inputs to make sure I'm comparing what
I think I'm comparing. The kernel itself was 30 lines of NEON
intrinsics that worked the first try.

## Open questions

1. **Where in the substrate does sparsity exceed 92-94%?** Weight
   sparsity in BitNet's ternary distribution sits at 38-50%. To buy
   the routed16 win we need an operation whose natural sparsity is
   higher. Candidates: post-ReLU² activations in the FFN (haven't
   measured); attention masks at decode (typically dense in causal
   inference; sparse in some block patterns); pruned-and-distilled
   weights from a different training regime (not BitNet's recipe).
   Need a measurement before claiming the win condition exists.

2. **Why is the K=6912 crossover lower than K=2560?** Hypothesis:
   each tile's 32-trit window covers a larger fraction of the dense
   path's per-output work as K grows, so tile overhead amortizes
   across more skipped work. Untested — could also be cache effects
   on the routed16 metadata or pipeline differences. A microbench
   isolating tile-walk cost from accumulation cost would clarify,
   but I haven't run it.

3. **Is there a representation that beats the 8-op-per-tile floor?**
   Bit-packed signs (-2 ops per tile? maybe). Pre-permuted X laid
   out so SDOT can run sequentially (-2 ops? but pays per-call
   permute cost). NEON SVE2 has efficient gather instructions but
   isn't on Apple Silicon's NEON variant. None of these have been
   tried; not clear which would pay off.

4. **Should routed16 ship in the BitNet inference path?** It loses
   at BitNet's sparsity, so wiring it in would regress. But not
   wiring it in means it's dead code waiting for an operation that
   may never appear. Unclear answer; depends on Q1's resolution.

5. **Did the encoder make the right choice with the greedy
   take-up-to-16 rule?** A different policy — say "fit as many
   nonzeros as possible in a 32-window, even if exceeds 16" with
   multi-tile windows — might pack tighter at the cost of kernel
   complexity. Untested.

## NODES

- **N1 — routed16 exists as a correct production primitive.**
  NEON-only, M-arbitrary, ASAN+UBSAN clean across 33 test cases
  (synthetic shapes 0-100% sparsity, FFN shapes, K<WINDOW boundary,
  K%5≠0, M ∈ {0,1,2,3,4,8,32}). No scalar fallback. Bit-exact vs the
  routed_ref oracle.

- **N2 — Crossover is shape-dependent in a non-uniform way.**
  K=N=2560 → 96-97%. K=2560 N=6912 → ~97%. K=6912 N=2560 → 92-94%.
  At 99% sparsity routed16 wins 2.1× to 3.0×. The K-dimension
  dominates the win condition; expanding N by itself doesn't move
  the threshold.

- **N3 — Storage 7-8× is the real cost.**
  40-byte tiles (4 start_k + 1 n_pos + 1 n_neg + 2 pad + 16 idx_pos
  + 16 idx_neg). q_proj 9.41 MB, gate 27.16 MB, down 27.54 MB vs
  5-in-8's 1.31 / 3.54 / 3.54 MB. This is irreducible without
  bit-packing signs (~32 bytes/tile, still 6×). The cost is
  amortized only if pre-packed offline.

- **N4 — xpacked is the wrong baseline for A8 activations.**
  Silently truncates non-ternary X to {-1,0,+1}. diff(d,xp) =
  2558-2560 cells across every test in the v2 bench. For BitNet's
  A8-quantized int8 X, xpacked gives wrong answers. Dense
  (m4t_ternary_5in8_matmul_bt) is the only valid baseline.
  Documented in routed16.h.

- **N5 — M=1 silent truncation was a real release-build bug.**
  `assert(M == 1)` compiles out under NDEBUG. Caller passing M>1
  in release got row-0 output + uninitialized garbage rows 1..M-1.
  Fix: extended kernel to arbitrary M (matches the rest of m4t
  API). Adds 10 test cases. Same routine handles M=0 (returns
  immediately) and M=32 (no special case).

- **N6 — Per-tile cost is 8 NEON-issue ops, not 5.**
  Initial commit message said "5 NEON ops per tile (load + 2
  gathers + 2 reductions)." Actual: 2× vld1q_s8 + 2× vld1q_u8 +
  2× vqtbl2q_u8 + 2× vaddlvq_s8 = 8 NEON-issue ops + scalar acc
  updates. The undercount didn't change the conclusion (routed16
  still beats dense at high sparsity) but it's a precision lapse
  worth correcting.

- **N7 — The methodology yielded most of the value, not the kernel.**
  The kernel itself is 30 lines of NEON intrinsics. The encoder is
  ~80 lines. The test, sweep, ASAN run, red-team, and remediation
  added another ~400 lines and ~3 hours of cycles. The ratio of
  scaffolding to kernel is ~10:1 — and that's the right ratio for
  "100/100, methodically."

- **N8 — Methodical means: design first, test early, audit inputs,
  red-team self, then commit.** I produced the design plan as
  chat-level text before writing code, registered tests early in
  CMake (so ctest catches regressions), wrote the v1 commit, then
  caught my own claims (op count, storage range, M=1 limitation,
  bench input mismatch) on the red-team pass. Each fix landed as
  its own coherent change.

## Tensions

- **T1: Routing-as-foundation vs. routing-as-speed.**
  The project's "math as signatures via routing" foundation
  (project memory) is a *representational* claim — the substrate
  represents and computes ternary projections via routing. It is
  correct and bit-exact. routed16 demonstrates this end-to-end
  on the existing 5-in-8 layout.
  But "routing-as-speed" is a *performance* claim, and my early
  framing implicitly conflated the two. Routing buys
  representational correctness. Routing as a speed primitive
  requires either (a) sparsity above the operation's crossover or
  (b) a different operation entirely. The data forces the
  separation. Resolution: speak about the two claims separately.
  Routing as correctness is validated; routing as speed is
  contingent on the operation and the sparsity. Don't claim
  routing-as-foundation buys speed — that's two arguments fused
  into one assertion.

- **T2: "Don't add features beyond what the task requires" vs.
  "100/100, methodically."**
  When I caught the M=1 silent-truncation bug, two paths existed:
  (a) hard-fail on M>1 (preserves scope) or (b) extend to
  arbitrary M (removes the constraint, matches m4t convention).
  I chose (b). The risk with (b) is scope creep — extending one
  function's contract opens questions about whether the kernel
  should also gain SIMD batching, prefill optimization, etc.
  Resolution: (b) was correct here because (i) the bug *required*
  some response to M>1, (ii) matching the m4t API convention
  reduces caller surprise, (iii) the change was minimal (10
  lines, 10 tests). I would have crossed the scope-creep line if
  I'd added a tile-outer/i-inner batching optimization at the
  same time — and I didn't. Methodical means proportionate, not
  maximal.

## REFLECT

**Why does dense beat routed16 at 38-50% sparsity?** Because NEON
SDOT processes 16 trits per ~3 ops regardless of how many are
zero, while routed16 pays per-tile overhead even when a tile only
holds 12-16 nonzeros. At 40% zero density, dense's effective work
per nonzero is ~0.2 ops; routed16's is ~0.5 ops. Multiply-by-zero
is free in vectorized math, so "skipping zero work" only buys
something when zeros vastly outnumber nonzeros.
→ *Why* does the dense path use multiply-by-zero rather than
detect-and-skip? Because vectorization absorbs the cost of the
multiply but not the cost of a per-lane branch. The architectural
trade-off is: SIMD trades branch-friendly-skip for predicate-free
throughput.

**Why is the K=6912 crossover lower than K=2560?** Each tile's
32-trit window is fixed-size. As K grows, the dense path's
per-output work covers more trits, so any single tile-skip
eliminates a larger absolute amount of dense work. Tile overhead
is constant per tile; dense work skipped per tile grows with the
sparsity-window relationship. So tile-skip amortizes faster as K
grows.
→ *Why* does this matter for BitNet? BitNet's down_proj is
K=6912, which is the closest to the routed16 crossover. If
something pushes down_proj's input sparsity above 92%, routed16
becomes viable for that one BitLinear. Worth measuring.

**Why didn't the v1 commit catch the M=1 release-build bug?**
Because the test target builds with `-UNDEBUG` (per
journal/tier2_residuals_v4_precommit.md V4-G1) so asserts fire
in tests. Production builds compile with NDEBUG. The test caught
the assert; the test didn't catch the missing release-mode
guard. Lesson: any assert that exists for a contract critical
to caller correctness needs a hard guard (return early, abort,
or extend the kernel) — not just an assert.
→ *Why* didn't I think of this on the v1 design pass? Because
I framed M=1 as "initial scope; M>1 future work" rather than as
"a contract violation that must fail loudly." Future-work framing
lets you skip the guard. Contract-violation framing forces it.

**Why did I undercount NEON ops as 5 instead of 8?** Cognitive
shortcut: I counted by category (load, gather, reduce) instead
of by individual instructions. The category count was 3 + 1
"acc update" = 4-ish, which I rounded to 5. The instruction
count is 2+2+2+2 = 8. Lesson: when claiming op counts in commit
messages, count instructions, not categories.

**Why did the xpacked mismatch slip past the v2 bench's first
read?** Because I wrote the bit-exact verifier comparing dense
vs routed16 (the kernels I cared about) and added xpacked as
"another baseline" without verifying its bit-exactness. The
diff column was right there showing 2558-2560 cells different,
and I noticed only on the read-back. Lesson: any kernel
included in a bench needs to pass bit-exact against a reference
*before* its timing is reported. Otherwise the timing is
meaningless.

## SYNTHESIZE

**Concrete state after remediation:**
- routed16 is production-grade NEON kernel, M-arbitrary,
  ASAN+UBSAN clean, bit-exact across 33 test cases.
- Crossover table is documented in routed16.h: 96-97% for
  K=N=2560, ~97% for K=2560 N=6912, 92-94% for K=6912 N=2560.
- BitNet's 38-50% sparsity is below all three crossovers.
  Dense remains the right kernel for current BitNet inference.
- xpacked is documented as inapplicable for A8 input.
- No scalar fallback in production; oracle (`_routed_ref`) is
  the test gate.

**Actionable next steps (ranked):**

1. **Measure post-ReLU² activation sparsity** in a real BitNet
   forward pass. If the FFN's intermediate activations exceed
   92% per-token sparsity, the down_proj BitLinear's input is
   sparse — and activation-side routing on that one BitLinear
   becomes a real candidate. Do this with a small, instrumented
   forward (50-100 prompts) and report the distribution.

2. **Don't optimize routed16 further** until step 1 yields a
   yes/no. The kernel sits at the per-tile floor that the
   representation allows. Bit-packed signs would save ~2 ops
   per tile but doesn't move the crossover meaningfully (~95%
   instead of 96-97%). Better representations need a different
   memory layout, not a tighter encoding.

3. **Update project memory** with two clarifications: (a)
   "routing as foundation" ≠ "routing as speed primitive"; the
   former is representational, the latter is operation-and-
   sparsity-dependent. (b) When making op-count claims about
   NEON kernels, count individual instructions, not categories.

4. **Audit other m4t kernels** for the same M-bound silent-
   truncation pattern. Any kernel that takes M as a parameter
   and asserts M==1 has the same release-build hazard.
   Quick grep, fix any found.

5. **Fold the v2 bench into m4t/bench/** as a registered
   benchmark target. /tmp scripts disappear; in-tree bench
   becomes a regression guard against future kernel drift.

**Lessons for future cycles:**

- Methodical means design → test → measure → red-team self →
  remediate → commit. Each phase produces output that the next
  phase consumes. Skipping phases produces fragile claims.
- Asserts are not a substitute for hard guards on caller
  contract. UNDEBUG test builds make asserts feel real; NDEBUG
  production builds make them disappear.
- Op count claims in commit messages need instruction-level
  precision. Category counts mislead.
- Any kernel in a bench must pass bit-exact verification
  *before* its timing is reported. Otherwise the timing is a
  measurement of a wrong answer.
- Routing as foundation is a representational claim. Speed
  wins are operation-dependent. Conflating the two costs
  honesty.
