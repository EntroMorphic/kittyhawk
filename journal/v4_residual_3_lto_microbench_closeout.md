# Closeout: V4 Residual #3 — LTO microbench

Per the V4 closeout's honest concern #3: "no bench delta" means "no delta on THIS bench." A different bench, designed adversarially in favor of LTO, might surface different findings.

## Verdict: CLOSED with REVISED finding

```
Cycle: design microbench → discover -fno-lto wasn't actually applied → fix CMake gating →
       measure → discover variant A still shows no delta → red-team → add variant B →
       discover 3x speedup → red-team again → doc/commit
Result: LTO IS doing useful work. V4-G5's "no delta" finding was correct for its
        workload shape (carry-dependent), but NOT a general statement about LTO.
```

## Headline finding

LTO produces dramatically different deltas depending on workload shape:

| Workload | LTO ns/call | no-LTO ns/call | LTO speedup | Bottleneck |
|----------|------------:|---------------:|------------:|------------|
| Variant A (carry-dependent, single dst accumulated) | 1.36 | 1.35 | **~1.0×** | Data dependency between iters |
| Variant B (pipelined, 64 independent dsts round-robin) | 0.23 | 0.68 | **~3.0×** | Call overhead |

Variant A reproduces V4-G5's "no observable LTO benefit" finding under controlled conditions. Variant B shows LTO IS doing meaningful cross-TU inlining — it just doesn't surface on the substrate's actual consumers, which are mostly variant-A-shaped (accumulating into state).

## Disasm verification

**LTO build, variant A inner loop (4 instructions):**
```
add.4s   v0, v0, v1
smin.4s  v0, v0, v2
smax.4s  v0, v0, v3
subs     w8, w8, #0x1
b.ne     <loop>
```
`m4t_mtfp_block_add` fully inlined. dst kept in register v0 across iterations. Bottlenecked by the `add → smin → smax → add` dependency chain (~5 cycles latency, observed 4.8).

**LTO build, variant B inner loop (7 instructions):**
```
ldr      q0, [x19, x9]      ; load dst[i % 64]
add.4s   v0, v0, v1
smin.4s  v0, v0, v2
smax.4s  v0, v0, v3
str      q0, [x19, x9]      ; store dst[i % 64]
add      w8, w8, #0x1
b.ne     <loop>
```
Inlined. Loads/stores per iter, but no carry dependency between iters. Apple Silicon's 8-wide superscalar issues these at ~9 IPC (0.8 cycles/iter).

**no-LTO build, variant A and B inner loops:**
```
bl       _m4t_mtfp_block_add
subs     w19, w19, #0x1
b.ne     <loop>
```
Function NOT inlined. Symbol `_m4t_mtfp_block_add` present in binary at fixed address. Per-call overhead ~5 cycles for branch+return, hidden by data dependency in variant A but exposed in variant B.

## What surfaced during the cycle

**Surprise #1: `-fno-lto` was being silently overridden.** First attempt at the no-LTO build passed `-DCMAKE_C_FLAGS="-fno-lto"`. CMake prepends user flags but the project's own `add_compile_options(-flto)` appends, so the compile line ended up `... -fno-lto ... -flto ...` and clang took the LATER flag. Both builds had LTO active, both showed identical timings — which I correctly recognized as suspicious. **Fix:** added `option(GESH_LTO "Enable link-time optimization" ON)` to top-level `CMakeLists.txt`, gated `add_compile_options(-flto)` and `add_link_options(-flto)` behind it. No-LTO build now uses `-DGESH_LTO=OFF`. Verified via `cmake --build ... -- VERBOSE=1` that `-flto` is absent from the no-LTO compile line.

**Surprise #2: variant A showed no LTO delta even after `-fno-lto` actually applied.** Both builds: ~1.35 ns/iter. Disasm confirmed LTO inlined block_add (no `bl`) while no-LTO didn't (one `bl` per iter). Yet the timings were equal. Hypothesis: workload is bottlenecked by data dependency between iters, not by per-call overhead. The CPU's branch prediction + out-of-order execution hides the `bl` cost when the dependency chain dominates.

**Confirmation via variant B:** restructured the workload to pipeline independent dsts. With no carry dependency, variant B exposes call overhead as the bottleneck. LTO eliminates the calls entirely → 3× speedup. This is the proof LTO does meaningful work.

## Red-team and remediation

Five findings examined:

| ID | Finding | Disposition |
|----|---------|-------------|
| RT-1 | Variant B's pipelined workload may not represent any real substrate consumer; the 3× win is real but irrelevant. | **DOCUMENTED** — explicit statement in bench source: "the substrate's real consumers are mostly (A)-shaped (accumulating into a state)." |
| RT-2 | Variant B's 0.8 cycles/iter — could indicate unexpected loop unrolling. | **VERIFIED** — disasm shows single-iter loop body, 7 instructions. The 0.8 cycles comes from Apple Silicon's 8-wide issue + micro-op cache + branch prediction, not unrolling. |
| RT-3 | Variant A's "no LTO benefit" duplicates V4-G5. | **ACCEPTED** — variant A is the controlled reproduction of V4-G5; variant B is the new evidence. Both are valuable. |
| RT-4 | Adding `GESH_LTO` option permanently alters the build API. | **ACCEPTED** — defaults ON (preserves prior behavior); enables the comparison; future cycles can flip it without source edits. |
| RT-10 | Variant B's 3× speedup could come from cross-TU inlining of block_add, OR from better libm4t internal optimization due to LTO compile of `m4t_mtfp.c`, OR both. The disasm proves cross-TU inlining (no `bl` in LTO build, `bl` present in no-LTO build), but doesn't isolate intra-TU LTO contributions. | **DOCUMENTED** — flagged as honest concern, not addressed (would require building libm4t and bench in mixed configs, e.g., libm4t-LTO + bench-no-LTO). The cross-TU inlining is the FIRST suspect (proven by the missing `bl`), so the bulk of the 3× is almost certainly from inlining. |

## What shipped

- `m4t/tests/bench_m4t_lto.c` — new microbench with two variants (A: carry-dep, B: pipelined). Min-of-3 sampling per variant. Print ns/call and approximate cycles/call. Build target only (not ctest).
- `m4t/CMakeLists.txt` — added `bench_m4t_lto` build target linked against production `m4t` (not `m4t_test`; perf bench, not regression test).
- `CMakeLists.txt` (top-level) — gated `-flto` behind `option(GESH_LTO "Enable link-time optimization" ON)`. Default behavior unchanged. Enables the comparison via `-DGESH_LTO=OFF` for a parallel build tree.

## What's now structurally true

**LTO is enabled, applied, and produces meaningful per-instruction wins on cross-TU inlinable functions.** The V4-G5 finding ("no observable bench delta") is correct narrowly: for the substrate's actual consumers (accumulating into state, data-dep-bound), LTO doesn't change observed perf. But LTO IS doing structural work — it inlines `m4t_mtfp_block_add` cleanly into bench main, eliminating ~5 cycles of call overhead per iter. That overhead is invisible in dependency-bound workloads but dominates in pipelined ones (3× speedup observed in variant B).

The takeaway: **keep LTO enabled.** It's free for the substrate's measured workloads and a 3× win for any future workload that becomes call-overhead-bound.

## Honest concerns from this cycle

**1. Did NOT formally isolate cross-TU vs intra-TU LTO contributions in variant B.** The 3× speedup almost certainly comes from cross-TU inlining (no `bl` in LTO build), but mixed configs (libm4t-LTO + bench-no-LTO, or vice-versa) would prove the decomposition. Defer until there's a perf reason to know exactly.

**2. The microbench measures one specific cross-TU function (`m4t_mtfp_block_add`).** Other library functions might inline differently under LTO. A more thorough audit would parameterize across substrate functions. Not done; flagged for future work if substrate perf becomes a bottleneck.

**3. The 3× speedup on variant B has no current consumer.** No substrate consumer is structured to benefit. If we ever build a workload that pipelines block ops across independent buffers (e.g., batched matmul over many small tiles), this finding becomes load-bearing. Until then, it's evidence of correct LTO behavior, not a perf win.

**4. The CMake `GESH_LTO` option is not exercised in CI.** Only the default (ON) configuration is built and tested. A regression that breaks the OFF path would not be caught. Risk: low; defer.

## Methodology lifted

**1. Always prove LTO actually applied.** Verbose make output (`cmake --build ... -- VERBOSE=1`) is the source of truth. CMake flag prepending vs appending behavior bit me — fixed by gating with `option()` so the `add_compile_options` is conditional, not silently overridden.

**2. Compiler flags can lie about their effect; disasm cannot.** Always cross-check optimization claims with `otool -tv` (or equivalent disassembly).

**3. Workload shape determines bottleneck, not the compiler.** A workload's bottleneck (data dependency, memory bandwidth, call overhead, etc.) determines what optimizations CAN help. Measuring a workload in only one shape under-determines the conclusion.

**4. "No delta" findings should be tested with at least one adversarial variant.** If a workload designed adversarially in favor of optimization X also shows no delta, the finding generalizes. If it shows a delta, the original "no delta" was specific to that workload shape.

## Status

CLOSED — V4 residual #3 (LTO opacity) is structurally remediated. The microbench provides:
- Direct proof LTO applies (disasm shows inlining).
- Direct measurement that LTO produces no benefit on carry-dep workloads (matching V4-G5).
- Direct measurement that LTO produces 3× benefit on pipelined workloads (the new finding).

The V4-G5 "no observable LTO benefit" stands but is now precisely scoped: it's a property of the substrate's workload shape, not a property of LTO.
