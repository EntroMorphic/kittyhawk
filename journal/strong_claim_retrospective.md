# RETROSPECTIVE: the strong claim's current precise verdict

Per `docs/TECHNICAL_DEBT.md` TD-11. Consolidates the strong-claim verdict (vision claim 3 in operational form) across 7+ addenda + red-teams. Single landing point for readers who don't want to traverse the cycle history.

## The claim

**Vision claim 3 (operational form):** Base-3 carries information that base-2 collapses, in a way structurally cheaper or more accurate than base-2's workaround (e.g., binary value + separate masking machinery).

## Verdict precision (post all red-teams)

**SUPPORTED at the density-ceiling layer. CONDITIONAL at the kernel-cost layer.**

### Density-ceiling (UNCONDITIONAL structural advantage)

Base-3 has a sub-2-bits/cell packing. B2-B (sign + mask) does not.

- Base-3 5-in-8 packing: **1.6 bits/cell** (5 trits in 8 bits; 243 valid byte codes, 13 reserved). Approaches log2(3) ≈ 1.585.
- B2-B floor: **2 bits/cell** (sign and mask are independent — they cannot share information).

**B2-B cannot follow base-3 below 2 bits/cell.** This is hardware-independent, structurally provable. It's the strongest defensible position.

### Encoding-label equivalence at fixed density (proved by R-G1)

At 2 bits/cell, "base-3 ternary" and "B2-B with optimal unified-LUT decode" are byte-for-byte identical kernels (same TBL pattern, different LUT contents). At fixed density, the encoding label is a relabeling — not a structural difference.

This means: any wall-clock advantage at 2 b/c is from the IMPLEMENTATION CHOICE (e.g., decoding sign+mask separately vs unified LUT), not from the encoding itself. Path A (base-3 4-in-8) ≡ Path C (B2-B optimal) at the disasm level.

### Kernel-cost (regime-dependent, hardware-specific)

- **At sub-2-bit density on Apple Silicon:** Path D (base-3 5-in-8) wins ~1.8× wall-clock vs Path A (base-3 4-in-8 packed) when both register-tiled, due to better SDOT amortization (5 SDOTs per outer block vs 1; mechanism directly measured per `journal/p0_concern1_mechanism.md`).
- **In libm4t: §20 vs ternary_dot.** §20 (5-in-8 packed) is 1.14-1.5× SLOWER than `m4t_ternary_dot_matmul_bt` (which uses UNPACKED W = 8 b/c). DIFFERENT comparison axis. §20's value is the storage-vs-decode tradeoff (5× less storage for 1.14-1.5× compute cost) — pays off at memory-cost-bound consumers, loses at L1-resident inference.
- **Trajectory in memory-bandwidth-bound regimes:** plateaus, doesn't crossover (per `journal/tristate_strong_membw_addendum.md` red-team R-G2). Apple Silicon's unified memory bandwidth (~70-200 GB/s) is generous enough that the ~5 MB savings per call from denser packing don't dominate the decode cost.

### L2 generalization (per concern 2 remediation)

- **Encoding-label equivalence at L2 (Path E ≡ Path F):** verified directly via disassembly. Symmetry argument from R-G1 extends to activation packing.
- **Wall-clock advantage at L2:** does NOT extend at L1-resident M=8 workloads (Path E is 20% slower than Path A — X decode cost without bandwidth savings since X stays L1-resident).

## How the verdict moved (history)

| Round | Headline at the time | What changed it |
|---|---|---|
| Initial L1 cycle | "STRONG CLAIM SUPPORTED — base-3 wins by 3 ops/block" | R-G1 caught B2-B-honest as a strawman. |
| R-G1 remediation | "ENCODING-LABEL EQUIVALENCE at fixed density" | Path C added; byte-identical to Path A. |
| 5-in-8 addendum | "DENSITY CEILING is the structural foothold" | Sub-2-bit packing demonstrated; B2-B floored at 2 b/c. |
| Membw addendum draft | "Trajectory toward DRAM-bound crossover" | Membw red-team C3: tested DRAM-bound; trajectory PLATEAUS, not crossover. |
| P0-1, P0-2, P0-3 | "Path D 1.8× faster than Path A apples-to-apples tile" | P0-3 red-team caught initial 3× as tile asymmetry; honest 1.8× preserved. |
| P0-Concern-1 | Mechanism inferred ("SDOT amortization") | SDOT throughput microbench: ratio 1.78× (matches wall-clock 1.8× exactly). Mechanism empirically grounded. |
| P0-Concern-2 | "Verdict L1-only, generalization premature" | Path E added for L2; encoding-label equivalence extends; wall-clock advantage doesn't (at our workloads). |
| Production-shoring red-team | "1.8× advantage" carried into libm4t | Reframed: libm4t's comparison axis differs (5-in-8 vs UNPACKED, not vs 4-in-8). §20 is 1.14-1.5× SLOWER in libm4t — storage-vs-decode tradeoff, not throughput win. |

The verdict moved through EIGHT distinct refinement rounds. Each red-team caught an overclaim or a reframing necessity. The current position is the most defensible point reached so far.

## What's verified vs untested

### Verified
- **L1 weights** at 2 b/c: encoding-label equivalence (Path A ≡ Path C disasm).
- **L1 weights** at 1.6 b/c: density-ceiling structural advantage (B2-B can't follow).
- **L1 weights** kernel-cost on Apple Silicon: Path D ~1.8× of Path A apples-to-apples tile (SDOT amortization mechanism directly measured).
- **L2 activations** at 2 b/c: encoding-label equivalence extends (Path E ≡ Path F disasm).
- **L2 activations** wall-clock at L1-resident M=8: 20% slower (X decode cost, no bandwidth savings).
- **Memory-bandwidth-regime** behavior up to W = 25.6 MB (exceeds L2 partially): trajectory plateaus, doesn't crossover.

### Untested
- **L4** (cross-layer requantization) strong-claim cycle (TD-4).
- **L5** (cross-exp accumulator) strong-claim cycle (TD-5; needs residual-style workload).
- **L6** (post-ternarization activations) strong-claim cycle (TD-6; symmetry argument suggests follows L1/L2).
- **Sub-2-bit X-packing** (5-in-8 X) wall-clock impact (TD-7).
- **DRAM-bound crossover** at workloads where W > L2 substantially (TD-9; would need N=2048+ or K > 1M).
- **Real LLM workload shapes** (large batch / KV cache / training-shape activations).
- **Non-Apple-Silicon hardware** (the kernel-cost direction depends on memory bandwidth vs compute ratio; M-series's generous bandwidth is a specific hardware trait).

## What this means for consumers

For a consumer choosing between base-3 and base-2-with-mask:
1. **At 2 bits/cell density:** the choice is a labeling preference. Same kernel performance, same density. Pick whichever fits the calling convention.
2. **At sub-2-bit density:** only base-3 is available. Use it if you need sub-2-bit storage (e.g., LLM weight compression, KV cache density).
3. **For raw kernel throughput on Apple Silicon:** unpacked SDOT (`m4t_ternary_dot_matmul_bt`) is fastest at L1-resident workloads. Switch to packed (5-in-8 §20 or 4-in-8 default) when storage matters.

## Where to look for more depth

- Density-ceiling argument: `journal/tristate_strong_5in8_addendum.md`.
- Mechanism for the L1 wall-clock win: `journal/p0_concern1_mechanism.md`.
- Why "trajectory crossover" was wrong: `journal/tristate_strong_membw_redteam.md`.
- L2 generalization scope: `journal/p0_concern2_l2.md`.
- Production-shoring reframing: `journal/production_shoring_redteam.md`.
- Full cycle docs: `journal/tristate_strong_*.md` series.
- Spec amendment for sub-2-bit packing: `m4t/docs/M4T_SUBSTRATE.md` §20.

## Status

Strong-claim cycle as a body of work is complete (for L1; L2 partially; L4/L5/L6 deferred per TD-4/5/6). Verdict is precise and defensible. Future work either (a) tests untested layers, (b) tests untested workload regimes (DRAM-bound, non-Apple-Silicon), or (c) consumer-side validation when consumer-layer rebuild begins.
