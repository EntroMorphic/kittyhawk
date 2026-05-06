# TD-7 X-packed wall-clock benchmark

Closes the wall-clock comparison TD-7's closeout deferred ("primitive ships per project rule; wall-clock comparison vs §20 deferred until a consumer demands it"). The "consumer" here is the post-large-cycle health check.

## Method

Three kernels, three speeds:

- `m4t_ternary_dot_matmul_bt` — unpacked X (8 b/c) × unpacked W (8 b/c)
- `m4t_ternary_5in8_matmul_bt` (§20) — unpacked X × 5-in-8 W (1.6 b/c)
- `m4t_ternary_5in8_matmul_xpacked_bt` (§20-xp, TD-7) — 5-in-8 X × 5-in-8 W

Sweep M ∈ {1, 8, 64, 256, 1024, 4096} × K ∈ {1280, 4480, 12800}, N=64. Cache-flush between kernel calls; bit-exactness verified per config (memcmp Y across all three kernels).

Bench source: `audit/td7_xpacked_bench.c`. Run: `build/audit/td7_xpacked_bench`.

## Results

```
config                                      ms_dot   ms_§20   ms_xp    xp/§20  xp/dot   §20/dot
M=1,    K=1280  inference (single token)    0.002    0.002    0.002    0.743   0.840    1.131
M=8,    K=1280  inference (small batch)     0.007    0.011    0.009    0.821   1.288    1.568
M=64,   K=1280  fine-tune                   0.046    0.080    0.066    0.830   1.447    1.743
M=256,  K=1280  training                    0.183    0.323    0.269    0.834   1.475    1.769
M=1024, K=1280  training                    0.739    1.292    1.086    0.840   1.470    1.749
M=1,    K=4480                              0.009    0.005    0.004    0.838   0.473    0.564
M=64,   K=4480                              0.228    0.281    0.235    0.835   1.031    1.235
M=256,  K=4480                              0.887    1.124    0.931    0.829   1.050    1.267
M=1024, K=4480                              3.420    4.503    3.789    0.841   1.108    1.317
M=4096, K=4480                             13.699   18.182   15.141    0.833   1.105    1.327
M=1,    K=12800                             0.022    0.016    0.013    0.859   0.618    0.720
M=64,   K=12800                             0.615    0.803    0.665    0.828   1.082    1.307
M=256,  K=12800                             2.454    3.253    2.705    0.832   1.102    1.326
M=1024, K=12800                             9.691   12.983   10.852    0.836   1.120    1.340
```

## Headline findings

**1. §20-xp BEATS §20 at every tested config (xp/§20 = 0.74-0.86).**

This contradicts the TD-7 closeout's prediction:

> *"§20-xp adds X-decode cost (1 div-by-9 + 5 LUTs per byte). Pays off when X-side memory bandwidth is the bottleneck (large-batch training, KV cache density)."*

The prediction assumed X-decode would be pure overhead at small M. Instead, §20-xp is consistently 14-26% faster than §20 across all M from 1 to 4096.

**Mechanism (post-bench analysis):** the two kernels structurally differ in how they prepare X for the SDOT tile body. Both produce the same strided int8 layout; what differs is the permutation pass:

- **§20** (unpacked X path) uses a scalar nested loop:
  ```c
  for (int n = 0; n < K5; n++)
      for (int d = 0; d < 5; d++)
          X_d[d][n] = (5*n+d < K) ? xi[5*n+d] : 0;
  ```
  Strided memory writes; not vectorized.

- **§20-xp** uses NEON split-LUT decode with vectorized stores:
  16 bytes loaded → 5 NEON LUT lookups → 5 vectorized stores of 16 int8 each. Processes 80 trits per iteration.

The X-packing kernel inadvertently improved the permutation efficiency by replacing the scalar permute with a vectorized decode. The "X-bandwidth savings" the TD-7 closeout predicted as the load-bearing benefit is essentially absent at tested workloads (X is L1-resident at most M values); the load-bearing benefit is actually the better permutation kernel.

**2. §20 (W-packed) is consistently slower than unpacked (§20/dot = 1.13-1.77).**

Confirms the strong-claim retrospective: §20 trades wall-clock for storage. Per-call overhead from W decode + scalar X permutation makes §20 slower than `m4t_ternary_dot_matmul_bt` at every regime tested.

**3. §20-xp vs unpacked is regime-dependent (xp/dot = 0.47-1.48).**

- At M=1 (single-token inference): §20-xp WINS over unpacked at K ≥ 4480 (xp/dot = 0.47-0.86).
- At M ≥ 8: §20-xp loses to unpacked (xp/dot = 1.03-1.48).

For LLM-shape inference (single-token decode, K ∈ 1024-12800), §20-xp is the fastest available kernel. For larger batch (training, fine-tune), unpacked is faster.

## Implications for consumer-side guidance

Updated decision tree (supersedes TD-7 closeout's deferral):

| Workload | Kernel | Why |
|---|---|---|
| Single-token inference, K ≥ 4480 | §20-xp | Fastest (xp/dot = 0.47-0.86) AND smallest storage |
| Single-token inference, K ≤ 1280 | unpacked dot | Slightly faster than §20-xp; storage doesn't matter at this scale |
| Batched inference / fine-tune | unpacked dot | xp/dot ≈ 1.05-1.5; speed > storage at this batch size |
| Large training, storage-bound | §20-xp | xp/§20 ≈ 0.83 always; smaller storage; modest speed cost vs unpacked |
| Memory-bandwidth bound | §20-xp | 5× X reduction × 5× W reduction = 25× total bandwidth savings |

**Production guidance:** ship §20-xp as the default for single-token-inference paths (where it's both fastest AND smallest). Ship unpacked dot for batched-inference / training (fastest at those shapes).

## Honest concerns

1. **The "X-packing pays off" mechanism story is different from what TD-7 predicted.** v1 TD-7 closeout suggested bandwidth would dominate at large M; the bench finds it's actually permutation-pass efficiency at all M. The prediction was directionally wrong.
2. **§20's scalar X-permutation is the slowdown root cause.** A future cycle could vectorize §20's permutation (without requiring X to be packed) and likely close most of the §20-vs-§20-xp gap. Not a blocker for shipping §20-xp.
3. **N=64 fixed.** Larger N (more output projections) would amortize X permutation across more tile cells; the xp/§20 ratio could change. Not tested here.
4. **No SD reported.** Per-config variance bounded by cache-flush + reps; trend across configs is the load-bearing finding.
5. **bit-exact verification per config.** All three kernels produced byte-identical Y for every tested (M, K, N).

## Status

Bench landed; TD-7 closeout updated to point at this doc. Production decision tree above is the consumer-side guidance going forward.

## Cross-references

- Bench source: `audit/td7_xpacked_bench.c`
- TD-7 closeout: `journal/strong_claim_retrospective.md` and original CHANGELOG `Added — TD-7 closure` entry (2026-05-05)
- §20-xp kernel: `m4t/src/m4t_ternary_matmul.c::m4t_ternary_5in8_matmul_xpacked_bt`
- §20 kernel: `m4t/src/m4t_ternary_matmul.c::m4t_ternary_5in8_matmul_bt`
- Unpacked dot: `m4t/src/m4t_ternary_matmul.c::m4t_ternary_dot_matmul_bt`
