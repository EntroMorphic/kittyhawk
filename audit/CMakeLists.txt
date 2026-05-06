# Tri-state operationalization audit + strong-claim test.
#
# Per journal/tristate_op_synthesize.md (audit) and
# journal/tristate_strong_synthesize.md (strong-claim).
#
# Build targets only — not in ctest. Science deliverables, run manually.

cmake_minimum_required(VERSION 3.16)

# Audit binary — measures third-state utilization across substrate layers.
add_executable(tristate_audit tristate_audit.c)
target_link_libraries(tristate_audit PRIVATE m4t m)

# Strong-claim bench — compares base-3 vs B2-B (sign+mask) NEON kernels.
# All kernels NEON-only; verification is NEON-vs-NEON cross-check, plus
# external grounding via libm4t's externally-validated ternary matmul
# (R-G2 from the strong-claim red-team).
add_executable(tristate_strong_bench
    tristate_strong_bench.c
    b2b_matmul.c
)
target_link_libraries(tristate_strong_bench PRIVATE m4t m)

# L4 strong-claim cycle (TD-4) — tests three operationalization candidates
# for L4: A.1 absmean ternarization (BitNet b1.58 rule), A.2 zero-flag
# forwarding, A.3 sign+magnitude two-channel split. Compares cos(L4) under
# each rule against quantile baseline; verdict on whether L4 can be made
# more load-bearing.
add_executable(tristate_l4_strong tristate_l4_strong.c)
target_link_libraries(tristate_l4_strong PRIVATE m4t m)

# L5 strong-claim cycle (TD-5) — residual-style cross-exp accumulator
# workload. Tests whether the third-state outcome of cross-exp summation
# (exact-zero result from cancelling addends) is load-bearing downstream.
add_executable(tristate_l5_strong tristate_l5_strong.c)
target_link_libraries(tristate_l5_strong PRIVATE m4t m)

# L6 strong-claim cycle (TD-6) — base-3 vs B2-B at the L6 layer
# (post-ternarization activations). Symmetric to L1; reuses kernel paths.
add_executable(tristate_l6_strong
    tristate_l6_strong.c
    b2b_matmul.c
)
target_link_libraries(tristate_l6_strong PRIVATE m4t m)

# DRAM-bound regime test (TD-9) — extends tristate_strong_bench shapes
# into N=2048+ / K=large territory where W substantially exceeds L2 cache.
add_executable(tristate_dram_regime
    tristate_dram_regime.c
    b2b_matmul.c
)
target_link_libraries(tristate_dram_regime PRIVATE m4t m)

# TD-7 X-packed §20 vs §20 vs unpacked benchmark — closes the wall-clock
# comparison TD-7's closeout deferred. Sweeps M, K to find the regime
# where 5-in-8 X-packing begins to beat unpacked-X §20.
add_executable(td7_xpacked_bench td7_xpacked_bench.c)
target_link_libraries(td7_xpacked_bench PRIVATE m4t m)

# No-scalar audit setup-time bench (2026-05-06): quantifies the speedup
# from NEON-izing pack/unpack/conversion/shift3-multiply primitives.
# NEON public dispatcher vs _scalar_ref test oracle, per function, across
# L1/L2/DRAM-resident N values plus tail-exercise rows.
add_executable(no_scalar_audit_bench no_scalar_audit_bench.c)
target_link_libraries(no_scalar_audit_bench PRIVATE m4t m)

# NOTE: sdot_pipeline_bench moved to m4t/tools/ on 2026-05-05 (per Item 3
# of the production-shoring plan). Compile manually with:
#   cc -O3 -mcpu=native m4t/tools/sdot_pipeline_bench.c -o /tmp/sdb
