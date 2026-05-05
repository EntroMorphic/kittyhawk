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
# All kernels NEON-only; verification is NEON-vs-NEON cross-check.
add_executable(tristate_strong_bench
    tristate_strong_bench.c
    b2b_matmul.c
)
target_link_libraries(tristate_strong_bench PRIVATE m)
