# Tri-state operationalization audit.
#
# Per journal/tristate_op_synthesize.md. Measures third-state utilization
# across substrate layers on a 2-layer ternary GEMM workload.
#
# Build target only — not in ctest. Audit is a science deliverable, run
# manually to produce the per-layer measurements that feed the closeout.

cmake_minimum_required(VERSION 3.16)

add_executable(tristate_audit tristate_audit.c)
target_link_libraries(tristate_audit PRIVATE m4t m)
