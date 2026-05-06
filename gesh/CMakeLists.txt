# Gesh — base-3 native routing layer.
#
# The substrate's first measured consumer. Phase A.1 ships forward-pass
# only; Phase A.2 will add lattice-update training.
#
# See gesh/README.md and journal/gesh_design_*.md for design context.

# Synthetic benchmark — separate library so it's available to both gesh
# tests and any future end-to-end tools.
add_library(gesh_bench STATIC
    bench/synth_proto.c
    bench/synth_wildcard.c
    bench/synth_close_proto.c
    bench/synth_compose_hier.c
)
target_include_directories(gesh_bench PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bench>
)
target_link_libraries(gesh_bench PUBLIC m4t)
set_target_properties(gesh_bench PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Gesh consumer library.
add_library(gesh STATIC
    src/gesh_bank.c
    src/gesh_forward.c
    src/gesh_project.c
    src/gesh_train.c
    src/expr.c
    src/expr_signature.c
    src/expr_bank.c
    src/expr_random.c
)
target_include_directories(gesh PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
target_link_libraries(gesh PUBLIC m4t)
set_target_properties(gesh PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Image canon — substrate-legal MNIST loader + ternary quantization
# pipeline (no random projections; one trit per pixel). Lifted from
# 01MAY26_archived/src/{glyph_dataset.c, glyph_sig.c}. Declared before
# the test block so test_image_canon can link against it.
add_library(gesh_image_canon STATIC
    bench/image_canon.c
)
target_include_directories(gesh_image_canon PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bench>
)
target_link_libraries(gesh_image_canon PUBLIC m4t)
set_target_properties(gesh_image_canon PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Test variants of gesh, gesh_bench, gesh_image_canon. Same sources, but
# compiled with -UNDEBUG so the libraries' internal asserts (precondition
# checks) actually fire when tests trigger them. Each links against m4t_test
# transitively so substrate asserts also fire when reached via these libs.
# Per V4-G1 from journal/tier2_residuals_v4_precommit.md.
add_library(gesh_bench_test STATIC
    bench/synth_proto.c
    bench/synth_wildcard.c
    bench/synth_close_proto.c
    bench/synth_compose_hier.c
)
target_include_directories(gesh_bench_test PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bench>
)
target_compile_options(gesh_bench_test PRIVATE -UNDEBUG)
target_link_libraries(gesh_bench_test PUBLIC m4t_test)
set_target_properties(gesh_bench_test PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(gesh_test STATIC
    src/gesh_bank.c
    src/gesh_forward.c
    src/gesh_project.c
    src/gesh_train.c
    src/expr.c
    src/expr_signature.c
    src/expr_bank.c
    src/expr_random.c
)
target_include_directories(gesh_test PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
target_compile_options(gesh_test PRIVATE -UNDEBUG)
target_link_libraries(gesh_test PUBLIC m4t_test)
set_target_properties(gesh_test PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(gesh_image_canon_test STATIC
    bench/image_canon.c
)
target_include_directories(gesh_image_canon_test PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bench>
)
target_compile_options(gesh_image_canon_test PRIVATE -UNDEBUG)
target_link_libraries(gesh_image_canon_test PUBLIC m4t_test)
set_target_properties(gesh_image_canon_test PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Tests.
option(GESH_BUILD_TESTS "Build Gesh tests" ON)
if(GESH_BUILD_TESTS)
    add_executable(test_synth_proto tests/test_synth_proto.c)
    target_link_libraries(test_synth_proto PRIVATE gesh_bench_test)
    gesh_test_undebug(test_synth_proto)
    add_test(NAME gesh_synth_proto COMMAND test_synth_proto)

    add_executable(test_gesh_bank tests/test_gesh_bank.c)
    target_link_libraries(test_gesh_bank PRIVATE gesh_test gesh_bench_test)
    gesh_test_undebug(test_gesh_bank)
    add_test(NAME gesh_bank COMMAND test_gesh_bank)

    add_executable(test_gesh_forward tests/test_gesh_forward.c)
    target_link_libraries(test_gesh_forward PRIVATE gesh_test gesh_bench_test)
    gesh_test_undebug(test_gesh_forward)
    add_test(NAME gesh_forward COMMAND test_gesh_forward)

    add_executable(test_gesh_train tests/test_gesh_train.c)
    target_link_libraries(test_gesh_train PRIVATE gesh_test gesh_bench_test)
    gesh_test_undebug(test_gesh_train)
    add_test(NAME gesh_train COMMAND test_gesh_train)

    # Smoke test for image_canon (MNIST loader + normalize + quantize).
    # Lifted from M3 in the Phase B red-team.
    add_executable(test_image_canon tests/test_image_canon.c)
    target_link_libraries(test_image_canon PRIVATE gesh_image_canon_test)
    gesh_test_undebug(test_image_canon)
    add_test(NAME gesh_image_canon COMMAND test_image_canon)

    # Bit-equivalence test: kernel-routed projection vs reference loop.
    # Establishes that the substrate-discipline cleanup is semantically
    # neutral (identical ternary outputs for all sampled shapes/seeds).
    add_executable(test_gesh_project tests/test_gesh_project.c)
    target_link_libraries(test_gesh_project PRIVATE gesh_test)
    gesh_test_undebug(test_gesh_project)
    add_test(NAME gesh_project COMMAND test_gesh_project)
endif()

# Sweep tool (benchmark, not regression test). Builds with the library.
# Links `m` for sqrt (used for stddev computation in multi-seed reporting).
add_executable(gesh_sweep_dims bench/sweep_dims.c)
target_link_libraries(gesh_sweep_dims PRIVATE gesh gesh_bench m)

# Phase B Gate 1: MNIST canonical pipeline probe.
add_executable(gesh_mnist_probe bench/mnist_probe.c)
target_link_libraries(gesh_mnist_probe PRIVATE gesh gesh_image_canon m)

# Phase B Gate 2: H1 mechanism test (implicit denoising hypothesis).
add_executable(gesh_denoise_probe bench/denoise_probe.c)
target_link_libraries(gesh_denoise_probe PRIVATE gesh gesh_bench m)

# Finding 3 high-seed probe (capacity floor at sig_dim ≤ 4 from
# sweep_dims_results.md). 30 seeds vs original sweep's 5; tightens the
# confidence interval on the capacity-floor claim.
add_executable(gesh_finding3_probe bench/finding3_probe.c)
target_link_libraries(gesh_finding3_probe PRIVATE gesh gesh_bench m)

# Single-run MNIST probe: full 60K train, full 10K test, sig_dim=64,
# top_k=3, n_epochs=64, early_stop OFF. One decisive measurement.
add_executable(gesh_mnist_full_run bench/mnist_full_run.c)
target_link_libraries(gesh_mnist_full_run PRIVATE gesh gesh_image_canon m)

# k-means multi-prototype bank measurement on MNIST. Tests whether
# bank capacity (T=10) was the bottleneck, by replacing single-class-
# mean bank with k-means (T = k_per_class × C tiles).
add_executable(gesh_mnist_kmeans_run bench/mnist_kmeans_run.c)
target_link_libraries(gesh_mnist_kmeans_run PRIVATE gesh gesh_image_canon m)

# k-means trained: train R via lattice update with k-means bank refresh
# in the loop (vs class-mean refresh in mnist_full_run.c). Tests whether
# training contributes on top of the bank-capacity gain.
add_executable(gesh_mnist_kmeans_trained bench/mnist_kmeans_trained.c)
target_link_libraries(gesh_mnist_kmeans_trained PRIVATE gesh gesh_image_canon m)

# P0-1 verification: wildcard kernel + wildcard bank vs class_mean + Hamming
# baseline on synth_wildcard. Runs Gates 1, 2, 3 from the design SYNTHESIZE.
add_executable(gesh_wildcard_probe bench/wildcard_probe.c)
target_link_libraries(gesh_wildcard_probe PRIVATE gesh gesh_bench m)

# P0-1 Gate 4: MNIST regression check for wildcard bank.
add_executable(gesh_mnist_wildcard bench/mnist_wildcard.c)
target_link_libraries(gesh_mnist_wildcard PRIVATE gesh gesh_image_canon)

# P0-2 verification: confidence-weighted routing on synth_proto.
add_executable(gesh_confidence_probe bench/confidence_probe.c)
target_link_libraries(gesh_confidence_probe PRIVATE gesh gesh_bench m)

# P0-2 Gate 4: MNIST regression check for confidence routing.
add_executable(gesh_mnist_confidence bench/mnist_confidence.c)
target_link_libraries(gesh_mnist_confidence PRIVATE gesh gesh_image_canon)

# P0-3 verification: lattice-native geometric training on synth_proto.
add_executable(gesh_geometric_probe bench/geometric_probe.c)
target_link_libraries(gesh_geometric_probe PRIVATE gesh gesh_bench m)

# P0-3 Gate 4: MNIST regression check for geometric training.
add_executable(gesh_mnist_geometric bench/mnist_geometric.c)
target_link_libraries(gesh_mnist_geometric PRIVATE gesh gesh_image_canon)

# P0-4 verification: hierarchical (compositional) routing on synth_close_proto.
add_executable(gesh_compose_probe bench/compose_probe.c)
target_link_libraries(gesh_compose_probe PRIVATE gesh gesh_bench m)

# P0-4 RED-TEAM REMEDIATION: ablation across stage-1/stage-2/mask configs
# on a hierarchical synthetic (synth_compose_hier).
add_executable(gesh_compose_ablation bench/compose_ablation.c)
target_link_libraries(gesh_compose_ablation PRIVATE gesh gesh_bench m)

# P0-4 Gate 4 (red-team C3): MNIST regression check for P0-4 hier.
add_executable(gesh_mnist_compose bench/mnist_compose.c)
target_link_libraries(gesh_mnist_compose PRIVATE gesh gesh_image_canon)

# Vision claim #2 P0-4: equivalence-recognition probe for expression routing.
# See docs/PLAN_EXPRESSION_ROUTING.md and journal/expression_routing_*.md.
add_executable(gesh_expr_routing_probe bench/expr_routing_probe.c)
target_link_libraries(gesh_expr_routing_probe PRIVATE gesh)

# Vision claim #2 P0-4 RED-TEAM REMEDIATION: addresses C1, C2, H1, H2, H3,
# M1, M2 from journal/expression_routing_redteam.md against pre-committed
# gates in journal/expression_routing_remediation_precommit.md.
add_executable(gesh_expr_routing_remediation bench/expr_routing_remediation.c)
target_link_libraries(gesh_expr_routing_remediation PRIVATE gesh m)

# R1: per-expression-tau dual-threshold signature rule.
# Addresses red-team concerns 2, 3, 7, partial 8 from
# journal/expression_routing_redteam.md per the LMM cycle in
# journal/r1_signature_rule_synthesize.md, against pre-committed gates
# R1-A/B/C in docs/PLAN_EXPRESSION_ROUTING_R2.md.
add_executable(gesh_expr_routing_r1 bench/expr_routing_r1.c)
target_link_libraries(gesh_expr_routing_r1 PRIVATE gesh)

# R1 RED-TEAM REMEDIATION: addresses 13 findings from
# journal/r1_signature_rule_redteam.md per pre-committed gates in
# journal/r1_remediation_precommit.md.
add_executable(gesh_expr_routing_r1_remediation bench/expr_routing_r1_remediation.c)
target_link_libraries(gesh_expr_routing_r1_remediation PRIVATE gesh m)

# R1 FORK EXPERIMENT: distinguishes F1/F2/F3 framings of R1 FAIL per
# docs/PLAN_R1_FORK.md and journal/r1_path_forward_synthesize.md.
add_executable(gesh_expr_routing_r1_fork bench/expr_routing_r1_fork.c)
target_link_libraries(gesh_expr_routing_r1_fork PRIVATE gesh)

# F-G5: held-out routing accuracy for R1 (TD-8). Equivalence ground truth
# constructed via int64 evaluation on a separate input fingerprint set.
# Per journal/r1_falsify_closeout.md F-G5 + journal/r1_falsify_f_g5_*.md.
add_executable(gesh_expr_routing_r1_f_g5 bench/expr_routing_r1_f_g5.c)
target_link_libraries(gesh_expr_routing_r1_f_g5 PRIVATE gesh)

# BitNet b1.58-2B-4T inference harness (Phase 1 of the ternary-ML arc).
# Per LMM cycle journal/bitnet_phase1_*. Single-block forward-pass
# scaffolding; substrate-side validation that kernels compose into a
# real LLM. Not in ctest — manual invocation via the harness CLI.
add_executable(bitnet_harness
    bitnet/bitnet_harness.c
    bitnet/bitnet_stubs.c
    bitnet/bitnet_weights.c
)
target_include_directories(bitnet_harness PRIVATE bitnet)
target_link_libraries(bitnet_harness PRIVATE m4t m)
