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

# Tests.
option(GESH_BUILD_TESTS "Build Gesh tests" ON)
if(GESH_BUILD_TESTS)
    add_executable(test_synth_proto tests/test_synth_proto.c)
    target_link_libraries(test_synth_proto PRIVATE gesh_bench)
    add_test(NAME gesh_synth_proto COMMAND test_synth_proto)

    add_executable(test_gesh_bank tests/test_gesh_bank.c)
    target_link_libraries(test_gesh_bank PRIVATE gesh gesh_bench)
    add_test(NAME gesh_bank COMMAND test_gesh_bank)

    add_executable(test_gesh_forward tests/test_gesh_forward.c)
    target_link_libraries(test_gesh_forward PRIVATE gesh gesh_bench)
    add_test(NAME gesh_forward COMMAND test_gesh_forward)

    add_executable(test_gesh_train tests/test_gesh_train.c)
    target_link_libraries(test_gesh_train PRIVATE gesh gesh_bench)
    add_test(NAME gesh_train COMMAND test_gesh_train)

    # Smoke test for image_canon (MNIST loader + normalize + quantize).
    # Lifted from M3 in the Phase B red-team.
    add_executable(test_image_canon tests/test_image_canon.c)
    target_link_libraries(test_image_canon PRIVATE gesh_image_canon)
    add_test(NAME gesh_image_canon COMMAND test_image_canon)

    # Bit-equivalence test: kernel-routed projection vs reference loop.
    # Establishes that the substrate-discipline cleanup is semantically
    # neutral (identical ternary outputs for all sampled shapes/seeds).
    add_executable(test_gesh_project tests/test_gesh_project.c)
    target_link_libraries(test_gesh_project PRIVATE gesh)
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
