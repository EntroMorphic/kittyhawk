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
