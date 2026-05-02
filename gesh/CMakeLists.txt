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
endif()
