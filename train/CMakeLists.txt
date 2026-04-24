# train/ — routed autodiff MVP.
#
# Consumer-layer training scaffolding per NORTH_STAR §13 ("training
# artifacts live in the consumer"). Not linked into libm4t or libglyph.
#
# The MVP builds:
#   - libtrain.a              static lib with backward primitives
#   - test_gradient_linear     numerical gradient check (required)
#
# Later additions (planned):
#   - routed_trainer          CIFAR trainer tool
#   - test_toy_convergence     2-class linear separation toy
#   - NEON backward kernels   hot-loop vectorization

cmake_minimum_required(VERSION 3.16)

add_library(train STATIC
    src/backward_linear.c
    src/backward_routed.c
    src/requantize.c
)
target_include_directories(train PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
target_link_libraries(train PUBLIC m)

option(GLYPH_BUILD_TRAIN_TESTS "Build routed-autodiff gradient check tests" ON)
if(GLYPH_BUILD_TRAIN_TESTS)
    enable_testing()

    add_executable(test_gradient_linear tests/test_gradient_linear.c)
    target_link_libraries(test_gradient_linear PRIVATE train)
    add_test(NAME train_gradient_linear COMMAND test_gradient_linear)

    add_executable(test_gradient_routed tests/test_gradient_routed.c)
    target_link_libraries(test_gradient_routed PRIVATE train)
    add_test(NAME train_gradient_routed COMMAND test_gradient_routed)

    add_executable(test_toy_convergence tests/test_toy_convergence.c)
    target_link_libraries(test_toy_convergence PRIVATE train)
    add_test(NAME train_toy_convergence COMMAND test_toy_convergence)

    add_executable(test_toy_10class tests/test_toy_10class.c)
    target_link_libraries(test_toy_10class PRIVATE train)
    add_test(NAME train_toy_10class COMMAND test_toy_10class)

    add_executable(test_edge_cases tests/test_edge_cases.c)
    target_link_libraries(test_edge_cases PRIVATE train)
    add_test(NAME train_edge_cases COMMAND test_edge_cases)
endif()
