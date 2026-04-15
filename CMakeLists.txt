# Glyph — glass-box ternary routing architecture built on M4T.
# GLYPH IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.

cmake_minimum_required(VERSION 3.16)
project(glyph LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wstrict-prototypes
    -Werror
)

if(APPLE)
    add_compile_options(-mcpu=native)
endif()

# Build the M4T substrate first.
add_subdirectory(m4t)

# libglyph — higher-level routed k-NN primitives on top of libm4t.
#
# The substrate (m4t) provides ternary kernels, packed-trit signatures,
# and popcount_dist. libglyph adds the consumer-side infrastructure:
# MNIST dataset loading, signature builder (projection + tau calibration),
# sorted bucket index, ternary multi-probe enumeration, resolver variants,
# and CLI hyperparameter parsing.
#
# This library started as the Axis 5 / Axis 6 refactor: every cascade
# tool was duplicating MNIST loader, deskew, RNG, signature building,
# bucket index, and multi-probe enumeration. Consolidated into libglyph
# so tools become thin CLI wrappers.
add_library(glyph STATIC
    src/glyph_rng.c
    src/glyph_dataset.c
    src/glyph_sig.c
    src/glyph_bucket.c
    src/glyph_multiprobe.c
    src/glyph_resolver.c
    src/glyph_config.c
)
target_include_directories(glyph PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
target_link_libraries(glyph PUBLIC m4t)

# Consumers built on the substrate. Currently one: the Trit Lattice LSH
# tool, the provisional primary consumer per docs/THESIS.md.
option(GLYPH_BUILD_TOOLS "Build glyph consumer tools" ON)
if(GLYPH_BUILD_TOOLS)
    add_executable(mnist_trit_lattice tools/mnist_trit_lattice.c)
    target_link_libraries(mnist_trit_lattice PRIVATE m4t)

    add_executable(mnist_routed_lattice tools/mnist_routed_lattice.c)
    target_link_libraries(mnist_routed_lattice PRIVATE m4t)

    add_executable(mnist_routed_knn tools/mnist_routed_knn.c)
    target_link_libraries(mnist_routed_knn PRIVATE m4t)

    add_executable(mnist_routed_trace tools/mnist_routed_trace.c)
    target_link_libraries(mnist_routed_trace PRIVATE m4t)

    add_executable(mnist_routed_weighted tools/mnist_routed_weighted.c)
    target_link_libraries(mnist_routed_weighted PRIVATE m4t)

    add_executable(mnist_routed_amplified tools/mnist_routed_amplified.c)
    target_link_libraries(mnist_routed_amplified PRIVATE m4t)

    add_executable(mnist_full_sweep tools/mnist_full_sweep.c)
    target_link_libraries(mnist_full_sweep PRIVATE m4t)

    add_executable(mnist_probe_nproj16 tools/mnist_probe_nproj16.c)
    target_link_libraries(mnist_probe_nproj16 PRIVATE m4t)

    add_executable(mnist_cascade_nproj16 tools/mnist_cascade_nproj16.c)
    target_link_libraries(mnist_cascade_nproj16 PRIVATE m4t)

    add_executable(mnist_cascade_atomics tools/mnist_cascade_atomics.c)
    target_link_libraries(mnist_cascade_atomics PRIVATE m4t)

    add_executable(mnist_cascade_sweep tools/mnist_cascade_sweep.c)
    target_link_libraries(mnist_cascade_sweep PRIVATE m4t)

    add_executable(mnist_resolver_sweep tools/mnist_resolver_sweep.c)
    target_link_libraries(mnist_resolver_sweep PRIVATE m4t m)

    add_executable(mnist_local_vs_global tools/mnist_local_vs_global.c)
    target_link_libraries(mnist_local_vs_global PRIVATE m4t)

    add_executable(mnist_lvg_atomics tools/mnist_lvg_atomics.c)
    target_link_libraries(mnist_lvg_atomics PRIVATE m4t)

    add_executable(mnist_local_v2 tools/mnist_local_v2.c)
    target_link_libraries(mnist_local_v2 PRIVATE m4t)

    add_executable(mnist_routed_bucket tools/mnist_routed_bucket.c)
    target_link_libraries(mnist_routed_bucket PRIVATE m4t)

    add_executable(mnist_routed_bucket_multi tools/mnist_routed_bucket_multi.c)
    target_link_libraries(mnist_routed_bucket_multi PRIVATE glyph)
endif()

# Glyph tests compile against the glyph headers and can smoke-test routed
# consumers when the tool targets are enabled.
option(GLYPH_BUILD_TESTS "Build glyph wrapper tests" ON)
if(GLYPH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
