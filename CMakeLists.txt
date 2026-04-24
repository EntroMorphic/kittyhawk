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

# Routed autodiff MVP — consumer-layer training scaffolding per
# NORTH_STAR §13. Not on libm4t's build path. Opt-in via GLYPH_BUILD_TRAIN.
option(GLYPH_BUILD_TRAIN "Build routed autodiff MVP (train/)" ON)
if(GLYPH_BUILD_TRAIN)
    add_subdirectory(train)
endif()

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
    src/glyph_probe.c
    src/glyph_resolver.c
    src/glyph_config.c
)
target_include_directories(glyph PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
target_link_libraries(glyph PUBLIC m4t)

# Consumer tools built on libglyph (which transitively pulls libm4t).
option(GLYPH_BUILD_TOOLS "Build glyph consumer tools" ON)

# Legacy random-projection consumers. These tools generate random ternary
# projection matrices at startup and score queries against them — a pattern
# that violates the NO RANDOM PROJECTIONS / NO RANDOM WEIGHTS discipline
# for image classification (see docs/LIBGLYPH.md and NORTH_STAR.md). Their
# outputs are preserved as historical Axis 5/6 benchmark references and
# as atomic-probe diagnostic tooling. Default OFF; opt in only for
# benchmark reproduction. Production consumers (direct_lsh, structured_*,
# ig_scored, inverted_ig, sstt_precog, block_distance) do NOT require
# this flag.
option(GLYPH_BUILD_LEGACY_RP "Build legacy random-projection consumer tools" OFF)

if(GLYPH_BUILD_TOOLS)
    # -------- Production / current consumers (rule-compliant) --------

    add_executable(direct_lsh tools/direct_lsh.c)
    target_link_libraries(direct_lsh PRIVATE glyph m)

    add_executable(sstt_precog tools/sstt_precog.c)
    target_link_libraries(sstt_precog PRIVATE glyph)

    add_executable(ig_scored tools/ig_scored.c)
    target_link_libraries(ig_scored PRIVATE glyph)

    add_executable(inverted_ig tools/inverted_ig.c)
    target_link_libraries(inverted_ig PRIVATE glyph)

    add_executable(structured_gsh tools/structured_gsh.c)
    target_link_libraries(structured_gsh PRIVATE glyph)

    add_executable(structured_lsh tools/structured_lsh.c)
    target_link_libraries(structured_lsh PRIVATE glyph)

    add_executable(block_distance tools/block_distance.c)
    target_link_libraries(block_distance PRIVATE glyph m)

    # Class-Signature Argmax: ternary class prototypes scored via
    # popcount_dist. Derived from the lr_scaffold LMM cycle (journal/
    # lr_scaffold_*.md) — the substrate-native shape for the
    # per-class weighting problem that LR would have solved in base-2.
    add_executable(csa_classifier tools/csa_classifier.c)
    target_link_libraries(csa_classifier PRIVATE glyph)

    # go_probe — base3_benchmarks cycle probe (per journal/base3_benchmarks_synthesize.md).
    # Standalone SGF parser + brute-force Hamming k-NN on raw ternary Go
    # positions. Does not depend on libglyph; links only against libm.
    # Probe-scope tool; promote or retire based on outcome documented in
    # journal/base3_go_probe.md.
    add_executable(go_probe tools/go_probe.c)
    target_link_libraries(go_probe PRIVATE m)

    # -------- Legacy random-projection consumers (opt-in) --------

    if(GLYPH_BUILD_LEGACY_RP)
        add_executable(mnist_trit_lattice tools/mnist_trit_lattice.c)
        target_link_libraries(mnist_trit_lattice PRIVATE glyph)

        add_executable(mnist_routed_lattice tools/mnist_routed_lattice.c)
        target_link_libraries(mnist_routed_lattice PRIVATE glyph)

        add_executable(mnist_routed_knn tools/mnist_routed_knn.c)
        target_link_libraries(mnist_routed_knn PRIVATE glyph)

        add_executable(mnist_routed_trace tools/mnist_routed_trace.c)
        target_link_libraries(mnist_routed_trace PRIVATE glyph)

        add_executable(mnist_routed_weighted tools/mnist_routed_weighted.c)
        target_link_libraries(mnist_routed_weighted PRIVATE glyph)

        add_executable(mnist_routed_amplified tools/mnist_routed_amplified.c)
        target_link_libraries(mnist_routed_amplified PRIVATE glyph)

        add_executable(mnist_full_sweep tools/mnist_full_sweep.c)
        target_link_libraries(mnist_full_sweep PRIVATE glyph)

        add_executable(mnist_probe_nproj16 tools/mnist_probe_nproj16.c)
        target_link_libraries(mnist_probe_nproj16 PRIVATE glyph)

        add_executable(mnist_cascade_nproj16 tools/mnist_cascade_nproj16.c)
        target_link_libraries(mnist_cascade_nproj16 PRIVATE glyph)

        add_executable(mnist_cascade_atomics tools/mnist_cascade_atomics.c)
        target_link_libraries(mnist_cascade_atomics PRIVATE glyph)

        add_executable(mnist_cascade_sweep tools/mnist_cascade_sweep.c)
        target_link_libraries(mnist_cascade_sweep PRIVATE glyph)

        add_executable(mnist_resolver_sweep tools/mnist_resolver_sweep.c)
        target_link_libraries(mnist_resolver_sweep PRIVATE glyph m)

        add_executable(mnist_local_vs_global tools/mnist_local_vs_global.c)
        target_link_libraries(mnist_local_vs_global PRIVATE glyph)

        add_executable(mnist_lvg_atomics tools/mnist_lvg_atomics.c)
        target_link_libraries(mnist_lvg_atomics PRIVATE glyph)

        add_executable(mnist_local_v2 tools/mnist_local_v2.c)
        target_link_libraries(mnist_local_v2 PRIVATE glyph)

        add_executable(mnist_routed_bucket tools/mnist_routed_bucket.c)
        target_link_libraries(mnist_routed_bucket PRIVATE glyph)

        add_executable(mnist_routed_bucket_multi tools/mnist_routed_bucket_multi.c)
        target_link_libraries(mnist_routed_bucket_multi PRIVATE glyph)

        add_executable(fashion_atomics tools/fashion_atomics.c)
        target_link_libraries(fashion_atomics PRIVATE glyph)

        add_executable(cifar_seed_overlap tools/cifar_seed_overlap.c)
        target_link_libraries(cifar_seed_overlap PRIVATE glyph)

        add_executable(dynamic_nproj tools/dynamic_nproj.c)
        target_link_libraries(dynamic_nproj PRIVATE glyph)

        add_executable(subsetted_multi tools/subsetted_multi.c)
        target_link_libraries(subsetted_multi PRIVATE glyph)

        add_executable(bruteforce_nproj tools/bruteforce_nproj.c)
        target_link_libraries(bruteforce_nproj PRIVATE glyph)

        add_executable(layered_lsh tools/layered_lsh.c)
        target_link_libraries(layered_lsh PRIVATE glyph)

        add_executable(specialist_rerank tools/specialist_rerank.c)
        target_link_libraries(specialist_rerank PRIVATE glyph)

        add_executable(centroid_routed tools/centroid_routed.c)
        target_link_libraries(centroid_routed PRIVATE glyph)

        add_executable(conv_lsh tools/conv_lsh.c)
        target_link_libraries(conv_lsh PRIVATE glyph)
    endif()
endif()

# Glyph tests compile against the glyph headers and can smoke-test routed
# consumers when the tool targets are enabled.
option(GLYPH_BUILD_TESTS "Build glyph wrapper tests" ON)
if(GLYPH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
