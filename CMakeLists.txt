# Glyph — routing-first ternary compute stack for Apple Silicon.
# GLYPH IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
#
# Ground-zero rebuild begun 2026-05-01. The substrate library `m4t` is the
# first thing back online. Higher layers (libglyph, libtrain, tools) return
# when their kernels are ready. Until then, the build configures only m4t.

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

# Link-time optimization. Full LTO (cross-TU inlining + IR-level opt).
# Per V2-G1 root-cause investigation: the prior "full-LTO segfault" turned
# out to be a TEST CODE BUG (side-effecting load_mnist() inside assert(),
# which is removed under -DNDEBUG). Once the test was fixed, full LTO works
# clean. See journal/tier2_residuals_v2_pinpoint.md for the trace.
#
# Gated by GESH_LTO option (default ON) so the V4-residual-3 LTO microbench
# can build a parallel no-LTO tree via -DGESH_LTO=OFF for direct comparison.
option(GESH_LTO "Enable link-time optimization" ON)
if(GESH_LTO)
    add_compile_options(-flto)
    add_link_options(-flto)
endif()

enable_testing()

# Tests must run with assertions enabled regardless of the substrate's
# Release-build NDEBUG. Per V2-pinpoint red-team C1: assert(EXPR) under
# -DNDEBUG becomes ((void)0) and EXPR is never evaluated, silently disabling
# all test validation in Release builds. Apply -UNDEBUG to test executables
# only — production binaries (probes, benches, future tools) stay compiled
# with NDEBUG.
#
# V4 expansion (per journal/tier2_residuals_v4_*.md): -UNDEBUG on a test
# executable only affects code compiled INTO that executable's own .o files.
# It does not re-enable asserts inside static libraries the test links
# against. To make substrate-internal asserts (e.g., m4t_route_topk_abs's
# T <= M4T_ROUTE_MAX_T precondition) actually fire when triggered from
# tests, we ship parallel `_test` library variants that compile the same
# sources with -UNDEBUG. Test executables link against the `_test`
# variants (m4t_test, gesh_test, gesh_bench_test, gesh_image_canon_test);
# production binaries link against the production variants. The
# test_m4t_assert_live deliberate-abort meta-test verifies this end-to-end.
function(gesh_test_undebug target)
    target_compile_options(${target} PRIVATE -UNDEBUG)
endfunction()

add_subdirectory(m4t)

# Gesh — base-3 native routing layer; substrate's first measured consumer.
# Phase A.1 ships forward-pass + synthetic benchmark; Phase A.2 will add
# lattice-update training. See gesh/README.md.
add_subdirectory(gesh)
