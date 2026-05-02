# M4T — M4 Ternary Extensions
# Routing-first ternary/MTFP compute substrate for aarch64 + NEON.
# M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
#
# Single-threaded at the opcode level. No libdispatch, no pthreads.
# Threading is a consumer concern.
#
# Ground-zero rebuild status (2026-05-01):
#   Tier 1 — pure base-3 layer — ONLINE.
#   Tier 2 — route primitives + MTFP19 mantissa arithmetic — ONLINE.
#   Tier 3 — cross-exponent kernel + MTFP4 SDOT + MTFP19 ternary matmul
#            — ONLINE.
#
# See m4t/docs/M4T_SUBSTRATE.md for the canonical spec.

cmake_minimum_required(VERSION 3.16)
project(m4t LANGUAGES C)

if(NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64"))
    message(FATAL_ERROR
        "M4T requires aarch64 with NEON. "
        "Target processor '${CMAKE_SYSTEM_PROCESSOR}' is unsupported.")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(
    -O3
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

# All tiers — full substrate surface for routing-first base-3 compute.
set(M4T_CORE_SRCS
    src/m4t_trit_pack.c
    src/m4t_trit_ops.c
    src/m4t_trit_reducers.c
    src/m4t_mtfp.c
    src/m4t_mtfp4.c
    src/m4t_route.c
    src/m4t_ternary_matmul.c
)

add_library(m4t STATIC ${M4T_CORE_SRCS})
target_include_directories(m4t PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
set_target_properties(m4t PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

option(M4T_BUILD_TESTS "Build M4T tests" ON)
if(M4T_BUILD_TESTS)
    add_executable(test_m4t_trit_pack tests/test_m4t_trit_pack.c)
    target_link_libraries(test_m4t_trit_pack PRIVATE m4t)
    add_test(NAME m4t_trit_pack COMMAND test_m4t_trit_pack)

    add_executable(test_m4t_trit_ops tests/test_m4t_trit_ops.c)
    target_link_libraries(test_m4t_trit_ops PRIVATE m4t)
    add_test(NAME m4t_trit_ops COMMAND test_m4t_trit_ops)

    add_executable(test_m4t_trit_reducers tests/test_m4t_trit_reducers.c)
    target_link_libraries(test_m4t_trit_reducers PRIVATE m4t)
    add_test(NAME m4t_trit_reducers COMMAND test_m4t_trit_reducers)

    add_executable(test_m4t_mtfp tests/test_m4t_mtfp.c)
    target_link_libraries(test_m4t_mtfp PRIVATE m4t)
    add_test(NAME m4t_mtfp COMMAND test_m4t_mtfp)

    add_executable(test_m4t_route tests/test_m4t_route.c)
    target_link_libraries(test_m4t_route PRIVATE m4t)
    add_test(NAME m4t_route COMMAND test_m4t_route)

    # Property tests for the §14.2 cross-exponent accumulator. Bit-exact
    # int64 reference; no fp in the test path. 14 properties.
    add_executable(test_m4t_mtfp_accum_aligning tests/test_m4t_mtfp_accum_aligning.c)
    target_link_libraries(test_m4t_mtfp_accum_aligning PRIVATE m4t)
    add_test(NAME m4t_mtfp_accum_aligning COMMAND test_m4t_mtfp_accum_aligning)

    # SDOT MTFP4 matmul + cell-width conversions (§8.4 Case W).
    add_executable(test_m4t_mtfp4 tests/test_m4t_mtfp4.c)
    target_link_libraries(test_m4t_mtfp4 PRIVATE m4t)
    add_test(NAME m4t_mtfp4 COMMAND test_m4t_mtfp4)

    # MTFP19 × packed-ternary matmul (§8.5 Case S).
    add_executable(test_m4t_ternary_matmul tests/test_m4t_ternary_matmul.c)
    target_link_libraries(test_m4t_ternary_matmul PRIVATE m4t)
    add_test(NAME m4t_ternary_matmul COMMAND test_m4t_ternary_matmul)
endif()
