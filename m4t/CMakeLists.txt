# M4T — M4 Ternary Extensions
# Routing-first ternary/MTFP compute substrate for aarch64 + NEON.
# M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
#
# Single-threaded at the opcode level. No libdispatch, no pthreads.
# Threading is a consumer concern.
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

set(M4T_CORE_SRCS
    src/m4t_mtfp.c
    src/m4t_mtfp4.c
    src/m4t_trit_pack.c
    src/m4t_trit_ops.c
    src/m4t_trit_reducers.c
    src/m4t_ternary_matmul.c
    src/m4t_route.c
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
    enable_testing()

    add_executable(test_m4t_mtfp tests/test_m4t_mtfp.c)
    target_link_libraries(test_m4t_mtfp PRIVATE m4t)
    add_test(NAME m4t_mtfp COMMAND test_m4t_mtfp)

    add_executable(test_m4t_trit_ops tests/test_m4t_trit_ops.c)
    target_link_libraries(test_m4t_trit_ops PRIVATE m4t)
    add_test(NAME m4t_trit_ops COMMAND test_m4t_trit_ops)

    add_executable(test_m4t_trit_reducers tests/test_m4t_trit_reducers.c)
    target_link_libraries(test_m4t_trit_reducers PRIVATE m4t)
    add_test(NAME m4t_trit_reducers COMMAND test_m4t_trit_reducers)

    add_executable(test_m4t_route tests/test_m4t_route.c)
    target_link_libraries(test_m4t_route PRIVATE m4t)
    add_test(NAME m4t_route COMMAND test_m4t_route)

    add_executable(test_m4t_mtfp4 tests/test_m4t_mtfp4.c)
    target_link_libraries(test_m4t_mtfp4 PRIVATE m4t)
    add_test(NAME m4t_mtfp4 COMMAND test_m4t_mtfp4)
endif()

option(M4T_BUILD_TOOLS "Build M4T dev tools (trit_golden, lut_gen)" OFF)
if(M4T_BUILD_TOOLS)
    # Golden-value enumerator for trit op truth tables. Pure substrate,
    # no float. Useful when regenerating LUT_MUL / LUT_SAT_ADD / etc.
    add_executable(m4t_trit_golden tools/m4t_trit_golden.c)
    target_link_libraries(m4t_trit_golden PRIVATE m4t)

    # Offline LUT generator for (future) ternary-float nonlinearities.
    # This is the ONLY sanctioned place in the ecosystem where binary
    # float (libm) is used — at build time, not runtime. Currently has
    # no live consumer inside libm4t; archived nonlinearity consumers
    # would link its output back in when a consumer demands them.
    add_executable(m4t_lut_gen tools/m4t_lut_gen.c)
    target_link_libraries(m4t_lut_gen PRIVATE m)
endif()
