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
