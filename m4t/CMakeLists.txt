# M4T — M4 Ternary Extensions
# A generic, cache-resident ternary opcode substrate for aarch64 + NEON.
# M4T IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
#
# Single-threaded at the opcode level. No libdispatch, no pthreads.
# Threading is a consumer concern.

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
    src/m4t_trit_pack.c
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
    enable_testing()
    add_executable(test_m4t_smoke tests/test_m4t_smoke.c)
    target_link_libraries(test_m4t_smoke PRIVATE m4t)
    add_test(NAME m4t_smoke COMMAND test_m4t_smoke)
endif()

option(M4T_BUILD_TOOLS "Build M4T tools (bench, size check)" ON)
if(M4T_BUILD_TOOLS)
    add_executable(m4t_bench tools/m4t_bench.c)
    target_link_libraries(m4t_bench PRIVATE m4t)

    # M1: link-time size budget enforcement (24 KB = 24576 bytes)
    add_custom_command(TARGET m4t POST_BUILD
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/m4t_size_check.sh
                $<TARGET_FILE:m4t> 24576
        COMMENT "M4T: checking opcode body size budget (24 KB)"
    )
endif()
