# Glyph — NEON-optimized ternary / multi-trit / MTFP library for Apple M4.
# GLYPH IS TERNARY / MULTI-TRIT / MULTI-TRIT FLOATING POINT ONLY.
# See docs/REMEDIATION_PLAN.md for policy.

cmake_minimum_required(VERSION 3.16)
project(glyph LANGUAGES C)

if(NOT APPLE)
    message(FATAL_ERROR
        "Glyph targets Apple M4 only. Non-Apple builds are not supported in v0. "
        "See docs/REMEDIATION_PLAN.md D5.")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)  # libdispatch blocks extension

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
    -mcpu=native
)

set(GLYPH_CORE_SRCS
    src/glyph_mtfp.c
    src/glyph_trit_pack.c
    src/glyph_ternary_matmul.c
)

add_library(glyph STATIC ${GLYPH_CORE_SRCS})
target_include_directories(glyph PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
set_target_properties(glyph PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# P3: reference-code/ must never be on the glyph include path.
# That directory contains trix-z's float paths and violates glyph's
# ternary/multi-trit/MTFP policy. If it ever ends up on an include path
# by accident, fail the configure step.
get_target_property(_glyph_includes glyph INCLUDE_DIRECTORIES)
foreach(inc IN LISTS _glyph_includes)
    if(inc MATCHES "reference-code")
        message(FATAL_ERROR
            "reference-code/ must not be on glyph's include path "
            "(found: ${inc}). See docs/REMEDIATION_PLAN.md policy section.")
    endif()
endforeach()

option(GLYPH_BUILD_TESTS "Build glyph tests" ON)
if(GLYPH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
