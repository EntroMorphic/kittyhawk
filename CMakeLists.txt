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

enable_testing()
add_subdirectory(m4t)
