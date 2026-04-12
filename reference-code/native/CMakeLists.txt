cmake_minimum_required(VERSION 3.16)
project(trix_native LANGUAGES C)

option(TRIX_USE_PTHREADS "Parallelize routed FFN across tiles with pthreads" ON)
option(TRIX_BUILD_TESTS "Build native C test executables" OFF)

if(TRIX_USE_PTHREADS)
  find_package(Threads REQUIRED)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O3)

if(APPLE)
  add_compile_options(-mcpu=native)
  add_definitions(-DAPPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
endif()

set(TRIX_COMMON_DEFS
  TRIX_USE_PTHREADS=$<BOOL:${TRIX_USE_PTHREADS}>
)

# ── Core: ternary routed FFN and its foundations ──

set(TRIX_CORE_SRCS
  src/trix_atoms.c
  src/trix_mtfp.c
  src/trix_ternary_matmul.c
  src/trix_ternary_route.c
  src/trix_routed_proj.c
  src/trix_surgery.c
  src/trix_compiled_dispatch.c
  src/trix_xor_compress.c
  src/trix_routed_mixer.c
  src/trix_routed_block.c
  src/trix_routed_mixer_causal.c
)

add_library(trix SHARED ${TRIX_CORE_SRCS})
target_include_directories(trix PRIVATE src)
target_link_libraries(trix PRIVATE m)
if(APPLE)
  target_link_libraries(trix PRIVATE ${ACCELERATE_FRAMEWORK})
endif()
if(TRIX_USE_PTHREADS)
  target_link_libraries(trix PRIVATE Threads::Threads)
endif()
target_compile_definitions(trix PRIVATE ${TRIX_COMMON_DEFS})
set_target_properties(trix PROPERTIES
  C_STANDARD 11
  POSITION_INDEPENDENT_CODE ON
)

# ── Tests ──

include(CTest)
if(BUILD_TESTING AND TRIX_BUILD_TESTS)
  function(add_trix_test NAME)
    cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})
    add_executable(${NAME} ${ARG_SOURCES} ${TRIX_CORE_SRCS})
    target_include_directories(${NAME} PRIVATE src)
    target_link_libraries(${NAME} PRIVATE m)
    if(APPLE)
      target_link_libraries(${NAME} PRIVATE ${ACCELERATE_FRAMEWORK})
    endif()
    if(TRIX_USE_PTHREADS)
      target_link_libraries(${NAME} PRIVATE Threads::Threads)
    endif()
    target_compile_definitions(${NAME} PRIVATE ${TRIX_COMMON_DEFS})
    add_test(NAME ${NAME} COMMAND ${NAME})
  endfunction()

  add_trix_test(trix_atoms_smoke SOURCES tests/test_native_atoms_smoke.c)
  add_trix_test(trix_ternary_route_fair SOURCES tests/test_ternary_route_fair.c)
  add_trix_test(trix_ternary_scale SOURCES tests/test_ternary_scale.c)
  add_trix_test(trix_ternary_mnist SOURCES tests/test_ternary_mnist.c)
  add_trix_test(trix_ternary_vision SOURCES tests/test_ternary_vision.c)
  add_trix_test(trix_routed_proj SOURCES tests/test_routed_proj.c)
  add_trix_test(trix_mtfp SOURCES tests/test_mtfp.c)
  add_trix_test(trix_ternary_mnist_seeds SOURCES tests/test_ternary_mnist_seeds.c)
  add_trix_test(trix_ternary_mnist_sweep SOURCES tests/test_ternary_mnist_sweep.c)
  add_trix_test(trix_ternary_mnist_errors SOURCES tests/test_ternary_mnist_errors.c)
  add_trix_test(trix_attack2_bypass SOURCES tests/test_attack2_bypass.c)
  add_trix_test(trix_routed_mixer SOURCES tests/test_routed_mixer.c)
  add_trix_test(trix_mixer_trace SOURCES tests/test_mixer_trace.c)
  add_trix_test(trix_routed_lm SOURCES tests/test_routed_lm.c)
  add_trix_test(trix_training_debug SOURCES tests/test_training_debug.c)
  add_trix_test(trix_lm_inspect SOURCES tests/test_lm_inspect.c)
endif()
