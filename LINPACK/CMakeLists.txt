cmake_minimum_required(VERSION 3.1)
project(LINPACK VERSION 2.1)

set(DEFAULT_MATRIX_SIZE 1024 CACHE STRING "Default matrix size")
set(GLOBAL_MEM_UNROLL 16 CACHE STRING "Unrolling factor that is used for all loops in the kernels")
set(LOCAL_MEM_BLOCK_LOG 5 CACHE STRING "Used to define the width and height of the block stored in local memory")
set(REGISTER_BLOCK_LOG 3 CACHE STRING "Size of the block that will be manipulated in registers")
set(TEST_UNIFORM No CACHE BOOL "All tests executed by CTest will be executed with uniformly generated matrices")

if (TEST_UNIFORM)
    set(TEST_HOST_FLAGS "--uniform")
endif()

set(DATA_TYPE float)
if (DATA_TYPE STREQUAL "double")
    set(_DP Yes)
    message(STATUS "Set DP flag since data type seems to be double precision")
else()
    set(_DP No)
endif()

set(USE_OPENMP Yes)
include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)
