cmake_minimum_required(VERSION 3.1)
project(LINPACK VERSION 2.4)

set(USE_DEPRECATED_HPP_HEADER No)

set(DEFAULT_MATRIX_SIZE 2 CACHE STRING "Default matrix size")
set(LOCAL_MEM_BLOCK_LOG 5 CACHE STRING "Used to define the width and height of the block stored in local memory")
set(REGISTER_BLOCK_LOG 3 CACHE STRING "Size of the block that will be manipulated in registers")
set(NUM_REPLICATIONS 1 CACHE STRING "Number of times the matrix multiplication kernel will be replicated")
set(TEST_UNIFORM No CACHE BOOL "All tests executed by CTest will be executed with uniformly generated matrices")
set(TEST_EMULATION Yes CACHE BOOL "All tests executed by CTest will be executed with emulation kernels")
set(DISTRIBUTED_VALIDATION Yes CACHE BOOL "Use the distributed validation scheme instead of validation on rank 0")

set(COMMUNICATION_TYPE_SUPPORT_ENABLED Yes)

if (TEST_UNIFORM)
    set(TEST_HOST_FLAGS "--uniform")
endif()

if (TEST_EMULATION)
    set(TEST_HOST_FLAGS "--emulation")
endif()

if (DATA_TYPE STREQUAL "double")
    set(_DP Yes)
    message(STATUS "Set DP flag since data type seems to be double precision")
else()
    set(_DP No)
endif()

set(USE_OPENMP Yes)
set(USE_MPI Yes)

include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)
