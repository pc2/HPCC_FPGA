cmake_minimum_required(VERSION 3.1)
project(PTRANS VERSION 1.1)

set(KERNEL_NAME transpose CACHE STRING "Name of the OpenCL kernel")
set(DEFAULT_MATRIX_SIZE 8 CACHE STRING "Default size of the used matrices")
set(BLOCK_SIZE 512 CACHE STRING "Block size used in the FPGA kernel")
set(GLOBAL_MEM_UNROLL 16 CACHE STRING "Unrolling factor used to stream data")

include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)
