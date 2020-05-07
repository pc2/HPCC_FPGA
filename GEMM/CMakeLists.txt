cmake_minimum_required(VERSION 3.1)
project(GEMM VERSION 0.2.1 )

set(KERNEL_NAME gemm CACHE STRING "Name of the OpenCL kernel")
set(DEFAULT_MATRIX_SIZE 4096 CACHE STRING "Default size of the used matrices")
set(BLOCK_SIZE 512 CACHE STRING "Block size used in the FPGA kernel")
set(GEMM_BLOCK 8 CACHE STRING "Block size used in the FPGA kernel for the cannon algorithm")
set(GLOBAL_MEM_UNROLL 16 CACHE STRING "Unrolling factor used to stream data")

include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)

