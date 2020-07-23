cmake_minimum_required(VERSION 3.13)
project(GEMM VERSION 0.4)

set(KERNEL_NAME gemm CACHE STRING "Name of the OpenCL kernel")
set(DEFAULT_MATRIX_SIZE 8 CACHE STRING "Default size of the used matrices")
set(BLOCK_SIZE 512 CACHE STRING "Block size used in the FPGA kernel")
set(GEMM_BLOCK 8 CACHE STRING "Block size used in the FPGA kernel for the cannon algorithm")
set(GLOBAL_MEM_UNROLL 16 CACHE STRING "Unrolling factor used to stream data")
set(NUM_REPLICATIONS 4 CACHE STRING "Number of times the kernels will be replicated")

set(INTEL_MUL_SHIFT_REG 6 CACHE STRING "Size of the shift register used to relax memory dependencies in the local memory MM")


include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)

