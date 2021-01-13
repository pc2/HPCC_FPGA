cmake_minimum_required(VERSION 3.13)
project(GEMM VERSION 1.2)

set(KERNEL_NAME gemm CACHE STRING "Name of the OpenCL kernel")
set(DEFAULT_MATRIX_SIZE 8 CACHE STRING "Default size of the used matrices")
set(BLOCK_SIZE 512 CACHE STRING "Block size used in the FPGA kernel")
set(GEMM_BLOCK 8 CACHE STRING "Block size used in the FPGA kernel for the cannon algorithm")
set(GLOBAL_MEM_UNROLL 16 CACHE STRING "Unrolling factor used to stream data")
set(NUM_REPLICATIONS 4 CACHE STRING "Number of times the kernels will be replicated")

set(INTEL_MUL_SHIFT_REG 0 CACHE STRING "Size of the shift register used to relax memory dependencies in the local memory MM. If set to 0, the shift register is removed.")
set(XILINX_UNROLL_GLOBAL_MEM_PIPELINE Yes CACHE BOOL "Fully unroll the strided reads and writes to from global memory to get a single pipeline and memory bursts")
set(ENABLE_MIXED_PRECISION No CACHE BOOL "Will use float as input parameter for the kernel, independent of the chosen data type. This allows e.g. using single precision on the host side and calculating in half precision on the device side.")

mark_as_advanced(XILINX_UNROLL_GLOBAL_MEM_PIPELINE ENABLE_MIXED_PRECISION KERNEL_NAME)

# Use MPI if it is available
find_package(MPI)
if (MPI_FOUND)
    message(STATUS "Found MPI, so binaries will be built with MPI support")
    set(USE_MPI Yes)
endif()

include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)

