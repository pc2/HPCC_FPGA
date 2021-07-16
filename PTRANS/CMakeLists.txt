cmake_minimum_required(VERSION 3.13)
project(PTRANS VERSION 1.5)

set(READ_KERNEL_NAME transpose_read CACHE STRING "Name of the OpenCL kernel that reads A and sends it over external channel")
set(WRITE_KERNEL_NAME transpose_write CACHE STRING "Name of the OpenCL kernel that receives A, adds C to it and stores result")
set(DEFAULT_MATRIX_SIZE 8 CACHE STRING "Default size of the used matrices")
set(DEFAULT_COMM_TYPE "IEC" CACHE STRING "Default communication type if nothing else is given over the --connectivity parameter")
set(DEFAULT_DIST_TYPE "PQ" CACHE STRING "Default distribution type if nothing is specified over the --handler parameter")
set(BLOCK_SIZE 512 CACHE STRING "Block size used in the FPGA kernel")
set(CHANNEL_WIDTH 8 CACHE STRING "Width of a single channel in number of values. Also specifies the width of memory ")
# NUM_REPLICATIONS set to 2 by default to allow build and execution of both versions of the transpose kernel
set(NUM_REPLICATIONS 2 CACHE STRING "Number of times the kernels will be replicated")

mark_as_advanced(READ_KERNEL_NAME WRITE_KERNEL_NAME)

set(USE_MPI Yes)
set(USE_OPENMP Yes)

include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)
