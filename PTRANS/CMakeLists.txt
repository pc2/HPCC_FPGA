cmake_minimum_required(VERSION 3.13)
project(PTRANS VERSION 1.7)

set(READ_KERNEL_NAME transpose_read CACHE STRING "Name of the OpenCL kernel that reads A and sends it over external channel")
set(WRITE_KERNEL_NAME transpose_write CACHE STRING "Name of the OpenCL kernel that receives A, adds C to it and stores result")
set(DEFAULT_MATRIX_SIZE 8 CACHE STRING "Default size of the used matrices")
set(DEFAULT_COMM_TYPE "AUTO" CACHE STRING "Default communication type if nothing else is given over the --connectivity parameter")
set(DEFAULT_DIST_TYPE "AUTO" CACHE STRING "Default distribution type if nothing is specified over the --handler parameter")
set(DEFAULT_P_VALUE 1 CACHE STRING "Default value of P that sets the width of the PQ grid")
set(BLOCK_SIZE 512 CACHE STRING "Block size used in the FPGA kernel")
set(CHANNEL_WIDTH 8 CACHE STRING "Width of a single channel in number of values. Also specifies the width of memory ")
# NUM_REPLICATIONS set to 2 by default to allow build and execution of both versions of the transpose kernel
set(NUM_REPLICATIONS 2 CACHE STRING "Number of times the kernels will be replicated")
set(USE_BUFFER_WRITE_RECT_FOR_A No CACHE BOOL "Only valid for PQ with IEC. Use the enqueueWriteBufferRect call to copy only the relevant part of A to memory bank of each replication. Whole matrix A will be copied otherwise.")
set(XILINX_UNROLL_INNER_LOOPS No CACHE BOOL "When building for Xilinx devices, unroll the inner loops to create a single pipeline per block and keep memory bursts. This is a tradeoff between resource usage and performance.")

set(HOST_EMULATION_REORDER No CACHE BOOL "Reorder the scheduling of FPGA kernels for Intel fast emulator since channels are only read once!")

mark_as_advanced(READ_KERNEL_NAME WRITE_KERNEL_NAME USE_BUFFER_WRITE_RECT_FOR_A XILINX_UNROLL_INNER_LOOPS)

set(USE_MPI Yes)
set(USE_OPENMP Yes)
set(USE_DEPRECATED_HPP_HEADER No)
set(COMMUNICATION_TYPE_SUPPORT_ENABLED Yes)

include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)
