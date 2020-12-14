cmake_minimum_required(VERSION 3.13)
project(PTRANS VERSION 1.2)

set(READ_KERNEL_NAME transpose_read CACHE STRING "Name of the OpenCL kernel that reads A and sends it over external channel")
set(WRITE_KERNEL_NAME transpose_write CACHE STRING "Name of the OpenCL kernel that receives A, adds C to it and stores result")
set(DEFAULT_MATRIX_SIZE 8 CACHE STRING "Default size of the used matrices")
set(BLOCK_SIZE 512 CACHE STRING "Block size used in the FPGA kernel")
set(CHANNEL_WIDTH 8 CACHE STRING "Width of a single channel in number of values. Also specifies the width of memory ")

mark_as_advanced(READ_KERNEL_NAME WRITE_KERNEL_NAME)

include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)
