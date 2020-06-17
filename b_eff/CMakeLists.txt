cmake_minimum_required(VERSION 3.13)
project(b_eff VERSION 1.2)

set(SEND_KERNEL_NAME send)
set(RECV_KERNEL_NAME recv)
set(CHANNEL_WIDTH 32 CACHE STRING "Width of a single external channel in Byte")

set(USE_MPI Yes)

set(DATA_TYPE char)
include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)
unset(DATA_TYPE CACHE)
if (NOT MPI_FOUND) 
    message(ERROR "Benchmarks host code requires MPI, but MPI could not be found.")
endif()
if (NOT INTELFPGAOPENCL_FOUND)
    message(ERROR "Benchmark does only support the Intel OpenCL SDK")
endif()
