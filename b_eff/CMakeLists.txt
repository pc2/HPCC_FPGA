cmake_minimum_required(VERSION 3.1)
project(b_eff VERSION 1.1.1)

set(SEND_KERNEL_NAME send)
set(RECV_KERNEL_NAME recv)
set(CHANNEL_WIDTH 32 CACHE STRING "Width of a single external channel in Byte")

find_package(MPI REQUIRED)

set(DATA_TYPE char)
include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)
unset(DATA_TYPE CACHE)
if (NOT INTELFPGAOPENCL_FOUND)
    message(ERROR "Benchmark does only support the Intel OpenCL SDK")
endif()
