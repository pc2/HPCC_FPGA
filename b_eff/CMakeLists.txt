cmake_minimum_required(VERSION 3.13)
project(b_eff VERSION 1.3)

set(SEND_KERNEL_NAME send)
set(RECV_KERNEL_NAME recv)
set(CHANNEL_WIDTH 32 CACHE STRING "Width of a single external channel in Byte")
set(DEFAULT_MAX_MESSAGE_SIZE 20 CACHE STRING "Log2 of the maximum message size that should be used in the benchmark. Default is 2^20=1MB")
set(DEFAULT_MAX_LOOP_LENGTH 65536 CACHE STRING "Maximum number of repetitions done for a single message size")
set(DEFAULT_MIN_LOOP_LENGTH 256 CACHE STRING "Minimum number of repetitions done for a single message size")#
set(DEFAULT_LOOP_LENGTH_OFFSET 11 CACHE STRING "Offset that is used for the message sizes before reducing the number of repetitions")
set(DEFAULT_LOOP_LENGTH_DECREASE 7 CACHE STRING "Number of steps that are used to decrease the number of repetitions to its minimum")

set(USE_MPI Yes)
set(USE_DEPRECATED_HPP_HEADER No)

set(DATA_TYPE char)
include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)
unset(DATA_TYPE CACHE)
if (NOT MPI_FOUND) 
    message(ERROR "Benchmarks host code requires MPI, but MPI could not be found.")
endif()
if (NOT INTELFPGAOPENCL_FOUND)
    message(ERROR "Benchmark does only support the Intel OpenCL SDK")
endif()
