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
set(NUM_REPLICATIONS 2 CACHE STRING "")

set(HOST_EMULATION_REORDER No CACHE BOOL "Reorder the scheduling of FPGA kernels for Intel fast emulator since channels are only read once!")


set(USE_MPI Yes)
set(USE_DEPRECATED_HPP_HEADER No)

set(COMMUNICATION_TYPE_SUPPORT_ENABLED Yes)

set(DATA_TYPE char)
include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)
unset(DATA_TYPE CACHE)
find_package(MPI REQUIRED)
