cmake_minimum_required(VERSION 3.1)
project(b_eff VERSION 1.1)

set (CMAKE_CXX_STANDARD 11)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

# Include CMake module files provided by hlslib
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/../extern/hlslib/cmake)

set(SEND_KERNEL_NAME send)
set(RECV_KERNEL_NAME recv)
set(DEFAULT_PLATFORM -1 CACHE STRING "Default platform index that is used by all ranks. (-1: manual selection)")
set(DEFAULT_DEVICE -1 CACHE STRING "Default device index that is used by all ranks. (-1: manual selection)")
set(DEFAULT_REPETITIONS 10 CACHE STRING "Default number of repetitions")

set(AOC_FLAGS "-fpc -fp-relaxed" CACHE STRING "Used flags for the AOC compiler")
separate_arguments(AOC_FLAGS)

set(CHANNEL_WIDTH 32 CACHE STRING "Width of a single external channel in Byte")
set(FPGA_BOARD_NAME p520_max_sg280l CACHE STRING "Name of the target FPGA board")

set(HOST_DATA_TYPE cl_uchar)
set(DEVICE_DATA_TYPE uchar)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

configure_file(
        "${CMAKE_SOURCE_DIR}/src/common/parameters.h.in"
        "${CMAKE_BINARY_DIR}/src/common/parameters.h"
)

include_directories(${CMAKE_BINARY_DIR}/src/common)

find_package(IntelFPGAOpenCL REQUIRED)
find_package(MPI REQUIRED)

add_subdirectory(src/device)
add_subdirectory(src/host)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

